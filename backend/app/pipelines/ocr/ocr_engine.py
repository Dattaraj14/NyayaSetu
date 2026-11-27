"""OCR pipeline using Tesseract OCR.

Supports English, Hindi, and Marathi OCR with single-pass text +
confidence extraction and configurable language selection.
"""

from __future__ import annotations

import io
import os
from typing import Any, Optional

from PIL import Image

from app.core.config import get_settings
from app.core.exceptions import OCRError
from app.core.logging_config import get_logger


logger = get_logger("ocr")

ALLOWED_OCR_LANGUAGES = frozenset(
    {
        "eng",
        "hin",
        "mar",
        "eng+hin",
        "eng+mar",
        "hin+mar",
        "eng+hin+mar",
    }
)


def reconstruct_text_from_tsv_data(data: dict[str, list[Any]]) -> str:
    """
    Rebuild formatted text from pytesseract image_to_data output.

    Groups words by (block_num, par_num, line_num) in reading order,
    inserting newlines between lines and blank lines between paragraphs.
    """
    texts = data.get("text", [])
    block_nums = data.get("block_num", [])
    par_nums = data.get("par_num", [])
    line_nums = data.get("line_num", [])

    # Preserve document order using insertion-ordered dicts.
    paragraphs: dict[tuple[int, int], dict[int, list[str]]] = {}

    count = min(len(texts), len(block_nums), len(par_nums), len(line_nums))
    for i in range(count):
        word = str(texts[i]).strip()
        if not word:
            continue

        par_key = (int(block_nums[i]), int(par_nums[i]))
        line_key = int(line_nums[i])

        lines = paragraphs.setdefault(par_key, {})
        lines.setdefault(line_key, []).append(word)

    paragraph_strings: list[str] = []
    for lines in paragraphs.values():
        line_strings = [" ".join(words) for words in lines.values() if words]
        if line_strings:
            paragraph_strings.append("\n".join(line_strings))

    return "\n\n".join(paragraph_strings).strip()


class OCREngine:
    """Abstract OCR engine interface."""

    def ocr_image(
        self,
        image_bytes: bytes,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Perform OCR on an image.

        Returns:
            {
                "text": str,
                "confidence": float,
                "details": list
            }
        """
        raise NotImplementedError


class TesseractOCREngine(OCREngine):
    """Tesseract OCR engine supporting English, Hindi, and Marathi."""

    def __init__(self, language: Optional[str] = None) -> None:
        settings = get_settings()

        # Prevent Tesseract OpenMP threads from oversubscribing CPU cores
        # when pages are OCR'd concurrently in ThreadPoolExecutor.
        os.environ["OMP_THREAD_LIMIT"] = "1"

        try:
            import pytesseract

            if settings.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

            version = pytesseract.get_tesseract_version()

            self._pytesseract = pytesseract
            self._language = language or settings.ocr_languages or "eng"

            logger.info(
                "tesseract_initialized",
                version=str(version),
                language=self._language,
            )

        except Exception as exc:
            logger.error(
                "tesseract_init_failed",
                error=str(exc),
            )
            self._pytesseract = None
            self._language = language or settings.ocr_languages or "eng"

    def ocr_image(
        self,
        image_bytes: bytes,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run single-pass Tesseract OCR (text + word confidences) on one image."""
        if self._pytesseract is None:
            raise OCRError("Tesseract is not available")

        lang = language or self._language

        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Single Tesseract pass returns words, layout structure, and confidences.
            data = self._pytesseract.image_to_data(
                image,
                lang=lang,
                config="--psm 3",
                output_type=self._pytesseract.Output.DICT,
            )

            text = reconstruct_text_from_tsv_data(data)

            confidences: list[float] = []
            details: list[dict[str, Any]] = []

            for index, confidence in enumerate(data["conf"]):
                try:
                    confidence_value = float(confidence)
                except (TypeError, ValueError):
                    continue

                if confidence_value < 0:
                    continue

                word = str(data["text"][index]).strip()
                if not word:
                    continue

                confidences.append(confidence_value)
                details.append(
                    {
                        "text": word,
                        "confidence": confidence_value / 100.0,
                        "bbox": {
                            "left": data["left"][index],
                            "top": data["top"][index],
                            "width": data["width"][index],
                            "height": data["height"][index],
                        },
                    }
                )

            average_confidence = (
                sum(confidences) / len(confidences) / 100.0
                if confidences
                else 0.0
            )

            return {
                "text": text,
                "confidence": round(average_confidence, 4),
                "details": details,
            }

        except Exception as exc:
            logger.error(
                "tesseract_failed",
                language=lang,
                error=str(exc),
            )
            raise OCRError(f"Tesseract OCR processing failed: {str(exc)}") from exc


class OCRPipeline:
    """OCR pipeline using Tesseract with confidence tracking."""

    def __init__(self, language: Optional[str] = None) -> None:
        settings = get_settings()

        self._settings = settings
        self._language = language or settings.ocr_languages or "eng"
        self._engine: Optional[OCREngine] = None

        try:
            engine = TesseractOCREngine(language=self._language)
            if engine._pytesseract is not None:
                self._engine = engine
        except Exception as exc:
            logger.error(
                "ocr_pipeline_initialization_failed",
                error=str(exc),
            )

    def ocr_page(
        self,
        image_bytes: bytes,
        page_number: int = 1,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        """OCR a single page image."""
        if self._engine is None:
            raise OCRError(
                f"No OCR engine is available for page {page_number}",
                details={"page": page_number},
            )

        try:
            result = self._engine.ocr_image(
                image_bytes,
                language=language or self._language,
            )
        except OCRError as exc:
            logger.error(
                "ocr_page_failed",
                page=page_number,
                error=str(exc),
            )
            raise

        confidence = float(result.get("confidence", 0.0))
        is_low_confidence = confidence < self._settings.ocr_confidence_threshold

        logger.info(
            "ocr_page_complete",
            page=page_number,
            confidence=confidence,
            low_confidence=is_low_confidence,
        )

        return {
            "text": result.get("text", ""),
            "confidence": confidence,
            "engine": "tesseract",
            "is_low_confidence": is_low_confidence,
            "details": result.get("details", []),
        }