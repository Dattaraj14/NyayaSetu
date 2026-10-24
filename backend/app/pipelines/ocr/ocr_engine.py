"""OCR pipeline using Tesseract OCR.

Supports English, Hindi, and Marathi OCR with page-level
confidence tracking and low-confidence detection.
"""

import io
from typing import Optional

from PIL import Image

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.exceptions import OCRError


logger = get_logger("ocr")


class OCREngine:
    """Abstract OCR engine interface."""

    def ocr_image(self, image_bytes: bytes) -> dict:
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
    """Tesseract OCR engine supporting English, Hindi and Marathi."""

    def __init__(self) -> None:
        settings = get_settings()

        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

            # Verify that Tesseract is actually available.
            version = pytesseract.get_tesseract_version()

            self._pytesseract = pytesseract
            self._language = "eng+hin+mar"

            logger.info(
                "tesseract_initialized",
                version=str(version),
                language=self._language,
            )

        except Exception as e:
            logger.error(
                "tesseract_init_failed",
                error=str(e),
            )
            self._pytesseract = None

    def ocr_image(self, image_bytes: bytes) -> dict:
        """Run Tesseract OCR on a single image."""

        if self._pytesseract is None:
            raise OCRError("Tesseract is not available")

        try:
            image = Image.open(io.BytesIO(image_bytes))

            # Convert images to RGB to avoid problems with
            # grayscale/RGBA PDF renderings.
            if image.mode != "RGB":
                image = image.convert("RGB")

            # OCR text.
            text = self._pytesseract.image_to_string(
                image,
                lang=self._language,
                config="--psm 3",
            )

            # OCR confidence information.
            data = self._pytesseract.image_to_data(
                image,
                lang=self._language,
                config="--psm 3",
                output_type=self._pytesseract.Output.DICT,
            )

            confidences: list[float] = []
            details: list[dict] = []

            for index, confidence in enumerate(data["conf"]):
                try:
                    confidence_value = float(confidence)
                except (TypeError, ValueError):
                    continue

                # Tesseract uses -1 for non-text regions.
                if confidence_value < 0:
                    continue

                word = data["text"][index].strip()

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

            # Convert 0-100 Tesseract confidence to 0-1.
            average_confidence = (
                sum(confidences) / len(confidences) / 100.0
                if confidences
                else 0.0
            )

            return {
                "text": text.strip(),
                "confidence": round(average_confidence, 4),
                "details": details,
            }

        except Exception as e:
            logger.error(
                "tesseract_failed",
                error=str(e),
            )

            raise OCRError(
                f"Tesseract OCR processing failed: {str(e)}"
            )


class OCRPipeline:
    """OCR pipeline using Tesseract with confidence tracking."""

    def __init__(self) -> None:
        settings = get_settings()

        self._settings = settings
        self._engine: Optional[OCREngine] = None

        try:
            self._engine = TesseractOCREngine()

            if self._engine._pytesseract is None:
                self._engine = None

        except Exception as e:
            logger.error(
                "ocr_pipeline_initialization_failed",
                error=str(e),
            )

    def ocr_page(
        self,
        image_bytes: bytes,
        page_number: int = 1,
    ) -> dict:
        """
        OCR a single page image.

        Returns:
            {
                "text": str,
                "confidence": float,
                "engine": str,
                "is_low_confidence": bool,
                "details": list
            }
        """

        if self._engine is None:
            raise OCRError(
                f"No OCR engine is available for page {page_number}",
                details={"page": page_number},
            )

        try:
            result = self._engine.ocr_image(image_bytes)

        except OCRError as e:
            logger.error(
                "ocr_page_failed",
                page=page_number,
                error=str(e),
            )
            raise

        confidence = float(result.get("confidence", 0.0))

        is_low_confidence = (
            confidence < self._settings.ocr_confidence_threshold
        )

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