"""PDF text extraction using PyMuPDF."""

from __future__ import annotations

import fitz  # PyMuPDF

from app.core.exceptions import DocumentProcessingError
from app.core.logging_config import get_logger

logger = get_logger("extraction")


class PDFExtractor:
    """Extract text and rendered page images from PDF files using PyMuPDF."""

    @staticmethod
    def detect_pdf_type(pdf_path: str) -> tuple[str, dict]:
        """
        Detect whether PDF is text-based, scanned, or hybrid.
        Returns: (pdf_type, info_dict)
        """
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            text_pages = 0
            image_pages = 0
            empty_pages = 0

            for page in doc:
                text = page.get_text("text").strip()
                images = page.get_images()

                if len(text) > 50:
                    text_pages += 1
                elif images:
                    image_pages += 1
                else:
                    empty_pages += 1

            doc.close()

            info = {
                "total_pages": total_pages,
                "text_pages": text_pages,
                "image_pages": image_pages,
                "empty_pages": empty_pages,
            }

            if text_pages == total_pages:
                return "text", info
            if image_pages == total_pages or (text_pages == 0 and image_pages > 0):
                return "scanned", info
            if text_pages > 0 and image_pages > 0:
                return "hybrid", info
            return "unknown", info

        except Exception as exc:
            logger.error("pdf_type_detection_failed", error=str(exc), path=pdf_path)
            raise DocumentProcessingError(
                f"Failed to detect PDF type: {str(exc)}",
                details={"path": pdf_path},
            ) from exc

    @staticmethod
    def extract_text(pdf_path: str) -> list[dict]:
        """Extract text from each page of a text-based PDF."""
        try:
            doc = fitz.open(pdf_path)
            pages = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                images = page.get_images()

                pages.append(
                    {
                        "page_number": page_num + 1,
                        "raw_text": text,
                        "has_images": len(images) > 0,
                        "extraction_method": "text",
                        "char_count": len(text),
                    }
                )

            doc.close()
            logger.info("text_extraction_complete", pages=len(pages), path=pdf_path)
            return pages

        except Exception as exc:
            logger.error("text_extraction_failed", error=str(exc), path=pdf_path)
            raise DocumentProcessingError(
                f"Failed to extract text from PDF: {str(exc)}",
                details={"path": pdf_path},
            ) from exc

    @staticmethod
    def inspect_pages_for_ocr(
        pdf_path: str,
        text_threshold: int = 50,
    ) -> list[dict]:
        """
        Inspect every page once to determine which pages already have a usable
        text layer and which require OCR.
        """
        try:
            doc = fitz.open(pdf_path)
            plan: list[dict] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                images = page.get_images()
                has_usable_text = len(text.strip()) > text_threshold

                plan.append(
                    {
                        "page_number": page_num + 1,
                        "raw_text": text if has_usable_text else "",
                        "has_images": len(images) > 0,
                        "needs_ocr": not has_usable_text,
                    }
                )

            doc.close()
            return plan

        except Exception as exc:
            logger.error("pdf_page_inspection_failed", error=str(exc), path=pdf_path)
            raise DocumentProcessingError(
                f"Failed to inspect PDF pages: {str(exc)}",
                details={"path": pdf_path},
            ) from exc

    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """Get the number of pages in a PDF."""
        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to read PDF: {str(exc)}",
                details={"path": pdf_path},
            ) from exc

    @staticmethod
    def extract_page_as_image(
        pdf_path: str,
        page_number: int,
        dpi: int = 200,
    ) -> bytes:
        """Extract a single page as a PNG image for OCR."""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_number - 1]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            doc.close()
            return img_bytes
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to render page {page_number} as image: {str(exc)}",
                details={"path": pdf_path, "page": page_number},
            ) from exc

    @staticmethod
    def render_pages_as_images(
        pdf_path: str,
        page_numbers: list[int],
        dpi: int = 200,
    ) -> dict[int, bytes]:
        """Render multiple pages in a single PDF open call."""
        try:
            doc = fitz.open(pdf_path)
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            rendered: dict[int, bytes] = {}

            for page_number in page_numbers:
                page = doc[page_number - 1]
                pix = page.get_pixmap(matrix=mat)
                rendered[page_number] = pix.tobytes("png")

            doc.close()
            return rendered
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to render PDF pages as images: {str(exc)}",
                details={"path": pdf_path},
            ) from exc