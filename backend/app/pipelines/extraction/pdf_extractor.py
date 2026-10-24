"""PDF text extraction using PyMuPDF."""

from typing import Optional
import fitz  # PyMuPDF

from app.core.logging_config import get_logger
from app.core.exceptions import DocumentProcessingError

logger = get_logger("extraction")


class PDFExtractor:
    """Extract text from PDF files using PyMuPDF."""

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
            elif image_pages == total_pages or (text_pages == 0 and image_pages > 0):
                return "scanned", info
            elif text_pages > 0 and image_pages > 0:
                return "hybrid", info
            else:
                return "unknown", info

        except Exception as e:
            logger.error("pdf_type_detection_failed", error=str(e), path=pdf_path)
            raise DocumentProcessingError(
                f"Failed to detect PDF type: {str(e)}",
                details={"path": pdf_path}
            )

    @staticmethod
    def extract_text(pdf_path: str) -> list[dict]:
        """
        Extract text from each page of a text-based PDF.
        Returns list of dicts with page_number, raw_text, has_images.
        """
        try:
            doc = fitz.open(pdf_path)
            pages = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                images = page.get_images()

                pages.append({
                    "page_number": page_num + 1,
                    "raw_text": text,
                    "has_images": len(images) > 0,
                    "extraction_method": "text",
                    "char_count": len(text),
                })

            doc.close()
            logger.info("text_extraction_complete", pages=len(pages), path=pdf_path)
            return pages

        except Exception as e:
            logger.error("text_extraction_failed", error=str(e), path=pdf_path)
            raise DocumentProcessingError(
                f"Failed to extract text from PDF: {str(e)}",
                details={"path": pdf_path}
            )

    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """Get the number of pages in a PDF."""
        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to read PDF: {str(e)}",
                details={"path": pdf_path}
            )

    @staticmethod
    def extract_page_as_image(pdf_path: str, page_number: int, dpi: int = 300) -> bytes:
        """Extract a single page as a PNG image for OCR."""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_number - 1]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            doc.close()
            return img_bytes
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to render page {page_number} as image: {str(e)}",
                details={"path": pdf_path, "page": page_number}
            )
