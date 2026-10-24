"""Legal document text preprocessing.

Provides conservative cleaning for text extracted from PDFs and OCR.
Legal meaning, citations, dates, paragraph numbering, and important
punctuation are intentionally preserved.
"""

import re
import unicodedata

from app.core.logging_config import get_logger


logger = get_logger("preprocessing")


class LegalTextPreprocessor:
    """Conservative text preprocessing for Indian legal documents."""

    # Common OCR characters that can appear instead of normal punctuation.
    OCR_REPLACEMENTS = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }

    @classmethod
    def normalize_unicode(cls, text: str) -> str:
        """Normalize Unicode while preserving legal text."""
        text = unicodedata.normalize("NFKC", text)

        for old, new in cls.OCR_REPLACEMENTS.items():
            text = text.replace(old, new)

        return text

    @staticmethod
    def normalize_line_endings(text: str) -> str:
        """Convert Windows/Mac line endings to Unix line endings."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def remove_trailing_whitespace(text: str) -> str:
        """Remove spaces/tabs at the end of lines."""
        return "\n".join(
            line.rstrip()
            for line in text.split("\n")
        )

    @staticmethod
    def normalize_spaces(text: str) -> str:
        """Normalize repeated spaces without joining paragraphs."""
        lines = []

        for line in text.split("\n"):
            # Convert tabs to spaces.
            line = line.replace("\t", " ")

            # Collapse repeated horizontal whitespace.
            line = re.sub(r"[ ]{2,}", " ", line)

            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def normalize_blank_lines(text: str) -> str:
        """Limit excessive blank lines to a single paragraph gap."""
        return re.sub(r"\n{3,}", "\n\n", text)

    @staticmethod
    def fix_spacing_around_punctuation(text: str) -> str:
        """
        Fix obvious spacing issues around punctuation.

        This intentionally avoids aggressive transformations because
        legal citations and abbreviations must be preserved.
        """

        # Remove spaces before common punctuation.
        text = re.sub(r"[ \t]+([,.;:])", r"\1", text)

        # Ensure a space after punctuation when followed immediately
        # by a normal alphabetic/number character.
        text = re.sub(
            r"([,;:])([A-Za-z0-9])",
            r"\1 \2",
            text,
        )

        return text

    @staticmethod
    def clean_line(line: str) -> str:
        """Clean an individual line while preserving its content."""
        line = line.strip()

        # Do not modify empty lines.
        if not line:
            return ""

        return line

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Apply the complete conservative preprocessing pipeline.

        The method does NOT:
        - remove legal citations
        - remove dates
        - remove paragraph numbers
        - remove names
        - remove headings
        - summarize content
        - rewrite legal meaning
        """

        if not text:
            return ""

        original_length = len(text)

        text = cls.normalize_unicode(text)
        text = cls.normalize_line_endings(text)
        text = cls.remove_trailing_whitespace(text)
        text = cls.normalize_spaces(text)
        text = cls.fix_spacing_around_punctuation(text)

        # Clean individual lines.
        text = "\n".join(
            cls.clean_line(line)
            for line in text.split("\n")
        )

        text = cls.normalize_blank_lines(text)

        # Remove whitespace around the entire document.
        text = text.strip()

        logger.debug(
            "text_preprocessing_complete",
            original_characters=original_length,
            cleaned_characters=len(text),
        )

        return text

    @classmethod
    def preprocess_page(cls, text: str) -> dict:
        """Preprocess one document page."""

        cleaned_text = cls.clean_text(text)

        return {
            "raw_text": text,
            "processed_text": cleaned_text,
            "original_char_count": len(text),
            "processed_char_count": len(cleaned_text),
        }

    @classmethod
    def preprocess_document(
        cls,
        pages: list[dict],
    ) -> list[dict]:
        """
        Preprocess multiple document pages.

        Each input dictionary should contain:
            page_number
            raw_text

        Returns the same page information with processed text added.
        """

        processed_pages = []

        for page in pages:
            result = cls.preprocess_page(
                page.get("raw_text", "")
            )

            processed_pages.append(
                {
                    **page,
                    "processed_text": result["processed_text"],
                    "original_char_count": result[
                        "original_char_count"
                    ],
                    "processed_char_count": result[
                        "processed_char_count"
                    ],
                }
            )

        logger.info(
            "document_preprocessing_complete",
            pages=len(processed_pages),
        )

        return processed_pages