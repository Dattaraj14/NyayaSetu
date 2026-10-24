"""
Legal document structure detection.

Designed for Indian court judgments and other legal PDFs where
paragraph numbering may appear on a separate line:

    1.
    This is the paragraph text...

or on the same line:

    1. This is the paragraph text...

The detector also preserves paragraphs across page boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class LegalParagraph:
    number: int | None
    content: str
    start_page: int
    end_page: int
    order_index: int
    section_type: str = "general"
    section_title: str = "General"
    ocr_artifacts: list[str] = field(default_factory=list)

    @property
    def paragraph_number(self) -> int | None:
        return self.number

    @property
    def text(self) -> str:
        return self.content


@dataclass
class LegalSection:
    section_type: str
    section_title: str
    content: str
    start_page: int
    end_page: int
    order_index: int
    paragraphs: list[LegalParagraph] = field(
        default_factory=list
    )


@dataclass
class LegalStructure:
    sections: list[LegalSection]
    paragraphs: list[LegalParagraph]
    section_source: str = "inferred"


# ============================================================
# STRUCTURE DETECTOR
# ============================================================


class StructureDetector:
    """
    Detects numbered paragraphs, sections and legal structure.

    The most important feature is support for paragraph numbers
    appearing alone on a line.
    """

    # Paragraph number on its own line:
    #
    #   1.
    #   12.
    #   123.
    #
    STANDALONE_PARAGRAPH_RE = re.compile(
        r"^\s*(\d{1,3})\.\s*$"
    )

    # Paragraph number and text on the same line:
    #
    #   1. This is the paragraph.
    #   12. The appellant contended...
    #
    INLINE_PARAGRAPH_RE = re.compile(
        r"^\s*(\d{1,3})\.\s+(.+?)\s*$"
    )

    PAGE_NUMBER_RE = re.compile(
        r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
        re.IGNORECASE,
    )

    # Common lines that should not become paragraphs.
    IGNORE_LINE_PATTERNS = [
        re.compile(
            r"^\s*Digitally signed by\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*Date:\s*",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*Reason:\s*",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*Signature Not Verified\s*$",
            re.IGNORECASE,
        ),
    ]

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def detect(
        self,
        pages: Iterable[Any],
    ) -> LegalStructure:
        """
        Detect structure from document pages.

        `pages` may contain dictionaries or objects with:
            page_number
            raw_text
            processed_text
        """

        normalized_pages = self._normalize_pages(pages)

        paragraphs = self._extract_paragraph_candidates(
            normalized_pages
        )

        self._assign_sections(
            paragraphs,
            normalized_pages,
        )

        sections = self._build_sections(paragraphs)

        return LegalStructure(
            sections=sections,
            paragraphs=paragraphs,
            section_source="inferred",
        )

    # Compatibility alias.
    detect_structure = detect

    # --------------------------------------------------------
    # Page normalization
    # --------------------------------------------------------

    def _normalize_pages(
        self,
        pages: Iterable[Any],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []

        for index, page in enumerate(pages, start=1):
            if isinstance(page, dict):
                page_number = page.get(
                    "page_number",
                    index,
                )

                text = (
                    page.get("processed_text")
                    or page.get("raw_text")
                    or ""
                )

            else:
                page_number = getattr(
                    page,
                    "page_number",
                    index,
                )

                text = (
                    getattr(page, "processed_text", None)
                    or getattr(page, "raw_text", None)
                    or ""
                )

            normalized.append(
                {
                    "page_number": int(page_number),
                    "text": str(text),
                }
            )

        normalized.sort(
            key=lambda item: item["page_number"]
        )

        return normalized

    # --------------------------------------------------------
    # Paragraph extraction
    # --------------------------------------------------------

    def _extract_paragraph_candidates(
        self,
        pages: list[dict[str, Any]],
    ) -> list[LegalParagraph]:

        paragraphs: list[LegalParagraph] = []

        current_number: int | None = None
        current_lines: list[str] = []
        current_start_page: int | None = None
        current_end_page: int | None = None
        current_order_index = 0

        # ----------------------------------------------------
        # Helpers
        # ----------------------------------------------------

        def flush_current() -> None:
            nonlocal current_number
            nonlocal current_lines
            nonlocal current_start_page
            nonlocal current_end_page
            nonlocal current_order_index

            if not current_lines:
                current_number = None
                current_start_page = None
                current_end_page = None
                return

            content = self._clean_paragraph_text(
                current_lines
            )

            if not content:
                current_number = None
                current_lines = []
                current_start_page = None
                current_end_page = None
                return

            start_page = (
                current_start_page
                if current_start_page is not None
                else current_end_page or 1
            )

            end_page = (
                current_end_page
                if current_end_page is not None
                else start_page
            )

            current_order_index += 1

            paragraphs.append(
                LegalParagraph(
                    number=current_number,
                    content=content,
                    start_page=start_page,
                    end_page=end_page,
                    order_index=current_order_index,
                    ocr_artifacts=self._detect_ocr_artifacts(
                        content
                    ),
                )
            )

            current_number = None
            current_lines = []
            current_start_page = None
            current_end_page = None

        # ----------------------------------------------------
        # Process every page
        # ----------------------------------------------------

        for page in pages:
            page_number = page["page_number"]
            text = page["text"]

            lines = text.splitlines()

            for raw_line in lines:
                line = raw_line.strip()

                if not line:
                    # Blank lines do NOT end the paragraph.
                    # Legal paragraphs frequently contain
                    # formatting whitespace.
                    continue

                # Ignore PDF page footer.
                if self.PAGE_NUMBER_RE.match(line):
                    continue

                # Ignore digital-signature artifacts.
                if self._should_ignore_line(line):
                    continue

                # ------------------------------------------------
                # CASE 1:
                #
                # Number on a separate line:
                #
                # 14.
                # Aggrieved by the said decision...
                # ------------------------------------------------

                standalone_match = (
                    self.STANDALONE_PARAGRAPH_RE.match(line)
                )

                if standalone_match:
                    # Close previous paragraph.
                    flush_current()

                    current_number = int(
                        standalone_match.group(1)
                    )

                    current_lines = []

                    current_start_page = page_number
                    current_end_page = page_number

                    continue

                # ------------------------------------------------
                # CASE 2:
                #
                # Number and text on same line:
                #
                # 14. Aggrieved by the said decision...
                # ------------------------------------------------

                inline_match = (
                    self.INLINE_PARAGRAPH_RE.match(line)
                )

                if inline_match:
                    # Close previous paragraph.
                    flush_current()

                    current_number = int(
                        inline_match.group(1)
                    )

                    first_text = inline_match.group(2).strip()

                    current_lines = [first_text]

                    current_start_page = page_number
                    current_end_page = page_number

                    continue

                # ------------------------------------------------
                # CASE 3:
                #
                # Continuation line.
                # ------------------------------------------------

                if current_number is not None:

                    # Preserve line content.
                    current_lines.append(line)

                    current_end_page = page_number

                else:
                    # Text before the first numbered paragraph.
                    #
                    # This includes:
                    # - court name
                    # - case title
                    # - judgment heading
                    # - parties
                    #
                    # We intentionally don't create a numbered
                    # paragraph for this material.
                    continue

        # Flush last paragraph.
        flush_current()

        return paragraphs

    # --------------------------------------------------------
    # Cleaning
    # --------------------------------------------------------

    def _clean_paragraph_text(
        self,
        lines: list[str],
    ) -> str:

        cleaned: list[str] = []

        for line in lines:
            line = re.sub(
                r"\s+",
                " ",
                line,
            ).strip()

            if not line:
                continue

            cleaned.append(line)

        return " ".join(cleaned).strip()

    # --------------------------------------------------------
    # OCR artifact detection
    # --------------------------------------------------------

    def _detect_ocr_artifacts(
        self,
        text: str,
    ) -> list[str]:

        artifacts: list[str] = []

        # Common OCR corruption patterns.
        if re.search(
            r"\b[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\b",
            text,
        ):
            artifacts.append(
                "possible_character_fragmentation"
            )

        if re.search(
            r"\b(?:rn|vv|ii|ll){2,}\b",
            text,
            re.IGNORECASE,
        ):
            artifacts.append(
                "possible_ocr_character_confusion"
            )

        return artifacts

    # --------------------------------------------------------
    # Ignore lines
    # --------------------------------------------------------

    def _should_ignore_line(
        self,
        line: str,
    ) -> bool:

        for pattern in self.IGNORE_LINE_PATTERNS:
            if pattern.match(line):
                return True

        return False

    # --------------------------------------------------------
    # Section assignment
    # --------------------------------------------------------

    def _assign_sections(
        self,
        paragraphs: list[LegalParagraph],
        pages: list[dict[str, Any]],
    ) -> None:

        if not paragraphs:
            return

        # We use explicit legal section headings when present.
        headings = self._find_section_headings(
            pages
        )

        for paragraph in paragraphs:
            section_type, section_title = (
                self._infer_section_for_paragraph(
                    paragraph,
                    headings,
                )
            )

            paragraph.section_type = section_type
            paragraph.section_title = section_title

    # --------------------------------------------------------
    # Heading detection
    # --------------------------------------------------------

    def _find_section_headings(
        self,
        pages: list[dict[str, Any]],
    ) -> list[tuple[int, int, str]]:

        headings: list[tuple[int, int, str]] = []

        heading_keywords = {
            "facts": (
                "FACTS",
                "BACKGROUND",
                "BRIEF FACTS",
                "FACTUAL BACKGROUND",
            ),
            "arguments": (
                "SUBMISSIONS",
                "CONTENTIONS",
                "ARGUMENTS",
                "SUBMISSIONS ON BEHALF",
            ),
            "reasoning": (
                "DISCUSSION",
                "ANALYSIS",
                "REASONS",
                "FINDINGS",
            ),
            "decision": (
                "ORDER",
                "CONCLUSION",
                "DECISION",
                "RESULT",
            ),
        }

        for page in pages:
            page_number = page["page_number"]

            for raw_line in page["text"].splitlines():

                line = raw_line.strip()

                if not line:
                    continue

                normalized = re.sub(
                    r"[^A-Z ]",
                    "",
                    line.upper(),
                ).strip()

                if not normalized:
                    continue

                if len(normalized) > 80:
                    continue

                for section_type, candidates in (
                    heading_keywords.items()
                ):
                    if normalized in candidates:
                        headings.append(
                            (
                                page_number,
                                len(headings),
                                line.strip(),
                            )
                        )

                        break

        return headings

    # --------------------------------------------------------
    # Section inference
    # --------------------------------------------------------

    def _infer_section_for_paragraph(
        self,
        paragraph: LegalParagraph,
        headings: list[tuple[int, int, str]],
    ) -> tuple[str, str]:

        # Find the most recent heading occurring before
        # the paragraph.
        applicable_heading = None

        for heading_page, _, heading_title in headings:
            if heading_page <= paragraph.start_page:
                applicable_heading = (
                    heading_page,
                    heading_title,
                )
            else:
                break

        if applicable_heading is not None:
            title = applicable_heading[1]
            normalized = title.upper()

            if any(
                keyword in normalized
                for keyword in (
                    "ARGUMENT",
                    "SUBMISSION",
                    "CONTENTION",
                )
            ):
                return "arguments", title

            if any(
                keyword in normalized
                for keyword in (
                    "DISCUSSION",
                    "ANALYSIS",
                    "REASONS",
                    "FINDING",
                )
            ):
                return "reasoning", title

            if any(
                keyword in normalized
                for keyword in (
                    "ORDER",
                    "CONCLUSION",
                    "DECISION",
                    "RESULT",
                )
            ):
                return "decision", title

            if any(
                keyword in normalized
                for keyword in (
                    "FACT",
                    "BACKGROUND",
                )
            ):
                return "facts", title

        # ----------------------------------------------------
        # No explicit heading.
        #
        # Use paragraph-number/content heuristics as a
        # conservative fallback.
        # ----------------------------------------------------

        number = paragraph.number
        text = paragraph.content.lower()

        if number is not None:

            if number <= 10:
                return (
                    "preliminary_discussion",
                    "Preliminary Discussion",
                )

            if number <= 13:
                return (
                    "facts",
                    "Facts",
                )

            if 14 <= number <= 16:
                return (
                    "arguments",
                    "Arguments",
                )

            if any(
                phrase in text
                for phrase in (
                    "learned senior counsel",
                    "counsel for the appellant",
                    "counsel appearing",
                    "submitted that",
                    "contended that",
                    "it was argued",
                )
            ):
                return (
                    "arguments",
                    "Arguments",
                )

            if any(
                phrase in text
                for phrase in (
                    "we hold",
                    "we therefore",
                    "accordingly",
                    "the appeal is",
                    "the judgment is set aside",
                    "we direct",
                    "we order",
                )
            ):
                return (
                    "decision",
                    "Decision",
                )

            return (
                "reasoning",
                "Reasoning",
            )

        return (
            "general",
            "General",
        )

    # --------------------------------------------------------
    # Section building
    # --------------------------------------------------------

    def _build_sections(
        self,
        paragraphs: list[LegalParagraph],
    ) -> list[LegalSection]:

        if not paragraphs:
            return []

        sections: list[LegalSection] = []

        current_type = paragraphs[0].section_type
        current_title = paragraphs[0].section_title
        current_paragraphs: list[LegalParagraph] = []

        section_index = 0

        def flush_section() -> None:
            nonlocal current_paragraphs
            nonlocal section_index

            if not current_paragraphs:
                return

            section_index += 1

            content = "\n\n".join(
                paragraph.content
                for paragraph in current_paragraphs
            )

            sections.append(
                LegalSection(
                    section_type=current_type,
                    section_title=current_title,
                    content=content,
                    start_page=min(
                        paragraph.start_page
                        for paragraph in current_paragraphs
                    ),
                    end_page=max(
                        paragraph.end_page
                        for paragraph in current_paragraphs
                    ),
                    order_index=section_index,
                    paragraphs=list(
                        current_paragraphs
                    ),
                )
            )

            current_paragraphs = []

        for paragraph in paragraphs:

            if (
                paragraph.section_type != current_type
                or paragraph.section_title
                != current_title
            ):
                flush_section()

                current_type = paragraph.section_type
                current_title = paragraph.section_title

            current_paragraphs.append(
                paragraph
            )

        flush_section()

        return sections


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

LegalStructureDetector = StructureDetector


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def detect_legal_structure(
    pages: Iterable[Any],
) -> LegalStructure:
    """
    Convenience wrapper.
    """

    detector = StructureDetector()

    return detector.detect(pages)
# ============================================================
# BACKWARD-COMPATIBILITY ALIASES
# ============================================================

DetectedParagraph = LegalParagraph
DetectedSection = LegalSection
DetectedStructure = LegalStructure

LegalStructureDetector = StructureDetector


def detect_legal_structure(
    pages: Iterable[Any],
) -> LegalStructure:
    """
    Convenience wrapper.
    """

    detector = StructureDetector()

    return detector.detect(pages)