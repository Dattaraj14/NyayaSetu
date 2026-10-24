"""Legal document chunking pipeline.

Creates retrieval-friendly chunks from detected legal paragraphs
while preserving paragraph, section, and page metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.logging_config import get_logger


logger = get_logger("chunking")


@dataclass
class LegalChunk:
    """A retrieval-ready chunk of a legal document."""

    chunk_index: int
    content: str
    token_count: int

    start_page: int | None
    end_page: int | None

    section_type: str | None
    section_title: str | None

    paragraph_numbers: list[int]
    paragraph_sequences: list[int]

    ocr_artifacts: list[dict[str, Any]]


class LegalChunker:
    """Create paragraph-aware chunks for legal documents."""

    # These values are deliberately conservative for the MVP.
    #
    # Later, when we add the actual embedding model, we can tune
    # them using retrieval experiments.
    TARGET_TOKENS = 700
    MAX_TOKENS = 1000

    def __init__(
        self,
        target_tokens: int = TARGET_TOKENS,
        max_tokens: int = MAX_TOKENS,
    ) -> None:

        if target_tokens <= 0:
            raise ValueError(
                "target_tokens must be greater than zero"
            )

        if max_tokens < target_tokens:
            raise ValueError(
                "max_tokens must be greater than or equal "
                "to target_tokens"
            )

        self.target_tokens = target_tokens
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count without requiring a tokenizer package.

        This is intentionally an approximation.

        It is sufficient for chunk-size control during the MVP.
        The actual embedding model/tokenizer can later provide
        model-specific token counts.
        """

        if not text:
            return 0

        words = text.split()

        if not words:
            return 0

        # A conservative approximation for English/legal text.
        estimated = int(len(words) * 1.3)

        return max(1, estimated)

    # ------------------------------------------------------------------
    # Section mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _find_section(
        paragraph: dict[str, Any],
        sections: list[Any],
    ) -> Any | None:
        """Find the most appropriate section for a paragraph.

        Sections are currently page-range based because the structure
        detector exposes section page boundaries rather than explicit
        paragraph indexes.

        When multiple sections contain the same boundary page, prefer
        the section whose range is the most specific.
        """

        paragraph_page = paragraph.get("start_page")

        if paragraph_page is None:
            return None

        candidates: list[Any] = []

        for section in sections:

            start_page = getattr(
                section,
                "start_page",
                None,
            )

            end_page = getattr(
                section,
                "end_page",
                None,
            )

            if start_page is None:
                continue

            if end_page is None:
                end_page = start_page

            if start_page <= paragraph_page <= end_page:
                candidates.append(section)

        if not candidates:
            return None

        # Prefer the smallest page range when sections overlap.
        candidates.sort(
            key=lambda item: (
                (
                    getattr(item, "end_page", 0)
                    or 0
                )
                - (
                    getattr(item, "start_page", 0)
                    or 0
                ),
                getattr(item, "order_index", 0),
            )
        )

        return candidates[0]

    # ------------------------------------------------------------------
    # Paragraph normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_paragraph(
        paragraph: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Normalize one detector paragraph."""

        text = str(
            paragraph.get("text") or ""
        ).strip()

        if not text:
            return None

        sequence_number = paragraph.get(
            "sequence_number"
        )

        paragraph_number = paragraph.get(
            "paragraph_number"
        )

        ocr_artifact = bool(
            paragraph.get(
                "ocr_artifact",
                False,
            )
        )

        return {
            "text": text,
            "sequence_number": sequence_number,
            "paragraph_number": paragraph_number,
            "source_paragraph_number": paragraph.get(
                "source_paragraph_number"
            ),
            "expected_paragraph_number": paragraph.get(
                "expected_paragraph_number"
            ),
            "ocr_artifact": ocr_artifact,
            "ocr_artifact_reason": paragraph.get(
                "ocr_artifact_reason"
            ),
            "start_page": paragraph.get(
                "start_page"
            ),
            "end_page": paragraph.get(
                "end_page"
            ),
        }

    # ------------------------------------------------------------------
    # Chunk creation
    # ------------------------------------------------------------------

    def create_chunks(
        self,
        paragraphs: list[dict[str, Any]],
        sections: list[Any],
    ) -> list[LegalChunk]:
        """Create legal chunks from detected paragraphs."""

        normalized_paragraphs: list[
            dict[str, Any]
        ] = []

        for paragraph in paragraphs:

            normalized = (
                self._normalize_paragraph(
                    paragraph
                )
            )

            if normalized is not None:
                normalized_paragraphs.append(
                    normalized
                )

        if not normalized_paragraphs:
            logger.warning(
                "no_paragraphs_available_for_chunking"
            )

            return []

        chunks: list[LegalChunk] = []

        current_items: list[
            dict[str, Any]
        ] = []

        current_tokens = 0

        current_section: Any | None = None

        def flush_current_chunk() -> None:
            nonlocal current_items
            nonlocal current_tokens
            nonlocal current_section

            if not current_items:
                return

            text_parts: list[str] = []

            paragraph_numbers: list[int] = []
            paragraph_sequences: list[int] = []

            ocr_artifacts: list[
                dict[str, Any]
            ] = []

            start_page: int | None = None
            end_page: int | None = None

            for item in current_items:

                text_parts.append(
                    item["text"]
                )

                paragraph_number = item.get(
                    "paragraph_number"
                )

                if isinstance(
                    paragraph_number,
                    int,
                ):
                    paragraph_numbers.append(
                        paragraph_number
                    )

                sequence_number = item.get(
                    "sequence_number"
                )

                if isinstance(
                    sequence_number,
                    int,
                ):
                    paragraph_sequences.append(
                        sequence_number
                    )

                if item.get(
                    "ocr_artifact",
                    False,
                ):
                    ocr_artifacts.append(
                        {
                            "sequence_number": (
                                sequence_number
                            ),
                            "source_paragraph_number": (
                                item.get(
                                    "source_paragraph_number"
                                )
                            ),
                            "expected_paragraph_number": (
                                item.get(
                                    "expected_paragraph_number"
                                )
                            ),
                            "reason": (
                                item.get(
                                    "ocr_artifact_reason"
                                )
                            ),
                            "page": (
                                item.get(
                                    "start_page"
                                )
                            ),
                        }
                    )

                item_start_page = item.get(
                    "start_page"
                )

                item_end_page = item.get(
                    "end_page"
                )

                if item_start_page is not None:
                    if (
                        start_page is None
                        or item_start_page < start_page
                    ):
                        start_page = item_start_page

                if item_end_page is not None:
                    if (
                        end_page is None
                        or item_end_page > end_page
                    ):
                        end_page = item_end_page

            chunk_content = "\n\n".join(
                text_parts
            ).strip()

            if not chunk_content:
                current_items = []
                current_tokens = 0
                current_section = None
                return

            section_type = None
            section_title = None

            if current_section is not None:

                section_type = getattr(
                    current_section,
                    "section_type",
                    None,
                )

                section_title = getattr(
                    current_section,
                    "title",
                    None,
                )

            chunk = LegalChunk(
                chunk_index=len(chunks),
                content=chunk_content,
                token_count=self.estimate_tokens(
                    chunk_content
                ),
                start_page=start_page,
                end_page=end_page,
                section_type=section_type,
                section_title=section_title,
                paragraph_numbers=(
                    paragraph_numbers
                ),
                paragraph_sequences=(
                    paragraph_sequences
                ),
                ocr_artifacts=ocr_artifacts,
            )

            chunks.append(chunk)

            current_items = []
            current_tokens = 0
            current_section = None

        # --------------------------------------------------------------
        # Build chunks paragraph by paragraph.
        # --------------------------------------------------------------

        for paragraph in normalized_paragraphs:

            paragraph_section = (
                self._find_section(
                    paragraph,
                    sections,
                )
            )

            paragraph_tokens = (
                self.estimate_tokens(
                    paragraph["text"]
                )
            )

            # ----------------------------------------------------------
            # Start a new chunk.
            # ----------------------------------------------------------

            if not current_items:

                current_items.append(
                    paragraph
                )

                current_tokens = (
                    paragraph_tokens
                )

                current_section = (
                    paragraph_section
                )

                continue

            # ----------------------------------------------------------
            # Detect section transition.
            #
            # We avoid combining different legal sections into one
            # chunk unless a single paragraph itself is larger than
            # the target size.
            # ----------------------------------------------------------

            section_changed = (
                paragraph_section
                is not current_section
            )

            # ----------------------------------------------------------
            # Check whether adding the paragraph would exceed max size.
            # ----------------------------------------------------------

            exceeds_max = (
                current_tokens
                + paragraph_tokens
                > self.max_tokens
            )

            reaches_target = (
                current_tokens
                >= self.target_tokens
            )

            if section_changed:

                flush_current_chunk()

                current_items.append(
                    paragraph
                )

                current_tokens = (
                    paragraph_tokens
                )

                current_section = (
                    paragraph_section
                )

                continue

            if exceeds_max:

                flush_current_chunk()

                current_items.append(
                    paragraph
                )

                current_tokens = (
                    paragraph_tokens
                )

                current_section = (
                    paragraph_section
                )

                continue

            # ----------------------------------------------------------
            # Add paragraph to current chunk.
            # ----------------------------------------------------------

            current_items.append(
                paragraph
            )

            current_tokens += paragraph_tokens

            # ----------------------------------------------------------
            # Once target size has been reached, wait until the current
            # paragraph boundary and then close the chunk.
            # ----------------------------------------------------------

            if reaches_target:

                flush_current_chunk()

        # Flush final chunk.

        flush_current_chunk()

        logger.info(
            "legal_chunking_complete",
            paragraphs=len(
                normalized_paragraphs
            ),
            chunks=len(chunks),
        )

        return chunks