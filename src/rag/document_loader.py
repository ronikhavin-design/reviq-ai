"""
Document Loader: Phase 4b of the RevIQ AI RAG Copilot.

Reads Markdown reports from data/reports/markdown/ and splits them into
chunks at ## heading boundaries. Each chunk carries metadata so that
retrieved results can be attributed back to the source document and section.

No LLM, no embedding, and no file writes happen here. This module is purely
about reading and structuring the documents.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import REPORTS_DIR

MARKDOWN_DIR = REPORTS_DIR / "markdown"


@dataclass
class DocumentChunk:
    """A single retrievable unit from a Markdown report."""

    chunk_id: str       # e.g. "executive_summary__01"
    source_file: str    # e.g. "executive_summary.md"
    report_type: str    # e.g. "executive_summary"
    section_title: str  # e.g. "Key Metrics" or "document_header"
    content: str        # Full text of this chunk, including the ## heading line


def chunk_markdown(content: str, source_file: str) -> list[DocumentChunk]:
    """
    Split a Markdown string into chunks at ## heading boundaries.

    The text before the first ## heading is captured as a "document_header"
    chunk containing the title, generation date, and source file metadata.
    Each ## section becomes its own chunk with the heading line included in
    the content so that the section title is searchable.

    Args:
        content:     Raw Markdown string.
        source_file: Filename for metadata (e.g. "executive_summary.md").

    Returns:
        List of DocumentChunk objects, one per section plus one header chunk.
    """
    report_type = Path(source_file).stem
    heading_pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(content))

    chunks: list[DocumentChunk] = []

    header_text = content[: matches[0].start()].strip() if matches else content.strip()
    if header_text:
        chunks.append(
            DocumentChunk(
                chunk_id=f"{report_type}__00",
                source_file=source_file,
                report_type=report_type,
                section_title="document_header",
                content=header_text,
            )
        )

    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_content = content[start:end].strip()

        chunks.append(
            DocumentChunk(
                chunk_id=f"{report_type}__{i + 1:02d}",
                source_file=source_file,
                report_type=report_type,
                section_title=section_title,
                content=section_content,
            )
        )

    return chunks


def load_markdown_file(path: Path) -> list[DocumentChunk]:
    """
    Load and chunk a single Markdown file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Markdown report not found: {path}")
    content = path.read_text(encoding="utf-8")
    return chunk_markdown(content, path.name)


def load_all_reports(markdown_dir: Path = MARKDOWN_DIR) -> list[DocumentChunk]:
    """
    Load and chunk all Markdown files in a directory.

    Returns an empty list if the directory does not exist or contains no
    .md files. This is intentionally safe to call before any reports have
    been generated.
    """
    if not markdown_dir.exists():
        return []
    chunks: list[DocumentChunk] = []
    for md_file in sorted(markdown_dir.glob("*.md")):
        chunks.extend(load_markdown_file(md_file))
    return chunks
