# src/pdf_parser.py — Research Paper PDF Parser
import fitz  # pymupdf
import re
from dataclasses import dataclass


@dataclass
class ParsedPaper:
    title: str
    full_text: str
    sections: dict[str, str]
    word_count: int
    page_count: int


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from uploaded PDF file."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text()
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def extract_page_count(uploaded_file_bytes: bytes) -> int:
    """Get page count from PDF bytes."""
    doc = fitz.open(stream=uploaded_file_bytes, filetype="pdf")
    count = doc.page_count
    doc.close()
    return count


def detect_title(text: str, filename: str) -> str:
    """
    Try to detect paper title from first few lines.
    Falls back to filename.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # Title is usually one of the first non-empty lines, longer than 10 chars
    for line in lines[:8]:
        if 10 < len(line) < 200 and not line.startswith("http"):
            return line
    # Fallback to filename
    return filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()


def extract_sections(text: str) -> dict[str, str]:
    """
    Extract common research paper sections.
    Returns dict of {section_name: content}.
    """
    section_patterns = [
        r'\b(abstract)\b',
        r'\b(introduction)\b',
        r'\b(related work)\b',
        r'\b(methodology|methods|method)\b',
        r'\b(experiments?|experimental setup)\b',
        r'\b(results?)\b',
        r'\b(discussion)\b',
        r'\b(conclusion)\b',
        r'\b(references?|bibliography)\b',
    ]

    sections = {}
    text_lower = text.lower()

    for pattern in section_patterns:
        match = re.search(pattern, text_lower)
        if match:
            section_name = match.group(1).title()
            start = match.start()
            # Find next section or end
            next_start = len(text)
            for other_pattern in section_patterns:
                other_match = re.search(other_pattern, text_lower[start + 10:])
                if other_match:
                    candidate = start + 10 + other_match.start()
                    if candidate < next_start:
                        next_start = candidate

            content = text[start:next_start].strip()
            if len(content) > 50:
                sections[section_name] = content[:2000]  # Cap per section

    return sections


def parse_paper(uploaded_file, filename: str) -> ParsedPaper:
    """Full pipeline: extract + parse a research paper PDF."""
    full_text = extract_text_from_pdf(uploaded_file)
    title = detect_title(full_text, filename)
    sections = extract_sections(full_text)
    word_count = len(full_text.split())

    return ParsedPaper(
        title=title,
        full_text=full_text,
        sections=sections,
        word_count=word_count,
        page_count=len([p for p in full_text.split('\f') if p.strip()]) or 1
    )