# src/pdf_parser.py — Research Paper PDF Parser + Figure Extractor
import fitz  # pymupdf
import re
import base64
from dataclasses import dataclass, field


@dataclass
class Figure:
    fig_number: str        # e.g. "3"
    fig_label: str         # e.g. "Fig. 3"
    caption: str           # e.g. "Architecture of classical graph convolutional neural network"
    image_base64: str      # base64 encoded PNG
    page: int              # page number


@dataclass
class ParsedPaper:
    title: str
    full_text: str
    sections: dict
    word_count: int
    page_count: int
    figures: list = field(default_factory=list)  # list of Figure objects


# ─────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text()
        text = re.sub(r'\n{3,}', '\n\n', text)
        pages.append(text)
    doc.close()
    return "\n\n".join(pages)


# ─────────────────────────────────────────────
# FIGURE CAPTION EXTRACTION FROM TEXT
# ─────────────────────────────────────────────
def extract_figure_captions(full_text: str) -> dict:
    """
    Find all figure captions in text.
    Returns dict: {"3": "Architecture of classical graph...", "4": "Hybrid quantum..."}
    Matches patterns like:
      Fig. 3 — Architecture of...
      Figure 3: Architecture of...
      Fig 3. Architecture of...
    """
    captions = {}

    # Pattern to match Fig/Figure followed by number and caption text
    patterns = [
        r'[Ff]ig(?:ure)?\.?\s*(\d+)\s*[—\-–:\.]\s*([^\n]{5,120})',
        r'[Ff]ig(?:ure)?\s+(\d+)[\.:]?\s*([^\n]{5,120})',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, full_text)
        for match in matches:
            fig_num = match.group(1)
            caption_text = match.group(2).strip()
            # Clean up caption
            caption_text = re.sub(r'\s+', ' ', caption_text)
            if fig_num not in captions and len(caption_text) > 5:
                captions[fig_num] = caption_text

    return captions


# ─────────────────────────────────────────────
# IMAGE EXTRACTION FROM PDF
# ─────────────────────────────────────────────
def _get_page_captions(page_text: str) -> list:
    """Extract figure numbers mentioned on a specific page."""
    patterns = [
        r'[Ff]ig(?:ure)?\.?\s*(\d+)',
    ]
    nums = []
    for pattern in patterns:
        for m in re.finditer(pattern, page_text):
            nums.append(m.group(1))
    return list(dict.fromkeys(nums))  # deduplicated, order preserved


def extract_figures_from_pdf(pdf_bytes: bytes, full_text: str) -> list:
    """
    Extract figures from PDF and match them to captions by page proximity.
    - Skips logos/icons (too small or near-square tiny images on page 1)
    - Matches each image to the figure number referenced on the same page
    - Falls back to sequential numbering for unmatched images
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_captions = extract_figure_captions(full_text)

    figures = []
    used_fig_nums = set()
    fallback_counter = 100  # high number so fallbacks don't clash with real figs

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        if not image_list:
            continue

        # Get figure numbers referenced on this page
        page_text = page.get_text()
        page_fig_nums = _get_page_captions(page_text)

        # Find unused figure nums referenced on this page
        available_nums = [n for n in page_fig_nums if n not in used_fig_nums]

        img_idx = 0
        for img in image_list:
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Skip small images
                if width < 100 or height < 100:
                    continue

                # Skip images that look like logos:
                # - On page 1 (likely journal header logos)
                # - Very wide/narrow aspect ratio (banners)
                aspect = width / height if height > 0 else 0
                if page_num == 0 and (aspect > 4 or aspect < 0.25):
                    continue
                # Also skip page 1 images that are smaller than 200x200
                # (journal logos are typically small)
                if page_num == 0 and width < 200 and height < 200:
                    continue

                img_b64 = base64.b64encode(image_bytes).decode("utf-8")

                # Try to assign a real figure number from this page
                if img_idx < len(available_nums):
                    fig_num = available_nums[img_idx]
                    caption = all_captions.get(fig_num, "")
                    used_fig_nums.add(fig_num)
                else:
                    # Fallback — no figure number found on this page
                    fig_num = str(fallback_counter)
                    caption = ""
                    fallback_counter += 1

                figures.append(Figure(
                    fig_number=fig_num,
                    fig_label=f"Fig. {fig_num}",
                    caption=caption,
                    image_base64=img_b64,
                    page=page_num + 1
                ))
                img_idx += 1

            except Exception:
                continue

    doc.close()

    # Sort by figure number (real figures first, fallbacks last)
    def sort_key(f):
        try:
            return int(f.fig_number)
        except ValueError:
            return 9999

    figures.sort(key=sort_key)
    return figures


# ─────────────────────────────────────────────
# FIGURE SEARCH — match question to figures
# ─────────────────────────────────────────────

# Synonym map — query terms → related caption words
SYNONYM_MAP = {
    "architecture": ["architecture", "network", "model", "structure", "framework", "pipeline", "layer", "gcnn", "qgnn", "hybrid"],
    "methodology": ["architecture", "network", "framework", "pipeline", "methodology", "proposed", "method", "approach", "model", "structure"],
    "proposed": ["architecture", "hybrid", "network", "proposed", "framework", "qgnn", "quantum"],
    "method": ["architecture", "framework", "pipeline", "network", "method", "model"],
    "results": ["accuracy", "loss", "confusion", "matrix", "performance", "results", "comparison"],
    "accuracy": ["accuracy", "loss", "performance", "results", "training", "validation"],
    "loss": ["loss", "accuracy", "training", "validation", "results"],
    "confusion": ["confusion", "matrix", "classification"],
    "performance": ["accuracy", "loss", "confusion", "matrix", "performance", "table"],
    "comparison": ["accuracy", "loss", "performance", "comparison", "classical", "quantum"],
    "algorithm": ["algorithm", "circuit", "encoding", "quantum", "qgnn"],
    "circuit": ["circuit", "quantum", "encoding", "qubit"],
    "quantum": ["quantum", "circuit", "encoding", "qgnn", "qubit"],
    "encoding": ["encoding", "circuit", "data", "quantum"],
    "dataset": ["dataset", "class", "distribution", "tumor", "brain"],
    "data": ["dataset", "class", "distribution", "brain", "image", "tumor", "preprocessing"],
    "preprocessing": ["brain", "tumor", "image", "normalized", "resized", "original"],
    "brain": ["brain", "tumor", "mri", "image", "classification"],
    "tumor": ["tumor", "brain", "mri", "classification", "image"],
    "classification": ["classification", "tumor", "brain", "confusion", "matrix"],
    "training": ["training", "accuracy", "loss", "validation"],
    "validation": ["validation", "accuracy", "loss", "training"],
    "graph": ["architecture", "network", "gcnn", "qgnn", "graph"],
    "node": ["architecture", "network", "gcnn", "qgnn"],
    "embedding": ["encoding", "circuit", "quantum", "embedding"],
}


def find_relevant_figures(question: str, figures: list, top_k: int = 2) -> list:
    """
    Find figures most relevant to the question.
    Uses caption keyword matching + synonym expansion.
    Returns list of Figure objects, max top_k.
    """
    if not figures:
        return []

    question_lower = question.lower()
    question_words = set(re.sub(r'[^\w\s]', '', question_lower).split())

    stop_words = {
        'what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'how',
        'does', 'do', 'this', 'that', 'explain', 'show', 'me', 'tell',
        'about', 'describe', 'paper', 'used', 'and', 'or', 'are',
        'was', 'were', 'it', 'its', 'by', 'with', 'from', 'to', 'on'
    }
    question_words = question_words - stop_words

    if not question_words:
        return []

    # Expand keywords using synonyms
    expanded_words = set(question_words)
    for word in question_words:
        if word in SYNONYM_MAP:
            expanded_words.update(SYNONYM_MAP[word])

    scored = []
    for fig in figures:
        caption_lower = fig.caption.lower() if fig.caption else ""

        # Score: direct caption match (weight 3) + synonym match (weight 1)
        direct_score = sum(3 for word in question_words if word in caption_lower)
        synonym_score = sum(1 for word in expanded_words if word in caption_lower)
        total_score = direct_score + synonym_score

        # Partial match — check if any question word is a substring of caption words
        partial_score = sum(
            1 for qw in question_words
            if any(qw in cw or cw in qw for cw in caption_lower.split() if len(cw) > 3)
        )
        total_score += partial_score

        if total_score > 0:
            scored.append((total_score, fig))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [fig for score, fig in scored[:top_k]]


# ─────────────────────────────────────────────
# TITLE DETECTION
# ─────────────────────────────────────────────
def detect_title(text: str, filename: str) -> str:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:8]:
        if 10 < len(line) < 200 and not line.startswith("http"):
            return line
    return filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()


# ─────────────────────────────────────────────
# SECTION EXTRACTION
# ─────────────────────────────────────────────
def extract_sections(text: str) -> dict:
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
            next_start = len(text)
            for other_pattern in section_patterns:
                other_match = re.search(other_pattern, text_lower[start + 10:])
                if other_match:
                    candidate = start + 10 + other_match.start()
                    if candidate < next_start:
                        next_start = candidate
            content = text[start:next_start].strip()
            if len(content) > 50:
                sections[section_name] = content[:2000]
    return sections


# ─────────────────────────────────────────────
# MAIN PARSE FUNCTION
# ─────────────────────────────────────────────
def parse_paper(uploaded_file, filename: str) -> ParsedPaper:
    """Full pipeline: extract text + figures + parse paper."""
    pdf_bytes = uploaded_file.read()

    full_text = extract_text_from_pdf(pdf_bytes)
    title = detect_title(full_text, filename)
    sections = extract_sections(full_text)
    word_count = len(full_text.split())
    figures = extract_figures_from_pdf(pdf_bytes, full_text)

    return ParsedPaper(
        title=title,
        full_text=full_text,
        sections=sections,
        word_count=word_count,
        page_count=len([p for p in full_text.split('\f') if p.strip()]) or 1,
        figures=figures
    )