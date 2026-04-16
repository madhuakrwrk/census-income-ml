"""Render ``reports/project_report.md`` into a polished client-ready PDF.

This exists so the client gets a PDF they can read without touching a
markdown renderer. We parse the source markdown line-by-line and lay it
out with fpdf2 — NOT by going through an HTML rasterizer, which would
require system libraries we don't want to depend on.

The renderer is deliberately lightweight:

* Recognises ``# H1``, ``## H2``, ``### H3``, ``*italic*``, ``**bold**``,
  fenced code blocks, bullet lists, numbered lists, pipe tables, and
  inline ``![alt](path)`` image references.
* Emits images with a fixed page width so figures stay legible.
* Auto-breaks pages when content overflows.

This is enough for the project report. For more complex markdown we'd
reach for pandoc + wkhtmltopdf.

Run:

    python scripts/04_render_report_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from census_income import config  # noqa: E402


SRC = config.REPORTS_DIR / "project_report.md"
DST = config.REPORTS_DIR / "project_report.pdf"
FIG_DIR = config.REPORTS_DIR / "figures"


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

PAGE_MARGIN = 13  # mm
CONTENT_WIDTH_MM = 210 - 2 * PAGE_MARGIN  # A4

# Compact — target <=10 pages total. Line heights and font sizes are tight.
STYLE = {
    "h1": ("helvetica", "B", 15, 2.5, 1.5),
    "h2": ("helvetica", "B", 11, 2.5, 1),
    "h3": ("helvetica", "B", 9.5, 1.5, 0.8),
    "body": ("helvetica", "", 8.5, 0, 1),
    "italic": ("helvetica", "I", 8, 0, 1),
    "bullet": ("helvetica", "", 8.5, 0, 1),
    "code": ("courier", "", 7.5, 0, 1),
    "caption": ("helvetica", "I", 7, 0, 1),
}
BODY_LINE_H = 3.8


class ReportPDF(FPDF):
    """Subclassed FPDF with a compact header/footer and tidy defaults."""

    def header(self):  # noqa: D401 - FPDF API
        pass  # cleaner look without a running header

    def footer(self):  # noqa: D401 - FPDF API
        self.set_y(-10)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _set_style(pdf: FPDF, name: str) -> tuple[float, float]:
    family, style, size, top_pad, bottom_pad = STYLE[name]
    pdf.set_font(family, style, size)
    return top_pad, bottom_pad


def _sanitize(text: str) -> str:
    """fpdf2 handles unicode fine with helvetica+latin-1 but a few glyphs
    outside latin-1 will crash it. Replace the ones we actually use."""
    replacements = {
        "—": "-",
        "–": "-",
        "…": "...",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "→": "->",
        "×": "x",
        "≥": ">=",
        "≤": "<=",
        "≈": "~",
        "±": "+/-",
        "×": "x",
        "≥": ">=",
        "•": "*",
        "°": "deg",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Drop any remaining non-latin1 codepoints quietly.
    return text.encode("latin-1", "replace").decode("latin-1")


def _render_inline(pdf: FPDF, text: str) -> None:
    """Minimal inline markdown — bold, italic, inline code — via write()."""
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    for piece in pattern.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            pdf.set_font("helvetica", "B", 8.5)
            pdf.write(BODY_LINE_H, _sanitize(piece[2:-2]))
        elif piece.startswith("*") and piece.endswith("*"):
            pdf.set_font("helvetica", "I", 8.5)
            pdf.write(BODY_LINE_H, _sanitize(piece[1:-1]))
        elif piece.startswith("`") and piece.endswith("`"):
            pdf.set_font("courier", "", 7.5)
            pdf.write(BODY_LINE_H, _sanitize(piece[1:-1]))
        else:
            pdf.set_font("helvetica", "", 8.5)
            pdf.write(BODY_LINE_H, _sanitize(piece))
    pdf.ln(BODY_LINE_H)


def _render_image(pdf: FPDF, path: Path, caption: str | None = None) -> None:
    """Emit an image, centred, with a caption. Page-break if it won't fit."""
    if not path.exists():
        pdf.set_text_color(200, 0, 0)
        _set_style(pdf, "body")
        pdf.multi_cell(0, 5, f"[missing figure: {path}]")
        pdf.set_text_color(0, 0, 0)
        return

    # A4 width minus margins, image gets 55% of that so multiple figures fit
    # on a page.
    target_w = CONTENT_WIDTH_MM * 0.48
    # Preserve aspect ratio — fpdf2 will set height from width automatically
    # if we don't pass h. We estimate the rendered height by opening the
    # image to know whether we need a page break.
    from PIL import Image  # lazy import; Pillow is a fpdf2 dep

    with Image.open(path) as im:
        iw, ih = im.size
        aspect = ih / iw
    target_h = target_w * aspect

    # Page break guard
    if pdf.get_y() + target_h + 8 > pdf.h - pdf.b_margin:
        pdf.add_page()

    x = (pdf.w - target_w) / 2
    pdf.image(str(path), x=x, w=target_w)
    pdf.ln(1)

    if caption:
        _set_style(pdf, "caption")
        pdf.set_text_color(110, 110, 110)
        pdf.multi_cell(0, 4, _sanitize(caption), align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)


def _render_table(pdf: FPDF, rows: list[list[str]]) -> None:
    """Pipe-table renderer. Drops the alignment row and auto-sizes columns."""
    if not rows:
        return
    header, *body = rows
    n_cols = len(header)
    col_w = CONTENT_WIDTH_MM / n_cols

    pdf.set_font("helvetica", "B", 7.5)
    pdf.set_fill_color(232, 238, 250)
    for cell in header:
        pdf.cell(col_w, 4.5, _sanitize(cell.strip()), border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("helvetica", "", 7.5)
    for row in body:
        row = (row + [""] * n_cols)[:n_cols]
        row_height = 4
        if pdf.get_y() + row_height > pdf.h - pdf.b_margin - 6:
            pdf.add_page()
            pdf.set_font("helvetica", "B", 7.5)
            pdf.set_fill_color(232, 238, 250)
            for cell in header:
                pdf.cell(col_w, 4.5, _sanitize(cell.strip()), border=1, align="C", fill=True)
            pdf.ln()
            pdf.set_font("helvetica", "", 7.5)
        for cell in row:
            pdf.cell(col_w, row_height, _sanitize(cell.strip()), border=1, align="C")
        pdf.ln()
    pdf.ln(1)


def render(src: Path = SRC, dst: Path = DST) -> None:
    text = src.read_text()
    pdf = ReportPDF(format="A4", unit="mm")
    pdf.set_margins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    lines = text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        # Skip HTML comments / horizontal rules
        if line.strip() in ("---", "***", "___"):
            pdf.ln(2)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(
                PAGE_MARGIN, pdf.get_y(), pdf.w - PAGE_MARGIN, pdf.get_y()
            )
            pdf.ln(2)
            i += 1
            continue

        # Images on their own line
        img_match = re.fullmatch(r"\s*!\[([^\]]*)\]\(([^)]+)\)\s*", line)
        if img_match:
            alt, rel_path = img_match.groups()
            resolved = (src.parent / rel_path).resolve()
            _render_image(pdf, resolved, caption=alt)
            i += 1
            continue

        # Headings
        if line.startswith("### "):
            top, bot = _set_style(pdf, "h3")
            pdf.ln(top)
            pdf.multi_cell(0, 5, _sanitize(line[4:]))
            pdf.ln(bot)
            i += 1
            continue
        if line.startswith("## "):
            top, bot = _set_style(pdf, "h2")
            pdf.ln(top)
            pdf.set_text_color(20, 60, 130)
            pdf.multi_cell(0, 6, _sanitize(line[3:]))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(bot)
            i += 1
            continue
        if line.startswith("# "):
            top, bot = _set_style(pdf, "h1")
            pdf.ln(top)
            pdf.set_text_color(20, 40, 90)
            pdf.multi_cell(0, 8, _sanitize(line[2:]))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(bot)
            i += 1
            continue

        # Fenced code
        if line.startswith("```"):
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            _set_style(pdf, "code")
            pdf.set_fill_color(245, 245, 245)
            for cl in code_lines:
                pdf.cell(0, 4, _sanitize(cl) or " ",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                         fill=True)
            pdf.ln(2)
            continue

        # Tables (pipe syntax)
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            rows: list[list[str]] = []
            header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(header_cells)
            i += 2  # skip header + separator
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                row_cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(row_cells)
                i += 1
            _render_table(pdf, rows)
            continue

        # Bullet lists
        if re.match(r"^\s*[-*]\s+", line):
            _set_style(pdf, "bullet")
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                content = re.sub(r"^\s*[-*]\s+", "", lines[i])
                indent = (len(lines[i]) - len(lines[i].lstrip())) // 2
                pdf.set_x(PAGE_MARGIN + 3 + indent * 3)
                pdf.set_font("helvetica", "", 8.5)
                pdf.write(BODY_LINE_H, _sanitize("* "))
                _render_inline_nowrap(pdf, content)
                i += 1
            pdf.ln(0.5)
            continue

        # Numbered lists
        if re.match(r"^\s*\d+\.\s+", line):
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                m = re.match(r"^\s*(\d+)\.\s+(.*)", lines[i])
                num, content = m.group(1), m.group(2)
                pdf.set_x(PAGE_MARGIN + 3)
                pdf.set_font("helvetica", "", 8.5)
                pdf.write(BODY_LINE_H, f"{num}. ")
                _render_inline_nowrap(pdf, content)
                i += 1
            pdf.ln(0.5)
            continue

        # Blank line
        if not line.strip():
            pdf.ln(1)
            i += 1
            continue

        # Paragraph
        para = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].startswith(("#", "-", "*", "`", "!"))
            and not re.match(r"^\s*\d+\.\s+", lines[i])
            and "|" not in lines[i]
        ):
            para.append(lines[i])
            i += 1
        _set_style(pdf, "body")
        _render_inline(pdf, " ".join(p.strip() for p in para))

    pdf.output(str(dst))
    print(f"[pdf] wrote {dst} ({dst.stat().st_size / 1024:.1f} KB, {pdf.page_no()} pages)")


def _render_inline_nowrap(pdf: FPDF, text: str) -> None:
    """Inline renderer for bullet/number-list items."""
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    for piece in pattern.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            pdf.set_font("helvetica", "B", 8.5)
            pdf.write(BODY_LINE_H, _sanitize(piece[2:-2]))
        elif piece.startswith("*") and piece.endswith("*"):
            pdf.set_font("helvetica", "I", 8.5)
            pdf.write(BODY_LINE_H, _sanitize(piece[1:-1]))
        elif piece.startswith("`") and piece.endswith("`"):
            pdf.set_font("courier", "", 7.5)
            pdf.write(BODY_LINE_H, _sanitize(piece[1:-1]))
        else:
            pdf.set_font("helvetica", "", 8.5)
            pdf.write(BODY_LINE_H, _sanitize(piece))
    pdf.ln(BODY_LINE_H)


if __name__ == "__main__":
    render()
