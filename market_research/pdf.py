"""A4 PDF export of a brief, rendered from the same Markdown as the download (rendering.brief_to_markdown).

Adapted from the ReportLab generator of a CrewAI reference project: navy header bar, striped tables,
"Page X of Y" footer, plus the real generation date. ReportLab is imported lazily, so the rest of the app
works without it; `pdf_available()` says whether PDF export can be offered.
"""
from __future__ import annotations

import io
import re
import unicodedata
from datetime import datetime
from xml.sax.saxutils import escape

from .models import WorkflowState
from .rendering import brief_to_markdown

# A Markdown link whose label may contain escaped brackets, e.g. [\[F4 · foodmandu.com\]](https://...)
_LINK = re.compile(r"\[((?:\\.|[^\]\\])+)\]\((https?://[^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*([^*]+)\*(?![*\w])")
_TABLE_RULE = re.compile(r":?-{3,}:?")
_NUMBERED = re.compile(r"(\d{1,3})[.)]\s+")
_TO_WINANSI = str.maketrans({"‐": "-", "‒": "-", "―": "-", "−": "-", "⁃": "-"})

NAVY, BLUE, PALE, DARK, MID, LIGHT = "#17324D", "#2F6690", "#EAF1F7", "#263238", "#607D8B", "#D9E2EA"


def pdf_available() -> bool:
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return False
    return True


def _plain(text: str) -> str:
    """Unescape Markdown brackets, keep to the WinAnsi characters the built-in PDF fonts can draw, escape XML.

    NFKC turns narrow and thin spaces into plain spaces and the non-breaking hyphen (U+2011, common in LLM
    output) into U+2010, which is then mapped to '-'. Anything still outside WinAnsi becomes '?'.
    """
    text = text.replace("\\[", "[").replace("\\]", "]")
    text = unicodedata.normalize("NFKC", text).translate(_TO_WINANSI)
    text = text.encode("cp1252", "replace").decode("cp1252")
    return escape(text)


def _emphasis(markup: str) -> str:
    markup = _BOLD.sub(r"<b>\1</b>", markup)
    return _ITALIC.sub(r"<i>\1</i>", markup)


def inline_markup(text: str) -> str:
    """Markdown inline text (links, **bold**, *italic*) to ReportLab paragraph markup. Content is XML-escaped."""
    out, pos = [], 0
    for m in _LINK.finditer(text):
        out.append(_emphasis(_plain(text[pos:m.start()])))
        href = escape(m.group(2), {'"': "&quot;"})
        out.append(f'<link href="{href}" color="{BLUE}">{_emphasis(_plain(m.group(1)))}</link>')
        pos = m.end()
    out.append(_emphasis(_plain(text[pos:])))
    return "".join(out).strip()


def _styles():
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm

    base = getSampleStyleSheet()
    c = colors.HexColor
    body = ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=14,
                          textColor=c(DARK), spaceAfter=2.5 * mm)
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22,
                                textColor=c(NAVY), alignment=0, spaceAfter=2 * mm),
        "meta": ParagraphStyle("Meta", parent=body, fontSize=8, leading=11, textColor=c(MID), spaceAfter=1 * mm),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=13, leading=17,
                             textColor=c(NAVY), spaceBefore=5 * mm, spaceAfter=2 * mm, keepWithNext=True),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14,
                             textColor=c(BLUE), spaceBefore=4 * mm, spaceAfter=1.5 * mm, keepWithNext=True),
        "body": body,
        "bullet": ParagraphStyle("Bullet", parent=body, leftIndent=5 * mm, firstLineIndent=-3 * mm,
                                 spaceAfter=1.5 * mm),
        "callout": ParagraphStyle("Callout", parent=body, backColor=c(PALE), borderColor=c(BLUE), borderWidth=0.6,
                                  borderPadding=3 * mm, spaceBefore=2 * mm,
                                  spaceAfter=4 * mm),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=7.5, leading=10, spaceAfter=0),
    }


def _table(lines: list[str], styles, width: float):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(_TABLE_RULE.fullmatch(c) for c in cells if c):
            continue
        rows.append(cells)
    if not rows:
        return None
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    # Column widths proportional to the longest text in each column, within limits, so long reasons wrap.
    weights = [min(60, max(6, max(len(r[i]) for r in rows))) for i in range(n)]
    widths = [width * w / sum(weights) for w in weights]
    table = Table([[Paragraph(inline_markup(c), styles["cell"]) for c in r] for r in rows],
                  colWidths=widths, repeatRows=1, hAlign="LEFT")
    c = colors.HexColor
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c(PALE)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c("#F6F9FB")]),
        ("GRID", (0, 0), (-1, -1), 0.35, c(LIGHT)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _story(markdown: str, styles, width: float) -> list:
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Paragraph, Spacer

    story: list = []
    lines = (markdown or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            table = _table(block, styles, width)
            if table is not None:
                story += [Spacer(1, 1.5 * mm), table, Spacer(1, 3 * mm)]
            continue
        if line == "---":
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor(LIGHT),
                                    spaceBefore=3 * mm, spaceAfter=3 * mm))
        elif line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), styles["h2"]))
        elif line.startswith(("## ", "# ")):
            story.append(Paragraph(inline_markup(line.lstrip("#").strip()), styles["h1"]))
        elif line.startswith("> "):
            story.append(Paragraph(inline_markup(line[2:]), styles["callout"]))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph("&#8226;&nbsp;" + inline_markup(line[2:]), styles["bullet"]))
        elif m := _NUMBERED.match(line):
            story.append(Paragraph(f"{m.group(1)}.&nbsp;" + inline_markup(line[m.end():]), styles["bullet"]))
        else:
            story.append(Paragraph(inline_markup(line), styles["body"]))
        i += 1
    return story


def markdown_to_pdf(markdown: str, title: str, subtitle: str = "", generated: datetime | None = None) -> bytes:
    """Compile Markdown (headings, bullets, quotes, tables, links, bold/italic) into A4 PDF bytes."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

    generated = generated or datetime.now()
    stamp = f"Generated {generated:%Y-%m-%d %H:%M}"
    page_w, page_h = A4
    margin = 18 * mm

    class NumberedCanvas(canvas.Canvas):
        """Draws the header bar and a 'Page X of Y' footer once the page count is known."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pages: list[dict] = []

        def showPage(self):
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            for state in self._pages:
                self.__dict__.update(state)
                self._decorate(len(self._pages))
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

        def _decorate(self, total: int):
            self.saveState()
            self.setFillColor(colors.HexColor(NAVY))
            self.rect(0, page_h - 11 * mm, page_w, 11 * mm, fill=1, stroke=0)
            self.setFillColor(colors.white)
            self.setFont("Helvetica-Bold", 8)
            self.drawString(margin, page_h - 7 * mm, "MARKET RESEARCH BRIEF")
            self.setStrokeColor(colors.HexColor(LIGHT))
            self.setLineWidth(0.5)
            self.line(margin, 15 * mm, page_w - margin, 15 * mm)
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor(MID))
            self.drawString(margin, 10 * mm, stamp)
            self.drawRightString(page_w - margin, 10 * mm, f"Page {self._pageNumber} of {total}")
            self.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=20 * mm,
                            bottomMargin=22 * mm, title=title, author="Market Research Assistant",
                            subject=subtitle or title)
    styles = _styles()
    story: list = [Paragraph(inline_markup(title), styles["title"])]
    if subtitle:
        story.append(Paragraph(inline_markup(subtitle), styles["meta"]))
    story += [Paragraph(stamp, styles["meta"]), Spacer(1, 2 * mm),
              HRFlowable(width="100%", thickness=1, color=colors.HexColor(LIGHT), spaceAfter=2 * mm)]
    body = markdown.splitlines()
    if body and body[0].startswith("# "):  # the title is drawn above; do not repeat it
        body = body[1:]
    story += _story("\n".join(body), styles, doc.width)
    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()


def brief_to_pdf(state: WorkflowState) -> bytes:
    """The full brief (evidence, grounding check, Critic verdict, evaluation) as an A4 PDF."""
    mode = state.research.mode if state.research else ""
    subtitle = "Sample data (not live research)" if mode == "sample" else "Web research"
    if state.research and state.research.focus:
        subtitle += f" · Focus: {', '.join(state.research.focus)}"
    return markdown_to_pdf(brief_to_markdown(state), f"Research brief: {state.query}", subtitle)
