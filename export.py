from pathlib import Path
import textwrap
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Preformatted, Spacer, PageBreak
)

ROOT = Path(__file__).parent
OUTPUT = ROOT / "Project_Source_Code.pdf"

EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".html", ".css", ".scss", ".json",
    ".java", ".cpp", ".c", ".h", ".hpp",
    ".sql", ".xml", ".yaml", ".yml",
    ".md", ".sh", ".bat"
}

EXCLUDED_DIRS = {
    "node_modules", ".git", ".venv", "venv",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "dist", "build", ".next",
    "coverage", ".idea", ".vscode"
}

EXCLUDED_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "export_code_pdf.py", "Project_Source_Code.pdf"
}

styles = getSampleStyleSheet()
title_style = styles["Title"]

file_style = ParagraphStyle(
    "FileName",
    parent=styles["Heading2"],
    fontSize=12,
    leading=16,
    spaceBefore=10,
    spaceAfter=8,
    keepWithNext=1,
)

code_style = ParagraphStyle(
    "Code",
    fontName="Courier",
    fontSize=9,
    leading=12,
    leftIndent=5,
    rightIndent=5,
)

def collect_files():
    files = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(ROOT)

        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue

        if path.name in EXCLUDED_FILES:
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        files.append(path)

    return sorted(files)

def wrap_code(code, width=84):
    result = []

    for line in code.splitlines():
        if len(line) <= width:
            result.append(line)
        else:
            result.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    subsequent_indent="    ",
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )

    return "\n".join(result)

def build_pdf():
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    story = [
        Paragraph("Project Source Code", title_style),
        Spacer(1, 8),
    ]

    files = collect_files()

    story.append(Paragraph(
        f"Total source files: {len(files)}",
        styles["Normal"]
    ))

    story.append(PageBreak())

    for index, file in enumerate(files):
        if index:
            story.append(PageBreak())

        relative = file.relative_to(ROOT).as_posix()

        try:
            code = file.read_text(
                encoding="utf-8",
                errors="replace"
            )
        except Exception:
            continue

        story.append(Paragraph(escape(relative), file_style))
        story.append(Preformatted(
            wrap_code(code),
            code_style,
            maxLineLength=84
        ))

    doc.build(story)
    print(f"PDF created: {OUTPUT}")

if __name__ == "__main__":
    build_pdf()