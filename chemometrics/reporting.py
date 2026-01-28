"""Reporting utilities: generate simple PDF reports using PyLaTeX.

This module provides a minimal wrapper around PyLaTeX to create a short
summary report. It is intentionally simple to avoid heavy coupling.
"""
from pylatex import Document, Section, Subsection, Command
from pylatex.utils import NoEscape
from typing import Dict


def create_pdf_report(summary: Dict[str, object], filename: str = 'report.tex') -> str:
    """Create a simple LaTeX report and save to `filename` (TeX file).

    Returns the path to the generated .tex file. User can compile to PDF
    using `pdflatex` if they have a LaTeX toolchain installed.
    """
    doc = Document()
    doc.preamble.append(Command('title', 'Chemometrics Report'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    with doc.create(Section('Summary')):
        for k, v in summary.items():
            with doc.create(Subsection(str(k))):
                doc.append(str(v))

    tex_path = filename
    doc.generate_tex(tex_path.replace('.tex', ''))
    return tex_path
