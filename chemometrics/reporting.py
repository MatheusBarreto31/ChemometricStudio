"""Reporting utilities for LaTeX source and PDF compilation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess


def escape_latex_text(value: Any) -> str:
    """Escape user-provided text for safe LaTeX rendering."""
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def babel_language_for(code: str) -> str:
    """Map app language code to babel language key."""
    normalized = (code or "en").strip().lower()
    mapping = {
        "en": "english",
        "en-us": "english",
        "en-gb": "english",
        "pt": "brazil",
        "pt-br": "brazil",
        "pt_pt": "portuguese",
        "es": "spanish",
        "fr": "french",
        "de": "ngerman",
        "it": "italian",
    }
    return mapping.get(normalized, "english")


def apply_text_styles(text: str, settings: Dict[str, Any]) -> str:
    """Apply bold/italic/underline wrappers to escaped text."""
    styled = text
    if bool(settings.get("underline", False)):
        styled = rf"\underline{{{styled}}}"
    if bool(settings.get("italic", False)):
        styled = rf"\textit{{{styled}}}"
    if bool(settings.get("bold", False)):
        styled = rf"\textbf{{{styled}}}"
    return styled


def build_latex_document(elements: List[Dict[str, Any]], language_code: str, footer_text: str) -> str:
    """Build complete LaTeX source from report elements."""
    babel_lang = babel_language_for(language_code)
    footer_escaped = escape_latex_text(footer_text)
    lines: List[str] = [
        r"\documentclass[11pt,a4paper]{article}",
        rf"\usepackage[{babel_lang}]{{babel}}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{ltablex}",
        r"\usepackage{tabularx}",
        r"\keepXColumns",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{rotating}",
        r"\usepackage{pdflscape}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{lastpage}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        rf"\fancyfoot[L]{{\footnotesize {footer_escaped}}}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"\renewcommand{\footrulewidth}{0.4pt}",
        r"\begin{document}",
    ]

    for element in elements or []:
        element_type = element.get("type", "text")
        settings = element.get("settings", {}) or {}

        if element_type == "report_header":
            title = escape_latex_text(settings.get("text", "Chemometric Studio Report"))
            styled_title = apply_text_styles(title, settings)
            lines.append(rf"\begin{{center}}\Large {styled_title}\end{{center}}")
            lines.append(r"\vspace{0.5em}")
        elif element_type == "title":
            body = apply_text_styles(escape_latex_text(settings.get("text", "Title")), settings)
            lines.append(rf"\section*{{{body}}}")
        elif element_type == "section":
            body = apply_text_styles(escape_latex_text(settings.get("text", "Section")), settings)
            lines.append(rf"\section{{{body}}}")
        elif element_type == "subsection":
            body = apply_text_styles(escape_latex_text(settings.get("text", "Subsection")), settings)
            lines.append(rf"\subsection{{{body}}}")
        elif element_type == "subsubsection":
            body = apply_text_styles(escape_latex_text(settings.get("text", "Subsubsection")), settings)
            lines.append(rf"\subsubsection{{{body}}}")
        elif element_type == "text":
            align = settings.get("align", "left")
            align_env = {"left": "flushleft", "center": "center", "right": "flushright"}.get(align, "flushleft")
            font_size = int(settings.get("font_size", 11))
            body = apply_text_styles(escape_latex_text(settings.get("text", "")), settings)
            lines.append(rf"\begin{{{align_env}}}")
            lines.append(rf"{{\fontsize{{{font_size}}}{{{font_size + 2}}}\selectfont {body}}}")
            lines.append(rf"\end{{{align_env}}}")
            lines.append(r"\vspace{0.3em}")
        elif element_type == "graph":
            image_path = settings.get("image_path", "")
            caption = escape_latex_text(settings.get("title", "Graph"))
            width = float(settings.get("width", 0.9))
            lines.append(r"\begin{figure}[H]")
            lines.append(r"\centering")
            if image_path:
                safe_path = str(image_path).replace("\\", "/")
                lines.append(rf"\includegraphics[width={width}\textwidth]{{{safe_path}}}")
            else:
                lines.append(r"\fbox{\parbox{0.85\textwidth}{Graph preview unavailable.}}")
            lines.append(rf"\caption{{{caption}}}")
            lines.append(r"\end{figure}")
        elif element_type == "table":
            headers = settings.get("headers", []) or []
            rows = settings.get("rows", []) or []
            row_labels = settings.get("row_labels", []) or []
            omit_row_column = bool(settings.get("omit_row_column", False))
            force_landscape = bool(settings.get("force_landscape", False))
            explicit_metadata = bool(settings.get("is_metadata", False))
            row_column_header = settings.get("row_column_header", "Row")
            title = escape_latex_text(settings.get("title", "Table"))
            base_col_count = max(1, len(headers) if headers else (len(rows[0]) if rows else 1))
            col_count = base_col_count if omit_row_column else base_col_count + 1
            max_header_len = max((len(str(h)) for h in headers), default=0)
            max_cell_len = 0
            for row in rows:
                for value in (row or []):
                    max_cell_len = max(max_cell_len, len(str(value)))

            title_lower = title.lower()
            is_metadata_table = explicit_metadata or ("metadata" in title_lower) or ("metadados" in title_lower)
            rotate_table = force_landscape or is_metadata_table or col_count >= 7 or max_header_len >= 30 or max_cell_len >= 52
            is_long_table = len(rows) > 30

            def _header_values_list() -> List[str]:
                if headers:
                    values = [escape_latex_text(h) for h in headers]
                else:
                    values = [f"Col {i + 1}" for i in range(base_col_count)]
                if not omit_row_column:
                    values = [escape_latex_text(row_column_header)] + values
                return values

            header_values = _header_values_list()

            raw_lengths: List[int] = []
            if not omit_row_column:
                row_label_max = max((len(str(v)) for v in row_labels), default=1)
                raw_lengths.append(max(len(str(row_column_header)), row_label_max, 3))
            for col_idx in range(base_col_count):
                header_len = len(str(headers[col_idx])) if col_idx < len(headers) else 4
                cell_len = max((len(str(row[col_idx])) if col_idx < len(row or []) else 0) for row in rows) if rows else 0
                raw_lengths.append(max(header_len, cell_len, 4))

            weight_seeds = [min(3.2, max(0.8, length / 12.0)) for length in raw_lengths]
            if is_metadata_table:
                metadata_target_weights = {
                    'index': 0.85,
                    'label': 1.05,
                    'source': 1.05,
                    'file': 2.15,
                    'file size (bytes)': 1.10,
                    'created': 1.25,
                    'modified': 1.25,
                    'row index': 0.95,
                }
                blended_weights: List[float] = []
                for col_idx in range(base_col_count):
                    header_name = str(headers[col_idx]).strip().lower() if col_idx < len(headers) else ''
                    content_seed = min(2.6, max(0.85, raw_lengths[col_idx] / 14.0))
                    target_seed = metadata_target_weights.get(header_name, 1.05)
                    blended = (0.60 * target_seed) + (0.40 * content_seed)
                    blended_weights.append(min(2.8, max(0.80, blended)))
                weight_seeds = blended_weights
            weight_sum = sum(weight_seeds) or float(col_count)
            normalized_weights = [seed * (col_count / weight_sum) for seed in weight_seeds]
            col_spec_xltabular = "".join(
                [rf">{{\raggedright\arraybackslash\hsize={weight:.3f}\hsize}}X" for weight in normalized_weights]
            )

            if rotate_table:
                lines.append(r"\clearpage")
                lines.append(r"\begin{landscape}")

            width_anchor = r"\linewidth" if rotate_table else r"\textwidth"
            if is_metadata_table:
                width_anchor = r"\textwidth"
            content_density = max(max_header_len, max_cell_len)
            header_row_chars = sum(len(str(v)) for v in header_values)
            row_char_counts: List[int] = []
            for row_idx, row in enumerate(rows):
                row_values = [str(v) for v in (row or [])]
                row_values = (row_values + [""] * base_col_count)[:base_col_count]
                if not omit_row_column:
                    row_label = row_labels[row_idx] if row_idx < len(row_labels) else str(row_idx + 1)
                    row_values = [str(row_label)] + row_values
                row_char_counts.append(sum(len(cell) for cell in row_values))
            max_row_chars = max([header_row_chars] + row_char_counts) if row_char_counts else header_row_chars
            estimated_row_width = min(0.86, max(0.34, 0.26 + (0.0035 * min(max_row_chars, 160))))
            density_width = min(0.88, max(0.36, 0.28 + (0.018 * min(col_count, 10)) + (0.0007 * min(content_density, 90))))
            header_guard_width = min(0.88, max(0.34, 0.20 + (0.010 * min(max_header_len, 36))))
            if is_metadata_table:
                table_width = max(estimated_row_width, density_width)
            else:
                table_width = max(estimated_row_width, min(density_width, estimated_row_width + 0.05), header_guard_width)
            if rotate_table:
                table_width = min(0.94, table_width + 0.05)
            if is_long_table:
                table_width = min(0.94, table_width + 0.02)
            if is_metadata_table:
                metadata_content_max = max((len(str(v)) for row in rows for v in (row or [])), default=0)
                metadata_content_boost = min(0.08, 0.0012 * min(metadata_content_max, 120))
                table_width = min(0.96 if rotate_table else 0.90, table_width + metadata_content_boost)
                table_width = 1.00
                lines.append(r"\scriptsize")
                lines.append(r"\setlength{\tabcolsep}{3pt}")
            else:
                lines.append(r"\footnotesize")
                lines.append(r"\setlength{\tabcolsep}{1.5pt}")
            lines.append(r"\begin{tabularx}{" + f"{table_width:.2f}" + width_anchor + r"}{" + col_spec_xltabular + r"}")
            lines.append(rf"\caption{{{title}}}\label{{tab:auto_{abs(hash(title)) % 100000}}}\\")
            lines.append(r"\toprule")
            header_cells = [rf"\textbf{{{value}}}" for value in header_values]
            lines.append(" & ".join(header_cells) + r" \\")
            lines.append(r"\midrule")

            lines.append(r"\endfirsthead")
            lines.append(r"\multicolumn{" + str(col_count) + r"}{c}{\tablename\ \thetable{}: (continued from previous page)} \\")
            lines.append(r"\toprule")
            lines.append(" & ".join(header_cells) + r" \\")
            lines.append(r"\midrule")
            lines.append(r"\endhead")
            lines.append(r"\bottomrule \multicolumn{" + str(col_count) + r"}{r}{{Continues on next page.}}\\")
            lines.append(r"\endfoot")
            lines.append(r"\bottomrule")
            lines.append(r"\endlastfoot")

            for row_idx, row in enumerate(rows):
                row_values = [escape_latex_text(v) for v in (row or [])]
                row_values = (row_values + [""] * base_col_count)[:base_col_count]
                if not omit_row_column:
                    row_label = row_labels[row_idx] if row_idx < len(row_labels) else str(row_idx + 1)
                    row_values = [escape_latex_text(row_label)] + row_values
                lines.append(" & ".join(row_values) + r" \\")
            lines.append(r"\end{tabularx}")
            if rotate_table:
                lines.append(r"\end{landscape}")
                lines.append(r"\clearpage")
        elif element_type == "page_break":
            lines.append(r"\newpage")

    lines.append(r"\end{document}")
    return "\n".join(lines)


def compile_latex_to_pdf(latex_source: str, output_pdf_path: str) -> Dict[str, Optional[str]]:
    """Compile LaTeX source to PDF using pdflatex.

    Returns a dictionary with keys: success, pdf_path, tex_path, error.
    """
    output_pdf = Path(output_pdf_path)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    work_dir = output_pdf.parent
    stem = output_pdf.stem
    tex_path = work_dir / f"{stem}.tex"
    tex_path.write_text(latex_source, encoding="utf-8")

    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"{stem}.tex",
    ]

    try:
        for _ in range(2):
            subprocess.run(
                command,
                cwd=str(work_dir),
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

        generated_pdf = work_dir / f"{stem}.pdf"
        if not generated_pdf.exists():
            return {
                "success": False,
                "pdf_path": None,
                "tex_path": str(tex_path),
                "error": "PDF was not generated by pdflatex.",
            }

        return {
            "success": True,
            "pdf_path": str(generated_pdf),
            "tex_path": str(tex_path),
            "error": None,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "pdf_path": None,
            "tex_path": str(tex_path),
            "error": "pdflatex is not installed or not available on PATH.",
        }
    except subprocess.CalledProcessError as exc:
        error_output = (exc.stdout or "") + "\n" + (exc.stderr or "")
        tail = "\n".join([line for line in error_output.splitlines() if line.strip()][-20:])
        return {
            "success": False,
            "pdf_path": None,
            "tex_path": str(tex_path),
            "error": tail or "LaTeX compilation failed.",
        }


def create_pdf_report(summary: Dict[str, object], filename: str = "report.tex") -> str:
    """Backward-compatible helper kept for existing integrations."""
    elements = [{"type": "section", "settings": {"text": "Summary"}}]
    for key, value in (summary or {}).items():
        elements.append({"type": "subsection", "settings": {"text": str(key)}})
        elements.append({"type": "text", "settings": {"text": str(value)}})

    latex_source = build_latex_document(elements, "en", "Generated by Chemometric Studio")
    target = Path(filename)
    target.write_text(latex_source, encoding="utf-8")
    return str(target)
