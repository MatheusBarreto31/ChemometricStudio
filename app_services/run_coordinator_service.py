"""Shared helpers for run orchestration payload shaping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional


def build_timing_store_args(
    *,
    run_type_label: str,
    timing_report: Any,
    stop_at_function_alias: Optional[str],
) -> Dict[str, Any]:
    """Return kwargs payload for timing-report storage."""
    return {
        "run_type_label": run_type_label,
        "timing_report": timing_report,
        "stop_at_function_alias": stop_at_function_alias,
    }


def build_runtime_log_contents(
    captured_output: Any,
    execution_report: Optional[Mapping[str, Any]] = None,
    run_outcome: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build runtime log body from captured output and structured execution report."""
    output_text = str(captured_output or "")
    sections = [output_text.rstrip()] if output_text else []

    report = execution_report if isinstance(execution_report, Mapping) else {}
    entries = report.get("entries", []) if isinstance(report.get("entries", []), list) else []
    counts = report.get("counts", {}) if isinstance(report.get("counts", {}), Mapping) else {}

    notable_entries = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        level = str(entry.get("level", "message") or "message").lower()
        if level not in {"error", "warning"}:
            continue
        notable_entries.append(entry)

    report_lines = []
    if counts or notable_entries:
        report_lines.append("=== Execution Report Summary ===")
        report_lines.append(
            "counts: "
            + ", ".join(
                [
                    f"message={int(counts.get('message', 0) or 0)}",
                    f"warning={int(counts.get('warning', 0) or 0)}",
                    f"error={int(counts.get('error', 0) or 0)}",
                ]
            )
        )

    if notable_entries:
        report_lines.append("notable entries:")
        for entry in notable_entries:
            level = str(entry.get("level", "message") or "message").upper()
            instance_alias = str(entry.get("instance_alias", "") or "")
            base_alias = str(entry.get("base_alias", "") or "")
            source = str(entry.get("source", "") or "")
            text = str(entry.get("text", "") or "")
            location = f"{instance_alias} ({base_alias})".strip()
            if source:
                report_lines.append(f"- [{level}] {location} [{source}] {text}")
            else:
                report_lines.append(f"- [{level}] {location} {text}")

    outcome = run_outcome if isinstance(run_outcome, Mapping) else {}
    had_error = bool(outcome.get("had_error", False))
    error_text = str(outcome.get("error_text", "") or "").strip()
    if had_error and error_text:
        report_lines.append(f"outcome error: {error_text}")

    if report_lines:
        sections.append("\n".join(report_lines))

    return "\n\n".join(section for section in sections if section)


def build_runtime_error_log_contents(exception_text: Any, captured_output: Any) -> str:
    """Build runtime error log body with optional captured output."""
    return f"ERROR: {str(exception_text)}\n\n{build_runtime_log_contents(captured_output)}"
