"""Helpers for post-processing analyst timing reports."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def extract_execution_report(timing_report: Any) -> Dict[str, Any]:
    """Return execution report payload from a timing report dict."""
    if not isinstance(timing_report, Mapping):
        return {}

    report = timing_report.get("execution_report", {})
    return report if isinstance(report, Mapping) else {}


def summarize_run_outcome(timing_report: Any) -> Dict[str, Any]:
    """Normalize success/failure summary from timing report payload."""
    if not isinstance(timing_report, Mapping):
        return {"had_error": False, "error_text": ""}

    had_error = bool(timing_report.get("had_error"))
    error_text = str(timing_report.get("error") or "").strip()
    return {
        "had_error": had_error,
        "error_text": error_text,
    }
