"""Shared helpers for run exception recovery handling."""

from __future__ import annotations

from typing import Any, Callable


def try_restore_and_show_execution_report(
    *,
    consume_report_fn: Callable[[], Any],
    store_report_fn: Callable[[Any], None],
    show_popup_fn: Callable[[str], None],
    run_type_label: str,
) -> bool:
    """Best-effort restore+show of last execution report.

    Returns True when the recovery flow completed without raising.
    """
    try:
        store_report_fn(consume_report_fn())
        show_popup_fn(run_type_label)
        return True
    except Exception:
        return False


def build_run_to_here_exception_message(prefix: str, exception_text: Any) -> str:
    """Build user-facing run-to-here exception message."""
    return f"{prefix} {str(exception_text)}"
