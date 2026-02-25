"""Execution reporting utilities for function-level messages, warnings, and errors."""

from __future__ import annotations

from contextlib import contextmanager
from threading import local
from typing import Any, Callable, Dict, Optional


_ThreadState = local()
_LAST_EXECUTION_REPORT: Optional[Dict[str, Any]] = None


ReportHandler = Callable[[str, Optional[str], str, Optional[Dict[str, Any]]], None]


def _get_handler() -> Optional[ReportHandler]:
    return getattr(_ThreadState, "handler", None)


def _set_handler(handler: Optional[ReportHandler]) -> None:
    if handler is None:
        if hasattr(_ThreadState, "handler"):
            delattr(_ThreadState, "handler")
        return
    _ThreadState.handler = handler


@contextmanager
def execution_report_context(handler: ReportHandler):
    """Temporarily route emitted execution-report events to a handler."""
    previous = _get_handler()
    _set_handler(handler)
    try:
        yield
    finally:
        _set_handler(previous)


def emit_execution_event(level: str, code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
    """Emit an execution-report event if a handler is active.

    Args:
        level: One of 'message', 'warning', or 'error'.
        code: Optional stable code for localization lookup.
        text: Fallback free-form text.
        details: Optional metadata.
    """
    handler = _get_handler()
    if handler is None:
        return
    handler(level, code, text, details)


def emit_execution_message(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
    emit_execution_event("message", code=code, text=text, details=details)


def emit_execution_warning(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
    emit_execution_event("warning", code=code, text=text, details=details)


def emit_execution_error(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
    emit_execution_event("error", code=code, text=text, details=details)


def set_last_execution_report(report: Optional[Dict[str, Any]]) -> None:
    """Store most recent execution report for retrieval after run completion/failure."""
    global _LAST_EXECUTION_REPORT
    _LAST_EXECUTION_REPORT = report


def get_last_execution_report() -> Optional[Dict[str, Any]]:
    """Return most recent execution report (without clearing it)."""
    return _LAST_EXECUTION_REPORT


def consume_last_execution_report() -> Optional[Dict[str, Any]]:
    """Return and clear most recent execution report."""
    global _LAST_EXECUTION_REPORT
    report = _LAST_EXECUTION_REPORT
    _LAST_EXECUTION_REPORT = None
    return report
