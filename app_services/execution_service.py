"""Execution service wrappers for analyst runs and report retrieval."""

from __future__ import annotations

import sys
from io import StringIO
from typing import Any, Callable, Dict, Optional, Tuple
from app_services.run_postprocess_service import extract_execution_report


AnalystRunner = Callable[..., Tuple[Dict[str, Any], Dict[str, Any]]]


def run_analyst_with_capture(
    *,
    stop_at_function_idx: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    return_partial_on_error: bool = True,
    analyst_runner: Optional[AnalystRunner] = None,
) -> Dict[str, Any]:
    """Run analyst_main while capturing stdout/stderr and normalizing return shape."""
    if analyst_runner is None:
        from analyst import analyst_main

        analyst_runner = analyst_main

    output_buffer = StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = output_buffer
        sys.stderr = output_buffer

        outputs, timing_report = analyst_runner(
            stop_at_function_idx=stop_at_function_idx,
            progress_callback=progress_callback,
            return_timing=True,
            return_partial_on_error=return_partial_on_error,
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    safe_timing = timing_report if isinstance(timing_report, dict) else {}
    execution_report = extract_execution_report(safe_timing)

    return {
        "outputs": outputs,
        "timing_report": safe_timing,
        "execution_report": execution_report,
        "captured_output": output_buffer.getvalue(),
    }


def consume_last_execution_report_safe() -> Optional[Dict[str, Any]]:
    """Best-effort retrieval of last execution report without raising."""
    try:
        from execution_reporting import consume_last_execution_report

        return consume_last_execution_report()
    except Exception:
        return None
