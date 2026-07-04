"""Thin controller for run-flow orchestration helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from app_services.run_coordinator_service import build_timing_store_args
from app_services.run_exception_service import try_restore_and_show_execution_report


class RunController:
    """Shared run-controller helpers to keep GUI methods lean."""

    def build_progress_callback(
        self,
        update_progress_fn: Callable[[int, int, str, str], None],
    ) -> Callable[[int, int, str, str], None]:
        def _progress_callback(completed_steps: int, total_steps: int, current_instance: str, current_base: str):
            update_progress_fn(completed_steps, total_steps, current_instance, current_base)

        return _progress_callback

    def show_latest_execution_report_popup(
        self,
        *,
        show_popup_fn: Callable[[Any, str], None],
        latest_execution_report: Any,
        run_type_label: str,
    ) -> None:
        show_popup_fn(latest_execution_report, run_type_label)

    def restore_execution_report_popup(
        self,
        *,
        consume_report_fn: Callable[[], Any],
        store_report_fn: Callable[[Any], None],
        show_popup_for_label_fn: Callable[[str], None],
        run_type_label: str,
    ) -> None:
        try_restore_and_show_execution_report(
            consume_report_fn=consume_report_fn,
            store_report_fn=store_report_fn,
            show_popup_fn=show_popup_for_label_fn,
            run_type_label=run_type_label,
        )

    def store_timing_report_for_run(
        self,
        *,
        store_timing_report_fn: Callable[..., None],
        run_type_label: str,
        timing_report: Any,
        stop_at_function_alias: Optional[str],
    ) -> None:
        store_timing_report_fn(
            **build_timing_store_args(
                run_type_label=run_type_label,
                timing_report=timing_report,
                stop_at_function_alias=stop_at_function_alias,
            )
        )

    def apply_run_feedback(
        self,
        *,
        run_feedback: Dict[str, Any],
        show_success_fn: Callable[[str], None],
        show_error_fn: Callable[[str], None],
        finish_progress_fn: Callable[[bool], None],
    ) -> None:
        success = bool(run_feedback.get("success"))
        message = str(run_feedback.get("message", ""))
        if success:
            show_success_fn(message)
            finish_progress_fn(True)
            return

        show_error_fn(message)
        finish_progress_fn(False)
