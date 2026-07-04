"""API-first run orchestration helpers with no GUI dependencies."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence

from app_services.analysis_state_service import hydrate_analysis_execution_results
from app_services.execution_service import run_analyst_with_capture
from app_services.run_coordinator_service import (
    build_runtime_error_log_contents,
    build_runtime_log_contents,
    build_timing_store_args,
)
from app_services.run_feedback_service import (
    build_full_run_exception_feedback,
    build_full_run_feedback,
    build_partial_run_feedback,
)
from app_services.run_postprocess_service import summarize_run_outcome


def orchestrate_run_execution(
    *,
    run_mode: str,
    run_type_label: str,
    methodology_list: Sequence[str],
    function_base_aliases: Sequence[str],
    function_configs: Mapping[str, Mapping[str, Any]],
    gui_configs: Mapping[str, Mapping[str, Any]],
    analysis_data: MutableMapping[str, Dict[str, Any]],
    stop_at_function_idx: Optional[int] = None,
    stop_at_function_alias: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    analyst_runner: Optional[Callable[..., Any]] = None,
    log_path: str = "",
    messages: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Run analyst and shape a backend-ready payload for full or partial runs.

    Args:
        run_mode: "full" or "partial".
        run_type_label: Display label for run type.
        methodology_list/function_base_aliases/function_configs/gui_configs/analysis_data:
            Runtime workflow structures reused from the current app model.
        stop_at_function_idx: Optional partial-run index.
        stop_at_function_alias: Optional partial-run alias for timing payload.
        progress_callback: Optional analyst progress callback.
        analyst_runner: Optional callable used by tests to avoid importing analyst.
        log_path: Runtime log path for full-run feedback template expansion.
        messages: UI/api text templates.

    Returns:
        Structured dict with execution output, outcome summary, feedback message,
        timing-store args, and prepared log content.
    """
    text = {
        "execution_failed_prefix": "Model execution failed:",
        "partial_results_text": "Partial results were loaded for completed functions.",
        "executed_up_to_prefix": "Model executed up to",
        "results_loaded_text": "Results loaded for analysis.",
        "model_executed_template": "Model executed successfully. Output saved to {log_path}",
        "check_log_template": "Check the log file at {log_path} for details",
    }
    if isinstance(messages, Mapping):
        text.update({k: str(v) for k, v in messages.items()})

    try:
        run_result = run_analyst_with_capture(
            stop_at_function_idx=stop_at_function_idx,
            progress_callback=progress_callback,
            return_partial_on_error=True,
            analyst_runner=analyst_runner,
        )

        outputs = run_result.get("outputs", {})
        timing_report = run_result.get("timing_report", {})
        execution_report = run_result.get("execution_report", {})
        captured_output = str(run_result.get("captured_output", ""))

        target_count = None
        if run_mode == "partial" and stop_at_function_idx is not None:
            target_count = int(stop_at_function_idx) + 1

        hydrate_analysis_execution_results(
            analysis_data=analysis_data,
            methodology_list=methodology_list,
            function_base_aliases=function_base_aliases,
            function_configs=function_configs,
            gui_configs=gui_configs,
            outputs=outputs,
            timing_report=timing_report,
            target_count=target_count,
        )

        run_outcome = summarize_run_outcome(timing_report)
        had_error = bool(run_outcome.get("had_error", False))
        error_text = str(run_outcome.get("error_text", ""))

        if run_mode == "partial":
            feedback = build_partial_run_feedback(
                had_error=had_error,
                error_text=error_text,
                instance_alias=str(stop_at_function_alias or ""),
                execution_failed_prefix=text["execution_failed_prefix"],
                partial_results_text=text["partial_results_text"],
                executed_up_to_prefix=text["executed_up_to_prefix"],
                results_loaded_text=text["results_loaded_text"],
            )
        else:
            feedback = build_full_run_feedback(
                had_error=had_error,
                error_text=error_text,
                log_path=log_path,
                execution_failed_prefix=text["execution_failed_prefix"],
                partial_results_text=text["partial_results_text"],
                model_executed_template=text["model_executed_template"],
            )

        return {
            "ok": True,
            "run_mode": run_mode,
            "run_type_label": run_type_label,
            "outputs": outputs,
            "timing_report": timing_report,
            "execution_report": execution_report,
            "captured_output": captured_output,
            "log_text": build_runtime_log_contents(captured_output, execution_report, run_outcome),
            "run_outcome": run_outcome,
            "run_feedback": feedback,
            "timing_store_args": build_timing_store_args(
                run_type_label=run_type_label,
                timing_report=timing_report,
                stop_at_function_alias=stop_at_function_alias,
            ),
        }
    except Exception as exc:
        exception_text = str(exc)
        log_text = build_runtime_error_log_contents(exception_text, "")
        exception_feedback = build_full_run_exception_feedback(
            exception_text=exception_text,
            log_path=log_path,
            execution_failed_prefix=text["execution_failed_prefix"],
            check_log_template=text["check_log_template"],
        )

        return {
            "ok": False,
            "run_mode": run_mode,
            "run_type_label": run_type_label,
            "exception": exception_text,
            "log_text": log_text,
            "exception_feedback": exception_feedback,
        }
