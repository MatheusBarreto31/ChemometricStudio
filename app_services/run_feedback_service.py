"""Helpers for consistent run feedback message assembly."""

from __future__ import annotations

from typing import Dict, Optional


def build_partial_run_feedback(
    *,
    had_error: bool,
    error_text: str,
    instance_alias: str,
    execution_failed_prefix: str,
    partial_results_text: str,
    executed_up_to_prefix: str,
    results_loaded_text: str,
) -> Dict[str, object]:
    """Build UI-ready success/error feedback for run-to-here execution."""
    if had_error:
        message = execution_failed_prefix + ((f" {error_text}\n") if error_text else "\n") + partial_results_text
        return {"success": False, "message": message}

    message = executed_up_to_prefix + f" {instance_alias}\n\n" + results_loaded_text
    return {"success": True, "message": message}


def build_full_run_feedback(
    *,
    had_error: bool,
    error_text: str,
    log_path: str,
    execution_failed_prefix: str,
    partial_results_text: str,
    model_executed_template: str,
) -> Dict[str, object]:
    """Build UI-ready success/error feedback for full-model execution."""
    if had_error:
        message = execution_failed_prefix + ((f" {error_text}\n") if error_text else "\n") + partial_results_text
        return {"success": False, "message": message}

    return {
        "success": True,
        "message": model_executed_template.format(log_path=log_path),
    }


def build_full_run_exception_feedback(
    *,
    exception_text: str,
    log_path: str,
    execution_failed_prefix: str,
    check_log_template: str,
) -> str:
    """Build UI-ready exception feedback message for full run failures."""
    return execution_failed_prefix + f" {exception_text}\n" + check_log_template.format(log_path=log_path)
