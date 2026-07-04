"""Analysis state hydration helpers shared by UI entry points."""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence


_DEFAULT_PAGES = [{"title": "Default", "layout": "fp", "sections": [{"type": None}]}]


def hydrate_analysis_execution_results(
    *,
    analysis_data: Dict[str, Dict[str, Any]],
    methodology_list: Sequence[str],
    function_base_aliases: Sequence[str],
    function_configs: Mapping[str, Mapping[str, Any]],
    gui_configs: Mapping[str, Mapping[str, Any]],
    outputs: Optional[Mapping[str, Any]],
    timing_report: Optional[Mapping[str, Any]],
    target_count: Optional[int] = None,
) -> None:
    """Populate per-function analysis execution state from analyst outputs.

    The function mutates analysis_data in place, preserving existing page/section
    layout edits when an instance key already exists.
    """

    safe_outputs = outputs if isinstance(outputs, Mapping) else {}
    safe_timing = timing_report if isinstance(timing_report, Mapping) else {}

    execution_time_by_instance = {
        entry.get("instance_alias"): entry.get("execution_time", 0.0)
        for entry in (safe_timing.get("function_timings", []) if isinstance(safe_timing.get("function_timings", []), list) else [])
        if isinstance(entry, Mapping)
    }
    execution_history_by_instance = safe_timing.get("execution_history_by_instance", {})
    if not isinstance(execution_history_by_instance, Mapping):
        execution_history_by_instance = {}

    max_count = len(methodology_list) if target_count is None else max(0, min(int(target_count), len(methodology_list)))

    for idx in range(max_count):
        instance_alias = methodology_list[idx]
        base_alias = function_base_aliases[idx] if idx < len(function_base_aliases) else instance_alias

        if instance_alias not in analysis_data:
            analysis_config = gui_configs.get(base_alias, {}).get("analysis") if base_alias in gui_configs else None
            if isinstance(analysis_config, Mapping):
                analysis_data[instance_alias] = {
                    "pages": copy.deepcopy(analysis_config.get("pages", _DEFAULT_PAGES)),
                    "current_page": analysis_config.get("current_page", 0),
                }
            else:
                analysis_data[instance_alias] = {
                    "pages": copy.deepcopy(_DEFAULT_PAGES),
                    "current_page": 0,
                }

        input_parameters = dict(function_configs.get(instance_alias, {}))
        history_entries = copy.deepcopy(execution_history_by_instance.get(instance_alias, []))

        if history_entries:
            analysis_data[instance_alias]["execution_history"] = history_entries
            selected_history_idx = 0
            analysis_data[instance_alias]["current_result_idx"] = selected_history_idx
            analysis_data[instance_alias]["execution_results"] = history_entries[selected_history_idx].copy()
            continue

        execution_results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time_by_instance.get(instance_alias, 0.0),
            "outputs": safe_outputs.get(instance_alias, {}),
            "inputs": input_parameters,
        }
        analysis_data[instance_alias]["execution_results"] = execution_results
        analysis_data[instance_alias]["execution_history"] = [execution_results.copy()]
        analysis_data[instance_alias]["current_result_idx"] = 0
