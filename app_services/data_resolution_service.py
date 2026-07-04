"""Data resolution helpers for routing, inherited context, and merged sources."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Set


def _extract_nested(value: Any, nested_key: str):
    if not nested_key:
        return value
    current = value
    for part in str(nested_key).split('.'):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _pick_contextual_execution_from_analysis(
    analysis_info: Mapping[str, Any],
    target_context: Mapping[str, Any],
) -> Dict[str, Any]:
    history_entries = analysis_info.get('execution_history', []) if isinstance(analysis_info, Mapping) else []
    if not isinstance(history_entries, list) or not history_entries:
        exec_result = analysis_info.get('execution_results', {}) if isinstance(analysis_info, Mapping) else {}
        return exec_result if isinstance(exec_result, dict) else {}

    target_loop = target_context.get('loop_path', []) if isinstance(target_context, Mapping) else []
    target_parallel = target_context.get('parallel_path', []) if isinstance(target_context, Mapping) else []

    def _score(entry: dict) -> int:
        if not isinstance(entry, dict):
            return -1
        ctx = entry.get('history_context', {}) or {}
        loop_path = ctx.get('loop_path', []) if isinstance(ctx, dict) else []
        parallel_path = ctx.get('parallel_path', []) if isinstance(ctx, dict) else []

        if loop_path == target_loop and parallel_path == target_parallel:
            return 4
        if (
            isinstance(loop_path, list)
            and isinstance(target_loop, list)
            and len(loop_path) <= len(target_loop)
            and target_loop[:len(loop_path)] == loop_path
            and parallel_path == target_parallel
        ):
            return 3
        if (
            isinstance(parallel_path, list)
            and isinstance(target_parallel, list)
            and len(parallel_path) <= len(target_parallel)
            and target_parallel[:len(parallel_path)] == parallel_path
        ):
            return 2
        if not loop_path and not parallel_path:
            return 1
        return 0

    best_entry = None
    best_score = -1
    for entry in history_entries:
        score = _score(entry)
        # Prefer newer entries on tie to avoid stale snapshots when contexts are equally compatible.
        if score >= best_score:
            best_score = score
            best_entry = entry

    if isinstance(best_entry, dict):
        return best_entry

    exec_result = analysis_info.get('execution_results', {}) if isinstance(analysis_info, Mapping) else {}
    return exec_result if isinstance(exec_result, dict) else {}


def resolve_routed_inputs(
    *,
    instance_alias: str,
    methodology_list: Sequence[str],
    routing_lines: Mapping[Any, Any],
    analysis_data: Mapping[str, Mapping[str, Any]],
    target_execution_results: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve routed input values for a function from upstream snapshots."""
    resolved_inputs: Dict[str, Any] = {}

    if instance_alias not in methodology_list:
        return resolved_inputs

    dst_idx = methodology_list.index(instance_alias)
    target_context = {}
    if isinstance(target_execution_results, Mapping):
        target_context = target_execution_results.get('history_context', {}) or {}

    for key, routing_info in routing_lines.items():
        try:
            src_idx = None
            src_param_key = None
            dst_idx_key = None
            dst_param_key = None

            if isinstance(key, tuple) and len(key) >= 4:
                src_idx, src_param_key, dst_idx_key, dst_param_key = key[:4]
            elif isinstance(routing_info, dict):
                src_idx = routing_info.get('src_idx')
                src_param_key = routing_info.get('src_param_key')
                dst_idx_key = routing_info.get('dst_idx')
                dst_param_key = routing_info.get('dst_param_key')

            try:
                src_idx = int(src_idx) if src_idx is not None else None
                dst_idx_key = int(dst_idx_key) if dst_idx_key is not None else None
            except (TypeError, ValueError):
                continue

            if dst_idx_key != dst_idx or src_idx is None or src_param_key is None or dst_param_key is None:
                continue

            if src_idx < 0 or src_idx >= len(methodology_list):
                continue

            src_instance_alias = methodology_list[src_idx]
            src_exec = _pick_contextual_execution_from_analysis(
                analysis_data.get(src_instance_alias, {}),
                target_context,
            )
            if src_exec.get('status') != 'success':
                continue

            src_outputs = src_exec.get('outputs', {})
            src_inputs = src_exec.get('inputs', {})
            src_nested_key = routing_info.get('src_nested_key', '') if isinstance(routing_info, dict) else ''

            if isinstance(src_outputs, dict) and src_param_key in src_outputs:
                extracted_value = _extract_nested(src_outputs[src_param_key], src_nested_key)
                if extracted_value is not None:
                    resolved_inputs[dst_param_key] = extracted_value
            elif isinstance(src_inputs, dict) and src_param_key in src_inputs:
                extracted_value = _extract_nested(src_inputs[src_param_key], src_nested_key)
                if extracted_value is not None:
                    resolved_inputs[dst_param_key] = extracted_value
            else:
                # Contextually matched entries can miss some keys; fall back to latest snapshot for that key.
                latest_exec = analysis_data.get(src_instance_alias, {}).get('execution_results', {})
                if isinstance(latest_exec, dict) and latest_exec.get('status') == 'success':
                    latest_outputs = latest_exec.get('outputs', {})
                    latest_inputs = latest_exec.get('inputs', {})
                    if isinstance(latest_outputs, dict) and src_param_key in latest_outputs:
                        extracted_value = _extract_nested(latest_outputs[src_param_key], src_nested_key)
                        if extracted_value is not None:
                            resolved_inputs[dst_param_key] = extracted_value
                    elif isinstance(latest_inputs, dict) and src_param_key in latest_inputs:
                        extracted_value = _extract_nested(latest_inputs[src_param_key], src_nested_key)
                        if extracted_value is not None:
                            resolved_inputs[dst_param_key] = extracted_value
        except Exception:
            continue

    return resolved_inputs


def resolve_inherited_upstream_outputs(
    *,
    instance_alias: str,
    methodology_list: Sequence[str],
    analysis_data: Mapping[str, Mapping[str, Any]],
    target_execution_results: Optional[Mapping[str, Any]] = None,
    can_auto_route_between_fn: Optional[Callable[[int, int], bool]] = None,
) -> Dict[str, Any]:
    """Resolve inherited upstream outputs using contextual history matching."""
    inherited: Dict[str, Any] = {}

    if instance_alias not in methodology_list:
        return inherited

    dst_idx = methodology_list.index(instance_alias)
    target_context = {}
    if isinstance(target_execution_results, Mapping):
        target_context = target_execution_results.get('history_context', {}) or {}

    for src_idx in range(dst_idx):
        try:
            if callable(can_auto_route_between_fn) and not can_auto_route_between_fn(src_idx, dst_idx):
                continue

            src_instance_alias = methodology_list[src_idx]
            src_exec = _pick_contextual_execution_from_analysis(
                analysis_data.get(src_instance_alias, {}),
                target_context,
            )
            if src_exec.get('status') != 'success':
                continue

            src_outputs = src_exec.get('outputs', {})
            if isinstance(src_outputs, dict):
                inherited.update(src_outputs)
        except Exception:
            continue

    return inherited


def build_execution_data_sources(
    *,
    execution_results: Mapping[str, Any],
    instance_alias: Optional[str] = None,
    routed_inputs: Optional[Mapping[str, Any]] = None,
    inherited_inputs: Optional[Mapping[str, Any]] = None,
    active_passforward_output_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Build merged data sources with prefixed aliases.

    Precedence: direct inputs < routed inputs < outputs.
    """
    combined_sources: Dict[str, Any] = {}
    try:
        if not isinstance(execution_results, Mapping):
            return combined_sources

        inputs = execution_results.get('inputs', {})
        outputs = execution_results.get('outputs', {})
        routed_inputs = routed_inputs if isinstance(routed_inputs, Mapping) else {}
        inherited_inputs = inherited_inputs if isinstance(inherited_inputs, Mapping) else {}
        pf_output_keys = set(active_passforward_output_keys or set())

        def _is_pf_output_key(key: Any) -> bool:
            return isinstance(key, str) and key in pf_output_keys

        def _add_prefixed_aliases() -> None:
            if isinstance(inputs, Mapping):
                for key, value in inputs.items():
                    combined_sources[f"in.{key}"] = value
                if instance_alias and instance_alias in inputs and isinstance(inputs[instance_alias], Mapping):
                    for key, value in inputs[instance_alias].items():
                        combined_sources[f"in.{key}"] = value

            for key, value in routed_inputs.items():
                combined_sources[f"in.{key}"] = value

            for key, value in inherited_inputs.items():
                combined_sources.setdefault(f"in.{key}", value)

            if isinstance(outputs, Mapping):
                for key, value in outputs.items():
                    if _is_pf_output_key(key):
                        combined_sources[f"pf.{key}"] = value
                    else:
                        combined_sources[f"out.{key}"] = value
                if instance_alias and instance_alias in outputs and isinstance(outputs[instance_alias], Mapping):
                    for key, value in outputs[instance_alias].items():
                        if _is_pf_output_key(key):
                            combined_sources[f"pf.{key}"] = value
                        else:
                            combined_sources[f"out.{key}"] = value

            for key, value in list(combined_sources.items()):
                if not isinstance(key, str):
                    continue
                if key.startswith('in.') or key.startswith('out.') or key.startswith('pf.'):
                    continue
                combined_sources.setdefault(f"in.{key}", value)

        if isinstance(inputs, Mapping):
            combined_sources.update(inputs)
            if instance_alias and instance_alias in inputs and isinstance(inputs[instance_alias], Mapping):
                combined_sources.update(inputs[instance_alias])

        combined_sources.update(routed_inputs)

        for key, value in inherited_inputs.items():
            combined_sources.setdefault(key, value)

        if isinstance(outputs, Mapping):
            for key, value in outputs.items():
                if _is_pf_output_key(key):
                    combined_sources[f"pf.{key}"] = value
                else:
                    combined_sources[key] = value
            if instance_alias and instance_alias in outputs and isinstance(outputs[instance_alias], Mapping):
                for key, value in outputs[instance_alias].items():
                    if _is_pf_output_key(key):
                        combined_sources[f"pf.{key}"] = value
                    else:
                        combined_sources[key] = value

        _add_prefixed_aliases()
        return combined_sources
    except Exception:
        return combined_sources
