"""Shared helpers for dialog data-source aliasing and nested-key discovery."""

from typing import Any, Dict, List, Optional

from app_services.data_resolution_service import (
    build_execution_data_sources,
    resolve_inherited_upstream_outputs,
    resolve_routed_inputs,
)
from app_services.passforward_service import get_active_passforward_output_keys
from app_services.routing_context_service import can_auto_route_between


def _normalize_bool_setting(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return bool(value)


def resolve_execution_data_sources(
    execution_results: Dict[str, Any],
    *,
    instance_alias: str,
    runtime_snapshot: Optional[Dict[str, Any]] = None,
    main_gui: Any = None,
) -> Dict[str, Any]:
    """Resolve combined dialog data sources via snapshot services, with GUI fallback."""
    if isinstance(runtime_snapshot, dict):
        methodology_list = runtime_snapshot.get("methodology_list", [])
        function_base_aliases = runtime_snapshot.get("function_base_aliases", [])
        routing_lines = runtime_snapshot.get("routing_lines", {})
        analysis_data = runtime_snapshot.get("analysis_data", {})
        function_configs = runtime_snapshot.get("function_configs", {})
        gui_configs = runtime_snapshot.get("gui_configs", {})
        workflow_control_aliases = set(runtime_snapshot.get("workflow_control_aliases", set()))

        routed_inputs = resolve_routed_inputs(
            instance_alias=instance_alias,
            methodology_list=methodology_list,
            routing_lines=routing_lines,
            analysis_data=analysis_data,
            target_execution_results=execution_results,
        )
        inherited_inputs = resolve_inherited_upstream_outputs(
            instance_alias=instance_alias,
            methodology_list=methodology_list,
            analysis_data=analysis_data,
            target_execution_results=execution_results,
            can_auto_route_between_fn=lambda src_idx, dst_idx: can_auto_route_between(
                function_base_aliases=function_base_aliases,
                workflow_control_aliases=workflow_control_aliases,
                src_idx=src_idx,
                dst_idx=dst_idx,
            ),
        )
        active_pf_keys = get_active_passforward_output_keys(
            instance_alias=instance_alias,
            base_alias=None,
            methodology_list=methodology_list,
            function_base_aliases=function_base_aliases,
            function_configs=function_configs,
            gui_configs=gui_configs,
            normalize_bool_setting=_normalize_bool_setting,
        )
        return build_execution_data_sources(
            execution_results=execution_results,
            instance_alias=instance_alias,
            routed_inputs=routed_inputs,
            inherited_inputs=inherited_inputs,
            active_passforward_output_keys=active_pf_keys,
        )

    if hasattr(main_gui, '_get_execution_data_sources'):
        try:
            resolved = main_gui._get_execution_data_sources(execution_results, instance_alias)
            if isinstance(resolved, dict):
                return resolved
        except Exception:
            pass

    inputs = execution_results.get('inputs', {}) if isinstance(execution_results, dict) else {}
    outputs = execution_results.get('outputs', {}) if isinstance(execution_results, dict) else {}
    combined_sources: Dict[str, Any] = {}
    if isinstance(inputs, dict):
        combined_sources.update(inputs)
    if isinstance(outputs, dict):
        combined_sources.update(outputs)
    append_prefixed_data_sources(
        combined_sources,
        execution_results,
        main_gui=main_gui,
        instance_alias=instance_alias,
    )
    return combined_sources


def append_prefixed_data_sources(
    combined_sources: Dict[str, Any],
    execution_results: Dict[str, Any],
    *,
    main_gui: Any,
    instance_alias: str,
) -> None:
    """Add explicit in./out. aliases while preserving unprefixed precedence behavior."""
    if not isinstance(combined_sources, dict) or not isinstance(execution_results, dict):
        return

    inputs = execution_results.get('inputs', {})
    outputs = execution_results.get('outputs', {})
    pf_output_keys = set()
    if hasattr(main_gui, '_get_active_passforward_output_keys'):
        try:
            pf_output_keys = main_gui._get_active_passforward_output_keys(instance_alias)
        except Exception:
            pf_output_keys = set()

    def _is_pf_output_key(key: Any) -> bool:
        return isinstance(key, str) and key in pf_output_keys

    if isinstance(inputs, dict):
        for key, value in inputs.items():
            combined_sources[f"in.{key}"] = value
        if instance_alias in inputs and isinstance(inputs[instance_alias], dict):
            for key, value in inputs[instance_alias].items():
                combined_sources[f"in.{key}"] = value

    if hasattr(main_gui, '_resolve_routed_inputs'):
        try:
            routed_inputs = main_gui._resolve_routed_inputs(instance_alias, execution_results)
            if isinstance(routed_inputs, dict):
                for key, value in routed_inputs.items():
                    combined_sources[f"in.{key}"] = value
        except Exception:
            pass

    if hasattr(main_gui, '_resolve_inherited_upstream_outputs'):
        try:
            inherited_inputs = main_gui._resolve_inherited_upstream_outputs(instance_alias)
            if isinstance(inherited_inputs, dict):
                for key, value in inherited_inputs.items():
                    combined_sources.setdefault(f"in.{key}", value)
        except Exception:
            pass

    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if _is_pf_output_key(key):
                combined_sources[f"pf.{key}"] = value
            else:
                combined_sources[f"out.{key}"] = value
        if instance_alias in outputs and isinstance(outputs[instance_alias], dict):
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


def get_available_data_sources(outputs: Optional[Dict[str, Any]]) -> List[str]:
    """Return UI source keys with unprefixed outputs and explicit in.* inputs.

    out.* aliases are intentionally hidden from listings.
    """
    if not outputs:
        return []

    keys = [str(key) for key in outputs.keys() if isinstance(key, str)]
    filtered: List[str] = []

    for key in keys:
        if key.startswith('out.'):
            continue
        filtered.append(key)

    def _sort_key(value: str):
        if value.startswith('in.'):
            return (1, value)
        if value.startswith('pf.'):
            return (2, value)
        return (0, value)

    return sorted(set(filtered), key=_sort_key)


def get_data_source_value(outputs: Optional[Dict[str, Any]], data_source: str):
    """Resolve a data source with prefixed/unprefixed compatibility."""
    if not data_source or not outputs:
        return None

    if data_source in outputs:
        return outputs[data_source]

    source_key = str(data_source)
    fallback_keys = []
    if source_key.startswith('in.') or source_key.startswith('out.') or source_key.startswith('pf.'):
        fallback_keys.append(source_key.split('.', 1)[1])
    else:
        fallback_keys.extend([f"out.{source_key}", f"in.{source_key}", f"pf.{source_key}"])

    for key in fallback_keys:
        if key in outputs:
            return outputs[key]

    return None


def collect_nested_key_paths(data: Dict[str, Any], prefix: str = "") -> List[str]:
    """Collect nested dictionary paths in dot notation (e.g., 'a.b.c')."""
    paths: List[str] = []
    for key, value in data.items():
        key_str = str(key)
        full_key = f"{prefix}.{key_str}" if prefix else key_str
        paths.append(full_key)
        if isinstance(value, dict):
            paths.extend(collect_nested_key_paths(value, full_key))
    return sorted(paths)


def get_nested_keys(outputs: Optional[Dict[str, Any]], data_source: str) -> List[str]:
    """Get nested key paths for a resolved dictionary data source."""
    data = get_data_source_value(outputs, data_source)
    if isinstance(data, dict):
        return collect_nested_key_paths(data)
    return []
