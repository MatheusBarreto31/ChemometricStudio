"""Shared helpers for dialog data-source aliasing and nested-key discovery."""

from typing import Any, Dict, List, Optional


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

    if isinstance(inputs, dict):
        for key, value in inputs.items():
            combined_sources[f"in.{key}"] = value
        if instance_alias in inputs and isinstance(inputs[instance_alias], dict):
            for key, value in inputs[instance_alias].items():
                combined_sources[f"in.{key}"] = value

    if hasattr(main_gui, '_resolve_routed_inputs'):
        try:
            routed_inputs = main_gui._resolve_routed_inputs(instance_alias)
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
            combined_sources[f"out.{key}"] = value
        if instance_alias in outputs and isinstance(outputs[instance_alias], dict):
            for key, value in outputs[instance_alias].items():
                combined_sources[f"out.{key}"] = value

    for key, value in list(combined_sources.items()):
        if not isinstance(key, str):
            continue
        if key.startswith('in.') or key.startswith('out.'):
            continue
        combined_sources.setdefault(f"in.{key}", value)


def get_available_data_sources(outputs: Optional[Dict[str, Any]]) -> List[str]:
    """Return UI source keys with redundant out.<key> aliases removed."""
    if not outputs:
        return []

    keys = [str(key) for key in outputs.keys() if isinstance(key, str)]
    key_set = set(keys)
    filtered: List[str] = []

    for key in keys:
        if key.startswith('out.'):
            unprefixed = key.split('.', 1)[1]
            if unprefixed in key_set:
                continue
        filtered.append(key)

    def _sort_key(value: str):
        if value.startswith('in.'):
            return (1, value)
        if value.startswith('out.'):
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
    if source_key.startswith('in.') or source_key.startswith('out.'):
        fallback_keys.append(source_key.split('.', 1)[1])
    else:
        fallback_keys.extend([f"out.{source_key}", f"in.{source_key}"])

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
