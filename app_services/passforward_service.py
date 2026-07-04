"""Passforward configuration and state helpers.

This module contains pure helpers that can be shared by GUI and API layers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence


def get_passforward_config(
    gui_configs: Mapping[str, Mapping[str, Any]],
    func_alias: str,
    *,
    normalize_bool_setting: Callable[[Any, bool], bool],
) -> Dict[str, Any]:
    """Return normalized passforward config for a function, or empty dict."""
    config = gui_configs.get(func_alias, {})
    raw_cfg = config.get("passforward", {}) if isinstance(config, Mapping) else {}
    if not isinstance(raw_cfg, dict):
        return {}
    if not normalize_bool_setting(raw_cfg.get("compatible", False), default=False):
        return {}

    raw_mappings = raw_cfg.get("mappings", {})
    if not isinstance(raw_mappings, dict):
        return {}

    normalized_mappings: Dict[str, Dict[str, Any]] = {}
    for dst_key, mapping in raw_mappings.items():
        if not isinstance(dst_key, str) or not dst_key.strip():
            continue

        source_key = ""
        nested_key = ""
        label = ""
        has_constant = False
        constant_value: Any = None
        builder = ""

        if isinstance(mapping, str):
            source_key = mapping.strip()
        elif isinstance(mapping, dict):
            source_key = str(mapping.get("source", "") or mapping.get("src_key", "")).strip()
            nested_key = str(mapping.get("nested_key", "") or "").strip()
            label = str(mapping.get("label", "") or mapping.get("display_name", "")).strip()
            builder = str(mapping.get("builder", "") or "").strip()
            if "value" in mapping:
                has_constant = True
                constant_value = mapping.get("value")

        if not source_key and not has_constant and not builder:
            continue

        normalized_mapping: Dict[str, Any] = {
            "source": source_key,
            "nested_key": nested_key,
            "label": label,
        }
        if has_constant:
            normalized_mapping["value"] = constant_value
        if builder:
            normalized_mapping["builder"] = builder
        if isinstance(mapping, dict):
            for extra_key, extra_value in mapping.items():
                if extra_key in normalized_mapping:
                    continue
                normalized_mapping[extra_key] = extra_value

        normalized_mappings[dst_key.strip()] = normalized_mapping

    if not normalized_mappings:
        return {}

    return {
        "compatible": True,
        "description": str(raw_cfg.get("description", "") or "").strip(),
        "mappings": normalized_mappings,
    }


def is_passforward_compatible(
    gui_configs: Mapping[str, Mapping[str, Any]],
    func_alias: str,
    *,
    normalize_bool_setting: Callable[[Any, bool], bool],
) -> bool:
    return bool(
        get_passforward_config(
            gui_configs,
            func_alias,
            normalize_bool_setting=normalize_bool_setting,
        )
    )


def is_passforward_enabled(
    *,
    instance_alias: str,
    base_alias: Optional[str],
    methodology_list: Sequence[str],
    function_base_aliases: Sequence[str],
    function_configs: Mapping[str, Mapping[str, Any]],
    gui_configs: Mapping[str, Mapping[str, Any]],
    normalize_bool_setting: Callable[[Any, bool], bool],
) -> bool:
    if not instance_alias:
        return False

    resolved_base_alias = base_alias
    if resolved_base_alias is None and instance_alias in methodology_list:
        idx = methodology_list.index(instance_alias)
        if idx < len(function_base_aliases):
            resolved_base_alias = function_base_aliases[idx]

    if resolved_base_alias and not is_passforward_compatible(
        gui_configs,
        resolved_base_alias,
        normalize_bool_setting=normalize_bool_setting,
    ):
        return False

    cfg = function_configs.get(instance_alias, {})
    return normalize_bool_setting(cfg.get("__passforward_enabled__", False), default=False)


def get_passforward_output_aliases(
    *,
    instance_alias: str,
    func_alias: str,
    methodology_list: Sequence[str],
    function_base_aliases: Sequence[str],
    function_configs: Mapping[str, Mapping[str, Any]],
    gui_configs: Mapping[str, Mapping[str, Any]],
    normalize_bool_setting: Callable[[Any, bool], bool],
) -> Dict[str, str]:
    passforward_cfg = get_passforward_config(
        gui_configs,
        func_alias,
        normalize_bool_setting=normalize_bool_setting,
    )
    if not passforward_cfg or not is_passforward_enabled(
        instance_alias=instance_alias,
        base_alias=func_alias,
        methodology_list=methodology_list,
        function_base_aliases=function_base_aliases,
        function_configs=function_configs,
        gui_configs=gui_configs,
        normalize_bool_setting=normalize_bool_setting,
    ):
        return {}

    aliases: Dict[str, str] = {}
    mappings = passforward_cfg.get("mappings", {})
    for dst_key, mapping in mappings.items():
        label = str(mapping.get("label", "") or "").strip()
        if label:
            low = label.lower()
            if low.startswith("passforward "):
                label = f"(PF) {label[12:]}"
            elif "passforward" in low:
                label = label.replace("Passforward", "(PF)").replace("passforward", "(PF)")
        aliases[dst_key] = label or f"(PF) {dst_key}"
    return aliases


def get_active_passforward_output_keys(
    *,
    instance_alias: str,
    base_alias: Optional[str],
    methodology_list: Sequence[str],
    function_base_aliases: Sequence[str],
    function_configs: Mapping[str, Mapping[str, Any]],
    gui_configs: Mapping[str, Mapping[str, Any]],
    normalize_bool_setting: Callable[[Any, bool], bool],
) -> set:
    """Return passforward output destination keys active for this instance."""
    if not instance_alias:
        return set()

    resolved_base = base_alias
    if resolved_base is None and instance_alias in methodology_list:
        idx = methodology_list.index(instance_alias)
        if idx < len(function_base_aliases):
            resolved_base = function_base_aliases[idx]

    if not resolved_base or not is_passforward_enabled(
        instance_alias=instance_alias,
        base_alias=resolved_base,
        methodology_list=methodology_list,
        function_base_aliases=function_base_aliases,
        function_configs=function_configs,
        gui_configs=gui_configs,
        normalize_bool_setting=normalize_bool_setting,
    ):
        return set()

    pf_cfg = get_passforward_config(
        gui_configs,
        resolved_base,
        normalize_bool_setting=normalize_bool_setting,
    )
    mappings = pf_cfg.get("mappings", {}) if isinstance(pf_cfg, dict) else {}
    if not isinstance(mappings, dict):
        return set()

    return {str(key) for key in mappings.keys() if isinstance(key, str) and str(key).strip()}
