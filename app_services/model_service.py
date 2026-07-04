"""Model payload assembly and persistence helpers."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


_PATH_PARAMETER_KEYS = {
    "data_path",
    "var_path",
    "smp_path",
    "y_path",
    "y_val_path",
    "X_val_path",
    "Y_val_path",
}


def _normalize_path_like(value: Any) -> str:
    return str(value).replace("\\", "/")


def _normalize_path_parameter(key: str, value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_path_like(path) for path in value]

    value_text = str(value)
    if ";" in value_text:
        return [_normalize_path_like(item.strip()) for item in value_text.split(";")]

    if key == "data_path":
        if value_text.startswith("["):
            try:
                parsed = json.loads(value_text)
                if isinstance(parsed, list):
                    return [_normalize_path_like(path) for path in parsed]
            except Exception:
                pass
        return [_normalize_path_like(value_text)]

    return _normalize_path_like(value_text)


def _build_all_function_params(
    methodology_list: Sequence[str],
    function_base_aliases: Sequence[str],
    function_configs: Mapping[str, Mapping[str, Any]],
    gui_configs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    all_function_params: Dict[str, Dict[str, Any]] = {}

    for idx, instance_alias in enumerate(methodology_list):
        params = dict(function_configs.get(instance_alias, {}))
        base_alias = function_base_aliases[idx] if idx < len(function_base_aliases) else instance_alias

        func_config = gui_configs.get(base_alias, {})
        layout = func_config.get("setup", {}).get("layout", [])

        for field_info in layout:
            field_name = field_info.get("name", "")
            if field_info.get("input_type") != "inherited":
                continue

            for prev_idx in range(idx - 1, -1, -1):
                prev_alias = methodology_list[prev_idx]
                prev_base_alias = function_base_aliases[prev_idx] if prev_idx < len(function_base_aliases) else prev_alias
                prev_config = gui_configs.get(prev_base_alias, {})
                prev_layout = prev_config.get("setup", {}).get("layout", [])

                for prev_field_info in prev_layout:
                    same_field = prev_field_info.get("name") == field_name
                    not_inherited = prev_field_info.get("input_type") != "inherited"
                    if not same_field or not not_inherited:
                        continue

                    prev_params = all_function_params.get(prev_alias, {})
                    if field_name in prev_params:
                        params[field_name] = prev_params[field_name]
                        break
                if field_name in params:
                    break

        all_function_params[instance_alias] = params

    return all_function_params


def _build_layout_field_index(func_config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    layout = func_config.get("setup", {}).get("layout", [])
    if not isinstance(layout, list):
        return {}

    indexed: Dict[str, Dict[str, Any]] = {}
    for field_info in layout:
        if not isinstance(field_info, dict):
            continue
        field_name = field_info.get("name")
        if isinstance(field_name, str) and field_name:
            indexed[field_name] = field_info
    return indexed


def _build_alias_to_actual(field_info: Mapping[str, Any]) -> Dict[str, Any]:
    values = field_info.get("values", [])
    aliases = field_info.get("value_aliases", values)
    if not isinstance(values, list) or not isinstance(aliases, list):
        return {}
    return {str(alias): actual for alias, actual in zip(aliases, values)}


def _canonicalize_choice_parameter(field_info: Mapping[str, Any], value: Any) -> Any:
    alias_to_actual = _build_alias_to_actual(field_info)
    if not alias_to_actual:
        return value

    widget_type = str(field_info.get("widget", "") or "").strip().lower()

    if widget_type == "checklist":
        if isinstance(value, str):
            parts = [segment.strip() for segment in value.split(",") if segment.strip()]
            normalized = [alias_to_actual.get(part, part) for part in parts]
            return ",".join(str(item) for item in normalized)
        if isinstance(value, list):
            return [alias_to_actual.get(str(item), item) for item in value]
        return alias_to_actual.get(str(value), value)

    if isinstance(value, list):
        return [alias_to_actual.get(str(item), item) for item in value]
    return alias_to_actual.get(str(value), value)


def build_model_payload(
    *,
    methodology_list: Sequence[str],
    function_base_aliases: Sequence[str],
    function_configs: Mapping[str, Mapping[str, Any]],
    gui_configs: Mapping[str, Mapping[str, Any]],
    routing_lines: Mapping[Any, Any],
    parameter_types: Mapping[str, Mapping[str, str]],
    addon_registry: Mapping[str, Any],
    app_version: str,
    is_passforward_enabled: Callable[[str, Optional[str]], bool],
    analysis_data: Optional[Mapping[str, Any]] = None,
    serialize_analysis_data: Optional[Callable[[], Dict[str, Any]]] = None,
    report_data: Optional[Mapping[str, Any]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    all_function_params = _build_all_function_params(
        methodology_list=methodology_list,
        function_base_aliases=function_base_aliases,
        function_configs=function_configs,
        gui_configs=gui_configs,
    )

    functions_array: List[Dict[str, Any]] = []
    for idx, instance_alias in enumerate(methodology_list):
        base_alias = function_base_aliases[idx]
        params = all_function_params.get(instance_alias, {})
        func_config = gui_configs.get(base_alias, {})
        field_index = _build_layout_field_index(func_config)
        display_name = func_config.get("display_name", base_alias)
        passforward_enabled = is_passforward_enabled(instance_alias, base_alias)

        processed_params: Dict[str, Any] = {}
        params_with_types: Dict[str, str] = {}
        func_types = parameter_types.get(base_alias, {})

        for key, value in params.items():
            if isinstance(key, str) and key.startswith("__"):
                continue
            if value is None or (isinstance(value, str) and not value):
                continue

            field_info = field_index.get(str(key), {}) if isinstance(key, str) else {}
            if field_info:
                value = _canonicalize_choice_parameter(field_info, value)

            if key in _PATH_PARAMETER_KEYS:
                processed_params[key] = _normalize_path_parameter(key, value)
            else:
                processed_params[key] = value

            params_with_types[key] = func_types.get(key, "str")

        functions_array.append(
            {
                "instance_alias": instance_alias,
                "base_alias": base_alias,
                "display_name": display_name,
                "parameters": processed_params,
                "parameter_types": params_with_types,
                "passforward": {"enabled": passforward_enabled},
            }
        )

    routing_array: List[Dict[str, Any]] = []
    for key, conn_info in routing_lines.items():
        if not isinstance(conn_info, dict):
            continue

        src_idx = conn_info.get("src_idx", key[0] if isinstance(key, tuple) else 0)
        dst_idx = conn_info.get("dst_idx", key[2] if isinstance(key, tuple) and len(key) > 2 else 1)
        src_param_key = conn_info.get("src_param_key", key[1] if isinstance(key, tuple) else "")
        dst_param_key = conn_info.get("dst_param_key", key[3] if isinstance(key, tuple) and len(key) > 3 else "")
        src_param_name = conn_info.get("src_param_name", src_param_key)
        dst_param_name = conn_info.get("dst_param_name", dst_param_key)
        src_nested_key = conn_info.get("src_nested_key", "")
        auto_created = conn_info.get("auto_created", False)

        src_instance_alias = methodology_list[src_idx] if src_idx < len(methodology_list) else ""
        dst_instance_alias = methodology_list[dst_idx] if dst_idx < len(methodology_list) else ""

        if not src_instance_alias or not dst_instance_alias:
            continue

        routing_array.append(
            {
                "source": {
                    "instance_alias": src_instance_alias,
                    "param_key": src_param_key,
                    "param_name": src_param_name,
                    "nested_key": src_nested_key,
                },
                "destination": {
                    "instance_alias": dst_instance_alias,
                    "param_key": dst_param_key,
                    "param_name": dst_param_name,
                },
                "auto_created": auto_created,
            }
        )

    function_to_addon = addon_registry.get("function_to_addon", {})
    available_addons = addon_registry.get("available_addons", {})
    required_addon_ids = sorted(
        {
            function_to_addon.get(base_alias)
            for base_alias in function_base_aliases
            if function_to_addon.get(base_alias)
        }
    )

    required_addons = [
        {
            "id": addon_id,
            "name": available_addons.get(addon_id, {}).get("name", addon_id),
        }
        for addon_id in required_addon_ids
    ]

    model_data: Dict[str, Any] = {
        "metadata": {
            "version": app_version,
            "created": created_at or datetime.now().isoformat(),
            "description": "Chemometric Studio Model Configuration",
            "required_addons": required_addons,
        },
        "functions": functions_array,
        "routing": routing_array,
    }

    if analysis_data and serialize_analysis_data is not None:
        model_data["analysis"] = serialize_analysis_data()

    if isinstance(report_data, Mapping):
        report_elements = report_data.get("elements", [])
        if report_elements:
            model_data["report"] = {"elements": copy.deepcopy(report_elements)}

    return model_data


def write_model_payload(model_data: Mapping[str, Any], model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(model_data, handle, indent=2, ensure_ascii=False)
