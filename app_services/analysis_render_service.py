"""Analysis render helpers shared by GUI dialogs and API paths."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


def get_data_from_source(outputs: Mapping[str, Any], data_source: Any, nested_key: str = None) -> Any:
    """Extract data from a source, supporting nested dictionary/list access and special markers."""
    if isinstance(data_source, (list, tuple)):
        for candidate in data_source:
            candidate_source = candidate
            candidate_nested = nested_key

            if isinstance(candidate, dict):
                candidate_source = candidate.get('data_source', candidate.get('var', candidate.get('source')))
                candidate_nested = candidate.get('nested_key', nested_key)

            if candidate_source is None:
                continue

            value = get_data_from_source(outputs, candidate_source, candidate_nested)
            if value is not None:
                return value
        return None

    if isinstance(data_source, dict):
        source_name = data_source.get('data_source', data_source.get('var', data_source.get('source')))
        if source_name is None:
            return None
        resolved_nested = data_source.get('nested_key', nested_key)
        return get_data_from_source(outputs, source_name, resolved_nested)

    if data_source in ('__index__', 'row_index'):
        return '__index__'

    if data_source not in outputs:
        source_key = str(data_source) if data_source is not None else ''
        fallback_keys = []
        if source_key.startswith('in.') or source_key.startswith('out.') or source_key.startswith('pf.'):
            dot_idx = source_key.find('.')
            if dot_idx >= 0 and dot_idx + 1 < len(source_key):
                fallback_keys.append(source_key[dot_idx + 1:])
        elif source_key:
            fallback_keys.extend([f'out.{source_key}', f'in.{source_key}', f'pf.{source_key}'])

        first_dot_idx = source_key.find('.')
        if first_dot_idx > 0:
            root_key = source_key[:first_dot_idx]
            root_nested = source_key[first_dot_idx + 1:]
            if root_key in outputs:
                root_value = outputs[root_key]
                if root_nested:
                    data = get_data_from_source({'__root__': root_value}, '__root__', root_nested)
                    if data is not None:
                        return data

        data = None
        found = False
        for key in fallback_keys:
            if key in outputs:
                data = outputs[key]
                found = True
                break
        if not found:
            return None
    else:
        data = outputs[data_source]

    if not nested_key:
        return data

    nested_key_normalized = str(nested_key).strip().lower() if nested_key is not None else ''
    if nested_key_normalized in ('shape', 'ndim', 'size', 'len', 'length'):
        try:
            if nested_key_normalized == 'shape':
                if isinstance(data, np.ndarray):
                    return tuple(data.shape)
                if isinstance(data, (list, tuple)):
                    return (len(data),)
                return None
            if nested_key_normalized == 'ndim':
                if isinstance(data, np.ndarray):
                    return int(data.ndim)
                if isinstance(data, (list, tuple)):
                    return 1
                return 0
            if nested_key_normalized == 'size':
                if isinstance(data, np.ndarray):
                    return int(data.size)
                if isinstance(data, (list, tuple)):
                    return int(len(data))
                if data is None:
                    return 0
                return 1
            if isinstance(data, (list, tuple, dict, np.ndarray)):
                return int(len(data))
            return None
        except Exception:
            return None

    if isinstance(data, (dict, list, tuple, np.ndarray)):
        keys_path = nested_key.split('.') if '.' in nested_key else [nested_key]
        try:
            for key in keys_path:
                if isinstance(data, dict):
                    if key in data:
                        data = data[key]
                    else:
                        return None
                elif isinstance(data, (list, tuple, np.ndarray)):
                    try:
                        idx = int(key)
                    except (ValueError, TypeError):
                        return None
                    if 0 <= idx < len(data):
                        data = data[idx]
                    else:
                        return None
                else:
                    return None
            return data
        except (KeyError, IndexError, TypeError, AttributeError):
            return None

    return None


def extract_axis_data(outputs: Mapping[str, Any], axis_config: Mapping[str, Any], indices: Any = None, ref_data: Any = None) -> Optional[np.ndarray]:
    """Extract data for an axis from outputs, including __index__ and slicing semantics."""
    if not axis_config:
        return None

    def _resolve_reference_axis(value: Any, ndim: int) -> int:
        if ndim <= 0:
            return 0

        if isinstance(value, str):
            token = value.strip().lower()
            if token in {'first', 'rows', 'row'}:
                return 0
            if token in {'last', 'columns', 'column', 'cols', 'col'}:
                return ndim - 1
            try:
                value = int(token)
            except (TypeError, ValueError):
                value = None

        if isinstance(value, (int, np.integer)):
            axis = int(value)
            if axis < 0:
                axis += ndim
            if 0 <= axis < ndim:
                return axis

        return 0 if ndim == 1 else ndim - 1

    data_source = axis_config.get('data_source')
    if not data_source:
        return None

    if data_source in ('__index__', 'row_index'):
        length = None
        reference_axis_cfg = axis_config.get('reference_axis')

        if ref_data is not None:
            if not isinstance(ref_data, np.ndarray):
                try:
                    ref_data = np.array(ref_data)
                except (ValueError, TypeError):
                    ref_data = None

            if ref_data is not None and hasattr(ref_data, 'shape') and ref_data.ndim > 0:
                ref_axis = _resolve_reference_axis(reference_axis_cfg, int(ref_data.ndim))
                length = int(ref_data.shape[ref_axis])

        if length is None:
            reference_source = axis_config.get('reference_source')
            if reference_source:
                reference_nested = axis_config.get('reference_nested_key')
                ref_data = get_data_from_source(outputs, reference_source, reference_nested)
                if ref_data is not None:
                    if not isinstance(ref_data, np.ndarray):
                        ref_data = np.array(ref_data)
                    if ref_data.ndim > 0:
                        ref_axis = _resolve_reference_axis(reference_axis_cfg, int(ref_data.ndim))
                        length = int(ref_data.shape[ref_axis])
            elif 'reference_length' in axis_config:
                length = axis_config.get('reference_length')

        if length is None:
            return None
        data = np.arange(1, length + 1)
    else:
        nested_key = axis_config.get('nested_key')
        data = get_data_from_source(outputs, data_source, nested_key)
        if data is None:
            return None

    nested_key = axis_config.get('nested_key')
    is_list_source = False
    if isinstance(data, list) and len(data) > 0:
        first_elem = data[0]
        is_list_source = isinstance(first_elem, (list, np.ndarray))
        if is_list_source and nested_key:
            force_list_source = axis_config.get('treat_as_list_source', False)
            is_list_source = bool(force_list_source)

    if is_list_source:
        list_index = axis_config.get('index', 0)
        if list_index < len(data):
            data = data[list_index] if isinstance(data[list_index], np.ndarray) else np.array(data[list_index])
        else:
            data = data[0] if isinstance(data[0], np.ndarray) else np.array(data[0])

    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except (ValueError, TypeError):
            return None

    if not is_list_source:
        config_index = axis_config.get('index')
        if config_index is not None:
            if isinstance(config_index, int) and data.ndim > 1:
                try:
                    data = data[config_index] if config_index < data.shape[0] else data
                except (IndexError, TypeError):
                    pass
            elif isinstance(config_index, list) and data.ndim > 1:
                try:
                    for idx in config_index:
                        data = data[idx]
                except (IndexError, TypeError):
                    pass

    if not is_list_source and isinstance(indices, dict) and indices:
        index_list = []
        for dim in range(data.ndim):
            if dim in indices:
                idx = indices[dim]
                max_idx = data.shape[dim] - 1
                if idx > max_idx:
                    idx = max_idx
                elif idx < 0:
                    idx = 0
                index_list.append(idx)
            else:
                index_list.append(slice(None))
        try:
            data = data[tuple(index_list)]
        except (IndexError, TypeError):
            pass
    elif not is_list_source and isinstance(indices, int) and data.ndim > 1:
        try:
            if 0 <= indices < data.shape[0]:
                data = data[indices]
        except (IndexError, TypeError):
            pass

    return data


def extract_sliced_data(data: Any, indices: Mapping[int, int]) -> Any:
    """Slice a multi-dimensional array using per-dimension indices."""
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except (ValueError, TypeError):
            return data

    if not isinstance(data, np.ndarray) or data.ndim == 0:
        return data

    result = data.copy()
    if isinstance(indices, Mapping):
        for dim in sorted(indices.keys(), reverse=True):
            idx = indices[dim]
            try:
                if dim < len(result.shape) and 0 <= idx < result.shape[dim]:
                    result = np.take(result, idx, axis=dim)
            except (IndexError, TypeError):
                pass

    return result


def compute_dimension_combinations(data_shape: tuple, specified_dims: set, ndim: int) -> List[tuple]:
    """Compute combinations of unspecified dimensions for 4D+ navigation."""
    from itertools import combinations

    all_dims = set(range(len(data_shape)))
    remaining_dims = sorted(all_dims - specified_dims)

    if not remaining_dims:
        return []
    if ndim <= 0 or ndim > len(remaining_dims):
        return []

    return list(combinations(remaining_dims, ndim))


def normalize_class_data_matrix(value: Any) -> Optional[np.ndarray]:
    """Normalize class labels into 2D object array: rows=samples, cols=class layers."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=object)
        if arr.ndim == 0:
            return np.asarray([[arr.item()]], dtype=object)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr.reshape(arr.shape[0], -1)
    except Exception:
        return None


def normalize_class_labels_for_plot(value: Any) -> Optional[np.ndarray]:
    """Normalize class labels to one value per sample (first column if multi-layer)."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=object)
        if arr.ndim == 0:
            return np.array([str(arr.item())], dtype=object)
        if arr.ndim == 1:
            return arr
        reshaped = arr.reshape(arr.shape[0], -1)
        return reshaped[:, 0]
    except Exception:
        return None


def _resolve_text_selector_index(token: Any, length: int, default: int = 0) -> int:
    if length <= 0:
        return 0

    if token is None:
        idx = default
    elif isinstance(token, (int, np.integer)):
        idx = int(token)
    else:
        raw = str(token).strip().lower()
        if raw in ('', 'none'):
            idx = default
        elif raw == 'first':
            idx = 0
        elif raw == 'last':
            idx = length - 1
        else:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                idx = default

    if idx < 0:
        idx = length + idx
    return max(0, min(length - 1, idx))


def _format_text_binding_atom(value: Any, value_format: str = '') -> str:
    if value is None:
        return ''

    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    if isinstance(value, np.ndarray):
        try:
            return json.dumps(value.tolist(), ensure_ascii=False)
        except Exception:
            return str(value)

    fmt = str(value_format or '').strip()
    if fmt:
        try:
            return format(value, fmt)
        except Exception:
            return str(value)

    return str(value)


def _coerce_text_binding_sequence(value: Any, dict_mode: str = 'values', value_format: str = '') -> List[str]:
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        try:
            flat_values = value.reshape(-1).tolist()
        except Exception:
            flat_values = [value]
    elif isinstance(value, (list, tuple)):
        flat_values = list(value)
    elif isinstance(value, dict):
        mode = str(dict_mode or 'values').strip().lower()
        if mode == 'keys':
            flat_values = list(value.keys())
        elif mode == 'items':
            flat_values = [f"{k}: {v}" for k, v in value.items()]
        else:
            flat_values = list(value.values())
    else:
        flat_values = [value]

    return [_format_text_binding_atom(item, value_format=value_format) for item in flat_values]


def _extract_text_binding_value(outputs: Mapping[str, Any], binding: Mapping[str, Any]) -> Any:
    if not isinstance(binding, Mapping):
        return None

    data_source = binding.get('data_source')
    nested_key = binding.get('nested_key')
    value = get_data_from_source(outputs, data_source, nested_key)
    if value is None:
        return None

    selector = binding.get('selector', {}) if isinstance(binding.get('selector', {}), Mapping) else {}
    mode = str(selector.get('mode', 'value')).strip().lower() or 'value'

    if mode == 'value':
        return value

    if not isinstance(value, np.ndarray):
        if isinstance(value, (list, tuple)):
            array_value = np.array(value, dtype=object)
        elif isinstance(value, dict):
            array_value = np.array(list(value.values()), dtype=object)
        else:
            array_value = None
    else:
        array_value = value

    if array_value is None:
        return value if mode == 'value' else None

    flat_values = list(array_value.reshape(-1).tolist())
    if not flat_values:
        return None

    if mode == 'index':
        token = selector.get('index')
        idx = _resolve_text_selector_index(token, len(flat_values), default=0)
        return flat_values[idx]

    if mode == 'range':
        start_token = selector.get('start')
        end_token = selector.get('end')
        start_idx = _resolve_text_selector_index(start_token, len(flat_values), default=0)
        end_idx = _resolve_text_selector_index(end_token, len(flat_values), default=len(flat_values) - 1)
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        return flat_values[start_idx:end_idx + 1]

    return value


def _resolve_text_table_binding(outputs: Mapping[str, Any], binding: Mapping[str, Any]) -> str:
    table_cfg = binding.get('table', {}) if isinstance(binding.get('table', {}), Mapping) else {}
    columns_cfg = table_cfg.get('columns', [])
    if not isinstance(columns_cfg, list) or not columns_cfg:
        return ''

    column_separator = str(table_cfg.get('column_separator', '\t'))
    row_separator = str(table_cfg.get('row_separator', '\n'))
    missing_value = str(table_cfg.get('missing_value', ''))
    include_header = bool(table_cfg.get('include_header', True))
    row_count_mode = str(table_cfg.get('row_count_mode', 'max')).strip().lower() or 'max'

    headers: List[str] = []
    columns_data: List[List[str]] = []

    for idx, col_cfg in enumerate(columns_cfg):
        if not isinstance(col_cfg, Mapping):
            continue

        data_source = col_cfg.get('data_source')
        nested_key = col_cfg.get('nested_key')
        selector = col_cfg.get('selector', {'mode': 'value'})
        dict_mode = str(col_cfg.get('dict_mode', 'values'))
        value_format = str(col_cfg.get('value_format', ''))

        header = col_cfg.get('header')
        if header is None or str(header).strip() == '':
            header = str(col_cfg.get('name', data_source if data_source is not None else f'C{idx + 1}'))
        headers.append(str(header))

        temp_binding = {
            'data_source': data_source,
            'nested_key': nested_key,
            'selector': selector if isinstance(selector, Mapping) else {'mode': 'value'},
        }
        extracted = _extract_text_binding_value(outputs, temp_binding)
        sequence = _coerce_text_binding_sequence(extracted, dict_mode=dict_mode, value_format=value_format)
        columns_data.append(sequence)

    if not columns_data:
        return ''

    lengths = [len(col) for col in columns_data]
    row_count = min(lengths) if row_count_mode == 'min' else max(lengths)

    lines: List[str] = []
    if include_header and headers:
        lines.append(column_separator.join(headers))

    for row_idx in range(row_count):
        row_values: List[str] = []
        for col in columns_data:
            row_values.append(col[row_idx] if row_idx < len(col) else missing_value)
        lines.append(column_separator.join(row_values))

    return row_separator.join(lines)


def resolve_text_section_content(outputs: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """Resolve text template using configured bindings and extracted data."""
    if not isinstance(config, Mapping):
        return ''

    template = config.get('text_template', '')
    if not isinstance(template, str):
        template = str(template)

    bindings = config.get('bindings', [])
    if not isinstance(bindings, list):
        bindings = []

    values_map: Dict[str, str] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        name = str(binding.get('name', '')).strip()
        if not name:
            continue

        if isinstance(binding.get('table', None), Mapping):
            values_map[name] = _resolve_text_table_binding(outputs, binding)
            continue

        extracted = _extract_text_binding_value(outputs, binding)
        value_format = binding.get('value_format', '')
        separator = binding.get('separator', ', ')

        if isinstance(extracted, np.ndarray):
            extracted = extracted.reshape(-1).tolist()

        if isinstance(extracted, (list, tuple)):
            values_map[name] = str(separator).join(
                _format_text_binding_atom(item, value_format=value_format)
                for item in extracted
            )
        else:
            values_map[name] = _format_text_binding_atom(extracted, value_format=value_format)

    class _SafeMap(dict):
        def __missing__(self, key):
            return '{' + str(key) + '}'

    try:
        return template.format_map(_SafeMap(values_map))
    except Exception:
        return template
