from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from chemometrics.input_parsing import parse_numeric_spec

Number = Union[int, float]


def _normalize_mode(value: Any, default: str = "Remove") -> str:
    text = str(value or default).strip().lower()
    return "select" if text.startswith("select") else "remove"


def _split_text_tokens(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    tokens: List[str] = []
    for chunk in text.replace("\n", ";").replace("\t", ";").split(";"):
        for piece in chunk.split(","):
            token = piece.strip()
            if token:
                tokens.append(token)
    return tokens


def _parse_selector(raw_value: Any) -> Tuple[str, List[Union[Number, str]]]:
    text = "" if raw_value is None else str(raw_value).strip()
    if not text:
        return "empty", []

    try:
        numeric_values = parse_numeric_spec(text)
        if numeric_values:
            return "numeric", numeric_values
    except Exception:
        pass

    return "text", _split_text_tokens(text)


def _slice_axis0(value: Any, keep_indices: Sequence[int]) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value[np.asarray(keep_indices, dtype=int)]
    if isinstance(value, list):
        return [value[i] for i in keep_indices if 0 <= i < len(value)]
    if isinstance(value, tuple):
        return tuple(value[i] for i in keep_indices if 0 <= i < len(value))
    return value


def _normalize_sample_labels(labels: Optional[Any], sample_count: int, prefix: str) -> List[str]:
    if isinstance(labels, np.ndarray):
        raw_labels = labels.reshape(-1).tolist()
    elif isinstance(labels, (list, tuple)):
        raw_labels = list(labels)
    else:
        raw_labels = []

    if raw_labels:
        normalized = [str(x) for x in raw_labels]
        if len(normalized) >= sample_count:
            return normalized[:sample_count]
        start = len(normalized)
        normalized.extend([f"{prefix}{idx + 1}" for idx in range(start, sample_count)])
        return normalized
    return [f"{prefix}{idx + 1}" for idx in range(sample_count)]


def _build_sample_index_map(metadata: Dict[str, Any]) -> Dict[int, List[Tuple[str, Any]]]:
    index_map: Dict[int, List[Tuple[str, Any]]] = {}
    fallback_idx = 0
    for key, entry in metadata.items():
        mapped_idx = fallback_idx
        if isinstance(entry, dict) and "sample_index" in entry:
            try:
                mapped_idx = int(entry.get("sample_index")) - 1
            except Exception:
                mapped_idx = fallback_idx
        index_map.setdefault(mapped_idx, []).append((key, entry))
        fallback_idx += 1
    return index_map


def _subset_metadata(
    metadata: Optional[Dict[str, Any]],
    keep_indices: Sequence[int],
    new_sample_labels: Optional[Sequence[str]],
) -> Optional[Dict[str, Any]]:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        return None

    index_map = _build_sample_index_map(metadata)
    subset: Dict[str, Any] = {}
    used_keys: set[str] = set()

    for new_pos, old_idx in enumerate(keep_indices):
        chosen_key: Optional[str] = None
        chosen_entry: Any = None

        entries = index_map.get(int(old_idx), [])
        if entries:
            chosen_key, chosen_entry = entries.pop(0)
        else:
            fallback_key = None
            for key, entry in metadata.items():
                if key in used_keys:
                    continue
                fallback_key = key
                chosen_entry = entry
                break
            chosen_key = fallback_key

        if chosen_key is None:
            chosen_key = str(new_sample_labels[new_pos]) if new_sample_labels and new_pos < len(new_sample_labels) else f"sample_{new_pos + 1}"
            chosen_entry = {}

        if chosen_key in used_keys:
            suffix = 2
            new_key = f"{chosen_key}_{suffix}"
            while new_key in used_keys:
                suffix += 1
                new_key = f"{chosen_key}_{suffix}"
            chosen_key = new_key

        used_keys.add(chosen_key)

        normalized_entry = deepcopy(chosen_entry) if isinstance(chosen_entry, dict) else {"value": deepcopy(chosen_entry)}
        normalized_entry["sample_index"] = new_pos + 1
        if new_sample_labels and new_pos < len(new_sample_labels):
            normalized_entry["sample_label"] = str(new_sample_labels[new_pos])

        subset[chosen_key] = normalized_entry

    return subset


def _indices_from_numeric_positions(values: Sequence[Number], size: int) -> List[int]:
    selected: List[int] = []
    for value in values:
        try:
            numeric = float(value)
        except Exception:
            continue
        nearest = int(round(numeric))
        if abs(numeric - nearest) > 1e-9:
            continue
        idx = nearest - 1
        if 0 <= idx < size:
            selected.append(idx)
    return selected


def _indices_from_numeric_axis(values: Sequence[Number], axis_values: np.ndarray) -> List[int]:
    if axis_values.size == 0:
        return []

    selected: List[int] = []
    axis_arr = np.asarray(axis_values, dtype=float).reshape(-1)
    for value in values:
        try:
            target = float(value)
        except Exception:
            continue
        matches = np.where(np.isclose(axis_arr, target, atol=1e-9, rtol=0.0))[0]
        selected.extend([int(idx) for idx in matches.tolist()])
    return selected


def _indices_from_text(values: Sequence[str], labels: Sequence[Any]) -> List[int]:
    target_set = {str(v).strip() for v in values if str(v).strip()}
    if not target_set:
        return []
    selected: List[int] = []
    for idx, label in enumerate(labels):
        if str(label).strip() in target_set:
            selected.append(idx)
    return selected


def _apply_mode_indices(total_size: int, selected_indices: Sequence[int], mode: str) -> List[int]:
    if total_size <= 0:
        return []

    selected_set = {int(i) for i in selected_indices if 0 <= int(i) < total_size}
    if not selected_set:
        return list(range(total_size))

    normalized_mode = _normalize_mode(mode)
    if normalized_mode == "select":
        return [idx for idx in range(total_size) if idx in selected_set]
    return [idx for idx in range(total_size) if idx not in selected_set]


def _normalize_axis_list(axis_list: Optional[List[Any]]) -> List[Any]:
    if not isinstance(axis_list, list):
        return []
    return [deepcopy(item) for item in axis_list]


def _resolve_axis_slot(axis_info: List[Any], non_sample_dims: int, dim_idx: int) -> Optional[int]:
    if not axis_info:
        return None

    if len(axis_info) == non_sample_dims + 1:
        slot = dim_idx + 1
    elif len(axis_info) == non_sample_dims:
        slot = dim_idx
    else:
        slot = dim_idx + 1 if dim_idx + 1 < len(axis_info) else dim_idx

    if slot < 0 or slot >= len(axis_info):
        return None
    return slot


def _slice_axis_entry(entry: Any, keep_indices: Sequence[int]) -> Any:
    if entry is None:
        return None
    if isinstance(entry, np.ndarray):
        arr = np.asarray(entry).reshape(-1)
        return arr[np.asarray(keep_indices, dtype=int)]
    if isinstance(entry, list):
        return [entry[i] for i in keep_indices if 0 <= i < len(entry)]
    if isinstance(entry, tuple):
        return tuple(entry[i] for i in keep_indices if 0 <= i < len(entry))
    arr = np.asarray(entry).reshape(-1)
    return arr[np.asarray(keep_indices, dtype=int)]


def _normalize_list_param(raw: Any, target_len: int, default_value: str) -> List[str]:
    if isinstance(raw, list):
        values = [str(v) for v in raw]
    elif raw is None:
        values = []
    else:
        values = [str(raw)]

    if len(values) < target_len:
        values.extend([default_value] * (target_len - len(values)))
    return values[:target_len]


def _select_sample_indices(sample_labels: Sequence[str], selector_raw: Any, mode: str) -> List[int]:
    selector_type, selector_values = _parse_selector(selector_raw)
    total = len(sample_labels)

    if selector_type == "empty":
        return list(range(total))

    if selector_type == "numeric":
        candidate_indices = _indices_from_numeric_positions(selector_values, total)
    else:
        candidate_indices = _indices_from_text([str(v) for v in selector_values], sample_labels)

    return _apply_mode_indices(total, candidate_indices, mode)


def _select_variable_indices(
    axis_numeric_entry: Optional[Any],
    axis_text_entry: Optional[Any],
    axis_size: int,
    selector_raw: Any,
    mode: str,
) -> List[int]:
    selector_type, selector_values = _parse_selector(selector_raw)
    if selector_type == "empty":
        return list(range(axis_size))

    if selector_type == "numeric":
        if axis_numeric_entry is not None:
            axis_vector = np.asarray(axis_numeric_entry, dtype=float).reshape(-1)
            candidate_indices = _indices_from_numeric_axis(selector_values, axis_vector)
        else:
            candidate_indices = _indices_from_numeric_positions(selector_values, axis_size)
    else:
        if axis_text_entry is not None:
            if isinstance(axis_text_entry, np.ndarray):
                labels = axis_text_entry.reshape(-1).tolist()
            elif isinstance(axis_text_entry, (list, tuple)):
                labels = list(axis_text_entry)
            else:
                labels = [axis_text_entry]
            candidate_indices = _indices_from_text([str(v) for v in selector_values], labels)
        else:
            candidate_indices = []

    return _apply_mode_indices(axis_size, candidate_indices, mode)


def data_selection(
    nway_flag: int,
    X_cal: np.ndarray,
    Y_cal: Optional[np.ndarray] = None,
    cal_s_mask: Optional[np.ndarray] = None,
    smp_cal: Optional[List[str]] = None,
    class_data_cal: Optional[List[Any]] = None,
    cal_metadata: Optional[Dict[str, Any]] = None,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    val_s_mask: Optional[np.ndarray] = None,
    smp_val: Optional[List[str]] = None,
    class_data_val: Optional[List[Any]] = None,
    val_metadata: Optional[Dict[str, Any]] = None,
    axis_n_info: Optional[List[Any]] = None,
    axis_t_info: Optional[List[Any]] = None,
    dim_labels: Optional[List[str]] = None,
    sample_mode_cal: str = "Remove",
    sample_values_cal: str = "",
    sample_mode_val: str = "Remove",
    sample_values_val: str = "",
    variable_modes: Optional[List[str]] = None,
    variable_values: Optional[List[str]] = None,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    List[str],
    Optional[List[Any]],
    Optional[Dict[str, Any]],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    List[str],
    Optional[List[Any]],
    Optional[Dict[str, Any]],
    List[Any],
    List[Any],
]:
    """Apply sample and variable selection/removal to calibration and validation sets.

    Notes:
    - Samples are assumed to be on axis 0.
    - Variable selection acts on the non-sample dimensions (1..nway_flag).
    - axis_n_info/axis_t_info may include sample axis at position 0 or only non-sample axes.
    """
    if X_cal is None:
        raise ValueError("X_cal is required")

    X_cal_out = np.asarray(X_cal)
    X_val_out = np.asarray(X_val) if X_val is not None else None
    cal_s_mask_out = np.asarray(cal_s_mask, dtype=bool) if cal_s_mask is not None else None
    val_s_mask_out = np.asarray(val_s_mask, dtype=bool) if val_s_mask is not None else None

    non_sample_dims = max(0, int(nway_flag or 0))
    expected_dims = 1 + non_sample_dims
    if X_cal_out.ndim < expected_dims:
        # Keep compatibility with ambiguous nway inputs by inferring from X when needed.
        non_sample_dims = max(0, int(X_cal_out.ndim - 1))

    smp_cal_out = _normalize_sample_labels(smp_cal, X_cal_out.shape[0], prefix="CAL_")
    smp_val_out = _normalize_sample_labels(smp_val, X_val_out.shape[0], prefix="VAL_") if X_val_out is not None else []

    axis_n_out = _normalize_axis_list(axis_n_info)
    axis_t_out = _normalize_axis_list(axis_t_info)

    # 1) Sample-level selection per set.
    cal_keep = _select_sample_indices(smp_cal_out, sample_values_cal, sample_mode_cal)
    X_cal_out = _slice_axis0(X_cal_out, cal_keep)
    Y_cal_out = _slice_axis0(Y_cal, cal_keep)
    cal_s_mask_out = _slice_axis0(cal_s_mask_out, cal_keep)
    smp_cal_out = _slice_axis0(smp_cal_out, cal_keep)
    class_data_cal_out = _slice_axis0(class_data_cal, cal_keep)
    cal_metadata_out = _subset_metadata(cal_metadata, cal_keep, smp_cal_out)

    # Only axis lists with non_sample_dims + 1 entries are treated as including sample axis.
    if axis_n_out and len(axis_n_out) == non_sample_dims + 1:
        axis_n_out[0] = _slice_axis_entry(axis_n_out[0], cal_keep)
    if axis_t_out and len(axis_t_out) == non_sample_dims + 1:
        axis_t_out[0] = _slice_axis_entry(axis_t_out[0], cal_keep)

    if X_val_out is not None:
        val_keep = _select_sample_indices(smp_val_out, sample_values_val, sample_mode_val)
        X_val_out = _slice_axis0(X_val_out, val_keep)
        Y_val_out = _slice_axis0(Y_val, val_keep)
        val_s_mask_out = _slice_axis0(val_s_mask_out, val_keep)
        smp_val_out = _slice_axis0(smp_val_out, val_keep)
        class_data_val_out = _slice_axis0(class_data_val, val_keep)
        val_metadata_out = _subset_metadata(val_metadata, val_keep, smp_val_out)
    else:
        Y_val_out = Y_val
        class_data_val_out = class_data_val
        val_metadata_out = val_metadata

    # 2) Variable-level selection shared by cal/val and axis descriptors.
    mode_list = _normalize_list_param(variable_modes, non_sample_dims, default_value="Remove")
    value_list = _normalize_list_param(variable_values, non_sample_dims, default_value="")

    for dim_idx in range(non_sample_dims):
        axis_in_tensor = dim_idx + 1
        if axis_in_tensor >= X_cal_out.ndim:
            continue

        axis_size = int(X_cal_out.shape[axis_in_tensor])
        if axis_size <= 0:
            continue

        n_slot = _resolve_axis_slot(axis_n_out, non_sample_dims, dim_idx)
        t_slot = _resolve_axis_slot(axis_t_out, non_sample_dims, dim_idx)

        n_axis_entry = axis_n_out[n_slot] if n_slot is not None else None
        t_axis_entry = axis_t_out[t_slot] if t_slot is not None else None

        keep_indices = _select_variable_indices(
            axis_numeric_entry=n_axis_entry,
            axis_text_entry=t_axis_entry,
            axis_size=axis_size,
            selector_raw=value_list[dim_idx],
            mode=mode_list[dim_idx],
        )

        # Apply same non-sample dimension selection to calibration and validation tensors.
        X_cal_out = np.take(X_cal_out, np.asarray(keep_indices, dtype=int), axis=axis_in_tensor)
        if cal_s_mask_out is not None and axis_in_tensor < cal_s_mask_out.ndim:
            cal_s_mask_out = np.take(cal_s_mask_out, np.asarray(keep_indices, dtype=int), axis=axis_in_tensor)
        if X_val_out is not None and axis_in_tensor < X_val_out.ndim:
            X_val_out = np.take(X_val_out, np.asarray(keep_indices, dtype=int), axis=axis_in_tensor)
        if val_s_mask_out is not None and axis_in_tensor < val_s_mask_out.ndim:
            val_s_mask_out = np.take(val_s_mask_out, np.asarray(keep_indices, dtype=int), axis=axis_in_tensor)

        if n_slot is not None:
            axis_n_out[n_slot] = _slice_axis_entry(axis_n_out[n_slot], keep_indices)
        if t_slot is not None:
            axis_t_out[t_slot] = _slice_axis_entry(axis_t_out[t_slot], keep_indices)

    return (
        X_cal_out,
        Y_cal_out,
        cal_s_mask_out,
        smp_cal_out,
        class_data_cal_out,
        cal_metadata_out,
        X_val_out,
        Y_val_out,
        val_s_mask_out,
        smp_val_out,
        class_data_val_out,
        val_metadata_out,
        axis_n_out,
        axis_t_out,
    )
