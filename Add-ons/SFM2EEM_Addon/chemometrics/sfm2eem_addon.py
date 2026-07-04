from __future__ import annotations

from fractions import Fraction
from math import ceil
from typing import Any, List, Optional, Tuple

import numpy as np


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _ratio_as_integers(s_dl: float, s_exc: float) -> Tuple[int, int]:
    if s_dl <= 0 or s_exc <= 0:
        raise ValueError("S_dl and S_exc must be positive numbers")

    ratio = Fraction(str(float(s_dl) / float(s_exc))).limit_denominator(1000000)
    return int(ratio.numerator), int(ratio.denominator)


def _convert_single_matrix(X_s: np.ndarray, s_dl: float, s_exc: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X_s.ndim != 2:
        raise ValueError("Each sample matrix must be 2D")

    l, c = X_s.shape
    m_min, n_min = _ratio_as_integers(s_dl, s_exc)
    groups = l // m_min

    if l != (groups * m_min):
        cols = (l - m_min * ((groups + 1) - 1)) + m_min * (c - 1) + m_min * n_min * ((groups + 1) - 1)
    else:
        cols = (l - m_min * (groups - 1)) + m_min * (c - 1) + m_min * n_min * (groups - 1)

    X = np.full((l, int(cols)), np.nan, dtype=float)

    i_last = 0
    for k in range(1, groups + 2):
        if k <= groups:
            counter = m_min * k
        else:
            counter = l

        for i in range(i_last + 1, counter + 1):
            for j in range(1, c + 1):
                col = (i - m_min * (k - 1)) + m_min * (j - 1) + m_min * n_min * (k - 1)
                X[i - 1, col - 1] = X_s[i - 1, j - 1]
        i_last = counter

    Xnr = X.copy()
    Xlnr = ~np.isnan(Xnr)

    if m_min != 1 and n_min != 1:
        ce = X.shape[1]
        ng = int(ceil(ce / m_min))
        s_subm = np.arange(0, n_min * m_min, n_min)
        s_total = np.tile(s_subm, ng) + np.repeat(np.arange(0, m_min * ng, m_min), m_min)
        s_total = s_total[:ce]
        order = np.argsort(s_total, kind="stable")
        sequence = s_total[order].astype(float) * (float(s_dl) / float(m_min))
        X = X[:, order]
    else:
        sequence = np.arange(X.shape[1], dtype=float) * (float(s_dl) / float(m_min))

    Xl = ~np.isnan(X)
    return X, Xnr, sequence, Xl, Xlnr


def _prepare_axis_vector(axis_values: Optional[Any], expected_len: int, default_step: float) -> np.ndarray:
    if expected_len <= 0:
        return np.array([], dtype=float)

    if axis_values is not None:
        try:
            arr = np.asarray(axis_values, dtype=float).reshape(-1)
        except Exception:
            arr = np.array([], dtype=float)
    else:
        arr = np.array([], dtype=float)

    if arr.size >= expected_len:
        return arr[:expected_len].astype(float)

    if arr.size == 0:
        start = 0.0
        step = float(default_step)
    elif arr.size == 1:
        start = float(arr[0])
        step = float(default_step)
    else:
        start = float(arr[0])
        step = float(np.median(np.diff(arr)))

    return start + step * np.arange(expected_len, dtype=float)


def _collapse_transformed_mapping_columns(mapped_matrix: np.ndarray) -> np.ndarray:
    """Collapse transformed mapping matrix column-wise into a 1D emission axis."""
    if mapped_matrix.ndim != 2:
        raise ValueError("Mapped matrix must be 2D")

    collapsed = np.full(mapped_matrix.shape[1], np.nan, dtype=float)
    for col_idx in range(mapped_matrix.shape[1]):
        col_vals = mapped_matrix[:, col_idx]
        finite_vals = col_vals[np.isfinite(col_vals)]
        if finite_vals.size == 0:
            continue

        ref = float(finite_vals[0])
        if np.allclose(finite_vals, ref, rtol=0.0, atol=1e-9):
            collapsed[col_idx] = ref
        else:
            collapsed[col_idx] = float(np.mean(finite_vals))

    return collapsed


def _compute_emission_axis_from_mapping_matrix(
    s_dl: float,
    s_exc: float,
    excitation_axis: np.ndarray,
    offset_axis: np.ndarray,
) -> np.ndarray:
    """Build excitation+offset mapping matrix, transform it, then collapse to emission axis."""
    mapping_matrix = np.asarray(excitation_axis, dtype=float).reshape(-1, 1) + np.asarray(offset_axis, dtype=float).reshape(1, -1)
    mapped_transformed, _mapped_nr, _seq, _mapped_l, _mapped_lnr = _convert_single_matrix(mapping_matrix, s_dl=s_dl, s_exc=s_exc)
    return _collapse_transformed_mapping_columns(mapped_transformed)


def _apply_to_tensor(tensor: np.ndarray, s_dl: float, s_exc: float, transpose_each: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = []
    transformed_nr = []
    sequence_ref: Optional[np.ndarray] = None

    for sample_idx in range(tensor.shape[0]):
        mat = np.asarray(tensor[sample_idx], dtype=float)
        if transpose_each:
            mat = mat.T

        X, Xnr, sequence, _Xl, _Xlnr = _convert_single_matrix(mat, s_dl=s_dl, s_exc=s_exc)

        if transpose_each:
            # Preserve original orientation after using transposed data for conversion.
            X = X.T
            Xnr = Xnr.T

        if sequence_ref is None:
            sequence_ref = sequence
        elif sequence_ref.shape != sequence.shape or not np.allclose(sequence_ref, sequence, equal_nan=True):
            raise ValueError("Inconsistent output axis sequence across samples")

        transformed.append(X)
        transformed_nr.append(Xnr)

    return np.stack(transformed, axis=0), np.stack(transformed_nr, axis=0), sequence_ref if sequence_ref is not None else np.array([])


def _copy_axis_info(axis_info: Any) -> Any:
    if not isinstance(axis_info, list):
        return axis_info
    copied = []
    for item in axis_info:
        if isinstance(item, np.ndarray):
            copied.append(item.copy())
        elif isinstance(item, list):
            copied.append(list(item))
        else:
            copied.append(item)
    return copied


def _update_axis_info(
    axis_n_info: Optional[List[Any]],
    axis_t_info: Optional[List[Any]],
    emission_axis: np.ndarray,
    transpose_each: bool,
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    axis_n_out = _copy_axis_info(axis_n_info)
    axis_t_out = _copy_axis_info(axis_t_info)

    if not isinstance(axis_n_out, list) or len(axis_n_out) < 3:
        return axis_n_out, axis_t_out

    if transpose_each:
        original_offset_axis_idx = 1
        original_excitation_axis_idx = 2
        output_emission_axis_idx = 1
    else:
        original_excitation_axis_idx = 1
        original_offset_axis_idx = 2
        output_emission_axis_idx = 2

    _ = original_excitation_axis_idx
    _ = original_offset_axis_idx

    new_emission_axis = np.asarray(emission_axis, dtype=float)

    axis_n_out[output_emission_axis_idx] = new_emission_axis

    if isinstance(axis_t_out, list) and len(axis_t_out) >= 3:
        axis_t_out[output_emission_axis_idx] = [f"{float(v):g}" for v in new_emission_axis]

    return axis_n_out, axis_t_out


def _update_dim_labels(
    dim_labels: Optional[List[Any]],
    transpose_each: bool,
    emission_axis_title: str,
) -> Optional[List[Any]]:
    if not isinstance(dim_labels, list):
        return dim_labels

    labels_out = list(dim_labels)
    if len(labels_out) < 3:
        return labels_out

    emission_axis_idx = 1 if transpose_each else 2
    labels_out[emission_axis_idx] = str(emission_axis_title)
    return labels_out


def sfm2eem_transform(
    X_cal: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    S_dl: float = 1.0,
    S_exc: float = 1.0,
    Transpose: bool = False,
    nway_flag: Optional[int] = None,
    axis_n_info: Optional[List[Any]] = None,
    axis_t_info: Optional[List[Any]] = None,
    dim_labels: Optional[List[Any]] = None,
    emission_axis_title: str = "$\\lambda_{Emi}$ (nm)",
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[List[Any]],
    Optional[List[Any]],
    Optional[List[Any]],
]:
    """Apply the MATLAB SFM2EEM conversion to every sample matrix in X_cal/X_val."""
    X_cal_arr = np.asarray(X_cal, dtype=float)

    if nway_flag is not None and int(nway_flag) != 2:
        raise ValueError("SFM2EEM add-on supports second-order data only (nway_flag must be 2)")

    if X_cal_arr.ndim != 3:
        raise ValueError("SFM2EEM add-on requires a second-order tensor shaped as [samples, axis1, axis2]")

    transpose_each = _as_bool(Transpose, default=False)

    X_cal_out, X_cal_nr, sequence = _apply_to_tensor(X_cal_arr, s_dl=float(S_dl), s_exc=float(S_exc), transpose_each=transpose_each)

    X_val_out = None
    X_val_nr = None
    if X_val is not None:
        X_val_arr = np.asarray(X_val, dtype=float)
        if X_val_arr.ndim != 3:
            raise ValueError("X_val must be a second-order tensor shaped as [samples, axis1, axis2]")
        X_val_out, X_val_nr, _sequence_val = _apply_to_tensor(X_val_arr, s_dl=float(S_dl), s_exc=float(S_exc), transpose_each=transpose_each)

    if sequence.size > 0:
        conversion_sample = np.asarray(X_cal_arr[0], dtype=float)
        if transpose_each:
            conversion_sample = conversion_sample.T
            excitation_axis_idx = 2
            offset_axis_idx = 1
        else:
            excitation_axis_idx = 1
            offset_axis_idx = 2

        n_rows, n_cols = conversion_sample.shape
        raw_exc_axis = axis_n_info[excitation_axis_idx] if isinstance(axis_n_info, list) and len(axis_n_info) > excitation_axis_idx else None
        raw_off_axis = axis_n_info[offset_axis_idx] if isinstance(axis_n_info, list) and len(axis_n_info) > offset_axis_idx else None
        excitation_axis = _prepare_axis_vector(raw_exc_axis, n_rows, float(S_exc))
        offset_axis = _prepare_axis_vector(raw_off_axis, n_cols, float(S_dl))

        sequence_out = _compute_emission_axis_from_mapping_matrix(
            s_dl=float(S_dl),
            s_exc=float(S_exc),
            excitation_axis=excitation_axis,
            offset_axis=offset_axis,
        )
    else:
        sequence_out = np.asarray(sequence, dtype=float)

    axis_n_out, axis_t_out = _update_axis_info(axis_n_info, axis_t_info, sequence_out, transpose_each=transpose_each)
    dim_labels_out = _update_dim_labels(dim_labels, transpose_each=transpose_each, emission_axis_title=emission_axis_title)

    return X_cal_out, X_val_out, axis_n_out, axis_t_out, dim_labels_out
