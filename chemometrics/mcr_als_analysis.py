"""MCR-ALS analysis based on pyMCR with optional calibration and CV support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import inspect

import numpy as np

from chemometrics.input_parsing import parse_numeric_spec

try:
    from execution_reporting import emit_execution_warning
except ImportError:
    def emit_execution_warning(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        return

try:
    from chemometrics.cv_pipeline import CVConfig, CVPipeline
    HAS_CV = True
except ImportError:
    CVConfig = Any  # type: ignore
    HAS_CV = False

try:
    from pymcr.constraints import ConstraintNonneg, ConstraintNorm  # type: ignore[import-not-found]
    from pymcr.mcr import McrAR  # type: ignore[import-not-found]
    HAS_PYMCR = True
except Exception:
    HAS_PYMCR = False


def _is_cv_fold_call() -> bool:
    stack = inspect.stack()
    for frame_info in stack:
        name = str(frame_info.filename).lower()
        if "cv_pipeline" in name:
            return True
    return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _as_2d_y(y: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if y is None:
        return None
    arr = np.asarray(y, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _normalize_y_labels(y_labels: Any) -> List[str]:
    if y_labels is None:
        return []
    if isinstance(y_labels, np.ndarray):
        raw = np.asarray(y_labels).reshape(-1).tolist()
    elif isinstance(y_labels, (list, tuple)):
        raw = list(y_labels)
    elif isinstance(y_labels, str):
        text = y_labels.strip()
        if not text:
            return []
        raw = [item.strip() for item in text.replace("\t", ",").split(",")]
    else:
        raw = [y_labels]
    return [str(item).strip() for item in raw if str(item).strip()]


def _coerce_mapping_target_index(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        _, right = text.split(":", 1)
        text = right.strip()
    try:
        idx = int(float(text)) - 1
    except Exception:
        return None
    return idx if idx >= 0 else None


def _parse_component_mapping_input(
    mapping_input: Any,
    n_components: int,
) -> Tuple[Dict[int, int], bool, bool]:
    """Return (mapping, explicit_input_present, all_entries_empty)."""
    if mapping_input is None:
        return {}, False, False

    if isinstance(mapping_input, dict):
        out: Dict[int, int] = {}
        for k, v in mapping_input.items():
            try:
                comp_idx = int(float(k)) - 1
            except Exception:
                continue
            y_idx = _coerce_mapping_target_index(v)
            if comp_idx >= 0 and comp_idx < n_components and y_idx is not None:
                out[comp_idx] = y_idx
        return out, True, len(out) == 0

    if isinstance(mapping_input, list):
        out: Dict[int, int] = {}
        non_empty_seen = False
        for comp_idx, raw in enumerate(mapping_input[:n_components]):
            text = "" if raw is None else str(raw).strip()
            if text:
                non_empty_seen = True
            y_idx = _coerce_mapping_target_index(raw)
            if y_idx is not None:
                out[comp_idx] = y_idx
        return out, True, not non_empty_seen

    text_input = str(mapping_input).strip()
    if not text_input:
        return {}, False, False

    out: Dict[int, int] = {}
    for token in text_input.replace(";", ",").split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        left, right = token.split(":", 1)
        try:
            comp_idx = int(float(left.strip())) - 1
            y_idx = int(float(right.strip())) - 1
        except Exception:
            continue
        if comp_idx >= 0 and comp_idx < n_components and y_idx >= 0:
            out[comp_idx] = y_idx
    return out, True, len(out) == 0


def _fit_linear_1d(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(xv) & np.isfinite(yv)
    n_valid = int(np.count_nonzero(valid))

    yhat_full = np.full_like(yv, np.nan, dtype=float)
    if n_valid < 2:
        return {
            "intercept": float("nan"),
            "slope": float("nan"),
            "y_pred": yhat_full,
            "metrics": {"R2": float("nan"), "RMSEP": float("nan"), "n_samples_used": n_valid},
        }

    xv_fit = xv[valid]
    yv_fit = yv[valid]
    Xd = np.column_stack([np.ones_like(xv_fit), xv_fit])
    beta, *_ = np.linalg.lstsq(Xd, yv_fit, rcond=None)
    yhat_fit = Xd @ beta
    yhat_full[valid] = yhat_fit

    residuals = yv_fit - yhat_fit
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((yv_fit - np.mean(yv_fit)) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    rmsep = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "y_pred": yhat_full,
        "metrics": {"R2": r2, "RMSEP": rmsep, "n_samples_used": n_valid},
    }


def _auto_mapping(scores: np.ndarray, y: np.ndarray) -> Dict[int, int]:
    n_comp = int(scores.shape[1])
    n_y = int(y.shape[1])
    mapping: Dict[int, int] = {}
    for comp_idx in range(n_comp):
        best_y = None
        best_r2 = -np.inf
        for y_idx in range(n_y):
            fit = _fit_linear_1d(scores[:, comp_idx], y[:, y_idx])
            r2 = _safe_float(fit.get("metrics", {}).get("R2"), default=-np.inf)
            if np.isfinite(r2) and r2 > best_r2:
                best_r2 = r2
                best_y = y_idx
        if best_y is not None:
            mapping[int(comp_idx)] = int(best_y)
    return mapping


def _flatten_samples_features(x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim < 2:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr, tuple(arr.shape)
    return arr.reshape(arr.shape[0], -1), tuple(arr.shape)


def _parse_row_counts(value: Any, expected_samples: Optional[int]) -> Optional[List[int]]:
    if value is None:
        return None

    if isinstance(value, str):
        tokens = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        parsed: List[int] = []
        for token in tokens:
            try:
                parsed.append(int(float(token)))
            except Exception:
                return None
        values = parsed
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = []
        for item in list(value):
            try:
                values.append(int(float(item)))
            except Exception:
                return None
    else:
        return None

    if not values or any(v <= 0 for v in values):
        return None
    if expected_samples is not None and expected_samples > 0 and len(values) != expected_samples:
        return None
    return [int(v) for v in values]


def _infer_equal_row_counts(total_rows: int, n_samples: Optional[int]) -> List[int]:
    if n_samples is None or n_samples <= 0:
        return [int(total_rows)]
    if int(total_rows) % int(n_samples) != 0:
        raise ValueError(
            "Cannot infer augmented sample row ranges: total rows are not divisible by sample count. "
            "Provide explicit augmented row counts for this dataset."
        )
    rows_per_sample = int(total_rows) // int(n_samples)
    if rows_per_sample <= 0:
        raise ValueError("Invalid augmented row count per sample inferred from input matrix.")
    return [rows_per_sample] * int(n_samples)


def _row_ranges_from_counts(row_counts: Sequence[int]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    for count in row_counts:
        stop = start + int(count)
        ranges.append((start, stop))
        start = stop
    return ranges


def _prepare_augmented_fit_matrix(
    x: np.ndarray,
    nway_flag: int,
    aug_direction: int,
    n_samples_hint: Optional[int],
    explicit_row_counts: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim < 2:
        arr = arr.reshape(-1, 1)

    if int(nway_flag) < 2:
        raise ValueError("aug_direction requires nway_flag >= 2.")

    if arr.ndim == int(nway_flag) + 1:
        sample_count = int(arr.shape[0])
        if n_samples_hint is not None and int(n_samples_hint) != sample_count:
            raise ValueError("Sample count inferred from Y does not match X sample dimension for augmented fitting.")

        aug_axis = int(aug_direction) - 1
        if aug_axis < 0 or aug_axis >= arr.ndim - 1:
            raise ValueError(
                f"aug_direction={int(aug_direction)} is out of range for nway_flag={int(nway_flag)}. "
                f"Expected an integer in [1, {int(nway_flag)}]."
            )

        row_blocks: List[np.ndarray] = []
        row_counts: List[int] = []
        for smp_idx in range(sample_count):
            sample_tensor = np.asarray(arr[smp_idx], dtype=float)
            sample_tensor = np.moveaxis(sample_tensor, aug_axis, 0)
            rows = int(sample_tensor.shape[0])
            row_counts.append(rows)
            row_blocks.append(sample_tensor.reshape(rows, -1, order="F"))

        D = np.vstack(row_blocks) if row_blocks else np.zeros((0, 0), dtype=float)
        return {
            "D": np.asarray(D, dtype=float),
            "fit_shape": tuple(D.shape),
            "sample_count": sample_count,
            "row_counts": row_counts,
            "row_ranges": _row_ranges_from_counts(row_counts),
            "is_augmented": True,
        }

    if arr.ndim != 2:
        raise ValueError(
            "Augmented MCR-ALS expects either an unfolded 2D matrix or a tensor with explicit sample axis "
            "(shape: samples x dim1 x ... x dimN)."
        )

    explicit = list(explicit_row_counts) if explicit_row_counts is not None else None
    row_counts = explicit if explicit else _infer_equal_row_counts(int(arr.shape[0]), n_samples_hint)
    if int(sum(row_counts)) != int(arr.shape[0]):
        raise ValueError("Explicit augmented row counts do not match the number of matrix rows in X.")

    return {
        "D": np.asarray(arr, dtype=float),
        "fit_shape": tuple(arr.shape),
        "sample_count": int(len(row_counts)),
        "row_counts": [int(v) for v in row_counts],
        "row_ranges": _row_ranges_from_counts(row_counts),
        "is_augmented": True,
    }


def _prepare_fit_matrix(
    x: np.ndarray,
    nway_flag: int,
    aug_direction: Optional[int],
    n_samples_hint: Optional[int],
    explicit_row_counts: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if aug_direction is not None and int(nway_flag) >= 2:
        return _prepare_augmented_fit_matrix(
            x=x,
            nway_flag=int(nway_flag),
            aug_direction=int(aug_direction),
            n_samples_hint=n_samples_hint,
            explicit_row_counts=explicit_row_counts,
        )

    D, fit_shape = _flatten_samples_features(np.asarray(x, dtype=float))
    return {
        "D": np.asarray(D, dtype=float),
        "fit_shape": tuple(fit_shape),
        "sample_count": int(D.shape[0]),
        "row_counts": [1] * int(D.shape[0]),
        "row_ranges": [(i, i + 1) for i in range(int(D.shape[0]))],
        "is_augmented": False,
    }


def _aggregate_scores_auc(C_rows: np.ndarray, row_ranges: Sequence[Tuple[int, int]]) -> np.ndarray:
    C = np.asarray(C_rows, dtype=float)
    if C.ndim != 2:
        C = np.asarray(C, dtype=float).reshape(-1, 1)

    out = np.full((int(len(row_ranges)), int(C.shape[1])), np.nan, dtype=float)
    for idx, (start, stop) in enumerate(row_ranges):
        s = int(start)
        e = int(stop)
        if e <= s or s < 0 or e > C.shape[0]:
            continue
        seg = C[s:e, :]
        if seg.shape[0] == 1:
            out[idx, :] = seg[0, :]
        else:
            out[idx, :] = np.trapezoid(seg, dx=1.0, axis=0)
    return out


def _build_pair_outputs(
    component_y_mapping: Dict[str, int],
    y_labels: Sequence[str],
    y_cal_pred: Optional[np.ndarray],
    y_cv_pred: Optional[np.ndarray],
    y_cal_true: Optional[np.ndarray],
    y_cal_error: Optional[np.ndarray],
    y_cv_error: Optional[np.ndarray],
    y_val_pred: Optional[np.ndarray],
    y_val_true: Optional[np.ndarray],
    y_val_error: Optional[np.ndarray],
) -> Dict[str, Any]:
    def _extract_pair_matrix(value: Optional[np.ndarray], y_cols_1based: List[int]) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols: List[np.ndarray] = []
        for y_col in y_cols_1based:
            idx = int(y_col) - 1
            if 0 <= idx < arr.shape[1]:
                cols.append(arr[:, idx:idx + 1])
            else:
                cols.append(np.full((arr.shape[0], 1), np.nan, dtype=float))
        return np.column_stack(cols) if cols else None

    pairs: List[Tuple[int, int]] = []
    for comp_key, y_col in component_y_mapping.items():
        try:
            comp_1 = int(comp_key)
            y_1 = int(y_col)
        except Exception:
            continue
        if comp_1 >= 1 and y_1 >= 1:
            pairs.append((comp_1, y_1))
    pairs = sorted(set(pairs), key=lambda item: item[0])

    pair_components = [int(comp) for comp, _ in pairs]
    pair_y_columns = [int(y_col) for _, y_col in pairs]

    y_titles: List[str] = []
    labels: List[str] = []
    labels_seq = list(y_labels)
    for comp_1, y_1 in pairs:
        y_idx = y_1 - 1
        y_title = f"Y{y_1}"
        if 0 <= y_idx < len(labels_seq):
            candidate = str(labels_seq[y_idx]).strip()
            if candidate:
                y_title = candidate
        y_titles.append(y_title)
        labels.append(f"C{comp_1} -> {y_title} (Y{y_1})")

    return {
        "mcr_pair_components": np.asarray(pair_components, dtype=int) if pair_components else np.asarray([], dtype=int),
        "mcr_pair_y_columns": np.asarray(pair_y_columns, dtype=int) if pair_y_columns else np.asarray([], dtype=int),
        "mcr_pair_y_titles": y_titles,
        "mcr_pairing_labels": labels,
        "mcr_pairing_labels_by_dimension": [[], labels],
        "y_cal_pred_pairs": _extract_pair_matrix(y_cal_pred, pair_y_columns),
        "y_cv_pred_pairs": _extract_pair_matrix(y_cv_pred, pair_y_columns),
        "y_cal_true_pairs": _extract_pair_matrix(y_cal_true, pair_y_columns),
        "y_cal_error_pairs": _extract_pair_matrix(y_cal_error, pair_y_columns),
        "y_cv_error_pairs": _extract_pair_matrix(y_cv_error, pair_y_columns),
        "y_val_pred_pairs": _extract_pair_matrix(y_val_pred, pair_y_columns),
        "y_val_true_pairs": _extract_pair_matrix(y_val_true, pair_y_columns),
        "y_val_error_pairs": _extract_pair_matrix(y_val_error, pair_y_columns),
    }


def _single_fit(
    X_cal: np.ndarray,
    Y_cal: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    Y_val: Optional[np.ndarray],
    n_components: int,
    max_iter: int,
    tol: float,
    random_state: Optional[int],
    c_regr: str,
    st_regr: str,
    c_nonneg: bool,
    st_nonneg: bool,
    c_norm: bool,
    component_y_mapping: Any,
    nway_flag: int,
    aug_direction: Optional[int],
    aug_row_counts_cal: Optional[Sequence[int]] = None,
    aug_row_counts_val: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if not HAS_PYMCR:
        raise ImportError("pyMCR is required for mcr_als_analysis. Install 'pyMCR'.")

    X_cal_raw = np.asarray(X_cal, dtype=float)
    X_val_raw = None if X_val is None else np.asarray(X_val, dtype=float)
    combine_fit_samples = bool(X_val_raw is not None)

    Yc = _as_2d_y(Y_cal)
    Yv = _as_2d_y(Y_val)

    cal_row_counts = _parse_row_counts(aug_row_counts_cal, expected_samples=None)
    val_row_counts = _parse_row_counts(aug_row_counts_val, expected_samples=None)

    cal_prep = _prepare_fit_matrix(
        x=X_cal_raw,
        nway_flag=int(nway_flag),
        aug_direction=aug_direction,
        n_samples_hint=None if Yc is None else int(Yc.shape[0]),
        explicit_row_counts=cal_row_counts,
    )

    val_prep = None
    if X_val_raw is not None:
        val_prep = _prepare_fit_matrix(
            x=X_val_raw,
            nway_flag=int(nway_flag),
            aug_direction=aug_direction,
            n_samples_hint=None if Yv is None else int(Yv.shape[0]),
            explicit_row_counts=val_row_counts,
        )

    if Yc is not None and int(Yc.shape[0]) != int(cal_prep["sample_count"]):
        raise ValueError("Y_cal sample count does not match X_cal sample structure after unfolding/augmentation.")
    if Yv is not None and val_prep is not None and int(Yv.shape[0]) != int(val_prep["sample_count"]):
        raise ValueError("Y_val sample count does not match X_val sample structure after unfolding/augmentation.")

    if combine_fit_samples:
        if int(val_prep["D"].shape[1]) != int(cal_prep["D"].shape[1]):
            raise ValueError(
                "X_val must match X_cal unfolded feature width when validation samples are included in MCR-ALS fitting."
            )
        D_fit = np.vstack([np.asarray(cal_prep["D"], dtype=float), np.asarray(val_prep["D"], dtype=float)])
    else:
        D_fit = np.asarray(cal_prep["D"], dtype=float)

    fit_shape = tuple(D_fit.shape)
    n_cal_rows = int(cal_prep["D"].shape[0])
    n_cal_samples = int(cal_prep["sample_count"])

    if np.any(~np.isfinite(D_fit)):
        raise ValueError("MCR-ALS does not support NaN/Inf in X_cal. Please impute or clean the data first.")

    rng = np.random.default_rng(random_state)

    c_constraints: List[Any] = []
    st_constraints: List[Any] = []
    if c_nonneg:
        c_constraints.append(ConstraintNonneg())
    if c_norm:
        c_constraints.append(ConstraintNorm())
    if st_nonneg:
        st_constraints.append(ConstraintNonneg())

    mcr = McrAR(
        c_regr=str(c_regr).upper(),
        st_regr=str(st_regr).upper(),
        c_constraints=c_constraints,
        st_constraints=st_constraints,
        max_iter=int(max(1, _safe_int(max_iter, default=250))),
        tol_err_change=float(max(0.0, _safe_float(tol, default=1e-8))),
        tol_increase=None,
        tol_n_increase=None,
        tol_n_above_min=None,
    )

    st_guess = np.abs(rng.normal(loc=0.0, scale=1.0, size=(int(n_components), D_fit.shape[1])))
    mcr.fit(D_fit, ST=st_guess)

    C_all = np.asarray(mcr.C_, dtype=float)
    ST = np.asarray(mcr.ST_, dtype=float)
    reconstructed_all = np.asarray(mcr.D_, dtype=float)

    if combine_fit_samples:
        C_cal_rows = np.asarray(C_all[:n_cal_rows], dtype=float)
        C_val_rows = np.asarray(C_all[n_cal_rows:], dtype=float)
        reconstructed = np.asarray(reconstructed_all[:n_cal_rows], dtype=float)
        residual = np.asarray(D_fit[:n_cal_rows] - reconstructed, dtype=float)
    else:
        C_cal_rows = np.asarray(C_all, dtype=float)
        reconstructed = np.asarray(reconstructed_all, dtype=float)
        residual = np.asarray(D_fit - reconstructed, dtype=float)
        if X_val is not None:
            D_val = np.asarray(val_prep["D"], dtype=float) if val_prep is not None else np.zeros((0, ST.shape[1]), dtype=float)
            coef, *_ = np.linalg.lstsq(ST.T, D_val.T, rcond=None)
            C_val_rows = np.asarray(coef.T, dtype=float)
            if c_nonneg:
                C_val_rows = np.maximum(C_val_rows, 0.0)
        else:
            C_val_rows = None

    if bool(cal_prep.get("is_augmented", False)):
        C_cal = _aggregate_scores_auc(C_cal_rows, cal_prep.get("row_ranges", []))
        if C_val_rows is not None and val_prep is not None:
            C_val = _aggregate_scores_auc(C_val_rows, val_prep.get("row_ranges", []))
        else:
            C_val = None
    else:
        C_cal = np.asarray(C_cal_rows, dtype=float)
        C_val = None if C_val_rows is None else np.asarray(C_val_rows, dtype=float)

    ss_res = float(np.sum((D_fit[:n_cal_rows] - reconstructed) ** 2))
    mean_ref = float(np.mean(D_fit[:n_cal_rows]))
    ss_tot = float(np.sum((D_fit[:n_cal_rows] - mean_ref) ** 2)) + 1e-12
    explained_variance = float(100.0 * (1.0 - ss_res / ss_tot))
    sfit = float(
        100.0 * (
            1.0 - np.linalg.norm(D_fit[:n_cal_rows] - reconstructed) / (np.linalg.norm(D_fit[:n_cal_rows]) + 1e-12)
        )
    )

    mapping: Dict[int, int] = {}
    auto_mapping_used = False
    calibration_models: List[Dict[str, Any]] = []
    y_cal_pred = None if Yc is None else np.full_like(Yc, np.nan, dtype=float)
    y_val_pred = None if Yv is None else np.full_like(Yv, np.nan, dtype=float)

    if Yc is not None and Yc.shape[0] == C_cal.shape[0]:
        parsed_map, explicit_map, all_empty = _parse_component_mapping_input(component_y_mapping, int(n_components))
        if explicit_map and not all_empty:
            for comp_idx, y_idx in parsed_map.items():
                if y_idx < Yc.shape[1]:
                    mapping[int(comp_idx)] = int(y_idx)
        else:
            mapping = _auto_mapping(C_cal, Yc)
            auto_mapping_used = True

        for comp_idx in sorted(mapping.keys()):
            y_idx = int(mapping[comp_idx])
            if y_idx >= Yc.shape[1]:
                continue
            fit = _fit_linear_1d(C_cal[:, comp_idx], Yc[:, y_idx])
            if y_cal_pred is not None:
                y_cal_pred[:, y_idx] = np.asarray(fit.get("y_pred"), dtype=float)

            val_metrics: Dict[str, float] = {}
            if C_val is not None and Yv is not None and Yv.shape[1] > y_idx:
                slope = _safe_float(fit.get("slope"), default=np.nan)
                intercept = _safe_float(fit.get("intercept"), default=np.nan)
                yv_hat = intercept + slope * np.asarray(C_val[:, comp_idx], dtype=float)
                if y_val_pred is not None:
                    y_val_pred[:, y_idx] = yv_hat
                valid = np.isfinite(yv_hat) & np.isfinite(Yv[:, y_idx])
                if np.any(valid):
                    resid = np.asarray(Yv[:, y_idx], dtype=float)[valid] - yv_hat[valid]
                    ss_res_v = float(np.sum(resid ** 2))
                    yy = np.asarray(Yv[:, y_idx], dtype=float)[valid]
                    ss_tot_v = float(np.sum((yy - np.mean(yy)) ** 2)) + 1e-12
                    val_metrics = {
                        "R2": float(1.0 - ss_res_v / ss_tot_v),
                        "RMSEP": float(np.sqrt(np.mean(resid ** 2))),
                        "n_samples_used": int(np.count_nonzero(valid)),
                    }

            calibration_models.append(
                {
                    "component": int(comp_idx + 1),
                    "y_column": int(y_idx + 1),
                    "intercept": _safe_float(fit.get("intercept"), default=np.nan),
                    "slope": _safe_float(fit.get("slope"), default=np.nan),
                    "calibration": dict(fit.get("metrics", {})),
                    "validation": val_metrics,
                }
            )

    y_cal_true = _as_2d_y(Y_cal)
    y_val_true = _as_2d_y(Y_val)
    y_cal_error = None if (y_cal_true is None or y_cal_pred is None) else np.asarray(y_cal_true, dtype=float) - np.asarray(y_cal_pred, dtype=float)
    y_val_error = None if (y_val_true is None or y_val_pred is None) else np.asarray(y_val_true, dtype=float) - np.asarray(y_val_pred, dtype=float)

    metrics = {
        "calibration": {
            "implementation_used": "MCR-ALS (pyMCR)",
            "n_iter": int(getattr(mcr, "n_iter", 0) or 0),
            "n_iter_opt": int(getattr(mcr, "n_iter_opt", 0) or 0),
            "SSR": ss_res,
            "mse": _safe_float((getattr(mcr, "err", []) or [np.nan])[-1], default=np.nan),
            "sfit": sfit,
            "explained_variance": explained_variance,
            "c_regr": str(c_regr).upper(),
            "st_regr": str(st_regr).upper(),
            "fit_combined_samples": bool(combine_fit_samples),
            "n_samples_fit": int(cal_prep["sample_count"] + (0 if val_prep is None else int(val_prep["sample_count"]))),
            "n_samples_calibration": int(n_cal_samples),
            "constraints": {
                "c_nonneg": bool(c_nonneg),
                "st_nonneg": bool(st_nonneg),
                "c_norm": bool(c_norm),
            },
            "original_shape": tuple(X_cal_raw.shape),
            "fit_shape": fit_shape,
            "nway_flag": int(nway_flag),
            "augmented_direction": None if aug_direction is None else int(aug_direction),
            "augmented_mode": bool(cal_prep.get("is_augmented", False)),
            "augmented_row_counts_cal": [int(v) for v in cal_prep.get("row_counts", [])],
            "augmented_row_counts_val": [] if val_prep is None else [int(v) for v in val_prep.get("row_counts", [])],
        }
    }

    report_lines = [
        "MCR-ALS Report",
        f"Implementation: MCR-ALS (pyMCR)",
        f"Components: {int(n_components)}",
        f"Calibration samples: {int(C_cal.shape[0])}",
        f"Calibration features (unfolded): {int(D_fit.shape[1])}",
        f"Fit combined samples: {bool(combine_fit_samples)}",
        f"Iterations: {int(getattr(mcr, 'n_iter', 0) or 0)}",
        f"sfit (%): {sfit:.4f}",
        f"Explained variance (%): {explained_variance:.4f}",
    ]
    if calibration_models:
        report_lines.append("")
        report_lines.append("Component calibration models:")
        for item in calibration_models:
            report_lines.append(
                f"- C{item['component']} -> Y{item['y_column']} | "
                f"Cal R2={_safe_float(item['calibration'].get('R2'), default=np.nan):.4f}, "
                f"Cal RMSEP={_safe_float(item['calibration'].get('RMSEP'), default=np.nan):.6g}"
            )

    output = {
        "scores_mode_a": C_cal,
        "val_scores_mode_a": C_val,
        "scores_mode_a_heatmap": np.asarray(C_cal, dtype=float).T,
        "mode_a_sample_axis": np.arange(1, int(C_cal.shape[0]) + 1, dtype=float),
        "mode_a_component_axis": np.arange(1, int(C_cal.shape[1]) + 1, dtype=float),
        "sweep_F": None,
        "sweep_sfit": None,
        "sweep_explained_variance": None,
        "components": ST,
        "concentrations": C_cal,
        "reconstructed": reconstructed,
        "residual": residual,
        "metrics": metrics,
        "calibration_models": calibration_models,
        "component_y_mapping": {str(k + 1): int(v + 1) for k, v in mapping.items()},
        "mcr_als_report": "\n".join(report_lines),
        "selected_n_components": int(n_components),
        "cv_results": None,
        "y_cv_pred": None,
        "y_cal_pred": y_cal_pred,
        "y_val_pred": y_val_pred,
        "y_cal_error": y_cal_error,
        "y_val_error": y_val_error,
        "y_cv_error": None,
        "y_cal_true": y_cal_true,
        "y_val_true": y_val_true,
        "data_shape": tuple(X_cal_raw.shape),
    }

    output.update(
        _build_pair_outputs(
            component_y_mapping=output.get("component_y_mapping", {}),
            y_labels=[],
            y_cal_pred=output.get("y_cal_pred"),
            y_cv_pred=output.get("y_cv_pred"),
            y_cal_true=output.get("y_cal_true"),
            y_cal_error=output.get("y_cal_error"),
            y_cv_error=output.get("y_cv_error"),
            y_val_pred=output.get("y_val_pred"),
            y_val_true=output.get("y_val_true"),
            y_val_error=output.get("y_val_error"),
        )
    )

    return output


def mcr_als_analysis(
    X_cal: Optional[np.ndarray] = None,
    Y_cal: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    n_components: int = 2,
    max_iter: int = 250,
    tol: float = 1e-8,
    random_state: Optional[Any] = 42,
    c_regr: str = "OLS",
    st_regr: str = "OLS",
    c_nonneg: Any = True,
    st_nonneg: Any = True,
    c_norm: Any = False,
    sweep_mode: bool = False,
    component_range: str = "",
    component_y_mapping: Any = "",
    cv_config: Optional[Any] = None,
    y_labels: Optional[Any] = None,
    nway_flag: Optional[Any] = None,
    aug_direction: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """MCR-ALS with optional component sweep, calibration, and CV support."""

    if X_cal is None and "X_cal_train" in kwargs:
        X_cal = kwargs["X_cal_train"]
    if Y_cal is None and "Y_cal_train" in kwargs:
        Y_cal = kwargs["Y_cal_train"]
    if X_cal is None:
        raise ValueError("X_cal is required for mcr_als_analysis")

    y_labels_resolved = _normalize_y_labels(y_labels if y_labels is not None else kwargs.get("y_labels"))
    resolved_nway_flag = _safe_int(nway_flag if nway_flag is not None else kwargs.get("nway_flag"), default=1)
    if resolved_nway_flag < 1:
        resolved_nway_flag = 1

    resolved_aug_direction: Optional[int] = None
    aug_raw = aug_direction if aug_direction is not None else kwargs.get("aug_direction")
    if aug_raw is not None and str(aug_raw).strip() != "":
        parsed_aug = _safe_int(aug_raw, default=0)
        if parsed_aug > 0 and resolved_nway_flag >= 2:
            if parsed_aug > resolved_nway_flag:
                raise ValueError(f"aug_direction must be between 1 and nway_flag ({resolved_nway_flag}).")
            resolved_aug_direction = int(parsed_aug)
    seed_value = None
    if random_state is not None and str(random_state).strip() != "":
        try:
            seed_value = int(float(random_state))
        except Exception:
            seed_value = None

    if cv_config is not None and HAS_CV and hasattr(cv_config, "is_enabled") and cv_config.is_enabled():
        if not _is_cv_fold_call():
            X_arr = np.asarray(X_cal, dtype=float)
            pipeline = CVPipeline(cv_config)
            splits = list(pipeline.splitter.get_splits(X_arr, None))

            Y2 = _as_2d_y(Y_cal)
            ycv_pred = None if Y2 is None else np.full_like(Y2, np.nan, dtype=float)
            fold_metrics: List[Dict[str, Any]] = []

            for fidx, (train_idx, test_idx) in enumerate(splits):
                fold_result = mcr_als_analysis(
                    X_cal=X_arr[train_idx],
                    Y_cal=None if Y_cal is None else np.asarray(Y_cal)[train_idx],
                    X_val=X_arr[test_idx],
                    Y_val=None if Y_cal is None else np.asarray(Y_cal)[test_idx],
                    n_components=n_components,
                    max_iter=max_iter,
                    tol=tol,
                    random_state=seed_value,
                    c_regr=c_regr,
                    st_regr=st_regr,
                    c_nonneg=c_nonneg,
                    st_nonneg=st_nonneg,
                    c_norm=c_norm,
                    sweep_mode=False,
                    component_y_mapping=component_y_mapping,
                    cv_config=None,
                    y_labels=y_labels_resolved,
                    nway_flag=resolved_nway_flag,
                    aug_direction=resolved_aug_direction,
                    aug_row_counts_cal=kwargs.get("aug_row_counts_cal"),
                    aug_row_counts_val=kwargs.get("aug_row_counts_val"),
                )

                fold_metrics.append(
                    {
                        "fold": int(fidx),
                        "n_test": int(len(test_idx)),
                        "sfit": _safe_float(fold_result.get("metrics", {}).get("calibration", {}).get("sfit"), default=np.nan),
                        "explained_variance": _safe_float(
                            fold_result.get("metrics", {}).get("calibration", {}).get("explained_variance"),
                            default=np.nan,
                        ),
                    }
                )

                if ycv_pred is not None:
                    fold_pred = _as_2d_y(fold_result.get("y_val_pred"))
                    if fold_pred is not None:
                        cols = min(ycv_pred.shape[1], fold_pred.shape[1])
                        ycv_pred[np.asarray(test_idx), :cols] = fold_pred[:, :cols]

            cv_agg: Dict[str, Any] = {"n_folds": int(len(splits)), "fold_metrics": fold_metrics}
            if fold_metrics:
                cv_agg["sfit_mean"] = float(np.nanmean([item["sfit"] for item in fold_metrics]))
                cv_agg["explained_variance_mean"] = float(np.nanmean([item["explained_variance"] for item in fold_metrics]))

            full_result = mcr_als_analysis(
                X_cal=X_arr,
                Y_cal=Y_cal,
                X_val=X_val,
                Y_val=Y_val,
                n_components=n_components,
                max_iter=max_iter,
                tol=tol,
                random_state=seed_value,
                c_regr=c_regr,
                st_regr=st_regr,
                c_nonneg=c_nonneg,
                st_nonneg=st_nonneg,
                c_norm=c_norm,
                sweep_mode=sweep_mode,
                component_range=component_range,
                component_y_mapping=component_y_mapping,
                cv_config=None,
                y_labels=y_labels_resolved,
                nway_flag=resolved_nway_flag,
                aug_direction=resolved_aug_direction,
                aug_row_counts_cal=kwargs.get("aug_row_counts_cal"),
                aug_row_counts_val=kwargs.get("aug_row_counts_val"),
            )

            full_result["cv_results"] = cv_agg
            full_result.setdefault("metrics", {})
            full_result["metrics"]["cv"] = cv_agg
            full_result["y_cv_pred"] = ycv_pred
            if ycv_pred is not None and Y2 is not None:
                full_result["y_cv_error"] = np.asarray(Y2, dtype=float) - np.asarray(ycv_pred, dtype=float)
            return full_result

    ranks: List[int] = [int(max(1, _safe_int(n_components, default=2)))]
    sweep_results: List[Dict[str, Any]] = []
    if _safe_bool(sweep_mode, default=False):
        parsed = parse_numeric_spec(component_range)
        if len(parsed) == 1:
            one = max(1, _safe_int(parsed[0], default=ranks[0]))
            if one >= 2:
                parsed = list(range(1, one + 1))
        candidate = sorted({max(1, _safe_int(item, default=1)) for item in parsed})
        if candidate:
            ranks = candidate

        for rk in ranks:
            try:
                fit_rk = _single_fit(
                    X_cal=np.asarray(X_cal, dtype=float),
                    Y_cal=Y_cal,
                    X_val=None,
                    Y_val=None,
                    n_components=int(rk),
                    max_iter=max_iter,
                    tol=tol,
                    random_state=seed_value,
                    c_regr=c_regr,
                    st_regr=st_regr,
                    c_nonneg=_safe_bool(c_nonneg, default=True),
                    st_nonneg=_safe_bool(st_nonneg, default=True),
                    c_norm=_safe_bool(c_norm, default=False),
                    component_y_mapping=component_y_mapping,
                    nway_flag=resolved_nway_flag,
                    aug_direction=resolved_aug_direction,
                    aug_row_counts_cal=kwargs.get("aug_row_counts_cal"),
                )
                cal_m = fit_rk.get("metrics", {}).get("calibration", {})
                sweep_results.append(
                    {
                        "n_components": int(rk),
                        "sfit": _safe_float(cal_m.get("sfit"), default=np.nan),
                        "explained_variance": _safe_float(cal_m.get("explained_variance"), default=np.nan),
                        "n_iter": _safe_int(cal_m.get("n_iter"), default=0),
                    }
                )
            except Exception as exc:
                emit_execution_warning(
                    code="mcr_als_sweep_rank_failed",
                    text=f"MCR-ALS sweep rank {rk} failed and was skipped.",
                    details={"rank": int(rk), "error": str(exc)},
                )

    selected_rank = int(max(1, _safe_int(n_components, default=2)))
    result = _single_fit(
        X_cal=np.asarray(X_cal, dtype=float),
        Y_cal=Y_cal,
        X_val=X_val,
        Y_val=Y_val,
        n_components=selected_rank,
        max_iter=max_iter,
        tol=tol,
        random_state=seed_value,
        c_regr=c_regr,
        st_regr=st_regr,
        c_nonneg=_safe_bool(c_nonneg, default=True),
        st_nonneg=_safe_bool(st_nonneg, default=True),
        c_norm=_safe_bool(c_norm, default=False),
        component_y_mapping=component_y_mapping,
        nway_flag=resolved_nway_flag,
        aug_direction=resolved_aug_direction,
        aug_row_counts_cal=kwargs.get("aug_row_counts_cal"),
        aug_row_counts_val=kwargs.get("aug_row_counts_val"),
    )

    if sweep_results:
        result["sweep_results"] = sweep_results
        result["sweep_F"] = np.asarray([_safe_float(item.get("n_components")) for item in sweep_results], dtype=float)
        result["sweep_sfit"] = np.asarray([_safe_float(item.get("sfit")) for item in sweep_results], dtype=float)
        result["sweep_explained_variance"] = np.asarray(
            [_safe_float(item.get("explained_variance")) for item in sweep_results],
            dtype=float,
        )
    else:
        result["sweep_results"] = None

    pair_outputs = _build_pair_outputs(
        component_y_mapping=result.get("component_y_mapping", {}),
        y_labels=y_labels_resolved,
        y_cal_pred=result.get("y_cal_pred"),
        y_cv_pred=result.get("y_cv_pred"),
        y_cal_true=result.get("y_cal_true"),
        y_cal_error=result.get("y_cal_error"),
        y_cv_error=result.get("y_cv_error"),
        y_val_pred=result.get("y_val_pred"),
        y_val_true=result.get("y_val_true"),
        y_val_error=result.get("y_val_error"),
    )
    result.update(pair_outputs)

    return result


_MCR_ALS_RETURN_ORDER: Tuple[str, ...] = (
    "scores_mode_a",
    "val_scores_mode_a",
    "scores_mode_a_heatmap",
    "mode_a_sample_axis",
    "mode_a_component_axis",
    "sweep_F",
    "sweep_sfit",
    "sweep_explained_variance",
    "components",
    "concentrations",
    "reconstructed",
    "residual",
    "metrics",
    "calibration_models",
    "component_y_mapping",
    "mcr_als_report",
    "sweep_results",
    "selected_n_components",
    "cv_results",
    "y_cv_pred",
    "y_cal_pred",
    "y_val_pred",
    "y_cal_error",
    "y_val_error",
    "y_cv_error",
    "y_cal_true",
    "y_val_true",
    "mcr_pair_components",
    "mcr_pair_y_columns",
    "mcr_pair_y_titles",
    "mcr_pairing_labels",
    "mcr_pairing_labels_by_dimension",
    "y_cal_pred_pairs",
    "y_cv_pred_pairs",
    "y_cal_true_pairs",
    "y_cal_error_pairs",
    "y_cv_error_pairs",
    "y_val_pred_pairs",
    "y_val_true_pairs",
    "y_val_error_pairs",
)


def mcr_als_analysis_standard(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:
    """Adapter for app execution pipeline: return outputs as ordered tuple."""
    result = mcr_als_analysis(*args, **kwargs)
    if not isinstance(result, dict):
        return (result,)
    return tuple(result.get(key) for key in _MCR_ALS_RETURN_ORDER)
