"""MCR-ALS analysis based on pyMCR with optional calibration and CV support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import nnls as _scipy_nnls

from chemometrics.input_parsing import parse_numeric_spec

try:
    from execution_reporting import emit_execution_message, emit_execution_warning
except ImportError:
    def emit_execution_message(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        return

    def emit_execution_warning(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        return

try:
    from pymcr.constraints import (  # type: ignore[import-not-found]
        ConstraintCompressAbove,
        ConstraintCompressBelow,
        ConstraintCumsumNonneg,
        ConstraintCutAbove,
        ConstraintCutBelow,
        ConstraintNonneg,
        ConstraintNorm,
        ConstraintPlanarize,
        ConstraintReplaceZeros,
        ConstraintZeroCumSumEndPoints,
        ConstraintZeroEndPoints,
    )
    from pymcr.mcr import McrAR  # type: ignore[import-not-found]
    HAS_PYMCR = True
except Exception:
    HAS_PYMCR = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _min_max(vals: List[float]) -> Tuple[float, float]:
    return (min(vals), max(vals))


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


def _safe_optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except Exception:
        return default


def _normalize_bool_list(values: Any, count: int) -> List[bool]:
    if isinstance(values, np.ndarray):
        raw = np.asarray(values).reshape(-1).tolist()
    elif isinstance(values, (list, tuple)):
        raw = list(values)
    elif isinstance(values, str):
        text = values.strip()
        if "," in text or ";" in text:
            raw = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
        elif text:
            val = _safe_bool(text, default=False)
            return [val] * int(max(1, count))
        else:
            raw = []
    elif values is None:
        raw = []
    else:
        val = _safe_bool(values, default=False)
        return [val] * int(max(1, count))

    out: List[bool] = []
    for item in raw[:count]:
        out.append(_safe_bool(item, default=False))
    if len(out) < count:
        out.extend([False] * (count - len(out)))
    return out


def _resolve_constraint_flags(
    values: Any,
    legacy_c: Any,
    legacy_st: Any,
    default_c: bool,
    default_st: bool,
) -> List[bool]:
    if values is None:
        return [
            _safe_bool(legacy_c, default=default_c),
            _safe_bool(legacy_st, default=default_st),
        ]
    return _normalize_bool_list(values, count=2)


def _normalize_non_negative_mode(value: Any, default: str = "nnls") -> str:
    text = str(value).strip().lower() if value is not None else str(default).strip().lower()
    normalized = text.replace("+", "_").replace(" ", "_").replace("-", "_")

    if normalized in {"nnls", "non_negative_least_squares"}:
        return "nnls"
    if normalized in {"project", "projection", "clip", "constraint", "constraint_nonneg"}:
        return "clip"
    if normalized in {
        "nnls_project",
        "project_nnls",
        "nnls_projection",
        "projection_nnls",
        "nnls_clip",
        "clip_nnls",
        "nnlsandclip",
        "nnls_and_clip",
    }:
        return "nnls_clip"
    return "nnls"


def _parse_one_based_index_list(value: Any) -> List[int]:
    parsed = parse_numeric_spec(value)
    out: List[int] = []
    for item in parsed:
        idx = _safe_int(item, default=0) - 1
        if idx >= 0:
            out.append(int(idx))
    return sorted(set(out))


def _parse_planarize_shape(value: Any, expected_rows: int) -> Tuple[int, int]:
    text = "" if value is None else str(value).strip().lower().replace("x", ",")
    if not text:
        return int(expected_rows), 1

    parts = [segment.strip() for segment in text.split(",") if segment.strip()]
    if len(parts) != 2:
        raise ValueError("Planarize shape must be provided as 'rows,cols'.")

    rows = max(1, _safe_int(parts[0], default=0))
    cols = max(1, _safe_int(parts[1], default=0))
    if int(rows * cols) != int(expected_rows):
        raise ValueError(
            f"Planarize shape product ({rows}x{cols}={rows*cols}) must equal matrix row count ({expected_rows})."
        )
    return int(rows), int(cols)


def _build_mcr_constraints(
    c_nonneg: Any,
    st_nonneg: Any,
    c_norm: Any,
    constraint_non_negative: Any,
    constraint_non_negative_clip: Any,
    constraint_cumsum_non_negative: Any,
    constraint_zero_end_points: Any,
    constraint_zero_cumsum_end_points: Any,
    constraint_normalize: Any,
    constraint_cut_below: Any,
    constraint_cut_above: Any,
    constraint_compress_below: Any,
    constraint_compress_above: Any,
    constraint_replace_zeros: Any,
    constraint_planarize: Any,
    constraint_zero_end_points_span: Any,
    constraint_zero_cumsum_nodes: Any,
    constraint_cut_below_value: Any,
    constraint_cut_above_value: Any,
    constraint_compress_below_value: Any,
    constraint_compress_above_value: Any,
    constraint_replace_zeros_feature: Any,
    constraint_replace_zeros_fval: Any,
    constraint_planarize_targets_c: Any,
    constraint_planarize_targets_st: Any,
    constraint_planarize_shape_c: Any,
    constraint_planarize_shape_st: Any,
    constraint_planarize_use_vals_above: Any,
    constraint_planarize_use_vals_below: Any,
    constraint_planarize_lims_to_plane: Any,
    constraint_planarize_scaler: Any,
    constraint_planarize_recalc_scaler: Any,
    c_rows: int,
    c_cols: int,
    st_rows: int,
    st_cols: int,
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    matrix_specs = [
        {"name": "C", "rows": int(c_rows), "cols": int(c_cols), "constraints": []},
        {"name": "ST", "rows": int(st_rows), "cols": int(st_cols), "constraints": []},
    ]

    flags_map: Dict[str, List[bool]] = {
        "non_negative": _resolve_constraint_flags(
            values=constraint_non_negative,
            legacy_c=c_nonneg,
            legacy_st=st_nonneg,
            default_c=True,
            default_st=True,
        ),
        "cumsum_non_negative": _resolve_constraint_flags(
            values=constraint_cumsum_non_negative,
            legacy_c=False,
            legacy_st=False,
            default_c=False,
            default_st=False,
        ),
        "zero_end_points": _resolve_constraint_flags(
            values=constraint_zero_end_points,
            legacy_c=False,
            legacy_st=False,
            default_c=False,
            default_st=False,
        ),
        "zero_cumsum_end_points": _resolve_constraint_flags(
            values=constraint_zero_cumsum_end_points,
            legacy_c=False,
            legacy_st=False,
            default_c=False,
            default_st=False,
        ),
        "normalize": _resolve_constraint_flags(
            values=constraint_normalize,
            legacy_c=c_norm,
            legacy_st=False,
            default_c=False,
            default_st=False,
        ),
        "cut_below": _resolve_constraint_flags(
            values=constraint_cut_below,
            legacy_c=False,
            legacy_st=False,
            default_c=False,
            default_st=False,
        ),
        "cut_above": _resolve_constraint_flags(
            values=constraint_cut_above,
            legacy_c=False,
            legacy_st=False,
            default_c=False,
            default_st=False,
        ),
        "compress_below": _resolve_constraint_flags(
            values=constraint_compress_below,
            legacy_c=False,
            legacy_st=False,
            default_c=False,
            default_st=False,
        ),
        "compress_above": _resolve_constraint_flags(
            values=constraint_compress_above,
            legacy_c=False,
            legacy_st=False,
            default_c=False,
            default_st=False,
        ),
        "replace_zeros": _resolve_constraint_flags(
            values=constraint_replace_zeros,
            legacy_c=False,
            legacy_st=False,
            default_c=False,
            default_st=False,
        ),
        "planarize": _resolve_constraint_flags(
            values=constraint_planarize,
            legacy_c=False,
            legacy_st=False,
            default_c=False,
            default_st=False,
        ),
    }
    apply_flags_map: Dict[str, List[bool]] = dict(flags_map)
    if constraint_non_negative_clip is not None:
        apply_flags_map["non_negative"] = _normalize_bool_list(constraint_non_negative_clip, count=2)

    span = max(1, _safe_int(constraint_zero_end_points_span, default=1))
    zero_cumsum_nodes = _parse_one_based_index_list(constraint_zero_cumsum_nodes)
    cut_below_value = _safe_float(constraint_cut_below_value, default=0.0)
    cut_above_value = _safe_float(constraint_cut_above_value, default=1.0)
    compress_below_value = _safe_float(constraint_compress_below_value, default=0.0)
    compress_above_value = _safe_float(constraint_compress_above_value, default=1.0)
    replace_feature = max(0, _safe_int(constraint_replace_zeros_feature, default=1) - 1)
    replace_fval = _safe_float(constraint_replace_zeros_fval, default=1.0)

    planarize_use_above = _safe_optional_float(constraint_planarize_use_vals_above, default=None)
    planarize_use_below = _safe_optional_float(constraint_planarize_use_vals_below, default=None)
    planarize_scaler = _safe_optional_float(constraint_planarize_scaler, default=None)
    planarize_lims = _safe_bool(constraint_planarize_lims_to_plane, default=True)
    planarize_recalc = _safe_bool(constraint_planarize_recalc_scaler, default=False)

    def _append_selected(name: str, factory) -> None:
        flags = apply_flags_map.get(name, [False, False])
        for matrix_idx, enabled in enumerate(flags[:2]):
            if not bool(enabled):
                continue
            matrix_specs[matrix_idx]["constraints"].append(factory(matrix_idx, matrix_specs[matrix_idx]))

    _append_selected("non_negative", lambda _idx, _spec: ConstraintNonneg())
    _append_selected("cumsum_non_negative", lambda _idx, _spec: ConstraintCumsumNonneg(axis=-1))
    _append_selected("zero_end_points", lambda _idx, _spec: ConstraintZeroEndPoints(axis=-1, span=span))
    _append_selected(
        "zero_cumsum_end_points",
        lambda _idx, _spec: ConstraintZeroCumSumEndPoints(nodes=(zero_cumsum_nodes or None), axis=-1),
    )
    _append_selected("normalize", lambda _idx, _spec: ConstraintNorm(axis=-1))
    _append_selected("cut_below", lambda _idx, _spec: ConstraintCutBelow(value=cut_below_value))
    _append_selected("cut_above", lambda _idx, _spec: ConstraintCutAbove(value=cut_above_value))
    _append_selected("compress_below", lambda _idx, _spec: ConstraintCompressBelow(value=compress_below_value))
    _append_selected("compress_above", lambda _idx, _spec: ConstraintCompressAbove(value=compress_above_value))
    _append_selected(
        "replace_zeros",
        lambda _idx, spec: ConstraintReplaceZeros(
            axis=-1,
            feature=min(replace_feature, max(0, int(spec.get("cols", 1)) - 1)),
            fval=replace_fval,
        ),
    )

    def _build_planarize(matrix_idx: int, matrix_spec: Dict[str, Any]) -> Any:
        if matrix_idx == 0:
            target_raw = constraint_planarize_targets_c
            shape_raw = constraint_planarize_shape_c
        else:
            target_raw = constraint_planarize_targets_st
            shape_raw = constraint_planarize_shape_st

        targets = [idx for idx in _parse_one_based_index_list(target_raw) if idx < int(matrix_spec["cols"])]
        if not targets:
            targets = [0]

        shape = _parse_planarize_shape(shape_raw, expected_rows=int(matrix_spec["rows"]))
        return ConstraintPlanarize(
            target=targets,
            shape=shape,
            use_vals_above=planarize_use_above,
            use_vals_below=planarize_use_below,
            lims_to_plane=planarize_lims,
            scaler=planarize_scaler,
            recalc_scaler=planarize_recalc,
        )

    _append_selected("planarize", _build_planarize)

    c_constraints = list(matrix_specs[0]["constraints"])
    st_constraints = list(matrix_specs[1]["constraints"])

    summary = {
        "enabled_flags": {
            key: {"C": bool(vals[0]), "ST": bool(vals[1])}
            for key, vals in flags_map.items()
        },
        "applied_flags": {
            key: {"C": bool(vals[0]), "ST": bool(vals[1])}
            for key, vals in apply_flags_map.items()
        },
        "applied_constraints": {
            "C": [type(item).__name__ for item in c_constraints],
            "ST": [type(item).__name__ for item in st_constraints],
        },
    }
    return c_constraints, st_constraints, summary


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


def _prediction_metrics(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(yt) & np.isfinite(yp)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 2:
        return {"R2": float("nan"), "RMSE": float("nan"), "n_samples_used": n_valid}
    yt_v = yt[valid]
    yp_v = yp[valid]
    residuals = yt_v - yp_v
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((yt_v - np.mean(yt_v)) ** 2)) + 1e-12
    return {
        "R2": float(1.0 - ss_res / ss_tot),
        "RMSE": float(np.sqrt(np.mean(residuals ** 2))),
        "n_samples_used": n_valid,
    }


def _build_sbs_text_report(
    rank: int,
    pair_labels: Sequence[str],
    sbs_y_cal_true_pairs: Any,
    sbs_y_cal_pred_pairs: Any,
    y_val_true_pairs: Any,
    y_val_pred_pairs: Any,
    auto_mapping_used: bool = False,
    sbs_fom_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    fom = sbs_fom_metrics or {}
    impl = str(fom.get("implementation_used", "") or "MCR-ALS (pyMCR)")
    lines: List[str] = [
        "MCR-ALS Report",
        "==============",
        f"Components: {int(rank)}",
        f"Implementation: {impl}",
        "Validation Processing: Sample by Sample",
    ]

    def _fmt_range(pair: Optional[Tuple[float, float]], fmt: str = ".4f") -> str:
        if pair is None:
            return "n/a"
        lo, hi = pair
        return f"{lo:{fmt}}-{hi:{fmt}}"

    ssr = fom.get("SSR")
    sfit = fom.get("sfit")
    ev = fom.get("explained_variance")
    n_iter = fom.get("n_iter")
    if ssr is not None:
        lines.append(f"SSR (range): {_fmt_range(ssr, '.6g')}")
    if sfit is not None:
        lines.append(f"sfit (range): {_fmt_range(sfit, '.6g')}")
    if ev is not None:
        lines.append(f"Explained variance % (range): {_fmt_range(ev)}")
    if n_iter is not None:
        lo_i, hi_i = int(n_iter[0]), int(n_iter[1])
        lines.append(f"Iterations (range): {lo_i}-{hi_i}")
    lines.append("")

    labels = list(pair_labels) if pair_labels is not None else []
    cal_true = np.asarray(sbs_y_cal_true_pairs, dtype=float) if sbs_y_cal_true_pairs is not None else None
    cal_pred = np.asarray(sbs_y_cal_pred_pairs, dtype=float) if sbs_y_cal_pred_pairs is not None else None

    lines.append("")
    if bool(auto_mapping_used):
        lines.append("Regression Report (Paired by maximum likelihood):")
    else:
        lines.append("Regression Report:")

    if (
        isinstance(cal_true, np.ndarray)
        and isinstance(cal_pred, np.ndarray)
        and cal_true.ndim == 3
        and cal_pred.ndim == 3
        and cal_true.shape == cal_pred.shape
        and cal_true.shape[2] > 0
    ):
        n_pairs = int(cal_true.shape[2])
        if len(labels) < n_pairs:
            labels.extend([f"Pair {i + 1}" for i in range(len(labels), n_pairs)])
        lines.append("")
        lines.append("Calibration (SBS per-model ranges):")
        for pair_idx in range(n_pairs):
            rmse_vals: List[float] = []
            r2_vals: List[float] = []
            for model_idx in range(int(cal_true.shape[0])):
                m = _prediction_metrics(cal_true[model_idx, :, pair_idx], cal_pred[model_idx, :, pair_idx])
                rmse = _safe_float(m.get("RMSE"), default=np.nan)
                r2 = _safe_float(m.get("R2"), default=np.nan)
                if np.isfinite(rmse):
                    rmse_vals.append(float(rmse))
                if np.isfinite(r2):
                    r2_vals.append(float(r2))
            label = labels[pair_idx] if pair_idx < len(labels) else f"Pair {pair_idx + 1}"
            if rmse_vals and r2_vals:
                lines.append(
                    f"- {label} | R2 (range): {min(r2_vals):.4f}-{max(r2_vals):.4f}, "
                    f"RMSEC (range): {min(rmse_vals):.6g}-{max(rmse_vals):.6g}"
                )
            else:
                lines.append(f"- {label} | R2 (range): n/a, RMSEC (range): n/a")
    elif labels:
        lines.append("")
        lines.append("Calibration (SBS per-model ranges):")
        for label in labels:
            lines.append(f"- {label} | R2 (range): n/a, RMSEP (range): n/a")

    val_true = np.asarray(y_val_true_pairs, dtype=float) if y_val_true_pairs is not None else None
    val_pred = np.asarray(y_val_pred_pairs, dtype=float) if y_val_pred_pairs is not None else None
    if (
        isinstance(val_true, np.ndarray)
        and isinstance(val_pred, np.ndarray)
        and val_true.ndim == 2
        and val_pred.ndim == 2
        and val_true.shape == val_pred.shape
        and val_true.shape[1] > 0
    ):
        n_pairs_val = int(val_true.shape[1])
        if len(labels) < n_pairs_val:
            labels.extend([f"Pair {i + 1}" for i in range(len(labels), n_pairs_val)])
        lines.append("")
        lines.append("Validation (aggregated SBS predictions):")
        for pair_idx in range(n_pairs_val):
            m = _prediction_metrics(val_true[:, pair_idx], val_pred[:, pair_idx])
            rmse = _safe_float(m.get("RMSE"), default=np.nan)
            r2 = _safe_float(m.get("R2"), default=np.nan)
            label = labels[pair_idx] if pair_idx < len(labels) else f"Pair {pair_idx + 1}"
            if np.isfinite(rmse) and np.isfinite(r2):
                lines.append(f"- {label} | R2={r2:.4f}, RMSEP={rmse:.6g}")
            else:
                lines.append(f"- {label} | R2=n/a, RMSEP=n/a")

    return "\n".join(lines)


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


def _normalize_sample_mask(sample_mask: Any, x_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """Normalize an optional sample mask to boolean ndarray with shape matching X."""
    if sample_mask is None:
        return None
    arr = np.asarray(sample_mask)
    if arr.shape != tuple(x_shape):
        raise ValueError(
            f"Sample mask shape {tuple(arr.shape)} does not match X shape {tuple(x_shape)}."
        )
    return np.asarray(arr, dtype=bool)


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
    sample_mask: Optional[np.ndarray] = None,
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

        mask_arr = _normalize_sample_mask(sample_mask, tuple(arr.shape))

        # aug_direction now identifies the S (spectral) dimension of each sample tensor.
        # That axis is moved to the last position (columns); all remaining axes are
        # flattened into rows, so C always spans a single instrumental direction.
        s_axis = int(aug_direction) - 1
        if s_axis < 0 or s_axis >= arr.ndim - 1:
            raise ValueError(
                f"aug_direction={int(aug_direction)} is out of range for nway_flag={int(nway_flag)}. "
                f"Expected an integer in [1, {int(nway_flag)}]."
            )

        row_blocks: List[np.ndarray] = []
        row_counts: List[int] = []
        row_mask_blocks: List[np.ndarray] = []
        sample_s_lengths: List[int] = []

        # First pass: unfold each sample and collect per-sample valid S-length.
        for smp_idx in range(sample_count):
            sample_tensor = np.asarray(arr[smp_idx], dtype=float)
            sample_tensor = np.moveaxis(sample_tensor, s_axis, -1)  # S axis -> columns
            n_cols = int(sample_tensor.shape[-1])
            n_rows = int(np.prod(sample_tensor.shape[:-1]))
            row_data = sample_tensor.reshape(n_rows, n_cols, order="F")

            if mask_arr is None:
                row_mask = np.ones_like(row_data, dtype=bool)
            else:
                sample_mask_tensor = np.moveaxis(np.asarray(mask_arr[smp_idx], dtype=bool), s_axis, -1)
                row_mask = sample_mask_tensor.reshape(n_rows, n_cols, order="F")

            # Determine real S-axis support for this sample (how many leading columns contain real data).
            valid_cols = np.any(row_mask, axis=0)
            s_len = int(np.count_nonzero(valid_cols))
            if s_len <= 0:
                raise ValueError(
                    f"Sample {smp_idx + 1} has no real (non-padding) values after mask filtering."
                )

            row_blocks.append(row_data)
            row_mask_blocks.append(row_mask)
            sample_s_lengths.append(s_len)

        # Use the common real S support so no padded columns are fed into MCR.
        common_s_len = int(min(sample_s_lengths)) if sample_s_lengths else 0
        if common_s_len <= 0:
            raise ValueError("Could not determine a valid common S-axis length from sample masks.")

        filtered_blocks: List[np.ndarray] = []
        for smp_idx, (row_data, row_mask) in enumerate(zip(row_blocks, row_mask_blocks)):
            data_trim = np.asarray(row_data[:, :common_s_len], dtype=float)
            mask_trim = np.asarray(row_mask[:, :common_s_len], dtype=bool)

            # Keep rows containing any real values in the common S support.
            valid_rows = np.any(mask_trim, axis=1)
            if not np.any(valid_rows):
                raise ValueError(
                    f"Sample {smp_idx + 1} has no usable rows in the common S-axis range after mask filtering."
                )

            filtered = data_trim[valid_rows, :]
            if np.any(~np.isfinite(filtered)):
                raise ValueError(
                    "Augmented fit matrix still contains NaN/Inf after mask filtering; "
                    "check loaded data and padding strategy."
                )

            row_counts.append(int(filtered.shape[0]))
            filtered_blocks.append(filtered)

        D = np.vstack(filtered_blocks) if filtered_blocks else np.zeros((0, common_s_len), dtype=float)
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
    sample_mask: Optional[np.ndarray] = None,
    explicit_row_counts: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if aug_direction is not None and int(nway_flag) >= 2:
        return _prepare_augmented_fit_matrix(
            x=x,
            nway_flag=int(nway_flag),
            aug_direction=int(aug_direction),
            n_samples_hint=n_samples_hint,
            sample_mask=sample_mask,
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


def _aggregate_scores(
    C_rows: np.ndarray,
    row_ranges: Sequence[Tuple[int, int]],
    method: str = "area",
) -> np.ndarray:
    """Aggregate augmented concentration rows per sample using area, max, or min."""
    C = np.asarray(C_rows, dtype=float)
    if C.ndim != 2:
        C = np.asarray(C, dtype=float).reshape(-1, 1)

    m = str(method).strip().lower()
    out = np.full((int(len(row_ranges)), int(C.shape[1])), np.nan, dtype=float)
    for idx, (start, stop) in enumerate(row_ranges):
        s = int(start)
        e = int(stop)
        if e <= s or s < 0 or e > C.shape[0]:
            continue
        seg = C[s:e, :]
        if seg.shape[0] == 1:
            out[idx, :] = seg[0, :]
        elif m == "max":
            out[idx, :] = np.max(seg, axis=0)
        elif m == "min":
            out[idx, :] = np.min(seg, axis=0)
        else:  # default: area (trapezoid integration)
            out[idx, :] = np.trapezoid(seg, dx=1.0, axis=0)
    return out


def _normalize_aggregation_method(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else "area"
    if text in {"max", "maximum"}:
        return "max"
    if text in {"min", "minimum"}:
        return "min"
    return "area"


def _c_aggregation_label(method: str) -> str:
    m = str(method).strip().lower()
    if m == "max":
        return "$C_{Max}$"
    if m == "min":
        return "$C_{Min}$"
    return "$C_{Area}$"


def _coerce_axis_vector_for_s(raw_axis: Any, expected_len: int) -> Optional[np.ndarray]:
    if raw_axis is None:
        return None
    try:
        arr = np.asarray(raw_axis, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size != int(expected_len):
        return None
    return arr


def _build_pair_outputs(
    component_y_mapping: Dict[str, int],
    y_labels: Sequence[str],
    y_cal_true: Optional[np.ndarray],
    y_val_true: Optional[np.ndarray],
    c_scores_cal: Optional[np.ndarray] = None,
    c_scores_val: Optional[np.ndarray] = None,
    calibration_models: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build per-component-pairing prediction matrices and labels.

    Predictions are computed independently for each (component, Y-column) pair
    using that component's own calibration model (intercept + slope × C score),
    so pairings sharing a Y column produce distinct predictions.
    """
    def _to_2d(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr if arr.ndim == 2 else None

    def _extract_pair_matrix(value: Any, y_cols_1based: List[int]) -> Optional[np.ndarray]:
        arr = _to_2d(value)
        if arr is None or not y_cols_1based:
            return None
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

    # Build lookup: component_1based -> (intercept, slope)
    model_lookup: Dict[int, Tuple[float, float]] = {}
    if isinstance(calibration_models, list):
        for entry in calibration_models:
            if not isinstance(entry, dict):
                continue
            comp = entry.get("component")
            intercept = entry.get("intercept")
            slope = entry.get("slope")
            if comp is not None and intercept is not None and slope is not None:
                try:
                    model_lookup[int(comp)] = (float(intercept), float(slope))
                except (TypeError, ValueError):
                    pass

    scores_cal = _to_2d(c_scores_cal)
    scores_val = _to_2d(c_scores_val)
    n_pairs = len(pairs)

    y_cal_true_pairs = _extract_pair_matrix(y_cal_true, pair_y_columns)
    y_val_true_pairs = _extract_pair_matrix(y_val_true, pair_y_columns)

    # Per-pairing calibration predictions: each column uses the component's own model
    y_cal_pred_pairs: Optional[np.ndarray] = None
    y_cal_error_pairs: Optional[np.ndarray] = None
    if scores_cal is not None and n_pairs > 0:
        n_cal = scores_cal.shape[0]
        pred_cal = np.full((n_cal, n_pairs), np.nan, dtype=float)
        for pi, comp_1 in enumerate(pair_components):
            comp_idx = comp_1 - 1
            if comp_1 not in model_lookup or comp_idx >= scores_cal.shape[1]:
                continue
            intercept, slope = model_lookup[comp_1]
            col = scores_cal[:, comp_idx]
            valid = np.isfinite(col)
            pred_cal[valid, pi] = intercept + slope * col[valid]
        y_cal_pred_pairs = pred_cal
        if y_cal_true_pairs is not None:
            y_cal_error_pairs = y_cal_true_pairs - pred_cal

    # Per-pairing validation predictions using each component's own model
    y_val_pred_pairs: Optional[np.ndarray] = None
    y_val_error_pairs: Optional[np.ndarray] = None
    if scores_val is not None and n_pairs > 0:
        n_val = scores_val.shape[0]
        pred_val = np.full((n_val, n_pairs), np.nan, dtype=float)
        for pi, comp_1 in enumerate(pair_components):
            comp_idx = comp_1 - 1
            if comp_1 not in model_lookup or comp_idx >= scores_val.shape[1]:
                continue
            intercept, slope = model_lookup[comp_1]
            col = scores_val[:, comp_idx]
            valid = np.isfinite(col)
            pred_val[valid, pi] = intercept + slope * col[valid]
        y_val_pred_pairs = pred_val
        if y_val_true_pairs is not None:
            y_val_error_pairs = y_val_true_pairs - pred_val

    # Per-pairing C score matrices (n_samples x n_pairs)
    c_scores_cal_pairs: Optional[np.ndarray] = None
    if scores_cal is not None and n_pairs > 0:
        cols: List[np.ndarray] = []
        for comp_1 in pair_components:
            comp_idx = comp_1 - 1
            if 0 <= comp_idx < scores_cal.shape[1]:
                cols.append(scores_cal[:, comp_idx:comp_idx + 1])
            else:
                cols.append(np.full((scores_cal.shape[0], 1), np.nan, dtype=float))
        c_scores_cal_pairs = np.column_stack(cols) if cols else None

    c_scores_val_pairs: Optional[np.ndarray] = None
    if scores_val is not None and n_pairs > 0:
        cols_v: List[np.ndarray] = []
        for comp_1 in pair_components:
            comp_idx = comp_1 - 1
            if 0 <= comp_idx < scores_val.shape[1]:
                cols_v.append(scores_val[:, comp_idx:comp_idx + 1])
            else:
                cols_v.append(np.full((scores_val.shape[0], 1), np.nan, dtype=float))
        c_scores_val_pairs = np.column_stack(cols_v) if cols_v else None

    # Effective validation Y: true reference where available, else predicted
    if y_val_true_pairs is not None and y_val_pred_pairs is not None:
        y_val_effective_pairs: Optional[np.ndarray] = np.where(
            np.isfinite(y_val_true_pairs), y_val_true_pairs, y_val_pred_pairs,
        )
    elif y_val_true_pairs is not None:
        y_val_effective_pairs = y_val_true_pairs.copy()
    elif y_val_pred_pairs is not None:
        y_val_effective_pairs = y_val_pred_pairs.copy()
    else:
        y_val_effective_pairs = None

    # Regression line endpoints per pairing (shape 2 x n_pairs)
    cal_regression_line_x_pairs: Optional[np.ndarray] = None
    cal_regression_line_y_pairs: Optional[np.ndarray] = None
    if c_scores_cal_pairs is not None and n_pairs > 0:
        line_x = np.full((2, n_pairs), np.nan, dtype=float)
        line_y = np.full((2, n_pairs), np.nan, dtype=float)
        for pi, comp_1 in enumerate(pair_components):
            score_col = np.asarray(c_scores_cal_pairs[:, pi], dtype=float)
            finite_score = np.isfinite(score_col)
            if not np.any(finite_score):
                continue

            # Plot uses Reference on x and Score on y. Prefer reference-domain
            # limits so the line spans the full calibration x-axis range.
            x1 = x2 = np.nan
            y1 = y2 = np.nan
            if y_cal_true_pairs is not None and pi < y_cal_true_pairs.shape[1]:
                ref_col = np.asarray(y_cal_true_pairs[:, pi], dtype=float)
                valid_xy = np.isfinite(ref_col) & np.isfinite(score_col)
                if int(np.count_nonzero(valid_xy)) >= 2:
                    fit_xy = _fit_linear_1d(ref_col[valid_xy], score_col[valid_xy])
                    b0_xy = _safe_float(fit_xy.get("intercept"), default=np.nan)
                    b1_xy = _safe_float(fit_xy.get("slope"), default=np.nan)
                    if np.isfinite(b0_xy) and np.isfinite(b1_xy):
                        ref_valid = ref_col[valid_xy]
                        ref_min = float(np.nanmin(ref_valid))
                        ref_max = float(np.nanmax(ref_valid))
                        ref_span = ref_max - ref_min if ref_max > ref_min else 1.0
                        ref_buf = ref_span * 0.05
                        x1 = ref_min - ref_buf
                        x2 = ref_max + ref_buf
                        y1 = b0_xy + b1_xy * x1
                        y2 = b0_xy + b1_xy * x2

            if (not (np.isfinite(x1) and np.isfinite(x2) and np.isfinite(y1) and np.isfinite(y2))) and comp_1 in model_lookup:
                intercept, slope = model_lookup[comp_1]
                if (
                    y_cal_true_pairs is not None
                    and pi < y_cal_true_pairs.shape[1]
                    and np.isfinite(intercept)
                    and np.isfinite(slope)
                ):
                    ref_col = np.asarray(y_cal_true_pairs[:, pi], dtype=float)
                    finite_ref = np.isfinite(ref_col)
                    if np.any(finite_ref):
                        ref_min = float(np.nanmin(ref_col[finite_ref]))
                        ref_max = float(np.nanmax(ref_col[finite_ref]))
                        ref_span = ref_max - ref_min if ref_max > ref_min else 1.0
                        ref_buf = ref_span * 0.05
                        x1 = ref_min - ref_buf
                        x2 = ref_max + ref_buf
                        y1 = (x1 - float(intercept)) / float(slope)
                        y2 = (x2 - float(intercept)) / float(slope)

            if not (np.isfinite(x1) and np.isfinite(x2) and np.isfinite(y1) and np.isfinite(y2)):
                score_min = float(np.nanmin(score_col[finite_score]))
                score_max = float(np.nanmax(score_col[finite_score]))
                score_span = score_max - score_min if score_max > score_min else 1.0
                score_buf = score_span * 0.05
                y1 = score_min - score_buf
                y2 = score_max + score_buf
                if comp_1 in model_lookup:
                    intercept, slope = model_lookup[comp_1]
                    x1 = float(intercept) + float(slope) * y1
                    x2 = float(intercept) + float(slope) * y2

            line_x[0, pi] = x1
            line_x[1, pi] = x2
            line_y[0, pi] = y1
            line_y[1, pi] = y2
        if not np.all(np.isnan(line_x)):
            # Keep naming semantic aligned with plotted axes: Reference on x, Score on y.
            cal_regression_line_x_pairs = line_x
            cal_regression_line_y_pairs = line_y

    # Diagonal line extent for Predicted vs Reference validation scatter
    val_pred_ref_diag_extent: Optional[float] = None
    val_pred_ref_diag_x: Optional[np.ndarray] = None
    val_pred_ref_diag_y: Optional[np.ndarray] = None
    candidates_for_diag: List[np.ndarray] = []
    if y_val_true_pairs is not None:
        candidates_for_diag.append(y_val_true_pairs)
    if y_val_pred_pairs is not None:
        candidates_for_diag.append(y_val_pred_pairs)
    if candidates_for_diag:
        all_vals = np.concatenate([arr.ravel() for arr in candidates_for_diag])
        finite_vals = all_vals[np.isfinite(all_vals)]
        if finite_vals.size > 0:
            max_abs = float(np.max(np.abs(finite_vals)))
            val_pred_ref_diag_extent = max_abs * 1.15 + 1e-6
            val_pred_ref_diag_x = np.array([-val_pred_ref_diag_extent, val_pred_ref_diag_extent], dtype=float)
            val_pred_ref_diag_y = val_pred_ref_diag_x.copy()

    return {
        "mcr_pair_components": np.asarray(pair_components, dtype=int) if pair_components else np.asarray([], dtype=int),
        "mcr_pair_y_columns": np.asarray(pair_y_columns, dtype=int) if pair_y_columns else np.asarray([], dtype=int),
        "mcr_pair_y_titles": y_titles,
        "mcr_pairing_labels": labels,
        "mcr_pairing_labels_by_dimension": [[], labels],
        "y_cal_pred_pairs": y_cal_pred_pairs,
        "y_cal_true_pairs": y_cal_true_pairs,
        "y_cal_error_pairs": y_cal_error_pairs,
        "y_val_pred_pairs": y_val_pred_pairs,
        "y_val_true_pairs": y_val_true_pairs,
        "y_val_error_pairs": y_val_error_pairs,
        "c_scores_cal_pairs": c_scores_cal_pairs,
        "c_scores_val_pairs": c_scores_val_pairs,
        "y_val_effective_pairs": y_val_effective_pairs,
        "cal_regression_line_x_pairs": cal_regression_line_x_pairs,
        "cal_regression_line_y_pairs": cal_regression_line_y_pairs,
        "val_pred_ref_diag_extent": val_pred_ref_diag_extent,
        "val_pred_ref_diag_x": val_pred_ref_diag_x,
        "val_pred_ref_diag_y": val_pred_ref_diag_y,
    }


def _build_unified_prediction_matrices(
    y_cal_true: Any,
    y_val_true: Any,
    y_cal_pred_pairs: Any,
    y_val_pred_pairs: Any,
    pair_y_columns: List[int],
) -> Dict[str, Any]:
    """Rebuild unified (n_samples × n_y) prediction matrices from per-pairing data.

    For each Y column j:
      - No pairing       → NaN column
      - One pairing      → use that pairing's predictions directly
      - Multiple pairings→ use the pairing with the highest Pearson r against
                           the cal reference; same selection applied to val
    """
    def _to_2d(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr if arr.ndim == 2 else None

    cal = _to_2d(y_cal_true)
    val = _to_2d(y_val_true)
    pred_cal = _to_2d(y_cal_pred_pairs)
    pred_val = _to_2d(y_val_pred_pairs)

    n_y: Optional[int] = None
    for arr in (cal, val):
        if arr is not None:
            n_y = arr.shape[1]
            break
    if n_y is None:
        return {"y_cal_pred": None, "y_val_pred": None, "y_cal_error": None, "y_val_error": None}

    col_to_pairs: Dict[int, List[int]] = {}
    for pi, y_col in enumerate(pair_y_columns):
        col_to_pairs.setdefault(int(y_col), []).append(pi)

    def _best_pair_idx(j_1based: int, candidates: List[int]) -> int:
        if len(candidates) == 1:
            return candidates[0]
        if cal is None or pred_cal is None:
            return candidates[0]
        j_idx = j_1based - 1
        if j_idx >= cal.shape[1]:
            return candidates[0]
        true_col = cal[:, j_idx]
        best_idx = candidates[0]
        best_r2 = -np.inf
        for pi in candidates:
            if pi >= pred_cal.shape[1]:
                continue
            pred_col = pred_cal[:, pi]
            mask = np.isfinite(true_col) & np.isfinite(pred_col)
            if mask.sum() < 2:
                continue
            r2 = float(np.corrcoef(true_col[mask], pred_col[mask])[0, 1]) ** 2
            if r2 > best_r2:
                best_r2 = r2
                best_idx = pi
        return best_idx

    best_pair_for_col: Dict[int, int] = {
        j_1based: _best_pair_idx(j_1based, candidates)
        for j_1based, candidates in col_to_pairs.items()
    }

    def _build_matrix(pred_pairs: Optional[np.ndarray], n_samples: int) -> np.ndarray:
        out = np.full((n_samples, n_y), np.nan, dtype=float)
        if pred_pairs is None:
            return out
        for j_1based, best_pi in best_pair_for_col.items():
            j_idx = j_1based - 1
            if j_idx < n_y and best_pi < pred_pairs.shape[1]:
                out[:, j_idx] = pred_pairs[:, best_pi]
        return out

    unified_cal_pred: Optional[np.ndarray] = None
    if pred_cal is not None:
        unified_cal_pred = _build_matrix(pred_cal, pred_cal.shape[0])
    elif cal is not None:
        unified_cal_pred = np.full((cal.shape[0], n_y), np.nan, dtype=float)

    unified_val_pred: Optional[np.ndarray] = None
    if pred_val is not None:
        unified_val_pred = _build_matrix(pred_val, pred_val.shape[0])
    elif val is not None:
        unified_val_pred = np.full((val.shape[0], n_y), np.nan, dtype=float)

    unified_cal_error: Optional[np.ndarray] = None
    if unified_cal_pred is not None and cal is not None:
        unified_cal_error = cal - unified_cal_pred

    unified_val_error: Optional[np.ndarray] = None
    if unified_val_pred is not None and val is not None:
        unified_val_error = val - unified_val_pred

    return {
        "y_cal_pred": unified_cal_pred,
        "y_val_pred": unified_val_pred,
        "y_cal_error": unified_cal_error,
        "y_val_error": unified_val_error,
    }


def _gather_sweep_outputs_mcr(
    sweep_model_results: Sequence[Dict[str, Any]],
    sweep_ranks: Sequence[int],
    y_labels: Sequence[str],
    y_cal_true: Any,
    y_val_true: Any,
) -> Dict[str, Any]:
    """Collect sweep-model tensors and pairwise calibration outputs for MCR-ALS."""
    out: Dict[str, Any] = {
        "sweep_model_axis": None,
        "sweep_model_labels": None,
        "sweep_component_labels": None,
        "sweep_c_component_axis": None,
        "sweep_c_sample_axis_full": None,
        "sweep_c_scores": None,
        "sweep_val_c_scores": None,
        "sweep_c_scores_heatmap_full": None,
        "sweep_concentrations_unfolded": None,
        "sweep_s_profiles": None,
        "sweep_mcr_pair_components": None,
        "sweep_mcr_pair_y_columns": None,
        "sweep_mcr_pair_y_titles": None,
        "sweep_mcr_pair_y_titles_by_model": None,
        "sweep_mcr_pairing_labels": None,
        "sweep_mcr_pairing_labels_by_model": None,
        "sweep_mcr_pairing_labels_by_dimension": None,
        "sweep_y_cal_pred_pairs": None,
        "sweep_y_cal_true_pairs": None,
        "sweep_y_cal_error_pairs": None,
        "sweep_y_val_pred_pairs": None,
        "sweep_y_val_true_pairs": None,
        "sweep_y_val_error_pairs": None,
        "sweep_c_scores_cal_pairs": None,
        "sweep_c_scores_val_pairs": None,
        "sweep_y_val_effective_pairs": None,
        "sweep_cal_regression_line_x_pairs": None,
        "sweep_cal_regression_line_y_pairs": None,
        "sweep_val_pred_ref_diag_extent": None,
        "sweep_val_pred_ref_diag_x": None,
        "sweep_val_pred_ref_diag_y": None,
        "sweep_ejcr_cal": None,
        "sweep_ejcr_val": None,
    }

    models = [m for m in sweep_model_results if isinstance(m, dict)]
    if not models:
        return out

    n_models = len(models)
    ranks = [int(r) for r in sweep_ranks[:n_models]]
    if len(ranks) < n_models:
        ranks.extend(list(range(1, n_models + 1))[len(ranks):])

    out["sweep_model_axis"] = np.arange(1, n_models + 1, dtype=float)
    out["sweep_model_labels"] = [f"C={r}" for r in ranks]

    max_comp = 0
    cal_scores = [m.get("c_scores") for m in models]
    cal_shapes = [arr.shape for arr in cal_scores if isinstance(arr, np.ndarray) and arr.ndim == 2]
    if cal_shapes:
        n_cal = int(max(shape[0] for shape in cal_shapes))
        max_comp = int(max(shape[1] for shape in cal_shapes))
        out["sweep_component_labels"] = [f"C{i + 1}" for i in range(max_comp)]
        out["sweep_c_component_axis"] = np.arange(1, max_comp + 1, dtype=float)

        sweep_cal = np.full((n_models, n_cal, max_comp), np.nan, dtype=float)
        for model_idx, arr in enumerate(cal_scores):
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                continue
            rr = min(n_cal, int(arr.shape[0]))
            cc = min(max_comp, int(arr.shape[1]))
            sweep_cal[model_idx, :rr, :cc] = np.asarray(arr[:rr, :cc], dtype=float)
        out["sweep_c_scores"] = sweep_cal

        val_scores = [m.get("val_c_scores") for m in models]
        val_shapes = [arr.shape for arr in val_scores if isinstance(arr, np.ndarray) and arr.ndim == 2]
        n_val = int(max(shape[0] for shape in val_shapes)) if val_shapes else 0
        if n_val > 0:
            sweep_val = np.full((n_models, n_val, max_comp), np.nan, dtype=float)
            for model_idx, arr in enumerate(val_scores):
                if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                    continue
                rr = min(n_val, int(arr.shape[0]))
                cc = min(max_comp, int(arr.shape[1]))
                sweep_val[model_idx, :rr, :cc] = np.asarray(arr[:rr, :cc], dtype=float)
            out["sweep_val_c_scores"] = sweep_val

        n_full = n_cal + n_val
        heat = np.full((n_models, max_comp, n_full), np.nan, dtype=float)
        for model_idx in range(n_models):
            cal = out["sweep_c_scores"][model_idx, :, :] if isinstance(out.get("sweep_c_scores"), np.ndarray) else None
            if not isinstance(cal, np.ndarray):
                continue
            full = cal
            if isinstance(out.get("sweep_val_c_scores"), np.ndarray):
                full = np.vstack([full, out["sweep_val_c_scores"][model_idx, :, :]])
            heat[model_idx, :, :full.shape[0]] = np.asarray(full, dtype=float).T
        out["sweep_c_scores_heatmap_full"] = np.nan_to_num(heat, nan=0.0)
        out["sweep_c_sample_axis_full"] = np.arange(1, n_full + 1, dtype=float)

    concentrations = [m.get("concentrations_unfolded") for m in models]
    conc_shapes = [arr.shape for arr in concentrations if isinstance(arr, np.ndarray) and arr.ndim == 2]
    if conc_shapes:
        if max_comp <= 0:
            max_comp = int(max(shape[0] for shape in conc_shapes))
            out["sweep_component_labels"] = [f"C{i + 1}" for i in range(max_comp)]
            out["sweep_c_component_axis"] = np.arange(1, max_comp + 1, dtype=float)
        n_rows = int(max(shape[1] for shape in conc_shapes))
        sweep_conc = np.full((n_models, max_comp, n_rows), np.nan, dtype=float)
        for model_idx, arr in enumerate(concentrations):
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                continue
            rr = min(max_comp, int(arr.shape[0]))
            cc = min(n_rows, int(arr.shape[1]))
            sweep_conc[model_idx, :rr, :cc] = np.asarray(arr[:rr, :cc], dtype=float)
        out["sweep_concentrations_unfolded"] = sweep_conc

    s_profiles = [m.get("s_profiles") for m in models]
    s_shapes = [arr.shape for arr in s_profiles if isinstance(arr, np.ndarray) and arr.ndim == 2]
    if s_shapes:
        if max_comp <= 0:
            max_comp = int(max(shape[0] for shape in s_shapes))
            out["sweep_component_labels"] = [f"C{i + 1}" for i in range(max_comp)]
            out["sweep_c_component_axis"] = np.arange(1, max_comp + 1, dtype=float)
        n_vars = int(max(shape[1] for shape in s_shapes))
        sweep_s = np.full((n_models, max_comp, n_vars), np.nan, dtype=float)
        for model_idx, arr in enumerate(s_profiles):
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                continue
            rr = min(max_comp, int(arr.shape[0]))
            cc = min(n_vars, int(arr.shape[1]))
            sweep_s[model_idx, :rr, :cc] = np.asarray(arr[:rr, :cc], dtype=float)
        out["sweep_s_profiles"] = sweep_s

    y_cal_arr = _as_2d_y(y_cal_true)
    y_val_arr = _as_2d_y(y_val_true)
    pair_payloads: List[Dict[str, Any]] = []
    max_pair_component = 0
    default_y_by_component: Dict[int, int] = {}
    for model in models:
        p = _build_pair_outputs(
            component_y_mapping=model.get("component_y_mapping"),
            y_labels=y_labels,
            y_cal_true=y_cal_arr,
            y_val_true=y_val_arr,
            c_scores_cal=model.get("c_scores"),
            c_scores_val=model.get("val_c_scores"),
            calibration_models=model.get("calibration_models"),
        )
        pair_payloads.append(p)
        comp = p.get("mcr_pair_components")
        ycol = p.get("mcr_pair_y_columns")
        if isinstance(comp, np.ndarray) and isinstance(ycol, np.ndarray):
            for j in range(min(comp.size, ycol.size)):
                comp_1 = int(comp[j])
                y_1 = int(ycol[j])
                if comp_1 < 1:
                    continue
                max_pair_component = max(max_pair_component, comp_1)
                if comp_1 not in default_y_by_component and y_1 >= 1:
                    default_y_by_component[comp_1] = y_1

    if max_pair_component <= 0:
        return out

    n_pairs = int(max_pair_component)
    out["sweep_mcr_pair_components"] = np.arange(1, n_pairs + 1, dtype=int)
    out["sweep_mcr_pair_y_columns"] = np.asarray(
        [int(default_y_by_component.get(comp_1, 0)) for comp_1 in range(1, n_pairs + 1)],
        dtype=int,
    )

    labels_seq = list(y_labels) if y_labels is not None else []
    y_titles: List[str] = []
    pair_labels: List[str] = []
    for comp_1 in range(1, n_pairs + 1):
        y_1 = int(default_y_by_component.get(comp_1, 0))
        y_title = f"Y{y_1}" if y_1 >= 1 else "Unmapped"
        y_idx = y_1 - 1
        if 0 <= y_idx < len(labels_seq):
            label_text = str(labels_seq[y_idx]).strip()
            if label_text:
                y_title = label_text
        y_titles.append(y_title)
        if y_1 >= 1:
            pair_labels.append(f"C{comp_1} -> {y_title} (Y{y_1})")
        else:
            pair_labels.append(f"C{comp_1}")

    out["sweep_mcr_pair_y_titles"] = y_titles
    out["sweep_mcr_pairing_labels"] = pair_labels
    out["sweep_mcr_pair_y_titles_by_model"] = [list(y_titles) for _ in range(n_models)]
    out["sweep_mcr_pairing_labels_by_model"] = [list(pair_labels) for _ in range(n_models)]
    out["sweep_mcr_pairing_labels_by_dimension"] = [
        out.get("sweep_model_labels") or [],
        [],
        pair_labels,
    ]

    n_cal = int(y_cal_arr.shape[0]) if isinstance(y_cal_arr, np.ndarray) else 0
    n_val = int(y_val_arr.shape[0]) if isinstance(y_val_arr, np.ndarray) else 0

    def _alloc(rows: int) -> Optional[np.ndarray]:
        return np.full((n_models, rows, n_pairs), np.nan, dtype=float) if rows > 0 else None

    out["sweep_y_cal_pred_pairs"] = _alloc(n_cal)
    out["sweep_y_cal_true_pairs"] = _alloc(n_cal)
    out["sweep_y_cal_error_pairs"] = _alloc(n_cal)
    out["sweep_y_val_pred_pairs"] = _alloc(n_val)
    out["sweep_y_val_true_pairs"] = _alloc(n_val)
    out["sweep_y_val_error_pairs"] = _alloc(n_val)
    out["sweep_c_scores_cal_pairs"] = _alloc(n_cal)
    out["sweep_c_scores_val_pairs"] = _alloc(n_val)
    out["sweep_y_val_effective_pairs"] = _alloc(n_val)
    out["sweep_cal_regression_line_x_pairs"] = np.full((n_models, 2, n_pairs), np.nan, dtype=float)
    out["sweep_cal_regression_line_y_pairs"] = np.full((n_models, 2, n_pairs), np.nan, dtype=float)
    out["sweep_val_pred_ref_diag_extent"] = np.full((n_models, n_pairs), np.nan, dtype=float)
    out["sweep_val_pred_ref_diag_x"] = np.full((n_models, 2, n_pairs), np.nan, dtype=float)
    out["sweep_val_pred_ref_diag_y"] = np.full((n_models, 2, n_pairs), np.nan, dtype=float)

    def _copy_matrix(src: Any, dst_key: str, model_idx: int, idx_map: Dict[int, int]) -> None:
        dst = out.get(dst_key)
        if not isinstance(dst, np.ndarray) or not isinstance(src, np.ndarray) or src.ndim != 2:
            return
        rr = min(dst.shape[1], src.shape[0])
        for g, l in idx_map.items():
            if l < src.shape[1]:
                dst[model_idx, :rr, g] = np.asarray(src[:rr, l], dtype=float)

    for model_idx, payload in enumerate(pair_payloads):
        comp = payload.get("mcr_pair_components")
        ycol = payload.get("mcr_pair_y_columns")
        if not isinstance(comp, np.ndarray):
            continue
        idx_map: Dict[int, int] = {}
        for local_idx in range(int(comp.size)):
            comp_1 = int(comp[local_idx])
            if 1 <= comp_1 <= n_pairs:
                idx_map[comp_1 - 1] = int(local_idx)

        model_titles: List[str] = []
        model_pair_labels: List[str] = []
        y_by_component: Dict[int, int] = {}
        if isinstance(ycol, np.ndarray):
            for local_idx in range(min(int(comp.size), int(ycol.size))):
                comp_1 = int(comp[local_idx])
                y_1 = int(ycol[local_idx])
                if comp_1 >= 1 and y_1 >= 1 and comp_1 not in y_by_component:
                    y_by_component[comp_1] = y_1
        for comp_1 in range(1, n_pairs + 1):
            y_1 = int(y_by_component.get(comp_1, 0))
            y_title = f"Y{y_1}" if y_1 >= 1 else "Unmapped"
            y_idx = y_1 - 1
            if 0 <= y_idx < len(labels_seq):
                label_text = str(labels_seq[y_idx]).strip()
                if label_text:
                    y_title = label_text
            model_titles.append(y_title)
            if y_1 >= 1:
                model_pair_labels.append(f"C{comp_1} -> {y_title} (Y{y_1})")
            else:
                model_pair_labels.append(f"C{comp_1}")
        if isinstance(out.get("sweep_mcr_pair_y_titles_by_model"), list) and model_idx < len(out["sweep_mcr_pair_y_titles_by_model"]):
            out["sweep_mcr_pair_y_titles_by_model"][model_idx] = model_titles
        if isinstance(out.get("sweep_mcr_pairing_labels_by_model"), list) and model_idx < len(out["sweep_mcr_pairing_labels_by_model"]):
            out["sweep_mcr_pairing_labels_by_model"][model_idx] = model_pair_labels

        _copy_matrix(payload.get("y_cal_pred_pairs"), "sweep_y_cal_pred_pairs", model_idx, idx_map)
        _copy_matrix(payload.get("y_cal_true_pairs"), "sweep_y_cal_true_pairs", model_idx, idx_map)
        _copy_matrix(payload.get("y_cal_error_pairs"), "sweep_y_cal_error_pairs", model_idx, idx_map)
        _copy_matrix(payload.get("y_val_pred_pairs"), "sweep_y_val_pred_pairs", model_idx, idx_map)
        _copy_matrix(payload.get("y_val_true_pairs"), "sweep_y_val_true_pairs", model_idx, idx_map)
        _copy_matrix(payload.get("y_val_error_pairs"), "sweep_y_val_error_pairs", model_idx, idx_map)
        _copy_matrix(payload.get("c_scores_cal_pairs"), "sweep_c_scores_cal_pairs", model_idx, idx_map)
        _copy_matrix(payload.get("c_scores_val_pairs"), "sweep_c_scores_val_pairs", model_idx, idx_map)
        _copy_matrix(payload.get("y_val_effective_pairs"), "sweep_y_val_effective_pairs", model_idx, idx_map)

        line_x = payload.get("cal_regression_line_x_pairs")
        line_y = payload.get("cal_regression_line_y_pairs")
        if isinstance(line_x, np.ndarray) and isinstance(line_y, np.ndarray) and line_x.ndim == 2 and line_y.ndim == 2:
            for g, l in idx_map.items():
                if l < line_x.shape[1] and l < line_y.shape[1]:
                    out["sweep_cal_regression_line_x_pairs"][model_idx, :, g] = line_x[:, l]
                    out["sweep_cal_regression_line_y_pairs"][model_idx, :, g] = line_y[:, l]

        yt = payload.get("y_val_true_pairs")
        yp = payload.get("y_val_pred_pairs")
        if isinstance(yt, np.ndarray) and isinstance(yp, np.ndarray) and yt.ndim == 2 and yp.ndim == 2:
            for g, l in idx_map.items():
                if l >= yt.shape[1] or l >= yp.shape[1]:
                    continue
                vals = np.concatenate([
                    np.asarray(yt[:, l], dtype=float).reshape(-1),
                    np.asarray(yp[:, l], dtype=float).reshape(-1),
                ])
                finite = vals[np.isfinite(vals)]
                if finite.size <= 0:
                    continue
                extent = float(np.max(np.abs(finite))) * 1.15 + 1e-6
                out["sweep_val_pred_ref_diag_extent"][model_idx, g] = extent
                out["sweep_val_pred_ref_diag_x"][model_idx, :, g] = np.array([-extent, extent], dtype=float)
                out["sweep_val_pred_ref_diag_y"][model_idx, :, g] = np.array([-extent, extent], dtype=float)

    try:
        from chemometrics.ejcr_analysis import compute_ejcr as _compute_ejcr

        _ejcr_n_pts = 100
        _ejcr_n_path = _ejcr_n_pts * 2 + 1
        _ejcr_levels = ("90", "95", "99")
        _ejcr_level_to_idx = {lvl: i for i, lvl in enumerate(_ejcr_levels)}
        _ejcr_item_count = len(_ejcr_levels) + 1

        def _new_packed_ejcr(color: str, nav_shape: Tuple[int, ...]) -> Dict[str, Any]:
            x_paths = np.full((_ejcr_item_count, *nav_shape, _ejcr_n_path), np.nan, dtype=float)
            y_paths = np.full((_ejcr_item_count, *nav_shape, _ejcr_n_path), np.nan, dtype=float)
            fit_slope = np.full((1, *nav_shape), np.nan, dtype=float)
            fit_intercept = np.full((1, *nav_shape), np.nan, dtype=float)
            x_paths[len(_ejcr_levels), ..., 0] = 1.0
            y_paths[len(_ejcr_levels), ..., 0] = 0.0
            return {
                "fit_slope": fit_slope,
                "fit_intercept": fit_intercept,
                "x_paths": x_paths,
                "y_paths": y_paths,
                "labels": ["90% EJCR", "95% EJCR", "99% EJCR", "Ideal (1, 0)"],
                "confidence_levels": ["90", "95", "99", None],
                "linestyles": [":", "-", "--", "none"],
                "linewidths": [1.2, 1.5, 1.2, 0.0],
                "alphas": [0.75, 0.90, 0.75, 0.95],
                "markers": [None, None, None, "o"],
                "markersizes": [7, 7, 7, 7],
                "colors": [color, color, color, "black"],
                "expand_limits": True,
            }

        def _fill_packed(y_ref_1d, y_pred_1d, packed: Dict[str, Any], nav_idx: Tuple[int, ...]) -> None:
            valid = np.isfinite(y_ref_1d) & np.isfinite(y_pred_1d)
            if valid.sum() < 3:
                return
            r = _compute_ejcr(y_ref_1d[valid], y_pred_1d[valid], n_points=_ejcr_n_pts)
            packed["fit_slope"][(0, *nav_idx)] = r["slope"]
            packed["fit_intercept"][(0, *nav_idx)] = r["intercept"]
            for ell in r["ellipses"]:
                pct = str(ell.get("confidence_pct", "")).strip()
                idx = _ejcr_level_to_idx.get(pct)
                if idx is None:
                    continue
                es = np.asarray(ell["ellipse_slope"], dtype=float)
                ei = np.asarray(ell["ellipse_intercept"], dtype=float)
                k = min(len(es), _ejcr_n_path)
                packed["x_paths"][(idx, *nav_idx, slice(0, k))] = es[:k]
                packed["y_paths"][(idx, *nav_idx, slice(0, k))] = ei[:k]

        sweep_ejcr_cal = _new_packed_ejcr("steelblue", (n_models, n_pairs))
        sweep_ejcr_val = _new_packed_ejcr("darkorange", (n_models, n_pairs))
        sw_cal_true = out.get("sweep_y_cal_true_pairs")
        sw_cal_pred = out.get("sweep_y_cal_pred_pairs")
        sw_val_true = out.get("sweep_y_val_true_pairs")
        sw_val_pred = out.get("sweep_y_val_pred_pairs")
        if (
            isinstance(sw_cal_true, np.ndarray)
            and isinstance(sw_cal_pred, np.ndarray)
            and sw_cal_true.ndim == 3
            and sw_cal_pred.ndim == 3
            and sw_cal_true.shape == sw_cal_pred.shape
        ):
            for m in range(sw_cal_true.shape[0]):
                for p in range(sw_cal_true.shape[2]):
                    _fill_packed(sw_cal_true[m, :, p], sw_cal_pred[m, :, p], sweep_ejcr_cal, (m, p))
                    if (
                        isinstance(sw_val_true, np.ndarray)
                        and isinstance(sw_val_pred, np.ndarray)
                        and sw_val_true.ndim == 3
                        and sw_val_pred.ndim == 3
                        and sw_val_true.shape == sw_val_pred.shape
                        and m < sw_val_true.shape[0]
                        and p < sw_val_true.shape[2]
                    ):
                        _fill_packed(sw_val_true[m, :, p], sw_val_pred[m, :, p], sweep_ejcr_val, (m, p))
            out["sweep_ejcr_cal"] = sweep_ejcr_cal
            out["sweep_ejcr_val"] = sweep_ejcr_val
    except Exception:
        pass

    return out


def _single_fit(
    X_cal: np.ndarray,
    Y_cal: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    Y_val: Optional[np.ndarray],
    cal_s_mask: Optional[np.ndarray],
    val_s_mask: Optional[np.ndarray],
    n_components: int,
    max_iter: int,
    tol: float,
    random_state: Optional[int],
    c_regr: str,
    st_regr: str,
    c_nonneg: bool,
    st_nonneg: bool,
    c_norm: bool,
    constraint_non_negative: Any,
    constraint_non_negative_mode_c: Any,
    constraint_non_negative_mode_st: Any,
    constraint_cumsum_non_negative: Any,
    constraint_zero_end_points: Any,
    constraint_zero_cumsum_end_points: Any,
    constraint_normalize: Any,
    constraint_cut_below: Any,
    constraint_cut_above: Any,
    constraint_compress_below: Any,
    constraint_compress_above: Any,
    constraint_replace_zeros: Any,
    constraint_planarize: Any,
    constraint_zero_end_points_span: Any,
    constraint_zero_cumsum_nodes: Any,
    constraint_cut_below_value: Any,
    constraint_cut_above_value: Any,
    constraint_compress_below_value: Any,
    constraint_compress_above_value: Any,
    constraint_replace_zeros_feature: Any,
    constraint_replace_zeros_fval: Any,
    constraint_planarize_targets_c: Any,
    constraint_planarize_targets_st: Any,
    constraint_planarize_shape_c: Any,
    constraint_planarize_shape_st: Any,
    constraint_planarize_use_vals_above: Any,
    constraint_planarize_use_vals_below: Any,
    constraint_planarize_lims_to_plane: Any,
    constraint_planarize_scaler: Any,
    constraint_planarize_recalc_scaler: Any,
    component_y_mapping: Any,
    nway_flag: int,
    aug_direction: Optional[int],
    c_aggregation_method: str = "area",
    axis_n_info: Optional[Any] = None,
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
        sample_mask=None if cal_s_mask is None else np.asarray(cal_s_mask, dtype=bool),
        explicit_row_counts=cal_row_counts,
    )

    val_prep = None
    if X_val_raw is not None:
        val_prep = _prepare_fit_matrix(
            x=X_val_raw,
            nway_flag=int(nway_flag),
            aug_direction=aug_direction,
            n_samples_hint=None if Yv is None else int(Yv.shape[0]),
            sample_mask=None if val_s_mask is None else np.asarray(val_s_mask, dtype=bool),
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

    non_negative_selected = _resolve_constraint_flags(
        values=constraint_non_negative,
        legacy_c=c_nonneg,
        legacy_st=st_nonneg,
        default_c=True,
        default_st=True,
    )
    mode_c = _normalize_non_negative_mode(constraint_non_negative_mode_c, default="nnls")
    mode_st = _normalize_non_negative_mode(constraint_non_negative_mode_st, default="nnls")

    c_clip = bool(non_negative_selected[0]) and mode_c in {"clip", "nnls_clip"}
    st_clip = bool(non_negative_selected[1]) and mode_st in {"clip", "nnls_clip"}
    effective_c_regr = "NNLS" if (bool(non_negative_selected[0]) and mode_c in {"nnls", "nnls_clip"}) else "OLS"
    effective_st_regr = "NNLS" if (bool(non_negative_selected[1]) and mode_st in {"nnls", "nnls_clip"}) else "OLS"

    c_constraints, st_constraints, constraint_summary = _build_mcr_constraints(
        c_nonneg=c_nonneg,
        st_nonneg=st_nonneg,
        c_norm=c_norm,
        constraint_non_negative=constraint_non_negative,
        constraint_non_negative_clip=[c_clip, st_clip],
        constraint_cumsum_non_negative=constraint_cumsum_non_negative,
        constraint_zero_end_points=constraint_zero_end_points,
        constraint_zero_cumsum_end_points=constraint_zero_cumsum_end_points,
        constraint_normalize=constraint_normalize,
        constraint_cut_below=constraint_cut_below,
        constraint_cut_above=constraint_cut_above,
        constraint_compress_below=constraint_compress_below,
        constraint_compress_above=constraint_compress_above,
        constraint_replace_zeros=constraint_replace_zeros,
        constraint_planarize=constraint_planarize,
        constraint_zero_end_points_span=constraint_zero_end_points_span,
        constraint_zero_cumsum_nodes=constraint_zero_cumsum_nodes,
        constraint_cut_below_value=constraint_cut_below_value,
        constraint_cut_above_value=constraint_cut_above_value,
        constraint_compress_below_value=constraint_compress_below_value,
        constraint_compress_above_value=constraint_compress_above_value,
        constraint_replace_zeros_feature=constraint_replace_zeros_feature,
        constraint_replace_zeros_fval=constraint_replace_zeros_fval,
        constraint_planarize_targets_c=constraint_planarize_targets_c,
        constraint_planarize_targets_st=constraint_planarize_targets_st,
        constraint_planarize_shape_c=constraint_planarize_shape_c,
        constraint_planarize_shape_st=constraint_planarize_shape_st,
        constraint_planarize_use_vals_above=constraint_planarize_use_vals_above,
        constraint_planarize_use_vals_below=constraint_planarize_use_vals_below,
        constraint_planarize_lims_to_plane=constraint_planarize_lims_to_plane,
        constraint_planarize_scaler=constraint_planarize_scaler,
        constraint_planarize_recalc_scaler=constraint_planarize_recalc_scaler,
        c_rows=int(D_fit.shape[0]),
        c_cols=int(n_components),
        st_rows=int(n_components),
        st_cols=int(D_fit.shape[1]),
    )
    c_nonneg_applied = any(type(item).__name__ == "ConstraintNonneg" for item in c_constraints)

    mcr = McrAR(
        c_regr=str(effective_c_regr).upper(),
        st_regr=str(effective_st_regr).upper(),
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
            if str(effective_c_regr).upper() == "NNLS":
                nnls_rows: List[np.ndarray] = []
                a_mat = np.asarray(ST.T, dtype=float)
                for row in np.asarray(D_val, dtype=float):
                    coef_row, _ = _scipy_nnls(a_mat, np.asarray(row, dtype=float))
                    nnls_rows.append(np.asarray(coef_row, dtype=float))
                C_val_rows = np.asarray(nnls_rows, dtype=float)
            else:
                coef, *_ = np.linalg.lstsq(ST.T, D_val.T, rcond=None)
                C_val_rows = np.asarray(coef.T, dtype=float)
            if c_nonneg_applied:
                C_val_rows = np.maximum(C_val_rows, 0.0)
        else:
            C_val_rows = None

    if bool(cal_prep.get("is_augmented", False)):
        C_cal = _aggregate_scores(C_cal_rows, cal_prep.get("row_ranges", []), method=c_aggregation_method)
        if C_val_rows is not None and val_prep is not None:
            C_val = _aggregate_scores(C_val_rows, val_prep.get("row_ranges", []), method=c_aggregation_method)
        else:
            C_val = None
    else:
        C_cal = np.asarray(C_cal_rows, dtype=float)
        C_val = None if C_val_rows is None else np.asarray(C_val_rows, dtype=float)

    residual_cal = np.asarray(D_fit[:n_cal_rows] - reconstructed, dtype=float)
    ss_res = float(np.sum(residual_cal ** 2))
    n_obs = int(np.isfinite(residual_cal).sum())
    mean_ref = float(np.mean(D_fit[:n_cal_rows]))
    ss_tot = float(np.sum((D_fit[:n_cal_rows] - mean_ref) ** 2)) + 1e-12
    explained_variance = float(100.0 * (1.0 - ss_res / ss_tot))
    sfit = float(np.sqrt(ss_res / max(n_obs, 1)))

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
            "c_regr": str(effective_c_regr).upper(),
            "st_regr": str(effective_st_regr).upper(),
            "fit_combined_samples": bool(combine_fit_samples),
            "n_samples_fit": int(cal_prep["sample_count"] + (0 if val_prep is None else int(val_prep["sample_count"]))),
            "n_samples_calibration": int(n_cal_samples),
            "constraints": {
                "c_nonneg": bool(non_negative_selected[0]),
                "st_nonneg": bool(non_negative_selected[1]),
                "c_norm": bool(c_norm),
                "selected": constraint_summary.get("enabled_flags", {}),
                "applied_flags": constraint_summary.get("applied_flags", {}),
                "applied": constraint_summary.get("applied_constraints", {}),
                "non_negative_mode": {"C": mode_c, "S": mode_st},
            },
            "original_shape": tuple(X_cal_raw.shape),
            "fit_shape": fit_shape,
            "nway_flag": int(nway_flag),
            "s_direction": None if aug_direction is None else int(aug_direction),
            "augmented_mode": bool(cal_prep.get("is_augmented", False)),
            "augmented_row_counts_cal": [int(v) for v in cal_prep.get("row_counts", [])],
            "augmented_row_counts_val": [] if val_prep is None else [int(v) for v in val_prep.get("row_counts", [])],
        }
    }

    c_constraints_report = list(constraint_summary.get("applied_constraints", {}).get("C", []))
    s_constraints_report = list(constraint_summary.get("applied_constraints", {}).get("ST", []))

    if bool(non_negative_selected[0]):
        if mode_c == "nnls":
            c_constraints_report.append("Non-negative (NNLS)")
        elif mode_c == "clip":
            c_constraints_report.append("Non-negative (Projection)")
        elif mode_c == "nnls_clip":
            c_constraints_report.append("Non-negative (NNLS + Projection)")

    if bool(non_negative_selected[1]):
        if mode_st == "nnls":
            s_constraints_report.append("Non-negative (NNLS)")
        elif mode_st == "clip":
            s_constraints_report.append("Non-negative (Projection)")
        elif mode_st == "nnls_clip":
            s_constraints_report.append("Non-negative (NNLS + Projection)")

    c_constraints_report = list(dict.fromkeys(c_constraints_report))
    s_constraints_report = list(dict.fromkeys(s_constraints_report))

    report_lines = [
        "MCR-ALS Report",
        "==============",
        f"Components: {int(n_components)}",
        f"SSR: {ss_res:.6g}",
        f"sfit: {sfit:.6g}",
        f"Explained variance (%): {explained_variance:.4f}",
        f"Iterations: {int(getattr(mcr, 'n_iter', 0) or 0)}",
        "Implementation: MCR-ALS (pyMCR)",
        "Validation Processing: Batch",
        f"Calibration samples: {int(C_cal.shape[0])}",
        f"Validation samples: {0 if C_val is None else int(C_val.shape[0])}",
        f"Calibration features (unfolded): {int(D_fit.shape[1])}",
        f"Validation samples included in fitting: {bool(combine_fit_samples)}",
        f"C regressor: {str(effective_c_regr).upper()} | S regressor: {str(effective_st_regr).upper()}",
        f"Constraints C: {', '.join(c_constraints_report) or 'None'}",
        f"Constraints S: {', '.join(s_constraints_report) or 'None'}",
    ]
    if calibration_models:
        report_lines.append("")
        if bool(auto_mapping_used):
            report_lines.append("Regression Report (Paired by maximum likelihood):")
        else:
            report_lines.append("Regression Report:")
        
        report_lines.append("")
        report_lines.append("Calibration:")
        for item in calibration_models:
            report_lines.append(
                f"- C{item['component']} -> Y{item['y_column']} | "
                f"R2={_safe_float(item['calibration'].get('R2'), default=np.nan):.4f}, "
                f"RMSEC={_safe_float(item['calibration'].get('RMSEP'), default=np.nan):.6g}"
            )
        
        report_lines.append("")
        report_lines.append("Validation:")
        for item in calibration_models:
            vm = item.get("validation", {}) if isinstance(item, dict) else {}
            if vm:
                report_lines.append(
                    f"- C{item['component']} -> Y{item['y_column']} | "
                    f"R2={_safe_float(vm.get('R2'), default=np.nan):.4f}, "
                    f"RMSEP={_safe_float(vm.get('RMSEP'), default=np.nan):.6g}"
                )
    elif mapping:
        report_lines.append("")
        report_lines.append("Regression Report:")
        report_lines.append("Calibration mapping provided but no compatible Y data was available.")

    c_rows_plot = np.asarray(C_cal_rows, dtype=float)
    if C_val_rows is not None and np.asarray(C_val_rows).size > 0:
        c_rows_plot = np.vstack([c_rows_plot, np.asarray(C_val_rows, dtype=float)])

    concentrations_unfolded = np.asarray(c_rows_plot, dtype=float).T
    concentration_row_axis = np.arange(1, int(c_rows_plot.shape[0]) + 1, dtype=float)

    s_axis_vector: Optional[np.ndarray] = None
    axis_vectors: List[Any] = []
    if isinstance(axis_n_info, (list, tuple)):
        axis_vectors = list(axis_n_info)

    # aug_direction is now the S dimension index (1-based, within the non-sample modes).
    # axis_n_info[0] = samples, axis_n_info[k] = mode k, so axis_idx == aug_direction.
    if int(nway_flag) <= 1:
        s_dim = 1
    else:
        s_dim = int(aug_direction) if aug_direction is not None else 2
        if s_dim < 1 or s_dim > int(nway_flag):
            s_dim = 2
    non_aug_dims: List[int] = [s_dim]

    # Use physical axis values only when S spans a single direction (always true now).
    if len(non_aug_dims) == 1:
        axis_idx = int(non_aug_dims[0])
        if 0 <= axis_idx < len(axis_vectors):
            s_axis_vector = _coerce_axis_vector_for_s(axis_vectors[axis_idx], int(ST.shape[1]))
    sample_boundary_positions: List[float] = []
    row_counts_for_boundaries = [int(v) for v in cal_prep.get("row_counts", [])]
    if val_prep is not None:
        row_counts_for_boundaries.extend([int(v) for v in val_prep.get("row_counts", [])])

    cal_val_boundary_position: Optional[float] = None
    if C_val_rows is not None and int(C_val_rows.shape[0]) > 0:
        cal_boundary_idx = int(C_cal_rows.shape[0])
        if 0 < cal_boundary_idx < int(c_rows_plot.shape[0]):
            cal_val_boundary_position = float(cal_boundary_idx) + 0.5
    one_row_per_sample = (
        len(row_counts_for_boundaries) == int(C_cal_rows.shape[0])
        and all(int(v) == 1 for v in row_counts_for_boundaries)
    )
    if int(nway_flag) <= 1:
        # First-order data has one row per sample by definition; only show cal/val divider.
        one_row_per_sample = True
    if row_counts_for_boundaries and not one_row_per_sample:
        cursor = 0
        n_rows = int(c_rows_plot.shape[0])
        for count in row_counts_for_boundaries[:-1]:
            cursor += max(0, int(count))
            if 0 < cursor < n_rows:
                sample_boundary_positions.append(float(cursor) + 0.5)

    output = {
        "c_scores": C_cal,
        "val_c_scores": C_val,
        "c_scores_heatmap": np.asarray(C_cal, dtype=float).T,
        "c_scores_heatmap_full": (
            np.asarray(np.vstack([C_cal, C_val]), dtype=float).T
            if isinstance(C_val, np.ndarray) and C_val.ndim == 2 and C_val.shape[1] == C_cal.shape[1] and C_val.shape[0] > 0
            else np.asarray(C_cal, dtype=float).T
        ),
        "c_sample_axis": np.arange(1, int(C_cal.shape[0]) + 1, dtype=float),
        "c_sample_axis_full": (
            np.arange(1, int(C_cal.shape[0] + C_val.shape[0]) + 1, dtype=float)
            if isinstance(C_val, np.ndarray) and C_val.ndim == 2 and C_val.shape[1] == C_cal.shape[1] and C_val.shape[0] > 0
            else np.arange(1, int(C_cal.shape[0]) + 1, dtype=float)
        ),
        "c_component_axis": np.arange(1, int(C_cal.shape[1]) + 1, dtype=float),
        "component_labels": [f"C{i + 1}" for i in range(int(C_cal.shape[1]))],
        "sweep_components": None,
        "sweep_sfit": None,
        "sweep_n_iter": None,
        "sweep_explained_variance": None,
        "s_profiles": ST,
        "concentrations": C_cal,
        "concentrations_unfolded": concentrations_unfolded,
        "concentration_row_axis": concentration_row_axis,
        "s_axis_vector": s_axis_vector,
        "sample_boundary_positions": np.asarray(sample_boundary_positions, dtype=float),
        "cal_val_boundary_position": cal_val_boundary_position,
        "reconstructed": reconstructed,
        "residual": residual,
        "metrics": metrics,
        "calibration_models": calibration_models,
        "component_y_mapping": {str(k + 1): int(v + 1) for k, v in mapping.items()},
        "auto_mapping_used": bool(auto_mapping_used),
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

    pair_outputs_inner = _build_pair_outputs(
        component_y_mapping=output.get("component_y_mapping", {}),
        y_labels=[],
        y_cal_true=output.get("y_cal_true"),
        y_val_true=output.get("y_val_true"),
        c_scores_cal=C_cal,
        c_scores_val=C_val,
        calibration_models=output.get("calibration_models"),
    )
    output.update(pair_outputs_inner)
    # Override unified prediction matrices using best-correlation pairing selection
    pair_y_cols_inner: List[int] = (
        output["mcr_pair_y_columns"].tolist()
        if isinstance(output.get("mcr_pair_y_columns"), np.ndarray)
        else list(output.get("mcr_pair_y_columns") or [])
    )
    output.update(
        _build_unified_prediction_matrices(
            y_cal_true=output.get("y_cal_true"),
            y_val_true=output.get("y_val_true"),
            y_cal_pred_pairs=output.get("y_cal_pred_pairs"),
            y_val_pred_pairs=output.get("y_val_pred_pairs"),
            pair_y_columns=pair_y_cols_inner,
        )
    )

    return output


# ---------------------------------------------------------------------------
# Sample-by-sample (SBS) validation helper for MCR-ALS
# ---------------------------------------------------------------------------

def _cosine_matrix_mcr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between rows of a and rows of b. (m,d)×(n,d)→(m,n)."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_unit = a / np.where(a_norm == 0, 1.0, a_norm)
    b_unit = b / np.where(b_norm == 0, 1.0, b_norm)
    return a_unit @ b_unit.T


def _align_st_to_reference_mcr(
    ref_st: np.ndarray,
    st: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align ST rows of `st` to `ref_st` using Hungarian algorithm on cosine similarity.

    ref_st, st: (n_comp, n_vars)
    Returns (perm, signs) where perm[ref_comp] = model_comp.
    """
    from scipy.optimize import linear_sum_assignment

    n_comp = ref_st.shape[0]
    sim = _cosine_matrix_mcr(ref_st, st)        # (n_comp_ref, n_comp_model)
    row_ind, col_ind = linear_sum_assignment(-np.abs(sim))
    perm = np.zeros(n_comp, dtype=int)
    signs = np.ones(n_comp, dtype=float)
    for r, c in zip(row_ind, col_ind):
        perm[r] = int(c)
        signs[r] = -1.0 if sim[r, c] < 0 else 1.0
    return perm, signs


def _sbs_mcr(
    X_cal: np.ndarray,
    Y_cal: Optional[np.ndarray],
    X_val: np.ndarray,
    Y_val: Optional[np.ndarray],
    cal_s_mask: Optional[np.ndarray],
    val_s_mask: Optional[np.ndarray],
    n_components: int,
    max_iter: int,
    tol: float,
    random_state: Optional[int],
    c_regr: str,
    st_regr: str,
    c_nonneg: bool,
    st_nonneg: bool,
    c_norm: bool,
    constraint_non_negative: Any,
    constraint_non_negative_mode_c: Any,
    constraint_non_negative_mode_st: Any,
    constraint_cumsum_non_negative: Any,
    constraint_zero_end_points: Any,
    constraint_zero_cumsum_end_points: Any,
    constraint_normalize: Any,
    constraint_cut_below: Any,
    constraint_cut_above: Any,
    constraint_compress_below: Any,
    constraint_compress_above: Any,
    constraint_replace_zeros: Any,
    constraint_planarize: Any,
    constraint_zero_end_points_span: Any,
    constraint_zero_cumsum_nodes: Any,
    constraint_cut_below_value: Any,
    constraint_cut_above_value: Any,
    constraint_compress_below_value: Any,
    constraint_compress_above_value: Any,
    constraint_replace_zeros_feature: Any,
    constraint_replace_zeros_fval: Any,
    constraint_planarize_targets_c: Any,
    constraint_planarize_targets_st: Any,
    constraint_planarize_shape_c: Any,
    constraint_planarize_shape_st: Any,
    constraint_planarize_use_vals_above: Any,
    constraint_planarize_use_vals_below: Any,
    constraint_planarize_lims_to_plane: Any,
    constraint_planarize_scaler: Any,
    constraint_planarize_recalc_scaler: Any,
    component_y_mapping: Any,
    nway_flag: int,
    aug_direction: Optional[int],
    c_aggregation_method: str,
    axis_n_info: Optional[Any],
    aug_row_counts_cal: Optional[Any],
) -> Dict[str, Any]:
    """Run MCR-ALS sample-by-sample validation.

    For each validation sample *i*, X_val[i:i+1] is passed as X_val so it
    joins the MCR-ALS fit for second-order advantage, while calibration
    regression uses Y_cal only.  S profiles are aligned across models with
    the Hungarian algorithm on cosine similarity.
    """
    X_cal_arr = np.asarray(X_cal, dtype=float)
    X_val_arr = np.asarray(X_val, dtype=float)
    cal_s_mask_arr = None if cal_s_mask is None else np.asarray(cal_s_mask, dtype=bool)
    val_s_mask_arr = None if val_s_mask is None else np.asarray(val_s_mask, dtype=bool)
    n_val = int(X_val_arr.shape[0])
    n_comp = int(n_components)
    Y_cal_2d = _as_2d_y(Y_cal)
    Y_val_2d = _as_2d_y(Y_val)
    n_y = int(Y_cal_2d.shape[1]) if Y_cal_2d is not None else 0

    # ------------------------------------------------------------------ #
    # Shared kwargs for _single_fit calls                                  #
    # ------------------------------------------------------------------ #
    common_kwargs: Dict[str, Any] = dict(
        n_components=n_comp,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        c_regr=c_regr,
        st_regr=st_regr,
        c_nonneg=c_nonneg,
        st_nonneg=st_nonneg,
        c_norm=c_norm,
        constraint_non_negative=constraint_non_negative,
        constraint_non_negative_mode_c=constraint_non_negative_mode_c,
        constraint_non_negative_mode_st=constraint_non_negative_mode_st,
        constraint_cumsum_non_negative=constraint_cumsum_non_negative,
        constraint_zero_end_points=constraint_zero_end_points,
        constraint_zero_cumsum_end_points=constraint_zero_cumsum_end_points,
        constraint_normalize=constraint_normalize,
        constraint_cut_below=constraint_cut_below,
        constraint_cut_above=constraint_cut_above,
        constraint_compress_below=constraint_compress_below,
        constraint_compress_above=constraint_compress_above,
        constraint_replace_zeros=constraint_replace_zeros,
        constraint_planarize=constraint_planarize,
        constraint_zero_end_points_span=constraint_zero_end_points_span,
        constraint_zero_cumsum_nodes=constraint_zero_cumsum_nodes,
        constraint_cut_below_value=constraint_cut_below_value,
        constraint_cut_above_value=constraint_cut_above_value,
        constraint_compress_below_value=constraint_compress_below_value,
        constraint_compress_above_value=constraint_compress_above_value,
        constraint_replace_zeros_feature=constraint_replace_zeros_feature,
        constraint_replace_zeros_fval=constraint_replace_zeros_fval,
        constraint_planarize_targets_c=constraint_planarize_targets_c,
        constraint_planarize_targets_st=constraint_planarize_targets_st,
        constraint_planarize_shape_c=constraint_planarize_shape_c,
        constraint_planarize_shape_st=constraint_planarize_shape_st,
        constraint_planarize_use_vals_above=constraint_planarize_use_vals_above,
        constraint_planarize_use_vals_below=constraint_planarize_use_vals_below,
        constraint_planarize_lims_to_plane=constraint_planarize_lims_to_plane,
        constraint_planarize_scaler=constraint_planarize_scaler,
        constraint_planarize_recalc_scaler=constraint_planarize_recalc_scaler,
        component_y_mapping=component_y_mapping,
        nway_flag=nway_flag,
        aug_direction=aug_direction,
        c_aggregation_method=c_aggregation_method,
        axis_n_info=axis_n_info,
        aug_row_counts_cal=aug_row_counts_cal,
    )

    # ------------------------------------------------------------------ #
    # Per-model outputs (before alignment)                                #
    # ------------------------------------------------------------------ #
    raw_st: List[Optional[np.ndarray]] = []         # (n_comp, n_vars)
    raw_c_scores_cal: List[Optional[np.ndarray]] = []  # (n_cal, n_comp)
    raw_val_c_scores: List[Optional[np.ndarray]] = []   # (1, n_comp)
    raw_conc_unfolded: List[Optional[np.ndarray]] = []   # (n_comp, n_rows_unfolded)
    raw_conc_row_axis: List[Optional[np.ndarray]] = []   # (n_rows_unfolded,)
    raw_sample_boundaries: List[Optional[np.ndarray]] = []  # (n_boundaries,)
    raw_cal_val_boundary: List[float] = []
    raw_y_val_pred: List[np.ndarray] = []
    raw_y_cal_pred: List[Optional[np.ndarray]] = []  # (n_cal, n_y)
    raw_fom: List[Optional[Dict[str, Any]]] = []      # per-model figures of merit
    first_auto_mapping: Optional[bool] = None

    nan_row = np.full((max(n_y, 1),), np.nan)

    for i in range(n_val):
        try:
            r = _single_fit(
                X_cal=X_cal_arr,
                Y_cal=Y_cal,
                X_val=X_val_arr[i : i + 1],
                Y_val=None,
                cal_s_mask=cal_s_mask_arr,
                val_s_mask=None if val_s_mask_arr is None else val_s_mask_arr[i : i + 1],
                **common_kwargs,
            )
        except Exception as exc:
            emit_execution_warning(
                code="mcr_als_sbs_sample_failed",
                text=f"SBS MCR-ALS model for validation sample {i + 1} failed: {exc}",
            )
            raw_st.append(None)
            raw_c_scores_cal.append(None)
            raw_val_c_scores.append(None)
            raw_conc_unfolded.append(None)
            raw_conc_row_axis.append(None)
            raw_sample_boundaries.append(None)
            raw_cal_val_boundary.append(np.nan)
            raw_y_val_pred.append(nan_row[:n_y] if n_y > 0 else np.array([]))
            raw_y_cal_pred.append(None)
            raw_fom.append(None)
            continue

        st = r.get("s_profiles")  # (n_comp, n_vars)
        raw_st.append(np.asarray(st, dtype=float) if st is not None else None)

        c_cal = r.get("c_scores")   # (n_cal, n_comp)
        raw_c_scores_cal.append(np.asarray(c_cal, dtype=float) if c_cal is not None else None)

        c_val = r.get("val_c_scores")  # (1, n_comp) or (n_comp,)
        if c_val is not None:
            c_val_arr = np.asarray(c_val, dtype=float)
            raw_val_c_scores.append(c_val_arr if c_val_arr.ndim == 2 else c_val_arr.reshape(1, -1))
        else:
            raw_val_c_scores.append(None)

        conc_unfolded = r.get("concentrations_unfolded")  # (n_comp, n_rows_unfolded)
        raw_conc_unfolded.append(np.asarray(conc_unfolded, dtype=float) if conc_unfolded is not None else None)

        conc_row_axis = r.get("concentration_row_axis")  # (n_rows_unfolded,)
        raw_conc_row_axis.append(np.asarray(conc_row_axis, dtype=float).reshape(-1) if conc_row_axis is not None else None)

        sample_boundaries = r.get("sample_boundary_positions")
        raw_sample_boundaries.append(
            np.asarray(sample_boundaries, dtype=float).reshape(-1)
            if sample_boundaries is not None else None
        )
        raw_cal_val_boundary.append(_safe_float(r.get("cal_val_boundary_position"), default=np.nan))

        yvp = r.get("y_val_pred")  # (1, n_y) or None
        if yvp is not None and np.asarray(yvp).size > 0:
            raw_y_val_pred.append(np.asarray(yvp, dtype=float).reshape(-1)[:n_y] if n_y > 0 else np.array([]))
        else:
            raw_y_val_pred.append(nan_row[:n_y] if n_y > 0 else np.array([]))

        # y_cal_pred from this SBS model: predictions on the calibration set
        ycp = r.get("y_cal_pred")  # (n_cal, n_y) or (n_cal,) or None
        if ycp is not None:
            ycp_arr = np.asarray(ycp, dtype=float)
            raw_y_cal_pred.append(ycp_arr if ycp_arr.ndim == 2 else ycp_arr.reshape(-1, max(n_y, 1)))
        else:
            raw_y_cal_pred.append(None)
        raw_fom.append(r.get("metrics", {}).get("calibration", {}) or {})
        if first_auto_mapping is None:
            first_auto_mapping = bool(r.get("auto_mapping_used", False))

    # ------------------------------------------------------------------ #
    # Factor alignment using S profiles (cosine on ST rows)               #
    # ------------------------------------------------------------------ #
    ref_st_arr: Optional[np.ndarray] = None
    for st in raw_st:
        if st is not None:
            ref_st_arr = np.asarray(st, dtype=float)
            break

    perms: List[np.ndarray] = []
    signs_list: List[np.ndarray] = []
    for i in range(n_val):
        if raw_st[i] is None or ref_st_arr is None:
            perms.append(np.arange(n_comp, dtype=int))
            signs_list.append(np.ones(n_comp, dtype=float))
            continue
        p, s = _align_st_to_reference_mcr(ref_st_arr, np.asarray(raw_st[i], dtype=float))
        perms.append(p)
        signs_list.append(s)

    # ------------------------------------------------------------------ #
    # Build aligned arrays                                                 #
    # ------------------------------------------------------------------ #
    n_cal_samples = (
        int(raw_c_scores_cal[next((j for j, x in enumerate(raw_c_scores_cal) if x is not None), 0)].shape[0])
        if any(x is not None for x in raw_c_scores_cal) else 0
    )
    n_vars = (
        int(raw_st[next((j for j, x in enumerate(raw_st) if x is not None), 0)].shape[1])
        if any(x is not None for x in raw_st) else 0
    )

    sbs_c_scores_list: List[np.ndarray] = []
    sbs_val_c_scores_list: List[np.ndarray] = []
    sbs_st_list: List[Optional[np.ndarray]] = []
    sbs_conc_unfolded_list: List[Optional[np.ndarray]] = []

    for i in range(n_val):
        p, s = perms[i], signs_list[i]

        # Cal C scores: (n_cal, n_comp) reordered
        if raw_c_scores_cal[i] is not None:
            cc = np.asarray(raw_c_scores_cal[i], dtype=float)[:, p]
            # Signs go into C (S matrix stays positive-semidefinite if constrained)
            cc = cc * s[np.newaxis, :]
        else:
            cc = np.full((n_cal_samples, n_comp), np.nan)
        sbs_c_scores_list.append(cc)

        # Val C score: (1, n_comp)
        if raw_val_c_scores[i] is not None:
            vc = np.asarray(raw_val_c_scores[i], dtype=float).reshape(1, n_comp)[:, p] * s[np.newaxis, :]
        else:
            vc = np.full((1, n_comp), np.nan)
        sbs_val_c_scores_list.append(vc)

        # S profiles: (n_comp, n_vars) — sign correction on S rows, consistent
        if raw_st[i] is not None:
            st_r = np.asarray(raw_st[i], dtype=float)[p, :] * s[:, np.newaxis]
        else:
            st_r = None
        sbs_st_list.append(st_r)

        if raw_conc_unfolded[i] is not None:
            cu = np.asarray(raw_conc_unfolded[i], dtype=float)
            if cu.ndim == 2 and cu.shape[0] == n_comp:
                cu = cu[p, :] * s[:, np.newaxis]
                sbs_conc_unfolded_list.append(cu)
            else:
                sbs_conc_unfolded_list.append(None)
        else:
            sbs_conc_unfolded_list.append(None)

    sbs_c_scores = np.stack(sbs_c_scores_list, axis=0)        # (n_val, n_cal, n_comp)
    sbs_val_c_scores = np.concatenate(sbs_val_c_scores_list, axis=0)  # (n_val, 1, n_comp) after concat
    # concatenate gives (n_val, n_comp) — reshape to (n_val, 1, n_comp)
    sbs_val_c_scores = sbs_val_c_scores.reshape(n_val, 1, n_comp)

    sbs_c_scores_heatmap = sbs_c_scores.transpose(0, 2, 1)    # (n_val, n_comp, n_cal)
    sbs_c_scores_full = np.concatenate([sbs_c_scores, sbs_val_c_scores], axis=1)
    sbs_c_scores_heatmap_full = sbs_c_scores_full.transpose(0, 2, 1)  # (n_val, n_comp, n_cal+1)

    n_rows_unfolded = max((int(arr.shape[1]) for arr in sbs_conc_unfolded_list if isinstance(arr, np.ndarray) and arr.ndim == 2), default=0)
    sbs_concentrations_unfolded: Optional[np.ndarray] = None
    if n_rows_unfolded > 0:
        sbs_concentrations_unfolded = np.full((n_val, n_comp, n_rows_unfolded), np.nan, dtype=float)
        for i, cu in enumerate(sbs_conc_unfolded_list):
            if isinstance(cu, np.ndarray) and cu.ndim == 2 and cu.shape[0] == n_comp:
                n_cols = min(int(cu.shape[1]), n_rows_unfolded)
                sbs_concentrations_unfolded[i, :, :n_cols] = cu[:, :n_cols]

    # Keep axes/boundaries slicable by model (dim 0) for linked SBS navigation.
    if n_rows_unfolded > 0:
        sbs_concentration_row_axis = np.tile(np.arange(1, n_rows_unfolded + 1, dtype=float), (n_val, 1))
    else:
        sbs_concentration_row_axis = None

    max_boundary_count = max(
        (int(arr.size) for arr in raw_sample_boundaries if isinstance(arr, np.ndarray)),
        default=0,
    )
    if max_boundary_count > 0:
        sbs_sample_boundary_positions = np.full((n_val, max_boundary_count), np.nan, dtype=float)
        for i, arr in enumerate(raw_sample_boundaries):
            if isinstance(arr, np.ndarray) and arr.size > 0:
                n_copy = min(int(arr.size), max_boundary_count)
                sbs_sample_boundary_positions[i, :n_copy] = arr[:n_copy]
    else:
        sbs_sample_boundary_positions = None

    if raw_cal_val_boundary:
        sbs_cal_val_boundary_position = np.asarray(raw_cal_val_boundary, dtype=float)
    else:
        sbs_cal_val_boundary_position = None

    st_valid = [x for x in sbs_st_list if x is not None]
    if st_valid and n_vars > 0:
        sbs_st_arr = np.full((n_val, n_comp, n_vars), np.nan, dtype=float)
        for i, st in enumerate(sbs_st_list):
            if st is not None:
                sbs_st_arr[i] = st
    else:
        sbs_st_arr = None

    # ------------------------------------------------------------------ #
    # Similarity matrix: (n_comp, n_val, n_val) from S profiles           #
    # ------------------------------------------------------------------ #
    sbs_similarity = np.full((n_comp, n_val, n_val), np.nan, dtype=float)
    if sbs_st_arr is not None:
        for f in range(n_comp):
            vecs = sbs_st_arr[:, f, :]      # (n_val, n_vars)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            units = vecs / norms
            sbs_similarity[f] = units @ units.T

    # ------------------------------------------------------------------ #
    # Assemble y_val_pred                                                  #
    # ------------------------------------------------------------------ #
    if n_y > 0 and raw_y_val_pred:
        y_val_pred_sbs = np.vstack([row.reshape(1, -1) for row in raw_y_val_pred])
    else:
        y_val_pred_sbs = None

    # ------------------------------------------------------------------ #
    # Assemble sbs_y_cal_pred / sbs_y_cal_true / sbs_y_cal_error          #
    # All shaped (n_val, n_cal, n_y) for consistent table/graph slicing.  #
    # ------------------------------------------------------------------ #
    if n_y > 0:
        sbs_y_cal_pred_arr = np.full((n_val, n_cal_samples, n_y), np.nan, dtype=float)
        for i, ycp in enumerate(raw_y_cal_pred):
            if ycp is not None:
                arr = np.asarray(ycp, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                rows = min(arr.shape[0], n_cal_samples)
                cols = min(arr.shape[1], n_y)
                sbs_y_cal_pred_arr[i, :rows, :cols] = arr[:rows, :cols]
        if Y_cal_2d is not None:
            y_cal_true_arr = np.asarray(Y_cal_2d, dtype=float)  # (n_cal, n_y)
            sbs_y_cal_true_arr = np.broadcast_to(
                y_cal_true_arr[np.newaxis, :, :], (n_val, n_cal_samples, n_y)
            ).copy()
            sbs_y_cal_error_arr = sbs_y_cal_true_arr - sbs_y_cal_pred_arr
        else:
            sbs_y_cal_true_arr = None
            sbs_y_cal_error_arr = None
    else:
        sbs_y_cal_pred_arr = None
        sbs_y_cal_true_arr = None
        sbs_y_cal_error_arr = None

    component_labels = [f"C{c + 1}" for c in range(n_comp)]
    val_sample_axis = np.arange(1, n_val + 1, dtype=float)
    c_sample_axis = np.arange(1, n_cal_samples + 1, dtype=float)
    c_sample_axis_full = np.arange(1, n_cal_samples + 2, dtype=float)
    c_component_axis = np.arange(1, n_comp + 1, dtype=float)

    # Align val true/pred matrices to (n_val, n_y) before expanding to (n_val, 1, n_y).
    # This avoids reshape errors when Y_val has fewer columns than Y_cal.
    sbs_y_val_true_2d: Optional[np.ndarray] = None
    sbs_y_val_pred_2d: Optional[np.ndarray] = None
    if n_y > 0:
        if Y_val_2d is not None:
            sbs_y_val_true_2d = np.full((n_val, n_y), np.nan, dtype=float)
            yv_true = np.asarray(Y_val_2d, dtype=float)
            if yv_true.ndim == 1:
                yv_true = yv_true.reshape(-1, 1)
            rows = min(yv_true.shape[0], n_val)
            cols = min(yv_true.shape[1], n_y)
            sbs_y_val_true_2d[:rows, :cols] = yv_true[:rows, :cols]
        if y_val_pred_sbs is not None:
            sbs_y_val_pred_2d = np.full((n_val, n_y), np.nan, dtype=float)
            yv_pred = np.asarray(y_val_pred_sbs, dtype=float)
            if yv_pred.ndim == 1:
                yv_pred = yv_pred.reshape(-1, 1)
            rows = min(yv_pred.shape[0], n_val)
            cols = min(yv_pred.shape[1], n_y)
            sbs_y_val_pred_2d[:rows, :cols] = yv_pred[:rows, :cols]

    sbs_y_val_true_3d = (
        sbs_y_val_true_2d.reshape(n_val, 1, n_y)
        if sbs_y_val_true_2d is not None else None
    )
    sbs_y_val_pred_3d = (
        sbs_y_val_pred_2d.reshape(n_val, 1, n_y)
        if sbs_y_val_pred_2d is not None else None
    )
    sbs_y_val_error_2d = (
        sbs_y_val_true_2d - sbs_y_val_pred_2d
        if (sbs_y_val_true_2d is not None and sbs_y_val_pred_2d is not None)
        else None
    )

    return {
        "sbs_c_scores": sbs_c_scores,                   # (n_val, n_cal, n_comp)
        "sbs_val_c_scores": sbs_val_c_scores,            # (n_val, 1, n_comp)
        "sbs_c_scores_heatmap": sbs_c_scores_heatmap,   # (n_val, n_comp, n_cal)
        "sbs_c_scores_heatmap_full": sbs_c_scores_heatmap_full,  # (n_val, n_comp, n_cal+1)
        "sbs_concentrations_unfolded": sbs_concentrations_unfolded,  # (n_val, n_comp, n_rows_unfolded)
        "sbs_concentration_row_axis": sbs_concentration_row_axis,  # (n_val, n_rows_unfolded)
        "sbs_sample_boundary_positions": sbs_sample_boundary_positions,  # (n_val, n_boundaries)
        "sbs_cal_val_boundary_position": sbs_cal_val_boundary_position,  # (n_val,)
        "sbs_s_profiles": sbs_st_arr,                   # (n_val, n_comp, n_vars) or None
        "sbs_similarity_matrix": sbs_similarity,         # (n_comp, n_val, n_val)
        # Tiled so the leading navigation dimension can be sliced without collapsing 1D vectors.
        "sbs_val_sample_axis": np.tile(val_sample_axis, (n_comp, 1)),    # (n_comp, n_val)
        "sbs_c_sample_axis": np.tile(c_sample_axis, (n_val, 1)),         # (n_val, n_cal)
        "sbs_c_sample_axis_full": np.tile(c_sample_axis_full, (n_val, 1)),  # (n_val, n_cal+1)
        "sbs_c_component_axis": np.tile(c_component_axis, (n_val, 1)),   # (n_val, n_comp)
        "sbs_component_labels": component_labels,
        "sbs_y_cal_pred": sbs_y_cal_pred_arr,          # (n_val, n_cal, n_y)
        "sbs_y_cal_true": sbs_y_cal_true_arr,           # (n_val, n_cal, n_y)
        "sbs_y_cal_error": sbs_y_cal_error_arr,         # (n_val, n_cal, n_y)
        "y_val_pred": y_val_pred_sbs,
        # (n_val, 1, n_y) — val-sample dim 0 + singleton dim 1 align with sbs_val_c_scores
        "sbs_y_val_true": sbs_y_val_true_3d,
        "sbs_y_val_pred": sbs_y_val_pred_3d,
        "y_val_error": sbs_y_val_error_2d,
        "sbs_fom_metrics": _compute_mcr_sbs_fom(raw_fom),
        "auto_mapping_used": first_auto_mapping if first_auto_mapping is not None else False,
    }


def _compute_mcr_sbs_fom(raw_fom: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    _fom_ssr: List[float] = []
    _fom_sfit: List[float] = []
    _fom_ev: List[float] = []
    _fom_iter: List[int] = []
    _fom_impl: Optional[str] = None
    for _fom in raw_fom:
        if not isinstance(_fom, dict):
            continue
        _v = _safe_float(_fom.get("SSR"), default=np.nan)
        if np.isfinite(_v):
            _fom_ssr.append(float(_v))
        _v = _safe_float(_fom.get("sfit"), default=np.nan)
        if np.isfinite(_v):
            _fom_sfit.append(float(_v))
        _v = _safe_float(_fom.get("explained_variance"), default=np.nan)
        if np.isfinite(_v):
            _fom_ev.append(float(_v))
        _vi = int(_safe_float(_fom.get("n_iter"), default=0))
        if _vi > 0:
            _fom_iter.append(_vi)
        if _fom_impl is None:
            _fom_impl = str(_fom.get("implementation_used", "") or "")
    return {
        "implementation_used": _fom_impl or "MCR-ALS (pyMCR)",
        "SSR": (_min_max(_fom_ssr) if _fom_ssr else None),
        "sfit": (_min_max(_fom_sfit) if _fom_sfit else None),
        "explained_variance": (_min_max(_fom_ev) if _fom_ev else None),
        "n_iter": ((min(_fom_iter), max(_fom_iter)) if _fom_iter else None),
    }


def mcr_als_analysis(
    X_cal: Optional[np.ndarray] = None,
    Y_cal: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    cal_s_mask: Optional[np.ndarray] = None,
    val_s_mask: Optional[np.ndarray] = None,
    n_components: int = 2,
    max_iter: int = 250,
    tol: float = 1e-8,
    random_state: Optional[Any] = 42,
    c_regr: str = "OLS",
    st_regr: str = "OLS",
    c_nonneg: Any = True,
    st_nonneg: Any = True,
    c_norm: Any = False,
    constraint_non_negative: Any = None,
    constraint_non_negative_mode_c: Any = "NNLS",
    constraint_non_negative_mode_st: Any = "NNLS",
    constraint_cumsum_non_negative: Any = None,
    constraint_zero_end_points: Any = None,
    constraint_zero_cumsum_end_points: Any = None,
    constraint_normalize: Any = None,
    constraint_cut_below: Any = None,
    constraint_cut_above: Any = None,
    constraint_compress_below: Any = None,
    constraint_compress_above: Any = None,
    constraint_replace_zeros: Any = None,
    constraint_planarize: Any = None,
    constraint_zero_end_points_span: Any = 1,
    constraint_zero_cumsum_nodes: Any = "",
    constraint_cut_below_value: Any = 0.0,
    constraint_cut_above_value: Any = 1.0,
    constraint_compress_below_value: Any = 0.0,
    constraint_compress_above_value: Any = 1.0,
    constraint_replace_zeros_feature: Any = 1,
    constraint_replace_zeros_fval: Any = 1.0,
    constraint_planarize_targets_c: Any = "",
    constraint_planarize_targets_st: Any = "",
    constraint_planarize_shape_c: Any = "",
    constraint_planarize_shape_st: Any = "",
    constraint_planarize_use_vals_above: Any = "",
    constraint_planarize_use_vals_below: Any = "",
    constraint_planarize_lims_to_plane: Any = True,
    constraint_planarize_scaler: Any = "",
    constraint_planarize_recalc_scaler: Any = False,
    sweep_mode: bool = False,
    component_range: str = "",
    mcr_component_y_mapping: Any = "",
    cv_config: Optional[Any] = None,
    y_labels: Optional[Any] = None,
    nway_flag: Optional[Any] = None,
    aug_direction: Optional[Any] = None,
    c_aggregation_method: Optional[Any] = None,
    validation_processing: str = "batch",
    **kwargs: Any,
) -> Dict[str, Any]:
    """MCR-ALS with optional component sweep, calibration, and CV support."""

    if X_cal is None and "X_cal_train" in kwargs:
        X_cal = kwargs["X_cal_train"]
    if Y_cal is None and "Y_cal_train" in kwargs:
        Y_cal = kwargs["Y_cal_train"]
    if X_cal is None:
        raise ValueError("X_cal is required for mcr_als_analysis")

    cal_s_mask_arr = None if cal_s_mask is None else np.asarray(cal_s_mask, dtype=bool)
    val_s_mask_arr = None if val_s_mask is None else np.asarray(val_s_mask, dtype=bool)

    component_y_mapping = mcr_component_y_mapping

    y_labels_resolved = _normalize_y_labels(y_labels if y_labels is not None else kwargs.get("y_labels"))
    resolved_nway_flag = _safe_int(nway_flag if nway_flag is not None else kwargs.get("nway_flag"), default=1)
    if resolved_nway_flag < 1:
        resolved_nway_flag = 1

    resolved_c_aggregation_method = _normalize_aggregation_method(
        c_aggregation_method if c_aggregation_method is not None else kwargs.get("c_aggregation_method")
    )

    resolved_aug_direction: Optional[int] = None
    aug_raw = aug_direction if aug_direction is not None else kwargs.get("aug_direction")
    if aug_raw is not None and str(aug_raw).strip() != "":
        parsed_aug = _safe_int(aug_raw, default=0)
        if parsed_aug > 0 and resolved_nway_flag >= 2:
            if parsed_aug > resolved_nway_flag:
                raise ValueError(f"aug_direction must be between 1 and nway_flag ({resolved_nway_flag}).")
            resolved_aug_direction = int(parsed_aug)
    if resolved_aug_direction is None and resolved_nway_flag >= 2:
        resolved_aug_direction = 2
    seed_value = None
    if random_state is not None and str(random_state).strip() != "":
        try:
            seed_value = int(float(random_state))
        except Exception:
            seed_value = None

    use_sbs = (
        str(validation_processing).strip().lower() == "sample_by_sample"
        and X_val is not None
        and np.asarray(X_val).shape[0] > 0
    )
    if use_sbs and bool(kwargs.get('__passforward_enabled__', False)):
        emit_execution_warning(
            code="mcr_als_sbs_passforward_fallback_batch",
            text=(
                "Sample-by-Sample validation is incompatible with passforward output mode. "
                "Falling back to Batch validation processing."
            ),
        )
        use_sbs = False
    effective_validation_processing = "sample_by_sample" if use_sbs else "batch"

    def _normalize_sweep_mode(value: Any) -> str:
        if isinstance(value, bool):
            return "on" if value else "off"
        text = "" if value is None else str(value).strip().lower()
        if text in {"", "0", "false", "no", "off", "none"}:
            return "off"
        if text in {"stats", "stats only", "stats_only", "statistics", "statistics only"}:
            return "stats_only"
        if text in {"1", "true", "yes", "on"}:
            return "on"
        return "on"

    sweep_mode_normalized = _normalize_sweep_mode(sweep_mode)
    run_sweep_stats = sweep_mode_normalized in {"stats_only", "on"}
    run_sweep_models = sweep_mode_normalized == "on"

    if use_sbs and run_sweep_stats:
        emit_execution_warning(
            code="mcr_als_sbs_sweep_first_layer",
            text=(
                "Sweep mode in Sample-by-Sample validation uses only the first SBS model "
                "(first validation sample) as sweep reference."
            ),
        )

    # Sweep fitting data:
    # - Batch mode: include all validation samples so sweep pages expose full val outputs.
    # - SBS mode: keep first validation sample only as documented sweep reference.
    sweep_x_val = None
    sweep_y_val = None
    if X_val is not None:
        xval_arr = np.asarray(X_val, dtype=float)
        sweep_x_val = xval_arr[0:1] if use_sbs else xval_arr
    if Y_val is not None:
        yv_arr = np.asarray(Y_val, dtype=float)
        if yv_arr.ndim == 1:
            sweep_y_val = yv_arr[0:1] if use_sbs else yv_arr.reshape(-1, 1)
        elif yv_arr.ndim >= 2 and yv_arr.shape[0] > 0:
            sweep_y_val = yv_arr[0:1, ...] if use_sbs else yv_arr

    ranks: List[int] = [int(max(1, _safe_int(n_components, default=2)))]
    sweep_results: List[Dict[str, Any]] = []
    sweep_model_results: List[Dict[str, Any]] = []
    sweep_ranks: List[int] = []
    if run_sweep_stats:
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
                    X_val=sweep_x_val,
                    Y_val=sweep_y_val,
                    cal_s_mask=cal_s_mask_arr,
                    val_s_mask=None if val_s_mask_arr is None or sweep_x_val is None else val_s_mask_arr[0:1],
                    n_components=int(rk),
                    max_iter=max_iter,
                    tol=tol,
                    random_state=seed_value,
                    c_regr=c_regr,
                    st_regr=st_regr,
                    c_nonneg=_safe_bool(c_nonneg, default=True),
                    st_nonneg=_safe_bool(st_nonneg, default=True),
                    c_norm=_safe_bool(c_norm, default=False),
                    constraint_non_negative=constraint_non_negative,
                    constraint_non_negative_mode_c=constraint_non_negative_mode_c,
                    constraint_non_negative_mode_st=constraint_non_negative_mode_st,
                    constraint_cumsum_non_negative=constraint_cumsum_non_negative,
                    constraint_zero_end_points=constraint_zero_end_points,
                    constraint_zero_cumsum_end_points=constraint_zero_cumsum_end_points,
                    constraint_normalize=constraint_normalize,
                    constraint_cut_below=constraint_cut_below,
                    constraint_cut_above=constraint_cut_above,
                    constraint_compress_below=constraint_compress_below,
                    constraint_compress_above=constraint_compress_above,
                    constraint_replace_zeros=constraint_replace_zeros,
                    constraint_planarize=constraint_planarize,
                    constraint_zero_end_points_span=constraint_zero_end_points_span,
                    constraint_zero_cumsum_nodes=constraint_zero_cumsum_nodes,
                    constraint_cut_below_value=constraint_cut_below_value,
                    constraint_cut_above_value=constraint_cut_above_value,
                    constraint_compress_below_value=constraint_compress_below_value,
                    constraint_compress_above_value=constraint_compress_above_value,
                    constraint_replace_zeros_feature=constraint_replace_zeros_feature,
                    constraint_replace_zeros_fval=constraint_replace_zeros_fval,
                    constraint_planarize_targets_c=constraint_planarize_targets_c,
                    constraint_planarize_targets_st=constraint_planarize_targets_st,
                    constraint_planarize_shape_c=constraint_planarize_shape_c,
                    constraint_planarize_shape_st=constraint_planarize_shape_st,
                    constraint_planarize_use_vals_above=constraint_planarize_use_vals_above,
                    constraint_planarize_use_vals_below=constraint_planarize_use_vals_below,
                    constraint_planarize_lims_to_plane=constraint_planarize_lims_to_plane,
                    constraint_planarize_scaler=constraint_planarize_scaler,
                    constraint_planarize_recalc_scaler=constraint_planarize_recalc_scaler,
                    component_y_mapping=component_y_mapping,
                    nway_flag=resolved_nway_flag,
                    aug_direction=resolved_aug_direction,
                    c_aggregation_method=resolved_c_aggregation_method,
                    axis_n_info=kwargs.get("axis_n_info"),
                    aug_row_counts_cal=kwargs.get("aug_row_counts_cal"),
                    aug_row_counts_val=None,
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
                if run_sweep_models and isinstance(fit_rk, dict):
                    sweep_model_results.append(fit_rk)
                    sweep_ranks.append(int(rk))
            except Exception as exc:
                emit_execution_warning(
                    code="mcr_als_sweep_rank_failed",
                    text=f"MCR-ALS sweep rank {rk} failed and was skipped.",
                    details={"rank": int(rk), "error": str(exc)},
                )

    selected_rank = int(max(1, _safe_int(n_components, default=2)))
    if not use_sbs:
        result = _single_fit(
            X_cal=np.asarray(X_cal, dtype=float),
            Y_cal=Y_cal,
            X_val=X_val,
            Y_val=Y_val,
            cal_s_mask=cal_s_mask_arr,
            val_s_mask=val_s_mask_arr,
            n_components=selected_rank,
            max_iter=max_iter,
            tol=tol,
            random_state=seed_value,
            c_regr=c_regr,
            st_regr=st_regr,
            c_nonneg=_safe_bool(c_nonneg, default=True),
            st_nonneg=_safe_bool(st_nonneg, default=True),
            c_norm=_safe_bool(c_norm, default=False),
            constraint_non_negative=constraint_non_negative,
            constraint_non_negative_mode_c=constraint_non_negative_mode_c,
            constraint_non_negative_mode_st=constraint_non_negative_mode_st,
            constraint_cumsum_non_negative=constraint_cumsum_non_negative,
            constraint_zero_end_points=constraint_zero_end_points,
            constraint_zero_cumsum_end_points=constraint_zero_cumsum_end_points,
            constraint_normalize=constraint_normalize,
            constraint_cut_below=constraint_cut_below,
            constraint_cut_above=constraint_cut_above,
            constraint_compress_below=constraint_compress_below,
            constraint_compress_above=constraint_compress_above,
            constraint_replace_zeros=constraint_replace_zeros,
            constraint_planarize=constraint_planarize,
            constraint_zero_end_points_span=constraint_zero_end_points_span,
            constraint_zero_cumsum_nodes=constraint_zero_cumsum_nodes,
            constraint_cut_below_value=constraint_cut_below_value,
            constraint_cut_above_value=constraint_cut_above_value,
            constraint_compress_below_value=constraint_compress_below_value,
            constraint_compress_above_value=constraint_compress_above_value,
            constraint_replace_zeros_feature=constraint_replace_zeros_feature,
            constraint_replace_zeros_fval=constraint_replace_zeros_fval,
            constraint_planarize_targets_c=constraint_planarize_targets_c,
            constraint_planarize_targets_st=constraint_planarize_targets_st,
            constraint_planarize_shape_c=constraint_planarize_shape_c,
            constraint_planarize_shape_st=constraint_planarize_shape_st,
            constraint_planarize_use_vals_above=constraint_planarize_use_vals_above,
            constraint_planarize_use_vals_below=constraint_planarize_use_vals_below,
            constraint_planarize_lims_to_plane=constraint_planarize_lims_to_plane,
            constraint_planarize_scaler=constraint_planarize_scaler,
            constraint_planarize_recalc_scaler=constraint_planarize_recalc_scaler,
            component_y_mapping=component_y_mapping,
            nway_flag=resolved_nway_flag,
            aug_direction=resolved_aug_direction,
            c_aggregation_method=resolved_c_aggregation_method,
            axis_n_info=kwargs.get("axis_n_info"),
            aug_row_counts_cal=kwargs.get("aug_row_counts_cal"),
            aug_row_counts_val=kwargs.get("aug_row_counts_val"),
        )
    else:
        result = {
            "c_scores": None,
            "val_c_scores": None,
            "c_scores_heatmap": None,
            "c_scores_heatmap_full": None,
            "c_sample_axis": None,
            "c_sample_axis_full": None,
            "c_component_axis": None,
            "component_labels": None,
            "sweep_components": None,
            "sweep_sfit": None,
            "sweep_n_iter": None,
            "sweep_explained_variance": None,
            "s_profiles": None,
            "concentrations": None,
            "concentrations_unfolded": None,
            "concentration_row_axis": None,
            "s_axis_vector": None,
            "sample_boundary_positions": None,
            "cal_val_boundary_position": None,
            "reconstructed": None,
            "residual": None,
            "metrics": None,
            "calibration_models": [],
            "component_y_mapping": {},
            "auto_mapping_used": False,
            "mcr_als_report": "",
            "selected_n_components": int(selected_rank),
            "cv_results": None,
            "y_cv_pred": None,
            "y_cal_pred": None,
            "y_val_pred": None,
            "y_cal_error": None,
            "y_val_error": None,
            "y_cv_error": None,
            "y_cal_true": _as_2d_y(Y_cal),
            "y_val_true": _as_2d_y(Y_val),
        }

    if (not use_sbs) and Y_cal is not None and not result.get("calibration_models"):
        emit_execution_warning(
            code="mcr_als_no_calibration_pairs",
            text=(
                "Y calibration data was provided but no valid component-to-Y calibration pairs could be established. "
                "Ensure the Y sample count matches the number of calibration samples."
            ),
        )

    # Resolve S profile x-axis label from dim_labels for the non-augmented direction
    dim_labels_resolved: List[str] = []
    raw_dim_labels = kwargs.get("dim_labels")
    if isinstance(raw_dim_labels, (list, tuple, np.ndarray)):
        dim_labels_resolved = [str(item).strip() for item in raw_dim_labels]

    # Patch Y column references in the batch report using resolved y_labels
    if not use_sbs and y_labels_resolved and result.get("mcr_als_report"):
        import re as _re
        _raw_report = result["mcr_als_report"]
        def _replace_y_ref(m: "_re.Match") -> str:
            y_num = m.group(1)
            y_idx = int(y_num) - 1
            if 0 <= y_idx < len(y_labels_resolved):
                label = str(y_labels_resolved[y_idx]).strip()
                if label:
                    return f"{label} (Y{y_num})"
            return m.group(0)
        _patched = _re.sub(r"Y(\d+)(?= \|)", _replace_y_ref, _raw_report)
        result["mcr_als_report"] = _patched

    s_axis_label: str = "Variable Index"
    if resolved_nway_flag <= 1:
        _s_dim_label = 1
    else:
        _s_dim_label = resolved_aug_direction if resolved_aug_direction is not None else 2
        if _s_dim_label < 1 or _s_dim_label > resolved_nway_flag:
            _s_dim_label = 2

    # dim_labels_resolved may start with "Samples" (PARAFAC pass-through) or be raw
    # user labels (no sample entry). Detect by checking if index 0 is "Samples".
    _offset = 1 if (dim_labels_resolved and dim_labels_resolved[0].lower() == "samples") else 0
    _label_idx = int(_s_dim_label) - 1 + _offset
    if 0 <= _label_idx < len(dim_labels_resolved) and dim_labels_resolved[_label_idx]:
        s_axis_label = dim_labels_resolved[_label_idx]

    result["s_axis_label"] = s_axis_label
    result["c_aggregation_label"] = _c_aggregation_label(resolved_c_aggregation_method)

    if run_sweep_stats and sweep_results:
        result["sweep_results"] = sweep_results
        result["sweep_components"] = np.asarray([_safe_float(item.get("n_components")) for item in sweep_results], dtype=float)
        result["sweep_sfit"] = np.asarray([_safe_float(item.get("sfit")) for item in sweep_results], dtype=float)
        result["sweep_n_iter"] = np.asarray([_safe_float(item.get("n_iter")) for item in sweep_results], dtype=float)
        result["sweep_explained_variance"] = np.asarray(
            [_safe_float(item.get("explained_variance")) for item in sweep_results],
            dtype=float,
        )

        report_text = str(result.get("mcr_als_report", "") or "")
        sweep_lines: List[str] = [report_text, "", "Sweep results:"]
        for item in sweep_results:
            sweep_lines.append(
                f"- F={item.get('n_components')} | sfit={_safe_float(item.get('sfit')):.6g} | "
                f"EV={_safe_float(item.get('explained_variance')):.4f}% | iter={int(_safe_int(item.get('n_iter'), default=0))}"
            )
        result["mcr_als_report"] = "\n".join(sweep_lines)
    else:
        result["sweep_results"] = None

    if run_sweep_models and sweep_model_results:
        result.update(
            _gather_sweep_outputs_mcr(
                sweep_model_results=sweep_model_results,
                sweep_ranks=sweep_ranks,
                y_labels=y_labels_resolved,
                y_cal_true=result.get("y_cal_true"),
                y_val_true=result.get("y_val_true"),
            )
        )

    if not use_sbs:
        pair_outputs = _build_pair_outputs(
            component_y_mapping=result.get("component_y_mapping", {}),
            y_labels=y_labels_resolved,
            y_cal_true=result.get("y_cal_true"),
            y_val_true=result.get("y_val_true"),
            c_scores_cal=result.get("c_scores"),
            c_scores_val=result.get("val_c_scores"),
            calibration_models=result.get("calibration_models"),
        )
        result.update(pair_outputs)
        # Per-pairing C score axis labels: aggregation label + component number
        _agg_lbl = result.get("c_aggregation_label", "C Score")
        _pair_comps_list = (
            result["mcr_pair_components"].tolist()
            if isinstance(result.get("mcr_pair_components"), np.ndarray)
            else list(result.get("mcr_pair_components") or [])
        )
        result["c_score_pair_labels"] = [f"{_agg_lbl} (C{comp})" for comp in _pair_comps_list]
        # Override unified prediction matrices using best-correlation pairing selection
        pair_y_cols_outer: List[int] = (
            result["mcr_pair_y_columns"].tolist()
            if isinstance(result.get("mcr_pair_y_columns"), np.ndarray)
            else list(result.get("mcr_pair_y_columns") or [])
        )
        result.update(
            _build_unified_prediction_matrices(
                y_cal_true=result.get("y_cal_true"),
                y_val_true=result.get("y_val_true"),
                y_cal_pred_pairs=result.get("y_cal_pred_pairs"),
                y_val_pred_pairs=result.get("y_val_pred_pairs"),
                pair_y_columns=pair_y_cols_outer,
            )
        )

    # ------------------------------------------------------------------ #
    # Sample-by-sample validation                                          #
    # ------------------------------------------------------------------ #
    result["validation_processing"] = effective_validation_processing

    for _sbs_key in (
        "sbs_c_scores", "sbs_val_c_scores", "sbs_c_scores_heatmap", "sbs_c_scores_heatmap_full",
        "sbs_concentrations_unfolded", "sbs_concentration_row_axis", "sbs_sample_boundary_positions", "sbs_cal_val_boundary_position",
        "sbs_s_profiles", "sbs_similarity_matrix", "sbs_val_sample_axis",
        "sbs_c_sample_axis", "sbs_c_sample_axis_full", "sbs_c_component_axis", "sbs_component_labels",
        "sbs_y_cal_pred", "sbs_y_cal_true", "sbs_y_cal_error",
        "sbs_y_val_true", "sbs_y_val_pred",
        "sbs_c_scores_cal_pairs", "sbs_c_scores_val_pairs",
        "sbs_y_cal_pred_pairs", "sbs_y_cal_true_pairs", "sbs_y_cal_error_pairs",
        "sbs_y_val_true_pairs", "sbs_y_val_pred_pairs", "sbs_y_val_effective_pairs", "sbs_y_val_error_pairs",
        "sbs_cal_regression_line_x_pairs", "sbs_cal_regression_line_y_pairs",
    ):
        result[_sbs_key] = None

    if use_sbs:
        sbs_out = _sbs_mcr(
            X_cal=np.asarray(X_cal, dtype=float),
            Y_cal=Y_cal,
            X_val=np.asarray(X_val, dtype=float),
            Y_val=Y_val,
            cal_s_mask=cal_s_mask_arr,
            val_s_mask=val_s_mask_arr,
            n_components=selected_rank,
            max_iter=max_iter,
            tol=tol,
            random_state=seed_value,
            c_regr=c_regr,
            st_regr=st_regr,
            c_nonneg=_safe_bool(c_nonneg, default=True),
            st_nonneg=_safe_bool(st_nonneg, default=True),
            c_norm=_safe_bool(c_norm, default=False),
            constraint_non_negative=constraint_non_negative,
            constraint_non_negative_mode_c=constraint_non_negative_mode_c,
            constraint_non_negative_mode_st=constraint_non_negative_mode_st,
            constraint_cumsum_non_negative=constraint_cumsum_non_negative,
            constraint_zero_end_points=constraint_zero_end_points,
            constraint_zero_cumsum_end_points=constraint_zero_cumsum_end_points,
            constraint_normalize=constraint_normalize,
            constraint_cut_below=constraint_cut_below,
            constraint_cut_above=constraint_cut_above,
            constraint_compress_below=constraint_compress_below,
            constraint_compress_above=constraint_compress_above,
            constraint_replace_zeros=constraint_replace_zeros,
            constraint_planarize=constraint_planarize,
            constraint_zero_end_points_span=constraint_zero_end_points_span,
            constraint_zero_cumsum_nodes=constraint_zero_cumsum_nodes,
            constraint_cut_below_value=constraint_cut_below_value,
            constraint_cut_above_value=constraint_cut_above_value,
            constraint_compress_below_value=constraint_compress_below_value,
            constraint_compress_above_value=constraint_compress_above_value,
            constraint_replace_zeros_feature=constraint_replace_zeros_feature,
            constraint_replace_zeros_fval=constraint_replace_zeros_fval,
            constraint_planarize_targets_c=constraint_planarize_targets_c,
            constraint_planarize_targets_st=constraint_planarize_targets_st,
            constraint_planarize_shape_c=constraint_planarize_shape_c,
            constraint_planarize_shape_st=constraint_planarize_shape_st,
            constraint_planarize_use_vals_above=constraint_planarize_use_vals_above,
            constraint_planarize_use_vals_below=constraint_planarize_use_vals_below,
            constraint_planarize_lims_to_plane=constraint_planarize_lims_to_plane,
            constraint_planarize_scaler=constraint_planarize_scaler,
            constraint_planarize_recalc_scaler=constraint_planarize_recalc_scaler,
            component_y_mapping=component_y_mapping,
            nway_flag=resolved_nway_flag,
            aug_direction=resolved_aug_direction,
            c_aggregation_method=resolved_c_aggregation_method,
            axis_n_info=kwargs.get("axis_n_info"),
            aug_row_counts_cal=kwargs.get("aug_row_counts_cal"),
        )
        for k in (
            "sbs_c_scores", "sbs_val_c_scores", "sbs_c_scores_heatmap", "sbs_c_scores_heatmap_full",
            "sbs_concentrations_unfolded", "sbs_concentration_row_axis", "sbs_sample_boundary_positions", "sbs_cal_val_boundary_position",
            "sbs_s_profiles", "sbs_similarity_matrix", "sbs_val_sample_axis",
            "sbs_c_sample_axis", "sbs_c_sample_axis_full", "sbs_c_component_axis", "sbs_component_labels",
            "sbs_y_cal_pred", "sbs_y_cal_true", "sbs_y_cal_error",
            "sbs_y_val_true", "sbs_y_val_pred",
        ):
            result[k] = sbs_out.get(k)
        result["auto_mapping_used"] = bool(sbs_out.get("auto_mapping_used", False))
        result["sbs_fom_metrics"] = sbs_out.get("sbs_fom_metrics")

        def _extract_pairs_3d(data: Any, indices_1based: List[int]) -> Optional[np.ndarray]:
            if data is None or not indices_1based:
                return None
            arr = np.asarray(data, dtype=float)
            if arr.ndim != 3:
                return None
            out = np.full(arr.shape[:2] + (len(indices_1based),), np.nan, dtype=float)
            for pair_idx, idx_1based in enumerate(indices_1based):
                idx0 = int(idx_1based) - 1
                if 0 <= idx0 < arr.shape[2]:
                    out[:, :, pair_idx] = arr[:, :, idx0]
            return out

        # Build SBS-specific component->Y mapping from aligned SBS calibration data,
        # instead of reusing batch/global mapping.
        _sbs_scores_all = result.get("sbs_c_scores")
        _sbs_y_cal_true_all = result.get("sbs_y_cal_true")
        if (
            isinstance(_sbs_scores_all, np.ndarray)
            and _sbs_scores_all.ndim == 3
            and isinstance(_sbs_y_cal_true_all, np.ndarray)
            and _sbs_y_cal_true_all.ndim == 3
            and _sbs_scores_all.shape[0] == _sbs_y_cal_true_all.shape[0]
            and _sbs_scores_all.shape[1] == _sbs_y_cal_true_all.shape[1]
        ):
            _n_val_sbs = int(_sbs_scores_all.shape[0])
            _n_comp_sbs = int(_sbs_scores_all.shape[2])
            _n_y_sbs = int(_sbs_y_cal_true_all.shape[2])
            _sbs_mapping: Dict[str, int] = {}
            _sbs_explicit_parsed, _sbs_explicit_map, _sbs_all_empty = _parse_component_mapping_input(
                component_y_mapping, _n_comp_sbs
            )
            if _sbs_explicit_map and not _sbs_all_empty:
                # Use the user-provided explicit mapping (convert 0-indexed keys/values → 1-indexed)
                for _comp_idx0, _y_idx0 in _sbs_explicit_parsed.items():
                    if int(_comp_idx0) < _n_comp_sbs and int(_y_idx0) < _n_y_sbs:
                        _sbs_mapping[str(int(_comp_idx0) + 1)] = int(_y_idx0) + 1
            else:
                # Auto-map by highest mean R2 across SBS models
                result["auto_mapping_used"] = True
                for _comp_idx in range(_n_comp_sbs):
                    _best_y = None
                    _best_score = -np.inf
                    for _y_idx in range(_n_y_sbs):
                        _r2_vals: List[float] = []
                        for _v in range(_n_val_sbs):
                            _x = np.asarray(_sbs_scores_all[_v, :, _comp_idx], dtype=float)
                            _y = np.asarray(_sbs_y_cal_true_all[_v, :, _y_idx], dtype=float)
                            _fit = _fit_linear_1d(_x, _y)
                            _m = _fit.get("metrics", {}) if isinstance(_fit, dict) else {}
                            _n_used = int(_safe_float(_m.get("n_samples_used"), default=0.0))
                            _r2 = _safe_float(_m.get("R2"), default=np.nan)
                            if _n_used >= 2 and np.isfinite(_r2):
                                _r2_vals.append(float(_r2))
                        if not _r2_vals:
                            continue
                        _score = float(np.mean(_r2_vals))
                        if _score > _best_score:
                            _best_score = _score
                            _best_y = int(_y_idx)
                    if _best_y is not None:
                        _sbs_mapping[str(_comp_idx + 1)] = int(_best_y + 1)

            if _sbs_mapping:
                result["component_y_mapping"] = _sbs_mapping

                _pairs_sbs: List[Tuple[int, int]] = []
                for _ck, _yv in _sbs_mapping.items():
                    try:
                        _c1 = int(_ck)
                        _y1 = int(_yv)
                    except (TypeError, ValueError):
                        continue
                    if _c1 >= 1 and _y1 >= 1:
                        _pairs_sbs.append((_c1, _y1))
                _pairs_sbs = sorted(set(_pairs_sbs), key=lambda _item: _item[0])

                _pair_components = [int(_c1) for _c1, _ in _pairs_sbs]
                _pair_y_columns = [int(_y1) for _, _y1 in _pairs_sbs]
                _pair_y_titles: List[str] = []
                _pair_labels: List[str] = []
                for _c1, _y1 in _pairs_sbs:
                    _y_idx0 = int(_y1) - 1
                    _title = f"Y{_y1}"
                    if 0 <= _y_idx0 < len(y_labels_resolved):
                        _yt = str(y_labels_resolved[_y_idx0]).strip()
                        if _yt:
                            _title = _yt
                    _pair_y_titles.append(_title)
                    _pair_labels.append(f"C{_c1} -> {_title} (Y{_y1})")

                result["mcr_pair_components"] = np.asarray(_pair_components, dtype=int)
                result["mcr_pair_y_columns"] = np.asarray(_pair_y_columns, dtype=int)
                result["mcr_pair_y_titles"] = _pair_y_titles
                result["mcr_pairing_labels"] = _pair_labels
                result["mcr_pairing_labels_by_dimension"] = [[], _pair_labels]
                _agg_lbl_sbs = result.get("c_aggregation_label", "C Score")
                result["c_score_pair_labels"] = [f"{_agg_lbl_sbs} (C{comp})" for comp in _pair_components]

        pair_components = (
            result.get("mcr_pair_components").tolist()
            if isinstance(result.get("mcr_pair_components"), np.ndarray)
            else list(result.get("mcr_pair_components") or [])
        )
        pair_y_columns = (
            result.get("mcr_pair_y_columns").tolist()
            if isinstance(result.get("mcr_pair_y_columns"), np.ndarray)
            else list(result.get("mcr_pair_y_columns") or [])
        )

        result["sbs_c_scores_cal_pairs"] = _extract_pairs_3d(result.get("sbs_c_scores"), pair_components)
        result["sbs_c_scores_val_pairs"] = _extract_pairs_3d(result.get("sbs_val_c_scores"), pair_components)
        result["sbs_y_cal_true_pairs"] = _extract_pairs_3d(result.get("sbs_y_cal_true"), pair_y_columns)
        result["sbs_y_val_true_pairs"] = _extract_pairs_3d(result.get("sbs_y_val_true"), pair_y_columns)

        _sbs_c_cal_pairs = result.get("sbs_c_scores_cal_pairs")
        _sbs_c_val_pairs = result.get("sbs_c_scores_val_pairs")
        _sbs_y_cal_true_pairs = result.get("sbs_y_cal_true_pairs")
        _sbs_y_val_true_pairs = result.get("sbs_y_val_true_pairs")

        _sbs_y_cal_pred_pairs: Optional[np.ndarray] = None
        _sbs_y_val_pred_pairs: Optional[np.ndarray] = None
        _pair_intercepts: Optional[np.ndarray] = None
        _pair_slopes: Optional[np.ndarray] = None

        if (
            isinstance(_sbs_c_cal_pairs, np.ndarray)
            and _sbs_c_cal_pairs.ndim == 3
            and isinstance(_sbs_y_cal_true_pairs, np.ndarray)
            and _sbs_y_cal_true_pairs.ndim == 3
            and _sbs_c_cal_pairs.shape[:2] == _sbs_y_cal_true_pairs.shape[:2]
            and _sbs_c_cal_pairs.shape[2] == _sbs_y_cal_true_pairs.shape[2]
        ):
            _n_val_sbs = int(_sbs_c_cal_pairs.shape[0])
            _n_cal = int(_sbs_c_cal_pairs.shape[1])
            _n_pairs = int(_sbs_c_cal_pairs.shape[2])
            _pair_intercepts = np.full((_n_val_sbs, _n_pairs), np.nan, dtype=float)
            _pair_slopes = np.full((_n_val_sbs, _n_pairs), np.nan, dtype=float)
            _pred_cal = np.full((_n_val_sbs, _n_cal, _n_pairs), np.nan, dtype=float)
            for _val_idx in range(_n_val_sbs):
                for _pair_idx, _comp_1based in enumerate(pair_components[:_n_pairs]):
                    _x = np.asarray(_sbs_c_cal_pairs[_val_idx, :, _pair_idx], dtype=float)
                    _y = np.asarray(_sbs_y_cal_true_pairs[_val_idx, :, _pair_idx], dtype=float)
                    _fit = _fit_linear_1d(_x, _y)
                    _m = _fit.get("metrics", {}) if isinstance(_fit, dict) else {}
                    _n_used = int(_safe_float(_m.get("n_samples_used"), default=0.0))
                    _intercept = _safe_float(_fit.get("intercept"), default=np.nan)
                    _slope = _safe_float(_fit.get("slope"), default=np.nan)
                    if _n_used < 2 or not (np.isfinite(_intercept) and np.isfinite(_slope)):
                        continue
                    _pair_intercepts[_val_idx, _pair_idx] = _intercept
                    _pair_slopes[_val_idx, _pair_idx] = _slope
                    _col = _sbs_c_cal_pairs[_val_idx, :, _pair_idx]
                    _mask = np.isfinite(_col)
                    _pred_cal[_val_idx, _mask, _pair_idx] = _intercept + _slope * _col[_mask]
            _sbs_y_cal_pred_pairs = _pred_cal

        if (
            isinstance(_sbs_c_val_pairs, np.ndarray)
            and _sbs_c_val_pairs.ndim == 3
            and isinstance(_pair_intercepts, np.ndarray)
            and isinstance(_pair_slopes, np.ndarray)
        ):
            _n_val_sbs = int(_sbs_c_val_pairs.shape[0])
            _n_val_rows = int(_sbs_c_val_pairs.shape[1])
            _n_pairs = int(_sbs_c_val_pairs.shape[2])
            _pred_val = np.full((_n_val_sbs, _n_val_rows, _n_pairs), np.nan, dtype=float)
            for _val_idx in range(_n_val_sbs):
                for _pair_idx, _comp_1based in enumerate(pair_components[:_n_pairs]):
                    _intercept = float(_pair_intercepts[_val_idx, _pair_idx])
                    _slope = float(_pair_slopes[_val_idx, _pair_idx])
                    if not (np.isfinite(_intercept) and np.isfinite(_slope)):
                        continue
                    _col = _sbs_c_val_pairs[_val_idx, :, _pair_idx]
                    _mask = np.isfinite(_col)
                    _pred_val[_val_idx, _mask, _pair_idx] = _intercept + _slope * _col[_mask]
            _sbs_y_val_pred_pairs = _pred_val

        result["sbs_y_cal_pred_pairs"] = _sbs_y_cal_pred_pairs
        result["sbs_y_val_pred_pairs"] = _sbs_y_val_pred_pairs
        result["sbs_y_cal_error_pairs"] = (
            np.asarray(_sbs_y_cal_true_pairs, dtype=float) - np.asarray(_sbs_y_cal_pred_pairs, dtype=float)
            if (_sbs_y_cal_true_pairs is not None and _sbs_y_cal_pred_pairs is not None)
            else None
        )

        _yv_true_pairs = _sbs_y_val_true_pairs
        _yv_pred_pairs = _sbs_y_val_pred_pairs
        result["sbs_y_val_effective_pairs"] = (
            np.where(
                np.isfinite(np.asarray(_yv_true_pairs, dtype=float)),
                np.asarray(_yv_true_pairs, dtype=float),
                np.asarray(_yv_pred_pairs, dtype=float),
            )
            if (_yv_true_pairs is not None and _yv_pred_pairs is not None)
            else (
                np.asarray(_yv_true_pairs, dtype=float).copy()
                if _yv_true_pairs is not None
                else (
                    np.asarray(_yv_pred_pairs, dtype=float).copy()
                    if _yv_pred_pairs is not None
                    else None
                )
            )
        )
        result["sbs_y_val_error_pairs"] = (
            np.asarray(_yv_true_pairs, dtype=float) - np.asarray(_yv_pred_pairs, dtype=float)
            if (_yv_true_pairs is not None and _yv_pred_pairs is not None)
            else None
        )

        # Keep legacy pairwise keys SBS-consistent in sample-by-sample mode.
        def _collapse_sbs_val_pairs(data: Any) -> Optional[np.ndarray]:
            if data is None:
                return None
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 3:
                if arr.shape[1] == 1:
                    return arr[:, 0, :]
                return np.nanmean(arr, axis=1)
            if arr.ndim == 2:
                return arr
            return None

        _yv_true_pairs_2d = _collapse_sbs_val_pairs(result.get("sbs_y_val_true_pairs"))
        _yv_pred_pairs_2d = _collapse_sbs_val_pairs(result.get("sbs_y_val_pred_pairs"))
        _yv_eff_pairs_2d = _collapse_sbs_val_pairs(result.get("sbs_y_val_effective_pairs"))
        _yv_err_pairs_2d = (
            _yv_true_pairs_2d - _yv_pred_pairs_2d
            if (_yv_true_pairs_2d is not None and _yv_pred_pairs_2d is not None)
            else None
        )

        result["y_val_true_pairs"] = _yv_true_pairs_2d
        result["y_val_pred_pairs"] = _yv_pred_pairs_2d
        result["y_val_effective_pairs"] = _yv_eff_pairs_2d
        result["y_val_error_pairs"] = _yv_err_pairs_2d

        # Recompute Predicted-vs-Reference diagonal from SBS-collapsed pair data.
        _diag_candidates: List[np.ndarray] = []
        if isinstance(_yv_true_pairs_2d, np.ndarray):
            _diag_candidates.append(_yv_true_pairs_2d)
        if isinstance(_yv_pred_pairs_2d, np.ndarray):
            _diag_candidates.append(_yv_pred_pairs_2d)
        if _diag_candidates:
            _all_diag_vals = np.concatenate([arr.ravel() for arr in _diag_candidates])
            _finite_diag_vals = _all_diag_vals[np.isfinite(_all_diag_vals)]
            if _finite_diag_vals.size > 0:
                _diag_extent = float(np.max(np.abs(_finite_diag_vals))) * 1.15 + 1e-6
                result["val_pred_ref_diag_extent"] = _diag_extent
                result["val_pred_ref_diag_x"] = np.array([-_diag_extent, _diag_extent], dtype=float)
                result["val_pred_ref_diag_y"] = np.array([-_diag_extent, _diag_extent], dtype=float)
            else:
                result["val_pred_ref_diag_extent"] = None
                result["val_pred_ref_diag_x"] = None
                result["val_pred_ref_diag_y"] = None
        else:
            result["val_pred_ref_diag_extent"] = None
            result["val_pred_ref_diag_x"] = None
            result["val_pred_ref_diag_y"] = None

        _line_x_pairs = result.get("cal_regression_line_x_pairs")
        _line_y_pairs = result.get("cal_regression_line_y_pairs")
        _sbs_scores = result.get("sbs_c_scores")
        _sbs_pair_scores = result.get("sbs_c_scores_cal_pairs")
        if (
            isinstance(_sbs_pair_scores, np.ndarray)
            and _sbs_pair_scores.ndim == 3
            and isinstance(_pair_intercepts, np.ndarray)
            and isinstance(_pair_slopes, np.ndarray)
        ):
            _n_val_sbs = int(_sbs_pair_scores.shape[0])
            _n_pairs = int(_sbs_pair_scores.shape[2])
            _line_x = np.full((_n_val_sbs, 2, _n_pairs), np.nan, dtype=float)
            _line_y = np.full((_n_val_sbs, 2, _n_pairs), np.nan, dtype=float)

            for _pair_idx, _comp_1based in enumerate(pair_components):
                for _val_idx in range(_n_val_sbs):
                    _intercept = float(_pair_intercepts[_val_idx, _pair_idx])
                    _slope = float(_pair_slopes[_val_idx, _pair_idx])
                    if not (np.isfinite(_intercept) and np.isfinite(_slope)):
                        continue
                    _col = _sbs_pair_scores[_val_idx, :, _pair_idx]
                    _finite = np.isfinite(_col)
                    if not np.any(_finite):
                        continue
                    _x1 = _x2 = np.nan
                    _y1 = _y2 = np.nan
                    if (
                        not (np.isfinite(_x1) and np.isfinite(_x2) and np.isfinite(_y1) and np.isfinite(_y2))
                        and
                        isinstance(_sbs_y_cal_true_pairs, np.ndarray)
                        and _sbs_y_cal_true_pairs.ndim == 3
                        and _pair_idx < _sbs_y_cal_true_pairs.shape[2]
                    ):
                        _ref_col = _sbs_y_cal_true_pairs[_val_idx, :, _pair_idx]
                        _valid_xy = np.isfinite(_ref_col) & np.isfinite(_col)
                        if int(np.count_nonzero(_valid_xy)) >= 2:
                            _fit_xy = _fit_linear_1d(_ref_col[_valid_xy], _col[_valid_xy])
                            _b0_xy = _safe_float(_fit_xy.get("intercept"), default=np.nan)
                            _b1_xy = _safe_float(_fit_xy.get("slope"), default=np.nan)
                            if np.isfinite(_b0_xy) and np.isfinite(_b1_xy):
                                _ref_valid = _ref_col[_valid_xy]
                                _ref_min = float(np.nanmin(_ref_valid))
                                _ref_max = float(np.nanmax(_ref_valid))
                                _ref_span = _ref_max - _ref_min if _ref_max > _ref_min else 1.0
                                _ref_buf = _ref_span * 0.05
                                _x1 = _ref_min - _ref_buf
                                _x2 = _ref_max + _ref_buf
                                _y1 = _b0_xy + _b1_xy * _x1
                                _y2 = _b0_xy + _b1_xy * _x2

                    if not (np.isfinite(_x1) and np.isfinite(_x2) and np.isfinite(_y1) and np.isfinite(_y2)):
                        _score_min = float(np.nanmin(_col[_finite]))
                        _score_max = float(np.nanmax(_col[_finite]))
                        _score_span = _score_max - _score_min if _score_max > _score_min else 1.0
                        _score_buf = _score_span * 0.05
                        _y1 = _score_min - _score_buf
                        _y2 = _score_max + _score_buf
                        _x1 = _intercept + _slope * _y1
                        _x2 = _intercept + _slope * _y2

                    _line_x[_val_idx, 0, _pair_idx] = _x1
                    _line_x[_val_idx, 1, _pair_idx] = _x2
                    _line_y[_val_idx, 0, _pair_idx] = _y1
                    _line_y[_val_idx, 1, _pair_idx] = _y2

            # Keep naming semantic aligned with plotted axes: Reference on x, Score on y.
            result["sbs_cal_regression_line_x_pairs"] = _line_x
            result["sbs_cal_regression_line_y_pairs"] = _line_y
        elif isinstance(_sbs_scores, np.ndarray) and _sbs_scores.ndim == 3:
            _n_val_sbs = int(_sbs_scores.shape[0])
            if isinstance(_line_x_pairs, np.ndarray) and _line_x_pairs.ndim == 2:
                result["sbs_cal_regression_line_x_pairs"] = np.broadcast_to(
                    _line_x_pairs[np.newaxis, :, :], (_n_val_sbs,) + _line_x_pairs.shape
                ).copy()
            if isinstance(_line_y_pairs, np.ndarray) and _line_y_pairs.ndim == 2:
                result["sbs_cal_regression_line_y_pairs"] = np.broadcast_to(
                    _line_y_pairs[np.newaxis, :, :], (_n_val_sbs,) + _line_y_pairs.shape
                ).copy()

        if sbs_out.get("y_val_pred") is not None:
            result["y_val_pred"] = sbs_out["y_val_pred"]
            result["y_val_error"] = sbs_out.get("y_val_error")

        result["mcr_als_report"] = _build_sbs_text_report(
            rank=int(selected_rank),
            pair_labels=list(result.get("mcr_pairing_labels") or []),
            sbs_y_cal_true_pairs=result.get("sbs_y_cal_true_pairs"),
            sbs_y_cal_pred_pairs=result.get("sbs_y_cal_pred_pairs"),
            y_val_true_pairs=result.get("y_val_true_pairs"),
            y_val_pred_pairs=result.get("y_val_pred_pairs"),
            auto_mapping_used=bool(result.get("auto_mapping_used", False)),
            sbs_fom_metrics=result.get("sbs_fom_metrics"),
        )
        if sweep_results:
            sweep_lines = ["", "Sweep results (first SBS model only):"]
            for item in sweep_results:
                sweep_lines.append(
                    f"- F={item.get('n_components')} | sfit={_safe_float(item.get('sfit')):.6g} | "
                    f"EV={_safe_float(item.get('explained_variance')):.4f}% | "
                    f"iter={int(_safe_int(item.get('n_iter'), default=0))}"
                )
            result["mcr_als_report"] = str(result.get("mcr_als_report") or "") + "\n".join(sweep_lines)

    # ─── EJCR packed payloads ────────────────────────────────────────────────
    _ejcr_n_pts = 100
    _ejcr_n_path = _ejcr_n_pts * 2 + 1

    _ejcr_n_pairs = 1
    _ejcr_pairs_ref = result.get("y_cal_pred_pairs")
    if (
        _ejcr_pairs_ref is not None
        and hasattr(_ejcr_pairs_ref, "shape")
        and _ejcr_pairs_ref.ndim >= 2
    ):
        _ejcr_n_pairs = max(_ejcr_n_pairs, int(_ejcr_pairs_ref.shape[1]))

    _sbs_ref = result.get("sbs_y_cal_pred_pairs")
    if (
        _sbs_ref is not None
        and hasattr(_sbs_ref, "shape")
        and _sbs_ref.ndim >= 3
    ):
        _ejcr_n_pairs = max(_ejcr_n_pairs, int(_sbs_ref.shape[2]))

    _pair_labels = result.get("mcr_pairing_labels")
    if isinstance(_pair_labels, (list, tuple)) and len(_pair_labels) > 0:
        _ejcr_n_pairs = max(_ejcr_n_pairs, len(_pair_labels))

    _ejcr_n_sbs = (
        int(_sbs_ref.shape[0])
        if _sbs_ref is not None
        and hasattr(_sbs_ref, "shape")
        and _sbs_ref.ndim >= 3
        else 1
    )

    _ejcr_levels = ("90", "95", "99")
    _ejcr_level_to_idx = {lvl: i for i, lvl in enumerate(_ejcr_levels)}
    _ejcr_item_count = len(_ejcr_levels) + 1

    def _new_packed_ejcr(color: str, nav_shape: Tuple[int, ...]) -> Dict[str, Any]:
        x_paths = np.full((_ejcr_item_count, *nav_shape, _ejcr_n_path), np.nan, dtype=float)
        y_paths = np.full((_ejcr_item_count, *nav_shape, _ejcr_n_path), np.nan, dtype=float)
        fit_slope = np.full((1, *nav_shape), np.nan, dtype=float)
        fit_intercept = np.full((1, *nav_shape), np.nan, dtype=float)
        x_paths[len(_ejcr_levels), ..., 0] = 1.0
        y_paths[len(_ejcr_levels), ..., 0] = 0.0
        return {
            "fit_slope": fit_slope,
            "fit_intercept": fit_intercept,
            "x_paths": x_paths,
            "y_paths": y_paths,
            "labels": ["90% EJCR", "95% EJCR", "99% EJCR", "Ideal (1, 0)"],
            "confidence_levels": ["90", "95", "99", None],
            "linestyles": [":", "-", "--", "none"],
            "linewidths": [1.2, 1.5, 1.2, 0.0],
            "alphas": [0.75, 0.90, 0.75, 0.95],
            "markers": [None, None, None, "o"],
            "markersizes": [7, 7, 7, 7],
            "colors": [color, color, color, "black"],
            "expand_limits": True,
        }

    ejcr_cal = _new_packed_ejcr("steelblue", (_ejcr_n_pairs,))
    ejcr_val = _new_packed_ejcr("darkorange", (_ejcr_n_pairs,))
    sbs_ejcr_cal = _new_packed_ejcr("steelblue", (_ejcr_n_pairs, _ejcr_n_sbs))
    sbs_ejcr_val = _new_packed_ejcr("darkorange", (_ejcr_n_pairs,))

    try:
        from chemometrics.ejcr_analysis import compute_ejcr as _compute_ejcr

        def _fill_packed(y_ref_1d, y_pred_1d, packed: Dict[str, Any], nav_idx: Tuple[int, ...]) -> None:
            valid = np.isfinite(y_ref_1d) & np.isfinite(y_pred_1d)
            if valid.sum() < 3:
                return
            r = _compute_ejcr(y_ref_1d[valid], y_pred_1d[valid], n_points=_ejcr_n_pts)
            packed["fit_slope"][(0, *nav_idx)] = r["slope"]
            packed["fit_intercept"][(0, *nav_idx)] = r["intercept"]
            for ell in r["ellipses"]:
                pct = str(ell.get("confidence_pct", "")).strip()
                idx = _ejcr_level_to_idx.get(pct)
                if idx is None:
                    continue
                es = np.asarray(ell["ellipse_slope"], dtype=float)
                ei = np.asarray(ell["ellipse_intercept"], dtype=float)
                k = min(len(es), _ejcr_n_path)
                packed["x_paths"][(idx, *nav_idx, slice(0, k))] = es[:k]
                packed["y_paths"][(idx, *nav_idx, slice(0, k))] = ei[:k]

        _cal_true_p = result.get("y_cal_true_pairs")
        _cal_pred_p = result.get("y_cal_pred_pairs")
        if (
            _cal_true_p is not None and _cal_pred_p is not None
            and hasattr(_cal_true_p, "shape") and _cal_true_p.ndim == 2
        ):
            _nc = min(_cal_true_p.shape[1], _cal_pred_p.shape[1], _ejcr_n_pairs)
            for _p in range(_nc):
                _fill_packed(_cal_true_p[:, _p], _cal_pred_p[:, _p], ejcr_cal, (_p,))

        _val_true_p = result.get("y_val_true_pairs")
        _val_pred_p = result.get("y_val_pred_pairs")
        if (
            _val_true_p is not None and _val_pred_p is not None
            and hasattr(_val_true_p, "shape") and _val_true_p.ndim == 2
        ):
            _nc = min(_val_true_p.shape[1], _val_pred_p.shape[1], _ejcr_n_pairs)
            for _p in range(_nc):
                _fill_packed(_val_true_p[:, _p], _val_pred_p[:, _p], ejcr_val, (_p,))

        _sbs_cal_true = result.get("sbs_y_cal_true_pairs")
        _sbs_cal_pred = result.get("sbs_y_cal_pred_pairs")
        if (
            _sbs_cal_true is not None and _sbs_cal_pred is not None
            and hasattr(_sbs_cal_true, "shape") and _sbs_cal_true.ndim == 3
        ):
            _n_sbs_c, _, _n_p_c = _sbs_cal_true.shape
            for _s in range(min(_n_sbs_c, _ejcr_n_sbs)):
                for _p in range(min(_n_p_c, _ejcr_n_pairs)):
                    _fill_packed(_sbs_cal_true[_s, :, _p], _sbs_cal_pred[_s, :, _p], sbs_ejcr_cal, (_p, _s))

        _sbs_val_true = result.get("sbs_y_val_true_pairs")
        _sbs_val_pred = result.get("sbs_y_val_pred_pairs")
        if (
            _sbs_val_true is not None and _sbs_val_pred is not None
            and hasattr(_sbs_val_true, "shape") and _sbs_val_true.ndim == 3
        ):
            _n_sbs_v, _, _n_p_v = _sbs_val_true.shape
            for _p in range(min(_n_p_v, _ejcr_n_pairs)):
                _fill_packed(_sbs_val_true[:_n_sbs_v, 0, _p], _sbs_val_pred[:_n_sbs_v, 0, _p], sbs_ejcr_val, (_p,))

    except Exception:
        pass

    result.update({
        "ejcr_cal": ejcr_cal,
        "ejcr_val": ejcr_val,
        "sbs_ejcr_cal": sbs_ejcr_cal,
        "sbs_ejcr_val": sbs_ejcr_val,
    })

    return result


_MCR_ALS_RETURN_ORDER: Tuple[str, ...] = (
    "c_scores",
    "val_c_scores",
    "c_scores_heatmap",
    "c_scores_heatmap_full",
    "c_sample_axis",
    "c_sample_axis_full",
    "c_component_axis",
    "component_labels",
    "sweep_components",
    "sweep_sfit",
    "sweep_n_iter",
    "sweep_explained_variance",
    "sweep_model_axis",
    "sweep_model_labels",
    "sweep_component_labels",
    "sweep_c_component_axis",
    "sweep_c_sample_axis_full",
    "sweep_c_scores",
    "sweep_val_c_scores",
    "sweep_c_scores_heatmap_full",
    "sweep_concentrations_unfolded",
    "sweep_s_profiles",
    "s_profiles",
    "concentrations",
    "concentrations_unfolded",
    "concentration_row_axis",
    "s_axis_vector",
    "s_axis_label",
    "c_aggregation_label",
    "sample_boundary_positions",
    "cal_val_boundary_position",
    "reconstructed",
    "residual",
    "metrics",
    "calibration_models",
    "component_y_mapping",
    "auto_mapping_used",
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
    "sweep_mcr_pair_components",
    "sweep_mcr_pair_y_columns",
    "sweep_mcr_pair_y_titles",
    "sweep_mcr_pair_y_titles_by_model",
    "sweep_mcr_pairing_labels",
    "sweep_mcr_pairing_labels_by_model",
    "sweep_mcr_pairing_labels_by_dimension",
    "y_cal_pred_pairs",
    "y_cal_true_pairs",
    "y_cal_error_pairs",
    "y_val_pred_pairs",
    "y_val_true_pairs",
    "y_val_error_pairs",
    "c_scores_cal_pairs",
    "c_scores_val_pairs",
    "c_score_pair_labels",
    "y_val_effective_pairs",
    "cal_regression_line_x_pairs",
    "cal_regression_line_y_pairs",
    "val_pred_ref_diag_extent",
    "val_pred_ref_diag_x",
    "val_pred_ref_diag_y",
    "sweep_y_cal_pred_pairs",
    "sweep_y_cal_true_pairs",
    "sweep_y_cal_error_pairs",
    "sweep_y_val_pred_pairs",
    "sweep_y_val_true_pairs",
    "sweep_y_val_error_pairs",
    "sweep_c_scores_cal_pairs",
    "sweep_c_scores_val_pairs",
    "sweep_y_val_effective_pairs",
    "sweep_cal_regression_line_x_pairs",
    "sweep_cal_regression_line_y_pairs",
    "sweep_val_pred_ref_diag_extent",
    "sweep_val_pred_ref_diag_x",
    "sweep_val_pred_ref_diag_y",
    "validation_processing",
    "sbs_c_scores",
    "sbs_val_c_scores",
    "sbs_c_scores_heatmap",
    "sbs_c_scores_heatmap_full",
    "sbs_concentrations_unfolded",
    "sbs_concentration_row_axis",
    "sbs_sample_boundary_positions",
    "sbs_cal_val_boundary_position",
    "sbs_s_profiles",
    "sbs_similarity_matrix",
    "sbs_val_sample_axis",
    "sbs_c_sample_axis",
    "sbs_c_sample_axis_full",
    "sbs_c_component_axis",
    "sbs_component_labels",
    "sbs_y_cal_pred",
    "sbs_y_cal_true",
    "sbs_y_cal_error",
    "sbs_y_val_true",
    "sbs_y_val_pred",
    "sbs_c_scores_cal_pairs",
    "sbs_c_scores_val_pairs",
    "sbs_y_cal_pred_pairs",
    "sbs_y_cal_true_pairs",
    "sbs_y_cal_error_pairs",
    "sbs_y_val_true_pairs",
    "sbs_y_val_pred_pairs",
    "sbs_y_val_effective_pairs",
    "sbs_y_val_error_pairs",
    "sbs_cal_regression_line_x_pairs",
    "sbs_cal_regression_line_y_pairs",
    "ejcr_cal",
    "ejcr_val",
    "sweep_ejcr_cal",
    "sweep_ejcr_val",
    "sbs_ejcr_cal",
    "sbs_ejcr_val",
)


def mcr_als_analysis_standard(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:
    """Adapter for app execution pipeline: return outputs as ordered tuple."""
    result = mcr_als_analysis(*args, **kwargs)
    if not isinstance(result, dict):
        return (result,)
    return tuple(result.get(key) for key in _MCR_ALS_RETURN_ORDER)
