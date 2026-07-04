"""N-PLS / U-PLS analysis for multiway calibration with canonical RBL/RTL correction."""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

try:
    from chemometrics.cv_pipeline import CVPipeline, CVConfig

    HAS_CV = True
except Exception:
    HAS_CV = False

try:
    from tensorly.regression import CP_PLSR
    from tensorly.decomposition import tucker
    from tensorly.tucker_tensor import tucker_to_tensor
    from tensorly.cp_tensor import cp_to_tensor

    HAS_TENSORLY = True
except Exception:
    HAS_TENSORLY = False

try:
    from execution_reporting import emit_execution_message, emit_execution_warning
except ImportError:
    def emit_execution_message(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        return

    def emit_execution_warning(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        return


def _as_2d_y(y: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if y is None:
        return None
    arr = np.asarray(y, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Y must be 1D or 2D.")
    return arr


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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


def _normalize_method(value: Any) -> str:
    raw = str(value).strip().lower() if value is not None else "n_pls"
    aliases = {
        "npls": "n_pls",
        "n-pls": "n_pls",
        "n_pls": "n_pls",
        "upls": "u_pls",
        "u-pls": "u_pls",
        "u_pls": "u_pls",
        "unfolded_pls": "u_pls",
        "unfolded pls": "u_pls",
    }
    return aliases.get(raw, "n_pls")


def _reshape_x_to_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt, axis=0, keepdims=True)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _tucker_supported_for_ndim(ndim: int) -> bool:
    if not HAS_TENSORLY:
        return False
    if ndim < 3:
        return False

    try:
        shape = tuple([2] * ndim)
        tensor = np.random.RandomState(0).randn(*shape)
        rank = tuple([min(2, s) for s in shape])
        core, factors = tucker(tensor, rank=rank, init="svd")
        _ = tucker_to_tensor((core, factors))
        return True
    except Exception:
        return False


def _build_pred_ref_diag(*arrays: Optional[np.ndarray], buffer_frac: float = 0.10) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build a y=x diagonal from the combined finite range of all supplied arrays.

    Each positional argument may be a 1-D array or None (None inputs are skipped).
    A fractional buffer is added on both ends so the line extends beyond the data.
    """
    all_vals: List[float] = []
    for a in arrays:
        if a is None:
            continue
        arr = np.asarray(a, dtype=float).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            all_vals.extend(finite.tolist())
    if not all_vals:
        return None, None
    lo = float(min(all_vals))
    hi = float(max(all_vals))
    span = hi - lo
    if span < 1e-12:
        span = max(abs(lo), abs(hi), 1.0) * 0.2
    buf = span * buffer_frac
    lo -= buf
    hi += buf
    return np.asarray([lo, hi], dtype=float), np.asarray([lo, hi], dtype=float)


def _parse_interferent_rank(multilinear_rank: Any, nway_flag: int, default_rank: int = 1) -> int:
    if multilinear_rank is None:
        return max(0, int(default_rank))

    if isinstance(multilinear_rank, (list, tuple, np.ndarray)):
        tokens = [str(v).strip() for v in multilinear_rank]
    else:
        tokens = [tok.strip() for tok in str(multilinear_rank).replace(";", ",").split(",")]

    ranks: List[int] = []
    for tok in tokens:
        if not tok:
            continue
        try:
            ranks.append(max(0, int(float(tok))))
        except Exception:
            continue

    if not ranks:
        return max(0, int(default_rank))

    # Canonical RBL/RTL uses a single interferent rank Aint across non-sample modes.
    return int(ranks[0])


def _low_rank_component(
    vector: np.ndarray,
    sample_shape: Sequence[int],
    rank: int,
    nway_flag: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    vec = np.asarray(vector, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vec)):
        raise ValueError(
            "Input data contains NaN or Inf values — RBL/RTL correction cannot proceed. "
            "Remove or impute missing values before running N-PLS/U-PLS with interferent correction."
        )
    shape = tuple(int(s) for s in sample_shape)
    if rank <= 0:
        return np.zeros_like(vec), {"method_used": None, "rank_used": 0}

    if len(shape) == 2 or int(nway_flag) == 2:
        mat = vec.reshape(shape)
        n_eff = max(1, min(int(rank), mat.shape[0], mat.shape[1]))

        # Canonical second-order correction is SVD-based bilinearization.
        u, s, vt = np.linalg.svd(mat, full_matrices=False)
        s_int = u[:, :n_eff] @ np.diag(s[:n_eff]) @ vt[:n_eff, :]

        return s_int.reshape(-1), {
            "method_used": "svd",
            "rank_used": int(n_eff),
            "scores": np.asarray(u[:, :n_eff] * s[:n_eff], dtype=float),
            "loadings": np.asarray(vt[:n_eff, :].T, dtype=float),
            "mode_factors": [np.asarray(u[:, :n_eff], dtype=float), np.asarray(vt[:n_eff, :].T, dtype=float)],
            "core": None,
            "reconstructed": np.asarray(s_int, dtype=float),
        }

    if not HAS_TENSORLY:
        raise ValueError("TensorLy is required for canonical RTL (nway_flag >= 3).")

    ten = vec.reshape(shape)
    rank_used = tuple(max(1, min(int(rank), int(s))) for s in shape)
    core, factors = tucker(ten, rank=rank_used, init="svd")
    rec = tucker_to_tensor((core, factors))
    return rec.reshape(-1), {
        "method_used": "tucker",
        "rank_used": [int(v) for v in rank_used],
        "scores": None,
        "loadings": [np.asarray(f, dtype=float) for f in factors],
        "mode_factors": [np.asarray(f, dtype=float) for f in factors],
        "core": np.asarray(core, dtype=float),
        "reconstructed": np.asarray(rec, dtype=float),
    }


def _fit_upls1(X_cal_2d: np.ndarray, y_cal_1d: np.ndarray, n_components: int) -> Dict[str, Any]:
    R = np.asarray(X_cal_2d, dtype=float).copy()
    yk = np.asarray(y_cal_1d, dtype=float).reshape(-1).copy()
    x0 = R.copy()
    y0 = yk.copy()
    n_comp_eff = max(1, min(int(n_components), R.shape[0], R.shape[1]))

    p_cols: List[np.ndarray] = []
    w_cols: List[np.ndarray] = []
    v_vals: List[float] = []
    t_cols: List[np.ndarray] = []
    ssx_percent: List[float] = []
    ssy_percent: List[float] = []
    x_denom = float(np.sum(x0 ** 2)) + 1e-12
    y_denom = float(np.sum(y0 ** 2)) + 1e-12

    for _ in range(n_comp_eff):
        denom_y = float(np.dot(yk, yk)) + 1e-12
        w = (R.T @ yk) / denom_y
        w_norm = float(np.linalg.norm(w)) + 1e-12
        wn = w / w_norm

        t = R @ wn
        denom_t = float(np.dot(t, t)) + 1e-12
        v = float(np.dot(t, yk) / denom_t)
        p = (R.T @ t) / denom_t

        ss_x_before = float(np.sum(R ** 2))
        ss_y_before = float(np.sum(yk ** 2))

        R = R - np.outer(t, p)
        yk = yk - v * t

        ssx_percent.append(float(100.0 * (ss_x_before - np.sum(R ** 2)) / x_denom))
        ssy_percent.append(float(100.0 * (ss_y_before - np.sum(yk ** 2)) / y_denom))

        p_cols.append(p)
        w_cols.append(wn)
        v_vals.append(v)
        t_cols.append(t)

    P = np.column_stack(p_cols)
    Wn = np.column_stack(w_cols)
    v_vec = np.asarray(v_vals, dtype=float)
    T = np.column_stack(t_cols)

    x_to_t = np.linalg.pinv(Wn.T @ P) @ Wn.T

    return {
        "P": P,
        "Wn": Wn,
        "v": v_vec,
        "T": T,
        "x_to_t": x_to_t,
        "x_res_cal": R,
        "y_res_cal": yk,
        "ssx_percent": np.asarray(ssx_percent, dtype=float),
        "ssy_percent": np.asarray(ssy_percent, dtype=float),
    }


def _predict_upls1_sample(
    model: Dict[str, Any],
    x_sample_1d: np.ndarray,
    sample_shape: Sequence[int],
    nway_flag: int,
    apply_correction: bool,
    interferent_rank: int,
    max_iter: int,
) -> Dict[str, Any]:
    x = np.asarray(x_sample_1d, dtype=float).reshape(-1)
    P = np.asarray(model["P"], dtype=float)
    v = np.asarray(model["v"], dtype=float)
    x_to_t = np.asarray(model["x_to_t"], dtype=float)

    beta0 = x_to_t @ x
    base_linear = P @ beta0
    base_res = x - base_linear
    y_base = float(v @ beta0)

    if (not apply_correction) or int(interferent_rank) <= 0:
        return {
            "y_base": y_base,
            "y_corrected": y_base,
            "beta_base": beta0,
            "beta_corrected": beta0,
            "res_before": base_res,
            "res_after": base_res,
            "low_rank_component": np.zeros_like(base_res),
            "correction_payload": None,
            "solver_success": True,
        }

    def residual_obj(beta: np.ndarray) -> np.ndarray:
        linear_res = x - (P @ beta)
        low_rank, _ = _low_rank_component(
            linear_res,
            sample_shape=sample_shape,
            rank=int(interferent_rank),
            nway_flag=int(nway_flag),
        )
        return linear_res - low_rank

    ls_res = least_squares(residual_obj, beta0, method="trf", max_nfev=max(20, int(max_iter)))
    beta_corr = np.asarray(ls_res.x if ls_res.success else beta0, dtype=float)

    linear_res_corr = x - (P @ beta_corr)
    low_rank_corr, corr_payload = _low_rank_component(
        linear_res_corr,
        sample_shape=sample_shape,
        rank=int(interferent_rank),
        nway_flag=int(nway_flag),
    )
    res_after = linear_res_corr - low_rank_corr
    y_corr = float(v @ beta_corr)

    return {
        "y_base": y_base,
        "y_corrected": y_corr,
        "beta_base": beta0,
        "beta_corrected": beta_corr,
        "res_before": base_res,
        "res_after": res_after,
        "low_rank_component": low_rank_corr,
        "correction_payload": corr_payload,
        "solver_success": bool(ls_res.success),
    }


def _build_cp_spatial_kron(model: CP_PLSR) -> np.ndarray:
    x_factors = [np.asarray(f, dtype=float) for f in model.X_factors]
    n_comp = int(x_factors[0].shape[1])

    cols: List[np.ndarray] = []
    spatial_factors = x_factors[1:]
    for comp_idx in range(n_comp):
        w = spatial_factors[-1][:, comp_idx]
        for mode_idx in range(len(spatial_factors) - 2, -1, -1):
            w = np.kron(w, spatial_factors[mode_idx][:, comp_idx])
        cols.append(w)
    return np.column_stack(cols)


def _predict_npls1_from_scores(model: CP_PLSR, scores_1d: np.ndarray, y_offset: float) -> float:
    t = np.asarray(scores_1d, dtype=float).reshape(1, -1)
    y_hat = t @ np.asarray(model.coef_, dtype=float) @ np.asarray(model.Y_factors[1], dtype=float).T
    # model was fitted on centered y; add external offset for canonical centering.
    return float(y_hat.reshape(-1)[0] + y_offset)


def _predict_npls1_sample(
    model: CP_PLSR,
    spatial_kron: np.ndarray,
    x_sample_tensor: np.ndarray,
    sample_shape: Sequence[int],
    nway_flag: int,
    apply_correction: bool,
    interferent_rank: int,
    max_iter: int,
    y_offset: float,
) -> Dict[str, Any]:
    x_t = np.asarray(x_sample_tensor, dtype=float)
    x_vec = x_t.reshape(-1)

    beta0 = np.asarray(model.transform(x_t[None, ...])[0], dtype=float)
    base_res = x_vec - (beta0 @ spatial_kron.T)
    y_base = _predict_npls1_from_scores(model, beta0, y_offset=y_offset)

    if (not apply_correction) or int(interferent_rank) <= 0:
        return {
            "y_base": y_base,
            "y_corrected": y_base,
            "beta_base": beta0,
            "beta_corrected": beta0,
            "res_before": base_res,
            "res_after": base_res,
            "low_rank_component": np.zeros_like(base_res),
            "correction_payload": None,
            "solver_success": True,
        }

    def residual_obj(beta: np.ndarray) -> np.ndarray:
        linear_res = x_vec - (beta @ spatial_kron.T)
        low_rank, _ = _low_rank_component(
            linear_res,
            sample_shape=sample_shape,
            rank=int(interferent_rank),
            nway_flag=int(nway_flag),
        )
        return linear_res - low_rank

    ls_res = least_squares(residual_obj, beta0, method="trf", max_nfev=max(20, int(max_iter)))
    beta_corr = np.asarray(ls_res.x if ls_res.success else beta0, dtype=float)

    linear_res_corr = x_vec - (beta_corr @ spatial_kron.T)
    low_rank_corr, corr_payload = _low_rank_component(
        linear_res_corr,
        sample_shape=sample_shape,
        rank=int(interferent_rank),
        nway_flag=int(nway_flag),
    )
    res_after = linear_res_corr - low_rank_corr
    y_corr = _predict_npls1_from_scores(model, beta_corr, y_offset=y_offset)

    return {
        "y_base": y_base,
        "y_corrected": y_corr,
        "beta_base": beta0,
        "beta_corrected": beta_corr,
        "res_before": base_res,
        "res_after": res_after,
        "low_rank_component": low_rank_corr,
        "correction_payload": corr_payload,
        "solver_success": bool(ls_res.success),
    }


def _unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    arr = np.asarray(tensor, dtype=float)
    moved = np.moveaxis(arr, mode, 0)
    return moved.reshape(arr.shape[mode], -1)


def _fold(unfolded: np.ndarray, shape: Sequence[int], mode: int) -> np.ndarray:
    shape_t = tuple(int(v) for v in shape)
    target = np.reshape(unfolded, (shape_t[mode],) + tuple(shape_t[i] for i in range(len(shape_t)) if i != mode))
    return np.moveaxis(target, 0, mode)


def _project_tensor_orthogonal(tensor: np.ndarray, mode_factors: Sequence[np.ndarray]) -> np.ndarray:
    projected = np.asarray(tensor, dtype=float)
    if len(mode_factors) != projected.ndim:
        return projected

    for mode, factor in enumerate(mode_factors):
        f = np.asarray(factor, dtype=float)
        if f.ndim != 2 or f.shape[0] != projected.shape[mode] or f.shape[1] == 0:
            continue
        unfolded = _unfold(projected, mode)
        unfolded = unfolded - (f @ (f.T @ unfolded))
        projected = _fold(unfolded, projected.shape, mode)
    return projected


def _compute_sensitivity_from_projection(
    loadings_matrix: np.ndarray,
    latent_to_y: np.ndarray,
    sample_shape: Sequence[int],
    interferent_mode_factors: Optional[Sequence[np.ndarray]] = None,
) -> float:
    loadings = np.asarray(loadings_matrix, dtype=float)
    v = np.asarray(latent_to_y, dtype=float).reshape(-1)
    if loadings.ndim != 2 or loadings.shape[1] == 0:
        return float("nan")

    if interferent_mode_factors is None:
        val = np.linalg.norm(np.linalg.pinv(loadings).T @ v)
        return float(1.0 / (val + 1e-12))

    peff_cols: List[np.ndarray] = []
    shape = tuple(int(s) for s in sample_shape)
    for comp_idx in range(loadings.shape[1]):
        comp_tensor = loadings[:, comp_idx].reshape(shape)
        comp_proj = _project_tensor_orthogonal(comp_tensor, interferent_mode_factors)
        peff_cols.append(comp_proj.reshape(-1))

    peff = np.column_stack(peff_cols)
    val = np.linalg.norm(np.linalg.pinv(peff).T @ v)
    return float(1.0 / (val + 1e-12))


def _compute_lod_loq_bounds(
    sr: float,
    sensitivity: float,
    sr_c: float,
    hmin: float,
    hmax: float,
    lod_confidence_factor: float,
) -> Tuple[float, float, float, float]:
    sen = max(float(abs(sensitivity)), 1e-12)
    lod_min = float(lod_confidence_factor * np.sqrt((sr / sen) ** 2 * (1.0 + hmin) + hmin * (sr_c ** 2)))
    lod_max = float(lod_confidence_factor * np.sqrt((sr / sen) ** 2 * (1.0 + hmax) + hmax * (sr_c ** 2)))
    loq_min = float(lod_confidence_factor * lod_min)
    loq_max = float(lod_confidence_factor * lod_max)
    return lod_min, lod_max, loq_min, loq_max


def _lod_factor_from_confidence_level(confidence_level: float, dof: Optional[int] = None) -> float:
    """Convert confidence level to LOD factor using one-sided alpha=beta.

    Uses Student's t quantile when dof is provided and finite; otherwise uses
    standard normal quantile. At confidence_level=0.95 and large dof, returns
    about 3.29, i.e. the canonical 3.3.
    """
    cl = float(confidence_level)
    if cl <= 0.5 or cl >= 1.0:
        raise ValueError("lod_confidence_level must be in (0.5, 1.0).")

    if dof is not None and int(dof) > 0:
        t_quant = float(student_t.ppf(cl, df=int(dof)))
        return float(2.0 * t_quant)

    z = float(NormalDist().inv_cdf(cl))
    return float(2.0 * z)


def _normalize_lod_distribution(value: Any) -> str:
    raw = str(value).strip().lower() if value is not None else "student"
    aliases = {
        "student": "student",
        "student_t": "student",
        "student-t": "student",
        "t": "student",
        "normal": "normal",
        "gaussian": "normal",
        "z": "normal",
    }
    return aliases.get(raw, "student")


def _compute_npls_explained_by_lv(
    X_centered: np.ndarray,
    y_centered: np.ndarray,
    n_components: int,
    tol: float,
    max_iter: int,
    random_state: Optional[int],
) -> Dict[str, np.ndarray]:
    if not HAS_TENSORLY:
        return {
            "x_percent": np.asarray([np.nan] * n_components, dtype=float),
            "y_percent": np.asarray([np.nan] * n_components, dtype=float),
        }

    x_out: List[float] = []
    y_out: List[float] = []
    y_vec = np.asarray(y_centered, dtype=float).reshape(-1)
    y_denom = float(np.sum(y_vec ** 2)) + 1e-12
    x_denom = float(np.sum(X_centered ** 2)) + 1e-12

    prev_y_cumul = 0.0
    prev_x_cumul = 0.0

    for comp in range(1, int(n_components) + 1):
        model = CP_PLSR(
            n_components=int(comp),
            tol=float(tol),
            n_iter_max=int(max_iter),
            random_state=random_state,
            verbose=False,
        )
        model.fit(X_centered, y_vec.reshape(-1, 1))
        y_hat = np.asarray(model.predict(X_centered), dtype=float).reshape(-1)
        y_res = y_vec - y_hat
        y_cumul = float(100.0 * (1.0 - np.sum(y_res ** 2) / y_denom))
        y_out.append(y_cumul - prev_y_cumul)
        prev_y_cumul = y_cumul

        try:
            x_rec = cp_to_tensor((np.ones(comp), model.X_factors))
            x_cumul = float(100.0 * (1.0 - np.sum((X_centered - x_rec) ** 2) / x_denom))
            x_out.append(x_cumul - prev_x_cumul)
            prev_x_cumul = x_cumul
        except Exception:
            x_out.append(float("nan"))

    return {
        "x_percent": np.asarray(x_out, dtype=float),
        "y_percent": np.asarray(y_out, dtype=float),
    }


def _build_npls_report(
    n_y: int,
    metrics: Dict[str, Any],
    model: Dict[str, Any],
    y_labels: Optional[List[str]] = None,
) -> str:
    """Build a per-response formatted text report for N-PLS / U-PLS results."""
    lines: List[str] = []
    lines.append("N-PLS / U-PLS Report")
    lines.append("====================")
    lines.append(f"Method: {model.get('method', 'unknown')}")
    lines.append(f"Latent Variables: {model.get('n_components', '?')}")
    lines.append(f"Data Order (nway): {model.get('nway_flag', '?')}")
    lines.append("")

    ml = metrics.get("residual_multilinearization") or {}
    if ml.get("attempted"):
        lines.append("Residual Correction")
        lines.append("-------------------")
        lines.append(f"Applied: {ml.get('applied')}")
        lines.append(f"Method: {ml.get('method_used', 'N/A')}")
        lines.append(f"Canonical Mode: {ml.get('canonical_mode', 'N/A')}")
        lines.append(f"Interferent Rank: {ml.get('interferent_rank', 0)}")
        lines.append(f"Solver Success: {ml.get('solver_success_count', 0)} / {ml.get('solver_total_count', 0)}")
        lines.append("")

    lines.append("Per-Response Performance")
    lines.append("------------------------")

    cal = metrics.get("calibration") or {}
    cv = metrics.get("cv") or {}
    val = metrics.get("validation") or {}
    r2_cal = cal.get("r2_per_response")
    rmse_cal = cal.get("rmse_per_response")
    r2_cv = cv.get("r2_per_response") if cv else None
    rmse_cv = cv.get("rmse_per_response") if cv else None
    r2_val = val.get("r2_per_response") if val else None
    rmse_val = val.get("rmse_per_response") if val else None

    for yi in range(n_y):
        _lbl = y_labels[yi] if (y_labels and yi < len(y_labels) and str(y_labels[yi]).strip()) else None
        _header = f"[Y{yi + 1} - {_lbl}]" if _lbl else f"[Y{yi + 1}]"
        lines.append(_header)
        if r2_cal is not None and yi < len(r2_cal):
            lines.append(f"  Calibration      -> R\u00b2: {float(r2_cal[yi]):.4f}, RMSEC: {float(rmse_cal[yi]):.4g}")
        if r2_cv is not None and yi < len(r2_cv):
            lines.append(f"  Cross-Validation -> R\u00b2: {float(r2_cv[yi]):.4f}, RMSECV: {float(rmse_cv[yi]):.4g}")
        if r2_val is not None and yi < len(r2_val):
            lines.append(f"  Validation       -> R\u00b2: {float(r2_val[yi]):.4f}, RMSEP: {float(rmse_val[yi]):.4g}")
        lines.append("")

    return "\n".join(lines)


def npls_analysis(
    X_cal: Optional[np.ndarray] = None,
    Y_cal: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    n_components: int = 2,
    nway_flag: int = 2,
    multiway_method: str = "n_pls",
    multilinear_rank: Any = None,
    lod_distribution: str = "student",
    lod_confidence_level: float = 0.95,
    y_labels: Optional[Any] = None,
    axis_n_info: Optional[Any] = None,
    dim_labels: Optional[Any] = None,
    pls_scale: bool = False,
    tol: float = 1e-9,
    max_iter: int = 200,
    random_state: Any = None,
    cv_config: Optional[Any] = None,
    fold: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    """Run U-PLS or N-PLS with canonical sample-wise residual correction.

    Canonical behavior:
    - nway_flag == 2: RBL bilinear correction (SVD)
    - nway_flag >= 3: Tucker-based multilinear residual correction (RTL extension)
    """
    if cv_config is not None and isinstance(cv_config, dict) and "cv_config" in cv_config:
        cv_config = cv_config["cv_config"]

    # CVPipeline routes split arrays as *_train/*_test keys.
    if X_cal is None and "X_cal_train" in kwargs:
        X_cal = kwargs["X_cal_train"]
    if Y_cal is None and "Y_cal_train" in kwargs:
        Y_cal = kwargs["Y_cal_train"]
    if X_val is None and "X_cal_test" in kwargs:
        X_val = kwargs["X_cal_test"]
    if Y_val is None and "Y_cal_test" in kwargs:
        Y_val = kwargs["Y_cal_test"]

    is_cv_fold_call = ("X_cal_train" in kwargs and "Y_cal_train" in kwargs)

    if (
        cv_config is not None
        and HAS_CV
        and hasattr(cv_config, "is_enabled")
        and cv_config.is_enabled()
        and int(fold) == 0
        and not is_cv_fold_call
    ):
        if X_cal is None or Y_cal is None:
            raise ValueError("X_cal and Y_cal are required.")

        pipeline = CVPipeline(cv_config)
        cv_results_dict = pipeline.run(
            npls_analysis,
            X_cal=np.asarray(X_cal, dtype=float),
            Y_cal=np.asarray(Y_cal, dtype=float),
            n_components=n_components,
            nway_flag=nway_flag,
            multiway_method=multiway_method,
            multilinear_rank=multilinear_rank,
            lod_distribution=lod_distribution,
            lod_confidence_level=lod_confidence_level,
            pls_scale=pls_scale,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            reference_input_key="Y_cal",
            comparison_output_key="y_val_pred",
            capture_output_keys=["y_val_pred"],
        )

        single_fit = npls_analysis(
            X_cal=X_cal,
            Y_cal=Y_cal,
            X_val=X_val,
            Y_val=Y_val,
            n_components=n_components,
            nway_flag=nway_flag,
            multiway_method=multiway_method,
            multilinear_rank=multilinear_rank,
            lod_distribution=lod_distribution,
            lod_confidence_level=lod_confidence_level,
            y_labels=y_labels,
            axis_n_info=axis_n_info,
            dim_labels=dim_labels,
            pls_scale=pls_scale,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            cv_config=None,
            fold=-1,
        )

        y_cv_pred = cv_results_dict.get("y_val_pred_cv", None)
        y_cv_error = None
        metrics_cv = None
        if y_cv_pred is not None:
            y_cv_pred = np.asarray(y_cv_pred, dtype=float)
            y_cal_true = np.asarray(single_fit.get("y_cal_true"), dtype=float)
            if y_cv_pred.shape == y_cal_true.shape:
                y_cv_error = y_cal_true - y_cv_pred
                _n_y_cv = y_cal_true.shape[1]
                metrics_cv = {
                    "n_samples": int(y_cal_true.shape[0]),
                    "r2": _r2(y_cal_true, y_cv_pred),
                    "rmse": _rmse(y_cal_true, y_cv_pred),
                    "r2_per_response": np.asarray([_r2(y_cal_true[:, _yi], y_cv_pred[:, _yi]) for _yi in range(_n_y_cv)], dtype=float),
                    "rmse_per_response": np.asarray([_rmse(y_cal_true[:, _yi], y_cv_pred[:, _yi]) for _yi in range(_n_y_cv)], dtype=float),
                }

        single_fit["y_cv_pred"] = y_cv_pred
        single_fit["y_cv_error"] = y_cv_error
        if isinstance(single_fit.get("metrics"), dict):
            single_fit["metrics"]["cv"] = metrics_cv

        # Rebuild npls_report now that CV metrics are available.
        if isinstance(single_fit.get("metrics"), dict) and isinstance(single_fit.get("model"), dict):
            _sf_n_y = np.asarray(single_fit["y_cal_true"]).shape[1] if single_fit.get("y_cal_true") is not None else 0
            single_fit["npls_report"] = _build_npls_report(_sf_n_y, single_fit["metrics"], single_fit["model"], y_labels)

        # Rebuild cal/cv diagonal to include y_cv_pred in the range calculation.
        if y_cv_pred is not None and y_cv_pred.shape == y_cal_true.shape:
            _y_cal_pred_sf = np.asarray(single_fit.get("y_cal_pred"), dtype=float)
            _n_y_cv = y_cal_true.shape[1]
            _d_x = np.full((2, _n_y_cv), np.nan, dtype=float)
            _d_y = np.full((2, _n_y_cv), np.nan, dtype=float)
            for _yi in range(_n_y_cv):
                _dx, _dy = _build_pred_ref_diag(y_cal_true[:, _yi], _y_cal_pred_sf[:, _yi], y_cv_pred[:, _yi])
                if _dx is not None:
                    _d_x[:, _yi] = _dx
                    _d_y[:, _yi] = _dy
            single_fit["pred_ref_diag_x"] = _d_x
            single_fit["pred_ref_diag_y"] = _d_y

        single_fit["cv_results"] = {
            k: v for k, v in cv_results_dict.items() if k != "y_val_pred_cv"
        }
        return single_fit

    Xc = np.asarray(X_cal, dtype=float)
    if Xc.ndim < 2:
        raise ValueError("X_cal must have at least 2 dimensions (samples + variables).")

    Yc = _as_2d_y(Y_cal)
    if Yc is None:
        raise ValueError("Y_cal is required and cannot be empty.")
    if Xc.shape[0] != Yc.shape[0]:
        raise ValueError("X_cal and Y_cal must have the same number of samples.")

    Xv = None if X_val is None else np.asarray(X_val, dtype=float)
    Yv = _as_2d_y(Y_val)
    if Xv is not None and Xv.ndim != Xc.ndim:
        raise ValueError("X_val must have the same number of dimensions as X_cal.")

    # Fail early with a clear message if any dataset contains NaN or Inf.
    if not np.all(np.isfinite(Xc)):
        raise ValueError("X_cal contains NaN or Inf values. Remove or impute missing values before fitting.")
    if not np.all(np.isfinite(Yc)):
        raise ValueError("Y_cal contains NaN or Inf values. Remove or impute missing values before fitting.")
    if Xv is not None and not np.all(np.isfinite(Xv)):
        raise ValueError("X_val contains NaN or Inf values. Remove or impute missing values before prediction.")
    if Yv is not None and not np.all(np.isfinite(Yv)):
        raise ValueError("Y_val contains NaN or Inf values. Remove or impute missing values before prediction.")

    method_norm = _normalize_method(multiway_method)
    n_comp = max(1, min(_safe_int(n_components, 2), Xc.shape[0]))
    expected_nway_flag = Xc.ndim - 1
    nway_flag_int = _safe_int(nway_flag, expected_nway_flag)
    if nway_flag_int != expected_nway_flag:
        raise ValueError(
            f"nway_flag ({nway_flag_int}) is inconsistent with X_cal dimensionality "
            f"({Xc.ndim}D, expected nway_flag={expected_nway_flag})."
        )

    model_payload: Dict[str, Any] = {
        "method": method_norm,
        "n_components": int(n_comp),
        "nway_flag": int(nway_flag_int),
        "tol": float(_safe_float(tol, 1e-9)),
        "max_iter": int(max(1, _safe_int(max_iter, 200))),
        "random_state": None if random_state in (None, "") else _safe_int(random_state, 0),
    }
    lod_level = _safe_float(lod_confidence_level, 0.95)
    lod_dist = _normalize_lod_distribution(lod_distribution)
    if lod_dist == "normal":
        lod_factor = _lod_factor_from_confidence_level(lod_level, dof=None)
    else:
        # Use calibration-sample dof for Student's t confidence factor.
        lod_factor = _lod_factor_from_confidence_level(lod_level, dof=max(1, int(Xc.shape[0]) - 1))
    # Canonical centering: use calibration means and apply same transform to unknown samples.
    x_mean = np.mean(Xc, axis=0, keepdims=True)
    Xc_cent = Xc - x_mean
    Xv_cent = None if Xv is None else (Xv - x_mean)

    y_mean = np.mean(Yc, axis=0, keepdims=True)
    Yc_cent = Yc - y_mean

    interferent_rank = _parse_interferent_rank(multilinear_rank, nway_flag_int, default_rank=0)
    apply_correction = bool(interferent_rank > 0)
    correction_disable_reason: Optional[str] = None
    if not apply_correction:
        correction_disable_reason = "rank_zero"

    bilinear_method = "svd"

    if apply_correction and nway_flag_int >= 3 and not _tucker_supported_for_ndim(int(Xc_cent.ndim)):
        apply_correction = False
        correction_disable_reason = "tucker_not_supported"
        emit_execution_warning(
            code="residual_multilinearization_disabled",
            text="Canonical RTL correction was disabled because Tucker decomposition is not supported for this tensor shape/environment.",
        )

    sample_shape = tuple(int(v) for v in Xc_cent.shape[1:])
    y_cal_pred = np.zeros_like(Yc, dtype=float)
    y_val_pred = None if Xv_cent is None else np.zeros((Xv_cent.shape[0], Yc.shape[1]), dtype=float)

    res_before_list: List[float] = []
    res_after_list: List[float] = []
    solver_success_count = 0
    solver_total_count = 0
    first_payload: Optional[Dict[str, Any]] = None
    _n_y = int(Yc_cent.shape[1])
    all_target_predictions: List[List[Dict[str, Any]]] = [[] for _ in range(_n_y)]
    res_before_matrix: List[List[float]] = [[] for _ in range(_n_y)]
    res_after_matrix: List[List[float]] = [[] for _ in range(_n_y)]
    target_fit_upls_list: List[Dict[str, Any]] = []
    target_cp_models_list: List[CP_PLSR] = []
    target_spatial_kron_list: List[np.ndarray] = []
    explained_variance_by_lv: Optional[Dict[str, np.ndarray]] = None

    if method_norm == "u_pls":
        Xc2d = _reshape_x_to_2d(Xc_cent)
        Xv2d = None if Xv_cent is None else _reshape_x_to_2d(Xv_cent)
        max_comp = max(1, min(Xc2d.shape[0], Xc2d.shape[1]))
        n_comp = min(n_comp, max_comp)

        target_models: List[Dict[str, Any]] = []
        ev_x_all: List[np.ndarray] = []
        ev_y_all: List[np.ndarray] = []
        for target_idx in range(Yc_cent.shape[1]):
            fit = _fit_upls1(Xc2d, Yc_cent[:, target_idx], n_comp)
            target_models.append(fit)
            target_fit_upls_list.append(fit)
            ev_x_all.append(np.asarray(fit.get("ssx_percent"), dtype=float))
            ev_y_all.append(np.asarray(fit.get("ssy_percent"), dtype=float))

            t_cal = Xc2d @ fit["x_to_t"].T
            y_cal_pred[:, target_idx] = (t_cal @ fit["v"]).reshape(-1) + float(y_mean[0, target_idx])

        explained_variance_by_lv = {
            "x_percent": np.column_stack(ev_x_all) if len(ev_x_all) > 1 else np.asarray(ev_x_all[0], dtype=float).reshape(-1, 1),
            "y_percent": np.column_stack(ev_y_all) if len(ev_y_all) > 1 else np.asarray(ev_y_all[0], dtype=float).reshape(-1, 1),
        }

        if Xv2d is not None:
            for sample_idx in range(Xv2d.shape[0]):
                x_sample = Xv2d[sample_idx, :]
                for target_idx, fit in enumerate(target_models):
                    pred = _predict_upls1_sample(
                        fit,
                        x_sample_1d=x_sample,
                        sample_shape=sample_shape,
                        nway_flag=nway_flag_int,
                        apply_correction=apply_correction,
                        interferent_rank=interferent_rank,
                        max_iter=model_payload["max_iter"],
                    )
                    y_val_pred[sample_idx, target_idx] = pred["y_corrected"] + float(y_mean[0, target_idx])
                    all_target_predictions[target_idx].append(pred)
                    res_before_matrix[target_idx].append(float(np.linalg.norm(pred["res_before"])))
                    res_after_matrix[target_idx].append(float(np.linalg.norm(pred["res_after"])))

                    if target_idx == 0:
                        solver_total_count += 1
                        if pred["solver_success"]:
                            solver_success_count += 1
                        if first_payload is None and pred.get("correction_payload") is not None:
                            first_payload = pred.get("correction_payload")

        model_payload["backend"] = "canonical.u_pls_rbl_rtl"

    else:
        if not HAS_TENSORLY:
            raise ValueError(
                "TensorLy is not available, so N-PLS (CP_PLSR) cannot be used. "
                "Install tensorly or switch to U-PLS."
            )

        rs = model_payload["random_state"]
        target_models_cp: List[CP_PLSR] = []
        target_spatial_kron: List[np.ndarray] = []

        ev_x_all_np: List[np.ndarray] = []
        ev_y_all_np: List[np.ndarray] = []
        for target_idx in range(Yc_cent.shape[1]):
            cp_model = CP_PLSR(
                n_components=n_comp,
                tol=float(model_payload["tol"]),
                n_iter_max=int(model_payload["max_iter"]),
                random_state=rs,
                verbose=False,
            )
            cp_model.fit(Xc_cent, Yc_cent[:, target_idx:target_idx + 1])
            target_models_cp.append(cp_model)
            target_spatial_kron.append(_build_cp_spatial_kron(cp_model))
            target_cp_models_list.append(cp_model)
            target_spatial_kron_list.append(target_spatial_kron[-1])

            y_cal_pred[:, target_idx] = np.asarray(cp_model.predict(Xc_cent), dtype=float).reshape(-1) + float(y_mean[0, target_idx])

            ev = _compute_npls_explained_by_lv(
                X_centered=Xc_cent,
                y_centered=Yc_cent[:, target_idx],
                n_components=n_comp,
                tol=float(model_payload["tol"]),
                max_iter=int(model_payload["max_iter"]),
                random_state=model_payload["random_state"],
            )
            ev_x_all_np.append(ev["x_percent"])
            ev_y_all_np.append(ev["y_percent"])

        if ev_x_all_np:
            explained_variance_by_lv = {
                "x_percent": np.column_stack(ev_x_all_np) if len(ev_x_all_np) > 1 else np.asarray(ev_x_all_np[0], dtype=float).reshape(-1, 1),
                "y_percent": np.column_stack(ev_y_all_np) if len(ev_y_all_np) > 1 else np.asarray(ev_y_all_np[0], dtype=float).reshape(-1, 1),
            }

        if Xv_cent is not None:
            for sample_idx in range(Xv_cent.shape[0]):
                x_sample_t = Xv_cent[sample_idx, ...]
                for target_idx, cp_model in enumerate(target_models_cp):
                    pred = _predict_npls1_sample(
                        cp_model,
                        spatial_kron=target_spatial_kron[target_idx],
                        x_sample_tensor=x_sample_t,
                        sample_shape=sample_shape,
                        nway_flag=nway_flag_int,
                        apply_correction=apply_correction,
                        interferent_rank=interferent_rank,
                        max_iter=model_payload["max_iter"],
                        y_offset=float(y_mean[0, target_idx]),
                    )
                    y_val_pred[sample_idx, target_idx] = pred["y_corrected"]
                    all_target_predictions[target_idx].append(pred)
                    res_before_matrix[target_idx].append(float(np.linalg.norm(pred["res_before"])))
                    res_after_matrix[target_idx].append(float(np.linalg.norm(pred["res_after"])))

                    if target_idx == 0:
                        solver_total_count += 1
                        if pred["solver_success"]:
                            solver_success_count += 1
                        if first_payload is None and pred.get("correction_payload") is not None:
                            first_payload = pred.get("correction_payload")

        model_payload["backend"] = "canonical.n_pls_rbl_rtl"

    y_cal_true = np.asarray(Yc, dtype=float)
    y_val_true = np.asarray(Yv, dtype=float) if Yv is not None else None

    y_cal_error = y_cal_true - np.asarray(y_cal_pred, dtype=float)
    y_val_error = None
    if y_val_true is not None and y_val_pred is not None and y_val_true.shape == y_val_pred.shape:
        y_val_error = y_val_true - y_val_pred

    # Build per-response 1:1 diagonal for scatter plots — shape (2, n_y).
    # Cal/CV diagonal (y_cv_pred included once available via CV outer path).
    _diag_x_cols = np.full((2, _n_y), np.nan, dtype=float)
    _diag_y_cols = np.full((2, _n_y), np.nan, dtype=float)
    # Validation diagonal — from val_true and val_pred.
    _val_diag_x_cols = np.full((2, _n_y), np.nan, dtype=float)
    _val_diag_y_cols = np.full((2, _n_y), np.nan, dtype=float)
    for _yi in range(_n_y):
        _dx, _dy = _build_pred_ref_diag(y_cal_true[:, _yi], y_cal_pred[:, _yi])
        if _dx is not None:
            _diag_x_cols[:, _yi] = _dx
            _diag_y_cols[:, _yi] = _dy
        _vt = y_val_true[:, _yi] if y_val_true is not None else None
        _vp = np.asarray(y_val_pred, dtype=float)[:, _yi] if y_val_pred is not None else None
        _vdx, _vdy = _build_pred_ref_diag(_vt, _vp)
        if _vdx is not None:
            _val_diag_x_cols[:, _yi] = _vdx
            _val_diag_y_cols[:, _yi] = _vdy
    pred_ref_diag_x = _diag_x_cols
    pred_ref_diag_y = _diag_y_cols
    pred_ref_diag_val_x = _val_diag_x_cols
    pred_ref_diag_val_y = _val_diag_y_cols

    canonical_mode = "none"
    if apply_correction:
        canonical_mode = "rbl" if nway_flag_int == 2 else ("rtl" if nway_flag_int >= 3 else "none")

    # Per-response calibration metrics (scalar for first response kept for text report).
    _r2_cal_all = np.asarray([_r2(y_cal_true[:, _yi], y_cal_pred[:, _yi]) for _yi in range(_n_y)], dtype=float)
    _rmse_cal_all = np.asarray([_rmse(y_cal_true[:, _yi], y_cal_pred[:, _yi]) for _yi in range(_n_y)], dtype=float)
    metrics: Dict[str, Any] = {
        "calibration": {
            "n_samples": int(y_cal_true.shape[0]),
            "r2": float(_r2_cal_all[0]),
            "rmse": float(_rmse_cal_all[0]),
            "r2_per_response": _r2_cal_all,
            "rmse_per_response": _rmse_cal_all,
        },
        "cv": None,
        "validation": None,
        "residual_multilinearization": {
            "attempted": bool(interferent_rank > 0),
            "applied": bool(apply_correction and (Xv_cent is not None)),
            "method_requested": bilinear_method,
            "method_used": "svd" if nway_flag_int == 2 else ("tucker" if nway_flag_int >= 3 else None),
            "reason": "ok" if (apply_correction and Xv_cent is not None) else ("no_unknown_samples" if apply_correction else (correction_disable_reason or "not_attempted")),
            "explained_variance": None,
            "canonical_mode": canonical_mode,
            "interferent_rank": int(interferent_rank),
            "solver_success_count": int(solver_success_count),
            "solver_total_count": int(solver_total_count),
        },
    }

    if y_val_true is not None and y_val_pred is not None and y_val_true.shape == y_val_pred.shape:
        _r2_val_all = np.asarray([_r2(y_val_true[:, _yi], y_val_pred[:, _yi]) for _yi in range(_n_y)], dtype=float)
        _rmse_val_all = np.asarray([_rmse(y_val_true[:, _yi], y_val_pred[:, _yi]) for _yi in range(_n_y)], dtype=float)
        metrics["validation"] = {
            "n_samples": int(y_val_true.shape[0]),
            "r2": float(_r2_val_all[0]),
            "rmse": float(_rmse_val_all[0]),
            "r2_per_response": _r2_val_all,
            "rmse_per_response": _rmse_val_all,
        }

    sensitivity_values: Optional[np.ndarray] = None
    concentration_sd: Optional[np.ndarray] = None
    lod_min: Optional[np.ndarray] = None
    lod_max: Optional[np.ndarray] = None
    loq_min: Optional[np.ndarray] = None
    loq_max: Optional[np.ndarray] = None
    residual_diagnostics: Optional[Dict[str, Any]] = None
    interferent_profiles: Optional[Dict[str, Any]] = None

    # Build per-sample, per-mode interferent profile arrays from target-0 predictions.
    # Structure mirrors PARAFAC SBS: (n_val, n_modes, max_rank, max_mode_len).
    _t0_preds = all_target_predictions[0] if _n_y > 0 else []
    _interf_mode_profile_factors: Optional[np.ndarray] = None
    _interf_mode_profile_axes: Optional[np.ndarray] = None
    _interf_mode_labels: Optional[List[str]] = None
    _interf_rank_labels: Optional[List[str]] = None
    _mode_dim_labels: Optional[List[str]] = None

    if len(_t0_preds) > 0 and apply_correction and interferent_rank > 0:
        _per_sample_mf: List[Optional[List[np.ndarray]]] = []
        for _pred in _t0_preds:
            _cp = _pred.get("correction_payload")
            if _cp is None:
                _per_sample_mf.append(None)
                continue
            _mf = _cp.get("mode_factors")
            if not isinstance(_mf, list) or len(_mf) == 0:
                _per_sample_mf.append(None)
            else:
                _per_sample_mf.append([np.asarray(_f, dtype=float) for _f in _mf])

        _valid_mf = [(i, mf) for i, mf in enumerate(_per_sample_mf) if mf is not None]
        if _valid_mf:
            _ref_mf = _valid_mf[0][1]
            _n_modes = len(_ref_mf)
            _n_val_int = len(_per_sample_mf)
            _max_rank = max(int(_f.shape[1]) for _f in _ref_mf)
            _max_mode_len = max(int(_f.shape[0]) for _f in _ref_mf)

            _arr = np.full((_n_val_int, _n_modes, _max_rank, _max_mode_len), np.nan, dtype=float)
            for _i, _mf in _valid_mf:
                for _m, _f in enumerate(_mf):
                    _ml = int(_f.shape[0])
                    _rk = int(_f.shape[1])
                    _arr[_i, _m, :_rk, :_ml] = _f.T
            _interf_mode_profile_factors = _arr

            _axes = np.full((_n_modes, _max_mode_len), np.nan, dtype=float)
            for _m, _f in enumerate(_ref_mf):
                _ml = int(_f.shape[0])
                if (
                    axis_n_info is not None
                    and isinstance(axis_n_info, list)
                    and (_m + 1) < len(axis_n_info)
                    and axis_n_info[_m + 1] is not None
                ):
                    _ax_src = np.asarray(axis_n_info[_m + 1], dtype=float).reshape(-1)
                    _take = min(_ml, len(_ax_src))
                    _axes[_m, :_take] = _ax_src[:_take]
                else:
                    _axes[_m, :_ml] = np.arange(1, _ml + 1, dtype=float)
            # Add a leading dummy sample dimension so the GUI can apply Val Sample
            # slicing (dim 0) uniformly across both axes and factors arrays.
            # Shape: (1, n_modes, max_mode_len) — sample index always clips to 0.
            _interf_mode_profile_axes = _axes[np.newaxis, :, :]
            _interf_mode_labels = [f"Mode {_m + 1}" for _m in range(_n_modes)]
            _interf_rank_labels = [f"Rank {_r + 1}" for _r in range(_max_rank)]
            if dim_labels is not None and isinstance(dim_labels, list) and len(dim_labels) > 1:
                _mode_dim_labels: List[str] = [str(dim_labels[_m + 1]) for _m in range(_n_modes) if (_m + 1) < len(dim_labels)]
            else:
                _mode_dim_labels = [f"Mode {_m + 1}" for _m in range(_n_modes)]

    if first_payload is not None or _interf_mode_profile_factors is not None:
        interferent_profiles = {
            "mode_profile_factors": _interf_mode_profile_factors,
            "mode_profile_axes": _interf_mode_profile_axes,
            "mode_profile_mode_labels": _interf_mode_labels,
            "mode_profile_rank_labels": _interf_rank_labels,
            "mode_dim_labels": _mode_dim_labels if _interf_mode_labels is not None else None,
            "interferent_rank": int(interferent_rank),
            "diagnostics_scope": "first_target_only",
        }

    # Per-target analytical outputs: sensitivity, LOD/LOQ, residual diagnostics.
    # Results are stacked to (n_val, n_y) after the per-target loop.
    _any_preds = any(len(all_target_predictions[t]) > 0 for t in range(_n_y))
    if _any_preds:
        i_cal = int(Xc.shape[0])
        n_features = int(np.prod(sample_shape))
        n_comp_eff = int(n_comp)
        dof_before = max(1, n_features - n_comp_eff)
        dof_after = max(1, int(np.prod([max(1, d - int(interferent_rank)) for d in sample_shape])) - n_comp_eff)

        _sensitivity_cols: List[np.ndarray] = []
        _sd_cols: List[np.ndarray] = []
        _lod_min_cols: List[np.ndarray] = []
        _lod_max_cols: List[np.ndarray] = []
        _loq_min_cols: List[np.ndarray] = []
        _loq_max_cols: List[np.ndarray] = []
        _sr_before_cols: List[np.ndarray] = []
        _sr_after_cols: List[np.ndarray] = []
        _norm_before_cols: List[np.ndarray] = []
        _norm_after_cols: List[np.ndarray] = []
        _calibration_fit_residual_0 = float("nan")

        mode_factors_for_projection: Optional[List[np.ndarray]] = None
        if first_payload is not None and isinstance(first_payload.get("mode_factors"), list):
            mode_factors_for_projection = [np.asarray(f, dtype=float) for f in first_payload.get("mode_factors")]

        for target_idx in range(_n_y):
            preds_t = all_target_predictions[target_idx]
            if not preds_t:
                _sensitivity_cols.append(np.full(0, np.nan))
                _sd_cols.append(np.full(0, np.nan))
                _lod_min_cols.append(np.full(0, np.nan))
                _lod_max_cols.append(np.full(0, np.nan))
                _loq_min_cols.append(np.full(0, np.nan))
                _loq_max_cols.append(np.full(0, np.nan))
                _sr_before_cols.append(np.full(0, np.nan))
                _sr_after_cols.append(np.full(0, np.nan))
                _norm_before_cols.append(np.asarray(res_before_matrix[target_idx], dtype=float))
                _norm_after_cols.append(np.asarray(res_after_matrix[target_idx], dtype=float))
                continue

            if method_norm == "u_pls" and target_idx < len(target_fit_upls_list):
                _fit_t = target_fit_upls_list[target_idx]
                loadings_matrix = np.asarray(_fit_t["P"], dtype=float)
                latent_to_y = np.asarray(_fit_t["v"], dtype=float)
                t_cal_t = np.asarray(_fit_t["T"], dtype=float)
                y_res_cal_t = np.asarray(_fit_t["y_res_cal"], dtype=float).reshape(-1)
                x_res_cal_t = np.asarray(_fit_t["x_res_cal"], dtype=float)
                cal_fit_res = float(np.linalg.norm(x_res_cal_t) / np.sqrt(max(1.0, (n_features - n_comp_eff) * i_cal)))
            elif method_norm == "n_pls" and target_idx < len(target_cp_models_list):
                _cp_t = target_cp_models_list[target_idx]
                _sk_t = target_spatial_kron_list[target_idx]
                loadings_matrix = np.asarray(_sk_t, dtype=float)
                latent_to_y = (
                    np.asarray(_cp_t.coef_, dtype=float) @ np.asarray(_cp_t.Y_factors[1], dtype=float).T
                ).reshape(-1)
                t_cal_t = np.asarray(_cp_t.X_factors[0], dtype=float)
                y_cal_hat_cent_t = np.asarray(_cp_t.predict(Xc_cent), dtype=float).reshape(-1)
                y_res_cal_t = np.asarray(Yc_cent[:, target_idx], dtype=float).reshape(-1) - y_cal_hat_cent_t
                x_cal_flat_t = Xc_cent.reshape(i_cal, -1)
                x_hat_flat_t = t_cal_t @ loadings_matrix.T
                cal_fit_res = float(
                    np.linalg.norm(x_cal_flat_t - x_hat_flat_t) / np.sqrt(max(1.0, (n_features - n_comp_eff) * i_cal))
                )
            else:
                loadings_matrix = np.asarray([])
                latent_to_y = np.asarray([])
                t_cal_t = np.asarray([])
                y_res_cal_t = np.asarray([])
                cal_fit_res = float("nan")

            if target_idx == 0:
                _calibration_fit_residual_0 = cal_fit_res

            sensitivity_base_t = _compute_sensitivity_from_projection(
                loadings_matrix=loadings_matrix,
                latent_to_y=latent_to_y,
                sample_shape=sample_shape,
                interferent_mode_factors=None,
            )

            y_cent_t = np.asarray(Yc_cent[:, target_idx], dtype=float).reshape(-1)
            mx_t = float(y_mean[0, target_idx])
            tpinv_t = np.linalg.pinv(np.asarray(t_cal_t, dtype=float).T) if np.size(t_cal_t) else None
            sr_c_t = float(np.linalg.norm(y_res_cal_t) / np.sqrt(max(1, y_res_cal_t.size))) if y_res_cal_t.size else float("nan")

            hcal_vals_t: List[float] = []
            if np.size(t_cal_t):
                for i_idx in range(t_cal_t.shape[0]):
                    hcal = float(np.linalg.norm(tpinv_t @ t_cal_t[i_idx, :].reshape(-1, 1)) ** 2)
                    if abs(mx_t) > 1e-12:
                        huni_t = float(mx_t ** 2 / (np.sum(y_cent_t ** 2) + 1e-12))
                        hcal_vals_t.append(float(hcal + huni_t * (1.0 - (y_cent_t[i_idx] / mx_t) ** 2)))
                    else:
                        hcal_vals_t.append(float(hcal))
            huni_t = float(mx_t ** 2 / (np.sum(y_cent_t ** 2) + 1e-12)) if abs(mx_t) > 1e-12 else 0.0
            hmax_t = float(1.0 / max(1, i_cal) + (max(hcal_vals_t) if hcal_vals_t else 0.0))
            hmin_t = float(1.0 / max(1, i_cal) + huni_t)

            _sens_list: List[float] = []
            _sd_list: List[float] = []
            _lmin_list: List[float] = []
            _lmax_list: List[float] = []
            _qmin_list: List[float] = []
            _qmax_list: List[float] = []
            _srb_list: List[float] = []
            _sra_list: List[float] = []

            for pred in preds_t:
                beta_used = np.asarray(pred["beta_corrected"] if apply_correction else pred["beta_base"], dtype=float)
                sr_before = float(np.linalg.norm(pred["res_before"]) / np.sqrt(dof_before))
                sr_after = float(np.linalg.norm(pred["res_after"]) / np.sqrt(dof_after if apply_correction else dof_before))
                sr_used = sr_after if apply_correction else sr_before

                if apply_correction and mode_factors_for_projection is not None:
                    sensitivity_used = _compute_sensitivity_from_projection(
                        loadings_matrix=loadings_matrix,
                        latent_to_y=latent_to_y,
                        sample_shape=sample_shape,
                        interferent_mode_factors=mode_factors_for_projection,
                    )
                else:
                    sensitivity_used = sensitivity_base_t

                lever = float("nan")
                if tpinv_t is not None:
                    h_vec = tpinv_t @ beta_used.reshape(-1, 1)
                    lever = float(np.linalg.norm(h_vec) ** 2 + 1.0 / max(1, i_cal))

                sen_abs = max(abs(float(sensitivity_used)), 1e-12)
                sd_val = float(np.sqrt(max(0.0, lever * sr_c_t ** 2 + lever * sr_used ** 2 / sen_abs ** 2 + sr_used ** 2 / sen_abs ** 2)))
                lmin, lmax, qmin, qmax = _compute_lod_loq_bounds(
                    sr=sr_used,
                    sensitivity=sensitivity_used,
                    sr_c=sr_c_t,
                    hmin=hmin_t,
                    hmax=hmax_t,
                    lod_confidence_factor=lod_factor,
                )

                _sens_list.append(float(sensitivity_used))
                _sd_list.append(sd_val)
                _lmin_list.append(lmin)
                _lmax_list.append(lmax)
                _qmin_list.append(qmin)
                _qmax_list.append(qmax)
                _srb_list.append(sr_before)
                _sra_list.append(sr_after)

            _sensitivity_cols.append(np.asarray(_sens_list, dtype=float))
            _sd_cols.append(np.asarray(_sd_list, dtype=float))
            _lod_min_cols.append(np.asarray(_lmin_list, dtype=float))
            _lod_max_cols.append(np.asarray(_lmax_list, dtype=float))
            _loq_min_cols.append(np.asarray(_qmin_list, dtype=float))
            _loq_max_cols.append(np.asarray(_qmax_list, dtype=float))
            _sr_before_cols.append(np.asarray(_srb_list, dtype=float))
            _sr_after_cols.append(np.asarray(_sra_list, dtype=float))
            _norm_before_cols.append(np.asarray(res_before_matrix[target_idx], dtype=float))
            _norm_after_cols.append(np.asarray(res_after_matrix[target_idx], dtype=float))

        def _stack_cols(cols: List[np.ndarray]) -> Optional[np.ndarray]:
            if not cols or all(c.size == 0 for c in cols):
                return None
            return np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)

        sensitivity_values = _stack_cols(_sensitivity_cols)
        concentration_sd = _stack_cols(_sd_cols)
        lod_min = _stack_cols(_lod_min_cols)
        lod_max = _stack_cols(_lod_max_cols)
        loq_min = _stack_cols(_loq_min_cols)
        loq_max = _stack_cols(_loq_max_cols)

        residual_diagnostics = {
            "sample_residual_norm_before": _stack_cols(_norm_before_cols),
            "sample_residual_norm_after": _stack_cols(_norm_after_cols),
            "sample_residual_std_before": _stack_cols(_sr_before_cols),
            "sample_residual_std_after": _stack_cols(_sr_after_cols),
            "calibration_fit_residual": _calibration_fit_residual_0,
        }

        metrics["analytical"] = {
            "sensitivity": sensitivity_values,
            "concentration_sd": concentration_sd,
            "lod_min": lod_min,
            "lod_max": lod_max,
            "loq_min": loq_min,
            "loq_max": loq_max,
            "lod_confidence_factor": float(lod_factor),
            "lod_confidence_level": float(lod_level),
            "lod_distribution": str(lod_dist),
            "loq_factor": float(lod_factor),
            "residual_diagnostics": residual_diagnostics,
            "explained_variance_by_lv": explained_variance_by_lv,
            "interferent_profiles": interferent_profiles,
        }

    residual_ml_payload: Optional[Dict[str, Any]] = None
    residual_ml_scores: Optional[np.ndarray] = None
    residual_ml_loadings: Optional[Any] = None
    residual_ml_core: Optional[np.ndarray] = None
    residual_ml_reconstructed: Optional[np.ndarray] = None

    if solver_total_count > 0:
        _t0_norms_before = np.asarray(res_before_matrix[0], dtype=float) if res_before_matrix else np.asarray([], dtype=float)
        _t0_norms_after = np.asarray(res_after_matrix[0], dtype=float) if res_after_matrix else np.asarray([], dtype=float)
        residual_ml_payload = {
            "canonical_mode": canonical_mode,
            "interferent_rank": int(interferent_rank),
            "sample_residual_norm_before": _t0_norms_before,
            "sample_residual_norm_after": _t0_norms_after,
            "solver_success_count": int(solver_success_count),
            "solver_total_count": int(solver_total_count),
        }

    if first_payload is not None:
        residual_ml_scores = first_payload.get("scores")
        residual_ml_loadings = first_payload.get("loadings")
        residual_ml_core = first_payload.get("core")
        residual_ml_reconstructed = first_payload.get("reconstructed")

    # Calibration LV scores (t_scores): (n_cal, n_comp, n_y) — one model per response.
    _t_score_cols: List[np.ndarray] = []
    if method_norm == "u_pls":
        for _fit_t in target_fit_upls_list:
            _t_score_cols.append(np.asarray(_fit_t["T"], dtype=float))
    elif method_norm == "n_pls":
        for _cp_t in target_cp_models_list:
            _t_score_cols.append(np.asarray(_cp_t.X_factors[0], dtype=float))
    if _t_score_cols:
        if len(_t_score_cols) == 1:
            model_payload["t_scores"] = _t_score_cols[0][:, :, np.newaxis]
        else:
            model_payload["t_scores"] = np.stack(_t_score_cols, axis=2)

    # LV labels for dynamic axis titling on the LV Scores diagnostics page.
    model_payload["lv_labels"] = [f"LV {_i + 1}" for _i in range(int(n_comp))]

    # X mode factor profiles for diagnostics:
    #   N-PLS: CP factor vectors per spatial mode (X_factors[1:] per response).
    #   U-PLS: unfolded X weight vectors (Wn columns per response).
    # Result shape: (n_modes, n_comp, n_y, max_mode_len)
    _mfp_list: List[List[np.ndarray]] = []
    _mfp_axes: List[np.ndarray] = []
    if method_norm == "n_pls":
        _n_spatial = nway_flag_int
        for _m in range(_n_spatial):
            _mode_resp_factors: List[np.ndarray] = []
            for _cp_t in target_cp_models_list:
                if (_m + 1) < len(_cp_t.X_factors):
                    _mode_resp_factors.append(np.asarray(_cp_t.X_factors[_m + 1], dtype=float))
            if _mode_resp_factors:
                _mfp_list.append(_mode_resp_factors)
                _dim_len = _mode_resp_factors[0].shape[0]
                if (
                    axis_n_info is not None and isinstance(axis_n_info, list)
                    and (_m + 1) < len(axis_n_info) and axis_n_info[_m + 1] is not None
                ):
                    _mfp_axes.append(np.asarray(axis_n_info[_m + 1], dtype=float).reshape(-1))
                else:
                    _mfp_axes.append(np.arange(1, _dim_len + 1, dtype=float))
    elif method_norm == "u_pls":
        _upls_wn_cols = [np.asarray(_fit_t["Wn"], dtype=float) for _fit_t in target_fit_upls_list]
        if _upls_wn_cols:
            _mfp_list.append(_upls_wn_cols)
            _dim_len = _upls_wn_cols[0].shape[0]
            if (
                axis_n_info is not None and isinstance(axis_n_info, list)
                and len(axis_n_info) > 1 and axis_n_info[1] is not None
            ):
                _mfp_axes.append(np.asarray(axis_n_info[1], dtype=float).reshape(-1))
            else:
                _mfp_axes.append(np.arange(1, _dim_len + 1, dtype=float))
    if _mfp_list:
        _n_modes_mf = len(_mfp_list)
        _n_comp_eff_mf = int(n_comp)
        _n_y_eff_mf = len(_mfp_list[0])
        _max_ml_mf = max(rf[0].shape[0] for rf in _mfp_list)
        _mfp_arr = np.full((_n_modes_mf, _n_comp_eff_mf, _n_y_eff_mf, _max_ml_mf), np.nan, dtype=float)
        for _m_mf, _resp_cols in enumerate(_mfp_list):
            _ml_mf = _resp_cols[0].shape[0]
            _mc_mf = _resp_cols[0].shape[1]
            for _r_mf, _rf_mf in enumerate(_resp_cols):
                _mfp_arr[_m_mf, :_mc_mf, _r_mf, :_ml_mf] = np.asarray(_rf_mf, dtype=float).T
        _mfp_axes_arr = np.full((_n_modes_mf, _max_ml_mf), np.nan, dtype=float)
        for _m_mf, _ax_mf in enumerate(_mfp_axes):
            _ml_mf = min(len(_ax_mf), _max_ml_mf)
            _mfp_axes_arr[_m_mf, :_ml_mf] = _ax_mf[:_ml_mf]
        model_payload["x_mode_factors"] = _mfp_arr
        model_payload["x_mode_factor_axes"] = _mfp_axes_arr
        if dim_labels is not None and isinstance(dim_labels, list) and len(dim_labels) > 1:
            _mfp_dim_labels: List[str] = [
                str(dim_labels[_m_mf + 1]) for _m_mf in range(_n_modes_mf)
                if (_m_mf + 1) < len(dim_labels)
            ]
            if len(_mfp_dim_labels) < _n_modes_mf:
                _mfp_dim_labels += [f"Mode {_m_mf + 1}" for _m_mf in range(len(_mfp_dim_labels), _n_modes_mf)]
        elif method_norm == "u_pls":
            _mfp_dim_labels = ["Feature (Unfolded)"]
        else:
            _mfp_dim_labels = [f"Mode {_m_mf + 1}" for _m_mf in range(_n_modes_mf)]
        model_payload["x_mode_factor_dim_labels"] = _mfp_dim_labels

    # U-PLS X loadings (P) and weights (Wn) for PLS-style diagnostics page.
    # Shape: (n_features, n_comp, n_y) — navigate by LV (dim 1) and Response (dim 2).
    if method_norm == "u_pls" and target_fit_upls_list:
        _upls_P_list = [np.asarray(_ft["P"], dtype=float) for _ft in target_fit_upls_list]
        _upls_Wn_list = [np.asarray(_ft["Wn"], dtype=float) for _ft in target_fit_upls_list]
        if _upls_P_list:
            _n_feat_lp = _upls_P_list[0].shape[0]
            _n_comp_lp = int(n_comp)
            _n_y_lp = len(_upls_P_list)
            _lp_arr = np.full((_n_feat_lp, _n_comp_lp, _n_y_lp), np.nan, dtype=float)
            _lw_arr = np.full((_n_feat_lp, _n_comp_lp, _n_y_lp), np.nan, dtype=float)
            for _r_lp, (_P_r, _Wn_r) in enumerate(zip(_upls_P_list, _upls_Wn_list)):
                _nc_r = min(_P_r.shape[1], _n_comp_lp)
                _lp_arr[:, :_nc_r, _r_lp] = _P_r[:, :_nc_r]
                _lw_arr[:, :_nc_r, _r_lp] = _Wn_r[:, :_nc_r]
            model_payload["x_loadings_upls"] = _lp_arr
            model_payload["x_weights_upls"] = _lw_arr

    # EJCR packed payload: one variable contains all ellipses + metadata per graph.
    _ejcr_n_pts = 100
    _ejcr_n_path = _ejcr_n_pts * 2 + 1
    _Y_cal_2d = np.asarray(y_cal_true, dtype=float)
    if _Y_cal_2d.ndim == 1:
        _Y_cal_2d = _Y_cal_2d.reshape(-1, 1)
    _ejcr_n_col = _Y_cal_2d.shape[1]
    _ejcr_levels = ("90", "95", "99")
    _ejcr_level_to_idx = {lvl: i for i, lvl in enumerate(_ejcr_levels)}
    _ejcr_item_count = len(_ejcr_levels) + 1

    def _new_packed_ejcr(color: str) -> Dict[str, Any]:
        x_paths = np.full((_ejcr_item_count, _ejcr_n_col, _ejcr_n_path), np.nan, dtype=float)
        y_paths = np.full((_ejcr_item_count, _ejcr_n_col, _ejcr_n_path), np.nan, dtype=float)
        fit_slope = np.full((1, _ejcr_n_col), np.nan, dtype=float)
        fit_intercept = np.full((1, _ejcr_n_col), np.nan, dtype=float)
        x_paths[len(_ejcr_levels), :, 0] = 1.0
        y_paths[len(_ejcr_levels), :, 0] = 0.0
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

    ejcr_cal = _new_packed_ejcr("steelblue")
    ejcr_val = _new_packed_ejcr("darkorange")

    try:
        from chemometrics.ejcr_analysis import compute_ejcr as _compute_ejcr

        def _fill_ejcr(y_ref_1d, y_pred_1d, col: int, packed: Dict[str, Any]) -> None:
            valid = np.isfinite(y_ref_1d) & np.isfinite(y_pred_1d)
            if valid.sum() < 3:
                return
            r = _compute_ejcr(y_ref_1d[valid], y_pred_1d[valid], n_points=_ejcr_n_pts)
            packed["fit_slope"][0, col] = r["slope"]
            packed["fit_intercept"][0, col] = r["intercept"]
            for ell in r["ellipses"]:
                pct = str(ell.get("confidence_pct", "")).strip()
                idx = _ejcr_level_to_idx.get(pct)
                if idx is None:
                    continue
                es = np.asarray(ell["ellipse_slope"], dtype=float)
                ei = np.asarray(ell["ellipse_intercept"], dtype=float)
                k = min(len(es), _ejcr_n_path)
                packed["x_paths"][idx, col, :k] = es[:k]
                packed["y_paths"][idx, col, :k] = ei[:k]

        _y_cal_pred_2d = np.asarray(y_cal_pred, dtype=float)
        if _y_cal_pred_2d.ndim == 1:
            _y_cal_pred_2d = _y_cal_pred_2d.reshape(-1, 1)
        for _col in range(_ejcr_n_col):
            _fill_ejcr(_Y_cal_2d[:, _col], _y_cal_pred_2d[:, _col], _col, ejcr_cal)

        if y_val_true is not None and y_val_pred is not None:
            _Y_val_2d = np.asarray(y_val_true, dtype=float)
            if _Y_val_2d.ndim == 1:
                _Y_val_2d = _Y_val_2d.reshape(-1, 1)
            _y_val_pred_2d = np.asarray(y_val_pred, dtype=float)
            if _y_val_pred_2d.ndim == 1:
                _y_val_pred_2d = _y_val_pred_2d.reshape(-1, 1)
            _nc_val = min(_Y_val_2d.shape[1], _y_val_pred_2d.shape[1], _ejcr_n_col)
            for _col in range(_nc_val):
                _fill_ejcr(_Y_val_2d[:, _col], _y_val_pred_2d[:, _col], _col, ejcr_val)
    except Exception:
        pass

    return {
        "y_cal_pred": np.asarray(y_cal_pred, dtype=float),
        "y_val_pred": None if y_val_pred is None else np.asarray(y_val_pred, dtype=float),
        "y_cv_pred": None,
        "y_cal_true": y_cal_true,
        "y_val_true": y_val_true,
        "y_cal_error": y_cal_error,
        "y_val_error": y_val_error,
        "y_cv_error": None,
        "pred_ref_diag_x": pred_ref_diag_x,
        "pred_ref_diag_y": pred_ref_diag_y,
        "pred_ref_diag_val_x": pred_ref_diag_val_x,
        "pred_ref_diag_val_y": pred_ref_diag_val_y,
        "metrics": metrics,
        "model": model_payload,
        "sensitivity": sensitivity_values,
        "concentration_sd": concentration_sd,
        "lod_min": lod_min,
        "lod_max": lod_max,
        "loq_min": loq_min,
        "loq_max": loq_max,
        "residual_diagnostics": residual_diagnostics,
        "explained_variance_by_lv": explained_variance_by_lv,
        "interferent_profiles": interferent_profiles,
        "residual_multilinearization": residual_ml_payload,
        "residual_multilinearization_scores": residual_ml_scores,
        "residual_multilinearization_loadings": residual_ml_loadings,
        "residual_multilinearization_core": residual_ml_core,
        "residual_multilinearization_reconstructed": residual_ml_reconstructed,
        "cv_results": None,
        "npls_report": _build_npls_report(_n_y, metrics, model_payload, y_labels),
        "ejcr_cal": ejcr_cal,
        "ejcr_val": ejcr_val,
    }


_NPLS_RETURN_ORDER: Tuple[str, ...] = (
    "y_cal_pred",
    "y_val_pred",
    "y_cv_pred",
    "y_cal_true",
    "y_val_true",
    "y_cal_error",
    "y_val_error",
    "y_cv_error",
    "pred_ref_diag_x",
    "pred_ref_diag_y",
    "pred_ref_diag_val_x",
    "pred_ref_diag_val_y",
    "metrics",
    "model",
    "sensitivity",
    "concentration_sd",
    "lod_min",
    "lod_max",
    "loq_min",
    "loq_max",
    "residual_diagnostics",
    "explained_variance_by_lv",
    "interferent_profiles",
    "residual_multilinearization",
    "residual_multilinearization_scores",
    "residual_multilinearization_loadings",
    "residual_multilinearization_core",
    "residual_multilinearization_reconstructed",
    "cv_results",
    "npls_report",
    "ejcr_cal",
    "ejcr_val",
)


def npls_analysis_standard(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:
    """Adapter for the app execution pipeline: return outputs as an ordered tuple.

    The core npls_analysis API keeps its dict return for internal / recursive use.
    """
    result = npls_analysis(*args, **kwargs)
    if not isinstance(result, dict):
        return (result,)
    return tuple(result.get(key) for key in _NPLS_RETURN_ORDER)
