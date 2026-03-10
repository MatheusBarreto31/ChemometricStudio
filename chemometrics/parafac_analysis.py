"""PARAFAC (CP decomposition) analysis with optional calibration and CV support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import inspect
import itertools
import os

import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from tensorly.solvers.admm import admm
from tensorly.tenalg import khatri_rao
from tensorly.tenalg.proximal import proximal_operator

try:
    from tensorly.decomposition import constrained_parafac
    HAS_CONSTRAINED_PARAFAC = True
except Exception:
    HAS_CONSTRAINED_PARAFAC = False

try:
    from execution_reporting import emit_execution_message, emit_execution_warning
except ImportError:
    def emit_execution_message(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        return

    def emit_execution_warning(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        return

try:
    from chemometrics.cv_pipeline import CVConfig, CVPipeline
    HAS_CV = True
except ImportError:
    CVConfig = Any  # type: ignore
    HAS_CV = False

try:
    from chemometrics.input_parsing import parse_numeric_spec
except ImportError:
    def parse_numeric_spec(raw_value: Any) -> List[float]:
        if raw_value is None:
            return []
        if isinstance(raw_value, (int, float)):
            return [float(raw_value)]
        out: List[float] = []
        for token in str(raw_value).replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            out.append(float(token))
        return out


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


def _safe_optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return int(float(text))
    except Exception:
        return default


def _normalize_missing_constrained_solver(value: Any) -> str:
    mode = str(value).strip().lower() if value is not None else "em"
    allowed = {"em", "em_adaptive", "weighted_ao_admm"}
    return mode if mode in allowed else "em"


def _as_2d_y(y: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if y is None:
        return None
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _normalize_bool_list(values: Any, count: int) -> List[bool]:
    if isinstance(values, list):
        raw = values
    elif isinstance(values, str):
        raw = [v.strip() for v in values.split(",")]
    else:
        raw = []

    out: List[bool] = []
    for v in raw[:count]:
        if isinstance(v, bool):
            out.append(v)
        else:
            out.append(str(v).strip().lower() in {"1", "true", "yes", "y"})
    if len(out) < count:
        out.extend([False] * (count - len(out)))
    return out


def _coerce_scalar_token(token: Any) -> Any:
    if isinstance(token, (bool, int, float)) or token is None:
        return token
    text = str(token).strip()
    low = text.lower()
    if low in {"true", "yes", "y", "1"}:
        return True
    if low in {"false", "no", "n", "0"}:
        return False
    try:
        if "." in text or "e" in low:
            return float(text)
        return int(text)
    except Exception:
        return text


def _build_tensorly_constraints(
    n_modes: int,
    constraint_non_negative: Any = None,
    constraint_l1_reg: Any = None,
    constraint_l2_reg: Any = None,
    constraint_l2_square_reg: Any = None,
    constraint_unimodality: Any = None,
    constraint_normalize: Any = None,
    constraint_simplex: Any = None,
    constraint_normalized_sparsity: Any = None,
    constraint_soft_sparsity: Any = None,
    constraint_smoothness: Any = None,
    constraint_monotonicity: Any = None,
    constraint_hard_sparsity: Any = None,
) -> Dict[str, Any]:
    mode_flags: Dict[str, List[bool]] = {
        "non_negative": _normalize_bool_list(constraint_non_negative, n_modes),
        "l1_reg": _normalize_bool_list(constraint_l1_reg, n_modes),
        "l2_reg": _normalize_bool_list(constraint_l2_reg, n_modes),
        "l2_square_reg": _normalize_bool_list(constraint_l2_square_reg, n_modes),
        "unimodality": _normalize_bool_list(constraint_unimodality, n_modes),
        "normalize": _normalize_bool_list(constraint_normalize, n_modes),
        "simplex": _normalize_bool_list(constraint_simplex, n_modes),
        "normalized_sparsity": _normalize_bool_list(constraint_normalized_sparsity, n_modes),
        "soft_sparsity": _normalize_bool_list(constraint_soft_sparsity, n_modes),
        "smoothness": _normalize_bool_list(constraint_smoothness, n_modes),
        "monotonicity": _normalize_bool_list(constraint_monotonicity, n_modes),
        "hard_sparsity": _normalize_bool_list(constraint_hard_sparsity, n_modes),
    }

    mode_defaults: Dict[str, Any] = {
        "non_negative": True,
        "l1_reg": 1.0,
        "l2_reg": 1.0,
        "l2_square_reg": 1.0,
        "unimodality": True,
        "normalize": True,
        "simplex": 1.0,
        "normalized_sparsity": 0.5,
        "soft_sparsity": 1.0,
        "smoothness": 1.0,
        "monotonicity": True,
        "hard_sparsity": 1,
    }

    out: Dict[str, Any] = {}
    for key, flags in mode_flags.items():
        if not any(bool(v) for v in flags):
            continue
        enabled_value = mode_defaults[key]
        out[key] = [enabled_value if bool(flag) else None for flag in flags]
    return out


def _sanitize_tensorly_constraints(constraints: Dict[str, Any], n_modes: int) -> Dict[str, Any]:
    """Drop inactive per-mode constraints to avoid TensorLy None-handling issues."""
    if not constraints:
        return {}

    bool_keys = {"non_negative", "unimodality", "normalize", "monotonicity"}
    out: Dict[str, Any] = {}

    for key, value in constraints.items():
        if isinstance(value, list):
            active: Dict[int, Any] = {}
            for mode, entry in enumerate(value[:n_modes]):
                if entry is None:
                    continue
                if key in bool_keys:
                    if bool(entry):
                        active[mode] = True
                else:
                    active[mode] = entry
            if active:
                out[key] = active
        else:
            if value is None:
                continue
            if key in bool_keys and not bool(value):
                continue
            out[key] = value

    return out


def _load_profile_matrix(path: str, expected_rows: int, n_components: int) -> np.ndarray:
    if not path:
        raise ValueError("Empty profile path.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profile file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        # Let numpy infer whitespace or comma delimiters.
        try:
            arr = np.loadtxt(path, delimiter=",")
        except Exception:
            arr = np.loadtxt(path)

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != expected_rows:
            raise ValueError(
                f"Profile {path} has length {arr.shape[0]}, expected {expected_rows}."
            )
        arr = arr.reshape(expected_rows, 1)

    if arr.shape[0] != expected_rows:
        raise ValueError(
            f"Profile {path} has {arr.shape[0]} rows, expected {expected_rows}."
        )

    if arr.shape[1] == 1 and n_components > 1:
        arr = np.repeat(arr, n_components, axis=1)
    elif arr.shape[1] != n_components:
        raise ValueError(
            f"Profile {path} has {arr.shape[1]} columns, expected {n_components}."
        )

    return arr


def _modewise_preprocess(
    X: np.ndarray,
    center_modes: Sequence[bool],
    normalize_modes: Sequence[bool],
) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    Xp = np.asarray(X, dtype=float).copy()
    mode_stats: List[Dict[str, np.ndarray]] = []
    n_modes = Xp.ndim

    for mode in range(n_modes):
        unfold = tl.unfold(Xp, mode)
        stat: Dict[str, np.ndarray] = {}

        if center_modes[mode]:
            mean = np.nanmean(unfold, axis=1, keepdims=True)
            mean = np.where(np.isfinite(mean), mean, 0.0)
            unfold = unfold - mean
            stat["mean"] = mean

        if normalize_modes[mode]:
            std = np.nanstd(unfold, axis=1, keepdims=True)
            std = np.where((~np.isfinite(std)) | (std <= 0), 1.0, std)
            unfold = unfold / std
            stat["std"] = std

        mode_stats.append(stat)
        Xp = tl.fold(unfold, mode, Xp.shape)

    return np.asarray(Xp, dtype=float), mode_stats


def _component_reconstruction_tensor(weights: np.ndarray, factors: Sequence[np.ndarray]) -> np.ndarray:
    full_shape = tuple(f.shape[0] for f in factors)
    r = len(weights)
    out = np.zeros((r,) + full_shape, dtype=float)

    for comp in range(r):
        comp_tensor = weights[comp]
        for mode, factor in enumerate(factors):
            vec = factor[:, comp]
            shape = [1] * len(full_shape)
            shape[mode] = full_shape[mode]
            comp_tensor = comp_tensor * vec.reshape(shape)
        out[comp] = comp_tensor

    return out


def _instrumental_profiles(weights: np.ndarray, factors: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    if len(factors) <= 1:
        return None
    return _component_reconstruction_tensor(weights, factors[1:])


def _core_consistency(X: np.ndarray, weights: np.ndarray, factors: Sequence[np.ndarray]) -> float:
    if X.ndim < 3:
        return float("nan")

    pinvs = [np.linalg.pinv(np.asarray(f, dtype=float)) for f in factors]
    core = tl.tenalg.multi_mode_dot(np.asarray(X, dtype=float), pinvs, modes=list(range(X.ndim)))
    r = len(weights)
    ideal = np.zeros((r,) * X.ndim, dtype=float)
    for i in range(r):
        ideal[(i,) * X.ndim] = 1.0

    num = float(np.sum((core - ideal) ** 2))
    den = float(np.sum(ideal ** 2)) + 1e-12
    return float(max(0.0, 100.0 * (1.0 - num / den)))


def _explained_variance(X: np.ndarray, residual: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is None:
        sst = float(np.nansum(np.asarray(X, dtype=float) ** 2))
        ssr = float(np.nansum(np.asarray(residual, dtype=float) ** 2))
    else:
        m = np.asarray(mask, dtype=bool)
        x_obs = np.where(m, np.asarray(X, dtype=float), np.nan)
        r_obs = np.where(m, np.asarray(residual, dtype=float), np.nan)
        sst = float(np.nansum(x_obs ** 2))
        ssr = float(np.nansum(r_obs ** 2))

    if sst <= 0:
        return 0.0
    return float(max(0.0, 100.0 * (1.0 - ssr / sst)))


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float).reshape(-1)
    bv = np.asarray(b, dtype=float).reshape(-1)
    denom = (np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-12
    c = float(np.dot(av, bv) / denom)
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _parse_mapping(mapping_text: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if not mapping_text:
        return mapping
    for token in str(mapping_text).replace(";", ",").split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        a, b = token.split(":", 1)
        comp = int(a.strip()) - 1
        ycol = int(b.strip()) - 1
        if comp >= 0 and ycol >= 0:
            mapping[comp] = ycol
    return mapping


def _normalize_str_sequence(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(v).strip() for v in values]
    text = str(values).strip()
    if not text:
        return []
    return [v.strip() for v in text.replace(";", ",").split(",")]


def _expand_profile_mode_settings(
    profile_paths: Any,
    profile_usage: Any,
    n_modes: int,
) -> Tuple[List[str], List[str]]:
    paths_raw = _normalize_str_sequence(profile_paths)
    usage_raw = [u.lower() for u in _normalize_str_sequence(profile_usage)]

    # New UI provides one value per non-sample mode (nway_flag); legacy models may
    # still provide one per full tensor mode (including sample mode at index 0).
    if len(paths_raw) == max(0, n_modes - 1):
        paths = [""] + paths_raw
    else:
        paths = list(paths_raw)

    if len(usage_raw) == max(0, n_modes - 1):
        usage = ["none"] + usage_raw
    else:
        usage = list(usage_raw)

    if len(paths) < n_modes:
        paths.extend([""] * (n_modes - len(paths)))
    if len(usage) < n_modes:
        usage.extend(["none"] * (n_modes - len(usage)))

    return paths[:n_modes], usage[:n_modes]


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
    out = _parse_mapping(text_input)
    return out, True, len(out) == 0


def _auto_mapping(scores_a: np.ndarray, Y: np.ndarray) -> Dict[int, int]:
    n_comp = scores_a.shape[1]
    n_y = Y.shape[1]
    if n_comp <= 0 or n_y <= 0:
        return {}

    # Map each component to its best Y column independently, so all components are reported.
    mapping: Dict[int, int] = {}
    for c in range(n_comp):
        best_y: Optional[int] = None
        best_score = -np.inf

        for y in range(n_y):
            fit = _fit_linear_1d(scores_a[:, c], Y[:, y])
            r2 = _safe_float(fit.get("metrics", {}).get("R2"), default=-np.inf)
            n_used = _safe_float(fit.get("metrics", {}).get("n_samples_used"), default=0.0)
            if (not np.isfinite(r2)) or n_used < 2:
                continue
            if r2 > best_score:
                best_score = float(r2)
                best_y = int(y)

        if best_y is not None:
            mapping[int(c)] = int(best_y)

    return mapping


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
            "metrics": {
                "R2": float("nan"),
                "RMSEP": float("nan"),
                "n_samples_used": int(n_valid),
            },
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
        "metrics": {
            "R2": r2,
            "RMSEP": rmsep,
            "n_samples_used": int(n_valid),
        },
    }


def _project_scores_mode_a(
    X: np.ndarray,
    weights: np.ndarray,
    factors: Sequence[np.ndarray],
) -> np.ndarray:
    n_samples = X.shape[0]
    rank = len(weights)
    scores = np.zeros((n_samples, rank), dtype=float)

    if len(factors) < 2:
        return scores

    kr = khatri_rao([np.asarray(f, dtype=float) for f in factors[1:]])
    design = kr * weights.reshape(1, -1)

    for i in range(n_samples):
        x_i = np.asarray(X[i], dtype=float).reshape(-1)
        valid = np.isfinite(x_i)
        if np.count_nonzero(valid) < rank:
            continue
        A = design[valid, :]
        b = x_i[valid]
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        scores[i] = coef

    return scores


def _initial_cp_factors(
    X_filled: np.ndarray,
    rank: int,
    init: Any,
    random_state: Optional[int],
) -> List[np.ndarray]:
    shape = X_filled.shape
    rng = np.random.default_rng(random_state)

    if isinstance(init, (tuple, list)) and len(init) == 2:
        _, factors = init
        return [np.asarray(f, dtype=float).copy() for f in factors]

    init_name = str(init).strip().lower()
    if init_name == "random":
        return [rng.normal(size=(shape[m], rank)) for m in range(len(shape))]

    # SVD initialization mode-by-mode with random padding when needed.
    factors: List[np.ndarray] = []
    for mode in range(len(shape)):
        unfold = np.asarray(tl.unfold(np.asarray(X_filled, dtype=float), mode), dtype=float)
        u, s, _ = np.linalg.svd(unfold, full_matrices=False)
        cols = min(rank, u.shape[1])
        f = np.zeros((shape[mode], rank), dtype=float)
        if cols > 0:
            f[:, :cols] = u[:, :cols]
            f[:, :cols] *= s[:cols].reshape(1, cols)
        if cols < rank:
            f[:, cols:] = rng.normal(size=(shape[mode], rank - cols))
        factors.append(f)
    return factors


def _weighted_constrained_ao_admm_missing(
    X_proc: np.ndarray,
    mask: np.ndarray,
    rank: int,
    max_iter: int,
    tol: float,
    init: Any,
    random_state: Optional[int],
    fixed_modes: Optional[Sequence[int]],
    constraints: Dict[str, Any],
) -> Tuple[Any, List[float]]:
    """Weighted constrained CP fit for missing data with AO-ADMM style updates."""
    X_obs = np.asarray(X_proc, dtype=float)
    observed = np.asarray(mask, dtype=bool)
    cleaned_constraints = _sanitize_tensorly_constraints(constraints, int(X_obs.ndim))

    fill_value = float(np.nanmean(X_obs)) if np.isfinite(np.nanmean(X_obs)) else 0.0
    X_filled = np.where(observed, X_obs, fill_value)
    factors = _initial_cp_factors(X_filled, int(rank), init, random_state)
    weights = np.ones(int(rank), dtype=float)

    n_modes = X_obs.ndim
    fixed = set(int(m) for m in (fixed_modes or []))

    dual_variables: List[np.ndarray] = [np.zeros_like(f) for f in factors]
    factors_aux: List[np.ndarray] = [np.zeros((f.shape[1], f.shape[0]), dtype=float) for f in factors]
    rec_errors: List[float] = []

    # Start from a feasible point for selected constraints.
    for mode in range(n_modes):
        factors[mode] = np.asarray(
            proximal_operator(
                np.asarray(factors[mode], dtype=float),
                non_negative=cleaned_constraints.get("non_negative"),
                l1_reg=cleaned_constraints.get("l1_reg"),
                l2_reg=cleaned_constraints.get("l2_reg"),
                l2_square_reg=cleaned_constraints.get("l2_square_reg"),
                unimodality=cleaned_constraints.get("unimodality"),
                normalize=cleaned_constraints.get("normalize"),
                simplex=cleaned_constraints.get("simplex"),
                normalized_sparsity=cleaned_constraints.get("normalized_sparsity"),
                soft_sparsity=cleaned_constraints.get("soft_sparsity"),
                smoothness=cleaned_constraints.get("smoothness"),
                monotonicity=cleaned_constraints.get("monotonicity"),
                hard_sparsity=cleaned_constraints.get("hard_sparsity"),
                n_const=n_modes,
                order=mode,
            ),
            dtype=float,
        )

    inner_iter = max(5, min(40, int(max_iter // 5) if max_iter > 0 else 5))

    for it in range(max(1, int(max_iter))):
        for mode in range(n_modes):
            if mode in fixed:
                continue

            x_unfold = np.asarray(tl.unfold(X_obs, mode), dtype=float)
            m_unfold = np.asarray(tl.unfold(observed, mode), dtype=bool)
            kr = np.asarray(khatri_rao([np.asarray(f, dtype=float) for f in factors], skip_matrix=mode), dtype=float)

            updated = np.asarray(factors[mode], dtype=float).copy()
            for row_idx in range(updated.shape[0]):
                obs = m_unfold[row_idx]
                n_obs = int(np.count_nonzero(obs))
                if n_obs == 0:
                    updated[row_idx, :] = 0.0
                    continue

                z = kr[obs, :]
                y = x_unfold[row_idx, obs]
                if n_obs < rank:
                    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
                else:
                    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
                updated[row_idx, :] = coef

            factors[mode], factors_aux[mode], dual_variables[mode] = admm(
                UtM=np.asarray(updated, dtype=float),
                UtU=np.eye(int(rank), dtype=float),
                x=np.asarray(factors[mode], dtype=float),
                dual_var=np.asarray(dual_variables[mode], dtype=float),
                n_iter_max=inner_iter,
                n_const=n_modes,
                order=mode,
                non_negative=cleaned_constraints.get("non_negative"),
                l1_reg=cleaned_constraints.get("l1_reg"),
                l2_reg=cleaned_constraints.get("l2_reg"),
                l2_square_reg=cleaned_constraints.get("l2_square_reg"),
                unimodality=cleaned_constraints.get("unimodality"),
                normalize=cleaned_constraints.get("normalize"),
                simplex=cleaned_constraints.get("simplex"),
                normalized_sparsity=cleaned_constraints.get("normalized_sparsity"),
                soft_sparsity=cleaned_constraints.get("soft_sparsity"),
                smoothness=cleaned_constraints.get("smoothness"),
                monotonicity=cleaned_constraints.get("monotonicity"),
                hard_sparsity=cleaned_constraints.get("hard_sparsity"),
                tol=max(float(tol), 1e-9),
            )
            factors[mode] = np.asarray(factors[mode], dtype=float)

        reconstructed = np.asarray(cp_to_tensor((weights, factors)), dtype=float)
        residual_obs = reconstructed[observed] - X_obs[observed]
        fit = float(np.sqrt(np.mean(residual_obs ** 2)))
        rec_errors.append(fit)

        if len(rec_errors) >= 2:
            fit_change = float(abs(rec_errors[-2] - rec_errors[-1]) / (abs(rec_errors[-2]) + 1e-12))
            if fit_change < float(tol):
                break

    return (weights, factors), rec_errors


def _masked_constrained_parafac_em(
    X_proc: np.ndarray,
    mask: np.ndarray,
    rank: int,
    max_iter: int,
    tol: float,
    init: Any,
    random_state: Optional[int],
    fixed_modes: Optional[Sequence[int]],
    constraints: Dict[str, Any],
    adaptive_inner: bool = False,
) -> Tuple[Any, List[float]]:
    """Fit constrained CP with missing values via EM outer iterations.

    M-step: constrained PARAFAC on the currently completed tensor.
    E-step: replace only missing entries with current reconstruction.
    """
    observed = np.asarray(mask, dtype=bool)
    cleaned_constraints = _sanitize_tensorly_constraints(constraints, int(X_proc.ndim))
    X_obs = np.asarray(X_proc, dtype=float)

    if np.all(observed):
        kwargs_direct: Dict[str, Any] = {
            "rank": int(rank),
            "n_iter_max": int(max_iter),
            "init": init,
            "random_state": random_state,
            "return_errors": True,
            "tol_outer": float(tol),
            "tol_inner": max(float(tol) * 10.0, 1e-9),
        }
        if fixed_modes:
            kwargs_direct["fixed_modes"] = list(fixed_modes)
        for key in (
            "non_negative",
            "l1_reg",
            "l2_reg",
            "l2_square_reg",
            "unimodality",
            "normalize",
            "simplex",
            "normalized_sparsity",
            "soft_sparsity",
            "smoothness",
            "monotonicity",
            "hard_sparsity",
        ):
            if key in cleaned_constraints:
                kwargs_direct[key] = cleaned_constraints[key]

        cp_tensor, errors = constrained_parafac(np.asarray(X_obs, dtype=float), **kwargs_direct)
        return cp_tensor, [float(e) for e in errors]

    missing = ~observed
    fill_value = float(np.nanmean(X_obs)) if np.isfinite(np.nanmean(X_obs)) else 0.0
    X_filled = np.where(observed, X_obs, fill_value)

    # Split the budget so each EM step performs a meaningful constrained update.
    em_max_iter = max(5, min(int(max_iter), 50))
    inner_iter = max(5, min(30, int(max_iter)))

    prev_fit: Optional[float] = None
    rec_errors: List[float] = []
    best_cp = None

    for em_it in range(em_max_iter):
        if adaptive_inner:
            frac = float(em_it + 1) / float(max(em_max_iter, 1))
            inner_iter = max(5, min(int(max_iter), int(5 + frac * max(5, int(max_iter) - 5))))

        kwargs_em: Dict[str, Any] = {
            "rank": int(rank),
            "n_iter_max": int(inner_iter),
            "init": init if best_cp is None else best_cp,
            "random_state": random_state,
            "return_errors": True,
            "tol_outer": float(tol),
            "tol_inner": max(float(tol) * 10.0, 1e-9),
        }
        if fixed_modes:
            kwargs_em["fixed_modes"] = list(fixed_modes)
        for key in (
            "non_negative",
            "l1_reg",
            "l2_reg",
            "l2_square_reg",
            "unimodality",
            "normalize",
            "simplex",
            "normalized_sparsity",
            "soft_sparsity",
            "smoothness",
            "monotonicity",
            "hard_sparsity",
        ):
            if key in cleaned_constraints:
                kwargs_em[key] = cleaned_constraints[key]

        cp_tensor, _ = constrained_parafac(np.asarray(X_filled, dtype=float), **kwargs_em)
        best_cp = cp_tensor
        reconstructed = np.asarray(cp_to_tensor(cp_tensor), dtype=float)

        prev_missing_values = X_filled[missing].copy()
        X_filled[missing] = reconstructed[missing]

        residual_obs = reconstructed[observed] - X_obs[observed]
        fit = float(np.sqrt(np.mean(residual_obs ** 2)))
        rec_errors.append(fit)

        denom = float(np.linalg.norm(prev_missing_values)) + 1e-12
        missing_change = float(np.linalg.norm(X_filled[missing] - prev_missing_values) / denom)

        fit_change = np.inf
        if prev_fit is not None:
            fit_change = float(abs(prev_fit - fit) / (abs(prev_fit) + 1e-12))
        prev_fit = fit

        if fit_change < float(tol) and missing_change < float(tol):
            break

    if best_cp is None:
        raise RuntimeError("Masked constrained PARAFAC did not produce a CP solution.")

    return best_cp, rec_errors


def _single_fit_once(
    X_cal: np.ndarray,
    Y_cal: Optional[np.ndarray],
    n_components: int,
    X_val: Optional[np.ndarray],
    Y_val: Optional[np.ndarray],
    nway_flag: Optional[int],
    init_method: str,
    max_iter: int,
    tol: float,
    random_state: Optional[int],
    mode_centering: Sequence[bool],
    mode_normalization: Sequence[bool],
    profile_paths: Sequence[str],
    profile_usage: Sequence[str],
    tensorly_constraints: Optional[Dict[str, Any]],
    unconstrained_sparsity: bool,
    unconstrained_linesearch: bool,
    unconstrained_orthogonalise: bool,
    missing_constrained_solver: str,
    component_y_mapping: Any,
    emit_missing_solver_notice: bool = True,
) -> Dict[str, Any]:
    X_raw = np.asarray(X_cal, dtype=float)
    if nway_flag is not None and X_raw.ndim != int(nway_flag) + 1:
        emit_execution_warning(
            code="parafac_nway_mismatch",
            text=(
                f"Provided nway_flag={nway_flag} implies {int(nway_flag)+1}D including samples, "
                f"but X_cal has ndim={X_raw.ndim}."
            ),
        )

    X_proc, mode_stats = _modewise_preprocess(X_raw, mode_centering, mode_normalization)
    mask = ~np.isnan(X_proc)
    X_filled = np.nan_to_num(X_proc, nan=0.0)

    n_modes = X_filled.ndim
    rank = int(n_components)

    loaded_profiles: List[Optional[np.ndarray]] = [None] * n_modes
    for mode_idx in range(n_modes):
        path = profile_paths[mode_idx] if mode_idx < len(profile_paths) else ""
        if path:
            loaded_profiles[mode_idx] = _load_profile_matrix(path, X_filled.shape[mode_idx], rank)

    init = str(init_method).strip().lower() or "svd"
    if init not in {"svd", "random", "custom"}:
        init = "svd"

    fixed_modes: List[int] = []
    init_factors: Optional[List[np.ndarray]] = None

    if init == "custom" or any(profile_usage[m] in {"initial", "fixed"} for m in range(n_modes)):
        rng = np.random.default_rng(random_state)
        init_factors = [rng.normal(size=(X_filled.shape[m], rank)) for m in range(n_modes)]
        for mode_idx in range(n_modes):
            use = profile_usage[mode_idx]
            profile = loaded_profiles[mode_idx]
            if profile is None:
                continue
            if use in {"initial", "fixed", "reference"}:
                init_factors[mode_idx] = profile.copy()
            if use == "fixed":
                fixed_modes.append(mode_idx)

    base_kwargs: Dict[str, Any] = {
        "rank": rank,
        "n_iter_max": int(max_iter),
        "init": init if init_factors is None else (np.ones(rank), init_factors),
        "random_state": random_state,
        "return_errors": True,
    }

    constraints = tensorly_constraints or {}
    constraints = _sanitize_tensorly_constraints(constraints, n_modes)
    has_missing = bool(np.isnan(X_proc).any())
    use_constrained = bool(HAS_CONSTRAINED_PARAFAC and constraints)

    if use_constrained and (bool(unconstrained_sparsity) or bool(unconstrained_linesearch) or bool(unconstrained_orthogonalise)):
        emit_execution_message(
            code="parafac_unconstrained_options_ignored",
            text=(
                "sparsity, linesearch, and orthogonalise are unconstrained PARAFAC options and were ignored "
                "because mode constraints are active."
            ),
        )

    solver_mode = _normalize_missing_constrained_solver(missing_constrained_solver)
    if has_missing and use_constrained and bool(emit_missing_solver_notice):
        emit_execution_message(
            code="parafac_constraints_missing_solver",
            text=(
                f"Using missing-data constrained solver mode: {solver_mode}."
            ),
        )

    if fixed_modes:
        base_kwargs["fixed_modes"] = fixed_modes

    if use_constrained and has_missing:
        if solver_mode == "weighted_ao_admm":
            cp_tensor, errors = _weighted_constrained_ao_admm_missing(
                X_proc=np.asarray(X_proc, dtype=float),
                mask=mask,
                rank=rank,
                max_iter=int(max_iter),
                tol=float(tol),
                init=base_kwargs["init"],
                random_state=random_state,
                fixed_modes=fixed_modes,
                constraints=constraints,
            )
        else:
            cp_tensor, errors = _masked_constrained_parafac_em(
                X_proc=np.asarray(X_proc, dtype=float),
                mask=mask,
                rank=rank,
                max_iter=int(max_iter),
                tol=float(tol),
                init=base_kwargs["init"],
                random_state=random_state,
                fixed_modes=fixed_modes,
                constraints=constraints,
                adaptive_inner=(solver_mode == "em_adaptive"),
            )
        parafac_result = (cp_tensor, errors)
    elif use_constrained:
        kwargs_constrained: Dict[str, Any] = dict(base_kwargs)
        kwargs_constrained["tol_outer"] = float(tol)
        kwargs_constrained["tol_inner"] = max(float(tol) * 10.0, 1e-9)
        for key in (
            "non_negative",
            "l1_reg",
            "l2_reg",
            "l2_square_reg",
            "unimodality",
            "normalize",
            "simplex",
            "normalized_sparsity",
            "soft_sparsity",
            "smoothness",
            "monotonicity",
            "hard_sparsity",
        ):
            if key in constraints:
                kwargs_constrained[key] = constraints[key]

        parafac_result = constrained_parafac(X_filled, **kwargs_constrained)
    else:
        kwargs_unconstrained: Dict[str, Any] = dict(base_kwargs)
        kwargs_unconstrained["tol"] = float(tol)
        kwargs_unconstrained["normalize_factors"] = False
        kwargs_unconstrained["mask"] = mask
        kwargs_unconstrained["sparsity"] = 1.0 if bool(unconstrained_sparsity) else None
        kwargs_unconstrained["l2_reg"] = 0.0
        kwargs_unconstrained["linesearch"] = bool(unconstrained_linesearch)
        kwargs_unconstrained["orthogonalise"] = bool(unconstrained_orthogonalise)

        parafac_result = parafac(X_filled, **kwargs_unconstrained)

    if isinstance(parafac_result, tuple) and len(parafac_result) == 2:
        cp_tensor, errors = parafac_result
        weights, factors = cp_tensor
    else:
        weights, factors = parafac_result
        errors = []
    factors = list(factors)

    reconstructed = cp_to_tensor((weights, factors))
    residual = np.asarray(X_filled - reconstructed, dtype=float)
    observed = np.asarray(mask, dtype=bool)

    ssr = float(np.sum((residual[observed]) ** 2))
    n_obs = int(np.count_nonzero(observed))
    sfit = float(np.sqrt(ssr / max(n_obs, 1)))
    explained = _explained_variance(X_filled, residual, observed)
    core_cons = _core_consistency(X_filled, np.asarray(weights, dtype=float), factors)

    scores_a = np.asarray(factors[0], dtype=float)
    val_scores_a = None
    if X_val is not None:
        val_scores_a = _project_scores_mode_a(np.asarray(X_val, dtype=float), np.asarray(weights, dtype=float), factors)

    Yc = _as_2d_y(Y_cal)
    Yv = _as_2d_y(Y_val)

    calibration_models: List[Dict[str, Any]] = []
    selected_component_y_mapping: Dict[int, int] = {}
    auto_mapping_used = False

    if Yc is not None and Yc.shape[0] == scores_a.shape[0]:
        parsed_map, explicit_map, all_empty_map = _parse_component_mapping_input(
            component_y_mapping,
            n_components=int(scores_a.shape[1]),
        )

        if explicit_map and not all_empty_map:
            for comp_idx, y_idx in parsed_map.items():
                if 0 <= comp_idx < scores_a.shape[1] and 0 <= y_idx < Yc.shape[1]:
                    selected_component_y_mapping[int(comp_idx)] = int(y_idx)
        else:
            selected_component_y_mapping = _auto_mapping(scores_a, Yc)
            auto_mapping_used = True

        for comp_idx in sorted(selected_component_y_mapping.keys()):
            y_idx = selected_component_y_mapping[comp_idx]
            fit_cal = _fit_linear_1d(scores_a[:, comp_idx], Yc[:, y_idx])
            cal_metrics = fit_cal.get("metrics", {}) if isinstance(fit_cal, dict) else {}
            n_used = int(_safe_float(cal_metrics.get("n_samples_used"), default=0.0))
            intercept = _safe_float(fit_cal.get("intercept"), default=np.nan)
            slope = _safe_float(fit_cal.get("slope"), default=np.nan)
            if n_used < 2 or (not np.isfinite(intercept)) or (not np.isfinite(slope)):
                continue

            entry = {
                "component": int(comp_idx + 1),
                "y_column": int(y_idx + 1),
                "intercept": float(intercept),
                "slope": float(slope),
                "calibration": cal_metrics,
            }

            if Yv is not None and val_scores_a is not None and Yv.shape[1] > y_idx:
                y_val_vec = np.asarray(Yv[:, y_idx], dtype=float).reshape(-1)
                score_val_vec = np.asarray(val_scores_a[:, comp_idx], dtype=float).reshape(-1)
                valid_val = np.isfinite(y_val_vec) & np.isfinite(score_val_vec)
                n_valid_val = int(np.count_nonzero(valid_val))

                if n_valid_val >= 2 and np.isfinite(intercept) and np.isfinite(slope):
                    y_pred_val = intercept + slope * score_val_vec[valid_val]
                    y_true_val = y_val_vec[valid_val]
                    val_res = y_true_val - y_pred_val
                    ss_res = float(np.sum(val_res ** 2))
                    ss_tot = float(np.sum((y_true_val - np.mean(y_true_val)) ** 2)) + 1e-12
                    entry["validation"] = {
                        "R2": float(1.0 - ss_res / ss_tot),
                        "RMSEP": float(np.sqrt(np.mean(val_res ** 2))),
                        "n_samples_used": int(n_valid_val),
                    }

            calibration_models.append(entry)

    reference_angles: List[Dict[str, Any]] = []
    for mode_idx in range(n_modes):
        if profile_usage[mode_idx] not in {"reference", "initial", "fixed"}:
            continue
        ref = loaded_profiles[mode_idx]
        if ref is None:
            continue

        fact = np.asarray(factors[mode_idx], dtype=float)
        for comp in range(min(fact.shape[1], ref.shape[1])):
            reference_angles.append(
                {
                    "mode": int(mode_idx),
                    "component": int(comp + 1),
                    "angle_degrees": _angle_degrees(fact[:, comp], ref[:, comp]),
                }
            )

    inst_profiles = _instrumental_profiles(np.asarray(weights, dtype=float), factors)

    metrics = {
        "calibration": {
            "core_consistency": core_cons,
            "SSR": ssr,
            "sfit": sfit,
            "explained_variance": explained,
            "n_iter": int(len(errors)) if errors is not None else 0,
            "used_constrained_parafac": bool(use_constrained),
            "missing_constrained_solver": solver_mode if (has_missing and use_constrained) else "n/a",
        },
        "calibration_models": calibration_models,
        "reference_angles": reference_angles,
    }

    return {
        "scores_mode_a": scores_a,
        "factors": [np.asarray(f, dtype=float) for f in factors],
        "weights": np.asarray(weights, dtype=float),
        "reconstructed": np.asarray(reconstructed, dtype=float),
        "residual": residual,
        "instrumental_profiles": inst_profiles,
        "metrics": metrics,
        "component_y_mapping": {str(k + 1): int(v + 1) for k, v in selected_component_y_mapping.items()},
        "auto_mapping_used": bool(auto_mapping_used),
        "calibration_models": calibration_models,
        "reference_angles": reference_angles,
        "mode_stats": mode_stats,
        "val_scores_mode_a": val_scores_a,
    }


def _single_fit(
    X_cal: np.ndarray,
    Y_cal: Optional[np.ndarray],
    n_components: int,
    X_val: Optional[np.ndarray],
    Y_val: Optional[np.ndarray],
    nway_flag: Optional[int],
    init_method: str,
    max_iter: int,
    tol: float,
    random_state: Optional[int],
    mode_centering: Sequence[bool],
    mode_normalization: Sequence[bool],
    profile_paths: Sequence[str],
    profile_usage: Sequence[str],
    tensorly_constraints: Optional[Dict[str, Any]],
    unconstrained_sparsity: bool,
    unconstrained_linesearch: bool,
    unconstrained_orthogonalise: bool,
    missing_constrained_solver: str,
    component_y_mapping: Any,
    random_multi_start: bool,
    random_multi_start_runs: int,
    emit_missing_solver_notice: bool = True,
) -> Dict[str, Any]:
    init_name = str(init_method).strip().lower()
    use_multi = bool(random_multi_start and init_name == "random")
    n_starts = int(max(1, random_multi_start_runs)) if use_multi else 1

    if use_multi:
        seed_stream = np.random.default_rng(random_state)
        start_seeds = [int(seed_stream.integers(0, 2**31 - 1)) for _ in range(n_starts)]
    else:
        start_seeds = [random_state]

    best_result: Optional[Dict[str, Any]] = None
    best_sfit = np.inf
    start_results: List[Dict[str, Any]] = []

    for seed in start_seeds:
        fit_result = _single_fit_once(
            X_cal=X_cal,
            Y_cal=Y_cal,
            n_components=n_components,
            X_val=X_val,
            Y_val=Y_val,
            nway_flag=nway_flag,
            init_method=init_method,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
            mode_centering=mode_centering,
            mode_normalization=mode_normalization,
            profile_paths=profile_paths,
            profile_usage=profile_usage,
            tensorly_constraints=tensorly_constraints,
            unconstrained_sparsity=unconstrained_sparsity,
            unconstrained_linesearch=unconstrained_linesearch,
            unconstrained_orthogonalise=unconstrained_orthogonalise,
            missing_constrained_solver=missing_constrained_solver,
            component_y_mapping=component_y_mapping,
            emit_missing_solver_notice=emit_missing_solver_notice,
        )
        sfit = _safe_float(
            fit_result.get("metrics", {}).get("calibration", {}).get("sfit"),
            default=np.inf,
        )
        start_results.append({"seed": None if seed is None else int(seed), "sfit": float(sfit)})
        if sfit < best_sfit:
            best_sfit = sfit
            best_result = fit_result

    if best_result is None:
        raise RuntimeError("PARAFAC fitting failed to produce any candidate solution.")

    best_result.setdefault("metrics", {}).setdefault("calibration", {})
    best_result["metrics"]["calibration"]["multi_start_used"] = bool(use_multi)
    best_result["metrics"]["calibration"]["multi_start_runs"] = int(n_starts)
    best_result["metrics"]["calibration"]["multi_start_results"] = start_results
    return best_result


def _build_text_report(
    rank: int,
    metrics: Dict[str, Any],
    mapping: Dict[str, int],
    sweep_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    cal = metrics.get("calibration", {}) if isinstance(metrics, dict) else {}
    lines: List[str] = []
    lines.append("PARAFAC Report")
    lines.append("==============")
    lines.append(f"Components: {rank}")
    lines.append(f"Core consistency: {_safe_float(cal.get('core_consistency')):.4f}")
    lines.append(f"SSR: {_safe_float(cal.get('SSR')):.6g}")
    lines.append(f"sfit: {_safe_float(cal.get('sfit')):.6g}")
    lines.append(f"Explained variance (%): {_safe_float(cal.get('explained_variance')):.4f}")
    lines.append(f"Iterations: {int(_safe_float(cal.get('n_iter')))}")
    lines.append("")

    angles = metrics.get("reference_angles", []) if isinstance(metrics, dict) else []
    if angles:
        lines.append("Reference angle differences (degrees):")
        for item in angles:
            lines.append(
                f"- Mode {item.get('mode')}, Component {item.get('component')}: "
                f"{_safe_float(item.get('angle_degrees')):.4f}"
            )
        lines.append("")

    models = metrics.get("calibration_models", []) if isinstance(metrics, dict) else []
    if models:
        lines.append("Calibration statistics:")
        ordered_models = sorted(
            [m for m in models if isinstance(m, dict)],
            key=lambda item: int(_safe_float(item.get("component"), default=np.inf)),
        )
        for m in ordered_models:
            comp = m.get("component")
            ycol = m.get("y_column")
            cm = m.get("calibration", {})
            vm = m.get("validation", {})
            lines.append(
                f"- Component {comp} -> Y column {ycol} | "
                f"Cal R2={_safe_float(cm.get('R2')):.4f}, Cal RMSEP={_safe_float(cm.get('RMSEP')):.6g}"
            )
            if vm:
                lines.append(
                    f"  Validation R2={_safe_float(vm.get('R2')):.4f}, "
                    f"Validation RMSEP={_safe_float(vm.get('RMSEP')):.6g}"
                )
    elif mapping:
        lines.append("Calibration mapping provided but no compatible Y data was available.")

    if sweep_results:
        lines.append("")
        lines.append("Sweep results:")
        for item in sweep_results:
            lines.append(
                f"- F={item.get('n_components')} | sfit={_safe_float(item.get('sfit')):.6g} | "
                f"core={_safe_float(item.get('core_consistency')):.4f} | EV={_safe_float(item.get('explained_variance')):.4f}%"
            )

    if bool(cal.get("multi_start_used", False)):
        lines.append("")
        lines.append(f"Random multi-start: {int(_safe_float(cal.get('multi_start_runs'), 1))} runs")
        for i, item in enumerate(cal.get("multi_start_results", []) or [], start=1):
            seed_text = "None" if item.get("seed") is None else str(item.get("seed"))
            lines.append(f"- Start {i}: seed={seed_text}, sfit={_safe_float(item.get('sfit')):.6g}")

    return "\n".join(lines)


def parafac_analysis(
    X_cal: Optional[np.ndarray] = None,
    Y_cal: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    n_components: int = 2,
    nway_flag: Optional[int] = None,
    mode_centering: Optional[Any] = None,
    mode_normalization: Optional[Any] = None,
    constraint_non_negative: Optional[Any] = None,
    constraint_l1_reg: Optional[Any] = None,
    constraint_l2_reg: Optional[Any] = None,
    constraint_l2_square_reg: Optional[Any] = None,
    constraint_unimodality: Optional[Any] = None,
    constraint_normalize: Optional[Any] = None,
    constraint_simplex: Optional[Any] = None,
    constraint_normalized_sparsity: Optional[Any] = None,
    constraint_soft_sparsity: Optional[Any] = None,
    constraint_smoothness: Optional[Any] = None,
    constraint_monotonicity: Optional[Any] = None,
    constraint_hard_sparsity: Optional[Any] = None,
    unconstrained_sparsity: bool = False,
    unconstrained_linesearch: bool = False,
    unconstrained_orthogonalise: bool = False,
    init_method: str = "svd",
    max_iter: int = 500,
    tol: float = 1e-7,
    random_state: Optional[Any] = 42,
    random_multi_start: bool = False,
    random_multi_start_runs: Optional[Any] = 5,
    allow_missing: bool = True,
    missing_constrained_solver: str = "em",
    profile_paths: Optional[Any] = None,
    profile_usage: Optional[Any] = None,
    sweep_mode: bool = False,
    component_range: str = "",
    component_y_mapping: Any = "",
    cv_config: Optional[Any] = None,
    fold: int = 0,
    axis_n_info: Optional[List[np.ndarray]] = None,
    dim_labels: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """PARAFAC with missing-data handling, sweep mode, calibration, and CV."""

    X_cal_test = kwargs.get("X_cal_test")
    Y_cal_test = kwargs.get("Y_cal_test")

    if X_cal is None and "X_cal_train" in kwargs:
        X_cal = kwargs["X_cal_train"]
    if Y_cal is None and "Y_cal_train" in kwargs:
        Y_cal = kwargs["Y_cal_train"]

    if X_cal is None:
        raise ValueError("X_cal is required for parafac_analysis")

    X_arr = np.asarray(X_cal, dtype=float)

    n_modes = X_arr.ndim
    center_list = _normalize_bool_list(mode_centering, n_modes)
    norm_list = _normalize_bool_list(mode_normalization, n_modes)
    parsed_constraints = _build_tensorly_constraints(
        n_modes=n_modes,
        constraint_non_negative=constraint_non_negative,
        constraint_l1_reg=constraint_l1_reg,
        constraint_l2_reg=constraint_l2_reg,
        constraint_l2_square_reg=constraint_l2_square_reg,
        constraint_unimodality=constraint_unimodality,
        constraint_normalize=constraint_normalize,
        constraint_simplex=constraint_simplex,
        constraint_normalized_sparsity=constraint_normalized_sparsity,
        constraint_soft_sparsity=constraint_soft_sparsity,
        constraint_smoothness=constraint_smoothness,
        constraint_monotonicity=constraint_monotonicity,
        constraint_hard_sparsity=constraint_hard_sparsity,
    )
    seed_value = _safe_optional_int(random_state, default=None)
    multi_start_runs_value = _safe_optional_int(random_multi_start_runs, default=5)
    if multi_start_runs_value is None:
        multi_start_runs_value = 5
    multi_start_runs_value = int(max(1, multi_start_runs_value))
    solver_mode = _normalize_missing_constrained_solver(missing_constrained_solver)
    path_list, usage_list = _expand_profile_mode_settings(profile_paths, profile_usage, n_modes)

    if cv_config is not None and HAS_CV and hasattr(cv_config, "is_enabled") and cv_config.is_enabled():
        if fold == 0 and not _is_cv_fold_call():
            pipeline = CVPipeline(cv_config)
            y_split = None

            try:
                splits = list(pipeline.splitter.get_splits(np.asarray(X_arr), y_split))
            except Exception:
                emit_execution_warning(
                    code="parafac_cv_fallback",
                    text="Selected CV strategy required unavailable stratification labels. Falling back to standard KFold logic where possible.",
                )
                splits = list(pipeline.splitter.get_splits(np.asarray(X_arr)))

            ycv_pred = None
            Y2 = _as_2d_y(Y_cal)
            if Y2 is not None:
                ycv_pred = np.full_like(Y2, np.nan, dtype=float)

            fold_metrics: List[Dict[str, Any]] = []

            for fidx, (train_idx, test_idx) in enumerate(splits):
                fold_result = parafac_analysis(
                    X_cal=np.asarray(X_arr)[train_idx],
                    Y_cal=None if Y_cal is None else np.asarray(Y_cal)[train_idx],
                    X_val=np.asarray(X_arr)[test_idx],
                    Y_val=None if Y_cal is None else np.asarray(Y_cal)[test_idx],
                    n_components=n_components,
                    nway_flag=nway_flag,
                    mode_centering=center_list,
                    mode_normalization=norm_list,
                    constraint_non_negative=constraint_non_negative,
                    constraint_l1_reg=constraint_l1_reg,
                    constraint_l2_reg=constraint_l2_reg,
                    constraint_l2_square_reg=constraint_l2_square_reg,
                    constraint_unimodality=constraint_unimodality,
                    constraint_normalize=constraint_normalize,
                    constraint_simplex=constraint_simplex,
                    constraint_normalized_sparsity=constraint_normalized_sparsity,
                    constraint_soft_sparsity=constraint_soft_sparsity,
                    constraint_smoothness=constraint_smoothness,
                    constraint_monotonicity=constraint_monotonicity,
                    constraint_hard_sparsity=constraint_hard_sparsity,
                    unconstrained_sparsity=unconstrained_sparsity,
                    unconstrained_linesearch=unconstrained_linesearch,
                    unconstrained_orthogonalise=unconstrained_orthogonalise,
                    init_method=init_method,
                    max_iter=max_iter,
                    tol=tol,
                    random_state=seed_value,
                    random_multi_start=random_multi_start,
                    random_multi_start_runs=multi_start_runs_value,
                    allow_missing=allow_missing,
                    missing_constrained_solver=solver_mode,
                    profile_paths=path_list,
                    profile_usage=usage_list,
                    sweep_mode=False,
                    component_y_mapping=component_y_mapping,
                    cv_config=None,
                    fold=fidx + 1,
                    axis_n_info=axis_n_info,
                    dim_labels=dim_labels,
                )

                cal_models = fold_result.get("calibration_models", [])
                fold_metrics.append(
                    {
                        "fold": int(fidx),
                        "n_test": int(len(test_idx)),
                        "sfit": _safe_float(fold_result.get("metrics", {}).get("calibration", {}).get("sfit")),
                        "explained_variance": _safe_float(fold_result.get("metrics", {}).get("calibration", {}).get("explained_variance")),
                        "calibration_models": cal_models,
                    }
                )

                if ycv_pred is not None and cal_models:
                    scores_test = fold_result.get("val_scores_mode_a")
                    if scores_test is not None:
                        for cm in cal_models:
                            c = int(cm.get("component", 1)) - 1
                            ycol = int(cm.get("y_column", 1)) - 1
                            if c < scores_test.shape[1] and ycol < ycv_pred.shape[1]:
                                ycv_pred[test_idx, ycol] = (
                                    _safe_float(cm.get("intercept")) +
                                    _safe_float(cm.get("slope")) * np.asarray(scores_test)[:, c]
                                )

            cv_agg: Dict[str, Any] = {"n_folds": int(len(splits)), "fold_metrics": fold_metrics}
            if fold_metrics:
                cv_agg["sfit_mean"] = float(np.mean([fm["sfit"] for fm in fold_metrics]))
                cv_agg["explained_variance_mean"] = float(np.mean([fm["explained_variance"] for fm in fold_metrics]))

            if ycv_pred is not None and Y2 is not None:
                valid = np.isfinite(ycv_pred) & np.isfinite(Y2)
                if np.any(valid):
                    diff = np.where(valid, Y2 - ycv_pred, np.nan)
                    rmsep = float(np.sqrt(np.nanmean(diff ** 2)))
                    cv_agg["calibration_rmsep"] = rmsep

            full_result = parafac_analysis(
                X_cal=X_arr,
                Y_cal=Y_cal,
                X_val=X_val,
                Y_val=Y_val,
                n_components=n_components,
                nway_flag=nway_flag,
                mode_centering=center_list,
                mode_normalization=norm_list,
                constraint_non_negative=constraint_non_negative,
                constraint_l1_reg=constraint_l1_reg,
                constraint_l2_reg=constraint_l2_reg,
                constraint_l2_square_reg=constraint_l2_square_reg,
                constraint_unimodality=constraint_unimodality,
                constraint_normalize=constraint_normalize,
                constraint_simplex=constraint_simplex,
                constraint_normalized_sparsity=constraint_normalized_sparsity,
                constraint_soft_sparsity=constraint_soft_sparsity,
                constraint_smoothness=constraint_smoothness,
                constraint_monotonicity=constraint_monotonicity,
                constraint_hard_sparsity=constraint_hard_sparsity,
                unconstrained_sparsity=unconstrained_sparsity,
                unconstrained_linesearch=unconstrained_linesearch,
                unconstrained_orthogonalise=unconstrained_orthogonalise,
                init_method=init_method,
                max_iter=max_iter,
                tol=tol,
                random_state=seed_value,
                random_multi_start=random_multi_start,
                random_multi_start_runs=multi_start_runs_value,
                allow_missing=allow_missing,
                missing_constrained_solver=solver_mode,
                profile_paths=path_list,
                profile_usage=usage_list,
                sweep_mode=sweep_mode,
                component_range=component_range,
                component_y_mapping=component_y_mapping,
                cv_config=None,
                fold=-1,
                axis_n_info=axis_n_info,
                dim_labels=dim_labels,
            )

            full_result["cv_results"] = cv_agg
            full_result.setdefault("metrics", {})
            full_result["metrics"]["cv"] = cv_agg
            full_result["y_cv_pred"] = ycv_pred
            return full_result

    # Sweep mode
    sweep_results: List[Dict[str, Any]] = []
    selected_rank = int(n_components)

    if sweep_mode:
        values = parse_numeric_spec(component_range)
        if len(values) == 1:
            try:
                single_value = int(float(values[0]))
            except Exception:
                single_value = 0
            if single_value >= 2:
                values = list(range(1, single_value + 1))
        ranks = sorted({int(v) for v in values if _safe_float(v) >= 1})
        if not ranks:
            ranks = [int(n_components)]

        for rk in ranks:
            fit = _single_fit(
                X_cal=X_arr,
                Y_cal=Y_cal,
                n_components=rk,
                X_val=X_val,
                Y_val=Y_val,
                nway_flag=nway_flag,
                init_method=init_method,
                max_iter=max_iter,
                tol=tol,
                random_state=seed_value,
                random_multi_start=random_multi_start,
                random_multi_start_runs=multi_start_runs_value,
                mode_centering=center_list,
                mode_normalization=norm_list,
                tensorly_constraints=parsed_constraints,
                unconstrained_sparsity=unconstrained_sparsity,
                unconstrained_linesearch=unconstrained_linesearch,
                unconstrained_orthogonalise=unconstrained_orthogonalise,
                missing_constrained_solver=solver_mode,
                profile_paths=path_list,
                profile_usage=usage_list,
                component_y_mapping=component_y_mapping,
                emit_missing_solver_notice=not bool(sweep_mode),
            )
            m = fit["metrics"]["calibration"]
            item = {
                "n_components": int(rk),
                "core_consistency": _safe_float(m.get("core_consistency")),
                "SSR": _safe_float(m.get("SSR")),
                "sfit": _safe_float(m.get("sfit")),
                "explained_variance": _safe_float(m.get("explained_variance")),
                "n_iter": int(_safe_float(m.get("n_iter"))),
            }
            sweep_results.append(item)

        # Sweep is reporting-only: keep final model rank equal to the user-specified n_components.
        selected_rank = int(n_components)
        result = _single_fit(
            X_cal=X_arr,
            Y_cal=Y_cal,
            n_components=selected_rank,
            X_val=X_val,
            Y_val=Y_val,
            nway_flag=nway_flag,
            init_method=init_method,
            max_iter=max_iter,
            tol=tol,
            random_state=seed_value,
            random_multi_start=random_multi_start,
            random_multi_start_runs=multi_start_runs_value,
            mode_centering=center_list,
            mode_normalization=norm_list,
            tensorly_constraints=parsed_constraints,
            unconstrained_sparsity=unconstrained_sparsity,
            unconstrained_linesearch=unconstrained_linesearch,
            unconstrained_orthogonalise=unconstrained_orthogonalise,
            missing_constrained_solver=solver_mode,
            profile_paths=path_list,
            profile_usage=usage_list,
            component_y_mapping=component_y_mapping,
            emit_missing_solver_notice=not bool(sweep_mode),
        )
    else:
        result = _single_fit(
            X_cal=X_arr,
            Y_cal=Y_cal,
            n_components=int(n_components),
            X_val=X_val,
            Y_val=Y_val,
            nway_flag=nway_flag,
            init_method=init_method,
            max_iter=max_iter,
            tol=tol,
            random_state=seed_value,
            random_multi_start=random_multi_start,
            random_multi_start_runs=multi_start_runs_value,
            mode_centering=center_list,
            mode_normalization=norm_list,
            tensorly_constraints=parsed_constraints,
            unconstrained_sparsity=unconstrained_sparsity,
            unconstrained_linesearch=unconstrained_linesearch,
            unconstrained_orthogonalise=unconstrained_orthogonalise,
            missing_constrained_solver=solver_mode,
            profile_paths=path_list,
            profile_usage=usage_list,
            component_y_mapping=component_y_mapping,
            emit_missing_solver_notice=True,
        )

    if Y_cal is not None and bool(result.get("auto_mapping_used", False)):
        emit_execution_warning(
            code="parafac_auto_mapping",
            text=(
                "No explicit component-to-Y mapping was provided (or all entries were empty). "
                "PARAFAC evaluated valid component-to-Y combinations and selected the best mapping."
            ),
        )

    report_text = _build_text_report(
        rank=selected_rank,
        metrics=result.get("metrics", {}),
        mapping={int(k) - 1: int(v) - 1 for k, v in result.get("component_y_mapping", {}).items()},
        sweep_results=sweep_results if sweep_mode else None,
    )

    output = {
        "scores_mode_a": result.get("scores_mode_a"),
        "val_scores_mode_a": result.get("val_scores_mode_a"),
        "scores_mode_a_heatmap": None,
        "mode_a_sample_axis": None,
        "mode_a_component_axis": None,
        "sweep_F": None,
        "sweep_sfit": None,
        "sweep_core_consistency": None,
        "sweep_explained_variance": None,
        "sweep_n_iter": None,
        "factors": result.get("factors"),
        "weights": result.get("weights"),
        "instrumental_profiles": result.get("instrumental_profiles"),
        "reconstructed": result.get("reconstructed"),
        "residual": result.get("residual"),
        "metrics": result.get("metrics"),
        "calibration_models": result.get("calibration_models"),
        "component_y_mapping": result.get("component_y_mapping"),
        "reference_angles": result.get("reference_angles"),
        "parafac_report": report_text,
        "sweep_results": sweep_results if sweep_mode else None,
        "selected_n_components": int(selected_rank),
        "axis_n_info": axis_n_info,
        "dim_labels": dim_labels,
        "nway_flag": int(nway_flag) if nway_flag is not None else int(max(1, X_arr.ndim - 1)),
        "cv_results": result.get("cv_results"),
        "y_cv_pred": result.get("y_cv_pred"),
    }

    scores_mode_a = result.get("scores_mode_a")
    if isinstance(scores_mode_a, np.ndarray) and scores_mode_a.ndim == 2:
        output["scores_mode_a_heatmap"] = np.asarray(scores_mode_a, dtype=float).T
        output["mode_a_sample_axis"] = np.arange(1, int(scores_mode_a.shape[0]) + 1, dtype=float)
        output["mode_a_component_axis"] = np.arange(1, int(scores_mode_a.shape[1]) + 1, dtype=float)

    if sweep_results:
        sweep_items = [item for item in sweep_results if isinstance(item, dict)]
        if sweep_items:
            output["sweep_F"] = np.asarray([_safe_float(item.get("n_components")) for item in sweep_items], dtype=float)
            output["sweep_sfit"] = np.asarray([_safe_float(item.get("sfit")) for item in sweep_items], dtype=float)
            output["sweep_core_consistency"] = np.asarray([_safe_float(item.get("core_consistency")) for item in sweep_items], dtype=float)
            output["sweep_explained_variance"] = np.asarray([_safe_float(item.get("explained_variance")) for item in sweep_items], dtype=float)
            output["sweep_n_iter"] = np.asarray([_safe_float(item.get("n_iter")) for item in sweep_items], dtype=float)

    return output


_PARAFAC_RETURN_ORDER: Tuple[str, ...] = (
    "scores_mode_a",
    "val_scores_mode_a",
    "scores_mode_a_heatmap",
    "mode_a_sample_axis",
    "mode_a_component_axis",
    "sweep_F",
    "sweep_sfit",
    "sweep_core_consistency",
    "sweep_explained_variance",
    "sweep_n_iter",
    "factors",
    "weights",
    "instrumental_profiles",
    "reconstructed",
    "residual",
    "metrics",
    "calibration_models",
    "component_y_mapping",
    "reference_angles",
    "parafac_report",
    "sweep_results",
    "selected_n_components",
    "axis_n_info",
    "dim_labels",
    "nway_flag",
    "cv_results",
    "y_cv_pred",
)


def parafac_analysis_standard(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:
    """Adapter for app execution pipeline: return outputs as ordered tuple.

    The core parafac_analysis API keeps its dict return for direct module users.
    """
    result = parafac_analysis(*args, **kwargs)
    if not isinstance(result, dict):
        return (result,)
    return tuple(result.get(key) for key in _PARAFAC_RETURN_ORDER)
