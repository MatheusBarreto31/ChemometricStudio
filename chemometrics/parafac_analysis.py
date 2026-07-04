"""PARAFAC (CP decomposition) analysis with optional calibration and CV support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _min_max(vals: List[float]) -> Tuple[float, float]:
    return (min(vals), max(vals))


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


def _normalize_missing_constrained_solver(value: Any) -> str:
    mode = str(value).strip().lower() if value is not None else "em"
    allowed = {"em", "em_adaptive", "weighted_ao_admm"}
    return mode if mode in allowed else "em"


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

    raw_items: List[Any]
    if isinstance(y_labels, np.ndarray):
        raw_items = np.asarray(y_labels).reshape(-1).tolist()
    elif isinstance(y_labels, (list, tuple)):
        raw_items = list(y_labels)
    elif isinstance(y_labels, str):
        text = y_labels.strip()
        if not text:
            return []
        if "," in text:
            raw_items = [item.strip() for item in text.split(",")]
        elif "\t" in text:
            raw_items = [item.strip() for item in text.split("\t")]
        else:
            raw_items = [item.strip() for item in text.split()]
    else:
        raw_items = [y_labels]

    out: List[str] = []
    for item in raw_items:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


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


def _normalize_optional_float_list(values: Any, count: int) -> List[Optional[float]]:
    if isinstance(values, np.ndarray):
        raw = np.asarray(values).reshape(-1).tolist()
    elif isinstance(values, (list, tuple)):
        raw = list(values)
    elif isinstance(values, str):
        text = values.strip()
        raw = [item.strip() for item in text.split(",")] if text else []
    elif values is None:
        raw = []
    else:
        raw = [values]

    out: List[Optional[float]] = []
    for item in raw[:count]:
        if item is None:
            out.append(None)
            continue
        text = str(item).strip()
        if text == "":
            out.append(None)
            continue
        try:
            out.append(float(text))
        except Exception:
            out.append(None)

    if len(out) < count:
        out.extend([None] * (count - len(out)))
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
    constraint_l1_reg_strength: Any = None,
    constraint_l2_reg: Any = None,
    constraint_l2_reg_strength: Any = None,
    constraint_l2_square_reg: Any = None,
    constraint_l2_square_reg_strength: Any = None,
    constraint_unimodality: Any = None,
    constraint_normalize: Any = None,
    constraint_simplex: Any = None,
    constraint_simplex_strength: Any = None,
    constraint_normalized_sparsity: Any = None,
    constraint_normalized_sparsity_strength: Any = None,
    constraint_soft_sparsity: Any = None,
    constraint_soft_sparsity_strength: Any = None,
    constraint_smoothness: Any = None,
    constraint_smoothness_strength: Any = None,
    constraint_monotonicity: Any = None,
    constraint_hard_sparsity: Any = None,
    constraint_hard_sparsity_strength: Any = None,
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
        "normalized_sparsity": 1.0,
        "soft_sparsity": 1.0,
        "smoothness": 1.0,
        "monotonicity": True,
        "hard_sparsity": 1,
    }

    mode_strengths: Dict[str, List[Optional[float]]] = {
        "l1_reg": _normalize_optional_float_list(constraint_l1_reg_strength, n_modes),
        "l2_reg": _normalize_optional_float_list(constraint_l2_reg_strength, n_modes),
        "l2_square_reg": _normalize_optional_float_list(constraint_l2_square_reg_strength, n_modes),
        "simplex": _normalize_optional_float_list(constraint_simplex_strength, n_modes),
        "normalized_sparsity": _normalize_optional_float_list(constraint_normalized_sparsity_strength, n_modes),
        "soft_sparsity": _normalize_optional_float_list(constraint_soft_sparsity_strength, n_modes),
        "smoothness": _normalize_optional_float_list(constraint_smoothness_strength, n_modes),
        "hard_sparsity": _normalize_optional_float_list(constraint_hard_sparsity_strength, n_modes),
    }

    out: Dict[str, Any] = {}
    for key, flags in mode_flags.items():
        if not any(bool(v) for v in flags):
            continue
        enabled_value = mode_defaults[key]
        strengths = mode_strengths.get(key)
        if strengths is None:
            out[key] = [enabled_value if bool(flag) else None for flag in flags]
            continue

        values: List[Any] = []
        for idx, flag in enumerate(flags):
            if not bool(flag):
                values.append(None)
                continue
            maybe_strength = strengths[idx] if idx < len(strengths) else None
            if maybe_strength is None:
                values.append(enabled_value)
            elif key == "hard_sparsity":
                values.append(max(1, int(round(float(maybe_strength)))))
            else:
                values.append(float(maybe_strength))
        out[key] = values
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


def _apply_modewise_stats(
    X: np.ndarray,
    mode_stats: Sequence[Dict[str, np.ndarray]],
) -> np.ndarray:
    """Apply previously learned mode-wise preprocessing statistics to new data."""
    Xp = np.asarray(X, dtype=float).copy()
    n_modes = Xp.ndim

    for mode in range(n_modes):
        unfold = tl.unfold(Xp, mode)
        stats = mode_stats[mode] if mode < len(mode_stats) else {}

        mean = stats.get("mean") if isinstance(stats, dict) else None
        if mean is not None:
            unfold = unfold - np.asarray(mean, dtype=float)

        std = stats.get("std") if isinstance(stats, dict) else None
        if std is not None:
            std_arr = np.asarray(std, dtype=float)
            std_arr = np.where((~np.isfinite(std_arr)) | (std_arr <= 0), 1.0, std_arr)
            unfold = unfold / std_arr

        Xp = tl.fold(unfold, mode, Xp.shape)

    return np.asarray(Xp, dtype=float)


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


def _build_mode_profile_outputs(
    factors: Any,
    axis_n_info: Optional[Any],
    dim_labels: Optional[Any],
) -> Dict[str, Any]:
    """Build navigation-friendly line-profile outputs for each PARAFAC mode."""
    if not isinstance(factors, (list, tuple)) or not factors:
        return {}

    factor_mats: List[np.ndarray] = []
    for factor in factors:
        try:
            arr = np.asarray(factor, dtype=float)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            continue
        factor_mats.append(arr)

    if not factor_mats:
        return {}

    n_modes = int(len(factor_mats))
    n_components = int(min(mat.shape[1] for mat in factor_mats))
    max_mode_len = int(max(mat.shape[0] for mat in factor_mats))

    mode_profile_axes = np.full((n_modes, max_mode_len), np.nan, dtype=float)
    mode_profile_factors = np.full((n_modes, n_components, max_mode_len), np.nan, dtype=float)

    labels_seq = list(dim_labels) if isinstance(dim_labels, (list, tuple, np.ndarray)) else []
    axis_seq = list(axis_n_info) if isinstance(axis_n_info, (list, tuple)) else []
    mode_labels: List[str] = []
    mode_y_titles: List[str] = []

    def _mode_letter(index: int) -> str:
        """Convert zero-based index to Excel-style letters: 0->A, 25->Z, 26->AA."""
        n = int(index)
        chars: List[str] = []
        while True:
            n, rem = divmod(n, 26)
            chars.append(chr(ord("A") + rem))
            if n == 0:
                break
            n -= 1
        return "".join(reversed(chars))

    for mode_idx, factor_mat in enumerate(factor_mats):
        mode_len = int(factor_mat.shape[0])
        mode_label = f"Mode {mode_idx + 1}"
        if mode_idx < len(labels_seq):
            label_text = str(labels_seq[mode_idx]).strip()
            if label_text:
                mode_label = label_text
        mode_labels.append(mode_label)
        mode_y_titles.append(f"{_mode_letter(mode_idx)} (a.u.)")

        mode_axis = None
        if mode_idx < len(axis_seq):
            try:
                candidate_axis = np.asarray(axis_seq[mode_idx], dtype=float).reshape(-1)
                if candidate_axis.size == mode_len:
                    mode_axis = candidate_axis
            except Exception:
                mode_axis = None
        if mode_axis is None:
            mode_axis = np.arange(1, mode_len + 1, dtype=float)
        else:
            # Keep provided axis values unless they are simple 0-based sample indices.
            finite_axis = np.asarray(mode_axis, dtype=float)
            if np.all(np.isfinite(finite_axis)):
                zero_based = np.arange(0, mode_len, dtype=float)
                if np.allclose(finite_axis, zero_based, rtol=0.0, atol=1e-12):
                    mode_axis = finite_axis + 1.0

        mode_profile_axes[mode_idx, :mode_len] = mode_axis[:mode_len]
        mode_profile_factors[mode_idx, :, :mode_len] = np.asarray(factor_mat[:, :n_components], dtype=float).T

    component_labels = [f"Component {idx + 1}" for idx in range(n_components)]

    return {
        "mode_profile_axes": mode_profile_axes,
        "mode_profile_factors": mode_profile_factors,
        "mode_profile_mode_labels": mode_labels,
        "mode_profile_y_titles": mode_y_titles,
        "mode_profile_component_labels": component_labels,
        "mode_profile_navigation_labels_by_dimension": [mode_labels, component_labels],
    }


def _build_mode_profile_outputs_with_full_a(
    mode_profile_axes: Any,
    mode_profile_factors: Any,
    scores_mode_a: Any,
    val_scores_mode_a: Any,
) -> Dict[str, Any]:
    """Return mode-profile arrays where mode A includes Cal+Val rows.

    This keeps legacy calibration-only mode-profile outputs unchanged and
    provides explicit full-A variants for pages that require all samples.
    """
    axes = np.asarray(mode_profile_axes, dtype=float) if mode_profile_axes is not None else None
    factors = np.asarray(mode_profile_factors, dtype=float) if mode_profile_factors is not None else None
    scores_cal = np.asarray(scores_mode_a, dtype=float) if scores_mode_a is not None else None
    scores_val = np.asarray(val_scores_mode_a, dtype=float) if val_scores_mode_a is not None else None

    if (
        not isinstance(axes, np.ndarray)
        or axes.ndim != 2
        or not isinstance(factors, np.ndarray)
        or factors.ndim != 3
        or not isinstance(scores_cal, np.ndarray)
        or scores_cal.ndim != 2
    ):
        return {
            "mode_profile_axes_full_a": mode_profile_axes,
            "mode_profile_factors_full_a": mode_profile_factors,
        }

    if isinstance(scores_val, np.ndarray) and scores_val.ndim == 2 and scores_val.shape[1] == scores_cal.shape[1]:
        scores_full = np.vstack([scores_cal, scores_val])
    else:
        scores_full = scores_cal

    n_modes = int(factors.shape[0])
    n_components = int(factors.shape[1])
    old_max_len = int(factors.shape[2])
    full_len = int(scores_full.shape[0])
    target_len = int(max(old_max_len, full_len))

    axes_full = np.full((n_modes, target_len), np.nan, dtype=float)
    axes_full[:, : min(old_max_len, target_len)] = axes[:, : min(old_max_len, target_len)]
    axes_full[0, :full_len] = np.arange(1, full_len + 1, dtype=float)

    factors_full = np.full((n_modes, n_components, target_len), np.nan, dtype=float)
    factors_full[:, :, : min(old_max_len, target_len)] = factors[:, :, : min(old_max_len, target_len)]
    n_comp_copy = min(n_components, int(scores_full.shape[1]))
    factors_full[0, :n_comp_copy, :full_len] = np.asarray(scores_full[:, :n_comp_copy], dtype=float).T

    return {
        "mode_profile_axes_full_a": axes_full,
        "mode_profile_factors_full_a": factors_full,
    }


def _mean_component_value(values: np.ndarray) -> float:
    vec = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(vec)
    n = int(np.count_nonzero(finite))
    if n <= 0:
        return 0.0
    return float(np.mean(vec[finite]))


def _orient_signs_by_negative_pairs(
    factors: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Flip component signs in mode-pairs when both vectors have negative mean.

    Flipping two modes within the same component preserves the represented tensor.
    """
    factor_list = [np.asarray(f, dtype=float).copy() for f in factors]
    if len(factor_list) < 2:
        return factor_list, []

    rank = int(min(f.shape[1] for f in factor_list))
    flips: List[Dict[str, Any]] = []

    for comp_idx in range(rank):
        mostly_negative_modes = [
            mode_idx
            for mode_idx, factor in enumerate(factor_list)
            if _mean_component_value(factor[:, comp_idx]) < 0.0
        ]
        while len(mostly_negative_modes) >= 2:
            first_mode = int(mostly_negative_modes.pop(0))
            second_mode = int(mostly_negative_modes.pop(0))
            factor_list[first_mode][:, comp_idx] *= -1.0
            factor_list[second_mode][:, comp_idx] *= -1.0
            flips.append(
                {
                    "component": int(comp_idx + 1),
                    "modes": [first_mode, second_mode],
                }
            )

    return factor_list, flips


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


def _constraint_enabled_for_mode(constraint_value: Any, mode: int) -> bool:
    if constraint_value is None:
        return False
    if isinstance(constraint_value, dict):
        return bool(constraint_value.get(int(mode), False))
    if isinstance(constraint_value, (list, tuple)):
        if 0 <= int(mode) < len(constraint_value):
            return bool(constraint_value[int(mode)])
        return False
    return bool(constraint_value)


def _stabilize_non_negative_init(
    factors: List[np.ndarray],
    non_negative_constraint: Any,
    eps: float = 1e-12,
) -> List[np.ndarray]:
    """Make constrained non-negative initial factors feasible without dead columns."""
    stabilized: List[np.ndarray] = []
    for mode, factor in enumerate(factors):
        arr = np.asarray(factor, dtype=float).copy()
        if _constraint_enabled_for_mode(non_negative_constraint, mode):
            arr = np.abs(arr)
            col_norms = np.linalg.norm(arr, axis=0)
            zero_cols = col_norms <= float(eps)
            if np.any(zero_cols):
                arr[:, zero_cols] = float(eps)
        stabilized.append(arr)
    return stabilized


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
    factors = _stabilize_non_negative_init(factors, cleaned_constraints.get("non_negative"))
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
    orient_mostly_negative_pairs: bool,
    emit_missing_solver_notice: bool = True,
) -> Dict[str, Any]:
    X_raw = np.asarray(X_cal, dtype=float)
    X_val_raw = None if X_val is None else np.asarray(X_val, dtype=float)
    combine_fit_samples = bool(X_val_raw is not None)

    if combine_fit_samples:
        if X_val_raw.ndim != X_raw.ndim or X_val_raw.shape[1:] != X_raw.shape[1:]:
            raise ValueError(
                "X_val must match X_cal dimensionality and non-sample dimensions when "
                "validation samples are included in PARAFAC fitting."
            )
        X_fit_raw = np.concatenate([X_raw, X_val_raw], axis=0)
    else:
        X_fit_raw = X_raw

    if nway_flag is not None and X_raw.ndim != int(nway_flag) + 1:
        emit_execution_warning(
            code="parafac_nway_mismatch",
            text=(
                f"Provided nway_flag={nway_flag} implies {int(nway_flag)+1}D including samples, "
                f"but X_cal has ndim={X_raw.ndim}."
            ),
        )

    X_proc, mode_stats = _modewise_preprocess(X_fit_raw, mode_centering, mode_normalization)
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
    if use_constrained and has_missing:
        solver_label_map = {
            "em": "EM",
            "em_adaptive": "EM (Adaptive Inner Iterations)",
            "weighted_ao_admm": "Weighted AO-ADMM",
        }
        solver_label = solver_label_map.get(solver_mode, str(solver_mode))
        implementation_used = f"Constrained PARAFAC (Missing Data, {solver_label})"
    elif use_constrained:
        implementation_used = "Constrained PARAFAC"
    else:
        implementation_used = "PARAFAC"

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

    sign_flip_pairs: List[Dict[str, Any]] = []
    if bool(orient_mostly_negative_pairs):
        factors, sign_flip_pairs = _orient_signs_by_negative_pairs(factors)

    reconstructed_full = cp_to_tensor((weights, factors))

    n_cal_samples = int(X_raw.shape[0])
    all_scores_a = np.asarray(factors[0], dtype=float)
    if combine_fit_samples:
        scores_a = np.asarray(all_scores_a[:n_cal_samples], dtype=float)
        val_scores_a = np.asarray(all_scores_a[n_cal_samples:], dtype=float)
        factors_for_output = [scores_a] + [np.asarray(f, dtype=float) for f in factors[1:]]
        reconstructed = cp_to_tensor((weights, factors_for_output))
        residual = np.asarray(X_filled[:n_cal_samples] - reconstructed, dtype=float)
        observed = np.asarray(mask[:n_cal_samples], dtype=bool)
        X_metrics = np.asarray(X_filled[:n_cal_samples], dtype=float)
        core_factors = factors_for_output
    else:
        scores_a = all_scores_a
        factors_for_output = [np.asarray(f, dtype=float) for f in factors]
        reconstructed = np.asarray(reconstructed_full, dtype=float)
        residual = np.asarray(X_filled - reconstructed, dtype=float)
        observed = np.asarray(mask, dtype=bool)
        X_metrics = np.asarray(X_filled, dtype=float)
        core_factors = factors_for_output

    ssr = float(np.sum((residual[observed]) ** 2))
    n_obs = int(np.count_nonzero(observed))
    sfit = float(np.sqrt(ssr / max(n_obs, 1)))
    explained = _explained_variance(X_metrics, residual, observed)
    core_cons = _core_consistency(X_metrics, np.asarray(weights, dtype=float), core_factors)

    if (not combine_fit_samples) and X_val is not None:
        X_val_proc = _apply_modewise_stats(np.asarray(X_val, dtype=float), mode_stats)
        val_scores_a = _project_scores_mode_a(X_val_proc, np.asarray(weights, dtype=float), factors)
    elif not combine_fit_samples:
        val_scores_a = None

    Yc = _as_2d_y(Y_cal)
    Yv = _as_2d_y(Y_val)

    calibration_models: List[Dict[str, Any]] = []
    prediction_candidates: List[Dict[str, Any]] = []
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
            prediction_candidates.append(
                {
                    "component": int(comp_idx),
                    "y_index": int(y_idx),
                    "intercept": float(intercept),
                    "slope": float(slope),
                    "cal_r2": _safe_float(cal_metrics.get("R2"), default=-np.inf),
                }
            )

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

    y_cal_pred = None
    y_val_pred = None
    y_cal_error = None
    y_val_error = None
    if Yc is not None and prediction_candidates:
        y_cal_pred = np.full_like(np.asarray(Yc, dtype=float), np.nan, dtype=float)
        n_val_samples = int(val_scores_a.shape[0]) if isinstance(val_scores_a, np.ndarray) and val_scores_a.ndim == 2 else 0
        n_val_pred_cols = int(max((int(candidate["y_index"]) + 1) for candidate in prediction_candidates)) if prediction_candidates else 0
        candidate_by_pair: Dict[Tuple[int, int], Dict[str, Any]] = {
            (int(candidate["component"]), int(candidate["y_index"])): candidate for candidate in prediction_candidates
        }
        for comp_idx in sorted(selected_component_y_mapping.keys()):
            y_index = int(selected_component_y_mapping[comp_idx])
            candidate = candidate_by_pair.get((int(comp_idx), int(y_index)))
            if not isinstance(candidate, dict):
                continue
            comp_idx = int(candidate["component"])
            if comp_idx >= scores_a.shape[1] or y_index >= y_cal_pred.shape[1]:
                continue
            score_vec = np.asarray(scores_a[:, comp_idx], dtype=float).reshape(-1)
            y_vec = np.asarray(Yc[:, y_index], dtype=float).reshape(-1)
            valid = np.isfinite(score_vec) & np.isfinite(y_vec) & ~np.isfinite(y_cal_pred[:, y_index])
            if np.any(valid):
                y_cal_pred[valid, y_index] = float(candidate["intercept"]) + float(candidate["slope"]) * score_vec[valid]

            if val_scores_a is not None and comp_idx < val_scores_a.shape[1]:
                if y_val_pred is None and n_val_samples > 0 and n_val_pred_cols > 0:
                    y_val_pred = np.full((n_val_samples, n_val_pred_cols), np.nan, dtype=float)
                if y_val_pred is None or y_index >= y_val_pred.shape[1]:
                    continue
                score_val_vec = np.asarray(val_scores_a[:, comp_idx], dtype=float).reshape(-1)
                valid_val = np.isfinite(score_val_vec) & ~np.isfinite(y_val_pred[:, y_index])
                if Yv is not None and y_index < Yv.shape[1] and Yv.shape[0] == score_val_vec.shape[0]:
                    y_val_vec = np.asarray(Yv[:, y_index], dtype=float).reshape(-1)
                    valid_val = valid_val & np.isfinite(y_val_vec)
                if np.any(valid_val):
                    y_val_pred[valid_val, y_index] = float(candidate["intercept"]) + float(candidate["slope"]) * score_val_vec[valid_val]

        y_cal_error = np.asarray(Yc, dtype=float) - np.asarray(y_cal_pred, dtype=float)
        if y_val_pred is not None and Yv is not None:
            y_val_error = np.asarray(Yv, dtype=float) - np.asarray(y_val_pred, dtype=float)

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

    inst_profiles = _instrumental_profiles(np.asarray(weights, dtype=float), factors_for_output)

    metrics = {
        "calibration": {
            "core_consistency": core_cons,
            "SSR": ssr,
            "sfit": sfit,
            "explained_variance": explained,
            "n_iter": int(len(errors)) if errors is not None else 0,
            "implementation_used": implementation_used,
            "used_constrained_parafac": bool(use_constrained),
            "missing_constrained_solver": solver_mode if (has_missing and use_constrained) else "n/a",
            "orient_mostly_negative_pairs": bool(orient_mostly_negative_pairs),
            "fit_combined_samples": bool(combine_fit_samples),
            "n_samples_fit": int(X_fit_raw.shape[0]),
            "n_samples_calibration": int(n_cal_samples),
            "sign_pair_flips": sign_flip_pairs,
            "sign_pair_flip_count": int(len(sign_flip_pairs)),
        },
        "calibration_models": calibration_models,
        "reference_angles": reference_angles,
    }

    return {
        "scores_mode_a": scores_a,
        "factors": [np.asarray(f, dtype=float) for f in factors_for_output],
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
        "y_cal_pred": y_cal_pred,
        "y_val_pred": y_val_pred,
        "y_cal_error": y_cal_error,
        "y_val_error": y_val_error,
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
    orient_mostly_negative_pairs: bool,
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
            orient_mostly_negative_pairs=orient_mostly_negative_pairs,
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
    y_labels: Optional[Sequence[str]] = None,
    auto_mapping_used: bool = False,
    validation_processing: str = "batch",
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
    lines.append(f"Implementation: {str(cal.get('implementation_used', 'unknown'))}")
    lines.append(f"Validation Processing: {validation_processing.replace('_', ' ').title()}")
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
        lines.append("")
        if bool(auto_mapping_used):
            lines.append("Regression Report (Paired by maximum likelihood):")
        else:
            lines.append("Regression Report:")
        ordered_models = sorted(
            [m for m in models if isinstance(m, dict)],
            key=lambda item: int(_safe_float(item.get("component"), default=np.inf)),
        )
        
        lines.append("")
        lines.append("Calibration:")
        for m in ordered_models:
            comp = m.get("component")
            ycol = m.get("y_column")
            cm = m.get("calibration", {})

            y_target_text = f"Y{ycol}"
            try:
                y_idx = int(_safe_float(ycol, default=0)) - 1
            except Exception:
                y_idx = -1
            if y_labels is not None and 0 <= y_idx < len(y_labels):
                label_text = str(y_labels[y_idx]).strip()
                if label_text:
                    y_target_text = f"{label_text} (Y{ycol})"

            lines.append(
                f"- F{comp} -> {y_target_text} | "
                f"R2={_safe_float(cm.get('R2')):.4f}, RMSEC={_safe_float(cm.get('RMSEP')):.6g}"
            )
        
        lines.append("")
        lines.append("Validation:")
        for m in ordered_models:
            comp = m.get("component")
            ycol = m.get("y_column")
            vm = m.get("validation", {})

            y_target_text = f"Y{ycol}"
            try:
                y_idx = int(_safe_float(ycol, default=0)) - 1
            except Exception:
                y_idx = -1
            if y_labels is not None and 0 <= y_idx < len(y_labels):
                label_text = str(y_labels[y_idx]).strip()
                if label_text:
                    y_target_text = f"{label_text} (Y{ycol})"

            if vm:
                lines.append(
                    f"- F{comp} -> {y_target_text} | "
                    f"R2={_safe_float(vm.get('R2')):.4f}, RMSEP={_safe_float(vm.get('RMSEP')):.6g}"
                )
    elif mapping:
        lines.append("")
        lines.append("Regression Report:")
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
    impl = str(fom.get("implementation_used", "") or "PARAFAC")
    lines: List[str] = [
        "PARAFAC Report",
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

    core = fom.get("core_consistency")
    ssr = fom.get("SSR")
    sfit = fom.get("sfit")
    ev = fom.get("explained_variance")
    n_iter = fom.get("n_iter")
    if core is not None:
        lines.append(f"Core consistency (range): {_fmt_range(core)}")
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
            lines.append(f"- {label} | R2 (range): n/a, RMSEC (range): n/a")

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


def _build_component_pair_prediction_outputs(
    component_y_mapping: Any,
    y_labels: Optional[Sequence[str]],
    y_cal_true: Any,
    y_val_true: Any,
    scores_mode_a: Any = None,
    val_scores_mode_a: Any = None,
    calibration_models: Any = None,
) -> Dict[str, Any]:
    """Build per-component-pairing prediction matrices and labels.

    Predictions are computed independently for each (component, Y-column) pair
    using that component's own calibration model, so pairings sharing a Y column
    produce distinct predictions.
    """

    def _to_2d(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            return None
        return arr

    def _extract_pair_matrix(value: Any, y_cols_1based: List[int]) -> Optional[np.ndarray]:
        arr = _to_2d(value)
        if arr is None:
            return None
        if not y_cols_1based:
            return None
        cols: List[np.ndarray] = []
        for y_col in y_cols_1based:
            y_idx = int(y_col) - 1
            if 0 <= y_idx < arr.shape[1]:
                cols.append(np.asarray(arr[:, y_idx], dtype=float).reshape(-1, 1))
            else:
                cols.append(np.full((arr.shape[0], 1), np.nan, dtype=float))
        return np.column_stack(cols) if cols else None

    def _extract_score_pair_matrix(scores: Any, comp_indices_1based: List[int]) -> Optional[np.ndarray]:
        """Extract columns from scores matrix by 1-based component indices."""
        arr = _to_2d(scores)
        if arr is None:
            return None
        if not comp_indices_1based:
            return None
        cols: List[np.ndarray] = []
        for comp_1based in comp_indices_1based:
            comp_idx = int(comp_1based) - 1
            if 0 <= comp_idx < arr.shape[1]:
                cols.append(np.asarray(arr[:, comp_idx], dtype=float).reshape(-1, 1))
            else:
                cols.append(np.full((arr.shape[0], 1), np.nan, dtype=float))
        return np.column_stack(cols) if cols else None

    mapping_dict = component_y_mapping if isinstance(component_y_mapping, dict) else {}
    pairs: List[Tuple[int, int]] = []
    for comp_key, y_col_value in mapping_dict.items():
        try:
            comp_1based = int(comp_key)
            y_col_1based = int(y_col_value)
        except (TypeError, ValueError):
            continue
        if comp_1based < 1 or y_col_1based < 1:
            continue
        pairs.append((comp_1based, y_col_1based))
    pairs = sorted(set(pairs), key=lambda item: item[0])

    pair_components = [int(comp) for comp, _ in pairs]
    pair_y_columns = [int(y_col) for _, y_col in pairs]

    y_titles: List[str] = []
    pair_labels: List[str] = []
    labels_seq = list(y_labels) if y_labels is not None else []
    for comp_1based, y_col_1based in pairs:
        y_idx = int(y_col_1based) - 1
        y_title = f"Y{y_col_1based}"
        if 0 <= y_idx < len(labels_seq):
            label_text = str(labels_seq[y_idx]).strip()
            if label_text:
                y_title = label_text
        y_titles.append(y_title)
        pair_labels.append(f"F{comp_1based} -> {y_title} (Y{y_col_1based})")

    # Per-pairing A score matrices (n_samples x n_pairs, each column = scores for paired component)
    a_scores_cal_pairs = _extract_score_pair_matrix(scores_mode_a, pair_components)
    a_scores_val_pairs = _extract_score_pair_matrix(val_scores_mode_a, pair_components)

    # Build lookup: component_1based -> (intercept, slope)
    model_lookup: Dict[int, Tuple[float, float]] = {}
    if isinstance(calibration_models, list):
        for model_entry in calibration_models:
            if not isinstance(model_entry, dict):
                continue
            comp = model_entry.get("component")
            intercept = model_entry.get("intercept")
            slope = model_entry.get("slope")
            if comp is not None and intercept is not None and slope is not None:
                try:
                    model_lookup[int(comp)] = (float(intercept), float(slope))
                except (TypeError, ValueError):
                    pass

    n_pairs = len(pairs)

    # Per-pairing calibration predictions: each column uses the component's own model.
    # This ensures C1->Y1 and C2->Y1 produce different predictions.
    y_cal_true_pairs = _extract_pair_matrix(y_cal_true, pair_y_columns)
    y_cal_pred_pairs: Optional[np.ndarray] = None
    y_cal_error_pairs: Optional[np.ndarray] = None
    if a_scores_cal_pairs is not None and n_pairs > 0:
        n_cal = a_scores_cal_pairs.shape[0]
        pred_cal = np.full((n_cal, n_pairs), np.nan, dtype=float)
        for pair_idx, comp_1based in enumerate(pair_components):
            if comp_1based not in model_lookup:
                continue
            intercept, slope = model_lookup[comp_1based]
            col = a_scores_cal_pairs[:, pair_idx]
            valid = np.isfinite(col)
            pred_cal[valid, pair_idx] = intercept + slope * col[valid]
        y_cal_pred_pairs = pred_cal
        if y_cal_true_pairs is not None:
            y_cal_error_pairs = y_cal_true_pairs - pred_cal

    # Per-pairing validation predictions using each component's own model.
    y_val_true_pairs = _extract_pair_matrix(y_val_true, pair_y_columns)
    y_val_pred_pairs: Optional[np.ndarray] = None
    y_val_error_pairs: Optional[np.ndarray] = None
    if a_scores_val_pairs is not None and n_pairs > 0:
        n_val = a_scores_val_pairs.shape[0]
        pred_val = np.full((n_val, n_pairs), np.nan, dtype=float)
        for pair_idx, comp_1based in enumerate(pair_components):
            if comp_1based not in model_lookup:
                continue
            intercept, slope = model_lookup[comp_1based]
            col = a_scores_val_pairs[:, pair_idx]
            valid = np.isfinite(col)
            pred_val[valid, pair_idx] = intercept + slope * col[valid]
        y_val_pred_pairs = pred_val
        if y_val_true_pairs is not None:
            y_val_error_pairs = y_val_true_pairs - pred_val

    # Effective validation Y: use true reference if available, fall back to predicted
    if y_val_true_pairs is not None and y_val_pred_pairs is not None:
        y_val_effective_pairs = np.where(
            np.isfinite(y_val_true_pairs),
            y_val_true_pairs,
            y_val_pred_pairs,
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
    if a_scores_cal_pairs is not None and n_pairs > 0:
        line_x = np.full((2, n_pairs), np.nan, dtype=float)
        line_y = np.full((2, n_pairs), np.nan, dtype=float)
        for pair_idx, comp_1based in enumerate(pair_components):
            score_col = np.asarray(a_scores_cal_pairs[:, pair_idx], dtype=float)
            finite_score = np.isfinite(score_col)
            if not np.any(finite_score):
                continue

            # Plot uses Reference on x and Score on y. Prefer reference-domain
            # limits so the line spans the full calibration x-axis range.
            x1 = x2 = np.nan
            y1 = y2 = np.nan
            if comp_1based in model_lookup:
                intercept, slope = model_lookup[comp_1based]
                if (
                    y_cal_true_pairs is not None
                    and pair_idx < y_cal_true_pairs.shape[1]
                    and np.isfinite(intercept)
                    and np.isfinite(slope)
                ):
                    ref_col = np.asarray(y_cal_true_pairs[:, pair_idx], dtype=float)
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
                if comp_1based in model_lookup:
                    intercept, slope = model_lookup[comp_1based]
                    x1 = float(intercept) + float(slope) * y1
                    x2 = float(intercept) + float(slope) * y2

            line_x[0, pair_idx] = x1
            line_x[1, pair_idx] = x2
            line_y[0, pair_idx] = y1
            line_y[1, pair_idx] = y2
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
        "parafac_pair_components": np.asarray(pair_components, dtype=int) if pair_components else np.asarray([], dtype=int),
        "parafac_pair_y_columns": np.asarray(pair_y_columns, dtype=int) if pair_y_columns else np.asarray([], dtype=int),
        "parafac_pair_y_titles": y_titles,
        "parafac_pairing_labels": pair_labels,
        "parafac_pairing_labels_by_dimension": [[], pair_labels],
        "y_cal_pred_pairs": y_cal_pred_pairs,
        "y_cal_true_pairs": y_cal_true_pairs,
        "y_cal_error_pairs": y_cal_error_pairs,
        "y_val_pred_pairs": y_val_pred_pairs,
        "y_val_true_pairs": y_val_true_pairs,
        "y_val_error_pairs": y_val_error_pairs,
        "a_scores_cal_pairs": a_scores_cal_pairs,
        "a_scores_val_pairs": a_scores_val_pairs,
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
    """Build unified (n_samples × n_y) prediction matrices from per-pairing data.

    For each Y column j:
      - No pairing       → NaN column
      - One pairing      → use that pairing's predictions directly
      - Multiple pairings→ use the pairing with the highest Pearson r between
                           its cal predictions and the reference cal Y column
    The same best-pairing selection (based on calibration) is applied to val.
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

    # Build Y-column → [pair_indices] mapping
    col_to_pairs: Dict[int, List[int]] = {}
    for pi, y_col in enumerate(pair_y_columns):
        col_to_pairs.setdefault(int(y_col), []).append(pi)

    def _best_pair_idx(j_1based: int, candidates: List[int]) -> int:
        """Return the pair index (from candidates) with highest cal Pearson r."""
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

    # Pre-compute best pair index for each y column that has pairings
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
        # Best-pair selection is based on calibration; same selection applied to val
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


# ---------------------------------------------------------------------------
# Sample-by-sample (SBS) validation helper for PARAFAC
# ---------------------------------------------------------------------------

def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between rows of a and rows of b.

    a: (m, d), b: (n, d) → result: (m, n)
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_unit = a / np.where(a_norm == 0, 1.0, a_norm)
    b_unit = b / np.where(b_norm == 0, 1.0, b_norm)
    return a_unit @ b_unit.T


def _align_factors_to_reference(
    ref_inst: np.ndarray,
    inst: np.ndarray,
    ref_a_cal: Optional[np.ndarray] = None,
    a_cal: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (permutation, signs) to align `inst` (n_comp, d) to `ref_inst`.

    Uses the Hungarian algorithm on absolute cosine similarity.

    If ``ref_a_cal`` and ``a_cal`` are provided (mode-A calibration scores
    transposed to shape ``(n_comp, n_cal)``), their absolute cosine similarities
    are averaged with the instrumental-profile similarities before the assignment
    step.  This yields a more robust alignment when spectral profiles are similar
    across components, because the sample scores add an independent signal.
    NaN values in mode-A arrays are replaced with zero before the cosine
    computation so degenerate samples are automatically excluded.
    Signs are always determined from the instrumental-profile similarity so that
    the existing sign-convention logic is preserved.

    Returns:
        perm   : 1-D int array of length n_comp — the column permutation
        signs  : 1-D float array of length n_comp — +1 or -1 per component
    """
    from scipy.optimize import linear_sum_assignment

    n_comp = ref_inst.shape[0]
    sim_inst = _cosine_matrix(ref_inst, inst)      # (n_comp, n_comp)
    abs_sim = np.abs(sim_inst)

    # Incorporate mode-A calibration scores when available and shapes match.
    if (
        ref_a_cal is not None
        and a_cal is not None
        and ref_a_cal.shape[0] == n_comp
        and a_cal.shape[0] == n_comp
        and ref_a_cal.shape[1] > 0
        and a_cal.shape[1] > 0
    ):
        ref_a_safe = np.nan_to_num(ref_a_cal, nan=0.0, posinf=0.0, neginf=0.0)
        a_safe = np.nan_to_num(a_cal, nan=0.0, posinf=0.0, neginf=0.0)
        sim_a = _cosine_matrix(ref_a_safe, a_safe)  # (n_comp, n_comp)
        abs_sim = (abs_sim + np.abs(sim_a)) / 2.0

    row_ind, col_ind = linear_sum_assignment(-abs_sim)
    perm = np.zeros(n_comp, dtype=int)
    signs = np.ones(n_comp, dtype=float)
    for r, c in zip(row_ind, col_ind):
        perm[r] = int(c)
        # Signs are derived from the instrumental-profile similarity so they
        # reflect the non-A mode sign convention (mode-A absorbs the flip).
        signs[r] = -1.0 if sim_inst[r, c] < 0 else 1.0
    return perm, signs


def _alignment_quality_score(
    ref_inst: np.ndarray,
    inst: np.ndarray,
    ref_a_cal: Optional[np.ndarray] = None,
    a_cal: Optional[np.ndarray] = None,
) -> float:
    """Return mean assigned absolute similarity used for alignment quality.

    This uses the same fused similarity logic as `_align_factors_to_reference`
    but returns only the assignment quality score, which is useful for choosing
    a robust reference model (medoid-like selection).
    """
    from scipy.optimize import linear_sum_assignment

    n_comp = ref_inst.shape[0]
    abs_sim = np.abs(_cosine_matrix(ref_inst, inst))

    if (
        ref_a_cal is not None
        and a_cal is not None
        and ref_a_cal.shape[0] == n_comp
        and a_cal.shape[0] == n_comp
        and ref_a_cal.shape[1] > 0
        and a_cal.shape[1] > 0
    ):
        ref_a_safe = np.nan_to_num(ref_a_cal, nan=0.0, posinf=0.0, neginf=0.0)
        a_safe = np.nan_to_num(a_cal, nan=0.0, posinf=0.0, neginf=0.0)
        abs_sim_a = np.abs(_cosine_matrix(ref_a_safe, a_safe))
        abs_sim = (abs_sim + abs_sim_a) / 2.0

    row_ind, col_ind = linear_sum_assignment(-abs_sim)
    if row_ind.size <= 0:
        return float("-inf")
    return float(np.mean(abs_sim[row_ind, col_ind]))


def _apply_perm_signs_to_parafac(
    factors: List[np.ndarray],
    perm: np.ndarray,
    signs: np.ndarray,
) -> List[np.ndarray]:
    """Reorder and sign-correct PARAFAC factor matrices.

    The sign correction is applied to mode-A (index 0) so that the tensor
    reconstruction is invariant; non-A modes are reordered only (sign already
    absorbed into mode-A).
    """
    aligned = []
    for mode_idx, fmat in enumerate(factors):
        fmat_arr = np.asarray(fmat, dtype=float)
        fmat_reordered = fmat_arr[:, perm]
        if mode_idx == 0:
            fmat_reordered = fmat_reordered * signs[np.newaxis, :]
        aligned.append(fmat_reordered)
    return aligned


def _sbs_parafac(
    X_cal: np.ndarray,
    Y_cal: Optional[np.ndarray],
    X_val: np.ndarray,
    Y_val: Optional[np.ndarray],
    n_components: int,
    nway_flag: Optional[int],
    init_method: str,
    max_iter: int,
    tol: float,
    random_state: Optional[int],
    random_multi_start: bool,
    random_multi_start_runs: int,
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
    orient_mostly_negative_pairs: bool,
    axis_n_info: Optional[Any],
    dim_labels: Optional[Any],
) -> Dict[str, Any]:
    """Run PARAFAC sample-by-sample validation.

    For each validation sample *i* a PARAFAC model is fit on X_cal while
    X_val[i:i+1] is concatenated into the tensor fit to exploit second-order
    advantage (same as batch mode, but one val sample at a time).
    Y_val is never used — calibration regression stays on Y_cal only.
    Factors are aligned across models by selecting a robust reference model
    (medoid-like selection using mean pairwise assignment quality) and then
    applying the Hungarian algorithm on a fused similarity matrix built from
    instrumental profiles plus mode-A calibration scores.  Mode-A uses only the
    calibration rows, so the validation-row insertion does not directly enter
    the score vectors used in the similarity calculation.

    Returns a dict of SBS-specific outputs ready to merge into the main
    ``parafac_analysis`` output dict.
    """
    X_cal_arr = np.asarray(X_cal, dtype=float)
    X_val_arr = np.asarray(X_val, dtype=float)
    n_val = int(X_val_arr.shape[0])
    n_cal = int(X_cal_arr.shape[0])
    n_comp = int(n_components)
    Y_cal_2d = _as_2d_y(Y_cal)
    Y_val_2d = _as_2d_y(Y_val)
    n_y = int(Y_cal_2d.shape[1]) if Y_cal_2d is not None else 0

    # ------------------------------------------------------------------ #
    # Per-model outputs (before alignment)                                #
    # ------------------------------------------------------------------ #
    raw_inst: List[Optional[np.ndarray]] = []        # (n_comp, n_b, n_c, ...)
    raw_factors: List[Optional[List[np.ndarray]]] = []
    raw_scores_cal: List[Optional[np.ndarray]] = []  # (n_cal, n_comp)
    raw_val_scores: List[Optional[np.ndarray]] = []  # (1, n_comp)
    raw_y_val_pred: List[np.ndarray] = []             # (n_y,)
    raw_y_cal_pred: List[Optional[np.ndarray]] = []  # (n_cal, n_y)
    raw_cal_models: List[List[Dict[str, Any]]] = []  # per-model calibration models
    raw_fom: List[Optional[Dict[str, Any]]] = []      # per-model figures of merit
    first_auto_mapping: Optional[bool] = None

    common_kwargs = dict(
        n_components=n_comp,
        nway_flag=nway_flag,
        init_method=init_method,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        random_multi_start=random_multi_start,
        random_multi_start_runs=random_multi_start_runs,
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
        orient_mostly_negative_pairs=orient_mostly_negative_pairs,
    )

    nan_row = np.full((max(n_y, 1),), np.nan)

    for i in range(n_val):
        # Pass val sample i as X_val so it joins the tensor fit (second-order
        # advantage). Y_val is intentionally omitted — calibration regression
        # uses only Y_cal, same as in batch mode.
        try:
            r = _single_fit(
                X_cal=X_cal_arr,
                Y_cal=Y_cal,
                X_val=X_val_arr[i : i + 1],
                Y_val=None,
                emit_missing_solver_notice=(i == 0),
                **common_kwargs,
            )
        except Exception as exc:
            emit_execution_warning(
                code="parafac_sbs_sample_failed",
                text=f"SBS PARAFAC model for validation sample {i + 1} failed: {exc}",
            )
            raw_inst.append(None)
            raw_factors.append(None)
            raw_scores_cal.append(None)
            raw_val_scores.append(None)
            raw_y_val_pred.append(nan_row[:n_y] if n_y > 0 else np.array([]))
            raw_y_cal_pred.append(None)
            raw_cal_models.append([])
            raw_fom.append(None)
            continue

        # Cal scores: (n_cal, n_comp) — from the calibration partition only
        sc = r.get("scores_mode_a")
        raw_scores_cal.append(np.asarray(sc, dtype=float) if sc is not None else None)

        # Val scores: (1, n_comp) — the single val sample's A-mode scores
        vs = r.get("val_scores_mode_a")
        if vs is not None:
            vs_arr = np.asarray(vs, dtype=float)
            raw_val_scores.append(vs_arr if vs_arr.ndim == 2 else vs_arr.reshape(1, -1))
        else:
            raw_val_scores.append(None)

        inst = r.get("instrumental_profiles")
        raw_inst.append(np.asarray(inst, dtype=float) if inst is not None else None)
        raw_factors.append(r.get("factors"))

        # y_cal_pred from this SBS model: predictions on the calibration set
        ycp = r.get("y_cal_pred")  # (n_cal, n_y) or (n_cal,) or None
        if ycp is not None:
            ycp_arr = np.asarray(ycp, dtype=float)
            raw_y_cal_pred.append(ycp_arr if ycp_arr.ndim == 2 else ycp_arr.reshape(-1, max(n_y, 1)))
        else:
            raw_y_cal_pred.append(None)

        cm = r.get("calibration_models")
        raw_cal_models.append(cm if isinstance(cm, list) else [])
        raw_fom.append(r.get("metrics", {}).get("calibration", {}) or {})
        if first_auto_mapping is None:
            first_auto_mapping = bool(r.get("auto_mapping_used", False))

        # y_val_pred from this SBS model: prediction for the single val sample
        yvp = r.get("y_val_pred")  # (1, n_y) or None
        if yvp is not None and np.asarray(yvp).size > 0:
            raw_y_val_pred.append(np.asarray(yvp, dtype=float).reshape(-1)[:n_y] if n_y > 0 else np.array([]))
        else:
            raw_y_val_pred.append(nan_row[:n_y] if n_y > 0 else np.array([]))

    # ------------------------------------------------------------------ #
    # Factor alignment (direct fused alignment to robust reference model) #
    # ------------------------------------------------------------------ #
    # Build per-model flattened instrumental and mode-A calibration arrays.
    inst_flat_list: List[Optional[np.ndarray]] = []
    a_cal_list: List[Optional[np.ndarray]] = []  # each item: (n_comp, n_cal)
    valid_model_indices: List[int] = []

    for i in range(n_val):
        inst_i = raw_inst[i]
        if inst_i is None:
            inst_flat_list.append(None)
            a_cal_list.append(None)
            continue

        inst_flat = np.asarray(inst_i, dtype=float).reshape(n_comp, -1)
        inst_flat_list.append(inst_flat)

        sc = raw_scores_cal[i]
        a_cal_i: Optional[np.ndarray] = None
        if sc is not None:
            sc_arr = np.asarray(sc, dtype=float)
            if sc_arr.ndim == 2 and sc_arr.shape[1] == n_comp:
                a_cal_i = sc_arr.T
        a_cal_list.append(a_cal_i)
        valid_model_indices.append(i)

    # Choose a robust reference model (medoid-like): the model with the best
    # mean pairwise fused assignment quality against all other valid models.
    ref_model_idx: Optional[int] = None
    if valid_model_indices:
        best_idx = valid_model_indices[0]
        best_score = float("-inf")
        for cand_idx in valid_model_indices:
            cand_inst = inst_flat_list[cand_idx]
            if cand_inst is None:
                continue

            pair_scores: List[float] = []
            for other_idx in valid_model_indices:
                other_inst = inst_flat_list[other_idx]
                if other_inst is None:
                    continue
                pair_scores.append(
                    _alignment_quality_score(
                        cand_inst,
                        other_inst,
                        ref_a_cal=a_cal_list[cand_idx],
                        a_cal=a_cal_list[other_idx],
                    )
                )

            if pair_scores:
                score = float(np.mean(pair_scores))
                if score > best_score:
                    best_score = score
                    best_idx = cand_idx
        ref_model_idx = best_idx

    perms: List[np.ndarray] = []
    signs: List[np.ndarray] = []
    for i in range(n_val):
        inst_i = inst_flat_list[i] if i < len(inst_flat_list) else None
        if inst_i is None or ref_model_idx is None:
            perms.append(np.arange(n_comp, dtype=int))
            signs.append(np.ones(n_comp, dtype=float))
            continue

        ref_inst_arr = inst_flat_list[ref_model_idx]
        if ref_inst_arr is None:
            perms.append(np.arange(n_comp, dtype=int))
            signs.append(np.ones(n_comp, dtype=float))
            continue

        p, s = _align_factors_to_reference(
            ref_inst_arr,
            inst_i,
            ref_a_cal=a_cal_list[ref_model_idx],
            a_cal=a_cal_list[i],
        )
        perms.append(p)
        signs.append(s)

    # ------------------------------------------------------------------ #
    # Build aligned arrays                                                 #
    # ------------------------------------------------------------------ #
    sbs_scores_cal_list: List[np.ndarray] = []
    sbs_val_scores_list: List[np.ndarray] = []
    sbs_inst_list: List[Optional[np.ndarray]] = []
    sbs_mode_profile_factors_list: List[Optional[np.ndarray]] = []
    sbs_calibration_coeffs_list: List[np.ndarray] = []  # (n_comp, 2) -> intercept, slope (aligned)
    ref_mode_profile_axes: Optional[np.ndarray] = None
    ref_mode_profile_labels: Optional[List[str]] = None
    ref_mode_profile_y_titles: Optional[List[str]] = None

    for i in range(n_val):
        p, s = perms[i], signs[i]
        n_cal_score_cols = n_comp

        # Cal A-scores
        if raw_scores_cal[i] is not None:
            sc = np.asarray(raw_scores_cal[i], dtype=float)[:, p] * s[np.newaxis, :]
        else:
            sc = np.full((n_cal, n_cal_score_cols), np.nan)
        sbs_scores_cal_list.append(sc)

        # Val A-score: raw_val_scores[i] is (1, n_comp) or None
        if raw_val_scores[i] is not None:
            vs_arr = np.asarray(raw_val_scores[i], dtype=float).reshape(1, n_comp)
            vs_arr = vs_arr[:, p] * s[np.newaxis, :]
        else:
            vs_arr = np.full((1, n_comp), np.nan)
        sbs_val_scores_list.append(vs_arr)

        # Instrumental profiles
        if raw_inst[i] is not None:
            ip_arr = np.asarray(raw_inst[i], dtype=float)
            ip_reordered = ip_arr[p] * s.reshape((-1,) + (1,) * (ip_arr.ndim - 1))
            sbs_inst_list.append(ip_reordered)
        else:
            sbs_inst_list.append(None)

        # Mode profiles (per-mode factor curves)
        if raw_factors[i] is not None:
            aligned_facs = _apply_perm_signs_to_parafac(raw_factors[i], p, s)
            mp = _build_mode_profile_outputs(
                factors=aligned_facs,
                axis_n_info=axis_n_info,
                dim_labels=dim_labels,
            )
            mpf = mp.get("mode_profile_factors")
            sbs_mode_profile_factors_list.append(np.asarray(mpf, dtype=float) if mpf is not None else None)
            if ref_mode_profile_axes is None:
                ref_mode_profile_axes = mp.get("mode_profile_axes")
                ref_mode_profile_labels = mp.get("mode_profile_mode_labels")
                ref_mode_profile_y_titles = mp.get("mode_profile_y_titles")
        else:
            sbs_mode_profile_factors_list.append(None)

        # Calibration coefficients aligned to reference component order/sign.
        # If aligned score is x' = s * x_old, then y = a + b*x_old = a + (b*s)*x'.
        coeff = np.full((n_comp, 2), np.nan, dtype=float)
        models_i = raw_cal_models[i] if i < len(raw_cal_models) else []
        if isinstance(models_i, list):
            for entry in models_i:
                if not isinstance(entry, dict):
                    continue
                comp_1b = entry.get("component")
                intercept = entry.get("intercept")
                slope = entry.get("slope")
                try:
                    old_idx = int(comp_1b) - 1
                    a = float(intercept)
                    b = float(slope)
                except (TypeError, ValueError):
                    continue
                if old_idx < 0 or old_idx >= n_comp:
                    continue
                ref_matches = np.where(p == old_idx)[0]
                if ref_matches.size <= 0:
                    continue
                ref_idx = int(ref_matches[0])
                coeff[ref_idx, 0] = a
                coeff[ref_idx, 1] = b * float(s[ref_idx])
        sbs_calibration_coeffs_list.append(coeff)

    # ------------------------------------------------------------------ #
    # Stack into (n_val, ...) arrays                                       #
    # ------------------------------------------------------------------ #
    sbs_scores_mode_a = np.stack(sbs_scores_cal_list, axis=0)            # (n_val, n_cal, n_comp)
    sbs_val_scores_mode_a = np.stack(sbs_val_scores_list, axis=0)        # (n_val, 1, n_comp)

    # Heatmap: (n_val, n_comp, n_cal)
    sbs_scores_mode_a_heatmap = sbs_scores_mode_a.transpose(0, 2, 1)
    sbs_scores_mode_a_full = np.concatenate([sbs_scores_mode_a, sbs_val_scores_mode_a], axis=1)
    sbs_scores_mode_a_heatmap_full = sbs_scores_mode_a_full.transpose(0, 2, 1)

    # Instrumental profiles: (n_val, n_comp, ...) or None if all failed
    inst_valid = [x for x in sbs_inst_list if x is not None]
    if inst_valid:
        # Determine a common shape
        inst_shape = inst_valid[0].shape     # (n_comp, ...)
        sbs_inst_arr = np.full((n_val,) + inst_shape, np.nan, dtype=float)
        for i, ip in enumerate(sbs_inst_list):
            if ip is not None:
                sbs_inst_arr[i] = ip
    else:
        sbs_inst_arr = None

    # Mode profile factors: (n_val, n_modes, n_comp, max_len)
    mpf_valid = [x for x in sbs_mode_profile_factors_list if x is not None]
    if mpf_valid:
        mpf_shape = mpf_valid[0].shape  # (n_modes, n_comp, max_len)
        sbs_mpf_arr = np.full((n_val,) + mpf_shape, np.nan, dtype=float)
        for i, mpf in enumerate(sbs_mode_profile_factors_list):
            if mpf is not None:
                sbs_mpf_arr[i] = mpf
    else:
        sbs_mpf_arr = None

    # ------------------------------------------------------------------ #
    # Similarity matrix: (n_comp, n_val, n_val)                           #
    # ------------------------------------------------------------------ #
    sbs_similarity = np.full((n_comp, n_val, n_val), np.nan, dtype=float)
    if sbs_inst_arr is not None:
        inst_flat_all = sbs_inst_arr.reshape(n_val, n_comp, -1)  # (n_val, n_comp, d)
        for f in range(n_comp):
            vecs = inst_flat_all[:, f, :]   # (n_val, d)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            units = vecs / norms
            sim_f = units @ units.T         # (n_val, n_val)
            sbs_similarity[f] = sim_f

    # ------------------------------------------------------------------ #
    # Assemble y_val_pred                                                  #
    # ------------------------------------------------------------------ #
    if n_y > 0 and raw_y_val_pred:
        y_val_pred_sbs = np.vstack([row.reshape(1, -1) for row in raw_y_val_pred])  # (n_val, n_y)
    else:
        y_val_pred_sbs = None

    # ------------------------------------------------------------------ #
    # Assemble sbs_y_cal_pred / sbs_y_cal_true / sbs_y_cal_error          #
    # All shaped (n_val, n_cal, n_y) for consistent table/graph slicing.  #
    # ------------------------------------------------------------------ #
    if n_y > 0:
        sbs_y_cal_pred_arr = np.full((n_val, n_cal, n_y), np.nan, dtype=float)
        for i, ycp in enumerate(raw_y_cal_pred):
            if ycp is not None:
                arr = np.asarray(ycp, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                rows = min(arr.shape[0], n_cal)
                cols = min(arr.shape[1], n_y)
                sbs_y_cal_pred_arr[i, :rows, :cols] = arr[:rows, :cols]
        # Y_cal_true is the same for every model — tile to match shape
        if Y_cal_2d is not None:
            y_cal_true_arr = np.asarray(Y_cal_2d, dtype=float)  # (n_cal, n_y)
            sbs_y_cal_true_arr = np.broadcast_to(
                y_cal_true_arr[np.newaxis, :, :], (n_val, n_cal, n_y)
            ).copy()
            sbs_y_cal_error_arr = sbs_y_cal_true_arr - sbs_y_cal_pred_arr
        else:
            sbs_y_cal_true_arr = None
            sbs_y_cal_error_arr = None
    else:
        sbs_y_cal_pred_arr = None
        sbs_y_cal_true_arr = None
        sbs_y_cal_error_arr = None

    factor_labels = [f"F{f + 1}" for f in range(n_comp)]
    val_sample_axis = np.arange(1, n_val + 1, dtype=float)
    mode_a_sample_axis = np.arange(1, n_cal + 1, dtype=float)
    mode_a_sample_axis_full = np.arange(1, n_cal + 2, dtype=float)
    mode_a_component_axis = np.arange(1, n_comp + 1, dtype=float)

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

    sbs_calibration_coeffs = np.stack(sbs_calibration_coeffs_list, axis=0)  # (n_val, n_comp, 2)

    # ------------------------------------------------------------------ #
    # Figures of merit ranges across all SBS models                        #
    # ------------------------------------------------------------------ #
    _fom_core: List[float] = []
    _fom_ssr: List[float] = []
    _fom_sfit: List[float] = []
    _fom_ev: List[float] = []
    _fom_iter: List[int] = []
    _fom_impl: Optional[str] = None
    for _fom in raw_fom:
        if not isinstance(_fom, dict):
            continue
        _v = _safe_float(_fom.get("core_consistency"), default=np.nan)
        if np.isfinite(_v):
            _fom_core.append(float(_v))
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

    sbs_fom_metrics: Dict[str, Any] = {
        "implementation_used": _fom_impl or "PARAFAC",
        "core_consistency": (_min_max(_fom_core) if _fom_core else None),
        "SSR": (_min_max(_fom_ssr) if _fom_ssr else None),
        "sfit": (_min_max(_fom_sfit) if _fom_sfit else None),
        "explained_variance": (_min_max(_fom_ev) if _fom_ev else None),
        "n_iter": ((min(_fom_iter), max(_fom_iter)) if _fom_iter else None),
    }

    return {
        "sbs_scores_mode_a": sbs_scores_mode_a,            # (n_val, n_cal, n_comp)
        "sbs_val_scores_mode_a": sbs_val_scores_mode_a,    # (n_val, 1, n_comp)
        "sbs_scores_mode_a_heatmap": sbs_scores_mode_a_heatmap,  # (n_val, n_comp, n_cal)
        "sbs_scores_mode_a_heatmap_full": sbs_scores_mode_a_heatmap_full,  # (n_val, n_comp, n_cal+1)
        "sbs_instrumental_profiles": sbs_inst_arr,          # (n_val, n_comp, ...) or None
        "sbs_mode_profile_factors": sbs_mpf_arr,            # (n_val, n_modes, n_comp, max_len) or None
        # Tile mode_profile_axes to (n_val, n_modes, max_len) so val-sample slicing
        # on dim 0 still leaves a (n_modes, max_len) array for the mode nav on dim 1.
        "sbs_mode_profile_axes": (
            np.tile(ref_mode_profile_axes[np.newaxis], (n_val, 1, 1))
            if ref_mode_profile_axes is not None else None
        ),
        "sbs_mode_profile_mode_labels": ref_mode_profile_labels,
        "sbs_mode_profile_y_titles": ref_mode_profile_y_titles,
        "sbs_similarity_matrix": sbs_similarity,            # (n_comp, n_val, n_val)
        # sbs_val_sample_axis tiled to (n_comp, n_val) so factor-slicing on dim 0
        # leaves a (n_val,) vector for the stability heatmap x/y axes.
        "sbs_val_sample_axis": np.tile(val_sample_axis, (n_comp, 1)),
        # sbs_mode_a_sample_axis tiled to (n_val, n_cal) so val-sample slicing on
        # dim 0 leaves a (n_cal,) vector for the A-scores heatmap x-axis.
        "sbs_mode_a_sample_axis": np.tile(mode_a_sample_axis, (n_val, 1)),
        # Full sample axis includes the appended validation sample.
        "sbs_mode_a_sample_axis_full": np.tile(mode_a_sample_axis_full, (n_val, 1)),
        # sbs_mode_a_component_axis tiled to (n_val, n_comp) so val-sample slicing on
        # dim 0 leaves a (n_comp,) vector for the A-scores heatmap y-axis.
        "sbs_mode_a_component_axis": np.tile(mode_a_component_axis, (n_val, 1)),
        "sbs_factor_labels": factor_labels,
        "sbs_y_cal_pred": sbs_y_cal_pred_arr,          # (n_val, n_cal, n_y)
        "sbs_y_cal_true": sbs_y_cal_true_arr,           # (n_val, n_cal, n_y)
        "sbs_y_cal_error": sbs_y_cal_error_arr,         # (n_val, n_cal, n_y)
        # (n_val, 1, n_y) — val-sample dim 0 + singleton dim 1 align with sbs_val_scores_mode_a
        "sbs_y_val_true": sbs_y_val_true_3d,
        "sbs_y_val_pred": sbs_y_val_pred_3d,
        "y_val_pred": y_val_pred_sbs,
        "y_val_error": sbs_y_val_error_2d,
        "sbs_calibration_coeffs": sbs_calibration_coeffs,
        "sbs_fom_metrics": sbs_fom_metrics,
        "auto_mapping_used": first_auto_mapping if first_auto_mapping is not None else False,
    }


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
    constraint_l1_reg_strength: Optional[Any] = None,
    constraint_l2_reg: Optional[Any] = None,
    constraint_l2_reg_strength: Optional[Any] = None,
    constraint_l2_square_reg: Optional[Any] = None,
    constraint_l2_square_reg_strength: Optional[Any] = None,
    constraint_unimodality: Optional[Any] = None,
    constraint_normalize: Optional[Any] = None,
    constraint_simplex: Optional[Any] = None,
    constraint_simplex_strength: Optional[Any] = None,
    constraint_normalized_sparsity: Optional[Any] = None,
    constraint_normalized_sparsity_strength: Optional[Any] = None,
    constraint_soft_sparsity: Optional[Any] = None,
    constraint_soft_sparsity_strength: Optional[Any] = None,
    constraint_smoothness: Optional[Any] = None,
    constraint_smoothness_strength: Optional[Any] = None,
    constraint_monotonicity: Optional[Any] = None,
    constraint_hard_sparsity: Optional[Any] = None,
    constraint_hard_sparsity_strength: Optional[Any] = None,
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
    orient_mostly_negative_pairs: Any = True,
    sweep_mode: bool = False,
    component_range: str = "",
    component_y_mapping: Any = "",
    cv_config: Optional[Any] = None,
    fold: int = 0,
    axis_n_info: Optional[List[np.ndarray]] = None,
    dim_labels: Optional[List[str]] = None,
    y_labels: Optional[Any] = None,
    validation_processing: str = "batch",
    **kwargs: Any,
) -> Dict[str, Any]:
    """PARAFAC with missing-data handling, sweep mode, calibration, and CV."""

    X_cal_test = kwargs.get("X_cal_test")
    Y_cal_test = kwargs.get("Y_cal_test")
    y_labels_raw = y_labels if y_labels is not None else kwargs.get("y_labels")
    y_labels_resolved = _normalize_y_labels(y_labels_raw)

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
        constraint_l1_reg_strength=constraint_l1_reg_strength,
        constraint_l2_reg=constraint_l2_reg,
        constraint_l2_reg_strength=constraint_l2_reg_strength,
        constraint_l2_square_reg=constraint_l2_square_reg,
        constraint_l2_square_reg_strength=constraint_l2_square_reg_strength,
        constraint_unimodality=constraint_unimodality,
        constraint_normalize=constraint_normalize,
        constraint_simplex=constraint_simplex,
        constraint_simplex_strength=constraint_simplex_strength,
        constraint_normalized_sparsity=constraint_normalized_sparsity,
        constraint_normalized_sparsity_strength=constraint_normalized_sparsity_strength,
        constraint_soft_sparsity=constraint_soft_sparsity,
        constraint_soft_sparsity_strength=constraint_soft_sparsity_strength,
        constraint_smoothness=constraint_smoothness,
        constraint_smoothness_strength=constraint_smoothness_strength,
        constraint_monotonicity=constraint_monotonicity,
        constraint_hard_sparsity=constraint_hard_sparsity,
        constraint_hard_sparsity_strength=constraint_hard_sparsity_strength,
    )
    seed_value = _safe_optional_int(random_state, default=None)
    multi_start_runs_value = _safe_optional_int(random_multi_start_runs, default=5)
    if multi_start_runs_value is None:
        multi_start_runs_value = 5
    multi_start_runs_value = int(max(1, multi_start_runs_value))
    solver_mode = _normalize_missing_constrained_solver(missing_constrained_solver)
    orient_negative_pairs_flag = _safe_bool(orient_mostly_negative_pairs, default=True)
    path_list, usage_list = _expand_profile_mode_settings(profile_paths, profile_usage, n_modes)

    requested_validation_processing = str(validation_processing).strip().lower()
    use_sbs = (
        requested_validation_processing == "sample_by_sample"
        and X_val is not None
        and np.asarray(X_val).shape[0] > 0
    )
    if use_sbs and bool(kwargs.get('__passforward_enabled__', False)):
        emit_execution_warning(
            code="parafac_sbs_passforward_fallback_batch",
            text=(
                "Sample-by-Sample validation is incompatible with passforward output mode. "
                "Falling back to Batch validation processing."
            ),
        )
        use_sbs = False

    effective_validation_processing = "sample_by_sample" if use_sbs else "batch"

    run_sweep = bool(sweep_mode)
    if use_sbs and run_sweep:
        emit_execution_warning(
            code="parafac_sbs_sweep_first_layer",
            text=(
                "Sweep mode in Sample-by-Sample validation uses only the first SBS model "
                "(first validation sample) as sweep reference."
            ),
        )
    sweep_x_val = np.asarray(X_val, dtype=float)[0:1] if (use_sbs and X_val is not None) else X_val
    sweep_y_val = None
    if use_sbs and Y_val is not None:
        yv_arr = np.asarray(Y_val, dtype=float)
        if yv_arr.ndim == 1:
            sweep_y_val = yv_arr[0:1]
        elif yv_arr.ndim >= 2 and yv_arr.shape[0] > 0:
            sweep_y_val = yv_arr[0:1, ...]
        else:
            sweep_y_val = None

    # Sweep mode
    sweep_results: List[Dict[str, Any]] = []
    selected_rank = int(n_components)
    result: Dict[str, Any] = {}

    if run_sweep:
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
                X_val=sweep_x_val,
                Y_val=sweep_y_val,
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
                orient_mostly_negative_pairs=orient_negative_pairs_flag,
                profile_paths=path_list,
                profile_usage=usage_list,
                component_y_mapping=component_y_mapping,
                emit_missing_solver_notice=not bool(run_sweep),
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
        if not use_sbs:
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
                orient_mostly_negative_pairs=orient_negative_pairs_flag,
                profile_paths=path_list,
                profile_usage=usage_list,
                component_y_mapping=component_y_mapping,
                emit_missing_solver_notice=not bool(run_sweep),
            )
    else:
        if not use_sbs:
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
                orient_mostly_negative_pairs=orient_negative_pairs_flag,
                profile_paths=path_list,
                profile_usage=usage_list,
                component_y_mapping=component_y_mapping,
                emit_missing_solver_notice=True,
            )

    report_text = (
        _build_text_report(
            rank=selected_rank,
            metrics=result.get("metrics", {}),
            mapping={int(k) - 1: int(v) - 1 for k, v in result.get("component_y_mapping", {}).items()},
            sweep_results=sweep_results if run_sweep else None,
            y_labels=y_labels_resolved,
            auto_mapping_used=bool(result.get("auto_mapping_used", False)),
            validation_processing=effective_validation_processing.replace("_", " ").title(),
        )
        if not use_sbs
        else ""
    )

    output = {
        "scores_mode_a": result.get("scores_mode_a"),
        "val_scores_mode_a": result.get("val_scores_mode_a"),
        "scores_mode_a_heatmap": None,
        "scores_mode_a_heatmap_full": None,
        "mode_a_sample_axis": None,
        "mode_a_sample_axis_full": None,
        "mode_a_component_axis": None,
        "factor_labels": None,
        "mode_profile_axes": None,
        "mode_profile_axes_full_a": None,
        "mode_profile_factors": None,
        "mode_profile_factors_full_a": None,
        "mode_profile_mode_labels": None,
        "mode_profile_y_titles": None,
        "mode_profile_component_labels": None,
        "mode_profile_navigation_labels_by_dimension": None,
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
        "sweep_results": sweep_results if run_sweep else None,
        "selected_n_components": int(selected_rank),
        "axis_n_info": axis_n_info,
        "dim_labels": dim_labels,
        "nway_flag": int(nway_flag) if nway_flag is not None else int(max(1, X_arr.ndim - 1)),
        "cv_results": result.get("cv_results"),
        "y_cv_pred": result.get("y_cv_pred"),
        "y_cal_pred": result.get("y_cal_pred"),
        "y_val_pred": result.get("y_val_pred"),
        "y_cal_error": result.get("y_cal_error"),
        "y_val_error": result.get("y_val_error"),
        "y_cv_error": result.get("y_cv_error"),
        "y_cal_true": _as_2d_y(Y_cal),
        "y_val_true": _as_2d_y(Y_val),
    }

    if output.get("y_cv_error") is None and output.get("y_cv_pred") is not None and output.get("y_cal_true") is not None:
        output["y_cv_error"] = np.asarray(output["y_cal_true"], dtype=float) - np.asarray(output["y_cv_pred"], dtype=float)

    if not use_sbs:
        pair_outputs = _build_component_pair_prediction_outputs(
            component_y_mapping=output.get("component_y_mapping"),
            y_labels=y_labels_resolved,
            y_cal_true=output.get("y_cal_true"),
            y_val_true=output.get("y_val_true"),
            scores_mode_a=output.get("scores_mode_a"),
            val_scores_mode_a=output.get("val_scores_mode_a"),
            calibration_models=result.get("calibration_models"),
        )
        output.update(pair_outputs)

        # Override unified prediction matrices: pick best pairing per Y column,
        # NaN for un-paired columns, highest-cal-correlation pairing for conflicts.
        pair_y_cols = output.get("parafac_pair_y_columns")
        pair_y_cols_list: List[int] = (
            pair_y_cols.tolist() if isinstance(pair_y_cols, np.ndarray) else
            list(pair_y_cols) if pair_y_cols is not None else []
        )
        output.update(
            _build_unified_prediction_matrices(
                y_cal_true=output.get("y_cal_true"),
                y_val_true=output.get("y_val_true"),
                y_cal_pred_pairs=output.get("y_cal_pred_pairs"),
                y_val_pred_pairs=output.get("y_val_pred_pairs"),
                pair_y_columns=pair_y_cols_list,
            )
        )

    scores_mode_a = result.get("scores_mode_a")
    if isinstance(scores_mode_a, np.ndarray) and scores_mode_a.ndim == 2:
        output["scores_mode_a_heatmap"] = np.asarray(scores_mode_a, dtype=float).T
        output["mode_a_sample_axis"] = np.arange(1, int(scores_mode_a.shape[0]) + 1, dtype=float)
        output["mode_a_component_axis"] = np.arange(1, int(scores_mode_a.shape[1]) + 1, dtype=float)
        output["factor_labels"] = [f"F{i + 1}" for i in range(scores_mode_a.shape[1])]
        val_scores_mode_a = output.get("val_scores_mode_a")
        if isinstance(val_scores_mode_a, np.ndarray) and val_scores_mode_a.ndim == 2 and val_scores_mode_a.shape[1] == scores_mode_a.shape[1]:
            scores_mode_a_full = np.vstack([scores_mode_a, val_scores_mode_a])
        else:
            scores_mode_a_full = np.asarray(scores_mode_a, dtype=float)
        output["scores_mode_a_heatmap_full"] = np.asarray(scores_mode_a_full, dtype=float).T
        output["mode_a_sample_axis_full"] = np.arange(1, int(scores_mode_a_full.shape[0]) + 1, dtype=float)
    else:
        output.setdefault("factor_labels", None)

    if sweep_results:
        sweep_items = [item for item in sweep_results if isinstance(item, dict)]
        if sweep_items:
            output["sweep_F"] = np.asarray([_safe_float(item.get("n_components")) for item in sweep_items], dtype=float)
            output["sweep_sfit"] = np.asarray([_safe_float(item.get("sfit")) for item in sweep_items], dtype=float)
            output["sweep_core_consistency"] = np.asarray([_safe_float(item.get("core_consistency")) for item in sweep_items], dtype=float)
            output["sweep_explained_variance"] = np.asarray([_safe_float(item.get("explained_variance")) for item in sweep_items], dtype=float)
            output["sweep_n_iter"] = np.asarray([_safe_float(item.get("n_iter")) for item in sweep_items], dtype=float)

    output.update(
        _build_mode_profile_outputs(
            factors=result.get("factors"),
            axis_n_info=axis_n_info,
            dim_labels=dim_labels,
        )
    )
    output.update(
        _build_mode_profile_outputs_with_full_a(
            mode_profile_axes=output.get("mode_profile_axes"),
            mode_profile_factors=output.get("mode_profile_factors"),
            scores_mode_a=output.get("scores_mode_a"),
            val_scores_mode_a=output.get("val_scores_mode_a"),
        )
    )

    # ------------------------------------------------------------------ #
    # Sample-by-sample validation                                          #
    # ------------------------------------------------------------------ #
    output["validation_processing"] = effective_validation_processing

    # SBS null defaults so keys are always present in output
    for _sbs_key in (
        "sbs_scores_mode_a", "sbs_val_scores_mode_a", "sbs_scores_mode_a_heatmap", "sbs_scores_mode_a_heatmap_full",
        "sbs_instrumental_profiles", "sbs_mode_profile_factors", "sbs_mode_profile_axes",
        "sbs_mode_profile_mode_labels", "sbs_mode_profile_y_titles",
        "sbs_similarity_matrix", "sbs_val_sample_axis",
        "sbs_mode_a_sample_axis", "sbs_mode_a_sample_axis_full", "sbs_mode_a_component_axis", "sbs_factor_labels",
        "sbs_y_cal_pred", "sbs_y_cal_true", "sbs_y_cal_error",
        "sbs_y_val_true", "sbs_y_val_pred",
        "sbs_a_scores_cal_pairs", "sbs_a_scores_val_pairs",
        "sbs_y_cal_pred_pairs", "sbs_y_cal_true_pairs", "sbs_y_cal_error_pairs",
        "sbs_y_val_true_pairs", "sbs_y_val_pred_pairs", "sbs_y_val_effective_pairs", "sbs_y_val_error_pairs",
        "sbs_cal_regression_line_x_pairs", "sbs_cal_regression_line_y_pairs",
    ):
        output[_sbs_key] = None

    if use_sbs:
        sbs_out = _sbs_parafac(
            X_cal=X_arr,
            Y_cal=Y_cal,
            X_val=np.asarray(X_val, dtype=float),
            Y_val=Y_val,
            n_components=int(selected_rank),
            nway_flag=nway_flag,
            init_method=init_method,
            max_iter=max_iter,
            tol=tol,
            random_state=seed_value,
            random_multi_start=random_multi_start,
            random_multi_start_runs=multi_start_runs_value,
            mode_centering=center_list,
            mode_normalization=norm_list,
            profile_paths=path_list,
            profile_usage=usage_list,
            tensorly_constraints=parsed_constraints,
            unconstrained_sparsity=unconstrained_sparsity,
            unconstrained_linesearch=unconstrained_linesearch,
            unconstrained_orthogonalise=unconstrained_orthogonalise,
            missing_constrained_solver=solver_mode,
            component_y_mapping=component_y_mapping,
            orient_mostly_negative_pairs=orient_negative_pairs_flag,
            axis_n_info=axis_n_info,
            dim_labels=dim_labels,
        )
        for k in (
            "sbs_scores_mode_a", "sbs_val_scores_mode_a", "sbs_scores_mode_a_heatmap", "sbs_scores_mode_a_heatmap_full",
            "sbs_instrumental_profiles", "sbs_mode_profile_factors", "sbs_mode_profile_axes",
            "sbs_mode_profile_mode_labels", "sbs_mode_profile_y_titles",
            "sbs_similarity_matrix", "sbs_val_sample_axis",
            "sbs_mode_a_sample_axis", "sbs_mode_a_sample_axis_full", "sbs_mode_a_component_axis", "sbs_factor_labels",
            "sbs_y_cal_pred", "sbs_y_cal_true", "sbs_y_cal_error",
            "sbs_y_val_true", "sbs_y_val_pred",
        ):
            output[k] = sbs_out.get(k)
        output["auto_mapping_used"] = bool(sbs_out.get("auto_mapping_used", False))
        output["sbs_fom_metrics"] = sbs_out.get("sbs_fom_metrics")

        # Build SBS-specific component->Y mapping from aligned SBS calibration data,
        # instead of reusing batch/global mapping.
        _sbs_scores_all = output.get("sbs_scores_mode_a")
        _sbs_y_cal_true_all = output.get("sbs_y_cal_true")
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
                output["component_y_mapping"] = _sbs_mapping

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
                    _pair_labels.append(f"F{_c1} -> {_title} (Y{_y1})")

                output["parafac_pair_components"] = np.asarray(_pair_components, dtype=int)
                output["parafac_pair_y_columns"] = np.asarray(_pair_y_columns, dtype=int)
                output["parafac_pair_y_titles"] = _pair_y_titles
                output["parafac_pairing_labels"] = _pair_labels
                output["parafac_pairing_labels_by_dimension"] = [[], _pair_labels]

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

        pair_components = (
            output.get("parafac_pair_components").tolist()
            if isinstance(output.get("parafac_pair_components"), np.ndarray)
            else list(output.get("parafac_pair_components") or [])
        )
        pair_y_columns = (
            output.get("parafac_pair_y_columns").tolist()
            if isinstance(output.get("parafac_pair_y_columns"), np.ndarray)
            else list(output.get("parafac_pair_y_columns") or [])
        )

        output["sbs_a_scores_cal_pairs"] = _extract_pairs_3d(output.get("sbs_scores_mode_a"), pair_components)
        output["sbs_a_scores_val_pairs"] = _extract_pairs_3d(output.get("sbs_val_scores_mode_a"), pair_components)

        def _extract_true_pairs_3d(data: Any, indices_1based: List[int]) -> Optional[np.ndarray]:
            """Extract pair-wise Y columns from (n_val, n_samples, n_y) arrays."""
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

        output["sbs_y_cal_true_pairs"] = _extract_true_pairs_3d(output.get("sbs_y_cal_true"), pair_y_columns)
        output["sbs_y_val_true_pairs"] = _extract_true_pairs_3d(output.get("sbs_y_val_true"), pair_y_columns)

        _sbs_a_cal_pairs = output.get("sbs_a_scores_cal_pairs")
        _sbs_a_val_pairs = output.get("sbs_a_scores_val_pairs")
        _sbs_y_cal_true_pairs = output.get("sbs_y_cal_true_pairs")
        _sbs_y_val_true_pairs = output.get("sbs_y_val_true_pairs")

        _sbs_y_cal_pred_pairs: Optional[np.ndarray] = None
        _sbs_y_val_pred_pairs: Optional[np.ndarray] = None

        _pair_intercepts: Optional[np.ndarray] = None
        _pair_slopes: Optional[np.ndarray] = None

        if (
            isinstance(_sbs_a_cal_pairs, np.ndarray)
            and _sbs_a_cal_pairs.ndim == 3
            and isinstance(_sbs_y_cal_true_pairs, np.ndarray)
            and _sbs_y_cal_true_pairs.ndim == 3
            and _sbs_a_cal_pairs.shape[:2] == _sbs_y_cal_true_pairs.shape[:2]
            and _sbs_a_cal_pairs.shape[2] == _sbs_y_cal_true_pairs.shape[2]
        ):
            _n_val_sbs = int(_sbs_a_cal_pairs.shape[0])
            _n_cal = int(_sbs_a_cal_pairs.shape[1])
            _n_pairs = int(_sbs_a_cal_pairs.shape[2])
            _pair_intercepts = np.full((_n_val_sbs, _n_pairs), np.nan, dtype=float)
            _pair_slopes = np.full((_n_val_sbs, _n_pairs), np.nan, dtype=float)
            _pred_cal = np.full((_n_val_sbs, _n_cal, _n_pairs), np.nan, dtype=float)
            for _val_idx in range(_n_val_sbs):
                for _pair_idx, _comp_1based in enumerate(pair_components[:_n_pairs]):
                    _x = np.asarray(_sbs_a_cal_pairs[_val_idx, :, _pair_idx], dtype=float)
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
                    _col = _sbs_a_cal_pairs[_val_idx, :, _pair_idx]
                    _mask = np.isfinite(_col)
                    _pred_cal[_val_idx, _mask, _pair_idx] = _intercept + _slope * _col[_mask]
            _sbs_y_cal_pred_pairs = _pred_cal

        if (
            isinstance(_sbs_a_val_pairs, np.ndarray)
            and _sbs_a_val_pairs.ndim == 3
            and isinstance(_pair_intercepts, np.ndarray)
            and isinstance(_pair_slopes, np.ndarray)
        ):
            _n_val_sbs = int(_sbs_a_val_pairs.shape[0])
            _n_val_rows = int(_sbs_a_val_pairs.shape[1])
            _n_pairs = int(_sbs_a_val_pairs.shape[2])
            _pred_val = np.full((_n_val_sbs, _n_val_rows, _n_pairs), np.nan, dtype=float)
            for _val_idx in range(_n_val_sbs):
                for _pair_idx, _comp_1based in enumerate(pair_components[:_n_pairs]):
                    _intercept = float(_pair_intercepts[_val_idx, _pair_idx])
                    _slope = float(_pair_slopes[_val_idx, _pair_idx])
                    if not (np.isfinite(_intercept) and np.isfinite(_slope)):
                        continue
                    _col = _sbs_a_val_pairs[_val_idx, :, _pair_idx]
                    _mask = np.isfinite(_col)
                    _pred_val[_val_idx, _mask, _pair_idx] = _intercept + _slope * _col[_mask]
            _sbs_y_val_pred_pairs = _pred_val

        output["sbs_y_cal_pred_pairs"] = _sbs_y_cal_pred_pairs
        output["sbs_y_val_pred_pairs"] = _sbs_y_val_pred_pairs
        output["sbs_y_cal_error_pairs"] = (
            np.asarray(_sbs_y_cal_true_pairs, dtype=float) - np.asarray(_sbs_y_cal_pred_pairs, dtype=float)
            if (_sbs_y_cal_true_pairs is not None and _sbs_y_cal_pred_pairs is not None)
            else None
        )

        _yv_true_pairs = _sbs_y_val_true_pairs
        _yv_pred_pairs = _sbs_y_val_pred_pairs
        output["sbs_y_val_effective_pairs"] = (
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
        output["sbs_y_val_error_pairs"] = (
            np.asarray(_yv_true_pairs, dtype=float) - np.asarray(_yv_pred_pairs, dtype=float)
            if (_yv_true_pairs is not None and _yv_pred_pairs is not None)
            else None
        )

        # Keep legacy pairwise keys SBS-consistent in sample-by-sample mode.
        # This prevents mixed semantics where some pages use global/batch-like
        # pair matrices while SBS tables use per-model predictions.
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

        _yv_true_pairs_2d = _collapse_sbs_val_pairs(output.get("sbs_y_val_true_pairs"))
        _yv_pred_pairs_2d = _collapse_sbs_val_pairs(output.get("sbs_y_val_pred_pairs"))
        _yv_eff_pairs_2d = _collapse_sbs_val_pairs(output.get("sbs_y_val_effective_pairs"))
        _yv_err_pairs_2d = (
            _yv_true_pairs_2d - _yv_pred_pairs_2d
            if (_yv_true_pairs_2d is not None and _yv_pred_pairs_2d is not None)
            else None
        )

        output["y_val_true_pairs"] = _yv_true_pairs_2d
        output["y_val_pred_pairs"] = _yv_pred_pairs_2d
        output["y_val_effective_pairs"] = _yv_eff_pairs_2d
        output["y_val_error_pairs"] = _yv_err_pairs_2d

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
                output["val_pred_ref_diag_extent"] = _diag_extent
                output["val_pred_ref_diag_x"] = np.array([-_diag_extent, _diag_extent], dtype=float)
                output["val_pred_ref_diag_y"] = np.array([-_diag_extent, _diag_extent], dtype=float)
            else:
                output["val_pred_ref_diag_extent"] = None
                output["val_pred_ref_diag_x"] = None
                output["val_pred_ref_diag_y"] = None
        else:
            output["val_pred_ref_diag_extent"] = None
            output["val_pred_ref_diag_x"] = None
            output["val_pred_ref_diag_y"] = None

        _line_x_pairs = output.get("cal_regression_line_x_pairs")
        _line_y_pairs = output.get("cal_regression_line_y_pairs")
        _sbs_scores = output.get("sbs_scores_mode_a")
        _sbs_pair_scores = output.get("sbs_a_scores_cal_pairs")
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
                        isinstance(_sbs_y_cal_true_pairs, np.ndarray)
                        and _sbs_y_cal_true_pairs.ndim == 3
                        and _pair_idx < _sbs_y_cal_true_pairs.shape[2]
                    ):
                        _ref_col = _sbs_y_cal_true_pairs[_val_idx, :, _pair_idx]
                        _finite_ref = np.isfinite(_ref_col)
                        if np.any(_finite_ref):
                            _ref_min = float(np.nanmin(_ref_col[_finite_ref]))
                            _ref_max = float(np.nanmax(_ref_col[_finite_ref]))
                            _ref_span = _ref_max - _ref_min if _ref_max > _ref_min else 1.0
                            _ref_buf = _ref_span * 0.05
                            _x1 = _ref_min - _ref_buf
                            _x2 = _ref_max + _ref_buf
                            _y1 = (_x1 - _intercept) / _slope
                            _y2 = (_x2 - _intercept) / _slope

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
            output["sbs_cal_regression_line_x_pairs"] = _line_x
            output["sbs_cal_regression_line_y_pairs"] = _line_y
        elif isinstance(_sbs_scores, np.ndarray) and _sbs_scores.ndim == 3:
            _n_val_sbs = int(_sbs_scores.shape[0])
            if isinstance(_line_x_pairs, np.ndarray) and _line_x_pairs.ndim == 2:
                output["sbs_cal_regression_line_x_pairs"] = np.broadcast_to(
                    _line_x_pairs[np.newaxis, :, :], (_n_val_sbs,) + _line_x_pairs.shape
                ).copy()
            if isinstance(_line_y_pairs, np.ndarray) and _line_y_pairs.ndim == 2:
                output["sbs_cal_regression_line_y_pairs"] = np.broadcast_to(
                    _line_y_pairs[np.newaxis, :, :], (_n_val_sbs,) + _line_y_pairs.shape
                ).copy()

        if sbs_out.get("y_val_pred") is not None:
            output["y_val_pred"] = sbs_out["y_val_pred"]
            output["y_val_error"] = sbs_out.get("y_val_error")

        # SBS mode-profile variant with full A mode (Cal + current Val sample).
        _sbs_axes = output.get("sbs_mode_profile_axes")
        _sbs_factors = output.get("sbs_mode_profile_factors")
        _sbs_scores_cal = output.get("sbs_scores_mode_a")
        _sbs_scores_val = output.get("sbs_val_scores_mode_a")
        output["sbs_mode_profile_axes_full_a"] = _sbs_axes
        output["sbs_mode_profile_factors_full_a"] = _sbs_factors
        if (
            isinstance(_sbs_axes, np.ndarray)
            and _sbs_axes.ndim == 3
            and isinstance(_sbs_factors, np.ndarray)
            and _sbs_factors.ndim == 4
            and isinstance(_sbs_scores_cal, np.ndarray)
            and _sbs_scores_cal.ndim == 3
            and isinstance(_sbs_scores_val, np.ndarray)
            and _sbs_scores_val.ndim == 3
        ):
            _n_val = int(_sbs_scores_cal.shape[0])
            _n_modes = int(_sbs_factors.shape[1])
            _n_comp = int(_sbs_factors.shape[2])
            _old_len = int(_sbs_factors.shape[3])
            _full_len = int(_sbs_scores_cal.shape[1] + _sbs_scores_val.shape[1])
            _target_len = int(max(_old_len, _full_len))

            _axes_full = np.full((_n_val, _n_modes, _target_len), np.nan, dtype=float)
            _axes_full[:, :, :min(_old_len, _target_len)] = _sbs_axes[:, :, :min(_old_len, _target_len)]
            _axes_full[:, 0, :_full_len] = np.arange(1, _full_len + 1, dtype=float)[np.newaxis, :]

            _factors_full = np.full((_n_val, _n_modes, _n_comp, _target_len), np.nan, dtype=float)
            _factors_full[:, :, :, :min(_old_len, _target_len)] = _sbs_factors[:, :, :, :min(_old_len, _target_len)]
            _scores_full = np.concatenate([_sbs_scores_cal, _sbs_scores_val], axis=1)  # (n_val, n_samples_full, n_comp)
            _n_comp_copy = min(_n_comp, int(_scores_full.shape[2]))
            _factors_full[:, 0, :_n_comp_copy, :_full_len] = np.transpose(
                _scores_full[:, :, :_n_comp_copy], (0, 2, 1)
            )

            output["sbs_mode_profile_axes_full_a"] = _axes_full
            output["sbs_mode_profile_factors_full_a"] = _factors_full

        output["parafac_report"] = _build_sbs_text_report(
            rank=int(selected_rank),
            pair_labels=list(output.get("parafac_pairing_labels") or []),
            sbs_y_cal_true_pairs=output.get("sbs_y_cal_true_pairs"),
            sbs_y_cal_pred_pairs=output.get("sbs_y_cal_pred_pairs"),
            y_val_true_pairs=output.get("y_val_true_pairs"),
            y_val_pred_pairs=output.get("y_val_pred_pairs"),
            auto_mapping_used=bool(output.get("auto_mapping_used", False)),
            sbs_fom_metrics=output.get("sbs_fom_metrics"),
        )
        if sweep_results:
            sweep_lines = ["", "Sweep results (first SBS model only):"]
            for item in sweep_results:
                sweep_lines.append(
                    f"- F={item.get('n_components')} | sfit={_safe_float(item.get('sfit')):.6g} | "
                    f"core={_safe_float(item.get('core_consistency')):.4f} | "
                    f"EV={_safe_float(item.get('explained_variance')):.4f}%"
                )
            output["parafac_report"] = str(output.get("parafac_report") or "") + "\n".join(sweep_lines)

    # ─── EJCR packed payloads ────────────────────────────────────────────────
    _ejcr_n_pts = 100
    _ejcr_n_path = _ejcr_n_pts * 2 + 1

    _ejcr_n_pairs = 1
    _ejcr_pairs_ref = output.get("y_cal_pred_pairs")
    if (
        _ejcr_pairs_ref is not None
        and hasattr(_ejcr_pairs_ref, "shape")
        and _ejcr_pairs_ref.ndim >= 2
    ):
        _ejcr_n_pairs = max(_ejcr_n_pairs, int(_ejcr_pairs_ref.shape[1]))

    _sbs_ref = output.get("sbs_y_cal_pred_pairs")
    if (
        _sbs_ref is not None
        and hasattr(_sbs_ref, "shape")
        and _sbs_ref.ndim >= 3
    ):
        _ejcr_n_pairs = max(_ejcr_n_pairs, int(_sbs_ref.shape[2]))

    _pair_labels = output.get("parafac_pairing_labels")
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

        _cal_true_p = output.get("y_cal_true_pairs")
        _cal_pred_p = output.get("y_cal_pred_pairs")
        if (
            _cal_true_p is not None and _cal_pred_p is not None
            and hasattr(_cal_true_p, "shape") and _cal_true_p.ndim == 2
        ):
            _nc = min(_cal_true_p.shape[1], _cal_pred_p.shape[1], _ejcr_n_pairs)
            for _p in range(_nc):
                _fill_packed(_cal_true_p[:, _p], _cal_pred_p[:, _p], ejcr_cal, (_p,))

        _val_true_p = output.get("y_val_true_pairs")
        _val_pred_p = output.get("y_val_pred_pairs")
        if (
            _val_true_p is not None and _val_pred_p is not None
            and hasattr(_val_true_p, "shape") and _val_true_p.ndim == 2
        ):
            _nc = min(_val_true_p.shape[1], _val_pred_p.shape[1], _ejcr_n_pairs)
            for _p in range(_nc):
                _fill_packed(_val_true_p[:, _p], _val_pred_p[:, _p], ejcr_val, (_p,))

        _sbs_cal_true = output.get("sbs_y_cal_true_pairs")
        _sbs_cal_pred = output.get("sbs_y_cal_pred_pairs")
        if (
            _sbs_cal_true is not None and _sbs_cal_pred is not None
            and hasattr(_sbs_cal_true, "shape") and _sbs_cal_true.ndim == 3
        ):
            _n_sbs_c, _, _n_p_c = _sbs_cal_true.shape
            for _s in range(min(_n_sbs_c, _ejcr_n_sbs)):
                for _p in range(min(_n_p_c, _ejcr_n_pairs)):
                    _fill_packed(_sbs_cal_true[_s, :, _p], _sbs_cal_pred[_s, :, _p], sbs_ejcr_cal, (_p, _s))

        _sbs_val_true = output.get("sbs_y_val_true_pairs")
        _sbs_val_pred = output.get("sbs_y_val_pred_pairs")
        if (
            _sbs_val_true is not None and _sbs_val_pred is not None
            and hasattr(_sbs_val_true, "shape") and _sbs_val_true.ndim == 3
        ):
            _n_sbs_v, _, _n_p_v = _sbs_val_true.shape
            for _p in range(min(_n_p_v, _ejcr_n_pairs)):
                _fill_packed(_sbs_val_true[:_n_sbs_v, 0, _p], _sbs_val_pred[:_n_sbs_v, 0, _p], sbs_ejcr_val, (_p,))

    except Exception:
        pass

    output.update({
        "ejcr_cal": ejcr_cal,
        "ejcr_val": ejcr_val,
        "sbs_ejcr_cal": sbs_ejcr_cal,
        "sbs_ejcr_val": sbs_ejcr_val,
    })

    return output


_PARAFAC_RETURN_ORDER: Tuple[str, ...] = (
    "scores_mode_a",
    "val_scores_mode_a",
    "scores_mode_a_heatmap",
    "scores_mode_a_heatmap_full",
    "mode_a_sample_axis",
    "mode_a_sample_axis_full",
    "mode_a_component_axis",
    "mode_profile_axes",
    "mode_profile_axes_full_a",
    "mode_profile_factors",
    "mode_profile_factors_full_a",
    "mode_profile_mode_labels",
    "mode_profile_y_titles",
    "mode_profile_component_labels",
    "mode_profile_navigation_labels_by_dimension",
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
    "cv_results",
    "y_cv_pred",
    "y_cal_pred",
    "y_val_pred",
    "y_cal_error",
    "y_val_error",
    "y_cv_error",
    "y_cal_true",
    "y_val_true",
    "parafac_pair_components",
    "parafac_pair_y_columns",
    "parafac_pair_y_titles",
    "parafac_pairing_labels",
    "parafac_pairing_labels_by_dimension",
    "y_cal_pred_pairs",
    "y_cal_true_pairs",
    "y_cal_error_pairs",
    "y_val_pred_pairs",
    "y_val_true_pairs",
    "y_val_error_pairs",
    "factor_labels",
    "a_scores_cal_pairs",
    "a_scores_val_pairs",
    "y_val_effective_pairs",
    "cal_regression_line_x_pairs",
    "cal_regression_line_y_pairs",
    "val_pred_ref_diag_extent",
    "val_pred_ref_diag_x",
    "val_pred_ref_diag_y",
    "validation_processing",
    "sbs_scores_mode_a",
    "sbs_val_scores_mode_a",
    "sbs_scores_mode_a_heatmap",
    "sbs_scores_mode_a_heatmap_full",
    "sbs_instrumental_profiles",
    "sbs_mode_profile_factors",
    "sbs_mode_profile_factors_full_a",
    "sbs_mode_profile_axes",
    "sbs_mode_profile_axes_full_a",
    "sbs_mode_profile_mode_labels",
    "sbs_mode_profile_y_titles",
    "sbs_similarity_matrix",
    "sbs_val_sample_axis",
    "sbs_mode_a_sample_axis",
    "sbs_mode_a_sample_axis_full",
    "sbs_mode_a_component_axis",
    "sbs_factor_labels",
    "sbs_y_cal_pred",
    "sbs_y_cal_true",
    "sbs_y_cal_error",
    "sbs_y_val_true",
    "sbs_y_val_pred",
    "sbs_a_scores_cal_pairs",
    "sbs_a_scores_val_pairs",
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
    "sbs_ejcr_cal",
    "sbs_ejcr_val",
)


def parafac_analysis_standard(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:
    """Adapter for app execution pipeline: return outputs as ordered tuple.

    The core parafac_analysis API keeps its dict return for direct module users.
    """
    result = parafac_analysis(*args, **kwargs)
    if not isinstance(result, dict):
        return (result,)
    return tuple(result.get(key) for key in _PARAFAC_RETURN_ORDER)
