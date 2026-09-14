from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, StratifiedKFold

try:
    from chemometrics.variable_selection.auswahl import CARS as AuswahlCARS
    from chemometrics.variable_selection.auswahl import BiPLS as AuswahlBiPLS
    from chemometrics.variable_selection.auswahl import FiPLS as AuswahlFiPLS
    from chemometrics.variable_selection.auswahl import IPLS as AuswahlIPLS
    from chemometrics.variable_selection.auswahl import IntervalRandomFrog as AuswahlIntervalRandomFrog
    from chemometrics.variable_selection.auswahl import MCUVE as AuswahlMCUVE
    from chemometrics.variable_selection.auswahl import RandomFrog as AuswahlRandomFrog
    from chemometrics.variable_selection.auswahl import SPA as AuswahlSPA
    from chemometrics.variable_selection.auswahl import VIP as AuswahlVIP
    from chemometrics.variable_selection.auswahl import VIP_SPA as AuswahlVIPSPA
    from chemometrics.variable_selection.auswahl import VISSA as AuswahlVISSA
except Exception:
    AuswahlCARS = None  # type: ignore
    AuswahlBiPLS = None  # type: ignore
    AuswahlFiPLS = None  # type: ignore
    AuswahlIPLS = None  # type: ignore
    AuswahlIntervalRandomFrog = None  # type: ignore
    AuswahlMCUVE = None  # type: ignore
    AuswahlRandomFrog = None  # type: ignore
    AuswahlSPA = None  # type: ignore
    AuswahlVIP = None  # type: ignore
    AuswahlVIPSPA = None  # type: ignore
    AuswahlVISSA = None  # type: ignore

try:
    from chemometrics.cv_pipeline import CVConfig, CVPipeline
except Exception:
    CVConfig = None  # type: ignore
    CVPipeline = None  # type: ignore


@dataclass
class SelectionResult:
    indices: np.ndarray
    scores: Optional[np.ndarray]
    metadata: Dict[str, Any]


def _parse_selected_indices_text(raw: str) -> List[int]:
    normalized = str(raw).replace(';', ',')
    tokens = [t.strip().replace(' ', '') for t in normalized.split(',') if t.strip()]
    indices: List[int] = []

    def _expand_interval_token(token: str) -> List[int]:
        parts = [p.strip() for p in token.split('-', 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid interval token '{token}' in override file.")
        try:
            start = int(parts[0])
            end = int(parts[1])
        except Exception as exc:
            raise ValueError(f"Invalid interval token '{token}' in override file.") from exc
        if end < start:
            raise ValueError(f"Invalid interval '{token}' in override file: end must be >= start.")
        return list(range(start, end + 1))

    for token in tokens:
        if '-' in token:
            indices.extend(_expand_interval_token(token))
            continue
        try:
            indices.append(int(token))
        except Exception as exc:
            raise ValueError(f"Invalid selected index token '{token}' in override file.") from exc
    return indices


def _indices_to_interval_text(indices: np.ndarray, *, one_based: bool = False) -> str:
    raw = np.asarray(indices, dtype=int).reshape(-1)
    if raw.size == 0:
        return ""

    values = sorted(set(int(v) for v in raw.tolist()))
    if one_based:
        values = [v + 1 for v in values]

    chunks: List[str] = []
    start = values[0]
    prev = values[0]
    for current in values[1:]:
        if current == prev + 1:
            prev = current
            continue
        if start == prev:
            chunks.append(str(start))
        else:
            chunks.append(f"{start}-{prev}")
        start = current
        prev = current

    if start == prev:
        chunks.append(str(start))
    else:
        chunks.append(f"{start}-{prev}")

    return ",".join(chunks)


def _parse_selection_override_file(file_path: str, n_features: int) -> Dict[str, Any]:
    path_text = str(file_path or '').strip()
    if not path_text:
        raise ValueError("selection_override_file_path is empty.")

    path = Path(path_text)
    if not path.exists():
        raise ValueError(f"selection_override_file_path does not exist: {path_text}")

    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        content = path.read_text(encoding="latin-1")

    parsed: Dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if '=' not in stripped:
            continue
        key, value = stripped.split('=', 1)
        parsed[key.strip().lower()] = value.strip()

    version = parsed.get('version', '')
    if version != '1':
        raise ValueError("Override file must contain 'version=1'.")

    index_base = parsed.get('index_base', '').strip().lower()
    if index_base not in ('zero_based', 'one_based'):
        raise ValueError("Override file must contain index_base=zero_based|one_based.")

    selected_indices_raw = parsed.get('selected_indices', '')
    if not selected_indices_raw:
        selected_indices_raw = parsed.get('selected_intervals', '')
    if not selected_indices_raw:
        raise ValueError("Override file must contain selected_indices (supports interval syntax like 20-36,3,6,80-95).")

    parsed_indices = _parse_selected_indices_text(selected_indices_raw)
    if not parsed_indices:
        raise ValueError("Override file selected_indices is empty.")

    converted: List[int] = []
    for value in parsed_indices:
        idx = value - 1 if index_base == 'one_based' else value
        if idx < 0 or idx >= int(n_features):
            raise ValueError(
                f"Override selected index out of range for current X_cal features: {value} (resolved {idx})."
            )
        converted.append(int(idx))

    unique_sorted = sorted(set(converted))
    if not unique_sorted:
        raise ValueError("Override selection resolves to empty feature set.")

    return {
        'selected_indices_zero_based': np.asarray(unique_sorted, dtype=int),
        'selected_intervals_zero_based': _indices_to_interval_text(np.asarray(unique_sorted, dtype=int), one_based=False),
        'selected_intervals_one_based': _indices_to_interval_text(np.asarray(unique_sorted, dtype=int), one_based=True),
        'selection_method': str(parsed.get('selection_method', 'manual') or 'manual').strip().lower(),
        'raw_content': content,
        'index_base': index_base,
        'path_used': str(path),
        'notes': parsed.get('notes', ''),
    }


def _build_selection_override_payload_text(
    *,
    selected_idx_zero_based: np.ndarray,
    selection_method: str,
    n_features_original: int,
) -> str:
    idx_zero = [int(v) for v in np.asarray(selected_idx_zero_based, dtype=int).tolist()]
    idx_one = [int(v) + 1 for v in idx_zero]
    idx_zero_text = _indices_to_interval_text(np.asarray(idx_zero, dtype=int), one_based=False)
    idx_one_text = _indices_to_interval_text(np.asarray(idx_zero, dtype=int), one_based=True)
    return "\n".join([
        "version=1",
        "index_base=zero_based",
        "selected_indices=" + idx_zero_text,
        "selected_indices_expanded=" + ",".join(str(v) for v in idx_zero),
        "selected_intervals_zero_based=" + idx_zero_text,
        f"selection_method={selection_method}",
        f"n_features_original={int(n_features_original)}",
        f"selected_count={len(idx_zero)}",
        "selected_indices_one_based=" + idx_one_text,
        "selected_indices_one_based_expanded=" + ",".join(str(v) for v in idx_one),
        "selected_intervals_one_based=" + idx_one_text,
    ])


def _ensure_2d(arr: Optional[Any]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr)
    if out.ndim == 1:
        return out.reshape(-1, 1)
    return out


def _ensure_1d_labels(arr: Optional[Any]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr, dtype=object)
    if out.ndim >= 2:
        out = out[:, 0]
    return out.reshape(-1)


def _coerce_numeric_target(y: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if y is None:
        return None
    y_arr = np.asarray(y)
    if y_arr.ndim >= 2:
        y_arr = y_arr[:, 0]
    y_arr = y_arr.reshape(-1)
    if y_arr.size == 0:
        return np.asarray([], dtype=float)
    try:
        return np.asarray(y_arr, dtype=float).reshape(-1)
    except Exception:
        unique_vals, inverse = np.unique(np.asarray(y_arr, dtype=object), return_inverse=True)
        if unique_vals.size == 0:
            return np.asarray([], dtype=float)
        return np.asarray(inverse, dtype=float).reshape(-1)


def _coerce_cv_config(cv_config: Optional[Any]) -> Optional[Any]:
    if cv_config is not None and isinstance(cv_config, dict) and "cv_config" in cv_config:
        cv_config = cv_config["cv_config"]

    if cv_config is None:
        return None

    if CVConfig is not None and isinstance(cv_config, CVConfig):
        return cv_config

    if isinstance(cv_config, dict) and CVConfig is not None:
        try:
            return CVConfig.from_dict(cv_config)
        except Exception:
            return None

    return None


def _default_cv_splits(
    X: np.ndarray,
    y: Optional[np.ndarray],
    task_type: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_samples = int(X.shape[0])
    if n_samples <= 2:
        return [(np.arange(n_samples, dtype=int), np.arange(n_samples, dtype=int))]

    if task_type == "classification" and y is not None:
        try:
            unique, counts = np.unique(np.asarray(y, dtype=object), return_counts=True)
            min_count = int(np.min(counts)) if counts.size else 0
            if unique.size >= 2 and min_count >= 2:
                n_splits = int(min(5, min_count, n_samples))
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                return list(splitter.split(X, y))
        except Exception:
            pass

    n_splits = int(min(5, n_samples))
    if n_splits < 2:
        n_splits = 2
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(splitter.split(X))


def _resolve_cv_splits(
    X: np.ndarray,
    y: Optional[np.ndarray],
    task_type: str,
    cv_config: Optional[Any],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    effective_cv = _coerce_cv_config(cv_config)
    if effective_cv is None or CVPipeline is None or not bool(effective_cv.is_enabled()):
        return _default_cv_splits(X, y, task_type)

    try:
        pipeline = CVPipeline(effective_cv)
        if str(effective_cv.cv_strategy).strip().lower() == "stratified_kfold" and y is not None:
            return list(pipeline.splitter.get_splits(X, y=np.asarray(y, dtype=object)))
        return list(pipeline.splitter.get_splits(X))
    except Exception:
        return _default_cv_splits(X, y, task_type)


def _as_axis_info_sequence(axis_info: Optional[Any]) -> Optional[List[Any]]:
    if axis_info is None:
        return None
    if isinstance(axis_info, list):
        return axis_info
    if isinstance(axis_info, tuple):
        return list(axis_info)
    if isinstance(axis_info, np.ndarray):
        arr = np.asarray(axis_info, dtype=object)
        if arr.ndim == 0:
            return [arr.item()]
        if arr.ndim == 1:
            return arr.tolist()
        return [arr[i, ...] for i in range(int(arr.shape[0]))]
    return None


def _axis_entry_length(entry: Any) -> Optional[int]:
    if entry is None:
        return None
    try:
        if isinstance(entry, np.ndarray):
            return int(np.asarray(entry).reshape(-1).shape[0])
        if isinstance(entry, (list, tuple)):
            return int(len(entry))
        return int(np.asarray(entry).reshape(-1).shape[0])
    except Exception:
        return None


def _infer_feature_axis_slot(
    axis_info: Optional[List[Any]],
    n_features_total: Optional[int] = None,
) -> Optional[int]:
    axis_seq = _as_axis_info_sequence(axis_info)
    if not isinstance(axis_seq, list) or not axis_seq:
        return None

    # Mirror data_selection slot semantics for first-order variable slicing:
    # - len == non_sample_dims + 1 -> slot 1 (sample axis present)
    # - len == non_sample_dims     -> slot 0 (sample axis omitted)
    # - otherwise prefer slot 1 when available
    non_sample_dims = 1
    if len(axis_seq) == non_sample_dims + 1:
        return 1
    if len(axis_seq) == non_sample_dims:
        return 0

    preferred_slot = 1 if len(axis_seq) > 1 else 0

    if n_features_total is not None:
        try:
            n_total = int(n_features_total)
        except Exception:
            n_total = -1

        if n_total > 0:
            exact_matches: List[int] = []
            for idx, entry in enumerate(axis_seq):
                entry_len = _axis_entry_length(entry)
                if entry_len == n_total:
                    exact_matches.append(idx)

            if exact_matches:
                if preferred_slot in exact_matches:
                    return preferred_slot
                return exact_matches[0]

    return preferred_slot


def _slice_axis_entry(entry: Any, indices: np.ndarray) -> Any:
    if entry is None:
        return None
    if isinstance(entry, np.ndarray):
        return np.asarray(entry).reshape(-1)[indices]
    if isinstance(entry, list):
        return [entry[int(i)] for i in indices if 0 <= int(i) < len(entry)]
    if isinstance(entry, tuple):
        return tuple(entry[int(i)] for i in indices if 0 <= int(i) < len(entry))
    arr = np.asarray(entry).reshape(-1)
    return arr[indices]


def _slice_axis_info(
    axis_info: Optional[List[Any]],
    indices: np.ndarray,
    n_features_total: Optional[int] = None,
) -> Optional[List[Any]]:
    axis_seq = _as_axis_info_sequence(axis_info)
    if not isinstance(axis_seq, list):
        return axis_info
    out = [item.copy() if isinstance(item, dict) else item for item in axis_seq]
    slot = _infer_feature_axis_slot(out, n_features_total=n_features_total)
    if slot is None or slot >= len(out):
        return out
    out[slot] = _slice_axis_entry(out[slot], indices)
    return out


def _feature_labels_from_axis_info(
    axis_t_info: Optional[List[Any]],
    axis_n_info: Optional[List[Any]],
    n_features: int,
) -> List[str]:
    def _extract_slot_values(axis_info: Optional[List[Any]]) -> Optional[List[Any]]:
        axis_seq = _as_axis_info_sequence(axis_info)
        if not isinstance(axis_seq, list) or len(axis_seq) == 0:
            return None
        slot = _infer_feature_axis_slot(axis_seq, n_features_total=n_features)
        if slot is None or slot >= len(axis_seq):
            return None
        entry = axis_seq[slot]
        if entry is None:
            return None
        if isinstance(entry, np.ndarray):
            return np.asarray(entry).reshape(-1).tolist()
        if isinstance(entry, (list, tuple)):
            return list(entry)
        try:
            return np.asarray(entry).reshape(-1).tolist()
        except Exception:
            return None

    # Prefer textual axis labels when available.
    text_values = _extract_slot_values(axis_t_info)
    if text_values is not None and len(text_values) >= int(n_features):
        out = []
        for idx in range(int(n_features)):
            label = str(text_values[idx]).strip()
            out.append(label if label else f"var_{idx}")
        return out

    numeric_values = _extract_slot_values(axis_n_info)
    if numeric_values is not None and len(numeric_values) >= int(n_features):
        out = []
        for idx in range(int(n_features)):
            label = str(numeric_values[idx]).strip()
            out.append(label if label else f"var_{idx}")
        return out

    return [f"var_{idx}" for idx in range(int(n_features))]


def _selected_variables_display_text(indices: np.ndarray, feature_labels: List[str]) -> str:
    idx = sorted(set(int(v) for v in np.asarray(indices, dtype=int).reshape(-1).tolist()))
    if not idx:
        return ""

    n_labels = int(len(feature_labels))
    parts: List[str] = []
    start = idx[0]
    prev = idx[0]
    for current in idx[1:]:
        if current == prev + 1:
            prev = current
            continue

        # Closed run [start, prev]
        if start == prev:
            if 0 <= start < n_labels:
                parts.append(str(feature_labels[start]))
            else:
                parts.append(f"var_{start}")
        else:
            left = str(feature_labels[start]) if 0 <= start < n_labels else f"var_{start}"
            right = str(feature_labels[prev]) if 0 <= prev < n_labels else f"var_{prev}"
            parts.append(f"{left}-{right}")

        start = current
        prev = current

    if start == prev:
        if 0 <= start < n_labels:
            parts.append(str(feature_labels[start]))
        else:
            parts.append(f"var_{start}")
    else:
        left = str(feature_labels[start]) if 0 <= start < n_labels else f"var_{start}"
        right = str(feature_labels[prev]) if 0 <= prev < n_labels else f"var_{prev}"
        parts.append(f"{left}-{right}")

    return ",".join(parts)


def _effective_feature_count(n_features_to_select: int, n_features_total: int) -> int:
    count = int(n_features_to_select)
    if count <= 0:
        count = max(1, n_features_total // 4)
    return max(1, min(n_features_total, count))


def _vip_scores(X: np.ndarray, Y: np.ndarray, n_components: int) -> np.ndarray:
    n_samples, n_features = X.shape
    max_comp = max(1, min(int(n_components), n_features, max(1, n_samples - 1)))
    pls = PLSRegression(n_components=max_comp, scale=False)
    pls.fit(X, Y)

    t = np.asarray(pls.x_scores_, dtype=float)
    w = np.asarray(pls.x_weights_, dtype=float)
    q = np.asarray(pls.y_loadings_, dtype=float)

    ss_components = np.sum(t ** 2, axis=0) * np.sum(q ** 2, axis=0)
    ss_total = float(np.sum(ss_components))
    if ss_total <= 0.0:
        return np.ones(n_features, dtype=float)

    weight_norm = np.sum(w ** 2, axis=0)
    weight_norm[weight_norm <= 0.0] = 1.0

    vip = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        contribution = np.sum(ss_components * (w[j, :] ** 2) / weight_norm)
        vip[j] = np.sqrt(max(0.0, n_features * contribution / ss_total))
    return vip


def _selected_metric_marker(
    feature_counts: List[int],
    values: List[float],
    selected_count: int,
) -> float:
    for idx, count in enumerate(feature_counts):
        if int(count) == int(selected_count) and idx < len(values):
            try:
                return float(values[idx])
            except Exception:
                return float("nan")
    return float("nan")


def _spa_indices(X: np.ndarray, n_select: int) -> np.ndarray:
    Xc = np.asarray(X, dtype=float) - np.mean(np.asarray(X, dtype=float), axis=0, keepdims=True)
    n_features = Xc.shape[1]
    n_select = max(1, min(n_features, int(n_select)))

    variances = np.var(Xc, axis=0)
    first = int(np.argmax(variances))
    selected: List[int] = [first]

    while len(selected) < n_select:
        best_idx = -1
        best_score = -np.inf
        S = Xc[:, selected]
        for j in range(n_features):
            if j in selected:
                continue
            xj = Xc[:, j]
            if S.ndim == 1:
                S2 = S.reshape(-1, 1)
            else:
                S2 = S
            try:
                proj = S2 @ np.linalg.pinv(S2) @ xj
                residual = xj - proj
                score = float(np.linalg.norm(residual))
            except Exception:
                score = float(np.linalg.norm(xj))
            if score > best_score:
                best_score = score
                best_idx = j
        if best_idx < 0:
            break
        selected.append(best_idx)

    return np.asarray(selected[:n_select], dtype=int)


def _mean_pairwise_jaccard(index_sets: List[np.ndarray]) -> Optional[float]:
    if not index_sets or len(index_sets) < 2:
        return None

    total = 0.0
    count = 0
    for i in range(len(index_sets)):
        a = set(int(v) for v in np.asarray(index_sets[i], dtype=int).tolist())
        for j in range(i + 1, len(index_sets)):
            b = set(int(v) for v in np.asarray(index_sets[j], dtype=int).tolist())
            union = a.union(b)
            if not union:
                continue
            inter = a.intersection(b)
            total += float(len(inter) / len(union))
            count += 1
    if count == 0:
        return None
    return float(total / count)


def _select_indices_single(
    X: np.ndarray,
    y: Optional[np.ndarray],
    task_type: str,
    selection_method: str,
    n_select: int,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    sfs_direction: str,
    vip_n_components: int,
    interval_pls_n_components: int,
    classification_family_hint: Optional[str] = None,
    selection_random_state: int = 42,
    selection_n_jobs: int = 1,
    mcuve_n_subsets: int = 100,
    mcuve_n_samples_per_subset: Optional[float] = None,
    cars_n_cars_runs: int = 20,
    cars_n_sample_runs: int = 100,
    cars_fit_samples_ratio: float = 0.9,
    random_frog_n_iterations: int = 10000,
    random_frog_n_initial_features: float = 0.1,
    random_frog_variance_factor: float = 0.3,
    random_frog_subset_expansion_factor: float = 3.0,
    random_frog_acceptance_factor: float = 0.1,
    interval_width: Any = None,
    vissa_n_submodels: int = 1000,
    vissa_ratio_submodel_selection: float = 0.05,
    vissa_max_iter: int = 100,
) -> SelectionResult:
    method = str(selection_method).strip().lower()

    def _resolve_interval_params() -> Tuple[int, int]:
        if interval_width in (None, ""):
            width_local = max(1, min(int(n_select), max(1, int(n_features // 10))))
        else:
            try:
                width_float_local = float(interval_width)
            except Exception:
                width_float_local = float(max(1, min(int(n_select), max(1, int(n_features // 10)))))
            if 0 < width_float_local < 1:
                width_local = max(1, int(round(width_float_local * n_features)))
            else:
                width_local = max(1, int(round(width_float_local)))
        width_local = max(1, min(width_local, max(1, n_features - 1)))

        n_intervals_local = max(1, int(round(float(n_select) / float(width_local))))
        while (n_intervals_local * width_local) >= n_features and n_intervals_local > 1:
            n_intervals_local -= 1
        n_intervals_local = max(1, n_intervals_local)
        return int(width_local), int(n_intervals_local)
    n_features = int(X.shape[1])
    n_select = max(1, min(n_features, int(n_select)))

    if method == "vip":
        if y is None:
            raise ValueError("VIP selection requires Y_cal for supervised ranking.")
        if AuswahlVIP is None:
            raise RuntimeError("auswahl VIP backend is unavailable.")

        y_num = _coerce_numeric_target(y)
        if y_num is None:
            raise ValueError("VIP selection requires a valid target vector.")
        max_comp = max(1, min(int(vip_n_components), n_features, max(1, int(X.shape[0]) - 1)))
        selector = AuswahlVIP(
            n_features_to_select=int(n_select),
            n_cv_folds=max(2, min(5, int(X.shape[0]))),
            pls=PLSRegression(n_components=max_comp, scale=False),
        )
        selector.fit(X, y_num)
        support = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
        idx = np.where(support)[0]
        scores = np.asarray(getattr(selector, "vips_", np.zeros(n_features, dtype=float)), dtype=float).reshape(-1)
        if idx.size == 0:
            idx = np.argsort(scores)[::-1][:n_select]
        return SelectionResult(
            indices=np.asarray(idx, dtype=int),
            scores=scores,
            metadata={"ranking": "descending", "backend": "auswahl.VIP"},
        )

    if method == "spa":
        if y is None:
            raise ValueError("SPA selection requires target data for model-based candidate evaluation.")
        if AuswahlSPA is None:
            raise RuntimeError("auswahl SPA backend is unavailable.")

        y_num = _coerce_numeric_target(y)
        if y_num is None or y_num.size != int(X.shape[0]):
            raise ValueError("SPA selection requires a valid target vector aligned to X samples.")

        max_comp = max(1, min(int(vip_n_components), n_features, max(1, int(X.shape[0]) - 1)))
        selector = AuswahlSPA(
            n_features_to_select=int(n_select),
            n_cv_folds=max(2, min(5, int(X.shape[0]))),
            pls=PLSRegression(n_components=max_comp, scale=False),
            n_jobs=1,
        )
        selector.fit(X, y_num)
        support = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
        idx = np.where(support)[0]
        if idx.size == 0:
            raise RuntimeError("auswahl SPA returned an empty selection.")
        return SelectionResult(
            indices=np.asarray(idx, dtype=int),
            scores=None,
            metadata={"selection": "projection_residual_greedy", "backend": "auswahl.SPA"},
        )

    if method == "vip_spa":
        if y is None:
            raise ValueError("VIP_SPA selection requires target data.")
        if AuswahlVIPSPA is None:
            raise RuntimeError("auswahl VIP_SPA backend is unavailable.")

        y_num = _coerce_numeric_target(y)
        if y_num is None or y_num.size != int(X.shape[0]):
            raise ValueError("VIP_SPA selection requires a valid target vector aligned to X samples.")

        max_comp = max(1, min(int(vip_n_components), n_features, max(1, int(X.shape[0]) - 1)))
        selector = AuswahlVIPSPA(
            n_features_to_select=int(n_select),
            n_cv_folds=max(2, int(len(cv_splits)) if cv_splits else min(5, int(X.shape[0]))),
            n_jobs=max(1, int(selection_n_jobs)),
            pls=PLSRegression(n_components=max_comp, scale=False),
        )
        selector.fit(X, y_num)
        support = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
        idx = np.where(support)[0]
        if idx.size == 0:
            raise RuntimeError("auswahl VIP_SPA returned an empty selection.")
        return SelectionResult(
            indices=np.asarray(idx, dtype=int),
            scores=None,
            metadata={"backend": "auswahl.VIP_SPA"},
        )

    if method == "mcuve":
        if y is None:
            raise ValueError("MCUVE selection requires target data.")
        if AuswahlMCUVE is None:
            raise RuntimeError("auswahl MCUVE backend is unavailable.")

        y_num = _coerce_numeric_target(y)
        if y_num is None or y_num.size != int(X.shape[0]):
            raise ValueError("MCUVE selection requires a valid target vector aligned to X samples.")

        max_comp = max(1, min(int(vip_n_components), n_features, max(1, int(X.shape[0]) - 1)))
        selector = AuswahlMCUVE(
            n_features_to_select=int(n_select),
            n_subsets=max(2, int(mcuve_n_subsets)),
            n_samples_per_subset=mcuve_n_samples_per_subset,
            pls=PLSRegression(n_components=max_comp, scale=False),
            n_cv_folds=max(2, int(len(cv_splits)) if cv_splits else min(5, int(X.shape[0]))),
            random_state=int(selection_random_state),
        )
        selector.fit(X, y_num)
        support = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
        idx = np.where(support)[0]
        stability = np.asarray(getattr(selector, 'stability_', np.zeros(n_features, dtype=float)), dtype=float).reshape(-1)
        return SelectionResult(
            indices=np.asarray(idx, dtype=int),
            scores=np.abs(stability),
            metadata={"backend": "auswahl.MCUVE"},
        )

    if method == "cars":
        if y is None:
            raise ValueError("CARS selection requires target data.")
        if AuswahlCARS is None:
            raise RuntimeError("auswahl CARS backend is unavailable.")

        y_num = _coerce_numeric_target(y)
        if y_num is None or y_num.size != int(X.shape[0]):
            raise ValueError("CARS selection requires a valid target vector aligned to X samples.")

        max_comp = max(1, min(int(vip_n_components), n_features, max(1, int(X.shape[0]) - 1)))
        selector = AuswahlCARS(
            n_features_to_select=int(n_select),
            n_cars_runs=max(1, int(cars_n_cars_runs)),
            n_jobs=max(1, int(selection_n_jobs)),
            n_sample_runs=max(2, int(cars_n_sample_runs)),
            fit_samples_ratio=float(cars_fit_samples_ratio),
            n_cv_folds=max(2, int(len(cv_splits)) if cv_splits else min(5, int(X.shape[0]))),
            pls=PLSRegression(n_components=max_comp, scale=False),
            random_state=int(selection_random_state),
        )
        selector.fit(X, y_num)
        support = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
        idx = np.where(support)[0]
        importance = np.asarray(getattr(selector, 'feature_importance_', np.zeros(n_features, dtype=float)), dtype=float).reshape(-1)
        return SelectionResult(
            indices=np.asarray(idx, dtype=int),
            scores=importance,
            metadata={"backend": "auswahl.CARS"},
        )

    if method == "random_frog":
        if y is None:
            raise ValueError("RandomFrog selection requires target data.")
        if AuswahlRandomFrog is None:
            raise RuntimeError("auswahl RandomFrog backend is unavailable.")

        y_num = _coerce_numeric_target(y)
        if y_num is None or y_num.size != int(X.shape[0]):
            raise ValueError("RandomFrog selection requires a valid target vector aligned to X samples.")

        max_comp = max(1, min(int(vip_n_components), n_features, max(1, int(X.shape[0]) - 1)))
        selector = AuswahlRandomFrog(
            n_features_to_select=int(n_select),
            n_iterations=max(1, int(random_frog_n_iterations)),
            n_initial_features=float(random_frog_n_initial_features),
            variance_factor=float(random_frog_variance_factor),
            subset_expansion_factor=float(random_frog_subset_expansion_factor),
            acceptance_factor=float(random_frog_acceptance_factor),
            pls=PLSRegression(n_components=max_comp, scale=False),
            n_cv_folds=max(2, int(len(cv_splits)) if cv_splits else min(5, int(X.shape[0]))),
            n_jobs=max(1, int(selection_n_jobs)),
            random_state=int(selection_random_state),
        )
        selector.fit(X, y_num)
        support = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
        idx = np.where(support)[0]
        frequency = np.asarray(getattr(selector, 'frequencies_', np.zeros(n_features, dtype=float)), dtype=float).reshape(-1)
        return SelectionResult(
            indices=np.asarray(idx, dtype=int),
            scores=frequency,
            metadata={"backend": "auswahl.RandomFrog"},
        )

    if method == "interval_random_frog":
        if y is None:
            raise ValueError("IntervalRandomFrog selection requires target data.")
        if AuswahlIntervalRandomFrog is None:
            raise RuntimeError("auswahl IntervalRandomFrog backend is unavailable.")

        y_num = _coerce_numeric_target(y)
        if y_num is None or y_num.size != int(X.shape[0]):
            raise ValueError("IntervalRandomFrog selection requires a valid target vector aligned to X samples.")

        max_comp = max(1, min(int(interval_pls_n_components), n_features, max(1, int(X.shape[0]) - 1)))

        width, n_intervals = _resolve_interval_params()

        selector = AuswahlIntervalRandomFrog(
            n_intervals_to_select=int(n_intervals),
            interval_width=int(width),
            n_iterations=max(1, int(random_frog_n_iterations)),
            n_initial_intervals=float(random_frog_n_initial_features),
            variance_factor=float(random_frog_variance_factor),
            subset_expansion_factor=float(random_frog_subset_expansion_factor),
            acceptance_factor=float(random_frog_acceptance_factor),
            pls=PLSRegression(n_components=max_comp, scale=False),
            n_cv_folds=max(2, int(len(cv_splits)) if cv_splits else min(5, int(X.shape[0]))),
            n_jobs=max(1, int(selection_n_jobs)),
            random_state=int(selection_random_state),
        )
        selector.fit(X, y_num)
        support = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
        idx = np.where(support)[0]
        frequency = np.asarray(getattr(selector, 'frequencies_', np.zeros(n_features, dtype=float)), dtype=float).reshape(-1)
        return SelectionResult(
            indices=np.asarray(idx, dtype=int),
            scores=frequency,
            metadata={
                "backend": "auswahl.IntervalRandomFrog",
                "interval_width": int(width),
                "n_intervals_to_select": int(n_intervals),
            },
        )

    if method == "ipls":
        if y is None:
            raise ValueError("IPLS selection requires target data.")
        if AuswahlIPLS is None:
            raise RuntimeError("auswahl IPLS backend is unavailable.")

        y_num = _coerce_numeric_target(y)
        if y_num is None or y_num.size != int(X.shape[0]):
            raise ValueError("IPLS selection requires a valid target vector aligned to X samples.")

        max_comp = max(1, min(int(interval_pls_n_components), n_features, max(1, int(X.shape[0]) - 1)))
        width, n_intervals = _resolve_interval_params()
        # IPLS supports one contiguous interval; fold the requested interval count into width.
        ipls_width = max(1, min(int(width) * int(n_intervals), max(1, n_features - 1)))
        selector = AuswahlIPLS(
            n_intervals_to_select=1,
            interval_width=int(ipls_width),
            n_cv_folds=max(2, int(len(cv_splits)) if cv_splits else min(5, int(X.shape[0]))),
            pls=PLSRegression(n_components=max_comp, scale=False),
            n_jobs=max(1, int(selection_n_jobs)),
            random_state=int(selection_random_state),
        )
        selector.fit(X, y_num)
        support = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
        idx = np.where(support)[0]
        return SelectionResult(
            indices=np.asarray(idx, dtype=int),
            scores=None,
            metadata={
                "backend": "auswahl.IPLS",
                "interval_width": int(ipls_width),
                "n_intervals_to_select": 1,
                "requested_interval_width": int(width),
                "requested_n_intervals": int(n_intervals),
            },
        )

    if method == "fipls":
        if y is None:
            raise ValueError("FiPLS selection requires target data.")
        if AuswahlFiPLS is None:
            raise RuntimeError("auswahl FiPLS backend is unavailable.")

        y_num = _coerce_numeric_target(y)
        if y_num is None or y_num.size != int(X.shape[0]):
            raise ValueError("FiPLS selection requires a valid target vector aligned to X samples.")

        max_comp = max(1, min(int(interval_pls_n_components), n_features, max(1, int(X.shape[0]) - 1)))
        width, n_intervals = _resolve_interval_params()
        selector = AuswahlFiPLS(
            n_intervals_to_select=int(n_intervals),
            interval_width=int(width),
            n_cv_folds=max(2, int(len(cv_splits)) if cv_splits else min(5, int(X.shape[0]))),
            pls=PLSRegression(n_components=max_comp, scale=False),
            n_jobs=max(1, int(selection_n_jobs)),
        )
        selector.fit(X, y_num)
        support = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
        idx = np.where(support)[0]
        return SelectionResult(
            indices=np.asarray(idx, dtype=int),
            scores=None,
            metadata={
                "backend": "auswahl.FiPLS",
                "interval_width": int(width),
                "n_intervals_to_select": int(n_intervals),
            },
        )

    if method == "bipls":
        if y is None:
            raise ValueError("BiPLS selection requires target data.")
        if AuswahlBiPLS is None:
            raise RuntimeError("auswahl BiPLS backend is unavailable.")

        y_num = _coerce_numeric_target(y)
        if y_num is None or y_num.size != int(X.shape[0]):
            raise ValueError("BiPLS selection requires a valid target vector aligned to X samples.")

        max_comp = max(1, min(int(interval_pls_n_components), n_features, max(1, int(X.shape[0]) - 1)))
        width, n_intervals = _resolve_interval_params()
        selector = AuswahlBiPLS(
            n_intervals_to_select=int(n_intervals),
            interval_width=int(width),
            n_cv_folds=max(2, int(len(cv_splits)) if cv_splits else min(5, int(X.shape[0]))),
            pls=PLSRegression(n_components=max_comp, scale=False),
            n_jobs=max(1, int(selection_n_jobs)),
        )
        selector.fit(X, y_num)
        support = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
        idx = np.where(support)[0]
        rank = np.asarray(getattr(selector, 'rank_', np.zeros(n_features, dtype=float)), dtype=float).reshape(-1)
        return SelectionResult(
            indices=np.asarray(idx, dtype=int),
            scores=rank,
            metadata={
                "backend": "auswahl.BiPLS",
                "interval_width": int(width),
                "n_intervals_to_select": int(n_intervals),
            },
        )

    if method == "vissa":
        if y is None:
            raise ValueError("VISSA selection requires target data.")
        if AuswahlVISSA is None:
            raise RuntimeError("auswahl VISSA backend is unavailable.")

        y_num = _coerce_numeric_target(y)
        if y_num is None or y_num.size != int(X.shape[0]):
            raise ValueError("VISSA selection requires a valid target vector aligned to X samples.")

        max_comp = max(1, min(int(vip_n_components), n_features, max(1, int(X.shape[0]) - 1)))
        selector = AuswahlVISSA(
            n_features_to_select=int(n_select),
            n_submodels=max(2, int(vissa_n_submodels)),
            ratio_submodel_selection=float(vissa_ratio_submodel_selection),
            max_iter=max(1, int(vissa_max_iter)),
            pls=PLSRegression(n_components=max_comp, scale=False),
            n_cv_folds=max(2, int(len(cv_splits)) if cv_splits else min(5, int(X.shape[0]))),
            random_state=int(selection_random_state),
            n_jobs=max(1, int(selection_n_jobs)),
        )
        selector.fit(X, y_num)
        support = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
        idx = np.where(support)[0]
        frequency = np.asarray(getattr(selector, 'frequency_', np.zeros(n_features, dtype=float)), dtype=float).reshape(-1)
        return SelectionResult(
            indices=np.asarray(idx, dtype=int),
            scores=frequency,
            metadata={"backend": "auswahl.VISSA"},
        )

    if method == "sfs":
        raise ValueError(
            "selection_method='sfs' runs in nested-only mode and must be orchestrated by workflow_variable_selection_start. "
            "Direct estimator-backed SFS is disabled."
        )

    if method == "ga":
        raise ValueError(
            "selection_method='ga' is orchestrator-owned for nested-metric evaluation and is not available as a direct wrapper mode. "
            "Use workflow_variable_selection_start with an enclosed body so GA fitness comes from nested outputs."
        )

    raise ValueError(
        "selection_method must be one of: vip, spa, vip_spa, mcuve, cars, random_frog, interval_random_frog, ipls, fipls, bipls, vissa, sfs, ga"
    )


def select_variables_for_workflow(
    X_cal: np.ndarray,
    Y_cal: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    class_data_cal: Optional[Any] = None,
    class_data_val: Optional[Any] = None,
    axis_n_info: Optional[List[Any]] = None,
    axis_t_info: Optional[List[Any]] = None,
    task_type: str = "auto",
    selection_method: str = "vip",
    n_features_to_select: int = 10,
    cv_config: Optional[Any] = None,
    sfs_direction: str = "forward",
    vip_n_components: int = 2,
    interval_pls_n_components: int = 2,
    selection_random_state: int = 42,
    selection_n_jobs: int = 1,
    mcuve_n_subsets: int = 100,
    mcuve_n_samples_per_subset: Optional[float] = None,
    cars_n_cars_runs: int = 20,
    cars_n_sample_runs: int = 100,
    cars_fit_samples_ratio: float = 0.9,
    random_frog_n_iterations: int = 10000,
    random_frog_n_initial_features: float = 0.1,
    random_frog_variance_factor: float = 0.3,
    random_frog_subset_expansion_factor: float = 3.0,
    random_frog_acceptance_factor: float = 0.1,
    interval_width: Any = None,
    vissa_n_submodels: int = 1000,
    vissa_ratio_submodel_selection: float = 0.05,
    vissa_max_iter: int = 100,
    selection_override_file_path: str = "",
    selected_indices_zero_based: Optional[Any] = None,
    classification_family_hint: str = "",
) -> Dict[str, Any]:
    X_cal_2d = _ensure_2d(X_cal)
    X_val_2d = _ensure_2d(X_val)

    if X_cal_2d is None:
        raise ValueError("X_cal is required.")
    if X_cal_2d.ndim != 2:
        raise ValueError("Variable-selection workflow currently supports 2D first-order X inputs only.")

    n_samples, n_features = X_cal_2d.shape
    if n_samples < 2 or n_features < 1:
        raise ValueError("X_cal must contain at least 2 samples and 1 variable.")

    feature_labels = _feature_labels_from_axis_info(axis_t_info, axis_n_info, n_features)

    task_norm = str(task_type).strip().lower()
    if task_norm not in ("auto", "regression", "classification"):
        raise ValueError("task_type must be 'auto', 'regression', or 'classification'.")

    override_path = str(selection_override_file_path or '').strip()
    has_selected_indices_override = selected_indices_zero_based is not None

    if task_norm == "auto":
        if Y_cal is not None:
            task_norm = "regression"
        elif class_data_cal is not None:
            task_norm = "classification"
        elif not override_path and not has_selected_indices_override:
            raise ValueError("task_type='auto' requires either Y_cal or class_data_cal.")

    selection_method_norm = str(selection_method).strip().lower()
    sfs_direction_norm = str(sfs_direction).strip().lower()
    if sfs_direction_norm not in ("forward", "backward"):
        sfs_direction_norm = "forward"

    override_selection_payload = None
    override_from_file = False
    selected_indices_from_orchestrator: Optional[np.ndarray] = None
    y_for_selection = None
    if has_selected_indices_override:
        try:
            if isinstance(selected_indices_zero_based, str):
                raw_idx = np.asarray(_parse_selected_indices_text(selected_indices_zero_based), dtype=int).reshape(-1)
            else:
                raw_idx = np.asarray(selected_indices_zero_based, dtype=int).reshape(-1)
        except Exception as exc:
            raise ValueError("selected_indices_zero_based must be an integer array-like or interval text (e.g., 20-36,3,6,80-95).") from exc
        if raw_idx.size == 0:
            raise ValueError("selected_indices_zero_based cannot be empty.")
        unique_sorted = np.asarray(sorted(set(int(v) for v in raw_idx.tolist())), dtype=int)
        if np.any(unique_sorted < 0) or np.any(unique_sorted >= int(n_features)):
            raise ValueError("selected_indices_zero_based contains out-of-range indices for X_cal.")
        selected_indices_from_orchestrator = unique_sorted
    elif override_path:
        override_selection_payload = _parse_selection_override_file(override_path, n_features)
        override_from_file = True
    elif task_norm == "regression":
        if Y_cal is None:
            raise ValueError("Y_cal is required when task_type='regression'.")
        y_for_selection = np.asarray(Y_cal, dtype=float)
        if y_for_selection.ndim == 1:
            y_for_selection = y_for_selection.reshape(-1, 1)
    else:
        class_vector = _ensure_1d_labels(class_data_cal)
        if class_vector is None:
            raise ValueError("class_data_cal is required when task_type='classification'.")
        y_for_selection = np.asarray(class_vector, dtype=object)

    if y_for_selection is not None and y_for_selection.shape[0] != n_samples:
        raise ValueError("X_cal and supervised targets must have the same number of samples.")

    feature_count_trajectory: List[int] = []
    trajectory_cv_score: List[float] = []
    trajectory_rmsecv: List[float] = []
    trajectory_r2cv: List[float] = []
    trajectory_accuracy: List[float] = []
    trajectory_f1_macro: List[float] = []
    trajectory_precision_macro: List[float] = []
    trajectory_recall_macro: List[float] = []
    trajectory_rmse_cal: List[float] = []
    trajectory_rmse_cv: List[float] = []
    trajectory_rmse_val: List[float] = []
    trajectory_r2_cal: List[float] = []
    trajectory_r2_cv: List[float] = []
    trajectory_r2_val: List[float] = []
    trajectory_accuracy_cal: List[float] = []
    trajectory_accuracy_cv: List[float] = []
    trajectory_accuracy_val: List[float] = []
    trajectory_f1_cal: List[float] = []
    trajectory_f1_cv: List[float] = []
    trajectory_f1_val: List[float] = []
    trajectory_precision_cal: List[float] = []
    trajectory_precision_cv: List[float] = []
    trajectory_precision_val: List[float] = []
    trajectory_recall_cal: List[float] = []
    trajectory_recall_cv: List[float] = []
    trajectory_recall_val: List[float] = []
    ga_count_stability: List[float] = []
    ga_count_restart_mean_score: List[float] = []
    optimization_model_labels: List[str] = []
    optimization_selected_variables_display: List[str] = []
    optimization_is_selected_model: List[int] = []
    cv_splits_used_count = 0

    if override_from_file:
        selected_idx = np.asarray(override_selection_payload['selected_indices_zero_based'], dtype=int)
        best_selection = SelectionResult(
            indices=selected_idx,
            scores=None,
            metadata={
                "selection_source": "override_file",
                "override_file_path": override_selection_payload['path_used'],
                "override_index_base": override_selection_payload.get('index_base'),
            },
        )
        best_count = int(selected_idx.size)
        candidate_counts = [best_count]
        feature_count_trajectory.append(int(selected_idx.size))
        optimization_model_labels.append(f"k={int(selected_idx.size)}")
        optimization_selected_variables_display.append(_selected_variables_display_text(selected_idx, feature_labels))
        trajectory_cv_score.append(float("nan"))
        trajectory_rmsecv.append(float("nan"))
        trajectory_r2cv.append(float("nan"))
        trajectory_accuracy.append(float("nan"))
        trajectory_f1_macro.append(float("nan"))
        trajectory_precision_macro.append(float("nan"))
        trajectory_recall_macro.append(float("nan"))
        trajectory_rmse_cal.append(float("nan"))
        trajectory_rmse_cv.append(float("nan"))
        trajectory_rmse_val.append(float("nan"))
        trajectory_r2_cal.append(float("nan"))
        trajectory_r2_cv.append(float("nan"))
        trajectory_r2_val.append(float("nan"))
        trajectory_accuracy_cal.append(float("nan"))
        trajectory_accuracy_cv.append(float("nan"))
        trajectory_accuracy_val.append(float("nan"))
        trajectory_f1_cal.append(float("nan"))
        trajectory_f1_cv.append(float("nan"))
        trajectory_f1_val.append(float("nan"))
        trajectory_precision_cal.append(float("nan"))
        trajectory_precision_cv.append(float("nan"))
        trajectory_precision_val.append(float("nan"))
        trajectory_recall_cal.append(float("nan"))
        trajectory_recall_cv.append(float("nan"))
        trajectory_recall_val.append(float("nan"))
        optimization_is_selected_model.append(0)
        ga_count_stability.append(float("nan"))
        ga_count_restart_mean_score.append(float("nan"))
    elif selected_indices_from_orchestrator is not None:
        selected_idx = np.asarray(selected_indices_from_orchestrator, dtype=int)
        best_selection = SelectionResult(
            indices=selected_idx,
            scores=None,
            metadata={
                "selection_source": "algorithm",
                "selected_indices_source": "orchestrator",
            },
        )
        best_count = int(selected_idx.size)
        candidate_counts = [best_count]
        feature_count_trajectory.append(int(selected_idx.size))
        optimization_model_labels.append(f"k={int(selected_idx.size)}")
        optimization_selected_variables_display.append(_selected_variables_display_text(selected_idx, feature_labels))
        trajectory_cv_score.append(float("nan"))
        trajectory_rmsecv.append(float("nan"))
        trajectory_r2cv.append(float("nan"))
        trajectory_accuracy.append(float("nan"))
        trajectory_f1_macro.append(float("nan"))
        trajectory_precision_macro.append(float("nan"))
        trajectory_recall_macro.append(float("nan"))
        trajectory_rmse_cal.append(float("nan"))
        trajectory_rmse_cv.append(float("nan"))
        trajectory_rmse_val.append(float("nan"))
        trajectory_r2_cal.append(float("nan"))
        trajectory_r2_cv.append(float("nan"))
        trajectory_r2_val.append(float("nan"))
        trajectory_accuracy_cal.append(float("nan"))
        trajectory_accuracy_cv.append(float("nan"))
        trajectory_accuracy_val.append(float("nan"))
        trajectory_f1_cal.append(float("nan"))
        trajectory_f1_cv.append(float("nan"))
        trajectory_f1_val.append(float("nan"))
        trajectory_precision_cal.append(float("nan"))
        trajectory_precision_cv.append(float("nan"))
        trajectory_precision_val.append(float("nan"))
        trajectory_recall_cal.append(float("nan"))
        trajectory_recall_cv.append(float("nan"))
        trajectory_recall_val.append(float("nan"))
        optimization_is_selected_model.append(0)
        ga_count_stability.append(float("nan"))
        ga_count_restart_mean_score.append(float("nan"))
    else:
        base_n_features = _effective_feature_count(int(n_features_to_select), n_features)
        cv_splits = _resolve_cv_splits(
            X=X_cal_2d,
            y=np.asarray(y_for_selection).reshape(-1),
            task_type=task_norm,
            cv_config=cv_config,
        )
        cv_splits_used_count = int(len(cv_splits))

        candidate_counts = [base_n_features]

        best_selection: Optional[SelectionResult] = None
        best_count = base_n_features

        for count in candidate_counts:
            current = _select_indices_single(
                X=X_cal_2d,
                y=np.asarray(y_for_selection),
                task_type=task_norm,
                selection_method=selection_method_norm,
                n_select=count,
                cv_splits=cv_splits,
                sfs_direction=sfs_direction_norm,
                vip_n_components=vip_n_components,
                interval_pls_n_components=interval_pls_n_components,
                classification_family_hint=classification_family_hint,
                selection_random_state=selection_random_state,
                selection_n_jobs=selection_n_jobs,
                mcuve_n_subsets=mcuve_n_subsets,
                mcuve_n_samples_per_subset=mcuve_n_samples_per_subset,
                cars_n_cars_runs=cars_n_cars_runs,
                cars_n_sample_runs=cars_n_sample_runs,
                cars_fit_samples_ratio=cars_fit_samples_ratio,
                random_frog_n_iterations=random_frog_n_iterations,
                random_frog_n_initial_features=random_frog_n_initial_features,
                random_frog_variance_factor=random_frog_variance_factor,
                random_frog_subset_expansion_factor=random_frog_subset_expansion_factor,
                random_frog_acceptance_factor=random_frog_acceptance_factor,
                interval_width=interval_width,
                vissa_n_submodels=vissa_n_submodels,
                vissa_ratio_submodel_selection=vissa_ratio_submodel_selection,
                vissa_max_iter=vissa_max_iter,
            )

            idx_sorted = np.asarray(sorted(np.unique(current.indices.tolist())), dtype=int)
            metrics = {
                "cv_score": None,
                "rmsecv": None,
                "r2cv": None,
                "accuracy": None,
                "f1_macro": None,
                "precision_macro": None,
                "recall_macro": None,
            }
            split_metrics = {
                "rmse_cal": None,
                "rmse_cv": None,
                "rmse_val": None,
                "r2_cal": None,
                "r2_cv": None,
                "r2_val": None,
                "accuracy_cal": None,
                "accuracy_cv": None,
                "accuracy_val": None,
                "f1_cal": None,
                "f1_cv": None,
                "f1_val": None,
                "precision_cal": None,
                "precision_cv": None,
                "precision_val": None,
                "recall_cal": None,
                "recall_cv": None,
                "recall_val": None,
                "cv_score": None,
            }
            feature_count_trajectory.append(int(idx_sorted.size))
            optimization_model_labels.append(f"k={int(idx_sorted.size)}")
            optimization_selected_variables_display.append(_selected_variables_display_text(idx_sorted, feature_labels))
            trajectory_cv_score.append(float(metrics["cv_score"]) if metrics["cv_score"] is not None else float("nan"))
            trajectory_rmsecv.append(float(metrics["rmsecv"]) if metrics["rmsecv"] is not None else float("nan"))
            trajectory_r2cv.append(float(metrics["r2cv"]) if metrics["r2cv"] is not None else float("nan"))
            trajectory_accuracy.append(float(metrics["accuracy"]) if metrics["accuracy"] is not None else float("nan"))
            trajectory_f1_macro.append(float(metrics["f1_macro"]) if metrics["f1_macro"] is not None else float("nan"))
            trajectory_precision_macro.append(float(metrics["precision_macro"]) if metrics["precision_macro"] is not None else float("nan"))
            trajectory_recall_macro.append(float(metrics["recall_macro"]) if metrics["recall_macro"] is not None else float("nan"))
            trajectory_rmse_cal.append(float(split_metrics["rmse_cal"]) if split_metrics["rmse_cal"] is not None else float("nan"))
            trajectory_rmse_cv.append(float(split_metrics["rmse_cv"]) if split_metrics["rmse_cv"] is not None else float("nan"))
            trajectory_rmse_val.append(float(split_metrics["rmse_val"]) if split_metrics["rmse_val"] is not None else float("nan"))
            trajectory_r2_cal.append(float(split_metrics["r2_cal"]) if split_metrics["r2_cal"] is not None else float("nan"))
            trajectory_r2_cv.append(float(split_metrics["r2_cv"]) if split_metrics["r2_cv"] is not None else float("nan"))
            trajectory_r2_val.append(float(split_metrics["r2_val"]) if split_metrics["r2_val"] is not None else float("nan"))
            trajectory_accuracy_cal.append(float(split_metrics["accuracy_cal"]) if split_metrics["accuracy_cal"] is not None else float("nan"))
            trajectory_accuracy_cv.append(float(split_metrics["accuracy_cv"]) if split_metrics["accuracy_cv"] is not None else float("nan"))
            trajectory_accuracy_val.append(float(split_metrics["accuracy_val"]) if split_metrics["accuracy_val"] is not None else float("nan"))
            trajectory_f1_cal.append(float(split_metrics["f1_cal"]) if split_metrics["f1_cal"] is not None else float("nan"))
            trajectory_f1_cv.append(float(split_metrics["f1_cv"]) if split_metrics["f1_cv"] is not None else float("nan"))
            trajectory_f1_val.append(float(split_metrics["f1_val"]) if split_metrics["f1_val"] is not None else float("nan"))
            trajectory_precision_cal.append(float(split_metrics["precision_cal"]) if split_metrics["precision_cal"] is not None else float("nan"))
            trajectory_precision_cv.append(float(split_metrics["precision_cv"]) if split_metrics["precision_cv"] is not None else float("nan"))
            trajectory_precision_val.append(float(split_metrics["precision_val"]) if split_metrics["precision_val"] is not None else float("nan"))
            trajectory_recall_cal.append(float(split_metrics["recall_cal"]) if split_metrics["recall_cal"] is not None else float("nan"))
            trajectory_recall_cv.append(float(split_metrics["recall_cv"]) if split_metrics["recall_cv"] is not None else float("nan"))
            trajectory_recall_val.append(float(split_metrics["recall_val"]) if split_metrics["recall_val"] is not None else float("nan"))
            optimization_is_selected_model.append(0)

            ga_count_stability.append(float("nan"))
            ga_count_restart_mean_score.append(float("nan"))

            if best_selection is None:
                best_selection = SelectionResult(indices=idx_sorted, scores=current.scores, metadata=current.metadata)
                best_count = int(idx_sorted.size)

        if best_selection is None or best_selection.indices.size == 0:
            raise ValueError("Feature selection failed to select variables.")

        selected_idx = np.asarray(sorted(np.unique(best_selection.indices.tolist())), dtype=int)

    selected_mask = np.zeros(n_features, dtype=bool)
    selected_mask[selected_idx] = True

    X_cal_sel = np.asarray(X_cal_2d[:, selected_idx], dtype=float)
    X_val_sel = np.asarray(X_val_2d[:, selected_idx], dtype=float) if X_val_2d is not None else None

    axis_n_info_sel = _slice_axis_info(axis_n_info, selected_idx, n_features_total=n_features)
    axis_t_info_sel = _slice_axis_info(axis_t_info, selected_idx, n_features_total=n_features)

    selection_source = "override_file" if override_from_file else "algorithm"
    selection_method_out = (
        str(override_selection_payload.get('selection_method', 'manual')).strip().lower()
        if override_from_file and override_selection_payload is not None
        else selection_method_norm
    )

    ga_selection_frequency = None
    ga_consensus_indices = None
    ga_restart_scores = None
    if selection_method_out == "ga" and isinstance(best_selection.metadata, dict):
        _freq = best_selection.metadata.get("ga_selection_frequency")
        _consensus = best_selection.metadata.get("ga_consensus_indices")
        _restart_scores = best_selection.metadata.get("ga_restart_scores")
        if _freq is not None:
            ga_selection_frequency = np.asarray(_freq, dtype=float)
        if _consensus is not None:
            ga_consensus_indices = np.asarray(_consensus, dtype=int)
        if isinstance(_restart_scores, list):
            ga_restart_scores = [float(v) for v in _restart_scores]
    selection_override_path_used = (
        str(override_selection_payload.get('path_used', override_path))
        if override_from_file and override_selection_payload is not None
        else ""
    )
    selection_override_payload_text = _build_selection_override_payload_text(
        selected_idx_zero_based=selected_idx,
        selection_method=selection_method_out,
        n_features_original=n_features,
    )
    selected_intervals_zero_based = _indices_to_interval_text(selected_idx, one_based=False)
    selected_intervals_one_based = _indices_to_interval_text(selected_idx, one_based=True)
    selected_variables_display = _selected_variables_display_text(selected_idx, feature_labels)

    selection_metadata: Dict[str, Any] = {
        "task_type": task_norm,
        "selection_method": selection_method_out,
        "n_features_original": int(n_features),
        "n_features_selected": int(selected_idx.size),
        "selected_indices_zero_based": [int(i) for i in selected_idx.tolist()],
        "selected_indices_one_based": [int(i) + 1 for i in selected_idx.tolist()],
        "selected_intervals_zero_based": str(selected_intervals_zero_based),
        "selected_intervals_one_based": str(selected_intervals_one_based),
        "selected_variables_display": str(selected_variables_display),
        "candidate_feature_counts": [int(v) for v in candidate_counts],
        "feature_count_optimized": False,
        "selected_feature_count": int(best_count),
        "selection_cv_score": None,
        "cv_splits_used": int(cv_splits_used_count),
        "sfs_direction": sfs_direction_norm if selection_method_out == "sfs" else None,
        "selection_source": selection_source,
        "selection_override_file_path_used": selection_override_path_used,
        "selection_override_file_applied": bool(override_from_file),
        "classification_family_hint": str(classification_family_hint or "") or None,
        "ga_n_restarts": None,
        "ga_consensus_mode": None,
        "ga_frequency_threshold": None,
        "feature_count_trajectory": [int(v) for v in feature_count_trajectory],
        "trajectory_cv_score": [float(v) for v in trajectory_cv_score],
        "trajectory_rmsecv": [float(v) for v in trajectory_rmsecv],
        "trajectory_r2cv": [float(v) for v in trajectory_r2cv],
        "trajectory_accuracy": [float(v) for v in trajectory_accuracy],
        "trajectory_f1_macro": [float(v) for v in trajectory_f1_macro],
        "trajectory_precision_macro": [float(v) for v in trajectory_precision_macro],
        "trajectory_recall_macro": [float(v) for v in trajectory_recall_macro],
        "trajectory_rmse_cal": [float(v) for v in trajectory_rmse_cal],
        "trajectory_rmse_cv": [float(v) for v in trajectory_rmse_cv],
        "trajectory_rmse_val": [float(v) for v in trajectory_rmse_val],
        "trajectory_r2_cal": [float(v) for v in trajectory_r2_cal],
        "trajectory_r2_cv": [float(v) for v in trajectory_r2_cv],
        "trajectory_r2_val": [float(v) for v in trajectory_r2_val],
        "trajectory_accuracy_cal": [float(v) for v in trajectory_accuracy_cal],
        "trajectory_accuracy_cv": [float(v) for v in trajectory_accuracy_cv],
        "trajectory_accuracy_val": [float(v) for v in trajectory_accuracy_val],
        "trajectory_f1_cal": [float(v) for v in trajectory_f1_cal],
        "trajectory_f1_cv": [float(v) for v in trajectory_f1_cv],
        "trajectory_f1_val": [float(v) for v in trajectory_f1_val],
        "trajectory_precision_cal": [float(v) for v in trajectory_precision_cal],
        "trajectory_precision_cv": [float(v) for v in trajectory_precision_cv],
        "trajectory_precision_val": [float(v) for v in trajectory_precision_val],
        "trajectory_recall_cal": [float(v) for v in trajectory_recall_cal],
        "trajectory_recall_cv": [float(v) for v in trajectory_recall_cv],
        "trajectory_recall_val": [float(v) for v in trajectory_recall_val],
        "ga_count_stability": [float(v) for v in ga_count_stability],
        "ga_count_restart_mean_score": [float(v) for v in ga_count_restart_mean_score],
        "optimization_model_labels": [str(v) for v in optimization_model_labels],
        "optimization_selected_variables_display": [str(v) for v in optimization_selected_variables_display],
        "optimization_is_selected_model": [int(v) for v in optimization_is_selected_model],
    }
    selection_metadata.update(best_selection.metadata)

    if feature_count_trajectory:
        optimization_is_selected_model = [
            1 if int(count) == int(best_count) else 0
            for count in feature_count_trajectory
        ]

    selected_count_marker = np.asarray([int(best_count)], dtype=int)
    selected_cv_score_marker = np.asarray([
        _selected_metric_marker(feature_count_trajectory, trajectory_cv_score, best_count)
    ], dtype=float)
    selected_rmsecv_marker = np.asarray([
        _selected_metric_marker(feature_count_trajectory, trajectory_rmsecv, best_count)
    ], dtype=float)
    selected_r2cv_marker = np.asarray([
        _selected_metric_marker(feature_count_trajectory, trajectory_r2cv, best_count)
    ], dtype=float)
    selected_accuracy_marker = np.asarray([
        _selected_metric_marker(feature_count_trajectory, trajectory_accuracy, best_count)
    ], dtype=float)
    selected_f1_macro_marker = np.asarray([
        _selected_metric_marker(feature_count_trajectory, trajectory_f1_macro, best_count)
    ], dtype=float)
    selected_precision_macro_marker = np.asarray([
        _selected_metric_marker(feature_count_trajectory, trajectory_precision_macro, best_count)
    ], dtype=float)
    selected_recall_macro_marker = np.asarray([
        _selected_metric_marker(feature_count_trajectory, trajectory_recall_macro, best_count)
    ], dtype=float)

    return {
        "task_type": task_norm,
        "X_cal": X_cal_sel,
        "X_val": X_val_sel,
        "axis_n_info": axis_n_info_sel,
        "axis_t_info": axis_t_info_sel,
        "selected_variable_indices": selected_idx,
        "selected_variable_intervals_zero_based": str(selected_intervals_zero_based),
        "selected_variable_intervals_one_based": str(selected_intervals_one_based),
        "selected_variables_display": str(selected_variables_display),
        "selected_variable_mask": selected_mask,
        "selected_variable_scores": best_selection.scores,
        "selected_variable_count": int(selected_idx.size),
        "selection_method": selection_method_out,
        "selection_source": selection_source,
        "selection_override_file_path_used": selection_override_path_used,
        "selection_override_payload_text": selection_override_payload_text,
        "ga_selection_frequency": ga_selection_frequency,
        "ga_consensus_indices": ga_consensus_indices,
        "ga_restart_scores": ga_restart_scores,
        "feature_count_trajectory": np.asarray(feature_count_trajectory, dtype=int),
        "trajectory_cv_score": np.asarray(trajectory_cv_score, dtype=float),
        "trajectory_rmsecv": np.asarray(trajectory_rmsecv, dtype=float),
        "trajectory_r2cv": np.asarray(trajectory_r2cv, dtype=float),
        "trajectory_accuracy": np.asarray(trajectory_accuracy, dtype=float),
        "trajectory_f1_macro": np.asarray(trajectory_f1_macro, dtype=float),
        "trajectory_precision_macro": np.asarray(trajectory_precision_macro, dtype=float),
        "trajectory_recall_macro": np.asarray(trajectory_recall_macro, dtype=float),
        "trajectory_rmse_cal": np.asarray(trajectory_rmse_cal, dtype=float),
        "trajectory_rmse_cv": np.asarray(trajectory_rmse_cv, dtype=float),
        "trajectory_rmse_val": np.asarray(trajectory_rmse_val, dtype=float),
        "trajectory_r2_cal": np.asarray(trajectory_r2_cal, dtype=float),
        "trajectory_r2_cv": np.asarray(trajectory_r2_cv, dtype=float),
        "trajectory_r2_val": np.asarray(trajectory_r2_val, dtype=float),
        "trajectory_accuracy_cal": np.asarray(trajectory_accuracy_cal, dtype=float),
        "trajectory_accuracy_cv": np.asarray(trajectory_accuracy_cv, dtype=float),
        "trajectory_accuracy_val": np.asarray(trajectory_accuracy_val, dtype=float),
        "trajectory_f1_cal": np.asarray(trajectory_f1_cal, dtype=float),
        "trajectory_f1_cv": np.asarray(trajectory_f1_cv, dtype=float),
        "trajectory_f1_val": np.asarray(trajectory_f1_val, dtype=float),
        "trajectory_precision_cal": np.asarray(trajectory_precision_cal, dtype=float),
        "trajectory_precision_cv": np.asarray(trajectory_precision_cv, dtype=float),
        "trajectory_precision_val": np.asarray(trajectory_precision_val, dtype=float),
        "trajectory_recall_cal": np.asarray(trajectory_recall_cal, dtype=float),
        "trajectory_recall_cv": np.asarray(trajectory_recall_cv, dtype=float),
        "trajectory_recall_val": np.asarray(trajectory_recall_val, dtype=float),
        "ga_count_stability": np.asarray(ga_count_stability, dtype=float),
        "ga_count_restart_mean_score": np.asarray(ga_count_restart_mean_score, dtype=float),
        "optimization_model_labels": np.asarray(optimization_model_labels, dtype=object),
        "optimization_selected_variables_display": np.asarray(optimization_selected_variables_display, dtype=object),
        "optimization_is_selected_model": np.asarray(optimization_is_selected_model, dtype=int),
        "selected_count_marker": selected_count_marker,
        "selected_cv_score_marker": selected_cv_score_marker,
        "selected_rmsecv_marker": selected_rmsecv_marker,
        "selected_r2cv_marker": selected_r2cv_marker,
        "selected_accuracy_marker": selected_accuracy_marker,
        "selected_f1_macro_marker": selected_f1_macro_marker,
        "selected_precision_macro_marker": selected_precision_macro_marker,
        "selected_recall_macro_marker": selected_recall_macro_marker,
        "selection_metadata": selection_metadata,
    }
