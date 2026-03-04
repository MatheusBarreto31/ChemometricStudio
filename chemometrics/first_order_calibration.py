from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

try:
    from chemometrics.cv_pipeline import CVConfig, CVPipeline
    HAS_CV = True
except ImportError:
    HAS_CV = False

from chemometrics.input_parsing import parse_numeric_spec


def _ensure_2d(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    rmsep = float(np.sqrt(np.mean(residual ** 2)))

    ss_res = float(np.sum(residual ** 2))
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        'RMSE': rmsep,
        'RMSEP': rmsep,
        'R2': r2,
        'n_samples': int(y_true.shape[0]),
    }


def _coerce_int_candidates(
    values: List[Any],
    default_max: int,
    from_one_on_single: bool = True,
) -> List[int]:
    parsed = [int(round(float(v))) for v in values if float(v) >= 1]
    if len(parsed) == 1 and from_one_on_single:
        single = max(1, parsed[0])
        parsed = list(range(1, single + 1))
    if not parsed:
        parsed = list(range(1, max(1, int(default_max)) + 1))
    unique_sorted = sorted(set(int(v) for v in parsed if int(v) >= 1))
    return unique_sorted


def _coerce_float_candidates(values: List[Any], default_values: List[float]) -> List[float]:
    parsed = [float(v) for v in values if float(v) > 0]
    if not parsed:
        parsed = list(default_values)
    return sorted(set(parsed))


def _parse_parameter_candidates(
    optimize_parameters: bool,
    parameter_range: Optional[Any],
    fixed_value: Any,
    model_type: str,
    X_cal: np.ndarray,
    local_method: Optional[str] = None,
) -> List[Any]:
    n_samples, n_features = X_cal.shape
    max_latent = max(1, min(n_features, n_samples - 1))
    max_neighbors = max(1, min(n_samples - 1, 25))

    if not optimize_parameters:
        return [fixed_value]

    parsed = parse_numeric_spec(parameter_range)

    if model_type in ('pls', 'pcr'):
        return _coerce_int_candidates(parsed, default_max=min(max_latent, 15), from_one_on_single=True)
    if model_type == 'ridge':
        return _coerce_float_candidates(parsed, default_values=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    if model_type == 'local':
        local_method_norm = str(local_method).strip().lower()
        if local_method_norm == 'idw':
            return _coerce_float_candidates(parsed, default_values=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        if local_method_norm == 'adaptive':
            return _coerce_float_candidates(parsed, default_values=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        if local_method_norm == 'radius':
            return _coerce_float_candidates(parsed, default_values=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        if local_method_norm == 'kernel':
            return _coerce_float_candidates(parsed, default_values=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        return _coerce_int_candidates(parsed, default_max=max_neighbors, from_one_on_single=True)
    return [fixed_value]


def _compute_query_distances(
    X_train: np.ndarray,
    x_query: np.ndarray,
    distance_metric: str,
    mahalanobis_vi: Optional[np.ndarray] = None,
) -> np.ndarray:
    metric = str(distance_metric).strip().lower()
    query = np.asarray(x_query, dtype=float).reshape(1, -1)

    if metric == 'euclidean':
        return np.linalg.norm(X_train - query, axis=1)
    if metric == 'manhattan':
        return cdist(X_train, query, metric='cityblock').reshape(-1)
    if metric == 'minkowski':
        return cdist(X_train, query, metric='minkowski', p=3).reshape(-1)
    if metric == 'chebyshev':
        return cdist(X_train, query, metric='chebyshev').reshape(-1)
    if metric == 'mahalanobis':
        if mahalanobis_vi is None:
            cov = np.cov(X_train, rowvar=False)
            if np.ndim(cov) == 0:
                cov = np.asarray([[float(cov)]], dtype=float)
            cov = np.asarray(cov, dtype=float)
            cov += np.eye(cov.shape[0], dtype=float) * 1e-10
            mahalanobis_vi = np.linalg.pinv(cov)
        return cdist(X_train, query, metric='mahalanobis', VI=mahalanobis_vi).reshape(-1)
    if metric == 'chord':
        train_norm = np.linalg.norm(X_train, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query)
        X_train_unit = X_train / np.maximum(train_norm, 1e-12)
        x_query_unit = query / max(float(query_norm), 1e-12)
        return np.linalg.norm(X_train_unit - x_query_unit, axis=1)

    raise ValueError(
        "local_distance must be one of: 'euclidean', 'mahalanobis', 'manhattan', 'minkowski', 'chebyshev', 'chord'."
    )


def _fit_predict_local(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    n_neighbors: int,
    local_method: str,
    idw_power: float,
    adaptive_alpha: float,
    radius_threshold: float,
    kernel_bandwidth: float,
    local_distance: str,
) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    n_neighbors = max(1, min(int(n_neighbors), X_train.shape[0]))
    idw_power = max(float(idw_power), 1e-12)
    adaptive_alpha = max(float(adaptive_alpha), 1e-12)
    radius_threshold = max(float(radius_threshold), 1e-12)
    kernel_bandwidth = max(float(kernel_bandwidth), 1e-12)
    local_distance = str(local_distance).strip().lower()
    local_method = str(local_method).strip().lower()
    outputs = np.zeros((X_test.shape[0], Y_train.shape[1]), dtype=float)

    mahalanobis_vi = None
    if local_distance == 'mahalanobis':
        cov = np.cov(X_train, rowvar=False)
        if np.ndim(cov) == 0:
            cov = np.asarray([[float(cov)]], dtype=float)
        cov = np.asarray(cov, dtype=float)
        cov += np.eye(cov.shape[0], dtype=float) * 1e-10
        mahalanobis_vi = np.linalg.pinv(cov)

    for idx, x_query in enumerate(X_test):
        distances = _compute_query_distances(
            X_train=X_train,
            x_query=x_query,
            distance_metric=local_distance,
            mahalanobis_vi=mahalanobis_vi,
        )
        if local_method == 'knn':
            nn_idx = np.argsort(distances, kind='mergesort')[:n_neighbors]
            outputs[idx] = np.mean(Y_train[nn_idx], axis=0)
            continue

        if local_method == 'idw':
            nn_idx = np.argsort(distances, kind='mergesort')[:n_neighbors]
            d = np.maximum(distances[nn_idx], 1e-12)
            weights = 1.0 / np.power(d, idw_power)
            w_sum = float(np.sum(weights))
            if w_sum <= 0:
                outputs[idx] = np.mean(Y_train[nn_idx], axis=0)
            else:
                outputs[idx] = np.sum(Y_train[nn_idx] * weights.reshape(-1, 1), axis=0) / w_sum
            continue

        if local_method == 'adaptive':
            similarities = 1.0 / (1.0 + np.maximum(distances, 0.0))
            ranked_idx = np.argsort(-similarities, kind='mergesort')
            ranked_sim = similarities[ranked_idx]
            thresholds = [((10 - k) * 0.1) ** adaptive_alpha for k in range(1, 11)]
            k_sel = 0
            for threshold in thresholds:
                k_sel = int(np.count_nonzero(ranked_sim > threshold))
                if k_sel > 0:
                    break
            if k_sel <= 0:
                k_sel = ranked_idx.shape[0]
            nn_idx = ranked_idx[:max(1, k_sel)]
            outputs[idx] = np.mean(Y_train[nn_idx], axis=0)
            continue

        if local_method == 'radius':
            nn_idx = np.flatnonzero(distances <= radius_threshold)
            if nn_idx.size == 0:
                nn_idx = np.argsort(distances, kind='mergesort')[:1]
            outputs[idx] = np.mean(Y_train[nn_idx], axis=0)
            continue

        if local_method == 'kernel':
            d = np.maximum(distances, 1e-12)
            weights = np.exp(-0.5 * np.square(d / kernel_bandwidth))
            w_sum = float(np.sum(weights))
            if w_sum <= 0:
                outputs[idx] = np.mean(Y_train, axis=0)
            else:
                outputs[idx] = np.sum(Y_train * weights.reshape(-1, 1), axis=0) / w_sum
            continue

        raise ValueError("Unsupported local_method for local regression.")

    return outputs


def _predict_local_leave_one_out(
    X_data: np.ndarray,
    Y_data: np.ndarray,
    n_neighbors: int,
    local_method: str,
    idw_power: float,
    adaptive_alpha: float,
    radius_threshold: float,
    kernel_bandwidth: float,
    local_distance: str,
) -> np.ndarray:
    X_data = np.asarray(X_data, dtype=float)
    Y_data = np.asarray(Y_data, dtype=float)
    outputs = np.zeros_like(Y_data, dtype=float)

    n_samples = X_data.shape[0]
    if n_samples <= 1:
        return np.mean(Y_data, axis=0, keepdims=True).repeat(n_samples, axis=0)

    for idx in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[idx] = False
        X_train = X_data[mask]
        Y_train = Y_data[mask]
        y_pred = _fit_predict_local(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_data[idx:idx + 1],
            n_neighbors=n_neighbors,
            local_method=local_method,
            idw_power=idw_power,
            adaptive_alpha=adaptive_alpha,
            radius_threshold=radius_threshold,
            kernel_bandwidth=kernel_bandwidth,
            local_distance=local_distance,
        )
        outputs[idx] = y_pred[0]

    return outputs


def _fit_model(
    model_type: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    n_components: int,
    ridge_alpha: float,
    n_neighbors: int,
    local_method: str,
    idw_power: float,
    adaptive_alpha: float,
    radius_threshold: float,
    kernel_bandwidth: float,
    local_distance: str,
) -> Dict[str, Any]:
    model_type = str(model_type).lower()
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)

    if model_type == 'ols':
        model = LinearRegression()
        model.fit(X_train, Y_train)
        return {'model_type': model_type, 'model': model}

    if model_type == 'ridge':
        model = Ridge(alpha=float(ridge_alpha))
        model.fit(X_train, Y_train)
        return {'model_type': model_type, 'model': model, 'alpha': float(ridge_alpha)}

    if model_type == 'pls':
        max_comp = max(1, min(X_train.shape[1], X_train.shape[0] - 1))
        n_comp = max(1, min(int(n_components), max_comp))
        model = PLSRegression(n_components=n_comp)
        model.fit(X_train, Y_train)
        return {'model_type': model_type, 'model': model, 'n_components': int(n_comp)}

    if model_type == 'pcr':
        max_comp = max(1, min(X_train.shape[1], X_train.shape[0] - 1))
        n_comp = max(1, min(int(n_components), max_comp))
        pca = PCA(n_components=n_comp)
        scores = pca.fit_transform(X_train)
        reg = LinearRegression()
        reg.fit(scores, Y_train)
        return {
            'model_type': model_type,
            'pca': pca,
            'reg': reg,
            'n_components': int(n_comp),
        }

    if model_type == 'local':
        return {
            'model_type': model_type,
            'X_train': X_train,
            'Y_train': Y_train,
            'n_neighbors': int(max(1, n_neighbors)),
            'local_method': str(local_method).lower(),
            'idw_power': float(max(idw_power, 1e-12)),
            'adaptive_alpha': float(max(adaptive_alpha, 1e-12)),
            'radius_threshold': float(max(radius_threshold, 1e-12)),
            'kernel_bandwidth': float(max(kernel_bandwidth, 1e-12)),
            'local_distance': str(local_distance).strip().lower(),
        }

    raise ValueError(f"Unsupported model_type '{model_type}'.")


def _predict_model(model_info: Dict[str, Any], X_data: np.ndarray) -> np.ndarray:
    model_type = model_info.get('model_type')
    X_data = np.asarray(X_data, dtype=float)

    if model_type in ('ols', 'ridge', 'pls'):
        return np.asarray(model_info['model'].predict(X_data), dtype=float)

    if model_type == 'pcr':
        pca = model_info['pca']
        reg = model_info['reg']
        scores = pca.transform(X_data)
        return np.asarray(reg.predict(scores), dtype=float)

    if model_type == 'local':
        return _fit_predict_local(
            model_info['X_train'],
            model_info['Y_train'],
            X_data,
            model_info['n_neighbors'],
            model_info['local_method'],
            model_info.get('idw_power', 1.0),
            model_info.get('adaptive_alpha', 1.0),
            model_info.get('radius_threshold', 1.0),
            model_info.get('kernel_bandwidth', 1.0),
            model_info.get('local_distance', 'euclidean'),
        )

    raise ValueError(f"Unsupported model_type '{model_type}'.")


def _build_cv_splits(X_cal: np.ndarray, Y_cal: np.ndarray, cv_config: CVConfig):
    pipeline = CVPipeline(cv_config)
    y_for_split = None

    strategy = str(getattr(cv_config, 'cv_strategy', '')).strip().lower()
    if strategy == 'stratified_kfold':
        y_vec = np.asarray(Y_cal).reshape(-1)
        n_bins = max(2, min(int(getattr(cv_config, 'n_splits', 5)), 10))
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.unique(np.quantile(y_vec, quantiles))
        if bin_edges.size <= 2:
            y_for_split = np.zeros_like(y_vec, dtype=int)
        else:
            y_for_split = np.digitize(y_vec, bin_edges[1:-1], right=True)

    return list(pipeline.splitter.get_splits(X_cal, y_for_split))


def _cross_validated_predictions(
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    model_type: str,
    n_components: int,
    ridge_alpha: float,
    n_neighbors: int,
    local_method: str,
    idw_power: float,
    adaptive_alpha: float,
    radius_threshold: float,
    kernel_bandwidth: float,
    local_distance: str,
    cv_config: Optional[CVConfig],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    if cv_config is None or not HAS_CV or not cv_config.is_enabled():
        return None, None

    splits = _build_cv_splits(X_cal, Y_cal, cv_config)
    y_cv_pred = np.zeros_like(Y_cal, dtype=float)
    fold_metrics: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X_cal[train_idx], X_cal[test_idx]
        Y_train, Y_test = Y_cal[train_idx], Y_cal[test_idx]

        fold_model = _fit_model(
            model_type=model_type,
            X_train=X_train,
            Y_train=Y_train,
            n_components=n_components,
            ridge_alpha=ridge_alpha,
            n_neighbors=n_neighbors,
            local_method=local_method,
            idw_power=idw_power,
            adaptive_alpha=adaptive_alpha,
            radius_threshold=radius_threshold,
            kernel_bandwidth=kernel_bandwidth,
            local_distance=local_distance,
        )
        fold_pred = _predict_model(fold_model, X_test)
        y_cv_pred[test_idx] = fold_pred

        fold_metrics.append({
            'fold': int(fold_idx),
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'metrics': _compute_metrics(Y_test, fold_pred),
        })

    cv_metrics = _compute_metrics(Y_cal, y_cv_pred)
    cv_results = {
        'n_folds': int(len(splits)),
        'fold_metrics': fold_metrics,
        'aggregated_metrics': cv_metrics,
        'cv_strategy': str(cv_config.cv_strategy),
    }
    return y_cv_pred, cv_results


def first_order_calibration(
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    model_type: str = 'pls',
    n_components: int = 2,
    ridge_alpha: float = 1.0,
    local_method: str = 'knn',
    n_neighbors: int = 5,
    idw_power: float = 1.0,
    adaptive_alpha: float = 1.0,
    radius_threshold: float = 1.0,
    kernel_bandwidth: float = 1.0,
    local_distance: str = 'euclidean',
    optimize_parameters: bool = False,
    n_components_range: Optional[Any] = None,
    ridge_alpha_range: Optional[Any] = None,
    n_neighbors_range: Optional[Any] = None,
    idw_power_range: Optional[Any] = None,
    adaptive_alpha_range: Optional[Any] = None,
    radius_threshold_range: Optional[Any] = None,
    kernel_bandwidth_range: Optional[Any] = None,
    parameter_range: Optional[Any] = None,
    cv_config: Optional[Any] = None,
    fold: int = 0,
    **kwargs,
) -> Tuple[Any, ...]:
    if cv_config is not None and isinstance(cv_config, dict) and 'cv_config' in cv_config:
        cv_config = cv_config['cv_config']

    if X_cal is None and 'X_cal_train' in kwargs:
        X_cal = kwargs['X_cal_train']
    if Y_cal is None and 'Y_cal_train' in kwargs:
        Y_cal = kwargs['Y_cal_train']
    if X_val is None and 'X_val_train' in kwargs:
        X_val = kwargs['X_val_train']
    if Y_val is None and 'Y_val_train' in kwargs:
        Y_val = kwargs['Y_val_train']

    X_cal = _ensure_2d(X_cal)
    Y_cal = _ensure_2d(Y_cal)
    X_val = _ensure_2d(X_val)
    Y_val = _ensure_2d(Y_val)

    if X_cal is None or Y_cal is None:
        raise ValueError('X_cal and Y_cal are required.')
    if X_cal.shape[0] != Y_cal.shape[0]:
        raise ValueError('X_cal and Y_cal must have the same number of samples.')

    model_type_norm = str(model_type).strip().lower()
    local_method_norm = str(local_method).strip().lower()
    if local_method_norm not in ('knn', 'idw', 'adaptive', 'radius', 'kernel'):
        raise ValueError("local_method must be one of: 'knn', 'idw', 'adaptive', 'radius', 'kernel'.")
    local_distance_norm = str(local_distance).strip().lower()
    valid_local_distances = ('euclidean', 'mahalanobis', 'manhattan', 'minkowski', 'chebyshev', 'chord')
    if local_distance_norm not in valid_local_distances:
        raise ValueError(
            "local_distance must be one of: 'euclidean', 'mahalanobis', 'manhattan', 'minkowski', 'chebyshev', 'chord'."
        )

    parameter_name = None
    fixed_value = None
    if model_type_norm in ('pls', 'pcr'):
        parameter_name = 'n_components'
        fixed_value = int(n_components)
    elif model_type_norm == 'ridge':
        parameter_name = 'ridge_alpha'
        fixed_value = float(ridge_alpha)
    elif model_type_norm == 'local':
        if local_method_norm == 'knn':
            parameter_name = 'n_neighbors'
            fixed_value = int(n_neighbors)
        elif local_method_norm == 'idw':
            parameter_name = 'idw_power'
            fixed_value = float(idw_power)
        elif local_method_norm == 'adaptive':
            parameter_name = 'adaptive_alpha'
            fixed_value = float(adaptive_alpha)
        elif local_method_norm == 'radius':
            parameter_name = 'radius_threshold'
            fixed_value = float(radius_threshold)
        elif local_method_norm == 'kernel':
            parameter_name = 'kernel_bandwidth'
            fixed_value = float(kernel_bandwidth)

    selected_range_input = parameter_range
    if model_type_norm in ('pls', 'pcr') and n_components_range not in (None, ''):
        selected_range_input = n_components_range
    elif model_type_norm == 'ridge' and ridge_alpha_range not in (None, ''):
        selected_range_input = ridge_alpha_range
    elif model_type_norm == 'local':
        if local_method_norm == 'knn' and n_neighbors_range not in (None, ''):
            selected_range_input = n_neighbors_range
        elif local_method_norm == 'idw' and idw_power_range not in (None, ''):
            selected_range_input = idw_power_range
        elif local_method_norm == 'adaptive' and adaptive_alpha_range not in (None, ''):
            selected_range_input = adaptive_alpha_range
        elif local_method_norm == 'radius' and radius_threshold_range not in (None, ''):
            selected_range_input = radius_threshold_range
        elif local_method_norm == 'kernel' and kernel_bandwidth_range not in (None, ''):
            selected_range_input = kernel_bandwidth_range

    candidates = _parse_parameter_candidates(
        optimize_parameters=bool(optimize_parameters and parameter_name is not None),
        parameter_range=selected_range_input,
        fixed_value=fixed_value,
        model_type=model_type_norm,
        X_cal=X_cal,
        local_method=local_method_norm,
    )

    optimization_parameter_values: List[Any] = []
    optimization_self_r2: List[float] = []
    optimization_cv_r2: List[float] = []
    optimization_val_r2: List[float] = []
    optimization_self_rmsep: List[float] = []
    optimization_cv_rmsep: List[float] = []
    optimization_val_rmsep: List[float] = []

    best_candidate = candidates[0] if candidates else fixed_value
    best_score = np.inf
    use_cv_for_selection = cv_config is not None and HAS_CV and cv_config.is_enabled()
    local_self_uses_loo = model_type_norm == 'local'

    for candidate in candidates:
        c_n_components = int(candidate) if model_type_norm in ('pls', 'pcr') else int(n_components)
        c_alpha = float(candidate) if model_type_norm == 'ridge' else float(ridge_alpha)
        c_neighbors = int(candidate) if (model_type_norm == 'local' and local_method_norm == 'knn') else int(n_neighbors)
        c_idw_power = float(candidate) if (model_type_norm == 'local' and local_method_norm == 'idw') else float(idw_power)
        c_adaptive_alpha = float(candidate) if (model_type_norm == 'local' and local_method_norm == 'adaptive') else float(adaptive_alpha)
        c_radius = float(candidate) if (model_type_norm == 'local' and local_method_norm == 'radius') else float(radius_threshold)
        c_kernel_bw = float(candidate) if (model_type_norm == 'local' and local_method_norm == 'kernel') else float(kernel_bandwidth)

        model_info = _fit_model(
            model_type=model_type_norm,
            X_train=X_cal,
            Y_train=Y_cal,
            n_components=c_n_components,
            ridge_alpha=c_alpha,
            n_neighbors=c_neighbors,
            local_method=local_method_norm,
            idw_power=c_idw_power,
            adaptive_alpha=c_adaptive_alpha,
            radius_threshold=c_radius,
            kernel_bandwidth=c_kernel_bw,
            local_distance=local_distance_norm,
        )
        if local_self_uses_loo:
            y_self_pred = _predict_local_leave_one_out(
                X_data=X_cal,
                Y_data=Y_cal,
                n_neighbors=c_neighbors,
                local_method=local_method_norm,
                idw_power=c_idw_power,
                adaptive_alpha=c_adaptive_alpha,
                radius_threshold=c_radius,
                kernel_bandwidth=c_kernel_bw,
                local_distance=local_distance_norm,
            )
        else:
            y_self_pred = _predict_model(model_info, X_cal)
        self_metrics = _compute_metrics(Y_cal, y_self_pred)

        y_val_pred_candidate = _predict_model(model_info, X_val) if X_val is not None else None
        val_metrics = _compute_metrics(Y_val, y_val_pred_candidate) if (Y_val is not None and y_val_pred_candidate is not None) else None

        y_cv_pred_candidate, _ = _cross_validated_predictions(
            X_cal=X_cal,
            Y_cal=Y_cal,
            model_type=model_type_norm,
            n_components=c_n_components,
            ridge_alpha=c_alpha,
            n_neighbors=c_neighbors,
            local_method=local_method_norm,
            idw_power=c_idw_power,
            adaptive_alpha=c_adaptive_alpha,
            radius_threshold=c_radius,
            kernel_bandwidth=c_kernel_bw,
            local_distance=local_distance_norm,
            cv_config=cv_config,
        )
        cv_metrics = _compute_metrics(Y_cal, y_cv_pred_candidate) if y_cv_pred_candidate is not None else None

        optimization_parameter_values.append(candidate)
        optimization_self_r2.append(float(self_metrics['R2']))
        optimization_self_rmsep.append(float(self_metrics['RMSEP']))
        optimization_cv_r2.append(float(cv_metrics['R2']) if cv_metrics is not None else float('nan'))
        optimization_cv_rmsep.append(float(cv_metrics['RMSEP']) if cv_metrics is not None else float('nan'))
        optimization_val_r2.append(float(val_metrics['R2']) if val_metrics is not None else float('nan'))
        optimization_val_rmsep.append(float(val_metrics['RMSEP']) if val_metrics is not None else float('nan'))

        if use_cv_for_selection and cv_metrics is not None:
            score = float(cv_metrics['RMSEP'])
        elif val_metrics is not None:
            score = float(val_metrics['RMSEP'])
        else:
            score = float(self_metrics['RMSEP'])
        if score < best_score:
            best_score = score
            best_candidate = candidate

    final_n_components = int(best_candidate) if model_type_norm in ('pls', 'pcr') else int(n_components)
    final_alpha = float(best_candidate) if model_type_norm == 'ridge' else float(ridge_alpha)
    final_neighbors = int(best_candidate) if (model_type_norm == 'local' and local_method_norm == 'knn') else int(n_neighbors)
    final_idw_power = float(best_candidate) if (model_type_norm == 'local' and local_method_norm == 'idw') else float(idw_power)
    final_adaptive_alpha = float(best_candidate) if (model_type_norm == 'local' and local_method_norm == 'adaptive') else float(adaptive_alpha)
    final_radius = float(best_candidate) if (model_type_norm == 'local' and local_method_norm == 'radius') else float(radius_threshold)
    final_kernel_bw = float(best_candidate) if (model_type_norm == 'local' and local_method_norm == 'kernel') else float(kernel_bandwidth)

    final_model = _fit_model(
        model_type=model_type_norm,
        X_train=X_cal,
        Y_train=Y_cal,
        n_components=final_n_components,
        ridge_alpha=final_alpha,
        n_neighbors=final_neighbors,
        local_method=local_method_norm,
        idw_power=final_idw_power,
        adaptive_alpha=final_adaptive_alpha,
        radius_threshold=final_radius,
        kernel_bandwidth=final_kernel_bw,
        local_distance=local_distance_norm,
    )

    if local_self_uses_loo:
        y_cal_pred = _predict_local_leave_one_out(
            X_data=X_cal,
            Y_data=Y_cal,
            n_neighbors=final_neighbors,
            local_method=local_method_norm,
            idw_power=final_idw_power,
            adaptive_alpha=final_adaptive_alpha,
            radius_threshold=final_radius,
            kernel_bandwidth=final_kernel_bw,
            local_distance=local_distance_norm,
        )
    else:
        y_cal_pred = _predict_model(final_model, X_cal)
    y_val_pred = _predict_model(final_model, X_val) if X_val is not None else None

    y_cv_pred, cv_results = _cross_validated_predictions(
        X_cal=X_cal,
        Y_cal=Y_cal,
        model_type=model_type_norm,
        n_components=final_n_components,
        ridge_alpha=final_alpha,
        n_neighbors=final_neighbors,
        local_method=local_method_norm,
        idw_power=final_idw_power,
        adaptive_alpha=final_adaptive_alpha,
        radius_threshold=final_radius,
        kernel_bandwidth=final_kernel_bw,
        local_distance=local_distance_norm,
        cv_config=cv_config,
    )

    metrics_cal = _compute_metrics(Y_cal, y_cal_pred)
    metrics_cv = _compute_metrics(Y_cal, y_cv_pred) if y_cv_pred is not None else None
    metrics_val = _compute_metrics(Y_val, y_val_pred) if (Y_val is not None and y_val_pred is not None) else None

    optimization_results = {
        'parameter_name': parameter_name,
        'parameter_values': [float(v) if isinstance(v, (float, np.floating)) else int(v) for v in optimization_parameter_values],
        'self_r2': optimization_self_r2,
        'cv_r2': optimization_cv_r2,
        'val_r2': optimization_val_r2,
        'self_rmsep': optimization_self_rmsep,
        'cv_rmsep': optimization_cv_rmsep,
        'val_rmsep': optimization_val_rmsep,
        'best_value': best_candidate,
        'selection_source': 'cv' if use_cv_for_selection else ('validation' if (X_val is not None and Y_val is not None) else 'self'),
        'optimization_used': bool(optimize_parameters and parameter_name is not None),
    }

    selected_parameter_value = None
    if parameter_name == 'n_components':
        selected_parameter_value = int(final_n_components)
    elif parameter_name == 'ridge_alpha':
        selected_parameter_value = float(final_alpha)
    elif parameter_name == 'n_neighbors':
        selected_parameter_value = int(final_neighbors)
    elif parameter_name == 'idw_power':
        selected_parameter_value = float(final_idw_power)
    elif parameter_name == 'adaptive_alpha':
        selected_parameter_value = float(final_adaptive_alpha)
    elif parameter_name == 'radius_threshold':
        selected_parameter_value = float(final_radius)
    elif parameter_name == 'kernel_bandwidth':
        selected_parameter_value = float(final_kernel_bw)

    model_payload = {
        'model_type': model_type_norm,
        'local_method': local_method_norm if model_type_norm == 'local' else None,
        'local_distance': local_distance_norm if model_type_norm == 'local' else None,
        'idw_power': float(final_idw_power) if model_type_norm == 'local' and local_method_norm == 'idw' else None,
        'adaptive_alpha': float(final_adaptive_alpha) if model_type_norm == 'local' and local_method_norm == 'adaptive' else None,
        'radius_threshold': float(final_radius) if model_type_norm == 'local' and local_method_norm == 'radius' else None,
        'kernel_bandwidth': float(final_kernel_bw) if model_type_norm == 'local' and local_method_norm == 'kernel' else None,
    }
    metrics_payload = {
        'calibration': metrics_cal,
        'cv': metrics_cv,
        'validation': metrics_val,
    }

    return (
        y_cal_pred,
        y_val_pred,
        y_cv_pred,
        Y_cal,
        Y_val,
        model_payload,
        parameter_name,
        selected_parameter_value,
        metrics_payload,
        cv_results,
        optimization_results,
    )
