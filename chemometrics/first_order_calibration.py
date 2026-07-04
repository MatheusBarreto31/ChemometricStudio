from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, MultiTaskLasso, ElasticNet, MultiTaskElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GPRRBFKernel, ConstantKernel
from sklearn.kernel_approximation import RBFSampler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

try:
    from execution_reporting import emit_execution_message, emit_execution_warning, emit_execution_error
except ImportError:
    def emit_execution_message(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        return

    def emit_execution_warning(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        return

    def emit_execution_error(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        return

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


def _align_prediction_shape(y_pred: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """Align prediction shape with reference shape to avoid broadcast artifacts."""
    yp = np.asarray(y_pred, dtype=float)
    yr = np.asarray(y_ref)

    if yr.ndim == 2 and yr.shape[1] == 1 and yp.ndim == 1:
        return yp.reshape(-1, 1)
    if yr.ndim == 1 and yp.ndim == 2 and yp.shape[1] == 1:
        return yp.reshape(-1)
    return yp


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)

    residual = y_true_arr - y_pred_arr
    rmsep = float(np.sqrt(np.mean(residual ** 2)))
    rmsep_per = np.sqrt(np.mean(residual ** 2, axis=0))

    ss_res = float(np.sum(residual ** 2))
    ss_res_per = np.sum(residual ** 2, axis=0)
    y_mean = np.mean(y_true_arr, axis=0, keepdims=True)
    ss_tot = float(np.sum((y_true_arr - y_mean) ** 2))
    ss_tot_per = np.sum((y_true_arr - y_mean) ** 2, axis=0)

    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    r2_per = np.full_like(rmsep_per, np.nan, dtype=float)
    valid = ss_tot_per > 0
    r2_per[valid] = 1.0 - (ss_res_per[valid] / ss_tot_per[valid])

    return {
        'RMSE': rmsep,
        'RMSEP': rmsep,
        'R2': r2,
        'RMSE_per_response': np.asarray(rmsep_per, dtype=float),
        'RMSEP_per_response': np.asarray(rmsep_per, dtype=float),
        'R2_per_response': np.asarray(r2_per, dtype=float),
        'n_samples': int(y_true_arr.shape[0]),
    }


def _resolve_response_labels(y_labels: Optional[Any], n_responses: int) -> List[str]:
    defaults = [f"Y{i + 1}" for i in range(int(n_responses))]
    if y_labels is None:
        return defaults

    try:
        raw = np.asarray(y_labels, dtype=object).reshape(-1).tolist()
    except Exception:
        return defaults

    labels: List[str] = []
    for idx in range(int(n_responses)):
        if idx < len(raw):
            text = str(raw[idx]).strip()
            labels.append(text if text else defaults[idx])
        else:
            labels.append(defaults[idx])
    return labels


def _response_header(response_index: int, response_label: Optional[str]) -> str:
    base = f"Y{int(response_index) + 1}"
    label_text = str(response_label).strip() if response_label is not None else ""
    if label_text and label_text.lower() != base.lower():
        return f"[{base} - {label_text}]"
    return f"[{base}]"


def _fmt_metric(value: Any, decimals: int = 6) -> str:
    try:
        val = float(value)
    except Exception:
        return "N/A"
    if not np.isfinite(val):
        return "N/A"
    return f"{val:.{int(decimals)}f}"


def _build_first_order_report(
    n_y: int,
    metrics: Dict[str, Any],
    model: Dict[str, Any],
    selected_parameter_name: Optional[str],
    selected_parameter_value: Optional[Any],
    optimization_results: Dict[str, Any],
    y_labels: Optional[Any] = None,
) -> str:
    labels = _resolve_response_labels(y_labels, n_y)
    lines: List[str] = []
    lines.append("First-Order Calibration Report")
    lines.append("==============================")
    lines.append(f"Method: {model.get('model_type', 'unknown')}")
    lines.append(f"Local mode: {model.get('local_method', 'N/A')}")
    lines.append(f"Model details: {(model.get('model_specific') or {}).get('summary', 'N/A')}")

    if selected_parameter_name:
        if isinstance(selected_parameter_value, (int, np.integer)):
            param_text = str(int(selected_parameter_value))
        elif isinstance(selected_parameter_value, (float, np.floating)):
            param_text = f"{float(selected_parameter_value):.4g}"
        else:
            param_text = "N/A"
        lines.append(f"Selected parameter: {selected_parameter_name} = {param_text}")
    else:
        lines.append("Selected parameter: N/A")

    lines.append("")
    lines.append("Per-Response Performance")
    lines.append("------------------------")

    cal = metrics.get('calibration') or {}
    cv = metrics.get('cv') or {}
    val = metrics.get('validation') or {}

    r2_cal = np.asarray(cal.get('R2_per_response', []), dtype=float)
    rmse_cal = np.asarray(cal.get('RMSEP_per_response', []), dtype=float)
    r2_cv = np.asarray(cv.get('R2_per_response', []), dtype=float) if cv else np.asarray([])
    rmse_cv = np.asarray(cv.get('RMSEP_per_response', []), dtype=float) if cv else np.asarray([])
    r2_val = np.asarray(val.get('R2_per_response', []), dtype=float) if val else np.asarray([])
    rmse_val = np.asarray(val.get('RMSEP_per_response', []), dtype=float) if val else np.asarray([])

    for yi in range(int(n_y)):
        lines.append(_response_header(yi, labels[yi]))
        lines.append(
            "Calibration      -> R2: "
            f"{_fmt_metric(r2_cal[yi] if yi < len(r2_cal) else np.nan)}, "
            f"RMSEC: {_fmt_metric(rmse_cal[yi] if yi < len(rmse_cal) else np.nan)}"
        )
        lines.append(
            "Cross-Validation -> R2: "
            f"{_fmt_metric(r2_cv[yi] if yi < len(r2_cv) else np.nan)}, "
            f"RMSECV: {_fmt_metric(rmse_cv[yi] if yi < len(rmse_cv) else np.nan)}"
        )
        lines.append(
            "Validation       -> R2: "
            f"{_fmt_metric(r2_val[yi] if yi < len(r2_val) else np.nan)}, "
            f"RMSEP: {_fmt_metric(rmse_val[yi] if yi < len(rmse_val) else np.nan)}"
        )
        lines.append("")

    diag = ((model.get('model_specific') or {}).get('diagnostics') or {})
    lines.append("Residual Summary (Calibration)")
    lines.append("------------------------------")
    lines.append(f"Mean: {_fmt_metric(diag.get('residual_mean', np.nan))}")
    lines.append(f"Std: {_fmt_metric(diag.get('residual_std', np.nan))}")
    lines.append(f"Max |residual|: {_fmt_metric(diag.get('residual_max_abs', np.nan))}")
    lines.append("")
    lines.append(f"Optimization enabled: {optimization_results.get('optimization_used', False)}")
    best_value = optimization_results.get('best_value')
    if isinstance(best_value, (int, np.integer)):
        best_text = str(int(best_value))
    elif isinstance(best_value, (float, np.floating)):
        best_text = f"{float(best_value):.4g}"
    else:
        best_text = "N/A"
    lines.append(f"Best optimized value: {best_text}")

    return "\n".join(lines)


def _fmt_metric_vector(values: Any, decimals: int = 6) -> str:
    if values is None:
        return "N/A"
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return "N/A"
    if arr.size == 0:
        return "N/A"
    if arr.size == 1:
        return _fmt_metric(arr[0], decimals)
    return "[" + ", ".join(_fmt_metric(v, decimals) for v in arr) + "]"


def _build_linear_model_summary_report(
    model_type_norm: str,
    linear_payload: Dict[str, Any],
    metrics_cal: Dict[str, Any],
    y_labels: Optional[Any] = None,
) -> str:
    titles = {
        'ols': 'OLS Model Summary',
        'ridge': 'Ridge Regression Summary',
        'lasso': 'Lasso Regression Summary',
        'elastic_net': 'Elastic Net Summary',
    }
    title = titles.get(model_type_norm, 'Linear Model Summary')

    r2_per = np.asarray(metrics_cal.get('R2_per_response', []), dtype=float)
    rmse_per = np.asarray(metrics_cal.get('RMSEP_per_response', []), dtype=float)
    n_y = int(max(len(r2_per), len(rmse_per), 1))
    labels = _resolve_response_labels(y_labels, n_y)

    lines: List[str] = []
    lines.append(title)
    lines.append("=" * len(title))

    if model_type_norm == 'ridge':
        lines.append(f"Regularization Alpha (lambda): {_fmt_metric(linear_payload.get('alpha'), 6)}")
    elif model_type_norm == 'lasso':
        lines.append(f"Selected Alpha (lambda): {_fmt_metric(linear_payload.get('alpha'), 6)}")
        lines.append(f"Non-zero Coefficients: {linear_payload.get('nonzero_coefficients', 'N/A')}")
    elif model_type_norm == 'elastic_net':
        lines.append(f"Alpha (lambda): {_fmt_metric(linear_payload.get('alpha'), 6)}")
        lines.append(f"L1 Ratio: {_fmt_metric(linear_payload.get('l1_ratio'), 6)}")
        lines.append(f"Non-zero Coefficients: {linear_payload.get('nonzero_coefficients', 'N/A')}")

    lines.append(f"Intercept: {_fmt_metric_vector(linear_payload.get('intercept'), 6)}")
    lines.append(f"Calibration samples: {metrics_cal.get('n_samples', 'N/A')}")
    lines.append(
        "Global calibration -> "
        f"R2: {_fmt_metric(metrics_cal.get('R2', np.nan), 6)}, "
        f"RMSEC: {_fmt_metric(metrics_cal.get('RMSEP', np.nan), 6)}"
    )
    lines.append("")
    lines.append("Per-Response (Calibration)")
    lines.append("--------------------------")
    for yi in range(n_y):
        r2_val = r2_per[yi] if yi < len(r2_per) else np.nan
        rmse_val = rmse_per[yi] if yi < len(rmse_per) else np.nan
        lines.append(
            f"{_response_header(yi, labels[yi])} R2: {_fmt_metric(r2_val, 6)}, RMSEC: {_fmt_metric(rmse_val, 6)}"
        )

    return "\n".join(lines)


def _to_serializable_vector(values: Any) -> Optional[List[float]]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return []
    return [float(v) for v in arr.tolist()]


def _to_serializable_matrix(values: Any) -> Optional[List[List[float]]]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        return None
    return [[float(v) for v in row] for row in arr.tolist()]


def _autoscale_from_calibration(
    X_cal: np.ndarray,
    X_val: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Autoscale data using calibration-derived mean/std statistics."""
    eps = 1e-10
    mean = np.mean(X_cal, axis=0, keepdims=True)
    std = np.std(X_cal, axis=0, keepdims=True)

    X_cal_scaled = (np.asarray(X_cal, dtype=float) - mean) / (std + eps)
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = (np.asarray(X_val, dtype=float) - mean) / (std + eps)

    return X_cal_scaled, X_val_scaled


def _coerce_optional_bool(value: Optional[Any]) -> Optional[bool]:
    """Normalize optional bool-like values from routing/UI payloads."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in ('true', '1', 'yes', 'y', 'on'):
            return True
        if norm in ('false', '0', 'no', 'n', 'off', ''):
            return False
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return None


def _resolve_translation_term(translation_keys: Optional[Dict[str, Any]], key: str, default: str) -> str:
    if not isinstance(translation_keys, dict):
        return default

    value = translation_keys.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return default


def _resolve_parameter_display_name(
    parameter_name: Optional[str],
    translation_keys: Optional[Dict[str, Any]],
    model_type: Optional[str] = None,
    local_method: Optional[str] = None,
) -> Optional[str]:
    if not parameter_name:
        return parameter_name

    model_type_norm = str(model_type).strip().lower() if isinstance(model_type, str) else ''
    local_method_norm = str(local_method).strip().lower() if isinstance(local_method, str) else ''

    # Priority 1: explicit model-specific mapping from config.
    parameter_display_names_by_model = (
        translation_keys.get('parameter_display_names_by_model')
        if isinstance(translation_keys, dict)
        else None
    )
    if isinstance(parameter_display_names_by_model, dict) and model_type_norm:
        model_map = parameter_display_names_by_model.get(model_type_norm)
        if isinstance(model_map, dict):
            model_value = model_map.get(parameter_name)
            if isinstance(model_value, str) and model_value.strip():
                return model_value

    parameter_display_names = translation_keys.get('parameter_display_names') if isinstance(translation_keys, dict) else None
    if isinstance(parameter_display_names, dict):
        # Priority 2: namespaced entries inside flat mapping.
        candidate_keys = []
        if model_type_norm:
            candidate_keys.extend([
                f'{model_type_norm}.{parameter_name}',
                f'{model_type_norm}:{parameter_name}',
            ])
        if model_type_norm == 'local' and local_method_norm:
            candidate_keys.extend([
                f'local.{local_method_norm}.{parameter_name}',
                f'local:{local_method_norm}:{parameter_name}',
            ])
        for candidate_key in candidate_keys:
            candidate_value = parameter_display_names.get(candidate_key)
            if isinstance(candidate_value, str) and candidate_value.strip():
                return candidate_value

        # Priority 3: plain parameter mapping.
        value = parameter_display_names.get(parameter_name)
        if isinstance(value, str) and value.strip():
            return value

    # Built-in defaults for shared parameters used by multiple model types.
    default_parameter_display_names_by_model = {
        'pls': {
            'n_components': 'Latent Variables',
        },
        'kernel_pls': {
            'n_components': 'Latent Variables',
        },
        'pcr': {
            'n_components': 'Principal Components',
        },
    }
    if model_type_norm:
        model_defaults = default_parameter_display_names_by_model.get(model_type_norm)
        if isinstance(model_defaults, dict):
            default_model_value = model_defaults.get(parameter_name)
            if isinstance(default_model_value, str) and default_model_value.strip():
                return default_model_value

    default_parameter_display_names = {
        'n_components': 'Latent Variables / PCs',
        'ridge_alpha': 'Ridge Alpha',
        'lasso_alpha': 'Lasso Alpha',
        'elastic_net_alpha': 'Elastic Net Alpha',
        'random_forest_n_estimators': 'Random Forest Trees',
        'svr_c': 'SVR C',
        'gradient_boosting_n_estimators': 'GB Trees',
        'hist_gradient_boosting_n_estimators': 'HistGB Iterations',
        'gaussian_process_alpha': 'GPR Alpha',
        'n_neighbors': 'Number of Neighbors',
        'idw_power': 'IDW Power',
        'adaptive_alpha': 'Adaptive Alpha',
        'radius_threshold': 'Radius Threshold',
        'kernel_bandwidth': 'Kernel Bandwidth',
    }
    return _resolve_translation_term(
        default_parameter_display_names,
        parameter_name,
        parameter_name.replace('_', ' ').title(),
    )


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
    selected_range_input: Optional[Any],
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

    parsed = parse_numeric_spec(selected_range_input)

    if model_type in ('pls', 'pcr', 'kernel_pls'):
        return _coerce_int_candidates(parsed, default_max=min(max_latent, 15), from_one_on_single=True)
    if model_type in ('ridge', 'lasso', 'elastic_net'):
        return _coerce_float_candidates(parsed, default_values=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    if model_type == 'svr':
        return _coerce_float_candidates(parsed, default_values=[0.1, 1.0, 10.0, 100.0])
    if model_type == 'gaussian_process':
        return _coerce_float_candidates(parsed, default_values=[1e-8, 1e-6, 1e-4, 1e-2, 1e-1])
    if model_type == 'random_forest':
        return _coerce_int_candidates(parsed, default_max=300, from_one_on_single=False)
    if model_type in ('gradient_boosting', 'hist_gradient_boosting'):
        return _coerce_int_candidates(parsed, default_max=400, from_one_on_single=False)
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


def _resolve_optimization_target(
    model_type_norm: str,
    local_method_norm: str,
    n_components: int,
    ridge_alpha: float,
    lasso_alpha: float,
    elastic_net_alpha: float,
    random_forest_n_estimators: int,
    svr_c: float,
    gradient_boosting_n_estimators: int,
    hist_gradient_boosting_n_estimators: int,
    gaussian_process_alpha: float,
    n_neighbors: int,
    idw_power: float,
    adaptive_alpha: float,
    radius_threshold: float,
    kernel_bandwidth: float,
    n_components_range: Optional[Any],
    ridge_alpha_range: Optional[Any],
    lasso_alpha_range: Optional[Any],
    elastic_net_alpha_range: Optional[Any],
    random_forest_n_estimators_range: Optional[Any],
    svr_c_range: Optional[Any],
    gradient_boosting_n_estimators_range: Optional[Any],
    hist_gradient_boosting_n_estimators_range: Optional[Any],
    gaussian_process_alpha_range: Optional[Any],
    n_neighbors_range: Optional[Any],
    idw_power_range: Optional[Any],
    adaptive_alpha_range: Optional[Any],
    radius_threshold_range: Optional[Any],
    kernel_bandwidth_range: Optional[Any],
) -> Tuple[Optional[str], Optional[Any], Optional[Any]]:
    parameter_name: Optional[str] = None
    fixed_value: Optional[Any] = None

    if model_type_norm in ('pls', 'pcr', 'kernel_pls'):
        parameter_name = 'n_components'
        fixed_value = int(n_components)
    elif model_type_norm == 'ridge':
        parameter_name = 'ridge_alpha'
        fixed_value = float(ridge_alpha)
    elif model_type_norm == 'lasso':
        parameter_name = 'lasso_alpha'
        fixed_value = float(lasso_alpha)
    elif model_type_norm == 'elastic_net':
        parameter_name = 'elastic_net_alpha'
        fixed_value = float(elastic_net_alpha)
    elif model_type_norm == 'random_forest':
        parameter_name = 'random_forest_n_estimators'
        fixed_value = int(random_forest_n_estimators)
    elif model_type_norm == 'svr':
        parameter_name = 'svr_c'
        fixed_value = float(svr_c)
    elif model_type_norm == 'gradient_boosting':
        parameter_name = 'gradient_boosting_n_estimators'
        fixed_value = int(gradient_boosting_n_estimators)
    elif model_type_norm == 'hist_gradient_boosting':
        parameter_name = 'hist_gradient_boosting_n_estimators'
        fixed_value = int(hist_gradient_boosting_n_estimators)
    elif model_type_norm == 'gaussian_process':
        parameter_name = 'gaussian_process_alpha'
        fixed_value = float(gaussian_process_alpha)
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

    selected_range_input: Optional[Any] = None
    if model_type_norm in ('pls', 'pcr', 'kernel_pls') and n_components_range not in (None, ''):
        selected_range_input = n_components_range
    elif model_type_norm == 'ridge' and ridge_alpha_range not in (None, ''):
        selected_range_input = ridge_alpha_range
    elif model_type_norm == 'lasso' and lasso_alpha_range not in (None, ''):
        selected_range_input = lasso_alpha_range
    elif model_type_norm == 'elastic_net' and elastic_net_alpha_range not in (None, ''):
        selected_range_input = elastic_net_alpha_range
    elif model_type_norm == 'random_forest' and random_forest_n_estimators_range not in (None, ''):
        selected_range_input = random_forest_n_estimators_range
    elif model_type_norm == 'svr' and svr_c_range not in (None, ''):
        selected_range_input = svr_c_range
    elif model_type_norm == 'gradient_boosting' and gradient_boosting_n_estimators_range not in (None, ''):
        selected_range_input = gradient_boosting_n_estimators_range
    elif model_type_norm == 'hist_gradient_boosting' and hist_gradient_boosting_n_estimators_range not in (None, ''):
        selected_range_input = hist_gradient_boosting_n_estimators_range
    elif model_type_norm == 'gaussian_process' and gaussian_process_alpha_range not in (None, ''):
        selected_range_input = gaussian_process_alpha_range
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

    return parameter_name, fixed_value, selected_range_input


def _candidate_hyperparameters(
    candidate: Any,
    model_type_norm: str,
    local_method_norm: str,
    n_components: int,
    ridge_alpha: float,
    lasso_alpha: float,
    elastic_net_alpha: float,
    elastic_net_l1_ratio: float,
    random_forest_n_estimators: int,
    random_forest_max_depth: int,
    random_forest_min_samples_leaf: int,
    random_forest_max_features: Any,
    random_forest_bootstrap: bool,
    random_forest_oob_score: bool,
    svr_c: float,
    svr_epsilon: float,
    svr_gamma: float,
    svr_kernel: str,
    svr_degree: int,
    svr_coef0: float,
    gradient_boosting_n_estimators: int,
    gradient_boosting_learning_rate: float,
    gradient_boosting_subsample: float,
    gradient_boosting_max_features: Any,
    gradient_boosting_max_depth: int,
    gradient_boosting_min_samples_leaf: int,
    hist_gradient_boosting_n_estimators: int,
    hist_gradient_boosting_learning_rate: float,
    hist_gradient_boosting_max_depth: int,
    hist_gradient_boosting_min_samples_leaf: int,
    hist_gradient_boosting_l2_regularization: float,
    hist_gradient_boosting_max_bins: int,
    hist_gradient_boosting_max_leaf_nodes: int,
    gaussian_process_alpha: float,
    gaussian_process_length_scale: float,
    gaussian_process_constant_value: float,
    gaussian_process_n_restarts_optimizer: int,
    gaussian_process_normalize_y: bool,
    kernel_pls_gamma: float,
    kernel_pls_n_features: int,
    n_neighbors: int,
    idw_power: float,
    adaptive_alpha: float,
    radius_threshold: float,
    kernel_bandwidth: float,
) -> Dict[str, Any]:
    return {
        'n_components': int(candidate) if model_type_norm in ('pls', 'pcr', 'kernel_pls') else int(n_components),
        'ridge_alpha': float(candidate) if model_type_norm == 'ridge' else float(ridge_alpha),
        'lasso_alpha': float(candidate) if model_type_norm == 'lasso' else float(lasso_alpha),
        'elastic_net_alpha': float(candidate) if model_type_norm == 'elastic_net' else float(elastic_net_alpha),
        'elastic_net_l1_ratio': float(elastic_net_l1_ratio),
        'random_forest_n_estimators': int(candidate) if model_type_norm == 'random_forest' else int(random_forest_n_estimators),
        'random_forest_max_depth': int(random_forest_max_depth),
        'random_forest_min_samples_leaf': int(random_forest_min_samples_leaf),
        'random_forest_max_features': random_forest_max_features,
        'random_forest_bootstrap': bool(random_forest_bootstrap),
        'random_forest_oob_score': bool(random_forest_oob_score),
        'svr_c': float(candidate) if model_type_norm == 'svr' else float(svr_c),
        'svr_epsilon': float(svr_epsilon),
        'svr_gamma': float(svr_gamma),
        'svr_kernel': str(svr_kernel),
        'svr_degree': int(svr_degree),
        'svr_coef0': float(svr_coef0),
        'gradient_boosting_n_estimators': int(candidate) if model_type_norm == 'gradient_boosting' else int(gradient_boosting_n_estimators),
        'gradient_boosting_learning_rate': float(gradient_boosting_learning_rate),
        'gradient_boosting_subsample': float(gradient_boosting_subsample),
        'gradient_boosting_max_features': gradient_boosting_max_features,
        'gradient_boosting_max_depth': int(gradient_boosting_max_depth),
        'gradient_boosting_min_samples_leaf': int(gradient_boosting_min_samples_leaf),
        'hist_gradient_boosting_n_estimators': int(candidate) if model_type_norm == 'hist_gradient_boosting' else int(hist_gradient_boosting_n_estimators),
        'hist_gradient_boosting_learning_rate': float(hist_gradient_boosting_learning_rate),
        'hist_gradient_boosting_max_depth': int(hist_gradient_boosting_max_depth),
        'hist_gradient_boosting_min_samples_leaf': int(hist_gradient_boosting_min_samples_leaf),
        'hist_gradient_boosting_l2_regularization': float(hist_gradient_boosting_l2_regularization),
        'hist_gradient_boosting_max_bins': int(hist_gradient_boosting_max_bins),
        'hist_gradient_boosting_max_leaf_nodes': int(hist_gradient_boosting_max_leaf_nodes),
        'gaussian_process_alpha': float(candidate) if model_type_norm == 'gaussian_process' else float(gaussian_process_alpha),
        'gaussian_process_length_scale': float(gaussian_process_length_scale),
        'gaussian_process_constant_value': float(gaussian_process_constant_value),
        'gaussian_process_n_restarts_optimizer': int(gaussian_process_n_restarts_optimizer),
        'gaussian_process_normalize_y': bool(gaussian_process_normalize_y),
        'kernel_pls_gamma': float(kernel_pls_gamma),
        'kernel_pls_n_features': int(kernel_pls_n_features),
        'n_neighbors': int(candidate) if (model_type_norm == 'local' and local_method_norm == 'knn') else int(n_neighbors),
        'idw_power': float(candidate) if (model_type_norm == 'local' and local_method_norm == 'idw') else float(idw_power),
        'adaptive_alpha': float(candidate) if (model_type_norm == 'local' and local_method_norm == 'adaptive') else float(adaptive_alpha),
        'radius_threshold': float(candidate) if (model_type_norm == 'local' and local_method_norm == 'radius') else float(radius_threshold),
        'kernel_bandwidth': float(candidate) if (model_type_norm == 'local' and local_method_norm == 'kernel') else float(kernel_bandwidth),
    }


def _selected_parameter_value(
    parameter_name: Optional[str],
    params: Dict[str, Any],
) -> Optional[Any]:
    if parameter_name is None:
        return None
    if parameter_name in (
        'n_components',
        'n_neighbors',
        'random_forest_n_estimators',
        'gradient_boosting_n_estimators',
        'hist_gradient_boosting_n_estimators',
    ):
        return int(params[parameter_name])
    return float(params[parameter_name])


def _build_model_specific_payload(
    model_type: str,
    fitted_model: Dict[str, Any],
    params: Dict[str, Any],
    local_method: str,
    local_distance: str,
    Y_train: Optional[np.ndarray] = None,
    X_train: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    model_type_norm = str(model_type).strip().lower()

    if model_type_norm == 'random_forest':
        model = fitted_model.get('model')
        importances = _to_serializable_vector(getattr(model, 'feature_importances_', None))
        n_trees = int(params.get('random_forest_n_estimators', fitted_model.get('n_estimators', 0)))
        oob_score_value = None
        if fitted_model.get('oob_score', False):
            raw_oob = getattr(model, 'oob_score_', None)
            if raw_oob is not None:
                oob_score_value = float(raw_oob)
        return {
            'summary': f"Random Forest with {n_trees} trees",
            'random_forest': {
                'n_estimators': n_trees,
                'max_depth': fitted_model.get('max_depth'),
                'min_samples_leaf': fitted_model.get('min_samples_leaf'),
                'max_features': fitted_model.get('max_features'),
                'bootstrap': fitted_model.get('bootstrap'),
                'oob_score': fitted_model.get('oob_score'),
                'oob_score_value': oob_score_value,
                'feature_importances': importances,
            },
        }

    if model_type_norm in ('ols', 'ridge', 'lasso', 'elastic_net'):
        model = fitted_model.get('model')
        coef = getattr(model, 'coef_', None)
        intercept = getattr(model, 'intercept_', None)
        nz = None
        if coef is not None:
            nz = int(np.count_nonzero(np.asarray(coef)))
        return {
            'summary': f"Linear coefficients available ({'sparse' if model_type_norm == 'lasso' else 'dense'})",
            'linear': {
                'coefficients': _to_serializable_vector(coef),
                'intercept': _to_serializable_vector(intercept),
                'nonzero_coefficients': nz,
                'alpha': float(fitted_model.get('alpha')) if 'alpha' in fitted_model else None,
                'l1_ratio': float(fitted_model.get('l1_ratio')) if 'l1_ratio' in fitted_model else None,
            },
        }

    if model_type_norm in ('svr', 'gradient_boosting', 'hist_gradient_boosting', 'gaussian_process'):
        model_obj = fitted_model.get('model')

        # Feature importances for gradient-boosted tree models
        feature_importances = None
        if model_type_norm in ('gradient_boosting', 'hist_gradient_boosting'):
            if hasattr(model_obj, 'feature_importances_'):
                feature_importances = _to_serializable_vector(model_obj.feature_importances_)
            elif hasattr(model_obj, 'estimators_'):
                imp_list = [
                    e.feature_importances_ for e in model_obj.estimators_
                    if hasattr(e, 'feature_importances_')
                ]
                if imp_list:
                    feature_importances = _to_serializable_vector(np.mean(imp_list, axis=0))

        # Fitted kernel parameters for Gaussian Process
        fitted_gpr_length_scale: Optional[float] = None
        fitted_gpr_constant_value: Optional[float] = None
        if model_type_norm == 'gaussian_process':
            gpr = model_obj
            if hasattr(model_obj, 'estimators_') and model_obj.estimators_:
                gpr = model_obj.estimators_[0]
            if hasattr(gpr, 'kernel_'):
                try:
                    for kname, kval in gpr.kernel_.get_params(deep=True).items():
                        if isinstance(kval, (float, int, np.floating, np.integer)):
                            if 'length_scale' in kname and fitted_gpr_length_scale is None:
                                fitted_gpr_length_scale = float(kval)
                            if 'constant_value' in kname and fitted_gpr_constant_value is None:
                                fitted_gpr_constant_value = float(kval)
                except Exception:
                    pass

        payload = {
            'summary': f"{model_type_norm.replace('_', ' ').title()} model diagnostics",
            model_type_norm: {
                'kernel': fitted_model.get('kernel'),
                'C': fitted_model.get('C'),
                'epsilon': fitted_model.get('epsilon'),
                'gamma': fitted_model.get('gamma'),
                'n_estimators': fitted_model.get('n_estimators'),
                'learning_rate': fitted_model.get('learning_rate'),
                'subsample': fitted_model.get('subsample'),
                'max_features': fitted_model.get('max_features'),
                'max_depth': fitted_model.get('max_depth'),
                'max_leaf_nodes': fitted_model.get('max_leaf_nodes'),
                'min_samples_leaf': fitted_model.get('min_samples_leaf'),
                'l2_regularization': fitted_model.get('l2_regularization'),
                'max_bins': fitted_model.get('max_bins'),
                'alpha': fitted_model.get('alpha'),
                'length_scale': fitted_model.get('length_scale'),
                'constant_value': fitted_model.get('constant_value'),
                'n_restarts_optimizer': fitted_model.get('n_restarts_optimizer'),
                'normalize_y': fitted_model.get('normalize_y'),
                'feature_importances': feature_importances,
                'fitted_length_scale': fitted_gpr_length_scale,
                'fitted_constant_value': fitted_gpr_constant_value,
            },
        }
        return payload

    if model_type_norm in ('pls', 'kernel_pls'):
        model = fitted_model.get('model')
        x_weights = getattr(model, 'x_weights_', None)
        x_loadings = getattr(model, 'x_loadings_', None)
        y_loadings = getattr(model, 'y_loadings_', None)
        x_scores = getattr(model, 'x_scores_', None)
        y_scores = getattr(model, 'y_scores_', None)
        weight_norms = None
        loading_norms = None
        y_residual_variance_by_lv = None
        n_components_out = int(fitted_model.get('n_components', params.get('n_components', 0)))

        if x_weights is not None:
            xw = np.asarray(x_weights, dtype=float)
            if xw.ndim == 2:
                weight_norms = [float(v) for v in np.linalg.norm(xw, axis=0).tolist()]
        if x_loadings is not None:
            xl = np.asarray(x_loadings, dtype=float)
            if xl.ndim == 2:
                loading_norms = [float(v) for v in np.linalg.norm(xl, axis=0).tolist()]

        if Y_train is not None and x_scores is not None and y_loadings is not None:
            y_arr = np.asarray(Y_train, dtype=float)
            t_arr = np.asarray(x_scores, dtype=float)
            q_arr = np.asarray(y_loadings, dtype=float)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            if t_arr.ndim == 2 and q_arr.ndim == 2 and t_arr.shape[0] == y_arr.shape[0]:
                n_comp = min(t_arr.shape[1], q_arr.shape[1])
                if n_comp > 0:
                    y_centered = y_arr - np.mean(y_arr, axis=0, keepdims=True)
                    y_residual_variance_by_lv = []
                    for k in range(1, n_comp + 1):
                        y_hat_centered = t_arr[:, :k] @ q_arr[:, :k].T
                        residual = y_centered - y_hat_centered
                        y_residual_variance_by_lv.append(float(np.mean(np.var(residual, axis=0))))

        # Outlier statistics (leverage, T2 Hotelling, Q-residuals in X space)
        pls_leverage: Optional[list] = None
        pls_t2: Optional[list] = None
        pls_q_residuals: Optional[list] = None
        if x_scores is not None and X_train is not None:
            try:
                T = np.asarray(x_scores, dtype=float)
                n_s, n_c = T.shape
                if n_s > 1 and n_c > 0:
                    TtT_inv = np.linalg.pinv(T.T @ T)
                    pls_leverage = np.sum((T @ TtT_inv) * T, axis=1).tolist()
                    ev = np.sum(T ** 2, axis=0) / (n_s - 1)
                    safe_ev = np.where(ev > 0, ev, np.nan)
                    pls_t2 = np.nansum(T ** 2 / safe_ev, axis=1).tolist()
                    if x_loadings is not None:
                        P = np.asarray(x_loadings, dtype=float)
                        x_mean = getattr(model, 'x_mean_', None)
                        x_std = getattr(model, 'x_std_', None)
                        X_arr = np.asarray(X_train, dtype=float)
                        if x_mean is not None:
                            X_sc = (X_arr - x_mean) / (np.where(x_std != 0, x_std, 1.0) if x_std is not None else 1.0)
                        else:
                            X_sc = X_arr
                        pls_q_residuals = np.sum((X_sc - T @ P.T) ** 2, axis=1).tolist()
            except Exception:
                pass

        lv_labels = [f"LV {i + 1}" for i in range(n_components_out)]

        return {
            'summary': f"{('Kernel PLS' if model_type_norm == 'kernel_pls' else 'PLS')} with {n_components_out} components",
            'pls': {
                'n_components': n_components_out,
                'pls_scale': bool(fitted_model.get('pls_scale', False)),
                'lv_labels': lv_labels,
                'x_weights_norm': weight_norms,
                'x_loadings_norm': loading_norms,
                'x_weights': _to_serializable_matrix(x_weights),
                'x_loadings': _to_serializable_matrix(x_loadings),
                'y_loadings': _to_serializable_matrix(y_loadings),
                'x_scores': _to_serializable_matrix(x_scores),
                'y_scores': _to_serializable_matrix(y_scores),
                'y_residual_variance_by_lv': y_residual_variance_by_lv,
                'leverage': pls_leverage,
                't2_hotelling': pls_t2,
                'q_residuals': pls_q_residuals,
                'kernel_gamma': float(fitted_model.get('gamma')) if model_type_norm == 'kernel_pls' else None,
                'mapped_features': int(fitted_model.get('mapped_features')) if model_type_norm == 'kernel_pls' else None,
            },
        }

    if model_type_norm == 'pcr':
        pca = fitted_model.get('pca')
        evr = _to_serializable_vector(getattr(pca, 'explained_variance_ratio_', None))
        cumsum = None
        if evr is not None:
            cumsum = [float(v) for v in np.cumsum(np.asarray(evr, dtype=float)).tolist()]
        pca_scores = None
        pca_loadings = None
        pca_scores_arr = None
        if pca is not None and X_train is not None:
            try:
                pca_scores_arr = pca.transform(np.asarray(X_train, dtype=float))
                pca_scores = _to_serializable_matrix(pca_scores_arr)
            except Exception:
                pass
        if pca is not None:
            components = getattr(pca, 'components_', None)
            if components is not None:
                # Transpose to (n_features, n_components) to match PLS loading convention
                pca_loadings = _to_serializable_matrix(np.asarray(components, dtype=float).T)

        # Outlier statistics from PCA decomposition
        pcr_leverage: Optional[list] = None
        pcr_t2: Optional[list] = None
        pcr_q_residuals: Optional[list] = None
        if pca_scores_arr is not None:
            try:
                T = pca_scores_arr
                n_s, n_c = T.shape
                if n_s > 1 and n_c > 0:
                    TtT_inv = np.linalg.pinv(T.T @ T)
                    pcr_leverage = np.sum((T @ TtT_inv) * T, axis=1).tolist()
                    ev = np.sum(T ** 2, axis=0) / (n_s - 1)
                    safe_ev = np.where(ev > 0, ev, np.nan)
                    pcr_t2 = np.nansum(T ** 2 / safe_ev, axis=1).tolist()
                    comps = getattr(pca, 'components_', None)  # (n_comp, n_features)
                    if comps is not None:
                        pca_mean = getattr(pca, 'mean_', None)
                        X_arr = np.asarray(X_train, dtype=float)
                        X_c = (X_arr - pca_mean) if pca_mean is not None else X_arr
                        pcr_q_residuals = np.sum((X_c - T @ comps) ** 2, axis=1).tolist()
            except Exception:
                pass

        pc_labels_list = []
        if evr is not None:
            for idx, pct in enumerate(evr):
                pct_f = float(pct) * 100
                pct_text = f"{pct_f:.1f}".rstrip('0').rstrip('.')
                pc_labels_list.append(f"PC {idx + 1} ({pct_text}% Variance)")
        if not pc_labels_list:
            n_comps_pcr = int(params.get('n_components', fitted_model.get('n_components', 0)))
            pc_labels_list = [f"PC {i + 1}" for i in range(n_comps_pcr)]

        return {
            'summary': f"PCR with {int(params.get('n_components', fitted_model.get('n_components', 0)))} components",
            'pcr': {
                'n_components': int(params.get('n_components', fitted_model.get('n_components', 0))),
                'explained_variance_ratio': evr,
                'cumulative_explained_variance_ratio': cumsum,
                'pc_labels': pc_labels_list,
                'pca_scores': pca_scores,
                'pca_loadings': pca_loadings,
                'leverage': pcr_leverage,
                't2_hotelling': pcr_t2,
                'q_residuals': pcr_q_residuals,
            },
        }

    if model_type_norm == 'local':
        return {
            'summary': f"Local regression ({local_method}, {local_distance})",
            'local': {
                'method': local_method,
                'distance': local_distance,
                'minkowski_p': float(params.get('local_minkowski_p', 3.0)),
                'n_neighbors': int(params.get('n_neighbors', 0)),
                'idw_power': float(params.get('idw_power', 1.0)),
                'adaptive_alpha': float(params.get('adaptive_alpha', 1.0)),
                'radius_threshold': float(params.get('radius_threshold', 1.0)),
                'kernel_bandwidth': float(params.get('kernel_bandwidth', 1.0)),
            },
        }

    return {'summary': f"Model details for '{model_type_norm}' are not available."}


def _compute_query_distances(
    X_train: np.ndarray,
    x_query: np.ndarray,
    distance_metric: str,
    local_minkowski_p: float,
    mahalanobis_vi: Optional[np.ndarray] = None,
) -> np.ndarray:
    metric = str(distance_metric).strip().lower()
    query = np.asarray(x_query, dtype=float).reshape(1, -1)

    if metric == 'euclidean':
        return np.linalg.norm(X_train - query, axis=1)
    if metric == 'manhattan':
        return cdist(X_train, query, metric='cityblock').reshape(-1)
    if metric == 'minkowski':
        return cdist(X_train, query, metric='minkowski', p=float(local_minkowski_p)).reshape(-1)
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
    local_minkowski_p: float,
) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    n_neighbors = max(1, min(int(n_neighbors), X_train.shape[0]))
    idw_power = max(float(idw_power), 1e-12)
    adaptive_alpha = max(float(adaptive_alpha), 1e-12)
    radius_threshold = max(float(radius_threshold), 1e-12)
    kernel_bandwidth = max(float(kernel_bandwidth), 1e-12)
    local_minkowski_p = float(local_minkowski_p)
    if not np.isfinite(local_minkowski_p) or local_minkowski_p <= 0.0:
        raise ValueError('local_minkowski_p must be a positive number.')
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
            local_minkowski_p=local_minkowski_p,
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
    local_minkowski_p: float,
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
            local_minkowski_p=local_minkowski_p,
        )
        outputs[idx] = y_pred[0]

    return outputs


def _fit_model(
    model_type: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    n_components: int,
    pls_scale: bool,
    ridge_alpha: float,
    lasso_alpha: float,
    elastic_net_alpha: float,
    elastic_net_l1_ratio: float,
    linear_max_iter: int,
    random_forest_n_estimators: int,
    random_forest_max_depth: int,
    random_forest_min_samples_leaf: int,
    random_forest_max_features: Any,
    random_forest_bootstrap: bool,
    random_forest_oob_score: bool,
    svr_c: float,
    svr_epsilon: float,
    svr_gamma: float,
    svr_kernel: str,
    svr_degree: int,
    svr_coef0: float,
    gradient_boosting_n_estimators: int,
    gradient_boosting_learning_rate: float,
    gradient_boosting_subsample: float,
    gradient_boosting_max_features: Any,
    gradient_boosting_max_depth: int,
    gradient_boosting_min_samples_leaf: int,
    hist_gradient_boosting_n_estimators: int,
    hist_gradient_boosting_learning_rate: float,
    hist_gradient_boosting_max_depth: int,
    hist_gradient_boosting_min_samples_leaf: int,
    hist_gradient_boosting_l2_regularization: float,
    hist_gradient_boosting_max_bins: int,
    hist_gradient_boosting_max_leaf_nodes: int,
    gaussian_process_alpha: float,
    gaussian_process_length_scale: float,
    gaussian_process_constant_value: float,
    gaussian_process_n_restarts_optimizer: int,
    gaussian_process_normalize_y: bool,
    kernel_pls_gamma: float,
    kernel_pls_n_features: int,
    n_neighbors: int,
    local_method: str,
    idw_power: float,
    adaptive_alpha: float,
    radius_threshold: float,
    kernel_bandwidth: float,
    local_distance: str,
    local_minkowski_p: float,
) -> Dict[str, Any]:
    model_type = str(model_type).lower()
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)

    if model_type == 'ols':
        n_samples, n_features = X_train.shape
        if int(n_samples) <= int(n_features):
            emit_execution_error(
                code='first_order_calibration_ols_underdetermined',
                details={
                    'function': '_fit_model',
                    'model_type': model_type,
                    'n_samples': int(n_samples),
                    'n_features': int(n_features),
                },
            )
            raise ValueError(
                "OLS requires more calibration samples than predictor variables "
                "(n_samples > n_features)."
            )

        model = LinearRegression()
        try:
            model.fit(X_train, Y_train)
        except Exception as exc:
            emit_execution_error(
                code='first_order_calibration_ols_fit_failed',
                text=str(exc),
                details={
                    'function': '_fit_model',
                    'model_type': model_type,
                    'n_samples': int(n_samples),
                    'n_features': int(n_features),
                },
            )
            raise ValueError(
                "OLS fitting failed. Consider using PLS/PCR/Ridge or reducing variable count."
            ) from exc
        return {'model_type': model_type, 'model': model}

    if model_type == 'ridge':
        model = Ridge(alpha=float(ridge_alpha), max_iter=max(1000, int(linear_max_iter)))
        model.fit(X_train, Y_train)
        return {'model_type': model_type, 'model': model, 'alpha': float(ridge_alpha)}

    if model_type == 'lasso':
        if Y_train.shape[1] > 1:
            model = MultiTaskLasso(alpha=float(lasso_alpha), max_iter=max(1000, int(linear_max_iter)))
            model.fit(X_train, Y_train)
        else:
            model = Lasso(alpha=float(lasso_alpha), max_iter=max(1000, int(linear_max_iter)))
            model.fit(X_train, np.asarray(Y_train).reshape(-1))
        return {'model_type': model_type, 'model': model, 'alpha': float(lasso_alpha)}

    if model_type == 'elastic_net':
        l1_ratio = min(1.0, max(0.0, float(elastic_net_l1_ratio)))
        if Y_train.shape[1] > 1:
            model = MultiTaskElasticNet(alpha=float(elastic_net_alpha), l1_ratio=l1_ratio, max_iter=max(1000, int(linear_max_iter)))
            model.fit(X_train, Y_train)
        else:
            model = ElasticNet(alpha=float(elastic_net_alpha), l1_ratio=l1_ratio, max_iter=max(1000, int(linear_max_iter)))
            model.fit(X_train, np.asarray(Y_train).reshape(-1))
        return {
            'model_type': model_type,
            'model': model,
            'alpha': float(elastic_net_alpha),
            'l1_ratio': l1_ratio,
        }

    if model_type == 'random_forest':
        rf_max_depth = int(random_forest_max_depth)
        rf_max_depth = None if rf_max_depth <= 0 else max(1, rf_max_depth)
        rf_min_samples_leaf = max(1, int(random_forest_min_samples_leaf))

        rf_max_features_value: Any = random_forest_max_features
        if isinstance(rf_max_features_value, str):
            norm = rf_max_features_value.strip().lower()
            if norm in ('', 'none', 'null'):
                rf_max_features_value = None
            elif norm in ('sqrt', 'log2'):
                rf_max_features_value = norm
            else:
                try:
                    numeric = float(norm)
                    if numeric > 0:
                        if float(numeric).is_integer() and numeric >= 1:
                            rf_max_features_value = int(numeric)
                        else:
                            rf_max_features_value = numeric
                    else:
                        rf_max_features_value = 1.0
                except Exception:
                    rf_max_features_value = 1.0

        rf_bootstrap = bool(random_forest_bootstrap)
        rf_oob_score = bool(random_forest_oob_score) and rf_bootstrap

        model = RandomForestRegressor(
            n_estimators=max(10, int(random_forest_n_estimators)),
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            max_features=rf_max_features_value,
            bootstrap=rf_bootstrap,
            oob_score=rf_oob_score,
            random_state=42,
            n_jobs=-1,
        )
        if Y_train.shape[1] == 1:
            model.fit(X_train, np.asarray(Y_train).reshape(-1))
        else:
            model.fit(X_train, Y_train)
        return {
            'model_type': model_type,
            'model': model,
            'n_estimators': int(max(10, int(random_forest_n_estimators))),
            'max_depth': rf_max_depth,
            'min_samples_leaf': int(rf_min_samples_leaf),
            'max_features': rf_max_features_value,
            'bootstrap': rf_bootstrap,
            'oob_score': rf_oob_score,
        }

    if model_type == 'svr':
        kernel = str(svr_kernel).strip().lower()
        if kernel not in ('rbf', 'linear', 'poly', 'sigmoid'):
            raise ValueError("svr_kernel must be one of: 'rbf', 'linear', 'poly', 'sigmoid'.")
        gamma_value: Any = float(svr_gamma)
        if gamma_value <= 0:
            gamma_value = 'scale'
        base = SVR(
            kernel=kernel,
            C=float(max(1e-12, svr_c)),
            epsilon=float(max(1e-12, svr_epsilon)),
            gamma=gamma_value,
            degree=max(1, int(svr_degree)),
            coef0=float(svr_coef0),
        )
        if Y_train.shape[1] > 1:
            model = MultiOutputRegressor(base)
            model.fit(X_train, Y_train)
        else:
            model = base
            model.fit(X_train, np.asarray(Y_train).reshape(-1))
        return {
            'model_type': model_type,
            'model': model,
            'kernel': kernel,
            'C': float(max(1e-12, svr_c)),
            'epsilon': float(max(1e-12, svr_epsilon)),
            'gamma': gamma_value,
            'degree': int(max(1, svr_degree)),
            'coef0': float(svr_coef0),
        }

    if model_type == 'gradient_boosting':
        gb_subsample = min(1.0, max(1e-12, float(gradient_boosting_subsample)))
        gb_max_features_value: Any = gradient_boosting_max_features
        if isinstance(gb_max_features_value, str):
            norm = gb_max_features_value.strip().lower()
            if norm in ('', 'none', 'null'):
                gb_max_features_value = None
            elif norm in ('sqrt', 'log2'):
                gb_max_features_value = norm
            else:
                try:
                    numeric = float(norm)
                    if numeric > 0:
                        if float(numeric).is_integer() and numeric >= 1:
                            gb_max_features_value = int(numeric)
                        else:
                            gb_max_features_value = numeric
                    else:
                        gb_max_features_value = 1.0
                except Exception:
                    gb_max_features_value = 1.0

        base = GradientBoostingRegressor(
            n_estimators=max(10, int(gradient_boosting_n_estimators)),
            learning_rate=float(max(1e-6, gradient_boosting_learning_rate)),
            subsample=gb_subsample,
            max_features=gb_max_features_value,
            max_depth=max(1, int(gradient_boosting_max_depth)),
            min_samples_leaf=max(1, int(gradient_boosting_min_samples_leaf)),
            random_state=42,
        )
        if Y_train.shape[1] > 1:
            model = MultiOutputRegressor(base)
            model.fit(X_train, Y_train)
        else:
            model = base
            model.fit(X_train, np.asarray(Y_train).reshape(-1))
        return {
            'model_type': model_type,
            'model': model,
            'n_estimators': int(max(10, int(gradient_boosting_n_estimators))),
            'learning_rate': float(max(1e-6, gradient_boosting_learning_rate)),
            'subsample': float(gb_subsample),
            'max_features': gb_max_features_value,
            'max_depth': int(max(1, gradient_boosting_max_depth)),
            'min_samples_leaf': int(max(1, gradient_boosting_min_samples_leaf)),
        }

    if model_type == 'hist_gradient_boosting':
        max_depth = int(hist_gradient_boosting_max_depth)
        max_leaf_nodes = int(hist_gradient_boosting_max_leaf_nodes)
        max_leaf_nodes = None if max_leaf_nodes <= 0 else max(2, max_leaf_nodes)
        hgb_l2_regularization = max(0.0, float(hist_gradient_boosting_l2_regularization))
        hgb_max_bins = min(255, max(2, int(hist_gradient_boosting_max_bins)))
        base = HistGradientBoostingRegressor(
            max_iter=max(10, int(hist_gradient_boosting_n_estimators)),
            learning_rate=float(max(1e-6, hist_gradient_boosting_learning_rate)),
            max_depth=max_depth if max_depth > 0 else None,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=max(1, int(hist_gradient_boosting_min_samples_leaf)),
            l2_regularization=hgb_l2_regularization,
            max_bins=hgb_max_bins,
            random_state=42,
        )
        if Y_train.shape[1] > 1:
            model = MultiOutputRegressor(base)
            model.fit(X_train, Y_train)
        else:
            model = base
            model.fit(X_train, np.asarray(Y_train).reshape(-1))
        return {
            'model_type': model_type,
            'model': model,
            'n_estimators': int(max(10, int(hist_gradient_boosting_n_estimators))),
            'learning_rate': float(max(1e-6, hist_gradient_boosting_learning_rate)),
            'max_depth': None if max_depth <= 0 else int(max_depth),
            'max_leaf_nodes': None if max_leaf_nodes is None else int(max_leaf_nodes),
            'min_samples_leaf': int(max(1, hist_gradient_boosting_min_samples_leaf)),
            'l2_regularization': float(hgb_l2_regularization),
            'max_bins': int(hgb_max_bins),
        }

    if model_type == 'gaussian_process':
        kernel = ConstantKernel(constant_value=max(1e-8, float(gaussian_process_constant_value))) * GPRRBFKernel(
            length_scale=max(1e-8, float(gaussian_process_length_scale))
        )
        base = GaussianProcessRegressor(
            kernel=kernel,
            alpha=max(1e-12, float(gaussian_process_alpha)),
            n_restarts_optimizer=max(0, int(gaussian_process_n_restarts_optimizer)),
            normalize_y=bool(gaussian_process_normalize_y),
            random_state=42,
        )
        if Y_train.shape[1] > 1:
            model = MultiOutputRegressor(base)
            model.fit(X_train, Y_train)
        else:
            model = base
            model.fit(X_train, np.asarray(Y_train).reshape(-1))
        return {
            'model_type': model_type,
            'model': model,
            'alpha': float(max(1e-12, gaussian_process_alpha)),
            'length_scale': float(max(1e-8, gaussian_process_length_scale)),
            'constant_value': float(max(1e-8, gaussian_process_constant_value)),
            'n_restarts_optimizer': int(max(0, gaussian_process_n_restarts_optimizer)),
            'normalize_y': bool(gaussian_process_normalize_y),
        }

    if model_type == 'pls':
        # sklearn PLS always mean-centers X and Y internally.
        # `pls_scale` controls only standard-deviation scaling after centering.
        max_comp = max(1, min(X_train.shape[1], X_train.shape[0]))
        n_comp = max(1, min(int(n_components), max_comp))
        model = PLSRegression(n_components=n_comp, scale=bool(pls_scale))
        model.fit(X_train, Y_train)
        return {
            'model_type': model_type,
            'model': model,
            'n_components': int(n_comp),
            'pls_scale': bool(pls_scale),
        }

    if model_type == 'pcr':
        max_comp = max(1, min(X_train.shape[1], X_train.shape[0]))
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

    if model_type == 'kernel_pls':
        max_comp = max(1, min(X_train.shape[0] - 1, int(max(1, kernel_pls_n_features)) - 1))
        n_comp = max(1, min(int(n_components), max_comp))
        sampler = RBFSampler(
            gamma=max(1e-12, float(kernel_pls_gamma)),
            n_components=max(10, int(kernel_pls_n_features)),
            random_state=42,
        )
        X_map = sampler.fit_transform(X_train)
        model = PLSRegression(n_components=n_comp, scale=bool(pls_scale))
        model.fit(X_map, Y_train)
        return {
            'model_type': model_type,
            'model': model,
            'sampler': sampler,
            'n_components': int(n_comp),
            'gamma': float(max(1e-12, kernel_pls_gamma)),
            'mapped_features': int(max(10, int(kernel_pls_n_features))),
            'pls_scale': bool(pls_scale),
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
            'local_minkowski_p': float(local_minkowski_p),
        }

    raise ValueError(f"Unsupported model_type '{model_type}'.")


def _predict_model(model_info: Dict[str, Any], X_data: np.ndarray) -> np.ndarray:
    model_type = model_info.get('model_type')
    X_data = np.asarray(X_data, dtype=float)

    if model_type in ('ols', 'ridge', 'pls', 'elastic_net', 'svr', 'gradient_boosting', 'hist_gradient_boosting', 'gaussian_process'):
        return np.asarray(model_info['model'].predict(X_data), dtype=float)

    if model_type == 'lasso':
        y_pred = np.asarray(model_info['model'].predict(X_data), dtype=float)
        if y_pred.ndim == 1:
            return y_pred.reshape(-1, 1)
        return y_pred

    if model_type == 'random_forest':
        y_pred = np.asarray(model_info['model'].predict(X_data), dtype=float)
        if y_pred.ndim == 1:
            return y_pred.reshape(-1, 1)
        return y_pred

    if model_type == 'pcr':
        pca = model_info['pca']
        reg = model_info['reg']
        scores = pca.transform(X_data)
        return np.asarray(reg.predict(scores), dtype=float)

    if model_type == 'kernel_pls':
        sampler = model_info['sampler']
        X_map = sampler.transform(X_data)
        return np.asarray(model_info['model'].predict(X_map), dtype=float)

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
            model_info.get('local_minkowski_p', 3.0),
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
    pls_scale: bool,
    ridge_alpha: float,
    lasso_alpha: float,
    elastic_net_alpha: float,
    elastic_net_l1_ratio: float,
    linear_max_iter: int,
    random_forest_n_estimators: int,
    random_forest_max_depth: int,
    random_forest_min_samples_leaf: int,
    random_forest_max_features: Any,
    random_forest_bootstrap: bool,
    random_forest_oob_score: bool,
    svr_c: float,
    svr_epsilon: float,
    svr_gamma: float,
    svr_kernel: str,
    svr_degree: int,
    svr_coef0: float,
    gradient_boosting_n_estimators: int,
    gradient_boosting_learning_rate: float,
    gradient_boosting_subsample: float,
    gradient_boosting_max_features: Any,
    gradient_boosting_max_depth: int,
    gradient_boosting_min_samples_leaf: int,
    hist_gradient_boosting_n_estimators: int,
    hist_gradient_boosting_learning_rate: float,
    hist_gradient_boosting_max_depth: int,
    hist_gradient_boosting_min_samples_leaf: int,
    hist_gradient_boosting_l2_regularization: float,
    hist_gradient_boosting_max_bins: int,
    hist_gradient_boosting_max_leaf_nodes: int,
    gaussian_process_alpha: float,
    gaussian_process_length_scale: float,
    gaussian_process_constant_value: float,
    gaussian_process_n_restarts_optimizer: int,
    gaussian_process_normalize_y: bool,
    kernel_pls_gamma: float,
    kernel_pls_n_features: int,
    n_neighbors: int,
    local_method: str,
    idw_power: float,
    adaptive_alpha: float,
    radius_threshold: float,
    kernel_bandwidth: float,
    local_distance: str,
    local_minkowski_p: float,
    cv_config: Optional[CVConfig],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    if cv_config is None or not HAS_CV or not cv_config.is_enabled():
        return None, None

    splits = _build_cv_splits(X_cal, Y_cal, cv_config)
    y_cv_pred = np.zeros_like(Y_cal, dtype=float)
    fold_metrics: List[Dict[str, Any]] = []
    fold_numbers: List[int] = []
    fold_n_train: List[int] = []
    fold_n_test: List[int] = []
    fold_r2_train: List[float] = []
    fold_r2_test: List[float] = []
    fold_rmse_train: List[float] = []
    fold_rmse_test: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X_cal[train_idx], X_cal[test_idx]
        Y_train, Y_test = Y_cal[train_idx], Y_cal[test_idx]

        fold_model = _fit_model(
            model_type=model_type,
            X_train=X_train,
            Y_train=Y_train,
            n_components=n_components,
            pls_scale=pls_scale,
            ridge_alpha=ridge_alpha,
            lasso_alpha=lasso_alpha,
            elastic_net_alpha=elastic_net_alpha,
            elastic_net_l1_ratio=elastic_net_l1_ratio,
            random_forest_n_estimators=random_forest_n_estimators,
            random_forest_max_depth=random_forest_max_depth,
            random_forest_min_samples_leaf=random_forest_min_samples_leaf,
            random_forest_max_features=random_forest_max_features,
            random_forest_bootstrap=random_forest_bootstrap,
            random_forest_oob_score=random_forest_oob_score,
            svr_c=svr_c,
            svr_epsilon=svr_epsilon,
            svr_gamma=svr_gamma,
            svr_kernel=svr_kernel,
            svr_degree=svr_degree,
            svr_coef0=svr_coef0,
            gradient_boosting_n_estimators=gradient_boosting_n_estimators,
            gradient_boosting_learning_rate=gradient_boosting_learning_rate,
            gradient_boosting_subsample=gradient_boosting_subsample,
            gradient_boosting_max_features=gradient_boosting_max_features,
            gradient_boosting_max_depth=gradient_boosting_max_depth,
            gradient_boosting_min_samples_leaf=gradient_boosting_min_samples_leaf,
            hist_gradient_boosting_n_estimators=hist_gradient_boosting_n_estimators,
            hist_gradient_boosting_learning_rate=hist_gradient_boosting_learning_rate,
            hist_gradient_boosting_max_depth=hist_gradient_boosting_max_depth,
            hist_gradient_boosting_min_samples_leaf=hist_gradient_boosting_min_samples_leaf,
            hist_gradient_boosting_l2_regularization=hist_gradient_boosting_l2_regularization,
            hist_gradient_boosting_max_bins=hist_gradient_boosting_max_bins,
            hist_gradient_boosting_max_leaf_nodes=hist_gradient_boosting_max_leaf_nodes,
            gaussian_process_alpha=gaussian_process_alpha,
            gaussian_process_length_scale=gaussian_process_length_scale,
            gaussian_process_constant_value=gaussian_process_constant_value,
            gaussian_process_n_restarts_optimizer=gaussian_process_n_restarts_optimizer,
            gaussian_process_normalize_y=gaussian_process_normalize_y,
            kernel_pls_gamma=kernel_pls_gamma,
            kernel_pls_n_features=kernel_pls_n_features,
            n_neighbors=n_neighbors,
            local_method=local_method,
            idw_power=idw_power,
            adaptive_alpha=adaptive_alpha,
            radius_threshold=radius_threshold,
            kernel_bandwidth=kernel_bandwidth,
            linear_max_iter=linear_max_iter,
            local_distance=local_distance,
            local_minkowski_p=local_minkowski_p,
        )
        fold_train_pred = _align_prediction_shape(_predict_model(fold_model, X_train), Y_train)
        fold_test_pred = _align_prediction_shape(_predict_model(fold_model, X_test), Y_test)
        y_cv_pred[test_idx] = fold_test_pred

        train_metrics = _compute_metrics(Y_train, fold_train_pred)
        test_metrics = _compute_metrics(Y_test, fold_test_pred)

        fold_number = int(fold_idx + 1)
        n_train = int(len(train_idx))
        n_test = int(len(test_idx))
        r2_train = float(train_metrics.get('R2', np.nan))
        r2_test = float(test_metrics.get('R2', np.nan))
        rmse_train = float(train_metrics.get('RMSEP', np.nan))
        rmse_test = float(test_metrics.get('RMSEP', np.nan))

        fold_numbers.append(fold_number)
        fold_n_train.append(n_train)
        fold_n_test.append(n_test)
        fold_r2_train.append(r2_train)
        fold_r2_test.append(r2_test)
        fold_rmse_train.append(rmse_train)
        fold_rmse_test.append(rmse_test)

        fold_metrics.append({
            'fold': fold_number,
            'n_train': n_train,
            'n_test': n_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
        })

    cv_metrics = _compute_metrics(Y_cal, y_cv_pred)
    cv_results = {
        'n_folds': int(len(splits)),
        'fold_metrics': fold_metrics,
        'fold': fold_numbers,
        'n_train': fold_n_train,
        'n_test': fold_n_test,
        'r2_train': fold_r2_train,
        'r2_test': fold_r2_test,
        'rmse_train': fold_rmse_train,
        'rmse_test': fold_rmse_test,
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
    pls_scale: bool = False,
    ridge_alpha: float = 1.0,
    lasso_alpha: float = 1.0,
    elastic_net_alpha: float = 1.0,
    elastic_net_l1_ratio: float = 0.5,
    linear_max_iter: int = 10000,
    random_forest_n_estimators: int = 200,
    random_forest_max_depth: int = 0,
    random_forest_min_samples_leaf: int = 1,
    random_forest_max_features: Any = 1.0,
    random_forest_bootstrap: bool = True,
    random_forest_oob_score: bool = False,
    svr_c: float = 1.0,
    svr_epsilon: float = 0.1,
    svr_gamma: float = 0.1,
    svr_kernel: str = 'rbf',
    svr_degree: int = 3,
    svr_coef0: float = 0.0,
    gradient_boosting_n_estimators: int = 200,
    gradient_boosting_learning_rate: float = 0.05,
    gradient_boosting_subsample: float = 1.0,
    gradient_boosting_max_features: Any = 1.0,
    gradient_boosting_max_depth: int = 3,
    gradient_boosting_min_samples_leaf: int = 1,
    hist_gradient_boosting_n_estimators: int = 200,
    hist_gradient_boosting_learning_rate: float = 0.05,
    hist_gradient_boosting_max_depth: int = 0,
    hist_gradient_boosting_min_samples_leaf: int = 20,
    hist_gradient_boosting_l2_regularization: float = 0.0,
    hist_gradient_boosting_max_bins: int = 255,
    hist_gradient_boosting_max_leaf_nodes: int = 31,
    gaussian_process_alpha: float = 1e-6,
    gaussian_process_length_scale: float = 1.0,
    gaussian_process_constant_value: float = 1.0,
    gaussian_process_n_restarts_optimizer: int = 0,
    gaussian_process_normalize_y: bool = True,
    kernel_pls_gamma: float = 0.1,
    kernel_pls_n_features: int = 300,
    local_method: str = 'knn',
    n_neighbors: int = 5,
    idw_power: float = 1.0,
    adaptive_alpha: float = 1.0,
    radius_threshold: float = 1.0,
    kernel_bandwidth: float = 1.0,
    local_distance: str = 'euclidean',
    local_minkowski_p: float = 3.0,
    optimize_parameters: bool = False,
    n_components_range: Optional[Any] = None,
    ridge_alpha_range: Optional[Any] = None,
    lasso_alpha_range: Optional[Any] = None,
    elastic_net_alpha_range: Optional[Any] = None,
    random_forest_n_estimators_range: Optional[Any] = None,
    svr_c_range: Optional[Any] = None,
    gradient_boosting_n_estimators_range: Optional[Any] = None,
    hist_gradient_boosting_n_estimators_range: Optional[Any] = None,
    gaussian_process_alpha_range: Optional[Any] = None,
    n_neighbors_range: Optional[Any] = None,
    idw_power_range: Optional[Any] = None,
    adaptive_alpha_range: Optional[Any] = None,
    radius_threshold_range: Optional[Any] = None,
    kernel_bandwidth_range: Optional[Any] = None,
    cv_config: Optional[Any] = None,
    was_scaled: Optional[bool] = None,
    translation_keys: Optional[Dict[str, Any]] = None,
    y_labels: Optional[Any] = None,
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
    if y_labels is None and 'y_labels' in kwargs:
        y_labels = kwargs.get('y_labels')

    X_cal = _ensure_2d(X_cal)
    Y_cal = _ensure_2d(Y_cal)
    X_val = _ensure_2d(X_val)
    Y_val = _ensure_2d(Y_val)
    if Y_val is not None and np.asarray(Y_val).size == 0:
        Y_val = None

    if X_cal is None or Y_cal is None:
        raise ValueError('X_cal and Y_cal are required.')
    if X_cal.shape[0] != Y_cal.shape[0]:
        raise ValueError('X_cal and Y_cal must have the same number of samples.')

    model_type_norm = str(model_type).strip().lower()
    pls_scale = bool(pls_scale)

    if was_scaled is None and 'was_scaled' in kwargs:
        was_scaled = kwargs.get('was_scaled')
    was_scaled = _coerce_optional_bool(was_scaled)

    scaling_fallback_applied = False
    if model_type_norm in ('ridge', 'lasso', 'elastic_net', 'svr', 'gaussian_process', 'kernel_pls') and was_scaled is not True:
        X_cal, X_val = _autoscale_from_calibration(X_cal=X_cal, X_val=X_val)
        scaling_fallback_applied = True
        emit_execution_warning(
            code='first_order_calibration_autoscale_fallback',
            details={
                'function': 'first_order_calibration',
                'model_type': model_type_norm,
                'autoscale_fallback_applied': True,
                'was_scaled': None if was_scaled is None else bool(was_scaled),
            },
        )

    local_method_norm = str(local_method).strip().lower()
    if local_method_norm not in ('knn', 'idw', 'adaptive', 'radius', 'kernel'):
        raise ValueError("local_method must be one of: 'knn', 'idw', 'adaptive', 'radius', 'kernel'.")
    local_distance_norm = str(local_distance).strip().lower()
    valid_local_distances = ('euclidean', 'mahalanobis', 'manhattan', 'minkowski', 'chebyshev', 'chord')
    if local_distance_norm not in valid_local_distances:
        raise ValueError(
            "local_distance must be one of: 'euclidean', 'mahalanobis', 'manhattan', 'minkowski', 'chebyshev', 'chord'."
        )
    local_minkowski_p_norm = float(local_minkowski_p)
    if local_distance_norm == 'minkowski':
        if not np.isfinite(local_minkowski_p_norm) or local_minkowski_p_norm <= 0.0:
            raise ValueError('local_minkowski_p must be a positive number when local_distance is minkowski.')

    parameter_name, fixed_value, selected_range_input = _resolve_optimization_target(
        model_type_norm=model_type_norm,
        local_method_norm=local_method_norm,
        n_components=n_components,
        ridge_alpha=ridge_alpha,
        lasso_alpha=lasso_alpha,
        elastic_net_alpha=elastic_net_alpha,
        random_forest_n_estimators=random_forest_n_estimators,
        svr_c=svr_c,
        gradient_boosting_n_estimators=gradient_boosting_n_estimators,
        hist_gradient_boosting_n_estimators=hist_gradient_boosting_n_estimators,
        gaussian_process_alpha=gaussian_process_alpha,
        n_neighbors=n_neighbors,
        idw_power=idw_power,
        adaptive_alpha=adaptive_alpha,
        radius_threshold=radius_threshold,
        kernel_bandwidth=kernel_bandwidth,
        n_components_range=n_components_range,
        ridge_alpha_range=ridge_alpha_range,
        lasso_alpha_range=lasso_alpha_range,
        elastic_net_alpha_range=elastic_net_alpha_range,
        random_forest_n_estimators_range=random_forest_n_estimators_range,
        svr_c_range=svr_c_range,
        gradient_boosting_n_estimators_range=gradient_boosting_n_estimators_range,
        hist_gradient_boosting_n_estimators_range=hist_gradient_boosting_n_estimators_range,
        gaussian_process_alpha_range=gaussian_process_alpha_range,
        n_neighbors_range=n_neighbors_range,
        idw_power_range=idw_power_range,
        adaptive_alpha_range=adaptive_alpha_range,
        radius_threshold_range=radius_threshold_range,
        kernel_bandwidth_range=kernel_bandwidth_range,
    )

    candidates = _parse_parameter_candidates(
        optimize_parameters=bool(optimize_parameters and parameter_name is not None),
        selected_range_input=selected_range_input,
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
    if bool(optimize_parameters and parameter_name is not None) and not use_cv_for_selection:
        if not HAS_CV:
            raise ValueError("Parameter optimization requires CV support, but CV pipeline is unavailable.")
        cv_config = CVConfig(
            use_cv=True,
            cv_strategy='loocv',
            n_splits=max(2, int(X_cal.shape[0])),
            random_state=42,
            shuffle=False,
        )
        use_cv_for_selection = True
        emit_execution_message(
            code='first_order_calibration_default_loocv',
            details={'function': 'first_order_calibration', 'cv_strategy': 'loocv'}
        )
    local_self_uses_loo = model_type_norm == 'local'

    for candidate in candidates:
        candidate_params = _candidate_hyperparameters(
            candidate=candidate,
            model_type_norm=model_type_norm,
            local_method_norm=local_method_norm,
            n_components=n_components,
            ridge_alpha=ridge_alpha,
            lasso_alpha=lasso_alpha,
            elastic_net_alpha=elastic_net_alpha,
            elastic_net_l1_ratio=elastic_net_l1_ratio,
            random_forest_n_estimators=random_forest_n_estimators,
            random_forest_max_depth=random_forest_max_depth,
            random_forest_min_samples_leaf=random_forest_min_samples_leaf,
            random_forest_max_features=random_forest_max_features,
            random_forest_bootstrap=random_forest_bootstrap,
            random_forest_oob_score=random_forest_oob_score,
            svr_c=svr_c,
            svr_epsilon=svr_epsilon,
            svr_gamma=svr_gamma,
            svr_kernel=svr_kernel,
            svr_degree=svr_degree,
            svr_coef0=svr_coef0,
            gradient_boosting_n_estimators=gradient_boosting_n_estimators,
            gradient_boosting_learning_rate=gradient_boosting_learning_rate,
            gradient_boosting_subsample=gradient_boosting_subsample,
            gradient_boosting_max_features=gradient_boosting_max_features,
            gradient_boosting_max_depth=gradient_boosting_max_depth,
            gradient_boosting_min_samples_leaf=gradient_boosting_min_samples_leaf,
            hist_gradient_boosting_n_estimators=hist_gradient_boosting_n_estimators,
            hist_gradient_boosting_learning_rate=hist_gradient_boosting_learning_rate,
            hist_gradient_boosting_max_depth=hist_gradient_boosting_max_depth,
            hist_gradient_boosting_min_samples_leaf=hist_gradient_boosting_min_samples_leaf,
            hist_gradient_boosting_l2_regularization=hist_gradient_boosting_l2_regularization,
            hist_gradient_boosting_max_bins=hist_gradient_boosting_max_bins,
            hist_gradient_boosting_max_leaf_nodes=hist_gradient_boosting_max_leaf_nodes,
            gaussian_process_alpha=gaussian_process_alpha,
            gaussian_process_length_scale=gaussian_process_length_scale,
            gaussian_process_constant_value=gaussian_process_constant_value,
            gaussian_process_n_restarts_optimizer=gaussian_process_n_restarts_optimizer,
            gaussian_process_normalize_y=gaussian_process_normalize_y,
            kernel_pls_gamma=kernel_pls_gamma,
            kernel_pls_n_features=kernel_pls_n_features,
            n_neighbors=n_neighbors,
            idw_power=idw_power,
            adaptive_alpha=adaptive_alpha,
            radius_threshold=radius_threshold,
            kernel_bandwidth=kernel_bandwidth,
        )

        model_info = _fit_model(
            model_type=model_type_norm,
            X_train=X_cal,
            Y_train=Y_cal,
            n_components=candidate_params['n_components'],
            pls_scale=pls_scale,
            ridge_alpha=candidate_params['ridge_alpha'],
            lasso_alpha=candidate_params['lasso_alpha'],
            elastic_net_alpha=candidate_params['elastic_net_alpha'],
            elastic_net_l1_ratio=candidate_params['elastic_net_l1_ratio'],
            random_forest_n_estimators=candidate_params['random_forest_n_estimators'],
            random_forest_max_depth=candidate_params['random_forest_max_depth'],
            random_forest_min_samples_leaf=candidate_params['random_forest_min_samples_leaf'],
            random_forest_max_features=candidate_params['random_forest_max_features'],
            random_forest_bootstrap=candidate_params['random_forest_bootstrap'],
            random_forest_oob_score=candidate_params['random_forest_oob_score'],
            svr_c=candidate_params['svr_c'],
            svr_epsilon=candidate_params['svr_epsilon'],
            svr_gamma=candidate_params['svr_gamma'],
            svr_kernel=candidate_params['svr_kernel'],
            svr_degree=candidate_params['svr_degree'],
            svr_coef0=candidate_params['svr_coef0'],
            gradient_boosting_n_estimators=candidate_params['gradient_boosting_n_estimators'],
            gradient_boosting_learning_rate=candidate_params['gradient_boosting_learning_rate'],
            gradient_boosting_subsample=candidate_params['gradient_boosting_subsample'],
            gradient_boosting_max_features=candidate_params['gradient_boosting_max_features'],
            gradient_boosting_max_depth=candidate_params['gradient_boosting_max_depth'],
            gradient_boosting_min_samples_leaf=candidate_params['gradient_boosting_min_samples_leaf'],
            hist_gradient_boosting_n_estimators=candidate_params['hist_gradient_boosting_n_estimators'],
            hist_gradient_boosting_learning_rate=candidate_params['hist_gradient_boosting_learning_rate'],
            hist_gradient_boosting_max_depth=candidate_params['hist_gradient_boosting_max_depth'],
            hist_gradient_boosting_min_samples_leaf=candidate_params['hist_gradient_boosting_min_samples_leaf'],
            hist_gradient_boosting_l2_regularization=candidate_params['hist_gradient_boosting_l2_regularization'],
            hist_gradient_boosting_max_bins=candidate_params['hist_gradient_boosting_max_bins'],
            hist_gradient_boosting_max_leaf_nodes=candidate_params['hist_gradient_boosting_max_leaf_nodes'],
            gaussian_process_alpha=candidate_params['gaussian_process_alpha'],
            gaussian_process_length_scale=candidate_params['gaussian_process_length_scale'],
            gaussian_process_constant_value=candidate_params['gaussian_process_constant_value'],
            gaussian_process_n_restarts_optimizer=candidate_params['gaussian_process_n_restarts_optimizer'],
            gaussian_process_normalize_y=candidate_params['gaussian_process_normalize_y'],
            kernel_pls_gamma=candidate_params['kernel_pls_gamma'],
            kernel_pls_n_features=candidate_params['kernel_pls_n_features'],
            n_neighbors=candidate_params['n_neighbors'],
            local_method=local_method_norm,
            idw_power=candidate_params['idw_power'],
            adaptive_alpha=candidate_params['adaptive_alpha'],
            radius_threshold=candidate_params['radius_threshold'],
            kernel_bandwidth=candidate_params['kernel_bandwidth'],
            linear_max_iter=linear_max_iter,
            local_distance=local_distance_norm,
            local_minkowski_p=local_minkowski_p_norm,
        )
        if local_self_uses_loo:
            y_self_pred = _predict_local_leave_one_out(
                X_data=X_cal,
                Y_data=Y_cal,
                n_neighbors=candidate_params['n_neighbors'],
                local_method=local_method_norm,
                idw_power=candidate_params['idw_power'],
                adaptive_alpha=candidate_params['adaptive_alpha'],
                radius_threshold=candidate_params['radius_threshold'],
                kernel_bandwidth=candidate_params['kernel_bandwidth'],
                local_distance=local_distance_norm,
                local_minkowski_p=local_minkowski_p_norm,
            )
        else:
            y_self_pred = _predict_model(model_info, X_cal)
        y_self_pred = _align_prediction_shape(y_self_pred, Y_cal)
        self_metrics = _compute_metrics(Y_cal, y_self_pred)

        y_val_pred_candidate = _predict_model(model_info, X_val) if X_val is not None else None
        if y_val_pred_candidate is not None and Y_val is not None:
            y_val_pred_candidate = _align_prediction_shape(y_val_pred_candidate, Y_val)
        val_metrics = _compute_metrics(Y_val, y_val_pred_candidate) if (Y_val is not None and y_val_pred_candidate is not None) else None

        y_cv_pred_candidate, _ = _cross_validated_predictions(
            X_cal=X_cal,
            Y_cal=Y_cal,
            model_type=model_type_norm,
            n_components=candidate_params['n_components'],
            pls_scale=pls_scale,
            ridge_alpha=candidate_params['ridge_alpha'],
            lasso_alpha=candidate_params['lasso_alpha'],
            elastic_net_alpha=candidate_params['elastic_net_alpha'],
            elastic_net_l1_ratio=candidate_params['elastic_net_l1_ratio'],
            random_forest_n_estimators=candidate_params['random_forest_n_estimators'],
            random_forest_max_depth=candidate_params['random_forest_max_depth'],
            random_forest_min_samples_leaf=candidate_params['random_forest_min_samples_leaf'],
            random_forest_max_features=candidate_params['random_forest_max_features'],
            random_forest_bootstrap=candidate_params['random_forest_bootstrap'],
            random_forest_oob_score=candidate_params['random_forest_oob_score'],
            svr_c=candidate_params['svr_c'],
            svr_epsilon=candidate_params['svr_epsilon'],
            svr_gamma=candidate_params['svr_gamma'],
            svr_kernel=candidate_params['svr_kernel'],
            svr_degree=candidate_params['svr_degree'],
            svr_coef0=candidate_params['svr_coef0'],
            gradient_boosting_n_estimators=candidate_params['gradient_boosting_n_estimators'],
            gradient_boosting_learning_rate=candidate_params['gradient_boosting_learning_rate'],
            gradient_boosting_subsample=candidate_params['gradient_boosting_subsample'],
            gradient_boosting_max_features=candidate_params['gradient_boosting_max_features'],
            gradient_boosting_max_depth=candidate_params['gradient_boosting_max_depth'],
            gradient_boosting_min_samples_leaf=candidate_params['gradient_boosting_min_samples_leaf'],
            hist_gradient_boosting_n_estimators=candidate_params['hist_gradient_boosting_n_estimators'],
            hist_gradient_boosting_learning_rate=candidate_params['hist_gradient_boosting_learning_rate'],
            hist_gradient_boosting_max_depth=candidate_params['hist_gradient_boosting_max_depth'],
            hist_gradient_boosting_min_samples_leaf=candidate_params['hist_gradient_boosting_min_samples_leaf'],
            hist_gradient_boosting_l2_regularization=candidate_params['hist_gradient_boosting_l2_regularization'],
            hist_gradient_boosting_max_bins=candidate_params['hist_gradient_boosting_max_bins'],
            hist_gradient_boosting_max_leaf_nodes=candidate_params['hist_gradient_boosting_max_leaf_nodes'],
            gaussian_process_alpha=candidate_params['gaussian_process_alpha'],
            gaussian_process_length_scale=candidate_params['gaussian_process_length_scale'],
            gaussian_process_constant_value=candidate_params['gaussian_process_constant_value'],
            gaussian_process_n_restarts_optimizer=candidate_params['gaussian_process_n_restarts_optimizer'],
            gaussian_process_normalize_y=candidate_params['gaussian_process_normalize_y'],
            kernel_pls_gamma=candidate_params['kernel_pls_gamma'],
            kernel_pls_n_features=candidate_params['kernel_pls_n_features'],
            n_neighbors=candidate_params['n_neighbors'],
            local_method=local_method_norm,
            idw_power=candidate_params['idw_power'],
            adaptive_alpha=candidate_params['adaptive_alpha'],
            radius_threshold=candidate_params['radius_threshold'],
            kernel_bandwidth=candidate_params['kernel_bandwidth'],
            linear_max_iter=linear_max_iter,
            local_distance=local_distance_norm,
            local_minkowski_p=local_minkowski_p_norm,
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
        else:
            score = float(self_metrics['RMSEP'])
        if score < best_score:
            best_score = score
            best_candidate = candidate

    final_params = _candidate_hyperparameters(
        candidate=best_candidate,
        model_type_norm=model_type_norm,
        local_method_norm=local_method_norm,
        n_components=n_components,
        ridge_alpha=ridge_alpha,
        lasso_alpha=lasso_alpha,
        elastic_net_alpha=elastic_net_alpha,
        elastic_net_l1_ratio=elastic_net_l1_ratio,
        random_forest_n_estimators=random_forest_n_estimators,
        random_forest_max_depth=random_forest_max_depth,
        random_forest_min_samples_leaf=random_forest_min_samples_leaf,
        random_forest_max_features=random_forest_max_features,
        random_forest_bootstrap=random_forest_bootstrap,
        random_forest_oob_score=random_forest_oob_score,
        svr_c=svr_c,
        svr_epsilon=svr_epsilon,
        svr_gamma=svr_gamma,
        svr_kernel=svr_kernel,
        svr_degree=svr_degree,
        svr_coef0=svr_coef0,
        gradient_boosting_n_estimators=gradient_boosting_n_estimators,
        gradient_boosting_learning_rate=gradient_boosting_learning_rate,
        gradient_boosting_subsample=gradient_boosting_subsample,
        gradient_boosting_max_features=gradient_boosting_max_features,
        gradient_boosting_max_depth=gradient_boosting_max_depth,
        gradient_boosting_min_samples_leaf=gradient_boosting_min_samples_leaf,
        hist_gradient_boosting_n_estimators=hist_gradient_boosting_n_estimators,
        hist_gradient_boosting_learning_rate=hist_gradient_boosting_learning_rate,
        hist_gradient_boosting_max_depth=hist_gradient_boosting_max_depth,
        hist_gradient_boosting_min_samples_leaf=hist_gradient_boosting_min_samples_leaf,
        hist_gradient_boosting_l2_regularization=hist_gradient_boosting_l2_regularization,
        hist_gradient_boosting_max_bins=hist_gradient_boosting_max_bins,
        hist_gradient_boosting_max_leaf_nodes=hist_gradient_boosting_max_leaf_nodes,
        gaussian_process_alpha=gaussian_process_alpha,
        gaussian_process_length_scale=gaussian_process_length_scale,
        gaussian_process_constant_value=gaussian_process_constant_value,
        gaussian_process_n_restarts_optimizer=gaussian_process_n_restarts_optimizer,
        gaussian_process_normalize_y=gaussian_process_normalize_y,
        kernel_pls_gamma=kernel_pls_gamma,
        kernel_pls_n_features=kernel_pls_n_features,
        n_neighbors=n_neighbors,
        idw_power=idw_power,
        adaptive_alpha=adaptive_alpha,
        radius_threshold=radius_threshold,
        kernel_bandwidth=kernel_bandwidth,
    )

    final_model = _fit_model(
        model_type=model_type_norm,
        X_train=X_cal,
        Y_train=Y_cal,
        n_components=final_params['n_components'],
        pls_scale=pls_scale,
        ridge_alpha=final_params['ridge_alpha'],
        lasso_alpha=final_params['lasso_alpha'],
        elastic_net_alpha=final_params['elastic_net_alpha'],
        elastic_net_l1_ratio=final_params['elastic_net_l1_ratio'],
        random_forest_n_estimators=final_params['random_forest_n_estimators'],
        random_forest_max_depth=final_params['random_forest_max_depth'],
        random_forest_min_samples_leaf=final_params['random_forest_min_samples_leaf'],
        random_forest_max_features=final_params['random_forest_max_features'],
        random_forest_bootstrap=final_params['random_forest_bootstrap'],
        random_forest_oob_score=final_params['random_forest_oob_score'],
        svr_c=final_params['svr_c'],
        svr_epsilon=final_params['svr_epsilon'],
        svr_gamma=final_params['svr_gamma'],
        svr_kernel=final_params['svr_kernel'],
        svr_degree=final_params['svr_degree'],
        svr_coef0=final_params['svr_coef0'],
        gradient_boosting_n_estimators=final_params['gradient_boosting_n_estimators'],
        gradient_boosting_learning_rate=final_params['gradient_boosting_learning_rate'],
        gradient_boosting_subsample=final_params['gradient_boosting_subsample'],
        gradient_boosting_max_features=final_params['gradient_boosting_max_features'],
        gradient_boosting_max_depth=final_params['gradient_boosting_max_depth'],
        gradient_boosting_min_samples_leaf=final_params['gradient_boosting_min_samples_leaf'],
        hist_gradient_boosting_n_estimators=final_params['hist_gradient_boosting_n_estimators'],
        hist_gradient_boosting_learning_rate=final_params['hist_gradient_boosting_learning_rate'],
        hist_gradient_boosting_max_depth=final_params['hist_gradient_boosting_max_depth'],
        hist_gradient_boosting_min_samples_leaf=final_params['hist_gradient_boosting_min_samples_leaf'],
        hist_gradient_boosting_l2_regularization=final_params['hist_gradient_boosting_l2_regularization'],
        hist_gradient_boosting_max_bins=final_params['hist_gradient_boosting_max_bins'],
        hist_gradient_boosting_max_leaf_nodes=final_params['hist_gradient_boosting_max_leaf_nodes'],
        gaussian_process_alpha=final_params['gaussian_process_alpha'],
        gaussian_process_length_scale=final_params['gaussian_process_length_scale'],
        gaussian_process_constant_value=final_params['gaussian_process_constant_value'],
        gaussian_process_n_restarts_optimizer=final_params['gaussian_process_n_restarts_optimizer'],
        gaussian_process_normalize_y=final_params['gaussian_process_normalize_y'],
        kernel_pls_gamma=final_params['kernel_pls_gamma'],
        kernel_pls_n_features=final_params['kernel_pls_n_features'],
        n_neighbors=final_params['n_neighbors'],
        local_method=local_method_norm,
        idw_power=final_params['idw_power'],
        adaptive_alpha=final_params['adaptive_alpha'],
        radius_threshold=final_params['radius_threshold'],
        kernel_bandwidth=final_params['kernel_bandwidth'],
        linear_max_iter=linear_max_iter,
        local_distance=local_distance_norm,
        local_minkowski_p=local_minkowski_p_norm,
    )

    y_val_pred = None

    if local_self_uses_loo:
        y_cal_pred = _predict_local_leave_one_out(
            X_data=X_cal,
            Y_data=Y_cal,
            n_neighbors=final_params['n_neighbors'],
            local_method=local_method_norm,
            idw_power=final_params['idw_power'],
            adaptive_alpha=final_params['adaptive_alpha'],
            radius_threshold=final_params['radius_threshold'],
            kernel_bandwidth=final_params['kernel_bandwidth'],
            local_distance=local_distance_norm,
            local_minkowski_p=local_minkowski_p_norm,
        )
        y_cal_pred = _align_prediction_shape(y_cal_pred, Y_cal)

        y_val_pred = _predict_model(final_model, X_val) if X_val is not None else None
        if y_val_pred is not None and Y_val is not None:
            y_val_pred = _align_prediction_shape(y_val_pred, Y_val)
    else:
        y_cal_pred = _predict_model(final_model, X_cal)
        y_cal_pred = _align_prediction_shape(y_cal_pred, Y_cal)

        y_val_pred = _predict_model(final_model, X_val) if X_val is not None else None
        if y_val_pred is not None and Y_val is not None:
            y_val_pred = _align_prediction_shape(y_val_pred, Y_val)

    y_cv_pred, cv_results = _cross_validated_predictions(
        X_cal=X_cal,
        Y_cal=Y_cal,
        model_type=model_type_norm,
        n_components=final_params['n_components'],
        pls_scale=pls_scale,
        ridge_alpha=final_params['ridge_alpha'],
        lasso_alpha=final_params['lasso_alpha'],
        elastic_net_alpha=final_params['elastic_net_alpha'],
        elastic_net_l1_ratio=final_params['elastic_net_l1_ratio'],
        random_forest_n_estimators=final_params['random_forest_n_estimators'],
        random_forest_max_depth=final_params['random_forest_max_depth'],
        random_forest_min_samples_leaf=final_params['random_forest_min_samples_leaf'],
        random_forest_max_features=final_params['random_forest_max_features'],
        random_forest_bootstrap=final_params['random_forest_bootstrap'],
        random_forest_oob_score=final_params['random_forest_oob_score'],
        svr_c=final_params['svr_c'],
        svr_epsilon=final_params['svr_epsilon'],
        svr_gamma=final_params['svr_gamma'],
        svr_kernel=final_params['svr_kernel'],
        svr_degree=final_params['svr_degree'],
        svr_coef0=final_params['svr_coef0'],
        gradient_boosting_n_estimators=final_params['gradient_boosting_n_estimators'],
        gradient_boosting_learning_rate=final_params['gradient_boosting_learning_rate'],
        gradient_boosting_subsample=final_params['gradient_boosting_subsample'],
        gradient_boosting_max_features=final_params['gradient_boosting_max_features'],
        gradient_boosting_max_depth=final_params['gradient_boosting_max_depth'],
        gradient_boosting_min_samples_leaf=final_params['gradient_boosting_min_samples_leaf'],
        hist_gradient_boosting_n_estimators=final_params['hist_gradient_boosting_n_estimators'],
        hist_gradient_boosting_learning_rate=final_params['hist_gradient_boosting_learning_rate'],
        hist_gradient_boosting_max_depth=final_params['hist_gradient_boosting_max_depth'],
        hist_gradient_boosting_min_samples_leaf=final_params['hist_gradient_boosting_min_samples_leaf'],
        hist_gradient_boosting_l2_regularization=final_params['hist_gradient_boosting_l2_regularization'],
        hist_gradient_boosting_max_bins=final_params['hist_gradient_boosting_max_bins'],
        hist_gradient_boosting_max_leaf_nodes=final_params['hist_gradient_boosting_max_leaf_nodes'],
        gaussian_process_alpha=final_params['gaussian_process_alpha'],
        gaussian_process_length_scale=final_params['gaussian_process_length_scale'],
        gaussian_process_constant_value=final_params['gaussian_process_constant_value'],
        gaussian_process_n_restarts_optimizer=final_params['gaussian_process_n_restarts_optimizer'],
        gaussian_process_normalize_y=final_params['gaussian_process_normalize_y'],
        kernel_pls_gamma=final_params['kernel_pls_gamma'],
        kernel_pls_n_features=final_params['kernel_pls_n_features'],
        n_neighbors=final_params['n_neighbors'],
        local_method=local_method_norm,
        idw_power=final_params['idw_power'],
        adaptive_alpha=final_params['adaptive_alpha'],
        radius_threshold=final_params['radius_threshold'],
        kernel_bandwidth=final_params['kernel_bandwidth'],
        linear_max_iter=linear_max_iter,
        local_distance=local_distance_norm,
        local_minkowski_p=local_minkowski_p_norm,
        cv_config=cv_config,
    )

    metrics_cal = _compute_metrics(Y_cal, y_cal_pred)
    metrics_cv = _compute_metrics(Y_cal, y_cv_pred) if y_cv_pred is not None else None
    metrics_val = _compute_metrics(Y_val, y_val_pred) if (Y_val is not None and y_val_pred is not None) else None

    selected_parameter_display_name = _resolve_parameter_display_name(
        parameter_name,
        translation_keys,
        model_type=model_type_norm,
        local_method=local_method_norm,
    )

    optimization_results = {
        'parameter_name': parameter_name,
        'parameter_display_name': selected_parameter_display_name,
        'parameter_values': [float(v) if isinstance(v, (float, np.floating)) else int(v) for v in optimization_parameter_values],
        'self_r2': optimization_self_r2,
        'cv_r2': optimization_cv_r2,
        'val_r2': optimization_val_r2,
        'self_rmsep': optimization_self_rmsep,
        'cv_rmsep': optimization_cv_rmsep,
        'val_rmsep': optimization_val_rmsep,
        'best_value': best_candidate,
        'selection_source': 'cv' if use_cv_for_selection else 'self',
        'optimization_used': bool(optimize_parameters and parameter_name is not None),
    }

    selected_parameter_value = _selected_parameter_value(parameter_name=parameter_name, params=final_params)

    model_specific_payload = _build_model_specific_payload(
        model_type=model_type_norm,
        fitted_model=final_model,
        params=final_params,
        local_method=local_method_norm,
        local_distance=local_distance_norm,
        Y_train=Y_cal,
        X_train=X_cal,
    )

    if model_type_norm in ('ols', 'ridge', 'lasso', 'elastic_net'):
        linear_payload = model_specific_payload.get('linear') if isinstance(model_specific_payload, dict) else None
        if isinstance(linear_payload, dict):
            model_specific_payload['summary_report_text'] = _build_linear_model_summary_report(
                model_type_norm=model_type_norm,
                linear_payload=linear_payload,
                metrics_cal=metrics_cal,
                y_labels=y_labels,
            )

    residual_cal = np.asarray(Y_cal, dtype=float) - np.asarray(y_cal_pred, dtype=float)
    residual_flat = residual_cal.reshape(-1)
    diagnostics = {
        'calibration_residuals': [float(v) for v in residual_flat.tolist()],
        'residual_mean': float(np.mean(residual_flat)) if residual_flat.size else 0.0,
        'residual_std': float(np.std(residual_flat)) if residual_flat.size else 0.0,
        'residual_max_abs': float(np.max(np.abs(residual_flat))) if residual_flat.size else 0.0,
    }
    if y_val_pred is not None and Y_val is not None:
        residual_val = np.asarray(Y_val, dtype=float) - np.asarray(y_val_pred, dtype=float)
        diagnostics['validation_residuals'] = [float(v) for v in residual_val.reshape(-1).tolist()]
    model_specific_payload['diagnostics'] = diagnostics

    model_payload = {
        'model_type': model_type_norm,
        'pls_scale': pls_scale if model_type_norm in ('pls', 'kernel_pls') else None,
        'was_scaled': bool(was_scaled) if was_scaled is not None else None,
        'autoscale_fallback_applied': bool(scaling_fallback_applied),
        'local_method': local_method_norm if model_type_norm == 'local' else None,
        'local_distance': local_distance_norm if model_type_norm == 'local' else None,
        'local_minkowski_p': float(local_minkowski_p_norm) if (model_type_norm == 'local' and local_distance_norm == 'minkowski') else None,
        'elastic_net_alpha': float(final_params['elastic_net_alpha']) if model_type_norm == 'elastic_net' else None,
        'elastic_net_l1_ratio': float(final_params['elastic_net_l1_ratio']) if model_type_norm == 'elastic_net' else None,
        'random_forest_n_estimators': int(final_params['random_forest_n_estimators']) if model_type_norm == 'random_forest' else None,
        'random_forest_max_depth': int(final_params['random_forest_max_depth']) if model_type_norm == 'random_forest' else None,
        'random_forest_min_samples_leaf': int(final_params['random_forest_min_samples_leaf']) if model_type_norm == 'random_forest' else None,
        'random_forest_max_features': final_params['random_forest_max_features'] if model_type_norm == 'random_forest' else None,
        'random_forest_bootstrap': bool(final_params['random_forest_bootstrap']) if model_type_norm == 'random_forest' else None,
        'random_forest_oob_score': bool(final_params['random_forest_oob_score']) if model_type_norm == 'random_forest' else None,
        'svr_c': float(final_params['svr_c']) if model_type_norm == 'svr' else None,
        'svr_epsilon': float(final_params['svr_epsilon']) if model_type_norm == 'svr' else None,
        'svr_gamma': float(final_params['svr_gamma']) if model_type_norm == 'svr' else None,
        'svr_kernel': str(final_params['svr_kernel']) if model_type_norm == 'svr' else None,
        'svr_degree': int(final_params['svr_degree']) if model_type_norm == 'svr' else None,
        'svr_coef0': float(final_params['svr_coef0']) if model_type_norm == 'svr' else None,
        'gradient_boosting_n_estimators': int(final_params['gradient_boosting_n_estimators']) if model_type_norm == 'gradient_boosting' else None,
        'gradient_boosting_learning_rate': float(final_params['gradient_boosting_learning_rate']) if model_type_norm == 'gradient_boosting' else None,
        'gradient_boosting_subsample': float(final_params['gradient_boosting_subsample']) if model_type_norm == 'gradient_boosting' else None,
        'gradient_boosting_max_features': final_params['gradient_boosting_max_features'] if model_type_norm == 'gradient_boosting' else None,
        'gradient_boosting_max_depth': int(final_params['gradient_boosting_max_depth']) if model_type_norm == 'gradient_boosting' else None,
        'gradient_boosting_min_samples_leaf': int(final_params['gradient_boosting_min_samples_leaf']) if model_type_norm == 'gradient_boosting' else None,
        'hist_gradient_boosting_n_estimators': int(final_params['hist_gradient_boosting_n_estimators']) if model_type_norm == 'hist_gradient_boosting' else None,
        'hist_gradient_boosting_learning_rate': float(final_params['hist_gradient_boosting_learning_rate']) if model_type_norm == 'hist_gradient_boosting' else None,
        'hist_gradient_boosting_max_depth': int(final_params['hist_gradient_boosting_max_depth']) if model_type_norm == 'hist_gradient_boosting' else None,
        'hist_gradient_boosting_max_leaf_nodes': int(final_params['hist_gradient_boosting_max_leaf_nodes']) if model_type_norm == 'hist_gradient_boosting' else None,
        'hist_gradient_boosting_min_samples_leaf': int(final_params['hist_gradient_boosting_min_samples_leaf']) if model_type_norm == 'hist_gradient_boosting' else None,
        'hist_gradient_boosting_l2_regularization': float(final_params['hist_gradient_boosting_l2_regularization']) if model_type_norm == 'hist_gradient_boosting' else None,
        'hist_gradient_boosting_max_bins': int(final_params['hist_gradient_boosting_max_bins']) if model_type_norm == 'hist_gradient_boosting' else None,
        'gaussian_process_alpha': float(final_params['gaussian_process_alpha']) if model_type_norm == 'gaussian_process' else None,
        'gaussian_process_length_scale': float(final_params['gaussian_process_length_scale']) if model_type_norm == 'gaussian_process' else None,
        'gaussian_process_constant_value': float(final_params['gaussian_process_constant_value']) if model_type_norm == 'gaussian_process' else None,
        'gaussian_process_n_restarts_optimizer': int(final_params['gaussian_process_n_restarts_optimizer']) if model_type_norm == 'gaussian_process' else None,
        'gaussian_process_normalize_y': bool(final_params['gaussian_process_normalize_y']) if model_type_norm == 'gaussian_process' else None,
        'kernel_pls_gamma': float(final_params['kernel_pls_gamma']) if model_type_norm == 'kernel_pls' else None,
        'kernel_pls_n_features': int(final_params['kernel_pls_n_features']) if model_type_norm == 'kernel_pls' else None,
        'idw_power': float(final_params['idw_power']) if model_type_norm == 'local' and local_method_norm == 'idw' else None,
        'adaptive_alpha': float(final_params['adaptive_alpha']) if model_type_norm == 'local' and local_method_norm == 'adaptive' else None,
        'radius_threshold': float(final_params['radius_threshold']) if model_type_norm == 'local' and local_method_norm == 'radius' else None,
        'kernel_bandwidth': float(final_params['kernel_bandwidth']) if model_type_norm == 'local' and local_method_norm == 'kernel' else None,
        'model_specific': model_specific_payload,
    }
    metrics_payload = {
        'calibration': metrics_cal,
        'cv': metrics_cv,
        'validation': metrics_val,
    }

    n_y = int(np.asarray(Y_cal).shape[1]) if np.asarray(Y_cal).ndim == 2 else 1
    model_payload['report_text'] = _build_first_order_report(
        n_y=n_y,
        metrics=metrics_payload,
        model=model_payload,
        selected_parameter_name=selected_parameter_display_name,
        selected_parameter_value=selected_parameter_value,
        optimization_results=optimization_results,
        y_labels=y_labels,
    )

    y_cal_error = np.asarray(Y_cal, dtype=float) - np.asarray(y_cal_pred, dtype=float)
    y_val_error = None
    if y_val_pred is not None and Y_val is not None:
        y_val_error = np.asarray(Y_val, dtype=float) - np.asarray(y_val_pred, dtype=float)
    y_cv_error = None
    if y_cv_pred is not None:
        y_cv_error = np.asarray(Y_cal, dtype=float) - np.asarray(y_cv_pred, dtype=float)

    # Diagonal line for Reference vs Predicted scatter (shared by all prediction graphs)
    pred_ref_diag_x = None
    pred_ref_diag_y = None
    _diag_candidates = [np.asarray(Y_cal, dtype=float).ravel(), np.asarray(y_cal_pred, dtype=float).ravel()]
    if y_cv_pred is not None:
        _diag_candidates.append(np.asarray(y_cv_pred, dtype=float).ravel())
    if Y_val is not None:
        _diag_candidates.append(np.asarray(Y_val, dtype=float).ravel())
    if y_val_pred is not None:
        _diag_candidates.append(np.asarray(y_val_pred, dtype=float).ravel())
    _all_diag_vals = np.concatenate(_diag_candidates)
    _finite_diag_vals = _all_diag_vals[np.isfinite(_all_diag_vals)]
    if _finite_diag_vals.size > 0:
        _diag_min = float(np.min(_finite_diag_vals))
        _diag_max = float(np.max(_finite_diag_vals))
        _diag_pad = (_diag_max - _diag_min) * 0.05 + 1e-6
        pred_ref_diag_x = np.array([_diag_min - _diag_pad, _diag_max + _diag_pad], dtype=float)
        pred_ref_diag_y = pred_ref_diag_x.copy()

    # EJCR packed payload: one variable contains all ellipses + metadata per graph.
    _ejcr_n_pts = 100
    _ejcr_n_path = _ejcr_n_pts * 2 + 1
    _Y_cal_2d = np.asarray(Y_cal, dtype=float)
    if _Y_cal_2d.ndim == 1:
        _Y_cal_2d = _Y_cal_2d.reshape(-1, 1)
    _ejcr_n_col = _Y_cal_2d.shape[1]
    _ejcr_levels = ("90", "95", "99")
    _ejcr_level_to_idx = {lvl: i for i, lvl in enumerate(_ejcr_levels)}
    _ejcr_item_count = len(_ejcr_levels) + 1  # + ideal point marker

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

        if Y_val is not None and y_val_pred is not None:
            _Y_val_2d = np.asarray(Y_val, dtype=float)
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

    return (
        y_cal_pred,
        y_val_pred,
        y_cv_pred,
        Y_cal,
        Y_val,
        model_payload,
        selected_parameter_display_name,
        selected_parameter_value,
        metrics_payload,
        cv_results,
        optimization_results,
        y_cal_error,
        y_val_error,
        y_cv_error,
        pred_ref_diag_x,
        pred_ref_diag_y,
        ejcr_cal,
        ejcr_val,
    )
