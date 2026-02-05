from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats

# Import CV pipeline
try:
    from chemometrics.cv_pipeline import CVPipeline, CVConfig
    HAS_CV = True
except ImportError:
    HAS_CV = False


def _ensure_2d_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _design_matrix(x: np.ndarray, degree: int, intercept: bool) -> np.ndarray:
    """Build polynomial design matrix for vector x.

    Returns matrix with columns corresponding to powers x^1..x^degree. If intercept=True,
    caller should prepend a column of ones when computing statistics (we handle that there).
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(x.reshape(-1, 1))


def _fit_ols(X_design: np.ndarray, y: np.ndarray, fit_intercept: bool) -> Dict[str, Any]:
    """Fit OLS and compute coefficient statistics (std err, t, p-values)."""
    # Use linear algebra to compute coefficients and statistics
    n_samples, n_feats = X_design.shape

    if fit_intercept:
        # prepend ones
        X = np.hstack([np.ones((n_samples, 1)), X_design])
    else:
        X = X_design

    # Solve least squares
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X.dot(beta)
    resid = y - y_hat
    ss_res = np.sum(resid ** 2)

    # Degrees of freedom and sigma2
    p = X.shape[1]
    dof = max(n_samples - p, 0)
    if dof > 0:
        sigma2 = ss_res / dof
    else:
        sigma2 = np.nan

    # Covariance matrix of beta: sigma2 * (X'X)^{-1}
    try:
        xtx_inv = np.linalg.inv(X.T.dot(X))
        cov_beta = xtx_inv * sigma2
        se = np.sqrt(np.diag(cov_beta))
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stats = beta / se
            p_values = stats.t.sf(np.abs(t_stats), dof) * 2
    except np.linalg.LinAlgError:
        se = np.full(beta.shape, np.nan)
        t_stats = np.full(beta.shape, np.nan)
        p_values = np.full(beta.shape, np.nan)

    return {
        'beta': beta,
        'y_hat': y_hat,
        'resid': resid,
        'ss_res': ss_res,
        'sigma2': sigma2,
        'se': se,
        't_stats': t_stats,
        'p_values': p_values,
        'dof': dof,
        'p': p,
        'n': n_samples,
    }


def _metrics(y: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    n = y.shape[0]
    rmse = np.sqrt(np.mean((y - y_hat) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return {'RMSE': float(rmse), 'R2': float(r2), 'SS_res': float(ss_res), 'SS_tot': float(ss_tot), 'n': int(n)}


def univariate_calibration(
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    degree: int = 1,
    intercept: bool = True,
    cv_config: Optional[Any] = None,
    fold: int = 0,
    reference_output_key: Optional[str] = None,
    capture_output_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build univariate polynomial calibration models for all column combinations.

    Args:
        X_cal: calibration X (vector or 2D matrix samples x variables)
        Y_cal: calibration Y (vector or 2D matrix samples x responses)
        X_val: optional validation X (samples x variables)
        Y_val: optional validation Y (samples x responses)
        degree: polynomial degree (required)
        intercept: whether to fit an intercept (set False to force zero intercept)
        cv_config: optional CVConfig object from cv_configuration() function.
                   If provided and use_cv=True, runs cross-validation.
                   Otherwise runs single fit.
        fold: fold index (passed by CVPipeline, not user-set)
        reference_output_key: (NEW) Which output to use as reference for single-fit comparison
                              e.g., 'y_cal_pred' to use calibration predictions as reference
        capture_output_keys: (NEW) Which outputs to capture per fold
                             e.g., ['y_cal_pred', 'metrics'] 

    Returns:
        Dict with keys:
        - y_cal_pred: dict mapping model_key -> calibration predicted values
        - y_val_pred: dict mapping model_key -> validation predicted values (or None)
        - models: dict with full model details (coefficients, stats, residuals, etc.)
        - metrics: dict aggregating RMSEC, RMSEV, R2 for each model
        - cv_results: (optional, if CV enabled) aggregated CV metrics across folds
        
    Example - Single-fit reference mode:
        results = univariate_calibration(
            X_cal, Y_cal,
            cv_config=my_config,
            reference_output_key='y_cal_pred',
            capture_output_keys=['y_cal_pred', 'metrics']
        )
        # Results now include per-fold comparisons to single fit
    """
    # Handle CV routing - unwrap cv_config if it's been wrapped in a dict by routing
    if cv_config is not None and isinstance(cv_config, dict) and 'cv_config' in cv_config:
        cv_config = cv_config['cv_config']
    
    # Handle CV routing
    if cv_config is not None and HAS_CV and cv_config.is_enabled():
        # If this is the initial call (fold=0 and not from pipeline), run CV
        if fold == 0 and not _is_cv_fold_call():
            pipeline = CVPipeline(cv_config)
            
            # For univariate calibration, y_cal_pred is a dict, not a single array.
            # We need to run CV differently - aggregate metrics per model key.
            # Use input reference mode: compare predictions against actual Y_cal
            cv_results = pipeline.run(
                _univariate_calibration_single_fit,
                X_cal=X_cal,
                Y_cal=Y_cal,
                X_val=X_val,
                Y_val=Y_val,
                degree=degree,
                intercept=intercept,
                reference_input_key='Y_cal',
                comparison_output_key='y_cal_pred_matrix',
                capture_output_keys=['y_cal_pred_matrix'],
            )
            
            # Also compute single fit on full data
            single_results = _univariate_calibration_single_fit(
                X_cal, Y_cal, X_val, Y_val, degree, intercept, fold=-1
            )
            return {
                **single_results,
                'cv_results': cv_results,
            }
    
    # Single fit (no CV or CV disabled)
    return _univariate_calibration_single_fit(
        X_cal, Y_cal, X_val, Y_val, degree, intercept, fold=fold
    )


def _is_cv_fold_call() -> bool:
    """Check if we're being called from CVPipeline (hacky but works)."""
    import inspect
    stack = inspect.stack()
    for frame_info in stack:
        if 'CVPipeline' in str(frame_info.filename) or 'cv_pipeline' in str(frame_info.filename):
            return True
    return False


def _univariate_calibration_single_fit(
    X_cal: Optional[np.ndarray] = None,
    Y_cal: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    degree: int = 1,
    intercept: bool = True,
    fold: int = 0,
    reference_output_key: Optional[str] = None,
    capture_output_keys: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Internal: single fit (used both standalone and by CVPipeline).
    
    Accepts both direct format (X_cal, Y_cal) and split format (X_cal_train, X_cal_test)
    for CVPipeline compatibility.
    """
    # Handle CVPipeline split format: X_cal_train/X_cal_test, Y_cal_train/Y_cal_test
    X_cal_test = kwargs.get('X_cal_test')
    Y_cal_test = kwargs.get('Y_cal_test')
    
    if X_cal is None and 'X_cal_train' in kwargs:
        X_cal = kwargs['X_cal_train']
    if Y_cal is None and 'Y_cal_train' in kwargs:
        Y_cal = kwargs['Y_cal_train']
    if X_val is None and 'X_val_train' in kwargs:
        X_val = kwargs['X_val_train']
    if Y_val is None and 'Y_val_train' in kwargs:
        Y_val = kwargs['Y_val_train']
    
    Xc = _ensure_2d_matrix(X_cal)
    Yc = _ensure_2d_matrix(Y_cal)
    
    # For CV: also handle test set
    if X_cal_test is not None:
        Xc_test = _ensure_2d_matrix(X_cal_test)
    else:
        Xc_test = None

    if X_val is not None:
        Xv = _ensure_2d_matrix(X_val)
    else:
        Xv = None
    if Y_val is not None:
        Yv = _ensure_2d_matrix(Y_val)
    else:
        Yv = None

    n_samples = Xc.shape[0]
    if Yc.shape[0] != n_samples:
        raise ValueError('X_cal and Y_cal must have the same number of rows')
    if Xv is not None and Xv.shape[0] != (Yv.shape[0] if Yv is not None else Xv.shape[0]):
        # no strict check here: assume user supplies corresponding validation Y
        pass

    results: Dict[str, Any] = {'models': {}, 'summary': {}}

    n_x = Xc.shape[1]
    n_y = Yc.shape[1]
    
    # Dictionaries to collect outputs
    y_cal_pred: Dict[str, Any] = {}
    y_val_pred: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    
    # Matrix to store predictions for CV (samples x responses)
    # For CV: use test set predictions; for single fit: use train predictions
    # Shape matches the appropriate set
    if Xc_test is not None:
        n_pred_samples = Xc_test.shape[0]
    else:
        n_pred_samples = n_samples
    Yc_pred_matrix = np.zeros((n_pred_samples, n_y), dtype=np.float64)

    for i in range(n_x):
        x_col_cal = Xc[:, i]
        # build design matrix for calibration (training)
        X_design_cal = _design_matrix(x_col_cal, degree, intercept)
        
        # prepare test design for CV if provided
        if Xc_test is not None:
            x_col_test = Xc_test[:, i] if i < Xc_test.shape[1] else None
            X_design_test = _design_matrix(x_col_test, degree, intercept) if x_col_test is not None else None
        else:
            X_design_test = None

        # prepare validation design if provided
        if Xv is not None:
            x_col_val = Xv[:, i] if i < Xv.shape[1] else None
            X_design_val = _design_matrix(x_col_val, degree, intercept) if x_col_val is not None else None
        else:
            X_design_val = None

        for j in range(n_y):
            y_col_cal = Yc[:, j]
            model_key = f'X{i}_Y{j}'

            fit = _fit_ols(X_design_cal, y_col_cal, fit_intercept=intercept)

            # metrics for calibration
            metrics_cal = _metrics(y_col_cal, fit['y_hat'])

            # validation predictions and metrics
            if X_design_val is not None and Yv is not None and j < Yv.shape[1]:
                # construct X for validation with intercept if needed
                if intercept:
                    Xv_mat = np.hstack([np.ones((X_design_val.shape[0], 1)), X_design_val])
                else:
                    Xv_mat = X_design_val
                y_hat_val = Xv_mat.dot(fit['beta'])
                y_col_val = Yv[:, j]
                metrics_val = _metrics(y_col_val, y_hat_val)
            elif X_design_val is not None:
                # no Y_val provided, still compute predicted values
                if intercept:
                    Xv_mat = np.hstack([np.ones((X_design_val.shape[0], 1)), X_design_val])
                else:
                    Xv_mat = X_design_val
                y_hat_val = Xv_mat.dot(fit['beta'])
                metrics_val = None
                y_col_val = None
            else:
                y_hat_val = None
                metrics_val = None
                y_col_val = None

            # assemble coefficient labels
            coeffs = {}
            labels: List[str] = []
            if intercept:
                coeffs['intercept'] = float(fit['beta'][0])
                labels.append('intercept')
                coefs_arr = fit['beta'][1:]
                se_arr = fit['se'][1:]
                p_arr = fit['p_values'][1:]
            else:
                coefs_arr = fit['beta']
                se_arr = fit['se']
                p_arr = fit['p_values']

            for k in range(coefs_arr.shape[0]):
                power = k + 1
                coeffs[f'coef_{power}'] = float(coefs_arr[k])

            # collect std err and p-values similarly
            stats_coeffs = {}
            if intercept:
                stats_coeffs['intercept'] = {'std_err': float(fit['se'][0]) if fit['se'].size>0 else np.nan,
                                             't_stat': float(fit['t_stats'][0]) if fit['t_stats'].size>0 else np.nan,
                                             'p_value': float(fit['p_values'][0]) if fit['p_values'].size>0 else np.nan}
                for k in range(se_arr.shape[0]):
                    stats_coeffs[f'coef_{k+1}'] = {'std_err': float(se_arr[k]) if not np.isnan(se_arr[k]) else np.nan,
                                                   't_stat': float(fit['t_stats'][k + (1 if intercept else 0)]) if fit['t_stats'].size>0 else np.nan,
                                                   'p_value': float(p_arr[k]) if not np.isnan(p_arr[k]) else np.nan}
            else:
                for k in range(se_arr.shape[0]):
                    stats_coeffs[f'coef_{k+1}'] = {'std_err': float(se_arr[k]) if not np.isnan(se_arr[k]) else np.nan,
                                                   't_stat': float(fit['t_stats'][k]) if fit['t_stats'].size>0 else np.nan,
                                                   'p_value': float(p_arr[k]) if not np.isnan(p_arr[k]) else np.nan}

            results['models'][model_key] = {
                'x_index': i,
                'y_index': j,
                'degree': degree,
                'intercept': bool(intercept),
                'coefficients': coeffs,
                'coeff_stats': stats_coeffs,
                'y_cal_predicted': fit['y_hat'],
                'y_cal_observed': y_col_cal,
                'metrics_cal': metrics_cal,
                'y_val_predicted': y_hat_val,
                'y_val_observed': y_col_val,
                'metrics_val': metrics_val,
                'residuals_cal': fit['resid'],
                'n_cal': int(fit['n']),
                'p': int(fit['p']),
                'dof': int(fit['dof']) if not np.isnan(fit['dof']) else None,
            }
            
            # Collect outputs for return tuple
            y_cal_pred[model_key] = fit['y_hat']
            y_val_pred[model_key] = y_hat_val
            metrics[model_key] = {
                'metrics_cal': metrics_cal,
                'metrics_val': metrics_val,
            }
            
            # Store predictions in matrix form for CV comparison
            # For CV: predict on test set; for single fit: predict on train set
            if i == 0:
                if X_design_test is not None:
                    # CV mode: predict on test set
                    if intercept:
                        Xtest_mat = np.hstack([np.ones((X_design_test.shape[0], 1)), X_design_test])
                    else:
                        Xtest_mat = X_design_test
                    y_hat_test = Xtest_mat.dot(fit['beta'])
                    Yc_pred_matrix[:, j] = y_hat_test
                else:
                    # Single fit: use train predictions
                    Yc_pred_matrix[:, j] = fit['y_hat']

    return {
        'y_cal_pred': y_cal_pred,
        'y_val_pred': y_val_pred,
        'y_cal_pred_matrix': Yc_pred_matrix,  # Matrix form for CV comparison
        'models': results['models'],
        'metrics': metrics,
    }
