"""Principal Component Analysis (PCA) for calibration and validation data."""
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sklearn.decomposition import PCA

# Import CV pipeline
try:
    from chemometrics.cv_pipeline import CVPipeline, CVConfig
    HAS_CV = True
except ImportError:
    HAS_CV = False


def _ensure_2d_matrix(X: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    """Ensure input is 2D matrix (samples x variables).
    
    For multiway data (ndim >= 3), unfolds all dimensions except samples
    using U-PCA (Unfolded-PCA) approach.
    
    Args:
        X: Input array (can be 1D, 2D, or multiway 3D/4D/etc)
        
    Returns:
        Tuple of (X_2d, original_shape) where:
        - X_2d: 2D array (samples x unfolded_variables)
        - original_shape: Original shape of input (for potential reshaping later)
    """
    X = np.asarray(X)
    original_shape = X.shape
    
    if X.ndim == 1:
        # 1D -> reshape to (n_samples, 1)
        X_2d = X.reshape(-1, 1)
    elif X.ndim == 2:
        # Already 2D
        X_2d = X
    else:
        # Multiway (3D, 4D, etc) -> U-PCA unfolding
        # Keep samples (axis 0), unfold rest into one dimension
        n_samples = X.shape[0]
        # Reshape to (n_samples, -1) which flattens all other dimensions
        X_2d = X.reshape(n_samples, -1)
    
    return X_2d, original_shape


def _is_cv_fold_call() -> bool:
    """Check if we're being called from CVPipeline (hacky but works)."""
    import inspect
    stack = inspect.stack()
    for frame_info in stack:
        if 'CVPipeline' in str(frame_info.filename) or 'cv_pipeline' in str(frame_info.filename):
            return True
    return False


def _compute_pca_metrics(scores: np.ndarray, X_data: np.ndarray, loadings: np.ndarray) -> Dict[str, Any]:
    """Compute PCA-relevant metrics.
    
    Args:
        scores: PCA scores (samples x n_components)
        X_data: Original data (samples x n_variables)
        loadings: PCA loadings (n_variables x n_components)
        
    Returns:
        Dictionary with PCA metrics
    """
    n_components = scores.shape[1]
    n_variables = X_data.shape[1]
    n_samples = X_data.shape[0]
    
    # Compute variance explained by each component
    explained_variance = np.sum(scores ** 2, axis=0) / (n_samples - 1)
    total_variance = np.sum(np.var(X_data, axis=0) * (n_samples / (n_samples - 1)))
    pct_var_explained = (explained_variance / total_variance) * 100
    cumsum_pct_var = np.cumsum(pct_var_explained)
    
    # Reconstruct data from scores and loadings
    X_reconstructed = scores @ loadings.T
    
    # Reconstruction error
    reconstruction_error = np.mean((X_data - X_reconstructed) ** 2)
    rmse_reconstruction = np.sqrt(reconstruction_error)
    
    return {
        'n_components': int(n_components),
        'n_variables': int(n_variables),
        'n_samples': int(n_samples),
        'explained_variance': explained_variance.tolist(),
        'pct_variance_explained': pct_var_explained.tolist(),
        'cumsum_pct_variance': cumsum_pct_var.tolist(),
        'total_variance': float(total_variance),
        'reconstruction_rmse': float(rmse_reconstruction),
    }


def pca_analysis(
    X_cal: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    n_components: int = 2,
    cv_config: Optional[Any] = None,
    fold: int = 0,
    axis_n_info: Optional[List[str]] = None,
    class_data_cal: Optional[List[str]] = None,
    class_data_val: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Principal Component Analysis with cross-validation and U-PCA support.
    
    Performs PCA on calibration data, projects validation data using 
    the learned loadings, and supports cross-validation for robust 
    component selection. Handles both univariate (2D) and multiway (3D+) data
    using U-PCA (Unfolded-PCA) approach.

    Args:
        X_cal: calibration X (samples x variables for 2D, or multiway 3D+)
               For multiway: samples x var1 x var2 x ... (U-PCA unfolds var dimensions)
        X_val: optional validation X (same shape as X_cal)
        n_components: number of PCA components to retain
        cv_config: optional CVConfig object from cv_configuration() function.
                   If provided and use_cv=True, runs cross-validation.
                   Otherwise runs single fit.
        fold: fold index (passed by CVPipeline, not user-set)
        axis_n_info: optional axis information for variable labels
        class_data_cal: optional list of class labels for calibration samples
        class_data_val: optional list of class labels for validation samples

    Returns:
        Dict with keys:
        - model_scores: calibration PCA scores (samples x n_components)
        - model_loadings: PCA loadings (unfolded_variables x n_components)
        - val_scores: validation scores (or None if no X_val)
        - model_scores_cv: per-fold calibration scores (if CV enabled)
        - model_loadings_cv: per-fold loadings (if CV enabled)
        - model_pca_object: fitted sklearn PCA object
        - metrics: dictionary with variance explained and other metrics
        - metrics_cv: per-fold metrics (if CV enabled)
        - cv_results: aggregated CV metrics (if CV enabled)
        - data_shape: original shape of input data (for multiway reference)
        
    Note on Multiway Data:
        When X_cal has ndim >= 3 (e.g., shape (100, 50, 10)):
        - Uses U-PCA (Unfolded-PCA) approach
        - Keeps sample dimension (axis 0) intact
        - Unfolds all other dimensions into single variable dimension
        - Example: (100, 50, 10) -> treated as (100, 500) for PCA
        - Loadings shape reflects unfolded space: (500, n_components)
        
    Example - Single fit:
        results = pca_analysis(X_cal, X_val, n_components=5)
        model_scores = results['model_scores']
        val_scores = results['val_scores']
        
    Example - Multiway (U-PCA):
        X_cal_3d = np.random.randn(100, 50, 10)  # Multiway data
        results = pca_analysis(X_cal_3d, n_components=5)
        # Internally: (100, 50, 10) -> (100, 500) for PCA
        data_shape = results['data_shape']  # (100, 50, 10)
        
    Example - With cross-validation:
        cv_config = CVConfig(use_cv=True, cv_strategy='kfold', n_splits=5)
        results = pca_analysis(X_cal, X_val, n_components=5, cv_config=cv_config)
        cv_results = results['cv_results']
    """
    # Handle CV routing - unwrap cv_config if it's been wrapped in a dict by routing
    if cv_config is not None and isinstance(cv_config, dict) and 'cv_config' in cv_config:
        cv_config = cv_config['cv_config']
    
    # Handle CV routing
    if cv_config is not None and HAS_CV and cv_config.is_enabled():
        # If this is the initial call (fold=0 and not from pipeline), run CV
        if fold == 0 and not _is_cv_fold_call():
            # Collect per-fold loadings via closure (avoids the ambiguity in
            # _reconstruct_from_folds when n_vars happens to equal n_test_samples).
            _fold_loadings_collector: list = []
            
            # Create a wrapper that returns dict format for CV pipeline
            def _pca_analysis_for_cv(**kwargs):
                """Wrapper that converts tuple output to dict for CV pipeline capture.
                
                Remaps outputs so CV pipeline captures the correct arrays:
                - 'model_scores' is overridden with the test-fold projected scores
                  (scores_cv) so _reconstruct_from_folds can align them by sample index.
                - Per-fold loadings are collected via closure for later tensor stacking.
                """
                result = _pca_analysis_single_fit(**kwargs)
                # Convert tuple to dict
                return_keys = ['model_scores', 'model_loadings', 'val_scores', 
                             'model_scores_cv', 'model_loadings_cv', 'metrics', 'cv_results',
                             'data_shape', 'data_shape_val', 'axis_n_info', 'pc_labels',
                             'class_data_cal', 'class_data_val']
                result_dict = dict(zip(return_keys, result))
                # Collect loadings per fold for tensor stacking
                _fold_loadings_collector.append(result_dict['model_loadings'])
                # Override model_scores with the test-fold projected scores so the
                # CV pipeline captures test scores (n_test, n_components) rather than
                # training scores (n_train, n_components).  Only test scores have
                # shape[0] == len(test_idx), which lets _reconstruct_from_folds place
                # them at the correct sample positions in the full array.
                if result_dict.get('model_scores_cv') is not None:
                    result_dict['model_scores'] = result_dict['model_scores_cv']
                return result_dict
            
            pipeline = CVPipeline(cv_config)
            
            # Run CV — only capture model_scores through the pipeline (scores are
            # sample-based and benefit from _reconstruct_from_folds).  Loadings are
            # collected via the closure above to avoid shape ambiguity.
            cv_results_dict = pipeline.run(
                _pca_analysis_for_cv,
                X_cal=X_cal,
                n_components=n_components,
                capture_output_keys=['model_scores'],
            )
            
            # Also compute single fit on full data (with X_val for projection)
            single_results = _pca_analysis_single_fit(
                X_cal, X_val, n_components, fold=-1, axis_n_info=axis_n_info,
                class_data_cal=class_data_cal, class_data_val=class_data_val
            )
            # single_results is a tuple, extract it
            model_scores, model_loadings, val_scores, scores_cv, loadings_cv, metrics, _, cal_shape, val_shape, axis_labels, pc_labels, _, _ = single_results
            
            # Extract CV scores from pipeline (reconstructed array: n_samples × n_components)
            cv_model_scores = cv_results_dict.get('model_scores_cv', None)
            
            # Build loadings tensor from the closure collector (n_folds × n_vars × n_components)
            if _fold_loadings_collector:
                cv_model_loadings = np.array(_fold_loadings_collector)
            else:
                cv_model_loadings = None
            
            # Build cv_results dict (remove the _cv suffixed keys which are outputs, not results)
            cv_results = {k: v for k, v in cv_results_dict.items() 
                         if k not in ['model_scores_cv', 'model_loadings_cv']}
            
            # Return as tuple (not dict) so analyst.py can handle it consistently
            return (
                model_scores,
                model_loadings,
                val_scores,
                cv_model_scores,  # Use CV scores from pipeline
                cv_model_loadings,  # Use CV loadings from pipeline
                metrics,
                cv_results,  # CV metrics and fold info
                cal_shape,
                val_shape,
                axis_labels,
                pc_labels,
                class_data_cal,  # calibration class data
                class_data_val,  # validation class data
            )
    
    # Single fit (no CV or CV disabled)
    return _pca_analysis_single_fit(
        X_cal, X_val, n_components, fold=fold, axis_n_info=axis_n_info,
        class_data_cal=class_data_cal, class_data_val=class_data_val
    )


def _pca_analysis_single_fit(
    X_cal: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    n_components: int = 2,
    fold: int = 0,
    axis_n_info: Optional[List[str]] = None,
    class_data_cal: Optional[List[str]] = None,
    class_data_val: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Internal: single PCA fit (used both standalone and by CVPipeline).
    
    Accepts both direct format (X_cal) and split format (X_cal_train, X_cal_test)
    for CVPipeline compatibility. Handles multiway data via U-PCA.
    """
    # Handle CVPipeline split format: X_cal_train/X_cal_test
    X_cal_test = kwargs.get('X_cal_test')
    
    if X_cal is None and 'X_cal_train' in kwargs:
        X_cal = kwargs['X_cal_train']
    if X_val is None and 'X_val_train' in kwargs:
        X_val = kwargs['X_val_train']
    
    # Convert to 2D using U-PCA approach for multiway data
    Xc, cal_shape = _ensure_2d_matrix(X_cal)
    
    if X_val is not None:
        Xv, val_shape = _ensure_2d_matrix(X_val)
    else:
        Xv = None
        val_shape = None
    
    if X_cal_test is not None:
        Xc_test, test_shape = _ensure_2d_matrix(X_cal_test)
    else:
        Xc_test = None
        test_shape = None
    
    # Fit PCA on calibration data (training set)
    pca = PCA(n_components=n_components)
    model_scores = pca.fit_transform(Xc)
    model_loadings = pca.components_.T  # Convert from (n_components, n_vars) to (n_vars, n_components)
    
    # Compute metrics for calibration
    metrics_cal = _compute_pca_metrics(model_scores, Xc, model_loadings)
    
    # Project validation data if provided
    if Xv is not None:
        val_scores = pca.transform(Xv)
        metrics_val = _compute_pca_metrics(val_scores, Xv, model_loadings)
    else:
        val_scores = None
        metrics_val = None
    
    # For CV: also compute scores on test set
    scores_cv = None
    metrics_cv = None
    if Xc_test is not None:
        scores_cv = pca.transform(Xc_test)
        metrics_cv = _compute_pca_metrics(scores_cv, Xc_test, model_loadings)
    
    # Build cv_results following FoldSegregatedOutput structure if this is a CV call
    cv_results = None
    if Xc_test is not None:
        # From CV pipeline
        cv_results = {
            'calibration': {'metrics': metrics_cal},
            'test': {'metrics': metrics_cv, 'scores': scores_cv},
        }
    
    # Return tuple in order specified by function_specs.json return_specs for pca_analysis
    # Generate PC component labels (e.g., ["PC 1", "PC 2", "PC 3"])
    pc_labels = [f"PC {i+1}" for i in range(n_components)]
    
    return (
        model_scores,
        model_loadings,
        val_scores,
        scores_cv,  # model_scores_cv
        model_loadings,  # model_loadings_cv (same across folds)
        {
            'calibration': metrics_cal,
            'validation': metrics_val,
            'pct_variance_explained': metrics_cal.get('pct_variance_explained', []),  # For easy access in plotting
        },  # metrics
        cv_results,  # cv_results
        cal_shape,  # data_shape (original shape for multiway reference)
        val_shape,  # data_shape_val
        axis_n_info,  # variable labels from load_data
        pc_labels,  # component labels (e.g., ["PC 1", "PC 2", "PC 3"])
        class_data_cal,  # calibration class data
        class_data_val,  # validation class data
    )
