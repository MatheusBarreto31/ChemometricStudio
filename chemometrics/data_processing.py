## Data processing functions for ChemometricsTool

from typing import Tuple, Optional, Union, List
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d


def _determine_dimensionality(X: np.ndarray) -> int:
    """
    Auto-determine data dimensionality.
    
    Args:
        X: Input data array
        
    Returns:
        Number of dimensions (1, 2, or 3+)
    """
    return X.ndim


def baseline_correction(
    X_cal: np.ndarray,
    method: str = 'msc',
    X_val: Optional[np.ndarray] = None,
    nway_flag: Optional[int] = None,
    direction: Optional[int] = None,
    window_size: int = 5,
    derivative_order: int = 1,
    axis_n_info: Optional[List[np.ndarray]] = None,
    axis_t_info: Optional[List[List[str]]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[np.ndarray]], Optional[List[List[str]]]]:
    """
    Apply baseline correction to spectroscopic data.
    
    Supports multiple baseline correction methods:
    - 'msc': Multiplicative Scatter Correction
    - 'snv'/'svn': Standard Normal Variate
    - 'moving_average': Moving Average baseline subtraction
    - 'derivative': Numerical derivative (np.diff)
    
    Args:
        X_cal: Calibration data (reference dataset)
        method: Baseline correction method ('msc', 'svn', 'moving_average')
        X_val: Optional validation data (will be corrected using X_cal reference)
        nway_flag: Number of ways (if None, auto-determined from X_cal)
        direction: Direction for multiway data (0, 1, 2, etc.). If None, applied on last axis
        window_size: Window size for moving average method
        derivative_order: Derivative order for derivative method
        axis_n_info: Optional numerical axis info to propagate/update
        axis_t_info: Optional text axis info to propagate/update
        
    Returns:
        Tuple with (X_cal, X_val, axis_n_info, axis_t_info)
    """
    if nway_flag is None:
        nway_flag = _determine_dimensionality(X_cal)

    method = str(method or 'msc').strip().lower()
    if method == 'svn':
        method = 'snv'

    axis_n_info_out = list(axis_n_info) if isinstance(axis_n_info, list) else axis_n_info
    axis_t_info_out = list(axis_t_info) if isinstance(axis_t_info, list) else axis_t_info
    
    if method == 'msc':
        msc_reference = _msc_reference(X_cal, direction=direction)
        X_cal = _msc(X_cal, nway_flag, direction, msc_reference)
        if X_val is not None:
            X_val = _msc(X_val, nway_flag, direction, msc_reference)
        return X_cal, X_val, axis_n_info_out, axis_t_info_out
    
    elif method == 'snv':
        X_cal = _snv(X_cal, nway_flag, direction)
        if X_val is not None:
            X_val = _snv(X_val, nway_flag, direction)
        return X_cal, X_val, axis_n_info_out, axis_t_info_out
    
    elif method == 'moving_average':
        X_cal = _moving_average_baseline(X_cal, nway_flag, direction, window_size)
        if X_val is not None:
            X_val = _moving_average_baseline(X_val, nway_flag, direction, window_size)
        return X_cal, X_val, axis_n_info_out, axis_t_info_out

    elif method == 'derivative':
        axis = _resolve_axis(direction, X_cal.ndim)
        X_cal = _derivative_baseline(X_cal, axis=axis, derivative_order=derivative_order)
        if X_val is not None:
            X_val = _derivative_baseline(X_val, axis=axis, derivative_order=derivative_order)
        axis_n_info_out, axis_t_info_out = _trim_axis_info_for_derivative(
            axis_n_info_out,
            axis_t_info_out,
            axis,
            derivative_order
        )
        return X_cal, X_val, axis_n_info_out, axis_t_info_out
    
    else:
        raise ValueError(f"Unknown baseline correction method: {method}")


def _resolve_axis(direction: Optional[int], ndim: int) -> int:
    if direction is None:
        return ndim - 1
    axis = int(direction)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Invalid direction/axis {direction} for data with ndim={ndim}")
    return axis


def _msc_reference(X_cal: np.ndarray, direction: Optional[int] = None) -> np.ndarray:
    """
    Compute MSC reference spectrum along the selected spectral axis.

    For a selected axis, all remaining axes are treated as replicate/sample
    dimensions and averaged to obtain the reference vector.
    """
    if X_cal.ndim == 1:
        return np.asarray(X_cal, dtype=float)

    axis = _resolve_axis(direction, X_cal.ndim)
    reduction_axes = tuple(idx for idx in range(X_cal.ndim) if idx != axis)
    return np.mean(X_cal, axis=reduction_axes)


def _msc(X: np.ndarray, nway_flag: int, direction: Optional[int] = None, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Multiplicative Scatter Correction (MSC).
    
    Args:
        X: Input data
        nway_flag: Number of dimensions
        direction: Direction for multiway data
        reference: Reference spectrum (if None, use mean of X)
        
    Returns:
        MSC-corrected data
    """
    axis = _resolve_axis(direction, X.ndim)
    if reference is None:
        reference = _msc_reference(X, direction=axis)

    X_float = np.asarray(X, dtype=float)
    ref = np.asarray(reference, dtype=float).reshape(-1)

    n_features = X_float.shape[axis]
    if ref.shape[0] != n_features:
        raise ValueError(
            f"MSC reference length ({ref.shape[0]}) does not match data size along axis {axis} ({n_features})"
        )

    # Bring spectral axis to the last dimension and treat all other dimensions
    # as independent spectra.
    moved = np.moveaxis(X_float, axis, -1)
    flat = moved.reshape(-1, n_features)

    ref_mean = np.mean(ref)
    ref_centered = ref - ref_mean
    ref_var = np.mean(ref_centered ** 2)
    eps = 1e-12
    if ref_var < eps:
        raise ValueError("MSC reference variance is zero or too small for stable correction")

    sample_means = np.mean(flat, axis=1)
    sample_centered = flat - sample_means[:, None]
    slopes = (sample_centered @ ref_centered) / (n_features * ref_var)
    intercepts = sample_means - slopes * ref_mean

    corrected_flat = flat.copy()
    stable_mask = np.abs(slopes) >= eps
    corrected_flat[stable_mask] = (
        (flat[stable_mask] - intercepts[stable_mask, None]) /
        slopes[stable_mask, None]
    )

    corrected_moved = corrected_flat.reshape(moved.shape)
    return np.moveaxis(corrected_moved, -1, axis)


def _snv(X: np.ndarray, nway_flag: int, direction: Optional[int] = None) -> np.ndarray:
    """
    Standard Normal Variate (SNV) normalization.
    
    Args:
        X: Input data
        nway_flag: Number of dimensions
        direction: Direction for multiway data
        
    Returns:
        SNV-normalized data
    """
    axis = _resolve_axis(direction, X.ndim)
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    return (X - mean) / (std + 1e-10)


def _moving_average_baseline(X: np.ndarray, nway_flag: int, direction: Optional[int] = None, window_size: int = 5) -> np.ndarray:
    """
    Moving average baseline correction.
    
    Args:
        X: Input data
        nway_flag: Number of dimensions
        direction: Direction for multiway data
        window_size: Window size for moving average
        
    Returns:
        Baseline-corrected data
    """
    axis = _resolve_axis(direction, X.ndim)
    baseline = uniform_filter1d(X, size=window_size, axis=axis, mode='nearest')
    return X - baseline


def _derivative_baseline(X: np.ndarray, axis: int, derivative_order: int = 1) -> np.ndarray:
    order = int(derivative_order)
    if order < 1:
        raise ValueError("derivative_order must be >= 1")
    if X.shape[axis] <= order:
        raise ValueError(
            f"derivative_order ({order}) must be smaller than data size along axis {axis} ({X.shape[axis]})"
        )
    return np.diff(X, n=order, axis=axis)


def _trim_axis_info_for_derivative(
    axis_n_info: Optional[List[np.ndarray]],
    axis_t_info: Optional[List[List[str]]],
    axis: int,
    derivative_order: int,
) -> Tuple[Optional[List[np.ndarray]], Optional[List[List[str]]]]:
    order = int(derivative_order)
    if order < 1:
        return axis_n_info, axis_t_info

    if isinstance(axis_n_info, list) and axis < len(axis_n_info):
        axis_vector = axis_n_info[axis]
        if axis_vector is not None and len(axis_vector) > order:
            axis_n_info[axis] = axis_vector[order:]

    if isinstance(axis_t_info, list) and axis < len(axis_t_info):
        axis_labels = axis_t_info[axis]
        if axis_labels is not None and len(axis_labels) > order:
            axis_t_info[axis] = axis_labels[order:]

    return axis_n_info, axis_t_info


def smoothing(
    X_cal: np.ndarray,
    method: str = 'moving_average',
    X_val: Optional[np.ndarray] = None,
    nway_flag: Optional[int] = None,
    direction: Optional[int] = None,
    window_size: int = 5,
    polyorder: int = 2
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply smoothing to data.
    
    Supports smoothing methods:
    - 'moving_average': Simple moving average filter
    - 'savitzky_golay': Savitzky-Golay filter (polynomial smoothing)
    
    Args:
        X_cal: Calibration data
        method: Smoothing method ('moving_average', 'savitzky_golay')
        X_val: Optional validation data
        nway_flag: Number of ways (if None, auto-determined)
        direction: Direction for multiway data
        window_size: Window/frame length for filter
        polyorder: Polynomial order for Savitzky-Golay filter
        
    Returns:
        Smoothed X_cal, or (X_cal, X_val) if X_val provided
    """
    if nway_flag is None:
        nway_flag = _determine_dimensionality(X_cal)
    
    if method == 'moving_average':
        X_cal = _moving_average_smooth(X_cal, nway_flag, direction, window_size)
        if X_val is not None:
            X_val = _moving_average_smooth(X_val, nway_flag, direction, window_size)
            return X_cal, X_val
        return X_cal
    
    elif method == 'savitzky_golay':
        X_cal = _savitzky_golay_smooth(X_cal, nway_flag, direction, window_size, polyorder)
        if X_val is not None:
            X_val = _savitzky_golay_smooth(X_val, nway_flag, direction, window_size, polyorder)
            return X_cal, X_val
        return X_cal
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def _moving_average_smooth(X: np.ndarray, nway_flag: int, direction: Optional[int] = None, window_size: int = 5) -> np.ndarray:
    """Apply moving average smoothing."""
    if nway_flag == 1 or X.ndim == 1:
        return uniform_filter1d(X, size=window_size, mode='nearest')
    
    if X.ndim == 2:
        return uniform_filter1d(X, size=window_size, axis=1, mode='nearest')
    
    axis = direction if direction is not None else X.ndim - 1
    return uniform_filter1d(X, size=window_size, axis=axis, mode='nearest')


def _savitzky_golay_smooth(X: np.ndarray, nway_flag: int, direction: Optional[int] = None, window_size: int = 5, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing."""
    # Ensure window_size is odd and valid
    if window_size % 2 == 0:
        window_size += 1
    if window_size <= polyorder:
        window_size = polyorder + 2
    
    if nway_flag == 1 or X.ndim == 1:
        return savgol_filter(X, window_size, polyorder)
    
    if X.ndim == 2:
        return savgol_filter(X, window_size, polyorder, axis=1)
    
    axis = direction if direction is not None else X.ndim - 1
    return savgol_filter(X, window_size, polyorder, axis=axis)


def center_and_scale(
    X_cal: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    scaling_method: str = "center",
    nway_flag: Optional[int] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply center/scaling preprocessing using calibration-derived statistics.
    
    This function handles both first-order (2D) and multiway data.
    When both X_cal and X_val are provided, both are processed using X_cal statistics.
    
    Args:
        X_cal: Calibration data (reference for centering/normalization)
        X_val: Optional validation data
        scaling_method: Preprocessing method. Supported values:
                   - 'none': no scaling
                   - 'center': mean centering only
                   - 'autoscale': mean centering + division by std
                   - 'frobenius': divide by Frobenius norm of X_cal
                   - 'range_0_1': min-max scale to [0, 1]
                   - 'range_-1_1': min-max scale to [-1, 1]
                   - 'variance': mean centering + division by variance
        nway_flag: Number of ways (if None, auto-determined from X_cal)
        
    Returns:
        Centered/normalized X_cal, or (X_cal, X_val) if X_val provided
        
    Examples:
        # 2D data: (samples, wavelengths) - center on wavelengths
        X_cal_centered = center_and_scale(X_cal, scaling_method="center")
        
        # Multiway data
        X_cal, X_val = center_and_scale(X_cal, X_val, scaling_method="autoscale",
                                            nway_flag=3)
    """

    if nway_flag is None:
        nway_flag = _determine_dimensionality(X_cal)


    method = str(scaling_method).strip().lower()
    
    axis = 0
    
    eps = 1e-10
    mean = np.mean(X_cal, axis=axis, keepdims=True)
    std = np.std(X_cal, axis=axis, keepdims=True)
    variance = np.var(X_cal, axis=axis, keepdims=True)
    min_val = np.min(X_cal, axis=axis, keepdims=True)
    max_val = np.max(X_cal, axis=axis, keepdims=True)
    range_val = max_val - min_val
    fro_norm = float(np.sqrt(np.sum(np.square(X_cal))))

    def _apply(arr: np.ndarray) -> np.ndarray:
        if method == "none":
            return arr
        if method == "center":
            return arr - mean
        if method == "autoscale":
            return (arr - mean) / (std + eps)
        if method == "frobenius":
            return arr / (fro_norm + eps)
        if method == "range_0_1":
            return (arr - min_val) / (range_val + eps)
        if method == "range_-1_1":
            return (2.0 * (arr - min_val) / (range_val + eps)) - 1.0
        if method == "variance":
            return (arr - mean) / (variance + eps)
        raise ValueError(f"Unknown scaling method: {method}")

    X_cal = _apply(X_cal.copy())

    if X_val is not None:
        X_val = _apply(X_val.copy())
        return X_cal, X_val
    
    return X_cal
