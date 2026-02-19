## Data processing functions for ChemometricsTool

from typing import Tuple, Optional, Union
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
    window_size: int = 5
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply baseline correction to spectroscopic data.
    
    Supports multiple baseline correction methods:
    - 'msc': Multiplicative Scatter Correction
    - 'svn': Standard Normal Variate
    - 'moving_average': Moving Average baseline subtraction
    
    Args:
        X_cal: Calibration data (reference dataset)
        method: Baseline correction method ('msc', 'svn', 'moving_average')
        X_val: Optional validation data (will be corrected using X_cal reference)
        nway_flag: Number of ways (if None, auto-determined from X_cal)
        direction: Direction for multiway data (0, 1, 2, etc.). If None, applied on last axis
        window_size: Window size for moving average method
        
    Returns:
        Baseline-corrected X_cal, or (X_cal, X_val) if X_val provided
    """
    if nway_flag is None:
        nway_flag = _determine_dimensionality(X_cal)
    
    if method == 'msc':
        X_cal = _msc(X_cal, nway_flag, direction)
        if X_val is not None:
            X_val = _msc(X_val, nway_flag, direction, X_cal)
            return X_cal, X_val
        return X_cal
    
    elif method == 'svn':
        X_cal = _svn(X_cal, nway_flag, direction)
        if X_val is not None:
            X_val = _svn(X_val, nway_flag, direction)
            return X_cal, X_val
        return X_cal
    
    elif method == 'moving_average':
        X_cal = _moving_average_baseline(X_cal, nway_flag, direction, window_size)
        if X_val is not None:
            X_val = _moving_average_baseline(X_val, nway_flag, direction, window_size)
            return X_cal, X_val
        return X_cal
    
    else:
        raise ValueError(f"Unknown baseline correction method: {method}")


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
    if nway_flag == 1 or X.ndim == 1:
        reference_spec = reference if reference is not None else np.mean(X)
        return X / reference_spec if reference is not None else X / np.mean(X)
    
    # For 2D and multiway, apply along last axis (spectra)
    if reference is None:
        reference = np.mean(X, axis=0)
    
    X_corrected = X / (reference + 1e-10)
    return X_corrected


def _svn(X: np.ndarray, nway_flag: int, direction: Optional[int] = None) -> np.ndarray:
    """
    Standard Normal Variate (SNV) normalization.
    
    Args:
        X: Input data
        nway_flag: Number of dimensions
        direction: Direction for multiway data
        
    Returns:
        SNV-normalized data
    """
    if nway_flag == 1 or X.ndim == 1:
        mean = np.mean(X)
        std = np.std(X)
        return (X - mean) / (std + 1e-10)
    
    # For 2D and multiway, apply along last axis
    mean = np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True)
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
    if nway_flag == 1 or X.ndim == 1:
        baseline = uniform_filter1d(X, size=window_size, mode='nearest')
        return X - baseline
    
    # For 2D, apply along last axis
    if X.ndim == 2:
        baseline = uniform_filter1d(X, size=window_size, axis=1, mode='nearest')
        return X - baseline
    
    # For multiway, apply along specified direction or last axis
    axis = direction if direction is not None else X.ndim - 1
    baseline = uniform_filter1d(X, size=window_size, axis=axis, mode='nearest')
    return X - baseline


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


def center_and_normalize(
    X_cal: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    center: bool = True,
    normalize: bool = False,
    nway_flag: Optional[int] = None,
    direction: Optional[int] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Center and/or normalize data on the average of calibration samples.
    
    This function handles both first-order (2D) and multiway data.
    When both X_cal and X_val are provided, both are processed using X_cal statistics.
    
    Args:
        X_cal: Calibration data (reference for centering/normalization)
        X_val: Optional validation data
        center: If True, subtract mean from X_cal
        normalize: If True, divide by standard deviation of X_cal
        nway_flag: Number of ways (if None, auto-determined from X_cal)
        direction: Direction for multiway data centering:
                   - None or -1: Apply along last axis (features/wavelengths)
                   - 0: Apply along first axis (samples)
                   - Other integers: Apply along specified axis
        
    Returns:
        Centered/normalized X_cal, or (X_cal, X_val) if X_val provided
        
    Examples:
        # 2D data: (samples, wavelengths) - center on wavelengths
        X_cal_centered = center_and_normalize(X_cal, center=True, normalize=False)
        
        # Multiway data with direction
        X_cal, X_val = center_and_normalize(X_cal, X_val, center=True, normalize=True, 
                                            nway_flag=3, direction=2)
    """
    if isinstance(direction, str):
        stripped = direction.strip()
        if stripped == "":
            direction = None
        else:
            direction = int(stripped)
    elif isinstance(direction, float):
        direction = int(direction)

    if nway_flag is None:
        nway_flag = _determine_dimensionality(X_cal)
    
    # Determine axis for centering
    # Default: center on samples (axis 0) for features; for multiway, respect direction
    if direction is None or direction == -1:
        # Default: center along axis 0 (samples) - compute mean across samples for each feature
        axis = 0
    else:
        axis = direction
    
    # Calculate statistics from X_cal
    mean = np.mean(X_cal, axis=axis, keepdims=True)
    std = np.std(X_cal, axis=axis, keepdims=True) if normalize else None
    
    # Process X_cal
    X_cal = X_cal.copy()
    if center:
        X_cal = X_cal - mean
    if normalize:
        X_cal = X_cal / (std + 1e-10)
    
    # Process X_val using X_cal statistics
    if X_val is not None:
        X_val = X_val.copy()
        if center:
            X_val = X_val - mean
        if normalize:
            X_val = X_val / (std + 1e-10)
        return X_cal, X_val
    
    return X_cal
