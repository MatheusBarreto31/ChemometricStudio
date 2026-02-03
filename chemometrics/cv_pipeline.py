"""
Cross-Validation Pipeline Module

Provides reusable CV infrastructure for all modeling functions.
Supports multiple CV strategies (K-Fold, Stratified, Time Series, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Any, Optional, Tuple, Iterator
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    RepeatedKFold,
    ShuffleSplit,
)


class FoldIndexedDict(dict):
    """
    Dictionary that allows indexed access to fold-segregated outputs.
    
    Example:
        results['loadings_cv_fold_0']  # Direct access via key
        results['loadings_cv'][0]      # Index-based access (calls the above)
    """
    def __getitem__(self, key):
        # Try direct key access first
        try:
            return super().__getitem__(key)
        except KeyError:
            # If key is an integer and we have fold keys, try fold access
            if isinstance(key, int):
                # This dict should contain 'parent_key' for context
                # We'll handle this via a wrapper below
                raise KeyError(f"Key {key} not found. Use string keys for fold access.")
            raise


class FoldSegregatedOutput:
    """
    Wrapper for segregated fold outputs that allows index-based access.
    
    Example:
        output = FoldSegregatedOutput({'fold_0': array1, 'fold_1': array2, ...})
        output[0]  # Returns array1 (fold_0)
        output[1]  # Returns array2 (fold_1)
        output['fold_0']  # Also works (direct string access)
        
    Can also iterate:
        for fold_array in output:
            print(fold_array.shape)
    """
    
    def __init__(self, fold_dict: Dict[str, Any]):
        """
        Args:
            fold_dict: Dictionary with keys like 'fold_0', 'fold_1', etc.
        """
        self._folds = fold_dict
        # Extract fold count
        self._fold_indices = sorted([
            int(k.split('_')[1]) for k in fold_dict.keys() 
            if k.startswith('fold_')
        ])
    
    def __getitem__(self, key):
        """Access by index or string key."""
        if isinstance(key, int):
            if key < 0 or key >= len(self._fold_indices):
                raise IndexError(f"Fold index {key} out of range [0, {len(self._fold_indices)-1}]")
            fold_key = f'fold_{self._fold_indices[key]}'
            return self._folds[fold_key]
        else:
            # String key access
            return self._folds[key]
    
    def __len__(self):
        """Return number of folds."""
        return len(self._fold_indices)
    
    def __iter__(self):
        """Iterate over fold arrays in order."""
        for idx in self._fold_indices:
            yield self._folds[f'fold_{idx}']
    
    def __repr__(self):
        return f"FoldSegregatedOutput(n_folds={len(self)}, keys={list(self._folds.keys())})"
    
    def as_array(self):
        """Stack all fold arrays into single array."""
        return np.array([self[i] for i in range(len(self))])
    
    def as_dict(self):
        """Get the underlying fold dictionary."""
        return self._folds.copy()



@dataclass
class CVConfig:
    """Cross-validation configuration object (serializable)."""

    use_cv: bool
    cv_strategy: str
    n_splits: int
    random_state: Optional[int] = 42
    shuffle: bool = True
    window_size: Optional[int] = None  # For moving_window and venetian_windows strategies
    n_repeats: Optional[int] = None  # For repeated_kfold strategy (default: 10)
    test_size: Optional[float] = None  # For shuffle_split strategy (default: 0.2)
    output_metrics: Optional[List[str]] = None
    capture_outputs: Optional[List[str]] = None  # NEW: outputs to preserve from each fold

    def __post_init__(self):
        if self.output_metrics is None:
            self.output_metrics = ["rmse", "r2"]
        if self.capture_outputs is None:
            self.capture_outputs = []
        # Set defaults for strategy-specific parameters
        if self.n_repeats is None:
            self.n_repeats = 10
        if self.test_size is None:
            self.test_size = 0.2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CVConfig":
        """Reconstruct from dict."""
        return cls(**data)

    def is_enabled(self) -> bool:
        """Check if CV is actually enabled."""
        return self.use_cv


class CVSplitter(ABC):
    """Base class for CV strategies."""

    @abstractmethod
    def get_splits(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, test_idx) tuples.

        Args:
            X: Feature matrix (used to determine n_samples)
            y: Target vector (optional, for stratified splits)
            groups: Group labels (optional, for group k-fold)

        Yields:
            Tuples of (train_indices, test_indices)
        """
        pass


class KFoldSplitter(CVSplitter):
    """K-Fold cross-validation splitter."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = 42):
        self.cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def get_splits(self, X, y=None, groups=None):
        return self.cv.split(X, y, groups)


class StratifiedKFoldSplitter(CVSplitter):
    """Stratified K-Fold (preserves class proportions per fold)."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = 42):
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def get_splits(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("StratifiedKFold requires y (target vector)")
        return self.cv.split(X, y)


class TimeSeriesSplitter(CVSplitter):
    """Time Series CV (forward chaining - no lookahead bias)."""

    def __init__(self, n_splits: int = 5):
        self.cv = TimeSeriesSplit(n_splits=n_splits)

    def get_splits(self, X, y=None, groups=None):
        return self.cv.split(X, y, groups)


class RepeatedKFoldSplitter(CVSplitter):
    """Repeated K-Fold (multiple iterations of KFold with different splits)."""

    def __init__(
        self, n_splits: int = 5, n_repeats: int = 10, random_state: Optional[int] = 42
    ):
        self.cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    def get_splits(self, X, y=None, groups=None):
        return self.cv.split(X, y, groups)


class ShuffleSplitSplitter(CVSplitter):
    """Shuffle Split (random train/test splits, useful for large datasets)."""

    def __init__(self, n_splits: int = 10, test_size: float = 0.2, random_state: Optional[int] = 42):
        self.cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    def get_splits(self, X, y=None, groups=None):
        return self.cv.split(X, y, groups)


class VenetianWindowsSplitter(CVSplitter):
    """
    Venetian Windows CV: Creates k overlapping windows that cover the full dataset.
    
    Useful for continuous data with systematic patterns. Each window contains roughly
    1/k of the samples, but windows overlap significantly.
    
    Example with n_samples=100, n_splits=5:
    - Window 0: indices 0-50 (test), 0-100 (train)
    - Window 1: indices 20-70 (test), 0-100 (train)
    - Window 2: indices 40-90 (test), 0-100 (train)
    - Window 3: indices 60-100 (test), 0-100 (train - includes some test indices)
    - Window 4: indices 80-100 (test), 0-100 (train - includes some test indices)
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def get_splits(self, X, y=None, groups=None):
        n_samples = len(X)
        window_size = max(1, n_samples // self.n_splits)
        
        indices = np.arange(n_samples)
        if self.shuffle and self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
        elif self.shuffle:
            np.random.shuffle(indices)
        
        for fold_idx in range(self.n_splits):
            test_start = fold_idx * window_size
            test_end = min(test_start + window_size, n_samples)
            
            # All samples are training (to include overlapping context)
            train_idx = indices
            test_idx = indices[test_start:test_end]
            
            yield train_idx, test_idx


class MovingWindowSplitter(CVSplitter):
    """
    Moving Window CV: Progressive sliding windows for time-dependent or sequential data.
    
    Each fold uses a different window position, with windows moving through the dataset.
    Useful for time series where you want to test generalization as the window moves forward.
    
    Example with n_samples=100, n_splits=5, window_size=50:
    - Fold 0: train indices 0-49, test indices 50-59
    - Fold 1: train indices 10-59, test indices 60-69
    - Fold 2: train indices 20-69, test indices 70-79
    - Fold 3: train indices 30-79, test indices 80-89
    - Fold 4: train indices 40-89, test indices 90-99
    """
    
    def __init__(self, n_splits: int = 5, window_size: Optional[int] = None, 
                 shuffle: bool = False, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.window_size = window_size
        self.shuffle = shuffle
        self.random_state = random_state
    
    def get_splits(self, X, y=None, groups=None):
        n_samples = len(X)
        
        # Default: window_size is total samples / n_splits
        if self.window_size is None:
            self.window_size = max(1, n_samples // (self.n_splits + 1))
        
        indices = np.arange(n_samples)
        if self.shuffle and self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
        elif self.shuffle:
            np.random.shuffle(indices)
        
        step = max(1, (n_samples - self.window_size) // self.n_splits)
        
        for fold_idx in range(self.n_splits):
            test_start = fold_idx * step
            test_end = min(test_start + self.window_size, n_samples)
            
            train_idx = indices
            test_idx = indices[test_start:test_end]
            
            yield train_idx, test_idx


class LOOCVSplitter(CVSplitter):
    """
    Leave-One-Out Cross-Validation (LOOCV).
    
    Each sample is left out once; the model trains on n-1 samples and tests on 1.
    Most thorough but computationally expensive (n iterations for n samples).
    Does not support shuffling (splits are deterministic).
    """
    
    def __init__(self, n_splits: Optional[int] = None, shuffle: bool = False, 
                 random_state: Optional[int] = None):
        # LOOCV ignores n_splits and shuffle parameters
        self.shuffle = False
        self.n_splits_param = n_splits  # Store for reference
    
    def get_splits(self, X, y=None, groups=None):
        n_samples = len(X)
        
        for fold_idx in range(n_samples):
            # Test: single sample at fold_idx
            test_idx = np.array([fold_idx])
            # Train: all except fold_idx
            train_idx = np.concatenate([np.arange(fold_idx), np.arange(fold_idx + 1, n_samples)])
            
            yield train_idx, test_idx


class BootstrapSplitter(CVSplitter):
    """
    Bootstrap Cross-Validation: Random sampling with replacement.
    
    Each iteration:
    - Training set: n random samples drawn WITH replacement from all samples
    - Test set: Samples NOT selected in that iteration (Out-of-Bag / OOB)
    
    Good for small datasets and provides robust error estimates.
    Useful when you want to assess variance of estimators.
    
    Example with n_samples=100, n_splits=10:
    - Each fold samples 100 indices randomly WITH replacement
    - Each fold's test set = indices not in that fold's bootstrap sample
    - Test set size varies (typically ~36.8% of original for large n)
    """
    
    def __init__(self, n_splits: int = 10, random_state: Optional[int] = 42):
        self.n_splits = n_splits
        self.random_state = random_state
    
    def get_splits(self, X, y=None, groups=None):
        n_samples = len(X)
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random.RandomState()
        
        for fold_idx in range(self.n_splits):
            # Bootstrap sample: draw n_samples WITH replacement
            train_idx = rng.choice(n_samples, size=n_samples, replace=True)
            
            # Out-of-Bag (OOB): samples not in bootstrap sample
            train_set = set(train_idx)
            test_idx = np.array([i for i in range(n_samples) if i not in train_set])
            
            # Only yield if OOB set is non-empty
            if len(test_idx) > 0:
                yield train_idx, test_idx


class CVPipeline:
    """
    Generic cross-validation pipeline that wraps any function.

    Usage:
        config = CVConfig(use_cv=True, cv_strategy='kfold', n_splits=5, ...)
        pipeline = CVPipeline(config)
        results = pipeline.run(func, X_cal=X_cal, Y_cal=Y_cal, ...)
    """

    def __init__(self, config: CVConfig):
        self.config = config
        self.splitter = self._get_splitter()
        self.results = {}

    def _get_splitter(self) -> CVSplitter:
        """Factory method to create appropriate splitter."""
        strategy_map = {
            "kfold": lambda cfg: KFoldSplitter(
                n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state
            ),
            "stratified_kfold": lambda cfg: StratifiedKFoldSplitter(
                n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state
            ),
            "timeseries": lambda cfg: TimeSeriesSplitter(n_splits=cfg.n_splits),
            "repeated_kfold": lambda cfg: RepeatedKFoldSplitter(
                n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.random_state
            ),
            "shuffle_split": lambda cfg: ShuffleSplitSplitter(
                n_splits=cfg.n_splits, test_size=cfg.test_size, random_state=cfg.random_state
            ),
            "venetian_windows": lambda cfg: VenetianWindowsSplitter(
                n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state
            ),
            "moving_window": lambda cfg: MovingWindowSplitter(
                n_splits=cfg.n_splits, window_size=cfg.window_size, shuffle=cfg.shuffle, random_state=cfg.random_state
            ),
            "loocv": lambda cfg: LOOCVSplitter(
                n_splits=cfg.n_splits, shuffle=False, random_state=cfg.random_state
            ),
            "bootstrap": lambda cfg: BootstrapSplitter(
                n_splits=cfg.n_splits, random_state=cfg.random_state
            ),
        }

        splitter_factory = strategy_map.get(self.config.cv_strategy)
        if not splitter_factory:
            raise ValueError(f"Unknown CV strategy: {self.config.cv_strategy}")

        return splitter_factory(self.config)

    def _split_array_on_samples(self, data: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split array along sample dimension (axis 0), preserving all other dimensions.
        
        Dimension-agnostic: Handles 1D, 2D, 3D, 4D, 5D, 6D, and higher dimensional data 
        correctly. No limit on number of dimensions.
        
        Args:
            data: Array of any shape; axis 0 is samples, remaining axes are preserved
            train_idx: Indices of training samples
            test_idx: Indices of test samples
            
        Returns:
            Tuple of (train_data, test_data) with same number of dimensions as input
            
        Examples:
            1D:  (100,) -> (80,), (20,)
            2D:  (100, 10) -> (80, 10), (20, 10)  [univariate]
            3D:  (100, 10, 50) -> (80, 10, 50), (20, 10, 50)  [multiway]
            4D:  (100, 10, 50, 3) -> (80, 10, 50, 3), (20, 10, 50, 3)  [multiway+aux]
            5D:  (100, 10, 50, 3, 2) -> (80, 10, 50, 3, 2), (20, 10, 50, 3, 2)
            6D+: (100, ...) -> (80, ...), (20, ...)  [any number of dimensions]
        """
        return data[train_idx], data[test_idx]

    def _reconstruct_from_folds(self, fold_outputs_list: List[Tuple[np.ndarray, np.ndarray]], 
                                 n_samples: int, test_indices_list: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct full-size array from fold outputs.
        
        Takes outputs from each fold's test set and places them back in original positions.
        Works differently based on output type:
        - Sample-based (axis 0 = samples): Reconstructs full-size array by positioning fold outputs
        - Non-sample-based (axis 0 ≠ samples): Returns segregated by fold (fold_0, fold_1, etc.)
        
        Args:
            fold_outputs_list: List of fold outputs (one per fold)
            n_samples: Total number of samples
            test_indices_list: List of test indices for each fold
            
        Returns:
            - If sample-based: Full-size reconstructed array
            - If non-sample-based: Dict with keys 'fold_0', 'fold_1', etc.
        """
        # Initialize output array with same shape as first fold output, but full n_samples
        first_output = fold_outputs_list[0]
        if isinstance(first_output, np.ndarray):
            # Check if first dimension matches test set size (i.e., samples in axis 0)
            if first_output.shape[0] == len(test_indices_list[0]):
                # Standard case: axis 0 is samples
                output_shape = (n_samples,) + first_output.shape[1:]
                reconstructed = np.zeros(output_shape, dtype=first_output.dtype)
                
                # Place each fold's output in its original positions
                for fold_output, test_idx in zip(fold_outputs_list, test_indices_list):
                    reconstructed[test_idx] = fold_output
                
                return reconstructed
            else:
                # Non-sample axis case (e.g., loadings, components, weights)
                # Return segregated by fold instead of averaging
                segregated = {}
                for fold_idx, fold_output in enumerate(fold_outputs_list):
                    segregated[f"fold_{fold_idx}"] = fold_output
                return segregated
        else:
            raise ValueError("Fold outputs must be numpy arrays for reconstruction")

    def run(
        self, func: Callable, reference_output_key: Optional[str] = None, 
        capture_output_keys: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute function across CV folds.

        Supports univariate (1D/2D) and multiway (3D/4D/etc) array data. Splitting 
        always occurs along the sample dimension (axis 0), preserving all other 
        dimensions.

        Args:
            func: callable that accepts **kwargs including fold=<int> parameter
                  and returns dict with metric keys specified in config.output_metrics
            reference_output_key: If specified, run single fit first, then CV compares 
                                fold outputs against single fit outputs. Metrics computed
                                as differences between fold outputs and single-fit reference.
                                (overrides config.output_metrics for metric computation)
            capture_output_keys: List of output keys to capture and reconstruct from folds.
                                If None, uses config.capture_outputs. Can be same or different
                                from reference_output_key.
            **kwargs: parameters for func; first array-like parameter will be used for splitting.
                      Arrays can be any shape (1D, 2D, 3D, 4D, etc.); splitting on axis 0.

        Returns:
            Dictionary with:
            - If reference_output_key specified:
              - metric (scalar): Overall metric comparing reconstructed vs single fit
              - metric_per_fold (list): Metric per fold
              - {output_key}_cv (array): Reconstructed full-size output from CV folds
              - {output_key}_single (array): Single fit output as reference
              - For all capture_output_keys: {output_key}_cv, {output_key}_single
            - Otherwise (traditional CV):
              - Aggregated metrics: {metric_mean, metric_std, metric_min, metric_max, metric_folds}
              - Captured outputs per fold: {output_fold_0, output_fold_1, ...}
        
        Example - Assess stability of PCA scores across folds:
            # Setup: Configure CV to capture scores
            cv_config = CVConfig(
                use_cv=True, 
                cv_strategy='kfold', 
                n_splits=5,
                output_metrics=['rmse'],
                capture_outputs=['scores']  # NEW: Capture PCA scores per fold
            )
            
            # Run PCA through CV
            pipeline = CVPipeline(cv_config)
            results = pipeline.run(_pca_fit, X=X)
            
            # Results now contains:
            # - 'rmse_folds': [0.88, 0.92, 0.85, 0.87, 0.90]  (metrics per fold)
            # - 'rmse_mean': 0.884, 'rmse_std': 0.028  (aggregated)
            # - 'scores_fold_0': array of shape (20, 3)  (scores from fold 0)
            # - 'scores_fold_1': array of shape (20, 3)  (scores from fold 1)
            # - ... (one per fold)
            
            # Assess stability: Check if scores vary by fold
            import numpy as np
            scores_list = [results[f'scores_fold_{i}'] for i in range(5)]
            std_across_folds = np.std(scores_list, axis=0)  # Std per score column
            
        Multiway data support:
            X: (n_samples, n_wavelengths, n_time_points)
            Y: (n_samples, n_responses)
            Both split to:
            X_train: (n_train, n_wavelengths, n_time_points)
            X_test: (n_test, n_wavelengths, n_time_points)
        """
        # Determine which outputs to capture
        capture_keys = capture_output_keys if capture_output_keys is not None else (self.config.capture_outputs or [])
        
        # Find first array-like input (X) and its size
        X = None
        y = None
        n_samples = None
        for key, val in kwargs.items():
            if isinstance(val, (np.ndarray, pd.DataFrame)):
                if X is None:
                    X = val
                    n_samples = len(val)
                elif key.lower() in ["y", "y_cal"]:
                    y = val
                    break

        if X is None:
            raise ValueError("No array-like input found in kwargs for splitting")

        # ==================== MODE 1: Single-Fit Reference ====================
        if reference_output_key is not None:
            # Run single fit on full data to get reference
            single_fit_kwargs = {"fold": -1}  # Special fold marker for single fit
            for key, data in kwargs.items():
                if isinstance(data, np.ndarray):
                    single_fit_kwargs[f"{key}_train"] = data
                    single_fit_kwargs[f"{key}_test"] = data
                elif isinstance(data, pd.DataFrame):
                    single_fit_kwargs[f"{key}_train"] = data
                    single_fit_kwargs[f"{key}_test"] = data
                else:
                    single_fit_kwargs[key] = data
            
            # Get single fit output
            single_fit_result = func(**single_fit_kwargs)
            if reference_output_key not in single_fit_result:
                raise ValueError(f"Single fit result does not contain '{reference_output_key}'")
            
            reference_output = single_fit_result[reference_output_key]
            
            # Initialize storage for fold outputs and metrics
            fold_metrics = []
            fold_outputs_dict = {key: [] for key in capture_keys}
            test_indices_list = []
            
            # Run CV folds
            for fold_idx, (train_idx, test_idx) in enumerate(self.splitter.get_splits(X, y)):
                fold_kwargs = {"fold": fold_idx}
                test_indices_list.append(test_idx)
                
                # Split data
                for key, data in kwargs.items():
                    if isinstance(data, np.ndarray):
                        train_data, test_data = self._split_array_on_samples(data, train_idx, test_idx)
                        fold_kwargs[f"{key}_train"] = train_data
                        fold_kwargs[f"{key}_test"] = test_data
                    elif isinstance(data, pd.DataFrame):
                        fold_kwargs[f"{key}_train"] = data.iloc[train_idx]
                        fold_kwargs[f"{key}_test"] = data.iloc[test_idx]
                    else:
                        fold_kwargs[key] = data
                
                # Run function
                fold_result = func(**fold_kwargs)
                
                if fold_result is not None:
                    # Get fold output for metric computation (only the reference output)
                    if reference_output_key in fold_result:
                        fold_output = fold_result[reference_output_key]
                        
                        # Compute metric: compare fold output against reference (same indices)
                        fold_reference = reference_output[test_idx]
                        
                        # Metric: RMSE between fold and reference
                        metric = float(np.sqrt(np.mean((fold_output - fold_reference) ** 2)))
                        fold_metrics.append(metric)
                    
                    # Capture specified outputs
                    for output_key in capture_keys:
                        if output_key in fold_result:
                            fold_outputs_dict[output_key].append(fold_result[output_key])
            
            # Build results for single-fit reference mode
            aggregated = {}
            
            # Overall metric and per-fold metrics
            if fold_metrics:
                aggregated[f"{reference_output_key}_rmse"] = float(np.mean(fold_metrics))
                aggregated[f"{reference_output_key}_rmse_per_fold"] = fold_metrics
                aggregated[f"{reference_output_key}_rmse_std"] = float(np.std(fold_metrics))
            
            # Reconstruct full-size arrays from fold outputs
            for output_key, fold_outputs_list in fold_outputs_dict.items():
                if fold_outputs_list:
                    reconstructed = self._reconstruct_from_folds(
                        fold_outputs_list, n_samples, test_indices_list
                    )
                    
                    # Handle both sample-based (full array) and non-sample-based (segregated dict) outputs
                    if isinstance(reconstructed, dict):
                        # Non-sample-based: segregated by fold
                        # Wrap in FoldSegregatedOutput for index-based access
                        fold_output_dict = {k.replace('fold_', f'fold_'): v 
                                           for k, v in reconstructed.items()}
                        aggregated[f"{output_key}_cv"] = FoldSegregatedOutput(fold_output_dict)
                    else:
                        # Sample-based: full reconstructed array
                        aggregated[f"{output_key}_cv"] = reconstructed
                    
                    aggregated[f"{output_key}_single"] = reference_output if output_key == reference_output_key else single_fit_result.get(output_key)
            
            aggregated["n_folds"] = len(test_indices_list)
            
            self.results = aggregated
            return aggregated
        
        # ==================== MODE 2: Traditional CV (no reference) ====================
        else:
            output_keys = self.config.output_metrics or []
            if not output_keys:
                raise ValueError("config.output_metrics must be specified")

            fold_results = {key: [] for key in output_keys}
            fold_outputs = {}
            fold_count = 0

            # Run across folds
            for fold_idx, (train_idx, test_idx) in enumerate(self.splitter.get_splits(X, y)):
                fold_kwargs = {"fold": fold_idx}

                # Split each array-like input
                for key, data in kwargs.items():
                    if isinstance(data, np.ndarray):
                        train_data, test_data = self._split_array_on_samples(data, train_idx, test_idx)
                        fold_kwargs[f"{key}_train"] = train_data
                        fold_kwargs[f"{key}_test"] = test_data
                    elif isinstance(data, pd.DataFrame):
                        fold_kwargs[f"{key}_train"] = data.iloc[train_idx]
                        fold_kwargs[f"{key}_test"] = data.iloc[test_idx]
                    else:
                        fold_kwargs[key] = data

                # Call function
                fold_output = func(**fold_kwargs)

                # Collect results
                if fold_output is not None:
                    # Collect metrics
                    for key in output_keys:
                        if key in fold_output:
                            fold_results[key].append(fold_output[key])
                    
                    # Capture specified non-metric outputs per fold
                    if self.config.capture_outputs:
                        for output_key in self.config.capture_outputs:
                            if output_key in fold_output:
                                fold_outputs[f"{output_key}_fold_{fold_idx}"] = fold_output[output_key]

                fold_count += 1

            # Aggregate metrics
            aggregated = {}
            for key, values in fold_results.items():
                if values:
                    aggregated[f"{key}_folds"] = values
                    aggregated[f"{key}_mean"] = float(np.mean(values))
                    aggregated[f"{key}_std"] = float(np.std(values))
                    aggregated[f"{key}_min"] = float(np.min(values))
                    aggregated[f"{key}_max"] = float(np.max(values))

            aggregated["n_folds"] = fold_count
            
            # Add captured outputs to results
            aggregated.update(fold_outputs)

            self.results = aggregated
            return aggregated


def cv_configuration(
    use_cv: bool,
    cv_strategy: str = "kfold",
    n_splits: int = 5,
    random_state: Optional[int] = 42,
    shuffle: bool = True,
    window_size: Optional[int] = None,
    n_repeats: Optional[int] = None,
    test_size: Optional[float] = None,
    output_metrics: Optional[List[str]] = None,
    capture_outputs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create a CV configuration object to be routed to modeling functions.

    Args:
        use_cv: Enable/disable CV
        cv_strategy: 'kfold', 'stratified_kfold', 'timeseries', 'repeated_kfold', 'shuffle_split',
                     'venetian_windows', 'moving_window', 'loocv', 'bootstrap'
        n_splits: Number of folds/splits (ignored for LOOCV; auto-set to n_samples)
        random_state: Seed for reproducibility
        shuffle: Randomize before splitting (applies to kfold, stratified_kfold, repeated_kfold,
                 venetian_windows, moving_window; not applicable to timeseries, loocv, bootstrap)
        window_size: Optional window size for moving_window and venetian_windows strategies.
                    If None, auto-calculated as n_samples // n_splits
        n_repeats: Number of repetitions for repeated_kfold strategy (default: 10)
        test_size: Test set proportion for shuffle_split strategy (default: 0.2 = 20%)
        output_metrics: Which metrics to compute and aggregate (e.g., ['rmse', 'r2'])
        capture_outputs: Which function outputs to capture per-fold for stability assessment
                        (e.g., ['scores', 'loadings'] for PCA). Captured outputs appear
                        in results as 'output_name_fold_0', 'output_name_fold_1', etc.

    Returns:
        dict with 'cv_config' key containing CVConfig object
    """
    config = CVConfig(
        use_cv=use_cv,
        cv_strategy=cv_strategy,
        n_splits=n_splits,
        random_state=random_state,
        shuffle=shuffle,
        window_size=window_size,
        n_repeats=n_repeats,
        test_size=test_size,
        output_metrics=output_metrics or ["rmse", "r2"],
        capture_outputs=capture_outputs or [],
    )

    return {"cv_config": config}
