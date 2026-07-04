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
    LeaveOneOut,
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
    window_size: Optional[int] = None  # For moving_window strategy
    n_repeats: Optional[int] = None  # For repeated_kfold strategy (default: 10)
    test_size: Optional[float] = None  # For shuffle_split strategy (default: 0.2)
    stratify_layer: int = 1  # 1-based class layer index for stratified_kfold
    output_metrics: Optional[List[str]] = None  # Which metrics to compute (e.g., ['rmse', 'r2'])
    capture_outputs: Optional[List[str]] = None  # Outputs to preserve from each fold
    reference_input_key: Optional[str] = None  # Input key to use as reference (e.g., 'Y_cal')
    reference_output_key: Optional[str] = None  # Output key for single-fit reference mode
    comparison_output_key: Optional[str] = None  # Output key to compare against reference

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
        try:
            self.stratify_layer = max(1, int(self.stratify_layer))
        except Exception:
            self.stratify_layer = 1

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
        effective_random_state = random_state if shuffle else None
        self.cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=effective_random_state)

    def get_splits(self, X, y=None, groups=None):
        return self.cv.split(X, y, groups)


class StratifiedKFoldSplitter(CVSplitter):
    """Stratified K-Fold (preserves class proportions per fold)."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = 42):
        effective_random_state = random_state if shuffle else None
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=effective_random_state)

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


class VenetianBlindsSplitter(CVSplitter):
    """
    Venetian Blinds CV.

    Splits samples by periodic indexing: fold i uses every k-th sample starting at i
    as test set, and all remaining samples as training set.

    This avoids train/test overlap and produces k disjoint test partitions.

    Example with n_samples=12, n_splits=4:
    - Fold 0 test: [0, 4, 8]
    - Fold 1 test: [1, 5, 9]
    - Fold 2 test: [2, 6, 10]
    - Fold 3 test: [3, 7, 11]
    """
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
    
    def get_splits(self, X, y=None, groups=None):
        n_samples = len(X)
        if self.n_splits < 2:
            raise ValueError("venetian_blinds requires n_splits >= 2")
        if n_samples < self.n_splits:
            raise ValueError(
                f"venetian_blinds requires n_samples >= n_splits (got n_samples={n_samples}, n_splits={self.n_splits})"
            )
        
        indices = np.arange(n_samples)
        
        for fold_idx in range(self.n_splits):
            test_idx = indices[fold_idx::self.n_splits]
            if len(test_idx) == 0:
                continue

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[fold_idx::self.n_splits] = False
            train_idx = indices[train_mask]
            
            yield train_idx, test_idx


class MovingWindowSplitter(CVSplitter):
    """
    Moving Window CV for sequential data.

    Each fold trains on a contiguous fixed-size window and tests on the immediately
    following block, then both windows slide forward.

    Train/test sets never overlap.
    """
    
    def __init__(self, n_splits: int = 5, window_size: Optional[int] = None):
        self.n_splits = n_splits
        self.window_size = window_size
    
    def get_splits(self, X, y=None, groups=None):
        n_samples = len(X)
        if self.n_splits < 1:
            raise ValueError("moving_window requires n_splits >= 1")
        
        # Default: training window is roughly half of each fold span
        window_size = self.window_size
        if window_size is None:
            window_size = max(1, n_samples // (self.n_splits + 2))
        if window_size >= n_samples:
            raise ValueError(
                f"moving_window requires window_size < n_samples (got window_size={window_size}, n_samples={n_samples})"
            )
        
        indices = np.arange(n_samples)

        remaining_after_train = n_samples - window_size
        test_block_size = max(1, remaining_after_train // (self.n_splits + 1))
        max_train_start = max(0, n_samples - window_size - test_block_size)
        step = max(1, max_train_start // max(1, self.n_splits - 1))
        
        for fold_idx in range(self.n_splits):
            train_start = min(fold_idx * step, max_train_start)
            train_end = train_start + window_size
            test_start = train_end
            test_end = min(test_start + test_block_size, n_samples)

            if test_start >= n_samples or test_end <= test_start:
                continue

            train_idx = indices[train_start:train_end]
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
        # LOOCV ignores n_splits/shuffle/random_state by definition.
        self.cv = LeaveOneOut()
    
    def get_splits(self, X, y=None, groups=None):
        return self.cv.split(X, y, groups)


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
        config = CVConfig(use_cv=True, cv_strategy='loocv', n_splits=5, ...)
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
            "venetian_blinds": lambda cfg: VenetianBlindsSplitter(n_splits=cfg.n_splits),
            "moving_window": lambda cfg: MovingWindowSplitter(
                n_splits=cfg.n_splits, window_size=cfg.window_size
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
        self, func: Callable, 
        reference_input_key: Optional[str] = None,
        reference_output_key: Optional[str] = None, 
        comparison_output_key: Optional[str] = None,
        capture_output_keys: Optional[List[str]] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute function across CV folds with unified reference-based metrics.

        Supports univariate (1D/2D) and multiway (3D/4D/etc) array data. Splitting 
        always occurs along the sample dimension (axis 0), preserving all other 
        dimensions.
        
        Reference Modes (mutually exclusive):
        1. Input Reference Mode (reference_input_key): Uses an input array (e.g., Y_cal)
           as the ground truth. Each fold's output is compared against the corresponding
           portion of this input. This is the standard CV use case.
           
        2. Output Reference Mode (reference_output_key): Runs single fit on full data first,
           then compares each fold's output against the single-fit output. Useful for
           assessing model stability across folds.

        Args:
            func: callable that accepts **kwargs including fold=<int> parameter
                  and returns dict with output arrays
            reference_input_key: Input kwarg key to use as reference (e.g., 'Y_cal').
                                The reference array is indexed by test_idx for each fold.
                                Takes precedence over reference_output_key.
            reference_output_key: Output key from single fit to use as reference.
                                Only used if reference_input_key is None.
            comparison_output_key: Which output key from function to compare against reference.
                                  Required when using reference_input_key or reference_output_key.
                                  For matrix outputs (e.g., PCA scores), comparison is element-wise.
            capture_output_keys: List of output keys to capture and reconstruct from folds.
                                If None, uses config.capture_outputs.
            **kwargs: parameters for func; array-like parameters will be split.
                      Arrays can be any shape (1D, 2D, 3D, 4D, etc.); splitting on axis 0.

        Returns:
            Dictionary with:
            - {metric}_mean, {metric}_std, {metric}_min, {metric}_max: Aggregated metrics
            - {metric}_folds: List of per-fold metrics
            - {output_key}_cv: Reconstructed full-size output from CV folds (sample-based)
            - {output_key}_cv: FoldSegregatedOutput for non-sample outputs (e.g., loadings)
            - reference: The reference array used (input or single-fit output)
            - n_folds: Number of folds executed
        
        Example 1 - Standard CV with input reference (compare predictions to actual Y):
            cv_config = CVConfig(use_cv=True, cv_strategy='kfold', n_splits=5)
            pipeline = CVPipeline(cv_config)
            results = pipeline.run(
                my_model_func,
                X_cal=X, Y_cal=Y,
                reference_input_key='Y_cal',      # Use actual Y as reference
                comparison_output_key='y_pred',   # Compare predictions to Y
                capture_output_keys=['y_pred']
            )
            # Results: rmse_mean, rmse_folds, y_pred_cv (reconstructed), reference
            
        Example 2 - Stability assessment with output reference (PCA scores):
            results = pipeline.run(
                pca_func,
                X=X,
                reference_output_key='scores',    # Single-fit scores as reference
                comparison_output_key='scores',   # Compare fold scores to single-fit
                capture_output_keys=['scores', 'loadings']
            )
            # Results: rmse_mean (stability metric), scores_cv, loadings_cv (segregated)
            
        Example 3 - Matrix reference (multi-response Y):
            # Y_cal shape: (100, 3) - 3 response variables
            # y_pred shape: (n_test, 3) - predictions for all 3 responses
            results = pipeline.run(
                multi_response_model,
                X_cal=X, Y_cal=Y,
                reference_input_key='Y_cal',
                comparison_output_key='y_pred'
            )
            # RMSE computed across all elements of the matrix
        """
        # Get config values with parameter overrides
        ref_input = reference_input_key or self.config.reference_input_key
        ref_output = reference_output_key or self.config.reference_output_key
        capture_keys = capture_output_keys if capture_output_keys is not None else (self.config.capture_outputs or [])
        
        # comparison_output_key defaults to reference_output_key if not specified
        # (in most cases they're the same - e.g., compare 'scores' to 'scores')
        comp_output = comparison_output_key or self.config.comparison_output_key
        if comp_output is None:
            comp_output = ref_output  # Default: compare same output as reference
        
        # Find first array-like input (X) and its size for splitting
        X = None
        y = None
        n_samples = None
        input_arrays = {}  # Store all input arrays for reference_input_key lookup
        
        for key, val in kwargs.items():
            if isinstance(val, (np.ndarray, pd.DataFrame)):
                input_arrays[key] = val
                if X is None:
                    X = val
                    n_samples = len(val)
                elif key.lower() in ["y", "y_cal"]:
                    y = val

        if X is None:
            raise ValueError("No array-like input found in kwargs for splitting")

        # Determine reference source
        reference_array = None
        single_fit_result = None
        
        if ref_input is not None:
            # Input reference mode: use input array directly
            if ref_input not in input_arrays:
                raise ValueError(f"reference_input_key '{ref_input}' not found in inputs. "
                               f"Available: {list(input_arrays.keys())}")
            reference_array = input_arrays[ref_input]
            if isinstance(reference_array, pd.DataFrame):
                reference_array = reference_array.values
                
        elif ref_output is not None:
            # Output reference mode: run single fit first
            single_fit_kwargs = {"fold": -1}
            for key, data in kwargs.items():
                if isinstance(data, np.ndarray):
                    single_fit_kwargs[f"{key}_train"] = data
                    single_fit_kwargs[f"{key}_test"] = data
                elif isinstance(data, pd.DataFrame):
                    single_fit_kwargs[f"{key}_train"] = data
                    single_fit_kwargs[f"{key}_test"] = data
                else:
                    single_fit_kwargs[key] = data
            
            single_fit_result = func(**single_fit_kwargs)
            # Normalize single_fit_result to dict format (handle both dict and tuple returns)
            if isinstance(single_fit_result, dict):
                single_fit_result_dict = single_fit_result
            elif isinstance(single_fit_result, (tuple, list)):
                single_fit_result_dict = {}
            else:
                single_fit_result_dict = {}
                
            if ref_output is not None:
                if isinstance(single_fit_result_dict, dict) and ref_output not in single_fit_result_dict:
                    raise ValueError(f"reference_output_key '{ref_output}' not found in function output. "
                                   f"Available: {list(single_fit_result_dict.keys())}")
                if isinstance(single_fit_result_dict, dict):
                    reference_array = single_fit_result_dict[ref_output]

        # Initialize storage
        fold_metrics = {metric: [] for metric in self.config.output_metrics}
        fold_outputs_dict = {key: [] for key in capture_keys}
        test_indices_list = []
        fold_count = 0

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
                # Normalize fold_result to dict format
                # (handle both dict and tuple returns)
                if isinstance(fold_result, dict):
                    fold_result_dict = fold_result
                elif isinstance(fold_result, (tuple, list)):
                    # Tuple/list result - only process if we have specific capture keys
                    # Otherwise skip (tuples from analyst-style functions)
                    fold_result_dict = {}
                else:
                    # Single value result
                    fold_result_dict = {}
                
                # Compute metrics if we have a reference and comparison output
                if reference_array is not None and comp_output is not None:
                    if isinstance(fold_result_dict, dict) and comp_output in fold_result_dict:
                        fold_output = fold_result_dict[comp_output]
                        
                        # Get corresponding portion of reference
                        # For input reference: index by test_idx
                        # For output reference: also index by test_idx
                        fold_reference = reference_array[test_idx]
                        
                        # Compute requested metrics
                        self._compute_fold_metrics(
                            fold_output, fold_reference, fold_metrics, len(test_idx)
                        )
                
                # Capture specified outputs
                for output_key in capture_keys:
                    if isinstance(fold_result_dict, dict) and output_key in fold_result_dict:
                        fold_outputs_dict[output_key].append(fold_result_dict[output_key])
            
            fold_count += 1

        # Build aggregated results
        aggregated = {}
        
        # Aggregate metrics
        for metric_name, values in fold_metrics.items():
            if values:
                aggregated[f"{metric_name}_folds"] = values
                aggregated[f"{metric_name}_mean"] = float(np.mean(values))
                aggregated[f"{metric_name}_std"] = float(np.std(values))
                aggregated[f"{metric_name}_min"] = float(np.min(values))
                aggregated[f"{metric_name}_max"] = float(np.max(values))
        
        # Reconstruct outputs from folds
        for output_key, fold_outputs_list in fold_outputs_dict.items():
            if fold_outputs_list:
                reconstructed = self._reconstruct_from_folds(
                    fold_outputs_list, n_samples, test_indices_list
                )
                
                if isinstance(reconstructed, dict):
                    # Non-sample-based: segregated by fold
                    aggregated[f"{output_key}_cv"] = FoldSegregatedOutput(reconstructed)
                else:
                    # Sample-based: full reconstructed array
                    aggregated[f"{output_key}_cv"] = reconstructed
                
                # Include single-fit output if available
                if single_fit_result is not None:
                    if isinstance(single_fit_result, dict) and output_key in single_fit_result:
                        aggregated[f"{output_key}_single"] = single_fit_result[output_key]
        
        # Include reference info
        if reference_array is not None:
            aggregated["reference"] = reference_array
            aggregated["reference_source"] = "input" if ref_input else "output"
        
        aggregated["n_folds"] = fold_count
        
        self.results = aggregated
        return aggregated
    
    def _compute_fold_metrics(
        self, 
        fold_output: np.ndarray, 
        fold_reference: np.ndarray, 
        fold_metrics: Dict[str, List[float]],
        n_samples: int
    ) -> None:
        """Compute requested metrics for a fold, supporting vector and matrix outputs.
        
        For matrix data (2D), returns a vector with one metric value per column.
        For vector data, computes metrics on the entire array.
        """
        fold_output = np.asarray(fold_output)
        fold_reference = np.asarray(fold_reference)

        # Normalize vectors to explicit 2D matrices to handle mixed-shape
        # cases safely (e.g., output=(n, m) and reference=(n,)).
        if fold_output.ndim == 1:
            fold_output = fold_output.reshape(-1, 1)
        if fold_reference.ndim == 1:
            fold_reference = fold_reference.reshape(-1, 1)

        if fold_output.ndim != 2 or fold_reference.ndim != 2:
            raise ValueError(
                "CV metrics expect fold_output and fold_reference to be vector/matrix-like with sample axis first"
            )

        if fold_output.shape[0] != fold_reference.shape[0]:
            raise ValueError(
                f"CV metrics sample mismatch: output has {fold_output.shape[0]} samples, "
                f"reference has {fold_reference.shape[0]}"
            )

        out_cols = fold_output.shape[1]
        ref_cols = fold_reference.shape[1]

        if ref_cols == 1 and out_cols > 1:
            # Broadcast single reference column across model/output columns.
            fold_reference = np.repeat(fold_reference, out_cols, axis=1)
        elif out_cols == 1 and ref_cols > 1:
            # Broadcast single output column across multi-column references.
            fold_output = np.repeat(fold_output, ref_cols, axis=1)

        if fold_output.shape[1] != fold_reference.shape[1]:
            raise ValueError(
                f"CV metrics column mismatch: output has {fold_output.shape[1]} columns, "
                f"reference has {fold_reference.shape[1]}"
            )

        # Check if data is matrix-valued (multiple columns) after normalization.
        is_matrix = fold_output.shape[1] > 1
        
        if is_matrix:
            # Compute metrics per column and return as vector
            n_cols = fold_output.shape[1]
            col_metrics = {metric_name: [] for metric_name in fold_metrics.keys()}
            
            for col in range(n_cols):
                diff = fold_output[:, col] - fold_reference[:, col]
                
                for metric_name in fold_metrics.keys():
                    metric_lower = metric_name.lower()
                    
                    if metric_lower == 'rmse':
                        value = float(np.sqrt(np.mean(diff ** 2)))
                    elif metric_lower == 'mse':
                        value = float(np.mean(diff ** 2))
                    elif metric_lower == 'mae':
                        value = float(np.mean(np.abs(diff)))
                    elif metric_lower == 'r2':
                        ss_res = np.sum(diff ** 2)
                        ss_tot = np.sum((fold_reference[:, col] - np.mean(fold_reference[:, col])) ** 2)
                        value = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else np.nan
                    elif metric_lower == 'bias':
                        value = float(np.mean(diff))
                    elif metric_lower == 'sep':  # Standard Error of Prediction
                        bias = np.mean(diff)
                        value = float(np.sqrt(np.mean((diff - bias) ** 2)))
                    else:
                        continue
                    
                    col_metrics[metric_name].append(value)
            
            # Store as vectors (one value per column)
            for metric_name in fold_metrics.keys():
                fold_metrics[metric_name].append(np.array(col_metrics[metric_name]))
        else:
            # Vector data: compute metrics on entire array
            diff = fold_output.flatten() - fold_reference.flatten()
            
            for metric_name in fold_metrics.keys():
                metric_lower = metric_name.lower()
                
                if metric_lower == 'rmse':
                    value = float(np.sqrt(np.mean(diff ** 2)))
                elif metric_lower == 'mse':
                    value = float(np.mean(diff ** 2))
                elif metric_lower == 'mae':
                    value = float(np.mean(np.abs(diff)))
                elif metric_lower == 'r2':
                    ss_res = np.sum(diff ** 2)
                    ss_tot = np.sum((fold_reference.flatten() - np.mean(fold_reference)) ** 2)
                    value = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else np.nan
                elif metric_lower == 'bias':
                    value = float(np.mean(diff))
                elif metric_lower == 'sep':  # Standard Error of Prediction
                    bias = np.mean(diff)
                    value = float(np.sqrt(np.mean((diff - bias) ** 2)))
                else:
                    # Unknown metric, skip
                    continue
                
                fold_metrics[metric_name].append(value)


def cv_configuration(
    use_cv: bool,
    cv_strategy: str = "loocv",
    n_splits: int = 5,
    random_state: Optional[int] = 42,
    shuffle: bool = True,
    window_size: Optional[int] = None,
    n_repeats: Optional[int] = None,
    test_size: Optional[float] = None,
    stratify_layer: int = 1,
    output_metrics: Optional[List[str]] = None,
    capture_outputs: Optional[List[str]] = None,
    reference_input_key: Optional[str] = None,
    reference_output_key: Optional[str] = None,
    comparison_output_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a CV configuration object to be routed to modeling functions.

    Args:
        use_cv: Enable/disable CV
        cv_strategy: 'loocv', 'kfold', 'stratified_kfold', 'timeseries', 'repeated_kfold',
                 'shuffle_split', 'venetian_blinds', 'moving_window', 'bootstrap'
        n_splits: Number of folds/splits (ignored for LOOCV; auto-set to n_samples)
        random_state: Seed for reproducibility
           shuffle: Randomize before splitting (applies to kfold, stratified_kfold;
              not applicable to venetian_blinds, timeseries, repeated_kfold,
             shuffle_split, moving_window, loocv, bootstrap)
        window_size: Optional training window size for moving_window strategy.
                If None, auto-calculated from n_samples and n_splits.
        n_repeats: Number of repetitions for repeated_kfold strategy (default: 10)
        test_size: Test set proportion for shuffle_split strategy (default: 0.2 = 20%)
        stratify_layer: 1-based class layer to use for stratified_kfold
        output_metrics: Which metrics to compute (e.g., ['rmse', 'r2', 'mae', 'bias', 'sep'])
        capture_outputs: Which function outputs to capture per-fold
                        (e.g., ['y_pred', 'scores']). Captured outputs appear
                        in results as '{output_key}_cv' (reconstructed).
        reference_input_key: Input key to use as ground truth reference (e.g., 'Y_cal').
                            Each fold's output is compared against reference[test_idx].
                            This is the standard CV use case.
        reference_output_key: Output key from single fit to use as reference.
                             Only used if reference_input_key is None.
                             Useful for stability assessment.
        comparison_output_key: Which function output to compare against the reference.
                              Required when using reference_input_key or reference_output_key.

    Returns:
        dict with 'cv_config' key containing CVConfig object
        
    Example - Standard CV (predictions vs actual Y):
        cv_config = cv_configuration(
            use_cv=True,
            cv_strategy='kfold',
            n_splits=5,
            output_metrics=['rmse', 'r2'],
            reference_input_key='Y_cal',      # Compare against actual Y
            comparison_output_key='y_pred',   # The model's predictions
            capture_outputs=['y_pred']
        )
        
    Example - Stability assessment (PCA scores vs single fit):
        cv_config = cv_configuration(
            use_cv=True,
            cv_strategy='kfold',
            n_splits=5,
            output_metrics=['rmse'],
            reference_output_key='scores',    # Single-fit scores as reference
            comparison_output_key='scores',   # Fold scores to compare
            capture_outputs=['scores', 'loadings']
        )
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
        stratify_layer=stratify_layer,
        output_metrics=output_metrics or ["rmse", "r2"],
        capture_outputs=capture_outputs or [],
        reference_input_key=reference_input_key,
        reference_output_key=reference_output_key,
        comparison_output_key=comparison_output_key,
    )

    strategy_display_names = {
        "loocv": "LOOCV (Leave-One-Out)",
        "kfold": "K-Fold (Standard)",
        "stratified_kfold": "Stratified K-Fold (Classification)",
        "timeseries": "Time Series (Forward Chaining)",
        "repeated_kfold": "Repeated K-Fold",
        "shuffle_split": "Shuffle Split (Random)",
        "venetian_blinds": "Venetian Blinds",
        "moving_window": "Moving Window",
        "bootstrap": "Bootstrap (OOB)",
    }
    strategy_name = strategy_display_names.get(config.cv_strategy, config.cv_strategy)

    if not config.use_cv:
        cv_report = (
            "Cross-Validation Settings\n"
            "=========================\n"
            "Enabled:           No\n"
            "\n"
            "Cross-validation is disabled. Downstream functions will run\n"
            "a single fit on all available data."
        )
    else:
        lines = [
            "Cross-Validation Settings",
            "=========================",
            f"Enabled:           Yes",
            f"Strategy:          {strategy_name}",
            "",
            "Parameters",
            "----------",
        ]

        strategies_with_splits = {"kfold", "stratified_kfold", "timeseries", "repeated_kfold",
                                   "shuffle_split", "venetian_blinds", "moving_window", "bootstrap"}
        if config.cv_strategy in strategies_with_splits:
            splits_label = "Bootstrap Iterations" if config.cv_strategy == "bootstrap" else "Splits / Folds"
            lines.append(f"{splits_label + ':':19} {config.n_splits}")

        strategies_with_random_state = {"kfold", "stratified_kfold", "repeated_kfold", "shuffle_split", "bootstrap"}
        if config.cv_strategy in strategies_with_random_state:
            random_str = str(config.random_state) if config.random_state is not None else "None (random)"
            lines.append(f"{'Random State:':19} {random_str}")

        if config.cv_strategy in {"kfold", "stratified_kfold"}:
            lines.append(f"{'Shuffle:':19} {config.shuffle}")

        if config.cv_strategy == "stratified_kfold":
            lines.append(f"{'Class Layer:':19} {config.stratify_layer}")

        if config.cv_strategy == "repeated_kfold":
            lines.append(f"{'Number of Repeats:':19} {config.n_repeats}")

        if config.cv_strategy == "shuffle_split":
            lines.append(f"{'Test Set Size:':19} {config.test_size}")

        if config.cv_strategy == "moving_window":
            window_str = str(config.window_size) if config.window_size is not None else "Auto"
            lines.append(f"{'Window Size:':19} {window_str}")

        cv_report = "\n".join(lines)

    return {"cv_config": config, "cv_report": cv_report}
