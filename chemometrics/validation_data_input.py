from chemometrics.data_input import load_data
from typing import Tuple, Optional, List, Any
import numpy as np
import os


def validation_data_main(X_cal: np.ndarray, Y_cal: Optional[np.ndarray], smp_cal: List[str],
                          class_data_cal: Optional[List[Any]] = None,
                          cal_metadata: Optional[dict] = None,
                          validation_mode: Optional[str] = None, createVal: Optional[bool] = None,
                          creationMethod: Optional[str] = None,
                          calProportion: Optional[float] = None, random_seed: Optional[int] = None,
                          selection_file: Optional[str] = None,
                          d_specs_separator: Optional[str] = None, d_specs_headlines: Optional[str] = None,
                          d_specs_type: Optional[str] = None, d_specs_dimensions: Optional[str] = None,
                          data_path: Optional[List[str]] = None,
                          y_path: Optional[str] = None, var_path: Optional[str] = None,
                          smp_path: Optional[str] = None, transpose: bool = False,
                          nway_flag: Optional[int] = None, reshape_order: str = 'F',
                          X_val_path: Optional[str] = None, Y_val_path: Optional[str] = None,
                          val_labels_path: Optional[str] = None,
                          cdata_path: Optional[str] = None,
                          source_metadata_overrides: Optional[dict] = None) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray], List[str], List[str], Optional[List[Any]], Optional[List[Any]], Optional[dict], Optional[dict]]:
    """
    Split data into calibration and validation sets.

    Args:
        X_cal: Input X data
        Y_cal: Input Y data
        smp_cal: Sample labels
        class_data_cal: Input classification data (optional)
        cal_metadata: Input per-sample calibration metadata dictionary (optional)
        validation_mode: "Create Validation Set" or "Load External Validation Set"
        createVal: (deprecated) If True, create validation from calibration; if False, load validation separately
        creationMethod: Method for creating validation ('random', 'kennard_stone', 'file')
        calProportion: Proportion of samples for calibration (0-1)
        random_seed: Seed for reproducible random selection (for 'random' method)
        selection_file: Path to file with 1s/2s for cal/val selection (for 'file' method)
        X_val_path, Y_val_path, val_labels_path: Paths to external validation files
        d_specs, data_path, etc.: Parameters for load_data if loading external validation
        cdata_path: Path to external classification data file

    Returns:
        X_cal, Y_cal, X_val, Y_val, smp_cal, smp_val, class_data_cal, class_data_val, cal_metadata, val_metadata
    """
    # Determine mode: prefer validation_mode if provided, fall back to createVal for backward compatibility
    if validation_mode is not None:
        should_create = "Create" in validation_mode
    elif createVal is not None:
        should_create = createVal
    else:
        # Default to loading external validation
        should_create = False
    
    n_samples = X_cal.shape[0]

    if not should_create:
        # Load validation data separately
        # Check if using new parameters (X_val_path, Y_val_path)
        if X_val_path or Y_val_path:
            # Load from external file paths
            X_val, Y_val, _, smp_val, _, _, class_data_val, val_metadata = load_data(
                d_specs_separator=d_specs_separator or "Auto detect",
                d_specs_headlines=d_specs_headlines if d_specs_headlines is not None else "",
                d_specs_type=d_specs_type or "Auto detect",
                d_specs_dimensions=d_specs_dimensions or "",
                data_path=[X_val_path] if X_val_path else [],
                nway_flag=nway_flag or 1,
                y_path=Y_val_path,
                var_path=None,
                smp_path=val_labels_path,
                transpose=transpose,
                reshape_order=reshape_order,
                cdata_path=cdata_path,
                source_metadata_overrides=source_metadata_overrides
            )
        else:
            # Use data_path parameters for loading external validation
            if data_path is None or nway_flag is None:
                raise ValueError("Either X_val_path/Y_val_path or data_path/nway_flag required when loading external validation")
            X_val, Y_val, _, smp_val, _, _, class_data_val, val_metadata = load_data(
                d_specs_separator=d_specs_separator or "Auto detect",
                d_specs_headlines=d_specs_headlines if d_specs_headlines is not None else "",
                d_specs_type=d_specs_type or "Auto detect",
                d_specs_dimensions=d_specs_dimensions or "",
                data_path=data_path,
                nway_flag=nway_flag,
                y_path=y_path,
                var_path=var_path,
                smp_path=smp_path,
                transpose=transpose,
                reshape_order=reshape_order,
                cdata_path=cdata_path,
                source_metadata_overrides=source_metadata_overrides
            )
        
        X_cal_out, Y_cal_out, smp_cal_out, class_data_cal_out = X_cal, Y_cal, smp_cal, class_data_cal
        cal_metadata_out = cal_metadata
    else:
        # Create validation from calibration
        if creationMethod is None or calProportion is None:
            raise ValueError("creationMethod and calProportion required when creating validation set")

        n_cal = int(n_samples * calProportion)
        indices = np.arange(n_samples)

        if creationMethod == 'random':
            rng = np.random.default_rng(random_seed)
            cal_indices = rng.choice(indices, size=n_cal, replace=False)
        elif creationMethod == 'kennard_stone':
            cal_indices = _kennard_stone_selection(X_cal, n_cal)
        elif creationMethod == 'file':
            if selection_file is None:
                raise ValueError("selection_file required for 'file' creationMethod")
            cal_indices = _load_selection_from_file(selection_file, n_samples)
        else:
            raise ValueError(f"Unknown creationMethod: {creationMethod}")

        val_indices = np.setdiff1d(indices, cal_indices)

        X_cal_out = X_cal[cal_indices]
        Y_cal_out = Y_cal[cal_indices] if Y_cal is not None else None
        smp_cal_out = [smp_cal[i] for i in cal_indices]
        class_data_cal_out = [class_data_cal[i] for i in cal_indices] if class_data_cal is not None else None
        cal_metadata_out, val_metadata = _split_metadata_by_indices(cal_metadata, cal_indices, val_indices)

        X_val = X_cal[val_indices]
        Y_val = Y_cal[val_indices] if Y_cal is not None else None
        smp_val = [smp_cal[i] for i in val_indices]
        class_data_val = [class_data_cal[i] for i in val_indices] if class_data_cal is not None else None

    return X_cal_out, Y_cal_out, X_val, Y_val, smp_cal_out, smp_val, class_data_cal_out, class_data_val, cal_metadata_out, val_metadata


def _split_metadata_by_indices(sample_metadata: Optional[dict], cal_indices: np.ndarray, val_indices: np.ndarray) -> Tuple[Optional[dict], Optional[dict]]:
    """Split metadata dictionary into calibration and validation subsets by sample index.

    Metadata sample_index values are expected to be 1-based.
    Internal cal/val indices are 0-based and converted accordingly.
    """
    if sample_metadata is None:
        return None, None

    if not isinstance(sample_metadata, dict):
        return None, None

    cal_set = {int(i) for i in np.asarray(cal_indices).tolist()}
    val_set = {int(i) for i in np.asarray(val_indices).tolist()}

    cal_metadata = {}
    val_metadata = {}

    fallback_order = 0
    for key, entry in sample_metadata.items():
        if isinstance(entry, dict) and "sample_index" in entry:
            try:
                sample_index_0b = int(entry["sample_index"]) - 1
            except (TypeError, ValueError):
                sample_index_0b = fallback_order
        else:
            sample_index_0b = fallback_order

        if sample_index_0b in cal_set:
            cal_metadata[key] = entry
        elif sample_index_0b in val_set:
            val_metadata[key] = entry

        fallback_order += 1

    return cal_metadata, val_metadata


def _kennard_stone_selection(X: np.ndarray, n_select: int) -> np.ndarray:
    """Select samples using Kennard-Stone algorithm."""
    n_samples = X.shape[0]
    selected = []
    remaining = list(range(n_samples))

    # Select first sample (closest to mean)
    mean = np.mean(X, axis=0)
    distances = np.linalg.norm(X - mean, axis=1)
    first = np.argmax(distances)
    selected.append(first)
    remaining.remove(first)

    while len(selected) < n_select:
        max_dist = -1
        next_idx = -1
        for i in remaining:
            min_dist_to_selected = min(np.linalg.norm(X[i] - X[j]) for j in selected)
            if min_dist_to_selected > max_dist:
                max_dist = min_dist_to_selected
                next_idx = i
        selected.append(next_idx)
        remaining.remove(next_idx)

    return np.array(selected)


def _load_selection_from_file(filepath: str, n_samples: int) -> np.ndarray:
    """Load selection from file: 1 for cal, 2 for val."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if len(lines) != n_samples:
        raise ValueError(f"File {filepath} must have {n_samples} lines")
    selection = [int(line.strip()) for line in lines]
    cal_indices = [i for i, s in enumerate(selection) if s == 1]
    return np.array(cal_indices)

