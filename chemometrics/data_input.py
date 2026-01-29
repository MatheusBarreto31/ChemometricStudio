
from typing import Tuple, Optional, List
import numpy as np
import os
import csv
import pandas as pd


def _load_file(path: str, separator: Optional[str], num_headlines: int) -> np.ndarray:
    """Load data from file, supporting text and Excel formats."""
    if path.lower().endswith(('.xlsx', '.xls')):
        # Load Excel file
        df = pd.read_excel(path, header=None, skiprows=num_headlines)
        return df.values
    else:
        # Load text file
        return np.loadtxt(path, delimiter=separator, skiprows=num_headlines)


def load_data(d_specs_separator: str, d_specs_headlines: str, d_specs_type: str, d_specs_dimensions: Optional[str] = None,
              data_path: Optional[List[str]] = None, nway_flag: int = 1, y_path: Optional[str] = None,
              var_path: Optional[str] = None, smp_path: Optional[str] = None,
              transpose: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]], List[str]]:
    """
    Load and organize chemometrics data.

    Supports text files (CSV, TSV, space-separated) and Excel files (.xlsx, .xls).

    Args:
        d_specs_separator: Separator type ('comma', 'tabs', 'spaces')
        d_specs_headlines: Number of header rows to skip
        d_specs_type: Data type ('x_vector', 'xy_vector', 'x_matrix', etc.)
        d_specs_dimensions: Dimensions for reshaping (optional, defaults to None)
        data_path: List of paths to X data files (text or Excel, defaults to None)
        nway_flag: Number of ways (1 for 1D/2D, 2+ for multi-way, defaults to 1)
        y_path: Optional path to Y data file (text or Excel)
        var_path: Optional path to variable labels file (text)
        smp_path: Optional path to sample labels file (text)
        transpose: Whether to transpose data (defaults to False)

    Returns:
        X_cal: X data array
        Y_cal: Y data array or None
        var_label: Variable labels or None
        smp_cal: Sample labels
    """
    # Parse d_specs parameters
    separator_map = {"comma": ",", "spaces": None, "tabs": "\t"}
    separator = separator_map.get(d_specs_separator, ",")
    num_headlines = int(d_specs_headlines)
    data_type = d_specs_type
    dimensions = d_specs_dimensions if d_specs_dimensions and d_specs_dimensions.strip() else None

    # Load X data
    X, row_counts = _load_x_data(data_path, separator, num_headlines, data_type, dimensions, transpose, nway_flag)

    # Load Y data (using same separator as X for consistency)
    Y = _load_y_data(y_path, separator, 0) if y_path else None

    # Load labels
    var_labels = _load_labels(var_path) if var_path else None
    if smp_path is None:
        if nway_flag == 1 and data_type == "x_matrix":
            smp_labels = _generate_row_labels(data_path, row_counts)
        else:
            smp_labels = _generate_sample_labels(data_path)
    else:
        smp_labels = _load_labels(smp_path)

    return X, Y, var_labels, smp_labels


def _load_x_data(data_path: List[str], separator: Optional[str], num_headlines: int,
                 data_type: str, dimensions: Optional[str], transpose: bool, nway_flag: int) -> Tuple[np.ndarray, List[int]]:
    """Load and organize X data based on nway_flag and data_type."""
    # global nway_flag

    if nway_flag == 1:
        return _load_x_1way(data_path, separator, num_headlines, data_type, transpose)
    else:
        X = _load_x_multiway(data_path, separator, num_headlines, data_type, dimensions, nway_flag, transpose)
        return X, []  # No row_counts for multiway


def _load_x_1way(data_path: List[str], separator: Optional[str], num_headlines: int,
                 data_type: str, transpose: bool) -> Tuple[np.ndarray, List[int]]:
    """Load 1-way X data."""
    samples = []
    row_counts = []
    for path in data_path:
        data = _load_file(path, separator, num_headlines)
        if data_type == "x_vector":
            # Single vector, transpose to row
            sample = data.flatten()
        elif data_type == "xy_vector":
            # Second column
            sample = data[:, 1] if data.ndim > 1 else data
        elif data_type == "x_matrix":
            # 2D matrix
            sample = data
            if transpose==True:
                sample = sample.T
            row_counts.append(sample.shape[0])
        else:
            raise ValueError(f"Unknown data_type for 1-way: {data_type}")
        samples.append(sample)

    # Concatenate samples
    if data_type == "x_matrix":
        X = np.concatenate(samples, axis=0)
    else:
        X = np.array(samples)
    return X, row_counts


def _load_x_multiway(data_path: List[str], separator: Optional[str], num_headlines: int,
                     data_type: str, dimensions: Optional[str], nway_flag: int, transpose: Optional[bool]) -> np.ndarray:
    """Load multi-way X data."""
    # global nway_flag

    dims = None
    if dimensions is not None:
        dims = [int(d) for d in dimensions.split(",")]
        if len(dims) != nway_flag:
            raise ValueError(f"Dimensions {dims} don't match nway_flag {nway_flag}")
    elif not (nway_flag == 2 and data_type in ["x_matrix", "xy_matrix"]):
        raise ValueError("Dimensions required for multi-way data unless nway_flag=2 and data_type is x_matrix or xy_matrix")

    samples = []
    for path in data_path:
        data = _load_file(path, separator, num_headlines)
        if data_type == "x_vector":
            sample = data.flatten()
            sample = sample.reshape(dims)
        elif data_type == "xy_vector":
            sample = data[:, 1] if data.ndim > 1 else data
            sample = sample.reshape(dims)
        elif data_type == "xyz_vector":
            sample = data[:, 2] if data.ndim > 1 else data
            sample = sample.reshape(dims)
        elif data_type == "x_matrix" and nway_flag == 2:
            sample = data
            if dims is None:
                dims = list(data.shape)
        elif data_type == "xy_matrix" and nway_flag == 2:
            sample = data[:, 1::2]  # Every second column starting from second
            if dims is None:
                dims = list(sample.shape)
        else:
            raise ValueError(f"Unknown or invalid data_type for {nway_flag}-way: {data_type}")
        if nway_flag == 2 and transpose==True:
                sample = sample.T
        samples.append(sample)

    # Stack into tensor with sample dimension first
    X = np.array(samples)
    return X


def _load_y_data(y_path: str, separator: Optional[str] = None, num_headlines: int = 0) -> np.ndarray:
    """Load Y data as 2D matrix using specified separator, preserving matrix structure."""
    data = _load_file(y_path, separator, num_headlines)
    # Keep as 2D if multiple columns, otherwise reshape to column vector
    if data.ndim == 1:
        return data.reshape(-1, 1)
    return data


def _load_labels(label_path: str) -> List[str]:
    """Load labels from file, one per line."""
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def _generate_sample_labels(data_path: List[str]) -> List[str]:
    """Generate sample labels from filenames without extension."""
    labels = []
    for path in data_path:
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        labels.append(name)
    return labels


def _generate_row_labels(data_path: List[str], row_counts: List[int]) -> List[str]:
    """Generate sample labels for each row in concatenated matrices."""
    labels = []
    for path, count in zip(data_path, row_counts):
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        for i in range(1, count + 1):
            labels.append(f"{name}_{i}")
    return labels


