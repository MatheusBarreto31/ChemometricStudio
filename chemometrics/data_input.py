
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import os
import csv
import pandas as pd
from datetime import datetime
import re


def _load_file(path: str, separator: Optional[str], num_headlines: int) -> np.ndarray:
    """Load data from file, supporting text and Excel formats.
    
    Handles empty cells and formatting issues gracefully.
    """
    if path.lower().endswith(('.xlsx', '.xls')):
        # Load Excel file
        df = pd.read_excel(path, header=None, skiprows=num_headlines)
        return df.values
    else:
        # Load text file - use pandas for more robust CSV handling
        try:
            # Map separator names to pandas parameters
            if separator is None:
                # Space-separated or whitespace
                sep = r'\s+'  # One or more whitespace characters
            elif separator == ',':
                sep = ','
            elif separator == '\t':
                sep = '\t'
            else:
                sep = separator
            
            # Use pandas to read, which handles edge cases better
            df = pd.read_csv(path, sep=sep, header=None, skiprows=num_headlines, 
                           engine='python', na_values=['', ' '], 
                           skip_blank_lines=True)
            
            # Convert to numeric, replacing any remaining non-numeric values
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Keep NaN values as-is for later handling by preprocessing functions
            return df.values
        except Exception as e:
            # Fallback to numpy loadtxt
            print(f"Warning: pandas read failed, falling back to numpy: {e}")
            return np.loadtxt(path, delimiter=separator, skiprows=num_headlines)


def _is_auto_detect_value(value: Any) -> bool:
    """Return True when a user value indicates automatic detection."""
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {
        "",
        "auto detect",
        "autodetect",
        "auto",
        "detecção automática",
        "detecao automatica",
        "automático",
        "automatico"
    }


def _normalize_separator_choice(choice: Any) -> str:
    """Normalize UI separator labels to internal tokens."""
    normalized = str(choice).strip().lower()
    mapping = {
        "comma": "comma",
        "vírgula": "comma",
        "virgula": "comma",
        "tabs": "tabs",
        "tabulações": "tabs",
        "tabulacoes": "tabs",
        "spaces": "spaces",
        "espaços": "spaces",
        "espacos": "spaces",
        "auto detect": "auto detect",
        "autodetect": "auto detect",
        "detecção automática": "auto detect",
        "detecao automatica": "auto detect"
    }
    return mapping.get(normalized, normalized)


def _detect_separator(data_paths: List[str]) -> str:
    """Detect file separator among comma, tabs, or spaces using first available file."""
    if not data_paths:
        return "comma"

    first_path = data_paths[0]
    if str(first_path).lower().endswith((".xlsx", ".xls")):
        return "comma"

    try:
        with open(first_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if '\t' in stripped:
                    return "tabs"
                if ',' in stripped:
                    return "comma"
                return "spaces"
    except Exception:
        return "comma"

    return "comma"


def _is_numeric_token(token: Any) -> bool:
    """Check whether token can be interpreted as numeric."""
    if token is None:
        return False
    text = str(token).strip()
    if text == "":
        return False
    try:
        float(text)
        return True
    except Exception:
        return False


def _detect_header_rows(path: str, separator: Optional[str]) -> int:
    """Detect number of non-data header rows before first numeric row."""
    if path.lower().endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(path, header=None, nrows=60)
        except Exception:
            return 0

        for idx, row in df.iterrows():
            values = [v for v in row.tolist() if str(v).strip() != "" and str(v).strip().lower() != "nan"]
            if not values:
                continue
            numeric_count = sum(1 for v in values if _is_numeric_token(v))
            if numeric_count / max(len(values), 1) >= 0.6:
                return int(idx)
        return 0

    split_pattern = None
    if separator is None:
        split_pattern = r'\s+'

    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f):
                stripped = line.strip()
                if not stripped:
                    continue
                if split_pattern is not None:
                    tokens = [tok for tok in re.split(split_pattern, stripped) if tok]
                elif separator == '\t':
                    tokens = [tok.strip() for tok in stripped.split('\t') if tok.strip()]
                elif separator == ',':
                    tokens = [tok.strip() for tok in stripped.split(',') if tok.strip()]
                else:
                    tokens = [tok.strip() for tok in stripped.split() if tok.strip()]

                if not tokens:
                    continue
                numeric_count = sum(1 for tok in tokens if _is_numeric_token(tok))
                if numeric_count / max(len(tokens), 1) >= 0.6:
                    return idx
    except Exception:
        return 0

    return 0


def _is_monotonic_numeric_column(values: np.ndarray) -> bool:
    """Return True when numeric column is monotonic increasing/decreasing."""
    if values is None:
        return False
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        return False
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return False
    diffs = np.diff(arr)
    return bool(np.all(diffs >= 0) or np.all(diffs <= 0))


def _infer_data_type(data_paths: List[str], separator: Optional[str], num_headlines: int, nway_flag: int) -> str:
    """Infer a suitable data type based on first data file shape/content."""
    if not data_paths:
        return "x_matrix"

    try:
        data = _load_file(data_paths[0], separator, num_headlines)
    except Exception:
        return "x_matrix"

    if data is None:
        return "x_matrix"

    arr = np.asarray(data)
    if arr.ndim == 0:
        return "x_vector"
    if arr.ndim == 1:
        return "x_vector"

    rows, cols = arr.shape[0], arr.shape[1] if arr.shape[1:] else 1
    if rows <= 1:
        if cols <= 1:
            return "x_vector"
        if cols == 2:
            return "xy_vector"
        if cols == 3:
            return "xyz_vector"
        return "x_vector"

    if cols <= 1:
        return "x_vector"

    first_col_monotonic = _is_monotonic_numeric_column(arr[:, 0])
    if cols == 2 and first_col_monotonic:
        return "xy_vector"
    if nway_flag > 1 and cols == 3 and first_col_monotonic and _is_monotonic_numeric_column(arr[:, 1]):
        return "xyz_vector"

    if nway_flag == 2 and cols >= 4 and cols % 2 == 0 and first_col_monotonic:
        return "xy_matrix"

    return "x_matrix"


def load_data(d_specs_separator: str = "Auto detect", d_specs_headlines: str = "", d_specs_type: str = "Auto detect", d_specs_dimensions: Optional[List[str]] = None,
              data_path: Optional[List[str]] = None, nway_flag: int = 1, y_path: Optional[str] = None,
              var_path: Optional[List[str]] = None, smp_path: Optional[str] = None,
              transpose: bool = False, axis_info: Optional[List[str]] = None, reshape_order: str = 'F',
              dim_labels: Optional[List[str]] = None, scale_type: Optional[List[str]] = None,
              multi_file_per_sample: bool = False, num_samples: Optional[int] = None,
              y_from_x: bool = False, class_from_x: bool = False, smp_from_x: bool = False,
              y_columns: str = "", class_columns: str = "", smp_column: str = "",
              cdata_path: Optional[str] = None,
              source_metadata_overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[List[str]]], List[str], Optional[List[np.ndarray]], List[str], Optional[List[Any]], Dict[str, Dict[str, Any]]]:
    """
    Load and organize chemometrics data.

    Supports text files (CSV, TSV, space-separated) and Excel files (.xlsx, .xls).

    Args:
        d_specs_separator: Separator type ('comma', 'tabs', 'spaces') or auto-detect token
        d_specs_headlines: Number of header rows to skip, or empty/auto-detect token
        d_specs_type: Data type ('x_vector', 'xy_vector', 'x_matrix', etc.) or auto-detect token
        d_specs_dimensions: List of dimensions for reshaping (or comma-separated string)
        data_path: List of paths to X data files (text or Excel, defaults to None)
        nway_flag: Number of ways (1 for 1D/2D, 2+ for multi-way, defaults to 1)
        y_path: Optional path to Y data file (text or Excel)
        var_path: Optional list of paths to axis labels files (one per dimension, based on nway_flag)
        smp_path: Optional path to sample labels file (text)
        transpose: Whether to transpose data (defaults to False)
        axis_info: Optional list of axis ranges (e.g., ["100 200", "1 10"]) or semicolon-separated string
        reshape_order: Reshape order for multiway data - 'F' (Fortran/MATLAB column-major, default) or 'C' (C row-major)
        dim_labels: Optional list of dimension names or comma-separated string
        scale_type: Optional list of scale types for axis generation ('Linear', 'Log10', 'Log2', 'Ln')
        multi_file_per_sample: If True, each sample consists of multiple files (only for nway_flag >= 3)
        num_samples: Number of samples when using multi_file_per_sample mode (files divided equally)
        y_from_x: If True, extract Y data from the X file using y_columns (only for x_matrix, nway_flag=1)
        class_from_x: If True, extract class labels from the X file using class_columns (only for x_matrix, nway_flag=1)
        smp_from_x: If True, extract sample labels from a single column in the X file (only for x_matrix, nway_flag=1)
        y_columns: Column spec string for Y extraction (1-based; supports ranges like '1:4', '1-4', '2:2:6')
        class_columns: Column spec string for class label extraction (same format as y_columns)
        smp_column: Single column number (1-based integer) whose string values become sample labels
        cdata_path: Optional path to classification data file (one row per sample; supports multi-column labels)
        source_metadata_overrides: Optional mapping of absolute file paths to pre-recorded metadata

    Returns:
        X_cal: X data array
        Y_cal: Y data array or None
        axis_t_info: List of string lists containing axis text labels for each dimension
        smp_cal: Sample labels
        axis_n_info: List of axis vectors matching data dimensions or None
        dim_labels: List of dimension labels
        class_data_cal: List of class labels (or list of class-label rows for multi-layer files) or None
        cal_metadata: Dict with per-sample metadata extracted from source files
    """
    normalized_data_paths = _normalize_data_path(data_path)

    # Parse d_specs parameters (with optional auto-detect)
    separator_choice = _normalize_separator_choice(d_specs_separator)
    if _is_auto_detect_value(separator_choice):
        separator_choice = _detect_separator(normalized_data_paths)

    separator_map = {"comma": ",", "spaces": None, "tabs": "\t"}
    separator = separator_map.get(separator_choice, ",")

    if _is_auto_detect_value(d_specs_headlines):
        num_headlines = _detect_header_rows(normalized_data_paths[0], separator) if normalized_data_paths else 0
    else:
        num_headlines = int(d_specs_headlines)

    if _is_auto_detect_value(d_specs_type):
        data_type = _infer_data_type(normalized_data_paths, separator, num_headlines, nway_flag)
    else:
        data_type = d_specs_type
    
    # Normalize d_specs_dimensions to string (handle both string and list inputs)
    dimensions = None
    if d_specs_dimensions:
        if isinstance(d_specs_dimensions, str):
            dimensions = d_specs_dimensions if d_specs_dimensions.strip() else None
        elif isinstance(d_specs_dimensions, list):
            # Join list with commas to create comma-separated string
            dims = [str(d).strip() for d in d_specs_dimensions if d]
            dimensions = ",".join(dims) if dims else None

    # Normalize var_path to list (handle both string and list inputs)
    var_path_list = None
    if var_path:
        if isinstance(var_path, str):
            # Parse from semicolon-separated string
            var_path_list = [p.strip() for p in var_path.split(';') if p.strip()]
        elif isinstance(var_path, list):
            var_path_list = [p.strip() if isinstance(p, str) else p for p in var_path]

    # Normalize scale_type to list (handle both string and list inputs)
    scale_type_list = None
    if scale_type:
        if isinstance(scale_type, str):
            # Parse from comma-separated string
            scale_type_list = [s.strip() for s in scale_type.split(',') if s.strip()]
        elif isinstance(scale_type, list):
            scale_type_list = [s.strip() if isinstance(s, str) else s for s in scale_type]

    # Normalize axis_info to list (handle both string and list inputs)
    axis_info_list = None
    if axis_info:
        if isinstance(axis_info, str):
            # Parse from semicolon-separated string
            axis_info_list = [a.strip() for a in axis_info.split(';') if a.strip()]
        elif isinstance(axis_info, list):
            axis_info_list = [a.strip() if isinstance(a, str) else a for a in axis_info if a]

    sample_paths = None
    # These may be populated by auxiliary extraction from X
    extracted_Y: Optional[np.ndarray] = None
    extracted_class: Optional[List[Any]] = None
    extracted_smp: Optional[List[str]] = None

    # Determine whether to use column-extraction mode (y_from_x / class_from_x / smp_from_x)
    use_extraction = (
        (y_from_x or class_from_x or smp_from_x)
        and data_type == "x_matrix"
        and nway_flag == 1
        and not multi_file_per_sample
    )

    # Load X data - check if multi_file_per_sample mode is enabled
    if multi_file_per_sample and nway_flag >= 3 and data_path and num_samples:
        # Multi-file per sample mode: divide files equally among samples
        file_list = normalized_data_paths
        
        if len(file_list) % num_samples != 0:
            raise ValueError(f"Cannot divide {len(file_list)} files equally among {num_samples} samples")
        
        files_per_sample = len(file_list) // num_samples
        sample_paths = [file_list[i * files_per_sample:(i + 1) * files_per_sample] 
                        for i in range(num_samples)]
        
        X = _load_x_multifile_per_sample(sample_paths, separator, num_headlines, data_type, 
                                         dimensions, nway_flag, transpose, reshape_order)
        row_counts = []
        # Generate sample labels from sample indices if no smp_path provided
        if smp_path is None:
            smp_labels = [f"Sample_{i+1}" for i in range(num_samples)]
        else:
            smp_labels = _load_labels(smp_path)
    elif use_extraction:
        # Extraction mode: pull Y and/or class and/or smp columns directly from the X file(s)
        y_col_indices = _parse_column_spec(y_columns) if y_from_x else []
        class_col_indices = _parse_column_spec(class_columns) if class_from_x else []
        smp_col_idx: Optional[int] = None
        if smp_from_x and str(smp_column).strip():
            try:
                val = int(str(smp_column).strip()) - 1  # convert to 0-based
                smp_col_idx = val if val >= 0 else None
            except ValueError:
                smp_col_idx = None

        X, extracted_Y, extracted_class, extracted_smp, row_counts = _load_x_matrix_with_extraction(
            normalized_data_paths, separator, num_headlines, transpose,
            y_col_indices, class_col_indices, smp_col_idx
        )
        # Load sample labels: extracted column wins, then explicit file, then auto-generated
        if smp_from_x and extracted_smp is not None:
            smp_labels = extracted_smp
        elif smp_path is not None:
            smp_labels = _load_labels(smp_path)
        else:
            smp_labels = _generate_row_labels(normalized_data_paths, row_counts)
    else:
        # Standard loading mode
        X, row_counts = _load_x_data(normalized_data_paths, separator, num_headlines, data_type, dimensions, transpose, nway_flag, reshape_order)
        # Load sample labels
        if smp_path is None:
            if nway_flag == 1 and data_type == "x_matrix":
                smp_labels = _generate_row_labels(normalized_data_paths, row_counts)
            else:
                smp_labels = _generate_sample_labels(normalized_data_paths)
        else:
            smp_labels = _load_labels(smp_path)

    # Load Y data
    # Prefer extraction result; fall back to y_path when y_from_x is not active
    if use_extraction and y_from_x:
        Y = extracted_Y
    else:
        Y = _load_y_data(y_path, separator, 0) if y_path else None

    # Load classification data
    # Prefer extraction result; fall back to cdata_path when class_from_x is not active
    if use_extraction and class_from_x:
        class_data = extracted_class
    else:
        class_data = _load_class_data(cdata_path) if cdata_path else None

    # Override smp_labels with extracted values when smp_from_x is active and extraction ran
    # (already handled above in the elif use_extraction branch, but guard here in case of future
    #  refactoring that splits the branches differently)
    if use_extraction and smp_from_x and extracted_smp is not None:
        smp_labels = extracted_smp

    # Generate axis information (numerical vectors)
    if axis_info_list:
        # Use provided axis information with scale types
        axis_n_info = _generate_axis_info(axis_info_list, X, scale_type_list)
    else:
        # Generate default axis information (1 to dimension size for each axis)
        axis_n_info = _generate_default_axis_info(X)

    # Load axis text labels from files (axis_t_info)
    axis_t_info = _load_axis_text_info(var_path_list, smp_labels, nway_flag, axis_n_info)

    # Generate dimension labels
    processed_dim_labels = _generate_dim_labels(dim_labels, nway_flag)

    cal_metadata = _build_sample_metadata(
        smp_labels=smp_labels,
        data_paths=normalized_data_paths,
        row_counts=row_counts,
        nway_flag=nway_flag,
        data_type=data_type,
        multi_file_per_sample=multi_file_per_sample,
        sample_paths=sample_paths,
        source_metadata_overrides=source_metadata_overrides
    )

    return X, Y, axis_t_info, smp_labels, axis_n_info, processed_dim_labels, class_data, cal_metadata


def _load_x_data(data_path: List[str], separator: Optional[str], num_headlines: int,
                 data_type: str, dimensions: Optional[str], transpose: bool, nway_flag: int, reshape_order: str = 'F') -> Tuple[np.ndarray, List[int]]:
    """Load and organize X data based on nway_flag and data_type."""
    # global nway_flag

    if nway_flag == 1:
        return _load_x_1way(data_path, separator, num_headlines, data_type, transpose)
    else:
        X = _load_x_multiway(data_path, separator, num_headlines, data_type, dimensions, nway_flag, transpose, reshape_order)
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
        elif data_type == "xyz_vector":
            # Third column
            sample = data[:, 2] if data.ndim > 1 else data
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
                     data_type: str, dimensions: Optional[str], nway_flag: int, transpose: Optional[bool], reshape_order: str = 'F') -> np.ndarray:
    """Load multi-way X data.
    
    Args:
        reshape_order: 'F' for Fortran/MATLAB column-major (default) or 'C' for C row-major
    """
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
            # Use specified reshape order (Fortran for MATLAB, C for NumPy default)
            sample = sample.reshape(dims, order=reshape_order)
        elif data_type == "xy_vector":
            sample = data[:, 1] if data.ndim > 1 else data
            # Use specified reshape order (Fortran for MATLAB, C for NumPy default)
            sample = sample.reshape(dims, order=reshape_order)
        elif data_type == "xyz_vector":
            sample = data[:, 2] if data.ndim > 1 else data
            # Use specified reshape order (Fortran for MATLAB, C for NumPy default)
            sample = sample.reshape(dims, order=reshape_order)
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


def _load_x_multifile_per_sample(sample_paths: List[List[str]], separator: Optional[str], num_headlines: int,
                                  data_type: str, dimensions: Optional[str], nway_flag: int, 
                                  transpose: Optional[bool], reshape_order: str = 'F') -> np.ndarray:
    """Load multi-way X data where each sample consists of multiple files.
    
    This function handles the case where each sample is stored across multiple files.
    For example, with dimensions [4, 3, 2] (excluding samples):
    - If data_type is vector: expects 4 values per file, 6 files per sample (3*2=6)
      Files fill dimensions from innermost to outermost (last dimension varies slowest)
    - If data_type is matrix: expects 4x3 matrices, 2 files per sample (one for each position in dim 2)
    
    Args:
        sample_paths: List of lists, where each inner list contains file paths for one sample
        separator: File separator character
        num_headlines: Number of header lines to skip
        data_type: Type of data in files ('x_vector', 'xy_vector', 'xyz_vector', 'x_matrix', 'xy_matrix')
        dimensions: Comma-separated dimension sizes (excluding sample dimension)
        nway_flag: Number of dimensions (excluding samples), must be >= 3
        transpose: Whether to transpose data
        reshape_order: 'F' for Fortran/MATLAB column-major (default) or 'C' for C row-major
        
    Returns:
        Tensor with shape (num_samples, dim1, dim2, ..., dimN)
    """
    if dimensions is None:
        raise ValueError("Dimensions are required for multi-file per sample loading")
    
    # Parse dimensions (excluding sample dimension)
    dims = [int(d) for d in dimensions.split(",")]
    if len(dims) != nway_flag:
        raise ValueError(f"Number of dimensions ({len(dims)}) doesn't match nway_flag ({nway_flag})")
    
    # Determine what shape of data each file contains based on data_type
    if data_type in ["x_vector", "xy_vector", "xyz_vector"]:
        # Vector types: each file contains a 1D array
        # The first dimension value tells us how many elements per file
        elements_per_file = dims[0]
        # Remaining dimensions tell us how many files per sample
        files_per_sample = int(np.prod(dims[1:]))
        file_shape = (elements_per_file,)
        # Shape to organize files into (excluding first dim which is from the vector)
        file_organization_shape = dims[1:]
    elif data_type in ["x_matrix", "xy_matrix"]:
        # Matrix types: each file contains a 2D array
        # First two dimensions tell us the matrix shape per file
        matrix_rows = dims[0]
        matrix_cols = dims[1]
        # Remaining dimensions tell us how many files per sample
        files_per_sample = int(np.prod(dims[2:])) if len(dims) > 2 else 1
        file_shape = (matrix_rows, matrix_cols)
        # Shape to organize files into (remaining dims after the matrix)
        file_organization_shape = dims[2:] if len(dims) > 2 else []
    else:
        raise ValueError(f"Unknown data_type for multi-file per sample loading: {data_type}")
    
    samples = []
    
    for sample_idx, file_list in enumerate(sample_paths):
        if not file_list:
            raise ValueError(f"No files provided for sample {sample_idx + 1}")
        
        if len(file_list) != files_per_sample:
            raise ValueError(f"Sample {sample_idx + 1}: expected {files_per_sample} files, got {len(file_list)}")
        
        # Load all files for this sample
        file_data_list = []
        for file_path in file_list:
            raw_data = _load_file(file_path, separator, num_headlines)
            
            # Extract the relevant data based on data_type
            if data_type == "x_vector":
                data = raw_data.flatten()
            elif data_type == "xy_vector":
                data = raw_data[:, 1] if raw_data.ndim > 1 else raw_data.flatten()
            elif data_type == "xyz_vector":
                data = raw_data[:, 2] if raw_data.ndim > 1 else raw_data.flatten()
            elif data_type == "x_matrix":
                data = raw_data
                if transpose:
                    data = data.T
            elif data_type == "xy_matrix":
                data = raw_data[:, 1::2]  # Every second column starting from second
                if transpose:
                    data = data.T
            
            file_data_list.append(data)
        
        # Organize the files into the sample tensor
        if data_type in ["x_vector", "xy_vector", "xyz_vector"]:
            # For vector types: stack vectors and reshape
            # file_data_list contains vectors of shape (elements_per_file,)
            # We need to arrange them into shape dims
            stacked = np.array(file_data_list)  # Shape: (files_per_sample, elements_per_file)
            
            # Reshape to the target dimensions
            # The organization is: innermost dimensions (last in dims) vary fastest
            # So if dims = [4, 3, 2], with 6 files:
            #   - Files 0-2 are for position 0 of dim 2, files 3-5 are for position 1 of dim 2
            #   - Within each group, file 0 is for pos 0 of dim 1, file 1 is for pos 1 of dim 1, etc.
            
            # Reshape stacked data: (files_per_sample, elements_per_file) -> (dim2, dim3, ..., dim1)
            # Then transpose to get (dim1, dim2, dim3, ...)
            if file_organization_shape:
                # Reshape to (file_organization_shape..., elements_per_file)
                intermediate_shape = list(file_organization_shape) + [elements_per_file]
                sample_tensor = stacked.reshape(intermediate_shape, order=reshape_order)
                # Move the last axis (elements) to the front
                sample_tensor = np.moveaxis(sample_tensor, -1, 0)
            else:
                sample_tensor = stacked.T  # Just transpose if no file organization needed
            
        elif data_type in ["x_matrix", "xy_matrix"]:
            # For matrix types: stack matrices along new dimensions
            # file_data_list contains matrices of shape (matrix_rows, matrix_cols)
            stacked = np.array(file_data_list)  # Shape: (files_per_sample, matrix_rows, matrix_cols)
            
            if file_organization_shape:
                # Reshape: (files_per_sample, rows, cols) -> (file_org_shape..., rows, cols)
                intermediate_shape = list(file_organization_shape) + [matrix_rows, matrix_cols]
                sample_tensor = stacked.reshape(intermediate_shape, order=reshape_order)
                # Move the matrix dimensions (last 2) to the front
                ndim = len(sample_tensor.shape)
                new_axes = list(range(ndim - 2, ndim)) + list(range(ndim - 2))
                sample_tensor = np.transpose(sample_tensor, new_axes)
            else:
                sample_tensor = stacked[0]  # Only one file, just use the matrix
        
        samples.append(sample_tensor)
    
    # Stack all samples into final tensor with sample dimension first
    X = np.array(samples)
    return X


def _normalize_data_path(data_path) -> List[str]:
    """
    Normalize data_path to a list of file paths.
    
    Args:
        data_path: Can be a list of paths, a semicolon-separated string, or a single path
        
    Returns:
        List of file path strings
    """
    if data_path is None:
        return []
    
    if isinstance(data_path, str):
        # Could be semicolon-separated or a single path
        if ';' in data_path:
            return [p.strip() for p in data_path.split(';') if p.strip()]
        else:
            return [data_path.strip()] if data_path.strip() else []
    elif isinstance(data_path, list):
        result = []
        for item in data_path:
            if isinstance(item, str):
                result.append(item.strip())
            else:
                result.append(str(item))
        return [p for p in result if p]
    else:
        return [str(data_path)]


def _load_y_data(y_path: str, separator: Optional[str] = None, num_headlines: int = 0) -> np.ndarray:
    """Load Y data as 2D matrix using specified separator, preserving matrix structure."""
    data = _load_file(y_path, separator, num_headlines)
    # Keep as 2D if multiple columns, otherwise reshape to column vector
    if data.ndim == 1:
        return data.reshape(-1, 1)
    return data


def _load_class_data(cdata_path: str) -> List[Any]:
    """Load classification data, supporting single or multi-layer labels per sample row."""
    rows: List[List[str]] = []
    max_cols = 1

    with open(cdata_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if ',' in stripped:
                values = [token.strip() for token in stripped.split(',') if token.strip()]
            elif '\t' in stripped:
                values = [token.strip() for token in stripped.split('\t') if token.strip()]
            else:
                values = [token.strip() for token in re.split(r'\s+', stripped) if token.strip()]

            if not values:
                continue

            rows.append(values)
            max_cols = max(max_cols, len(values))

    if max_cols <= 1:
        return [row[0] for row in rows]

    normalized_rows: List[List[str]] = []
    for row in rows:
        if len(row) < max_cols:
            normalized_rows.append(row + ["" for _ in range(max_cols - len(row))])
        else:
            normalized_rows.append(row)
    return normalized_rows


def _parse_column_spec(spec: str) -> List[int]:
    """Parse a column specification string into a sorted list of 0-based column indices.

    Supported formats (1-based indexing):
    - Single value:           "3"      → [2]
    - Comma/space separated:  "1,3,5"  → [0, 2, 4]
    - Colon range:            "1:4"    → [0, 1, 2, 3]
    - Dash range:             "1-4"    → [0, 1, 2, 3]
    - Stepped colon range:    "2:2:6"  → [1, 3, 5]  (start:step:end)
    - Mixed:                  "1,3:5"  → [0, 2, 3, 4]
    """
    if not spec or not spec.strip():
        return []

    indices: set = set()
    # Split by commas and whitespace, but keep range tokens intact
    tokens = re.split(r'[\s,]+', spec.strip())

    for token in tokens:
        if not token:
            continue

        if ':' in token:
            # Colon-separated: a:b  or  a:step:b
            parts = token.split(':')
            try:
                if len(parts) == 2:
                    start, end = int(parts[0]), int(parts[1])
                    step = 1
                elif len(parts) == 3:
                    start, step, end = int(parts[0]), int(parts[1]), int(parts[2])
                else:
                    continue
                if step <= 0:
                    continue
                for i in range(start, end + 1, step):
                    if i >= 1:
                        indices.add(i - 1)  # convert to 0-based
            except ValueError:
                continue

        elif re.match(r'^\d+-\d+$', token):
            # Dash-separated range: a-b
            parts = token.split('-')
            try:
                start, end = int(parts[0]), int(parts[1])
                for i in range(start, end + 1):
                    if i >= 1:
                        indices.add(i - 1)
            except ValueError:
                continue

        else:
            # Single integer
            try:
                idx = int(token)
                if idx >= 1:
                    indices.add(idx - 1)
            except ValueError:
                continue

    return sorted(indices)


def _load_file_raw_rows(path: str, separator: Optional[str], num_headlines: int) -> List[List[str]]:
    """Load a file and return every data row as a list of string tokens.

    Headline rows are skipped.  Empty lines are ignored.
    Works for both text files and Excel files.
    """
    rows: List[List[str]] = []

    if path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path, header=None, skiprows=num_headlines, dtype=str)
        df = df.fillna('')
        for _, row in df.iterrows():
            tokens = [str(v).strip() for v in row.tolist()]
            # Drop trailing empty tokens
            while tokens and tokens[-1] == '':
                tokens.pop()
            if tokens:
                rows.append(tokens)
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()

        # Skip headline rows
        data_lines = all_lines[num_headlines:]

        for line in data_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if separator is None:
                tokens = re.split(r'\s+', stripped)
            elif separator == '\t':
                tokens = stripped.split('\t')
            else:
                tokens = stripped.split(separator)
            tokens = [t.strip() for t in tokens]
            if tokens:
                rows.append(tokens)

    return rows


def _load_x_matrix_with_extraction(
    data_paths: List[str],
    separator: Optional[str],
    num_headlines: int,
    transpose: bool,
    y_col_indices: List[int],
    class_col_indices: List[int],
    smp_col_idx: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[Any]], Optional[List[str]], List[int]]:
    """Load 1-way X matrix files, extracting Y, class, and/or sample-label columns before numeric conversion.

    Y columns are numeric; class and sample-label columns are kept as raw strings so that
    non-numeric labels survive the process unchanged.  All extracted columns are removed from X.

    Args:
        data_paths:        List of file paths for the X data.
        separator:         Parsed separator character (or None for whitespace).
        num_headlines:     Number of header rows to skip.
        transpose:         Whether to transpose each file's data.
        y_col_indices:     0-based original column indices to use as Y (must be numeric).
        class_col_indices: 0-based original column indices to use as class labels.
        smp_col_idx:       0-based column index whose values become sample labels, or None.

    Returns:
        X:             Processed X matrix (samples × remaining variables).
        Y:             Y matrix (samples × y_cols) or None.
        class_data:    List of class labels (1-D list of strings or 2-D list) or None.
        smp_labels:    List of per-row sample label strings, or None.
        row_counts:    Number of rows contributed by each file.
    """
    remove_set = set(y_col_indices) | set(class_col_indices)
    if smp_col_idx is not None:
        remove_set.add(smp_col_idx)

    X_blocks: List[np.ndarray] = []
    Y_rows: List[List[float]] = []
    class_rows: List[List[str]] = []
    smp_rows: List[str] = []
    row_counts: List[int] = []

    y_only = [c for c in sorted(y_col_indices) if c not in set(class_col_indices)]

    for path in data_paths:
        raw = _load_file_raw_rows(path, separator, num_headlines)
        if not raw:
            continue

        n_cols = max(len(row) for row in raw)
        x_col_indices = [c for c in range(n_cols) if c not in remove_set]

        x_block: List[List[float]] = []
        for row in raw:
            # --- extract sample label value (single string column) ---
            if smp_col_idx is not None:
                smp_rows.append(row[smp_col_idx] if smp_col_idx < len(row) else '')

            # --- extract class values (strings) ---
            if class_col_indices:
                c_vals = []
                for ci in sorted(class_col_indices):
                    c_vals.append(row[ci] if ci < len(row) else '')
                class_rows.append(c_vals)

            # --- extract numeric X values ---
            x_vals: List[float] = []
            for ci in x_col_indices:
                raw_val = row[ci] if ci < len(row) else ''
                try:
                    x_vals.append(float(raw_val) if raw_val != '' else np.nan)
                except ValueError:
                    x_vals.append(np.nan)
            x_block.append(x_vals)

            # --- extract Y values ---
            if y_only:
                y_vals: List[float] = []
                for ci in y_only:
                    raw_val = row[ci] if ci < len(row) else ''
                    try:
                        y_vals.append(float(raw_val) if raw_val != '' else np.nan)
                    except ValueError:
                        y_vals.append(np.nan)
                Y_rows.append(y_vals)

        X_file = np.array(x_block, dtype=float)
        if transpose:
            X_file = X_file.T
        X_blocks.append(X_file)
        row_counts.append(X_file.shape[0])

    X = np.concatenate(X_blocks, axis=0) if X_blocks else np.empty((0, 0))

    Y: Optional[np.ndarray] = None
    if Y_rows:
        Y = np.array(Y_rows, dtype=float)
        if Y.ndim == 1 or (Y.ndim == 2 and Y.shape[1] == 1):
            Y = Y.reshape(-1, 1)

    class_data: Optional[List[Any]] = None
    if class_rows:
        max_c = max(len(r) for r in class_rows)
        if max_c <= 1:
            class_data = [r[0] if r else '' for r in class_rows]
        else:
            # Pad shorter rows
            class_data = [r + [''] * (max_c - len(r)) for r in class_rows]

    smp_labels_out: Optional[List[str]] = smp_rows if smp_rows else None

    return X, Y, class_data, smp_labels_out, row_counts


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


def _safe_stat(file_path: str, source_metadata_overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Return basic filesystem metadata for a file path."""
    abs_path = os.path.abspath(file_path)

    if isinstance(source_metadata_overrides, dict):
        override_entry = source_metadata_overrides.get(abs_path)
        if override_entry is None:
            override_entry = source_metadata_overrides.get(abs_path.replace('\\', '/'))
        if isinstance(override_entry, dict):
            return dict(override_entry)

    metadata: Dict[str, Any] = {
        "file_path": abs_path,
        "file_name": os.path.basename(abs_path),
        "file_stem": os.path.splitext(os.path.basename(abs_path))[0],
        "file_extension": os.path.splitext(abs_path)[1]
    }

    try:
        stat_info = os.stat(abs_path)
        metadata["file_size_bytes"] = stat_info.st_size
        metadata["created_time"] = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
        metadata["modified_time"] = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
    except OSError:
        metadata["file_size_bytes"] = None
        metadata["created_time"] = None
        metadata["modified_time"] = None

    return metadata


def _unique_sample_key(metadata: Dict[str, Dict[str, Any]], base_label: str, sample_index: int) -> str:
    """Create a unique dictionary key for metadata entries."""
    key = str(base_label).strip() if str(base_label).strip() else f"sample_{sample_index}"
    if key not in metadata:
        return key

    suffix = 2
    while f"{key}__{suffix}" in metadata:
        suffix += 1
    return f"{key}__{suffix}"


def _build_sample_metadata(smp_labels: List[str], data_paths: List[str], row_counts: List[int],
                           nway_flag: int, data_type: str, multi_file_per_sample: bool,
                           sample_paths: Optional[List[List[str]]] = None,
                           source_metadata_overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """Build per-sample metadata dictionary extracted from source file metadata."""
    metadata: Dict[str, Dict[str, Any]] = {}

    if multi_file_per_sample and sample_paths:
        for sample_index, sample_label in enumerate(smp_labels):
            files_for_sample = sample_paths[sample_index] if sample_index < len(sample_paths) else []
            sample_index_1b = sample_index + 1
            key = _unique_sample_key(metadata, sample_label, sample_index_1b)
            metadata[key] = {
                "sample_index": sample_index_1b,
                "sample_label": sample_label,
                "source_mode": "multi_file_per_sample",
                "source_files": [_safe_stat(path, source_metadata_overrides) for path in files_for_sample]
            }
        return metadata

    if nway_flag == 1 and data_type == "x_matrix" and row_counts:
        running_index = 0
        for path, count in zip(data_paths, row_counts):
            file_meta = _safe_stat(path, source_metadata_overrides)
            for row_index in range(count):
                if running_index >= len(smp_labels):
                    break
                sample_label = smp_labels[running_index]
                sample_index_1b = running_index + 1
                key = _unique_sample_key(metadata, sample_label, sample_index_1b)
                metadata[key] = {
                    "sample_index": sample_index_1b,
                    "sample_label": sample_label,
                    "source_mode": "matrix_row",
                    "source_file": file_meta,
                    "row_index_in_file": row_index
                }
                running_index += 1

        while running_index < len(smp_labels):
            sample_label = smp_labels[running_index]
            sample_index_1b = running_index + 1
            key = _unique_sample_key(metadata, sample_label, sample_index_1b)
            metadata[key] = {
                "sample_index": sample_index_1b,
                "sample_label": sample_label,
                "source_mode": "unknown"
            }
            running_index += 1
        return metadata

    for sample_index, sample_label in enumerate(smp_labels):
        sample_path = data_paths[sample_index] if sample_index < len(data_paths) else None
        sample_index_1b = sample_index + 1
        key = _unique_sample_key(metadata, sample_label, sample_index_1b)
        metadata[key] = {
            "sample_index": sample_index_1b,
            "sample_label": sample_label,
            "source_mode": "single_file" if sample_path else "unknown",
            "source_file": _safe_stat(sample_path, source_metadata_overrides) if sample_path else None
        }

    return metadata


def _generate_default_axis_info(X: np.ndarray) -> List[np.ndarray]:
    """
    Generate default axis vectors (1 to dimension size) for each axis.
    
    Args:
        X: Data array to match dimensions
        
    Returns:
        List of numpy arrays with values from 1 to dimension size for each axis
    """
    # Generate axis vectors from 1 to dimension size for ALL dimensions (including samples)
    axis_vectors = []
    for dim_size in X.shape:
        axis_vector = np.arange(1, dim_size + 1, dtype=float)
        axis_vectors.append(axis_vector)
    
    return axis_vectors


def _generate_axis_info(axis_info, X: np.ndarray, scale_type: Optional[List[str]] = None) -> Optional[List[np.ndarray]]:
    """
    Generate axis vectors from axis information with optional scale types.
    
    Args:
        axis_info: List of axis ranges (e.g., ["100 200", "1 10"]) or semicolon-separated string
        X: Data array to match dimensions
        scale_type: Optional list of scale types ('Linear', 'Log10', 'Log2', 'Ln') for each dimension
        
    Returns:
        List of numpy arrays representing axis vectors, or None if parsing fails
    """
    if not axis_info:
        return None
    
    try:
        # Normalize axis_info to list
        if isinstance(axis_info, str):
            axis_specs = [spec.strip() for spec in axis_info.split(';') if spec.strip()]
        elif isinstance(axis_info, list):
            axis_specs = [spec.strip() if isinstance(spec, str) else spec for spec in axis_info if spec]
        else:
            return None
        
        # Filter out empty specs
        axis_specs = [spec for spec in axis_specs if spec]
        
        if not axis_specs:
            return None
        
        # Get data shape (excluding sample dimension)
        data_shape = X.shape[1:]  # Skip first dimension (samples)
        
        # Check if number of axis specs matches data dimensions (excluding samples)
        if len(axis_specs) != len(data_shape):
            print(f"Warning: Number of axis specifications ({len(axis_specs)}) doesn't match data dimensions ({len(data_shape)})")
            return None
        
        # Generate axis vectors
        axis_vectors = []
        
        # First vector: sample indices (auto-generated)
        sample_vector = np.arange(1, X.shape[0] + 1, dtype=float)
        axis_vectors.append(sample_vector)
        
        # Remaining vectors: from axis_info specifications
        for i, (spec, dim_size) in enumerate(zip(axis_specs, data_shape)):
            parts = spec.split()
            if len(parts) != 2:
                print(f"Warning: Invalid axis specification '{spec}'. Expected format: 'start end'")
                return None
            
            try:
                start = float(parts[0])
                end = float(parts[1])
                
                # Get scale type for this dimension (default to Linear)
                current_scale = 'Linear'
                if scale_type and i < len(scale_type) and scale_type[i]:
                    current_scale = scale_type[i]
                
                axis_vector = _generate_scaled_vector(start, end, dim_size, current_scale)
                axis_vectors.append(axis_vector)
            except ValueError:
                print(f"Warning: Could not parse axis values from '{spec}'")
                return None
        
        return axis_vectors
    
    except Exception as e:
        print(f"Error generating axis information: {e}")
        return None


def _generate_scaled_vector(start: float, end: float, num_points: int, scale_type: str) -> np.ndarray:
    """
    Generate a vector from start to end with the specified scale type.
    
    Args:
        start: Start value
        end: End value
        num_points: Number of points in the vector
        scale_type: Scale type ('Linear', 'Log10', 'Log2', 'Ln')
        
    Returns:
        Numpy array with scaled values
    """
    if scale_type == 'Linear' or not scale_type:
        return np.linspace(start, end, num_points)
    elif scale_type == 'Log10':
        # Generate logarithmically spaced values (base 10)
        if start <= 0 or end <= 0:
            print(f"Warning: Log10 scale requires positive values. Using linear scale instead.")
            return np.linspace(start, end, num_points)
        return np.logspace(np.log10(start), np.log10(end), num_points)
    elif scale_type == 'Log2':
        # Generate logarithmically spaced values (base 2)
        if start <= 0 or end <= 0:
            print(f"Warning: Log2 scale requires positive values. Using linear scale instead.")
            return np.linspace(start, end, num_points)
        return np.logspace(np.log2(start), np.log2(end), num_points, base=2)
    elif scale_type == 'Ln':
        # Generate logarithmically spaced values (natural log)
        if start <= 0 or end <= 0:
            print(f"Warning: Ln scale requires positive values. Using linear scale instead.")
            return np.linspace(start, end, num_points)
        return np.logspace(np.log(start), np.log(end), num_points, base=np.e)
    else:
        print(f"Warning: Unknown scale type '{scale_type}'. Using linear scale.")
        return np.linspace(start, end, num_points)


def _load_axis_text_info(var_path: Optional[List[str]], smp_labels: List[str], nway_flag: int, 
                          axis_n_info: Optional[List[np.ndarray]]) -> List[List[str]]:
    """
    Load axis text labels from files and build axis_t_info structure.
    
    Args:
        var_path: List of file paths for axis labels (one per dimension)
        smp_labels: Sample labels (used for position 0)
        nway_flag: Number of dimensions (excluding samples)
        axis_n_info: Axis numerical vectors (used for fallback label generation)
        
    Returns:
        List of string lists, where position 0 is sample labels and subsequent positions
        are from loaded files, or fallback "V1, V2, ..." labels if not provided
    """
    # Initialize axis_t_info with sample labels as first element
    axis_t_info = [smp_labels.copy() if smp_labels else []]
    
    # Initialize remaining positions with empty lists (will be filled or get fallback)
    for i in range(nway_flag):
        axis_t_info.append([])
    
    # If var_path is provided, load files
    if var_path:
        for i, path in enumerate(var_path):
            if i >= nway_flag:
                break  # Don't exceed nway_flag dimensions
            
            if path and path.strip():
                try:
                    # Try to load the file content
                    file_content = _load_axis_file_content(path)
                    
                    if file_content is not None and len(file_content) > 0:
                        is_numeric, values = _check_if_numeric(file_content)
                        
                        if is_numeric and axis_n_info is not None:
                            # Override the corresponding axis_n_info vector
                            # Position in axis_n_info is i+1 (0 is samples)
                            if i + 1 < len(axis_n_info):
                                axis_n_info[i + 1] = values
                        
                        # Always store as text labels (numeric values become strings)
                        axis_t_info[i + 1] = file_content
                except Exception as e:
                    print(f"Warning: Could not load axis labels from '{path}': {e}")
                    axis_t_info[i + 1] = []
    
    # Generate fallback labels for any dimension that has empty list
    for i in range(1, len(axis_t_info)):
        if not axis_t_info[i]:  # Empty list
            # Determine size from axis_n_info if available
            dim_size = 0
            if axis_n_info is not None and i < len(axis_n_info) and axis_n_info[i] is not None:
                dim_size = len(axis_n_info[i])
            
            if dim_size > 0:
                # Generate V1, V2, V3, ... labels
                axis_t_info[i] = [f"V{j + 1}" for j in range(dim_size)]
    
    return axis_t_info


def _load_axis_file_content(path: str) -> Optional[List[str]]:
    """
    Load content from an axis labels file.
    
    Args:
        path: Path to the file
        
    Returns:
        List of strings (one per line) or None if loading fails
    """
    try:
        with open(path, 'r') as f:
            content = [line.strip() for line in f if line.strip()]
        return content
    except Exception as e:
        print(f"Error loading axis file '{path}': {e}")
        return None


def _check_if_numeric(content: List[str]) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Check if file content is numerical and convert to numpy array if so.
    
    Args:
        content: List of strings from file
        
    Returns:
        Tuple of (is_numeric, numpy_array or None)
    """
    try:
        values = []
        for item in content:
            # Try to parse as float
            val = float(item)
            values.append(val)
        return True, np.array(values, dtype=float)
    except (ValueError, TypeError):
        return False, None


def _generate_dim_labels(dim_labels, nway_flag: int) -> List[str]:
    """
    Generate dimension labels from user input.
    
    Args:
        dim_labels: List of dimension names or comma-separated string (e.g., "wavelength,time")
        nway_flag: Number of dimensions (excluding samples)
        
    Returns:
        List of dimension labels starting with "Samples" followed by user-provided names
        or position indices for empty values
    """
    # Start with "Samples" as the first label
    result = ["Samples"]
    
    # Normalize dim_labels to list (handle both string and list inputs)
    user_labels = []
    if dim_labels:
        if isinstance(dim_labels, str):
            # Parse from comma-separated string
            user_labels = [label.strip() for label in dim_labels.split(',') if label.strip()]
        elif isinstance(dim_labels, list):
            user_labels = [label.strip() if isinstance(label, str) else label for label in dim_labels]
    
    # Build labels for each dimension
    for i in range(nway_flag):
        if i < len(user_labels) and user_labels[i]:
            # Use user-provided label
            result.append(user_labels[i])
        else:
            # Use position index as string (1-based)
            result.append(str(i + 1))
    
    return result
