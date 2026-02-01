
from typing import Tuple, Optional, List
import numpy as np
import os
import csv
import pandas as pd


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
            
            # Check for NaN values (from conversion errors or empty cells)
            if df.isna().any().any():
                # Drop columns or rows with NaN if appropriate, or fill with 0
                # For now, replace NaN with 0
                df = df.fillna(0)
            
            return df.values
        except Exception as e:
            # Fallback to numpy loadtxt
            print(f"Warning: pandas read failed, falling back to numpy: {e}")
            return np.loadtxt(path, delimiter=separator, skiprows=num_headlines)


def load_data(d_specs_separator: str, d_specs_headlines: str, d_specs_type: str, d_specs_dimensions: Optional[str] = None,
              data_path: Optional[List[str]] = None, nway_flag: int = 1, y_path: Optional[str] = None,
              var_path: Optional[List[str]] = None, smp_path: Optional[str] = None,
              transpose: bool = False, axis_info: Optional[List[str]] = None, reshape_order: str = 'F',
              dim_labels: Optional[str] = None, scale_type: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[List[str]]], List[str], Optional[List[np.ndarray]], List[str]]:
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
        var_path: Optional list of paths to axis labels files (one per dimension, based on nway_flag)
        smp_path: Optional path to sample labels file (text)
        transpose: Whether to transpose data (defaults to False)
        axis_info: Optional list of axis ranges (e.g., ["100 200", "1 10"]) or semicolon-separated string
        reshape_order: Reshape order for multiway data - 'F' (Fortran/MATLAB column-major, default) or 'C' (C row-major)
        dim_labels: Optional comma-separated dimension names (e.g., "wavelength,time")
        scale_type: Optional list of scale types for axis generation ('Linear', 'Log10', 'Log2', 'Ln')

    Returns:
        X_cal: X data array
        Y_cal: Y data array or None
        axis_t_info: List of string lists containing axis text labels for each dimension
        smp_cal: Sample labels
        axis_n_info: List of axis vectors matching data dimensions or None
        dim_labels: List of dimension labels
    """
    # Parse d_specs parameters
    separator_map = {"comma": ",", "spaces": None, "tabs": "\t"}
    separator = separator_map.get(d_specs_separator, ",")
    num_headlines = int(d_specs_headlines)
    data_type = d_specs_type
    dimensions = d_specs_dimensions if d_specs_dimensions and d_specs_dimensions.strip() else None

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

    # Load X data
    X, row_counts = _load_x_data(data_path, separator, num_headlines, data_type, dimensions, transpose, nway_flag, reshape_order)

    # Load Y data (using same separator as X for consistency)
    Y = _load_y_data(y_path, separator, 0) if y_path else None

    # Load sample labels
    if smp_path is None:
        if nway_flag == 1 and data_type == "x_matrix":
            smp_labels = _generate_row_labels(data_path, row_counts)
        else:
            smp_labels = _generate_sample_labels(data_path)
    else:
        smp_labels = _load_labels(smp_path)

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

    return X, Y, axis_t_info, smp_labels, axis_n_info, processed_dim_labels


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
        axis_n_info: Axis numerical vectors (may be overridden if file contains numerical data)
        
    Returns:
        List of string lists, where position 0 is sample labels and subsequent positions
        are from loaded files or empty lists if not provided
    """
    # Initialize axis_t_info with sample labels as first element
    axis_t_info = [smp_labels.copy() if smp_labels else []]
    
    # Initialize remaining positions based on nway_flag
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
                    
                    if file_content is not None:
                        is_numeric, values = _check_if_numeric(file_content)
                        
                        if is_numeric and axis_n_info is not None:
                            # Override the corresponding axis_n_info vector
                            # Position in axis_n_info is i+1 (0 is samples)
                            if i + 1 < len(axis_n_info):
                                axis_n_info[i + 1] = values
                            # Store empty list in axis_t_info for this position
                            axis_t_info[i + 1] = []
                        else:
                            # Store as text labels
                            axis_t_info[i + 1] = file_content
                except Exception as e:
                    print(f"Warning: Could not load axis labels from '{path}': {e}")
                    axis_t_info[i + 1] = []
    
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
