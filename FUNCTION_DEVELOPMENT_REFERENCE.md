# Chemometric Studio - Function Development Reference

**Version 1.0** | Last Updated: January 28, 2026

This document serves as a comprehensive reference guide for developing new functions in Chemometric Studio. It covers architecture, file creation, JSON specifications, multi-language support, and code patterns.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Function Development Workflow](#function-development-workflow)
4. [Step 1: Create Core Function](#step-1-create-core-function)
5. [Step 2: Update Function Specifications](#step-2-update-function-specifications)
6. [Step 3: Create GUI Config Files](#step-3-create-gui-config-files)
7. [Step 4: Add Language Support](#step-4-add-language-support)
8. [JSON Format References](#json-format-references)
9. [Code Pattern Examples](#code-pattern-examples)
10. [Testing and Validation](#testing-and-validation)

---

## Architecture Overview

ChemometricsTool is a modular chemometrics analysis platform with:

- **Backend**: Core computational functions in `/chemometrics/` package
- **Frontend**: Tkinter-based GUI with Sun-Valley theme styling
- **Configuration**: JSON-based function specs and GUI configurations
- **Localization**: Multi-language support via JSON translation files
- **Pipeline**: Automatic routing and data flow between functions

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Core Functions | `chemometrics/` | Scientific computation modules |
| Function Registry | `function_specs.json` | Central specification registry |
| GUI Configs | `gui_configs/{lang}/` | UI element definitions per language |
| Language Files | `languages/` | Translation strings |
| Main GUI | `main_gui.py` | GUI application controller |
| Language Manager | `language_manager.py` | Translation and locale management |

---

## Project Structure

```
ChemometricsTool/
├── chemometrics/                    # Core computation package
│   ├── __init__.py
│   ├── data_input.py               # Data loading functions
│   ├── data_processing.py          # Preprocessing functions
│   ├── processing.py               # Additional processing
│   ├── univ_calibration.py         # Calibration models
│   ├── validation_data_input.py    # Validation logic
│   └── reporting.py                # Report generation
│
├── gui_configs/                     # GUI configuration per language
│   ├── en/
│   │   ├── load_data_config.json
│   │   ├── baseline_correction_config.json
│   │   ├── smoothing_config.json
│   │   ├── center_normalize_config.json
│   │   ├── univariate_calibration_config.json
│   │   └── validation_data_config.json
│   └── pt-br/
│       └── (same structure in Portuguese)
│
├── languages/                       # Translation files
│   ├── en.json                     # English translations
│   └── pt-br.json                  # Brazilian Portuguese translations
│
├── function_specs.json             # Master function registry
├── main_gui.py                     # Main GUI application
├── language_manager.py             # Language management
├── launcher.py                     # Application entry point
├── pyproject.toml                  # Project metadata
└── requirements.txt                # Python dependencies
```

---

## Function Development Workflow

### Overview Steps

When adding a new function:

1. ✅ Create Python function in `chemometrics/` module
2. ✅ Register function in `function_specs.json`
3. ✅ Create GUI config JSON for each supported language
4. ✅ Add translation strings to language files
5. ✅ Test integration and routing
6. ✅ Update documentation

---

## Step 1: Create Core Function

### Location
Create function in the appropriate `chemometrics/` submodule:

- **Data I/O**: `chemometrics/data_input.py`
- **Preprocessing**: `chemometrics/data_processing.py`
- **Calibration**: `chemometrics/univ_calibration.py`
- **Validation**: `chemometrics/validation_data_input.py`
- **Reporting**: `chemometrics/reporting.py`
- **Other**: `chemometrics/processing.py`

### Function Signature Requirements

```python
from typing import Tuple, Optional, List, Union, Dict, Any
import numpy as np

def my_new_function(
    input_param_1: np.ndarray,
    input_param_2: str,
    optional_param_1: Optional[np.ndarray] = None,
    optional_param_2: int = 5
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Brief description of what the function does.
    
    Detailed explanation including:
    - Purpose and use case
    - Input data structure requirements
    - Processing steps
    - Output format and interpretation
    
    Args:
        input_param_1: Description and expected format
        input_param_2: Description and expected format
        optional_param_1: Description (default: None)
        optional_param_2: Description (default: 5)
    
    Returns:
        If single output: np.ndarray
        If multiple outputs: Tuple of arrays/values
        
    Raises:
        ValueError: If inputs don't meet requirements
        TypeError: If input types are incorrect
    
    Examples:
        >>> import numpy as np
        >>> X = np.random.randn(100, 500)
        >>> result = my_new_function(X, "method1")
    """
    # Input validation
    if not isinstance(input_param_1, np.ndarray):
        raise TypeError("input_param_1 must be numpy array")
    
    if input_param_1.ndim < 2:
        raise ValueError("input_param_1 must be 2D or higher")
    
    # Function implementation
    # ...
    
    return output  # or (output1, output2, ...)
```

### Best Practices

1. **Type Hints**: Always include complete type annotations
2. **NumPy Arrays**: Use `np.ndarray` for data arrays
3. **Optional Validation**: Handle validation-specific data (X_val) gracefully
4. **Inheritance Support**: Accept `nway_flag`, `direction` for multiway data
5. **Return Consistency**: 
   - Single output: return array directly
   - Multiple outputs: return tuple `(output1, output2, ...)`
6. **Documentation**: Include docstring with Args, Returns, Raises sections

### Common Input/Output Patterns

**Pattern 1: Processing with Optional Validation Data**
```python
def my_processor(
    X_cal: np.ndarray,
    param1: str,
    X_val: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # Process calibration data
    X_cal_processed = process(X_cal, param1)
    
    # If validation data provided, process it too
    if X_val is not None:
        X_val_processed = process(X_val, param1)
        return X_cal_processed, X_val_processed
    
    return X_cal_processed
```

**Pattern 2: Function with Multiple Outputs**
```python
def my_model(
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    degree: int = 1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Build model
    model = build_model(X_cal, Y_cal, degree)
    
    # Calculate metrics
    metrics = compute_metrics(model)
    
    return model, metrics
```

---

## Step 2: Update Function Specifications

### File: `function_specs.json`

This is the central registry for all functions. Each function must have entries in four sections.

### Structure

```json
{
  "return_specs": {
    "function_name": ["output1", "output2", "..."]
  },
  "input_specs": {
    "function_name": ["input1", "input2", "..."]
  },
  "import_map": {
    "function_name": ["module.path", "function_name"]
  },
  "gui_listing": {
    "function_name": {
      "config_path": "gui_configs/function_name_config.json"
    }
  }
}
```

### Detailed Example

For function `baseline_correction` in `chemometrics/data_processing.py`:

```json
{
  "return_specs": {
    "baseline_correction": ["X_cal", "X_val"]
  },
  "input_specs": {
    "baseline_correction": ["X_cal", "method", "X_val", "nway_flag", "direction", "window_size"]
  },
  "import_map": {
    "baseline_correction": ["chemometrics.data_processing", "baseline_correction"]
  },
  "gui_listing": {
    "baseline_correction": {
      "config_path": "gui_configs/baseline_correction_config.json"
    }
  }
}
```

### Important Notes

1. **Function Name**: Must exactly match Python function name
2. **Module Path**: Use dot notation (e.g., `chemometrics.data_processing`)
3. **Return Specs**: List outputs in order they're returned
4. **Input Specs**: List ALL inputs (required and optional)
5. **Config Path**: Path relative to workspace root, includes language code

---

## Step 3: Create GUI Config Files

### Location
Create files in: `gui_configs/{language}/function_name_config.json`

Must be created for **each supported language**:
- `gui_configs/en/function_name_config.json`
- `gui_configs/pt-br/function_name_config.json`

### Input Type System

Every parameter in the setup layout must specify an `input_type`:

- **`"user"`**: User-provided input (visible in Setup tab) - methods, file paths, numeric parameters
- **`"routed"`**: Data from previous function output (configured in Routing tab) - X_cal, Y_cal, X_val, etc.
- **`"inherited"`**: Auto-passed from previous function (not user-visible) - nway_flag, direction

Only `"user"` type inputs appear in the Setup tab. The others are hidden automatically.

### Complete Example: `baseline_correction_config.json`

```json
{
  "display_name": "Baseline Correction",
  "category": "Preprocessing",
  "data_type": "firstorder",
  
  "input_aliases": {
    "X_cal": "X Calibration",
    "method": "Method",
    "X_val": "X Validation",
    "nway_flag": "Dimensionality",
    "direction": "Direction",
    "window_size": "Window Size"
  },
  
  "output_aliases": {
    "X_cal": "X Cal (Corrected)",
    "X_val": "X Val (Corrected)"
  },
  
  "short_description": "Removes baseline offset from spectra",
  
  "long_description": "Baseline correction removes systematic offsets and trends from spectroscopic measurements, improving data quality and model performance.\n\nAvailable Methods:\n\n1. Multiplicative Scatter Correction (MSC)\n   - Corrects for multiplicative and additive scatter effects\n   - Works well for diffuse reflectance and transmittance data\n\n2. Standard Normal Variate (SNV)\n   - Center-scales each spectrum independently\n   - Removes baseline drift and multiplicative effects\n\n3. Moving Average\n   - Simple smoothing-based baseline correction\n   - Uses configurable window size\n\nMultiway Data Support:\n- For 2D data: standard processing (nway_flag=1)\n- For 3D+ data: specify dimensionality and direction",
  
  "setup": {
    "layout": [
      {
        "name": "method",
        "label": "Correction Method",
        "widget": "combobox",
        "values": ["msc", "svn", "moving_average"],
        "value_aliases": ["Multiplicative Scatter Correction", "Standard Normal Variate", "Moving Average"],
        "default": "msc",
        "required": true,
        "input_type": "user",
        "tooltip": "Select baseline correction method"
      },
      {
        "name": "direction",
        "label": "Direction (for multiway)",
        "widget": "entry",
        "type": "int",
        "default": "-1",
        "required": false,
        "input_type": "user",
        "tooltip": "Dimension for correction (-1 = default/last)"
      },
      {
        "name": "window_size",
        "label": "Window Size (for moving average)",
        "widget": "entry",
        "type": "int",
        "default": "5",
        "required": false,
        "input_type": "user",
        "tooltip": "Points in moving average window",
        "visible_if": {"method": "moving_average"}
      },
      {
        "name": "X_cal",
        "label": "X Calibration Data",
        "widget": "data_input",
        "required": true,
        "input_type": "routed",
        "tooltip": "Routed from previous function output"
      },
      {
        "name": "X_val",
        "label": "X Validation Data",
        "widget": "data_input",
        "required": false,
        "input_type": "routed",
        "tooltip": "Routed from previous function output"
      },
      {
        "name": "nway_flag",
        "label": "Data Dimensionality",
        "widget": "entry",
        "type": "int",
        "default": "1",
        "required": true,
        "input_type": "inherited",
        "tooltip": "Automatically inherited from Load Data function"
      }
    ]
  }
}
```

### Config Field Reference

#### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `display_name` | string | Yes | User-facing function name |
| `category` | string | Yes | Function category (Data Import, Preprocessing, Calibration, etc.) |
| `data_type` | string | Yes | Data format (all, firstorder, secondorder, etc.) |
| `input_aliases` | object | Yes | User-friendly names for inputs |
| `output_aliases` | object | Yes | User-friendly names for outputs |
| `short_description` | string | Yes | One-line description |
| `long_description` | string | Yes | Detailed multi-line explanation |
| `setup.layout` | array | Yes | Array of widget configuration objects |

#### Widget Configuration

Each item in `setup.layout` can have:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Parameter name (must match function signature) |
| `label` | string | Yes | Display label |
| `widget` | string | Yes | Widget type (see Widget Types below) |
| `input_type` | string | Yes | Type of input: `"user"` (user-provided), `"routed"` (from routing), or `"inherited"` (auto-passed from previous function). Only `"user"` inputs appear in Setup tab |
| `default` | string/number | No | Default value |
| `required` | boolean | Yes | If parameter is required |
| `tooltip` | string | No | Help text on hover |
| `visible_if` | object | No | Conditional visibility (only for user inputs) |
| `type` | string | No | Data type (int, float, str) |
| `values` | array | No | Options for combobox/radiobutton |
| `value_aliases` | array | No | Display names for values |
| `multiple` | boolean | No | Multiple selection allowed |
| `ispath` | boolean | No | File/path selector |

#### Widget Types

| Widget Type | Usage | Config Example |
|------------|-------|-----------------|
| `combobox` | Dropdown selection | `{"values": ["opt1", "opt2"], "value_aliases": ["Option 1", "Option 2"]}` |
| `entry` | Text/numeric input | `{"type": "int", "default": "5"}` |
| `spinbox` | Numeric spinner | `{"type": "int", "from": 1, "to": 100}` |
| `checkbutton` | Toggle checkbox | `{"default": "true"}` |
| `radiobutton` | Radio group | `{"values": ["opt1", "opt2"]}` |
| `file_selector` | File picker | `{"multiple": true, "ispath": true}` |
| `data_input` | Routing data source | `{"inheritable": true}` |

#### Conditional Visibility

```json
"visible_if": {"parameter_name": "value"}
```

Example: Show `window_size` only when `method` is "moving_average":
```json
"visible_if": {"method": "moving_average"}
```

This only applies to `"user"` type inputs.

---

## Understanding Input Types

Each parameter in a function's setup layout must specify an `input_type` that determines how it's handled:

### `input_type: "user"`

**Purpose**: User provides this value in the Setup tab

**Characteristics**:
- Visible in the Setup tab UI
- User configures directly (combobox, entry, file selector, etc.)
- Can have conditional visibility with `visible_if`
- Examples: method selection, window size, file paths, degree parameters

**Example**:
```json
{
  "name": "method",
  "label": "Method",
  "widget": "combobox",
  "values": ["msc", "svn"],
  "input_type": "user",
  "tooltip": "Choose processing method"
}
```

### `input_type: "routed"`

**Purpose**: Data comes from a previous function's output through the Routing tab

**Characteristics**:
- NOT visible in Setup tab (hidden automatically)
- Configured in Routing tab by connecting outputs to inputs
- Always uses `widget: "data_input"`
- Examples: X_cal, X_val, Y_cal (data arrays passed between functions)

**Example**:
```json
{
  "name": "X_cal",
  "label": "Calibration Data",
  "widget": "data_input",
  "required": true,
  "input_type": "routed",
  "tooltip": "Routed from previous function"
}
```

### `input_type: "inherited"`

**Purpose**: Parameter automatically passed from a previous function (same value)

**Characteristics**:
- NOT visible in Setup tab (hidden automatically)
- Automatically inherits value from previous function that produces it
- Examples: nway_flag, direction (metadata about data structure)
- Useful for maintaining consistency across the pipeline

**Example**:
```json
{
  "name": "nway_flag",
  "label": "Data Dimensionality",
  "widget": "entry",
  "type": "int",
  "input_type": "inherited",
  "tooltip": "Automatically inherited"
}
```

---

## Conditional Visibility

```json
"visible_if": {"parameter_name": "value"}
```

Example: Show `window_size` only when `method` is "moving_average":
```json
"visible_if": {"method": "moving_average"}
```

This only applies to `"user"` type inputs.

---

## Step 4: Add Language Support

### File Structure

Language files are JSON dictionaries with hierarchical key structure.

Location: `languages/{language_code}.json`

Examples:
- `languages/en.json` - English
- `languages/pt-br.json` - Brazilian Portuguese

### Structure Pattern

Use dot notation for keys: `section.subsection.key`

```json
{
  "ui": {
    "main_title": "ChemometricsTool",
    "buttons": {
      "save_model": "Save Model",
      "load_model": "Load Model"
    }
  },
  "messages": {
    "no_methodology": "No functions selected...",
    "error": "Error"
  },
  "functions": {
    "baseline_correction": {
      "description": "Remove baseline offset",
      "method_msc": "Multiplicative Scatter Correction"
    }
  }
}
```

### Language Keys for New Functions

For each new function, add translation keys in both language files:

```json
{
  "functions": {
    "my_new_function": {
      "display_name": "Display Name",
      "category": "Category Name",
      "description": "Brief description",
      "long_description": "Detailed explanation with multiple lines...",
      "parameters": {
        "param1": "Parameter 1 Label",
        "param2": "Parameter 2 Label"
      },
      "tooltips": {
        "param1": "Tooltip for parameter 1",
        "param2": "Tooltip for parameter 2"
      },
      "values": {
        "option1": "Option 1 Display Name",
        "option2": "Option 2 Display Name"
      }
    }
  }
}
```

### Using Translations in Code

In Python code, use the language manager:

```python
from language_manager import get_language_manager, _

lm = get_language_manager()

# Get translation
title = lm.translate("functions.baseline_correction.display_name")

# Or use shorthand helper
title = _("functions.baseline_correction.display_name")

# With fallback
title = lm.translate("functions.baseline_correction.display_name", 
                     fallback="Baseline Correction")
```

---

## JSON Format References

### `function_specs.json` - Complete Structure

```json
{
  "return_specs": {
    "function_name": ["output_var1", "output_var2"]
  },
  
  "input_specs": {
    "function_name": ["input_var1", "input_var2", "optional_var"]
  },
  
  "import_map": {
    "function_name": ["module.path.to.module", "function_name"]
  },
  
  "gui_listing": {
    "function_name": {
      "config_path": "gui_configs/function_name_config.json"
    }
  }
}
```

### GUI Config JSON - Widget Examples

**User Input - Combobox (Dropdown)**
```json
{
  "name": "method",
  "label": "Method",
  "widget": "combobox",
  "values": ["msc", "svn", "moving_average"],
  "value_aliases": ["Multiplicative Scatter", "Standard Normal Variate", "Moving Average"],
  "default": "msc",
  "required": true,
  "input_type": "user",
  "tooltip": "Select processing method"
}
```

**User Input - Text/Number Entry**
```json
{
  "name": "window_size",
  "label": "Window Size",
  "widget": "entry",
  "type": "int",
  "default": "5",
  "required": false,
  "input_type": "user",
  "tooltip": "Size of the window in points"
}
```

**User Input - File Selector**
```json
{
  "name": "data_path",
  "label": "Data Files",
  "widget": "file_selector",
  "multiple": true,
  "required": true,
  "input_type": "user",
  "ispath": true,
  "tooltip": "Select one or more data files"
}
```

**Routed Input - From Previous Function Output**
```json
{
  "name": "X_cal",
  "label": "Calibration Data",
  "widget": "data_input",
  "required": true,
  "input_type": "routed",
  "tooltip": "Routed from previous function output"
}
```

**Inherited Input - Auto-Passed From Previous Function**
```json
{
  "name": "nway_flag",
  "label": "Data Dimensionality",
  "widget": "entry",
  "type": "int",
  "default": "1",
  "required": true,
  "input_type": "inherited",
  "tooltip": "Automatically inherited from previous function"
}
```

**User Input - Conditional Visibility**
```json
{
  "name": "polyorder",
  "label": "Polynomial Order",
  "widget": "entry",
  "type": "int",
  "default": "3",
  "required": false,
  "input_type": "user",
  "visible_if": {"method": "savgol"},
  "tooltip": "Order for Savitzky-Golay filter"
}
```

---

## Code Pattern Examples

### Pattern 1: Simple Data Processor

```python
def my_processor(X: np.ndarray, param: str) -> np.ndarray:
    """Apply processing to data."""
    if param == "option1":
        return X * 2
    elif param == "option2":
        return X / 2
    else:
        raise ValueError(f"Unknown parameter: {param}")
```

**function_specs.json:**
```json
{
  "return_specs": {"my_processor": ["X"]},
  "input_specs": {"my_processor": ["X", "param"]},
  "import_map": {"my_processor": ["chemometrics.data_processing", "my_processor"]},
  "gui_listing": {"my_processor": {"config_path": "gui_configs/my_processor_config.json"}}
}
```

### Pattern 2: Data Processor with Optional Validation

```python
def my_preprocessor(
    X_cal: np.ndarray,
    method: str,
    X_val: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Preprocess data with optional validation set."""
    
    # Process calibration
    X_cal_proc = _apply_method(X_cal, method)
    
    # Process validation if provided
    if X_val is not None:
        X_val_proc = _apply_method(X_val, method)
        return X_cal_proc, X_val_proc
    
    return X_cal_proc

def _apply_method(X: np.ndarray, method: str) -> np.ndarray:
    """Helper function for processing logic."""
    if method == "scale":
        return X / np.max(np.abs(X), axis=0)
    elif method == "center":
        return X - np.mean(X, axis=0)
    return X
```

### Pattern 3: Calibration Model

```python
def my_calibration(
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    degree: int = 1
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    """Build calibration model with optional validation."""
    
    # Build model
    model = _build_model(X_cal, Y_cal, degree)
    y_cal_pred = model.predict(X_cal)
    
    # Calculate metrics
    metrics = {
        "rmse_cal": np.sqrt(np.mean((Y_cal - y_cal_pred)**2)),
        "r2_cal": _r2_score(Y_cal, y_cal_pred)
    }
    
    # Validation predictions
    y_val_pred = None
    if X_val is not None:
        y_val_pred = model.predict(X_val)
        metrics["rmse_val"] = np.sqrt(np.mean((Y_val - y_val_pred)**2))
        metrics["r2_val"] = _r2_score(Y_val, y_val_pred)
    
    return y_cal_pred, y_val_pred, metrics, model
```

### Pattern 4: Multi-way Data Handler

```python
def my_multiway_function(
    X_cal: np.ndarray,
    method: str,
    X_val: Optional[np.ndarray] = None,
    nway_flag: Optional[int] = None,
    direction: Optional[int] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Process potentially multi-way data."""
    
    # Auto-detect dimensionality if not provided
    if nway_flag is None:
        nway_flag = X_cal.ndim
    
    # Set default direction to last axis
    if direction is None:
        direction = X_cal.ndim - 1
    
    # Process based on dimensionality
    if nway_flag == 1 or X_cal.ndim == 1:
        X_cal_proc = _process_1d(X_cal, method)
    elif nway_flag == 2 or X_cal.ndim == 2:
        X_cal_proc = _process_2d(X_cal, method, direction)
    else:
        X_cal_proc = _process_multiway(X_cal, method, direction)
    
    # Handle validation
    if X_val is not None:
        X_val_proc = _process_multiway(X_val, method, direction)
        return X_cal_proc, X_val_proc
    
    return X_cal_proc
```

---

## Testing and Validation

### Manual Testing Checklist

When adding a new function, verify:

1. **Code Import**
   - Function imports correctly: `from chemometrics.module import function_name`
   - No circular imports or missing dependencies

2. **Function Specs**
   - All four sections present: `return_specs`, `input_specs`, `import_map`, `gui_listing`
   - Parameter names match function signature
   - Return variable names match actual returns

3. **GUI Configs**
   - Created for all languages (en, pt-br)
   - All parameters covered in layout
   - Widget types appropriate for parameter types
   - Default values valid

4. **Language Files**
   - All display strings translated
   - Hierarchical keys properly formatted
   - No missing translations

5. **GUI Integration**
   - Function appears in function list
   - Parameters load correctly
   - Tooltips display properly
   - Routing options available

6. **Function Execution**
   - Function runs with test data
   - Returns correct output shapes/types
   - Handles optional parameters gracefully
   - Error messages are informative

### Example Test Code

```python
# test_my_function.py
import numpy as np
from chemometrics.data_processing import my_processor

def test_my_processor():
    """Test my_processor function."""
    
    # Create test data
    X = np.random.randn(100, 500)
    
    # Test basic functionality
    result = my_processor(X, "option1")
    assert result.shape == X.shape, "Output shape mismatch"
    
    # Test with validation data
    X_val = np.random.randn(50, 500)
    result_cal, result_val = my_processor(X, "option1", X_val)
    assert result_cal.shape == X.shape
    assert result_val.shape == X_val.shape
    
    # Test error handling
    try:
        my_processor(X, "invalid_option")
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✓ All tests passed")

if __name__ == "__main__":
    test_my_processor()
```

---

## Quick Reference Checklist

When creating a new function, create/update:

- [ ] **Core function** in `chemometrics/{module}.py`
  - Complete type annotations
  - Docstring with Args, Returns, Raises
  - Input validation
  - Handle optional validation data if applicable
  
- [ ] **function_specs.json** - Add four entries:
  - [ ] `return_specs`
  - [ ] `input_specs`
  - [ ] `import_map`
  - [ ] `gui_listing`

- [ ] **GUI Configs** (for each language):
  - [ ] `gui_configs/en/function_name_config.json`
  - [ ] `gui_configs/pt-br/function_name_config.json`
  - Must include: display_name, category, input_aliases, output_aliases, short/long descriptions, setup layout
  - **Each setup parameter must have `input_type`** ("user", "routed", or "inherited")
  - Only "user" type inputs appear in Setup tab

- [ ] **Language Files**:
  - [ ] `languages/en.json` - Add function translation keys
  - [ ] `languages/pt-br.json` - Add function translation keys

- [ ] **Testing**:
  - [ ] Test function with sample data
  - [ ] Test with optional parameters
  - [ ] Test error handling
  - [ ] Verify GUI integration

---

## Common Issues and Solutions

### Issue: Function Not Appearing in GUI

**Solution**: Check `function_specs.json` entry and GUI config file path

```json
"gui_listing": {
  "my_function": {
    "config_path": "gui_configs/my_function_config.json"  // Ensure this path exists
  }
}
```

### Issue: Parameters Not Appearing in Setup Tab

**Solution**: Check that `input_type` is set to `"user"` for user-provided inputs

```json
// Correct - will appear in Setup tab
{
  "name": "method",
  "label": "Method",
  "widget": "combobox",
  "values": ["opt1", "opt2"],
  "input_type": "user"
}

// Wrong - will NOT appear in Setup tab
{
  "name": "method",
  "label": "Method",
  "widget": "combobox",
  "values": ["opt1", "opt2"],
  "input_type": "routed"
}
```

### Issue: Routed Data Not Available for Routing

**Solution**: Mark data inputs as `"routed"` type in setup

```json
// Correct - available in Routing tab
{
  "name": "X_cal",
  "label": "Calibration Data",
  "widget": "data_input",
  "required": true,
  "input_type": "routed"
}
```

### Issue: Parameters Not Inheriting from Previous Function

**Solution**: Use `input_type: "inherited"` for parameters that should auto-pass

```json
// Correct - will inherit from previous function
{
  "name": "nway_flag",
  "label": "Dimensionality",
  "widget": "entry",
  "type": "int",
  "input_type": "inherited"
}
```

### Issue: Translations Not Loading

**Solution**: Verify JSON syntax and key hierarchy in language files

```json
// Correct
{"functions": {"my_func": {"param1": "Display Name"}}}

// Use in code
_("functions.my_func.param1")
```

### Issue: Validation Data Not Propagating

**Solution**: Function must return tuple when X_val is provided

```python
# Correct
if X_val is not None:
    return X_cal_result, X_val_result  # Returns tuple
return X_cal_result  # Returns single array
```

---

## Useful Code Snippets

### Get Language Manager

```python
from language_manager import get_language_manager, _

lm = get_language_manager()
text = lm.translate("key.path", fallback="Default Text")
# Or shorthand:
text = _("key.path")
```

### Load Function Specs

```python
import json
from pathlib import Path

specs_path = Path(__file__).parent / "function_specs.json"
with open(specs_path, encoding='utf-8') as f:
    FUNCTION_SPECS = json.load(f)
```

### Load GUI Config

```python
from pathlib import Path

config_path = Path("gui_configs/en/my_function_config.json")
with open(config_path, encoding='utf-8') as f:
    GUI_CONFIG = json.load(f)
```

### Validate NumPy Array Input

```python
def validate_input(X, min_dim=2):
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(X)}")
    if X.ndim < min_dim:
        raise ValueError(f"Expected {min_dim}D+ array, got {X.ndim}D")
    if X.size == 0:
        raise ValueError("Input array cannot be empty")
```

---

## References

- **Main Application**: [main_gui.py](main_gui.py)
- **Language Manager**: [language_manager.py](language_manager.py)
- **Function Specifications**: [function_specs.json](function_specs.json)
- **Existing Examples**: See `chemometrics/*.py` modules
- **Documentation**: See [Documentation/](Documentation/) folder

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 28, 2026 | Initial comprehensive reference guide |

---

**Last Updated**: January 28, 2026  
**For Questions**: Refer to specific sections or check existing function implementations as examples
