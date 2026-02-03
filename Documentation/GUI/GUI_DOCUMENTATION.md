# Chemometric Studio GUI - Implementation Summary

## Overview

The Chemometric Studio now features a complete tkinter-based GUI application with support for building analysis pipelines through a visual interface. The GUI allows users to select functions, configure parameters, define data routing between functions, and execute analysis workflows.

## Architecture

### Core Components

#### 1. **main_gui.py** (582 lines)
The main GUI application class that provides:
- **ChemometricsGUI**: Main application class inheriting from tkinter
- **Panel Layout**:
  - Left Panel: Available Functions (collapsible by category) + Methodology (selected functions)
  - Right Panel: Workspace (Setup, Routing, Analysis, Report tabs)

#### 2. **function_specs.json**
Centralized metadata registry containing:
- `return_specs`: Output names for each function
- `input_specs`: Input parameter names for each function
- `import_map`: Dynamic import information [module_path, function_name]
- `gui_listing`: Comprehensive metadata for each function:
  - `display_name`: UI-friendly name
  - `category`: Function category (Data Import, Preprocessing, etc.)
  - `data_type`: Data handling type (all, firstorder, multiway)
  - `input_aliases`: Friendly names for input parameters
  - `output_aliases`: Friendly names for output parameters
  - `config_path`: Path to function-specific GUI config file

#### 3. **gui_configs/** (6 JSON files)
Function-specific configuration files defining widget layout and properties:
- `load_data_config.json`
- `validation_data_config.json`
- `baseline_correction_config.json`
- `smoothing_config.json`
- `center_normalize_config.json`
- `univariate_calibration_config.json`

Each config file contains:
```json
{
  "setup": {
    "layout": [
      {
        "name": "parameter_name",
        "label": "Display Label",
        "widget": "entry|combobox|checkbutton|file_selector",
        "values": [...],
        "default": "value",
        "required": true|false,
        "type": "int|float|string",
        "multiple": true|false,
        "visible_if": { "param": value }
      }
    ]
  }
}
```

## Features

### 1. Functions Panel
- **Collapsible Categories**: Functions grouped by category (Data Import, Preprocessing, Calibration)
- **Dynamic Loading**: Functions loaded from `function_specs.json` gui_listing
- **One-Click Addition**: Click function button to add to methodology
- **Duplicate Handling**: Automatically suffixes duplicate functions (e.g., "Load Data #2")

### 2. Methodology Panel
- **Function List**: Shows selected functions in order
- **Selection**: Click to select function for configuration
- **Remove**: Remove individual functions
- **Clear All**: Clear entire methodology

### 3. Setup Tab
- **Dynamic Widget Generation**: Widgets created from function config JSON
- **Widget Types**:
  - `entry`: Text input (int/float/string)
  - `combobox`: Dropdown selection
  - `checkbutton`: Boolean toggle
  - `file_selector`: Single or multiple file selection
- **Auto-Save**: Values saved on widget focus loss or selection
- **Default Values**: Pre-populated from config files
- **Required Fields**: Marked with asterisk (*)

### 4. Routing Tab
- **Connection UI**: Select source output → select destination input → connect
- **Connection List**: Display all active connections
- **Connection Removal**: Delete connections with Delete key or selection
- **Function Aliasing**: Supports multiple instances of same function with index notation

### 5. Analysis Tab
- Placeholder for analysis results and statistics (under development)

### 6. Report Tab
- Placeholder for report generation (under development)

### 7. Run Model Button
- **Config Generation**: Creates functions.txt and routing.txt from current state
- **Execution**: Imports and runs analyst_main() from analyst module
- **Output Capture**: Captures stdout/stderr to StringIO
- **Log File**: Writes execution output to model_log.txt
- **Error Handling**: Logs errors to model_log.txt with traceback

## Data Flow

### 1. User Configuration
```
User selects functions → Configures parameters in Setup tab → Defines routing in Routing tab
```

### 2. File Generation
```
GUI State (methodology_list, function_configs, routing_lines)
    ↓
_generate_config_files()
    ↓
functions.txt (function calls with parameters)
routing.txt (connection definitions)
```

### 3. Model Execution
```
_run_model()
    ↓
_generate_config_files()
    ↓
import analyst.analyst_main()
    ↓
Run analysis pipeline
    ↓
Capture output to StringIO
    ↓
Write to model_log.txt
```

## Data Structures

### ChemometricsGUI Attributes
```python
self.methodology_list: List[str]
    # [func_alias, func_alias, ...]
    # e.g., ["load_data", "baseline_correction", "baseline_correction"]

self.function_configs: Dict[str, Dict[str, Any]]
    # {func_alias: {param_name: value, ...}}
    # e.g., {"load_data": {"d_specs": "comma", "data_path": "/path/to/file"}}

self.routing_lines: Dict[Tuple, str]
    # {(src_output, dst_input): value}
    # e.g., {("Load Data → X Data", "Baseline Correction → X_cal"): "X Data"}

self.selected_function_idx: Optional[int]
    # Index in methodology_list for currently selected function

self.gui_configs: Dict[str, Dict]
    # {func_alias: config_data}
    # Loaded from gui_configs/*.json files
```

## Key Methods

### UI Building
- `_build_ui()`: Main layout structure
- `_build_functions_panel()`: Functions section with canvas scrolling
- `_add_collapsible_category()`: Expandable category containers
- `_build_methodology_panel()`: Selected functions list
- `_build_control_bar()`: Tab buttons and Run Model button

### Tab Management
- `_show_setup_tab()`: Dynamic widget generation from config
- `_show_routing_tab()`: Connection interface
- `_show_analysis_tab()`: Analysis placeholder
- `_show_report_tab()`: Report placeholder
- `_clear_tab()`: Clean workspace before showing new tab

### Configuration
- `_load_gui_configs()`: Load all function config JSONs on startup
- `_save_widget_value()`: Save widget value to function_configs
- `_generate_config_files()`: Create functions.txt and routing.txt
- `_populate_source_functions()`: Fill routing source combobox

### Execution
- `_run_model()`: Execute analysis pipeline
- `_add_to_methodology()`: Add function to methodology with duplicate handling
- `_remove_from_methodology()`: Remove function from list
- `_clear_methodology()`: Clear all functions and configs

## Configuration Format Examples

### Function Entry in function_specs.json
```json
"baseline_correction": {
  "display_name": "Baseline Correction",
  "category": "Preprocessing",
  "data_type": "all",
  "input_aliases": {
    "X_cal": "Calibration Data",
    "X_val": "Validation Data",
    "method": "Correction Method"
  },
  "output_aliases": {
    "X_cal": "Corrected X_cal",
    "X_val": "Corrected X_val"
  },
  "config_path": "gui_configs/baseline_correction_config.json"
}
```

### Widget Configuration Example
```json
{
  "name": "method",
  "label": "Baseline Correction Method",
  "widget": "combobox",
  "values": ["msc", "svn", "moving_average"],
  "default": "msc",
  "required": true
}
```

## File Generation Output

### functions.txt Format
```
load_data d_specs:comma data_path:/path/to/file.txt nway_flag:1
baseline_correction X_cal:result_X_cal method:msc window_size:5
smoothing X_cal:result_X_cal method:savitzky_golay window_size:5 polyorder:2
```

### routing.txt Format
```
Load Data → X Data → Baseline Correction → X_cal
Baseline Correction → X_cal → Center & Normalize → X_cal
Center & Normalize → X_cal → Univariate Calibration → X_cal
```

## Error Handling

### Configuration Errors
- Missing required fields in widget specs
- Invalid JSON in config files
- Missing config file references

### Execution Errors
- Import failures for functions
- Missing data files
- Runtime errors during analysis

All errors are logged to `model_log.txt` with full traceback.

## Testing

### test_gui_init.py
Validates:
- Function specs JSON structure
- All required metadata in gui_listing
- GUI config files exist and are valid JSON
- All functions can be imported successfully

## Future Enhancements

1. **Canvas Line Drawing**: Visualize routing connections with canvas lines
2. **Conditional Widgets**: Hide/show widgets based on visible_if conditions
3. **Configuration Persistence**: Save/load project configurations to file
4. **Analysis Tab**: Display execution results and statistics
5. **Report Tab**: Generate formatted PDF/HTML reports
6. **Theme Integration**: Full Sun-Valley-ttk-theme styling
7. **Advanced Routing**: Visual drag-and-drop connection builder
8. **Function Validation**: Pre-execution validation of parameters
9. **Undo/Redo**: Revert configuration changes
10. **Batch Processing**: Run multiple parameter combinations

## Usage

### Starting the GUI
```bash
python launcher.py
```

Or directly:
```bash
python main_gui.py
```

### Building an Analysis Pipeline
1. Click function buttons in left panel to add to Methodology
2. Select function in Methodology to configure
3. Fill in parameters in Setup tab
4. Switch to Routing tab to define data flow between functions
5. Click "► Run Model" to execute pipeline
6. Check model_log.txt for execution results

## Dependencies

- `tkinter`: GUI framework (included with Python)
- `json`: Configuration file parsing
- `pathlib`: File path handling
- `io.StringIO`: Output capture
- Function modules: chemometrics.data_input, chemometrics.data_processing, etc.

## Notes

- GUI is responsive and doesn't block during execution (output is captured)
- All configurations are in-memory; use Save/Load features to persist (future)
- Functions are loaded dynamically based on function_specs.json
- Widget types are extensible; new types can be added to main_gui.py
- Routing uses human-readable function/output names for clarity
