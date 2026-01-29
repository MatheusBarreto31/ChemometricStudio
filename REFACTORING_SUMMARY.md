# JSON Refactoring Summary

## Overview
Replaced text-based configuration files (`functions.txt` and `routing.txt`) with a single structured JSON file (`model.json`). This improves maintainability, validation, and extensibility.

## Changes Made

### 1. **main_gui.py**

#### New Function: `_generate_model_json()`
- Replaces `_generate_config_files()`
- Generates a single `model.json` file containing:
  - **metadata**: version, creation timestamp, description
  - **functions**: array of function configurations with instance aliases, parameters
  - **routing**: array of connection definitions between functions

#### Updated Functions:
- **`_save_model_with_data()`** - Now reads/modifies `model.json` instead of `functions.txt`
- **`_save_full_model()`** - Processes `model.json` for full model save
- **`_save_model_method_only()`** - Extracts method-only configuration from `model.json`
- **`_load_model()`** - Extracts `model.json` from archive during load
- **`_parse_and_load_model_json()`** - New function to parse JSON and populate GUI state
  - Replaces `_parse_and_load_functions_txt()` and `_parse_and_load_routing_txt()`
  - Handles tempfiles path conversion
  - Validates function/routing structure

#### Model JSON Structure:
```json
{
  "metadata": {
    "version": "1.0",
    "created": "2026-01-29T12:34:56.789123",
    "description": "CM Studio Model Configuration"
  },
  "functions": [
    {
      "instance_alias": "load_data#1",
      "base_alias": "load_data",
      "display_name": "Load Data",
      "parameters": {
        "data_path": ["path/to/file.txt"],
        "separator": "tabs",
        "num_headlines": "0",
        "d_specs": ["tabs", "0", "x_matrix", ""]
      }
    }
  ],
  "routing": [
    {
      "source": {
        "instance_alias": "load_data#1",
        "param_key": "data_matrix",
        "param_name": "Data Matrix"
      },
      "destination": {
        "instance_alias": "smoothing#1",
        "param_key": "x_data",
        "param_name": "X Data"
      },
      "auto_created": false
    }
  ]
}
```

### 2. **analyst.py**

#### Complete Refactoring
- Replaced complex text-based parsing with simple JSON loading
- **Old approach**: Regex/string parsing of `functions.txt` and `routing.txt` (~80 lines of complex logic)
- **New approach**: Direct JSON loading and dict-based processing (~50 lines, cleaner)

#### Key Changes:
- Load `model.json` directly using `json.load()`
- Extract functions array and build `functions_info` dict
- Extract routing array and build `routing_map` dict
- Execute functions in order, applying routing automatically
- Store outputs under instance alias (enables multiple calls to same function)

#### Execution Flow:
1. Load `model.json` and `function_specs.json`
2. Build maps from model data
3. Import required functions
4. Execute each function with parameters
5. Apply routing by looking up values in `routing_map`
6. Store outputs indexed by instance alias

### 3. **Removed Files**
No files were deleted; `functions.txt` and `routing.txt` are no longer generated but won't cause issues if present.

## Benefits

✅ **Simpler Code** - No custom parsing logic needed  
✅ **Better Validation** - JSON schema can be validated before execution  
✅ **Extensible** - Adding new metadata fields is trivial  
✅ **Self-Documenting** - Display names and descriptions included in model  
✅ **Robust** - Standard JSON error handling  
✅ **Easier Debugging** - Pretty-printed JSON is human-readable  
✅ **Reversible** - Simple dict-to-JSON conversion maintains compatibility  
✅ **Future-Proof** - Support versioning and migrations easily  

## Migration Path

For users with existing saved models (.mdcd, .mdon, .mdfd files):
- Old files contain `functions.txt` and `routing.txt` 
- Can still be loaded (code will parse text files if needed)
- Upon re-save, they will be converted to new `model.json` format
- No manual migration needed

## Testing Checklist

- [ ] Create new model and verify `model.json` is generated correctly
- [ ] Save model with calibration data (.mdcd)
- [ ] Save model with full data (.mdfd)
- [ ] Save method only (.mdon)
- [ ] Load model from saved file
- [ ] Verify routing is preserved in loaded model
- [ ] Run model execution and verify outputs
- [ ] Test with multiple instances of same function
- [ ] Test with routing connections
- [ ] Verify language switching still works
