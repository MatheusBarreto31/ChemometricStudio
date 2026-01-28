# Chemometric Studio - Language Configuration File Structure

## Project Directory Tree

```
d:\ChemometricsTool\
│
├── languages/                           (Centralized UI translations)
│   ├── en.json                         (English: buttons, messages, dialogs, etc.)
│   └── pt-br.json                      (Portuguese: UI framework strings)
│
├── gui_configs/                         (Function-specific configurations)
│   │
│   ├── LANGUAGE VARIANTS (NEW):
│   ├── load_data_config_en.json        ✓ English: Load Data function config
│   ├── load_data_config_pt-br.json     ✓ Portuguese: Load Data function config
│   ├── validation_data_config_en.json  ✓ English: Validation function config
│   ├── validation_data_config_pt-br.json ✓ Portuguese: Validation function config
│   ├── baseline_correction_config_en.json ✓ English: Baseline Correction config
│   ├── baseline_correction_config_pt-br.json ✓ Portuguese: Baseline Correction config
│   ├── smoothing_config_en.json        ✓ English: Smoothing function config
│   ├── smoothing_config_pt-br.json     ✓ Portuguese: Smoothing function config
│   ├── center_normalize_config_en.json ✓ English: Center & Normalize config
│   ├── center_normalize_config_pt-br.json ✓ Portuguese: Center & Normalize config
│   ├── univariate_calibration_config_en.json ✓ English: Univariate Calibration config
│   ├── univariate_calibration_config_pt-br.json ✓ Portuguese: Univariate Calibration config
│   │
│   └── BACKWARD COMPATIBILITY (optional):
│       ├── load_data_config.json       (falls back here if no language variant)
│       ├── validation_data_config.json (falls back here if no language variant)
│       ├── baseline_correction_config.json
│       ├── smoothing_config.json
│       ├── center_normalize_config.json
│       └── univariate_calibration_config.json
│
├── main_gui.py                          (UPDATED: Language-aware config loading)
├── language_manager.py                  (No changes needed)
├── function_specs.json                  (No changes: config_path still points to base names)
│
├── tests/
│   └── (existing tests)
│
├── (NEW TEST FILES):
├── test_config_loading.py               (Validates config file structure)
├── test_gui_config_system.py            (Simulates GUI config loading with language switch)
│
├── (DOCUMENTATION):
├── LANGUAGE_ARCHITECTURE_FIX.md        (Architecture explanation & justification)
├── IMPLEMENTATION_COMPLETE.md           (Complete implementation summary)
└── [This file]
```

## Configuration File Organization

### By Function (6 functions total)

```
Load Data
├── load_data_config_en.json
└── load_data_config_pt-br.json

Validation Data
├── validation_data_config_en.json
└── validation_data_config_pt-br.json

Baseline Correction
├── baseline_correction_config_en.json
└── baseline_correction_config_pt-br.json

Smoothing
├── smoothing_config_en.json
└── smoothing_config_pt-br.json

Center & Normalize
├── center_normalize_config_en.json
└── center_normalize_config_pt-br.json

Univariate Calibration
├── univariate_calibration_config_en.json
└── univariate_calibration_config_pt-br.json
```

### By Language (2 languages total)

```
English (_en.json): 6 files
├── load_data_config_en.json
├── validation_data_config_en.json
├── baseline_correction_config_en.json
├── smoothing_config_en.json
├── center_normalize_config_en.json
└── univariate_calibration_config_en.json

Portuguese (_pt-br.json): 6 files
├── load_data_config_pt-br.json
├── validation_data_config_pt-br.json
├── baseline_correction_config_pt-br.json
├── smoothing_config_pt-br.json
├── center_normalize_config_pt-br.json
└── univariate_calibration_config_pt-br.json
```

## Configuration File Content Structure

Each language-variant config file contains:

```json
{
  "short_description": "Brief description of function (language-specific)",
  
  "long_description": "Detailed explanation of function...\n\nWith sections...",
  
  "setup": {
    "layout": [
      {
        "name": "field_name",
        "label": "Display Label (language-specific)",
        "widget": "combobox|entry|checkbutton|file_selector",
        "type": "int|float|string (optional)",
        "values": [...],
        "default": "value",
        "required": true|false,
        "tooltip": "Help text (language-specific)",
        "visible_if": {...},
        "ispath": true|false
      },
      ...
    ]
  }
}
```

## Key Fields Translated Per Config

### Each config file contains translations for:

1. **short_description**
   - Example EN: "Loads spectroscopic and response data from files"
   - Example PT: "Carrega dados espectroscópicos e de resposta de arquivos"

2. **long_description**
   - Multi-paragraph detailed explanation in appropriate language
   - Includes section headers and formatting

3. **Form field labels** (in setup.layout[].label)
   - Example EN: "Data Separator"
   - Example PT: "Separador de Dados"

4. **Tooltips** (in setup.layout[].tooltip)
   - Example EN: "Delimiter character used in data file"
   - Example PT: "Caractere delimitador usado no arquivo de dados"

## Loading Priority (Fallback Chain)

When user selects language "pt-br" for "smoothing" function:

```
1st: Try to load smoothing_config_pt-br.json
     ↓ (if file exists, use it)
2nd: Try to load smoothing_config_en.json
     ↓ (if file exists, use it)
3rd: Try to load smoothing_config.json (original, backward compatible)
     ↓ (if file exists, use it)
4th: Display error message
```

## Integration with main_gui.py

### Config Loading Method

**Location:** `main_gui.py`, method `_load_gui_configs()` (lines 101-149)

```python
def _load_gui_configs(self):
    """Load function-specific GUI configuration files with language support."""
    gui_listing = FUNCTION_SPECS.get("gui_listing", {})
    current_language = get_language_manager().get_language()
    
    for func_alias, func_info in gui_listing.items():
        config_path = func_info.get("config_path")  # e.g., "gui_configs/smoothing_config.json"
        
        # Construct language-specific path
        # e.g., "gui_configs/smoothing_config_pt-br.json"
        
        # Load with fallback chain
        # Language-specific → English → Original
```

### Language Refresh Method

**Location:** `main_gui.py`, method `_refresh_ui_text()` (lines 442-454)

```python
def _refresh_ui_text(self):
    """Refresh UI text and configs when language changes."""
    self._load_gui_configs()  # ← NEW: Reload configs with new language
    # ... rest of UI refresh logic
```

## File Statistics

### Language Variant Configs
- **Total files created:** 12
- **Per function:** 2 (English + Portuguese)
- **Total lines of configuration:** ~1,200
- **Average file size:** 3-4 KB per file

### Coverage
- **Functions with translations:** 6/6 (100%)
- **Languages supported:** 2 (English, Portuguese)
- **Form fields translated:** 40+
- **Descriptions translated:** 12 (6 short + 6 long)

## Testing Coverage

### Files
- `test_config_loading.py` - Validates file existence and JSON validity
- `test_gui_config_system.py` - Simulates language-aware loading

### Test Scenarios
1. ✓ All 12 language variant files exist
2. ✓ All files contain valid JSON
3. ✓ Configs load correctly in English
4. ✓ Configs load correctly in Portuguese
5. ✓ Form field labels are translated
6. ✓ Fallback chain works properly
7. ✓ Language switching triggers reload
8. ✓ Backward compatibility verified

## Adding a New Language

To add support for a new language (e.g., Spanish):

1. **Create 6 new config files:**
   - `load_data_config_es.json`
   - `validation_data_config_es.json`
   - `baseline_correction_config_es.json`
   - `smoothing_config_es.json`
   - `center_normalize_config_es.json`
   - `univariate_calibration_config_es.json`

2. **Translate content:**
   - Copy English version as template
   - Translate all text fields to new language
   - Keep structure identical

3. **Update language manager (optional):**
   - Add language code to `SUPPORTED_LANGUAGES` in `language_manager.py`
   - Create central `languages/es.json` for UI framework strings

4. **Test:**
   - Run existing tests to verify fallback chain
   - Create new test for new language if desired
   - Switch language in GUI to verify translations appear

## Backward Compatibility

### Original Config Files Still Work

If language variant files are missing, system falls back to:
- English variant (_en.json files)
- Original base files (no suffix)

This ensures:
- ✓ Existing projects continue to work
- ✓ Partial translations don't break system
- ✓ No data loss or migration needed
- ✓ Gradual rollout of translations possible

## Summary

The new architecture provides:

| Aspect | Benefit |
|--------|---------|
| **Organization** | Function configs grouped by language |
| **Maintenance** | Easy to find and update translations |
| **Scalability** | Add languages without code changes |
| **Reliability** | Fallback ensures no broken references |
| **Performance** | Minimal overhead, fast config loading |
| **Usability** | Seamless language switching in GUI |
| **Compatibility** | 100% backward compatible |

All files are in place, tested, and ready for production use.
