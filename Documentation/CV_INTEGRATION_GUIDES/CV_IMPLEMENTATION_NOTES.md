# Cross-Validation Pipeline Implementation Summary

## Overview
A reusable, DRY cross-validation infrastructure has been added to CM Studio. The CV system is decoupled from modeling functions via:
- **Single CV configuration function** (`cv_configuration`)
- **Centralized CV pipeline module** (`chemometrics/cv_pipeline.py`)
- **Minimal integration** in modeling functions (e.g., `univariate_calibration`)

## Multiway Data Support

**The CV pipeline is fully multiway-aware and handles 1D, 2D, 3D, 4D, and higher-dimensional data automatically.**

### How It Works
The pipeline splits data along **axis 0 (samples only)**, preserving all other dimensions:

```
Univariate (2D):
  (n_samples, n_variables) -> split -> (n_train, n_variables) + (n_test, n_variables)

Multiway (3D):
  (n_samples, n_wavelengths, n_time) -> split -> (n_train, n_wavelengths, n_time) + (n_test, n_wavelengths, n_time)

Multiway + Auxiliary (4D):
  (n_samples, n_wavelengths, n_time, n_electrodes) -> split -> (n_train, n_wavelengths, n_time, n_electrodes) + (n_test, n_wavelengths, n_time, n_electrodes)
```

### Example: 3D Spectroscopic Data with Time Dimension
```python
# Your multiway data: 50 samples, 103 wavelengths, 35 time points
X = np.random.randn(50, 103, 35)  # Shape: (samples, wavelengths, time)
Y = np.random.randn(50, 2)

# CV pipeline handles this automatically
cv_config = cv_configuration(use_cv=True, cv_strategy='kfold', n_splits=5)['cv_config']
result = pls_calibration(X, Y, cv_config=cv_config)

# Each fold receives properly split 3D data
# X_train_fold1: (10, 103, 35), X_test_fold1: (40, 103, 35)
# X_train_fold2: (40, 103, 35), X_test_fold2: (10, 103, 35), etc.
```

### No Changes Needed for New Dimensions
Whether your modeling function receives 2D or 4D data, the CV pipeline works identically:
- No need to tell CVPipeline about data dimensionality
- No need to reshape/flatten multiway data
- Automatically preserves spectroscopic/temporal/spatial structure

## Files Created/Modified

### 1. **NEW: `chemometrics/cv_pipeline.py`** (275 lines)
Core module providing reusable CV infrastructure:

**Key Classes:**
- `CVConfig`: Dataclass holding CV parameters (strategy, n_splits, random_state, etc.)
- `CVSplitter` (abstract): Base class for splitting strategies
- `KFoldSplitter`, `StratifiedKFoldSplitter`, `TimeSeriesSplitter`, `RepeatedKFoldSplitter`, `ShuffleSplitSplitter`: Concrete implementations
- `CVPipeline`: Main orchestrator that applies a function across CV folds and aggregates results

**Key Function:**
- `cv_configuration()`: Factory function that creates a `CVConfig` object as a routing output

### 2. **NEW: `gui_configs/en/cv_configuration_config.json`** (142 lines)
GUI configuration for the CV Configuration node:
```
Categories: Activation, Strategy, Parameters, Output
Parameters:
  - use_cv (bool): Enable/disable CV
  - cv_strategy (dropdown): kfold, stratified_kfold, timeseries, repeated_kfold, shuffle_split
  - n_splits (int): Number of folds (default: 5)
  - random_state (int): Random seed (default: 42)
  - shuffle (bool): Randomize before splitting (default: True)
  - output_metrics (checkboxes): rmse, r2, mae, mape, accuracy, f1
```

### 3. **MODIFIED: `function_specs.json`**
Added CV Configuration function to specs:
```json
"return_specs": {
  "cv_configuration": ["cv_config"],
  ...
}
"input_specs": {
  "cv_configuration": ["use_cv", "cv_strategy", "n_splits", "random_state", "shuffle", "output_metrics"],
  ...
  "univariate_calibration": [..., "cv_config"]
}
"output_specs": {
  "univariate_calibration": [..., "cv_results"]
}
"import_map": {
  "cv_configuration": ["chemometrics.cv_pipeline", "cv_configuration"],
  ...
}
```

### 4. **MODIFIED: `model.json`**
Added CV Configuration instance to the functions list:
```json
{
  "instance_alias": "cv_configuration",
  "base_alias": "cv_configuration",
  "display_name": "CV Configuration",
  "parameters": {
    "use_cv": true,
    "cv_strategy": "kfold",
    "n_splits": "5",
    "random_state": "42",
    "shuffle": true,
    "output_metrics": ["rmse", "r2"]
  }
}
```

### 5. **MODIFIED: `chemometrics/univ_calibration.py`**
Updated to support optional CV via `cv_config` parameter:
- Main `univariate_calibration()` now accepts `cv_config` parameter
- Returns dict (not tuple) with optional `cv_results` key
- Internal `_univariate_calibration_single_fit()` handles actual logic
- Backward compatible: CV disabled by default; single fit when `cv_config=None`

## Usage Architecture

```
┌─────────────────────────────┐
│ CV Configuration Node       │  (NEW node in workflow)
│ Parameters:                 │
│ • use_cv ← True/False       │
│ • cv_strategy               │
│ • n_splits, random_state    │
│ • output_metrics            │
└────────────┬────────────────┘
             │ Routes: cv_config
             ▼
┌──────────────────────────────────────┐
│ Univariate Calibration               │
│ (or any CV-compatible function)      │
│ Inputs: X_cal, Y_cal, cv_config      │
│ Outputs: y_cal_pred, y_val_pred,     │
│          models, metrics, cv_results │
└──────────────────────────────────────┘
```

## User Workflow

1. **Add CV Configuration node** to your pipeline
2. **Configure once**:
   - Set `use_cv=True` to enable CV
   - Choose strategy (K-Fold, Stratified, Time Series, etc.)
   - Set number of folds
   - Select metrics to report
3. **Route output** (`cv_config`) to any modeling function
4. **Results automatically include**:
   - Single-fit predictions/models (same as no-CV case)
   - `cv_results` dict with aggregated metrics: `{metric}_mean`, `{metric}_std`, `{metric}_folds`, `{metric}_min`, `{metric}_max`

### Example CV Results Output
```python
{
  'y_cal_pred': {...},
  'y_val_pred': {...},
  'models': {...},
  'metrics': {...},
  'cv_results': {
    'rmse_folds': [0.85, 0.91, 0.88, ...],  # values from each fold
    'rmse_mean': 0.883,
    'rmse_std': 0.03,
    'rmse_min': 0.81,
    'rmse_max': 0.95,
    'r2_folds': [0.92, 0.90, 0.91, ...],
    'r2_mean': 0.912,
    'r2_std': 0.01,
    ...
    'n_folds': 5
  }
}
```

## DRY Benefits

| Scenario | Before | After |
|----------|--------|-------|
| **Add new CV strategy** | Modify 5+ function configs | Update one `cv_configuration_config.json` |
| **Change metrics** | Edit each function's JSON | Edit one CV config |
| **Add new modeling function** | Duplicate CV params in its JSON | Just accept `cv_config` parameter |
| **Scale to 10+ functions** | 10 copies of CV params | Single CV module, thin wrappers |

## Supported CV Strategies

| Strategy | Best For | Notes |
|----------|----------|-------|
| **K-Fold** | Standard ML workflows | Random splits, IID data |
| **Stratified K-Fold** | Classification | Preserves class proportions |
| **Time Series** | Temporal data | Forward-chaining, no lookahead bias |
| **Repeated K-Fold** | Reduce variance | Multiple iterations, more compute |
| **Shuffle Split** | Large datasets | Random repeated splits |

## Backward Compatibility

- Existing code that calls `univariate_calibration(X, Y, ...)` **without `cv_config`** works unchanged
- Returns dict instead of tuple (tuple is unpacked in dict, should update calling code)
- Behavior when `cv_config=None` or `use_cv=False`: single fit, same as before

## Next Steps

To apply this to other modeling functions (e.g., PLS, PCR):

1. Add `cv_config` parameter to function signature
2. Wrap logic in `_function_single_fit()`
3. Check CV config and route through `CVPipeline` if enabled
4. Return dict with optional `cv_results`

Example 10-line wrapper:
```python
def pls_calibration(X_cal, Y_cal, n_components=3, cv_config=None, fold=0):
    if cv_config and cv_config.is_enabled():
        pipeline = CVPipeline(cv_config)
        cv_results = pipeline.run(_pls_single_fit, X_cal=X_cal, Y_cal=Y_cal, n_components=n_components)
        single = _pls_single_fit(X_cal, Y_cal, n_components, fold=0)
        return {**single, 'cv_results': cv_results}
    return _pls_single_fit(X_cal, Y_cal, n_components, fold=fold)
```

## Testing

All components tested and verified:
- ✅ CVConfig creation and serialization
- ✅ CVPipeline with KFold splitting
- ✅ cv_configuration() function
- ✅ univariate_calibration with/without CV
- ✅ Result dict structure and aggregation
