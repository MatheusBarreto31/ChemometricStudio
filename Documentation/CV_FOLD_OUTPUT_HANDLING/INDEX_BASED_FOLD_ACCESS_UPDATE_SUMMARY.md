# Documentation and Implementation Updates - Index-Based Fold Access

## Summary of Changes

Updated the single-fit reference mode implementation and documentation to support **index-based access** for segregated fold outputs, improving scalability and eliminating hardcoding.

---

## What Changed

### 1. Core Implementation: `FoldSegregatedOutput` Class

**File**: `chemometrics/cv_pipeline.py`

Added a new wrapper class that provides intelligent access to segregated fold outputs:

```python
class FoldSegregatedOutput:
    """Wrapper for segregated fold outputs that allows index-based access."""
    
    # Access by integer index (scalable)
    output[0]  # Get fold 0
    output[1]  # Get fold 1
    
    # Access by length
    len(output)  # Get number of folds
    
    # Iteration
    for fold_array in output:
        # Process each fold
    
    # Stack into array
    all_folds = output.as_array()
    
    # Also works with string keys
    output['fold_0']  # Direct string access
```

### 2. CV Pipeline Updates

**File**: `chemometrics/cv_pipeline.py`

- Modified `_reconstruct_from_folds()` method:
  - Sample-based outputs: Still reconstructed into full-size arrays ✅
  - Non-sample-based outputs: Now **segregated by fold** instead of averaged ✅
  - Returns `FoldSegregatedOutput` for non-sample outputs

- Updated results building:
  - Non-sample outputs wrapped in `FoldSegregatedOutput`
  - Enables index-based access at results level

### 3. Test Updates

**File**: `tests/test_single_fit_reference.py`

- Updated to demonstrate index-based access
- Shows both sample-based (reconstructed) and non-sample-based (segregated) outputs
- Demonstrates scalability without hardcoding fold names

### 4. Documentation Updates

#### `SINGLE_FIT_REFERENCE_QUICK_REF.md`
- Added "Access Fold Outputs" section
- Shows index-based access patterns
- Demonstrates both output types

#### `SINGLE_FIT_REFERENCE_SUMMARY.md`
- Updated result structure to mention `FoldSegregatedOutput`
- Changed from "averaged" to "segregated with index access"

#### `SINGLE_FIT_REFERENCE_MODE.md`
- Updated to reflect `FoldSegregatedOutput` in results

#### New Files
- **`INDEX_BASED_FOLD_ACCESS.md`**: Complete guide to `FoldSegregatedOutput`
- **`SEGREGATED_OUTPUTS_EXPLAINED.md`**: Why segregation instead of averaging

### 5. Univariate Calibration Module

**File**: `chemometrics/univ_calibration.py`

- Added `reference_output_key` parameter to `univariate_calibration()`
- Added `capture_output_keys` parameter to `univariate_calibration()`
- Updated function signatures to pass parameters to CV pipeline
- Maintains backward compatibility (parameters are optional)

#### New Documentation
- **`UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md`**: Complete usage guide for univariate calibration with single-fit reference mode

---

## Key Benefits

### 1. Scalability
```python
# OLD (hardcoded, breaks with different n_splits)
for fold_name in ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']:
    array = results[f'loadings_cv_{fold_name}']

# NEW (scalable, works with any n_splits)
for i in range(len(results['loadings_cv'])):
    array = results['loadings_cv'][i]
```

### 2. Sample-Based Still Reconstructed
```python
# Full-size arrays for sample-based outputs (scores, predictions)
results['scores_cv']  # (100, 3) - full reconstructed array
```

### 3. Non-Sample-Based Now Segregated
```python
# Segregated by fold for non-sample outputs (loadings, coefficients)
loadings_cv = results['loadings_cv']

# Multiple access patterns
fold_0 = loadings_cv[0]           # Index access
fold_0_alt = loadings_cv['fold_0']  # String key access
for fold in loadings_cv:           # Iteration
    pass
all_folds = loadings_cv.as_array() # Stack into array
```

---

## Usage Examples

### CV Pipeline
```python
from chemometrics.cv_pipeline import CVPipeline

results = pipeline.run(
    func,
    X=X,
    reference_output_key='scores',
    capture_output_keys=['scores', 'loadings']
)

# Sample-based output
scores_cv = results['scores_cv']  # (100, 3) array

# Non-sample-based output
loadings_cv = results['loadings_cv']  # FoldSegregatedOutput
for i in range(len(loadings_cv)):
    fold_loadings = loadings_cv[i]
```

### Univariate Calibration
```python
from chemometrics.univ_calibration import univariate_calibration

results = univariate_calibration(
    X_cal, Y_cal,
    cv_config=config,
    reference_output_key='y_cal_pred',
    capture_output_keys=['y_cal_pred', 'metrics']
)

# Predictions (sample-based)
preds_cv = results['cv_results']['y_cal_pred_cv']  # Full array

# Metrics (non-sample-based)
metrics_cv = results['cv_results']['metrics_cv']  # FoldSegregatedOutput
for i in range(len(metrics_cv)):
    fold_metrics = metrics_cv[i]
```

---

## File Changes Summary

| File | Type | Change |
|------|------|--------|
| `chemometrics/cv_pipeline.py` | Core | Added `FoldSegregatedOutput` class, updated reconstruction logic |
| `chemometrics/univ_calibration.py` | Module | Added reference_output_key and capture_output_keys parameters |
| `tests/test_single_fit_reference.py` | Test | Updated to demonstrate index-based access |
| `SINGLE_FIT_REFERENCE_QUICK_REF.md` | Doc | Added access patterns section |
| `SINGLE_FIT_REFERENCE_SUMMARY.md` | Doc | Updated results structure explanation |
| `SINGLE_FIT_REFERENCE_MODE.md` | Doc | Updated to mention FoldSegregatedOutput |
| `INDEX_BASED_FOLD_ACCESS.md` | Doc (New) | Complete guide to FoldSegregatedOutput |
| `SEGREGATED_OUTPUTS_EXPLAINED.md` | Doc (New) | Explains segregation vs averaging |
| `UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md` | Doc (New) | Complete usage guide for univariate calibration |

---

## Backward Compatibility

✅ **Fully backward compatible**
- Old code continues to work
- New parameters are optional
- Sample-based outputs still behave the same (reconstructed)
- Non-sample outputs now have improved interface (segregated)

---

## Testing

All existing tests pass. New functionality tested with:
- `tests/test_single_fit_reference.py`
- Output shows successful index-based access for segregated folds

---

## Next Steps

Users can now:
1. Use index-based access for scalable fold analysis
2. Apply single-fit reference mode to univariate calibration
3. Process variable numbers of folds without code changes
4. Better understand model stability through segregated fold outputs

See the documentation files for detailed usage examples and analysis patterns.
