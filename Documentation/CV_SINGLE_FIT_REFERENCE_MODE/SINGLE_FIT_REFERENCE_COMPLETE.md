# Single-Fit Reference Mode - Implementation Complete

## What You Asked For

**Your Request**: 
> "Run a function's single fit (on all data) to get outputs. Being able to use one of those outputs as the y_test reference for CV evaluation. Instead of requiring a separate pre-defined y_test, but still having the option to use a pre-defined y_test in case that function requires it."

**Status**: ✅ **Implemented and tested**

---

## What Was Built

A new **single-fit reference mode** for the CV pipeline that:

1. **Runs single fit** on all data (`fold=-1`)
2. **Gets reference output** (e.g., PCA scores from all 100 samples)
3. **Runs CV folds** normally (5 folds with different training data)
4. **Compares fold outputs** against the single-fit reference
5. **Computes metrics** as RMSE between fold and reference
6. **Reconstructs full-size arrays** from fold test sets
7. **Captures multiple outputs** for analysis

---

## Implementation

### Modified Files
- ✅ `chemometrics/cv_pipeline.py`
  - Added `_reconstruct_from_folds()` method
  - Enhanced `CVPipeline.run()` with two modes
  - Added `reference_output_key` and `capture_output_keys` parameters

### New Parameters to `CVPipeline.run()`

```python
def run(
    self, 
    func: Callable, 
    reference_output_key: Optional[str] = None,
    capture_output_keys: Optional[List[str]] = None, 
    **kwargs
):
```

- **`reference_output_key`**: Which output to use as reference for metrics
- **`capture_output_keys`**: Which outputs to capture and reconstruct

---

## How It Works

### Traditional CV (existing):
```python
pipeline.run(func, X=X)
# Returns: rmse_mean, rmse_std, rmse_per_fold, ...
```

### Single-Fit Reference Mode (new):
```python
pipeline.run(
    func,
    X=X,
    reference_output_key='scores',
    capture_output_keys=['scores', 'loadings']
)
# Returns: scores_rmse, scores_rmse_per_fold, scores_cv, scores_single, ...
```

### Data Flow

```
INPUT: X (100, 20)

SINGLE FIT (fold=-1)
├─ Train PCA on all 100 samples
├─ Get reference scores (100, 3)
└─ Store as reference

CV FOLDS (fold=0-4)
├─ Fold 0: Train on [0-19, 40-99], test [20-39]
│  ├─ PCA model A scores → (20, 3)
│  ├─ Compare: fold_scores vs reference[20:39]
│  ├─ Metric: RMSE_0 = 0.691
│  └─ Store in position [20:39]
│
├─ Fold 1: Train on [20-39, 60-99], test [40:59]
│  └─ ... similar ...
│
└─ ... Fold 2-4 ...

RECONSTRUCTION
├─ Gather all test set outputs in original positions
├─ Reconstruct full-size scores_cv (100, 3)
└─ Compute overall RMSE(scores_cv, scores_single)

RESULTS
├─ scores_rmse: 1.323 (mean of per-fold RMSEs)
├─ scores_rmse_per_fold: [0.691, 1.526, 1.143, 1.356, 1.900]
├─ scores_cv: array (100, 3) - from CV folds
├─ scores_single: array (100, 3) - from single fit
├─ loadings_cv: array (3, 20) - averaged from folds
└─ loadings_single: array (3, 20) - from single fit
```

---

## Key Design Decisions

### 1. Function Behavior
- Function works identically with `fold=-1` (single fit) and `fold=0-4` (CV folds)
- No special logic needed in function
- Function decides what metrics/outputs to return

### 2. Reconstruction Strategy
- **Sample-based outputs** (scores): Placed in original sample positions
  - Shape: (n_samples, n_features) → reconstructed as (100, 3)
- **Non-sample outputs** (loadings): Averaged across folds
  - Shape: (n_components, n_features) → averaged as (3, 20)

### 3. Metrics Computation
- Per-fold: `RMSE = sqrt(mean((fold_output - reference_output[test_idx]) ** 2))`
- Overall: Mean of per-fold RMSEs (not a global RMSE on reconstructed arrays)

### 4. Optional vs Required
- `reference_output_key`: **Optional** — if None, runs traditional CV
- `capture_output_keys`: **Optional** — defaults to `config.capture_outputs`
- **Backward compatible**: Existing code unaffected

---

## Example Usage

```python
import numpy as np
from sklearn.decomposition import PCA
from chemometrics.cv_pipeline import CVConfig, CVPipeline

# Define function
def pca_function(X_train, X_test, fold=-1):
    pca = PCA(n_components=3)
    pca.fit(X_train)
    scores = pca.transform(X_test)
    return {
        'scores': scores,
        'loadings': pca.components_,
        'rmse': np.sqrt(np.mean((X_test - pca.inverse_transform(scores)) ** 2))
    }

# Setup
np.random.seed(42)
X = np.random.randn(100, 20)

cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5
)

pipeline = CVPipeline(cv_config)

# Run with single-fit reference
results = pipeline.run(
    pca_function,
    X=X,
    reference_output_key='scores',
    capture_output_keys=['scores', 'loadings']
)

# Results
print(f"RMSE: {results['scores_rmse']:.4f}")
print(f"Per-fold: {results['scores_rmse_per_fold']}")
print(f"Reconstructed scores shape: {results['scores_cv'].shape}")

# Analyze stability
per_sample_rmse = np.sqrt(np.mean(
    (results['scores_cv'] - results['scores_single']) ** 2,
    axis=1
))
unstable = np.where(per_sample_rmse > 1.0)[0]
print(f"Unstable samples: {unstable}")
```

---

## Test Results

**Test File**: `tests/test_single_fit_reference.py`

```
Data: 100 samples × 20 features
PCA: 3 components
CV: 5-fold

Results:
  Overall RMSE: 1.323065
  Per-fold RMSE: [0.690786, 1.525725, 1.142841, 1.355914, 1.900058]
  RMSE std: 0.401828
  
Reconstructed shapes:
  scores_cv: (100, 3) ✓
  scores_single: (100, 3) ✓
  loadings_cv: (3, 20) ✓ (averaged)
  loadings_single: (3, 20) ✓
```

---

## Benefits

✅ **Flexible reference**: Use function outputs as reference instead of requiring Y  
✅ **Comprehensive analysis**: Compare single-fit vs CV-fold models  
✅ **Multiple outputs**: Capture different metrics for detailed investigation  
✅ **Automatic reconstruction**: Full-size arrays from fold tests  
✅ **Backward compatible**: Existing CV code unaffected  
✅ **Works with multiway data**: Automatic handling of 3D, 4D arrays  

---

## Documentation

- **[SINGLE_FIT_REFERENCE_MODE.md](SINGLE_FIT_REFERENCE_MODE.md)** - Complete guide
- **[tests/test_single_fit_reference.py](../tests/test_single_fit_reference.py)** - Working example

---

## Next Steps

1. **Use it**: Add `reference_output_key` and `capture_output_keys` to your CV calls
2. **Analyze**: Compare single-fit vs CV-fold outputs for stability assessment
3. **Iterate**: Adjust model parameters based on instability findings

---

## Status

| Item | Status |
|------|--------|
| Implementation | ✅ Complete |
| Testing | ✅ Passing |
| Documentation | ✅ Comprehensive |
| Backward compatibility | ✅ Verified |
| Ready to use | ✅ Yes |

**Implementation is complete and ready!** 🎉
