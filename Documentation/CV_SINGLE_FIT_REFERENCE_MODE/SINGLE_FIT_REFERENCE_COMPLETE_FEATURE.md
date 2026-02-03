# Single-Fit Reference Mode - Complete Feature Summary

## The Feature You Asked For ✅

> "Run a function's single fit (on all data) to get outputs. Being able to use one of those outputs as the y_test reference for CV evaluation... but also give me an option of outputting output captures of a list of multiple function outputs."

**Status: FULLY IMPLEMENTED**

---

## Core Implementation

### 1. Single Fit (fold=-1)
```python
# Your function runs once on all data
def pca_func(X_train, X_test, fold=-1):
    if fold == -1:  # Single fit marker
        # X_train and X_test are BOTH the full data
        # Run model once on everything
    return {'scores': scores, 'loadings': loadings}
```

### 2. CV Folds (fold=0-4)
```python
# Your function runs 5 times, each on a different training subset
def pca_func(X_train, X_test, fold=0):
    # fold=0: train on 80%, test on first 20%
    # fold=1: train on 80%, test on next 20%
    # ... etc
    return {'scores': scores, 'loadings': loadings}
```

### 3. Single-Fit Reference Mode
```python
from chemometrics.cv_pipeline import CVPipeline

results = pipeline.run(
    pca_func,
    X=X,
    reference_output_key='scores',        # Use this as reference
    capture_output_keys=['scores', 'loadings']  # Capture these
)

# ✅ What you get:
# - Single fit reference outputs
# - CV fold outputs
# - Comparison metrics (RMSE, etc.)
# - Full-size reconstructed arrays for sample-based outputs
# - Segregated fold outputs for non-sample outputs
```

---

## Results Structure

### Sample-Based Outputs (like predictions, scores)

```python
# Full-size arrays reconstructed from fold test outputs
results['scores_cv']      # shape (100, 3) - from all folds combined
results['scores_single']  # shape (100, 3) - reference from single fit

# Direct comparison
diff = results['scores_cv'] - results['scores_single']
stability = np.std(diff)
```

**Why reconstructed?** When you take the test predictions from each fold and line them up where they belong in the full dataset, you get the complete set of predictions.

### Non-Sample-Based Outputs (like parameters, loadings, coefficients)

```python
# Segregated by fold with index-based access
loadings_cv = results['loadings_cv']  # FoldSegregatedOutput

# Access by index (scalable!)
fold_0_loadings = loadings_cv[0]  # (3, 20)
fold_1_loadings = loadings_cv[1]  # (3, 20)

# Loop any number of folds
for i in range(len(loadings_cv)):
    fold_loadings = loadings_cv[i]
    # Process...

# Or iterate
for fold_loadings in loadings_cv:
    # Process...

# Get reference
loadings_single = results['loadings_single']  # (3, 20)
```

**Why segregated?** Model parameters (loadings, coefficients) don't correspond to specific samples - they belong to the entire model. You need to see each fold's version to understand parameter variation.

---

## Metrics and Comparison

```python
# Overall metric comparing folds to single fit
results['scores_rmse']                # 1.323 - how different overall?

# Per-fold breakdown
results['scores_rmse_per_fold']      # [0.69, 1.53, 1.14, 1.36, 1.90]

# Variability across folds
results['scores_rmse_std']            # 0.402 - how consistent?
```

---

## Complete Example: PCA Stability Assessment

```python
from sklearn.decomposition import PCA
from chemometrics.cv_pipeline import CVConfig, CVPipeline

# Define your function
def pca_func(X_train, X_test, fold=-1):
    rs = 42 if fold == -1 else fold
    pca = PCA(n_components=3, random_state=rs)
    pca.fit(X_train)
    return {
        'scores': pca.transform(X_test),
        'loadings': pca.components_
    }

# Configure CV
config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    random_state=42
)

# Run with single-fit reference
pipeline = CVPipeline(config)
results = pipeline.run(
    pca_func,
    X=X,
    reference_output_key='scores',
    capture_output_keys=['scores', 'loadings']
)

# Analysis
print(f"Overall score stability (RMSE): {results['scores_rmse']:.4f}")
print(f"Per-fold: {results['scores_rmse_per_fold']}")

# Scores: full arrays, direct comparison
score_diff = results['scores_cv'] - results['scores_single']
print(f"Score variation (std): {np.std(score_diff):.4f}")

# Loadings: segregated by fold, indexed access
loadings_cv = results['loadings_cv']
for i in range(len(loadings_cv)):
    fold_loadings = loadings_cv[i]
    diff = fold_loadings - results['loadings_single']
    param_stability = 1 - (np.linalg.norm(diff) / np.linalg.norm(results['loadings_single']))
    print(f"Fold {i} parameter stability: {param_stability:.2%}")
```

---

## Applied to Univariate Calibration

```python
from chemometrics.univ_calibration import univariate_calibration
from chemometrics.cv_pipeline import CVConfig

# Configure
config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5
)

# Run with single-fit reference
results = univariate_calibration(
    X_cal, Y_cal,
    X_val, Y_val,
    degree=2,
    cv_config=config,
    reference_output_key='y_cal_pred',
    capture_output_keys=['y_cal_pred', 'metrics']
)

# Single fit results (on all data)
print(f"Single fit RMSE: {results['metrics']['X0_Y0']['metrics_cal']['RMSE']:.4f}")

# CV comparison results
cv_results = results['cv_results']
print(f"CV stability RMSE: {cv_results['y_cal_pred_rmse']:.4f}")
print(f"Per-fold: {cv_results['y_cal_pred_rmse_per_fold']}")

# Prediction comparison
preds_diff = cv_results['y_cal_pred_cv'] - cv_results['y_cal_pred_single']
print(f"Prediction variation: {np.std(preds_diff):.4f}")

# Metrics per fold
metrics_cv = cv_results['metrics_cv']
for i in range(len(metrics_cv)):
    fold_metrics = metrics_cv[i]
    # Analyze fold-specific results...
```

---

## Key Design Decisions

### 1. Output Type Detection
Automatically determines if output is sample-based or not:
```python
if output.shape[0] == len(test_set):
    # Sample-based: reconstruct
    reconstructed[test_idx] = output
else:
    # Non-sample-based: segregate by fold
    segregated['fold_X'] = output
```

### 2. Index-Based Access (No Hardcoding)
```python
# ❌ Old way (doesn't scale)
results['loadings_cv_fold_0']
results['loadings_cv_fold_1']
# What if you change n_splits to 10?

# ✅ New way (scalable)
for i in range(len(results['loadings_cv'])):
    results['loadings_cv'][i]
```

### 3. Backward Compatibility
- Parameters optional
- Sample-based behavior unchanged
- Non-sample outputs improved (segregated instead of averaged)
- All existing code continues working

---

## File Organization

| Component | Location | Purpose |
|-----------|----------|---------|
| **Core Implementation** | `chemometrics/cv_pipeline.py` | CVPipeline with single-fit reference mode |
| **Univariate Module** | `chemometrics/univ_calibration.py` | Univariate calibration with CV support |
| **Test** | `tests/test_single_fit_reference.py` | Working example with PCA |
| **Quick Start** | `SINGLE_FIT_REFERENCE_QUICK_REF.md` | Copy-paste examples |
| **Complete Guide** | `SINGLE_FIT_REFERENCE_MODE.md` | Full reference documentation |
| **Index Access Guide** | `INDEX_BASED_FOLD_ACCESS.md` | FoldSegregatedOutput usage |
| **Univariate Guide** | `UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md` | Univariate calibration examples |

---

## Answers to Your Original Questions

### Q: Sample-dependent outputs continue being reconstructed, right?
**A:** ✅ Yes! Outputs where first dimension = number of samples are reconstructed into full-size arrays.

### Q: Can fold outputs be called from an index for scalability?
**A:** ✅ Yes! Non-sample outputs use `FoldSegregatedOutput` which supports:
- Index access: `output[0]`, `output[1]`, etc.
- Length: `len(output)`
- Iteration: `for fold in output:`
- Stacking: `output.as_array()`
- String keys: `output['fold_0']` (still works)

---

## Status: Complete and Ready ✅

- ✅ Single-fit reference mode implemented
- ✅ Index-based fold access for scalability
- ✅ Applied to univariate calibration module
- ✅ Comprehensive documentation
- ✅ Working test examples
- ✅ Backward compatible

Ready to use!
