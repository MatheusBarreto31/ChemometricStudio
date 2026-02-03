# Single-Fit Reference Mode for CV

## Overview

The CV pipeline now supports a **single-fit reference mode** that allows you to:

1. **Run a single fit** on all data to get reference outputs
2. **Run CV folds** comparing fold outputs against the single-fit reference
3. **Compute metrics** as differences between fold and reference
4. **Reconstruct full-size arrays** from fold test sets

This enables you to assess how stable a function's outputs are when trained on different data subsets, comparing against a reference model trained on all data.

---

## Quick Example

```python
from chemometrics.cv_pipeline import CVConfig, CVPipeline

# Setup
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
    reference_output_key='scores',      # Compare fold scores to single-fit scores
    capture_output_keys=['scores', 'loadings']  # Also capture loadings
)

# Results contain:
# - scores_rmse: Overall metric comparing reconstructed vs single fit
# - scores_rmse_per_fold: [fold_0_rmse, fold_1_rmse, ...]
# - scores_cv: Reconstructed full-size scores from folds (100, 3)
# - scores_single: Reference scores from single fit (100, 3)
# - loadings_cv: FoldSegregatedOutput with index access - loadings[0], loadings[1], etc.
# - loadings_single: Single-fit loadings (3, 20)
```

---

## How It Works

### Mode 1: Single Fit (fold=-1)
```
Input: X (100, 20) [all data]
↓
Single Fit: PCA on all 100 samples
↓
Output: scores (100, 3) ← This becomes the reference
```

### Mode 2: CV Folds (fold=0-4)
```
Fold 0: Train on [0-19, 40-99], Test on [20-39]
  PCA model A scores test[20:39] → rmse_0 vs reference[20:39]
  
Fold 1: Train on [20-39, 60-99], Test on [40-59]
  PCA model B scores test[40:59] → rmse_1 vs reference[40:59]
  
... etc
```

### Reconstruction
```
Fold outputs gathered in original positions:
- scores_fold_0[test_idx_0] → position [20:39]
- scores_fold_1[test_idx_1] → position [40:59]
- ... all 5 folds

Result: Full-size scores_cv (100, 3) reconstructed
Metric: RMSE(scores_cv, scores_single) compared across all samples
```

---

## Function Requirements

Your function must:
1. Accept `fold=-1` for single fit
2. Return dict with the `reference_output_key`
3. Work identically in single-fit and CV-fold modes

```python
def pca_function(X_train, X_test, fold=-1):
    """
    Args:
        X_train: Training data (or full data if fold=-1)
        X_test: Test data (or full data if fold=-1)
        fold: -1 for single fit, 0-4 for CV folds
    
    Returns:
        Dict with 'scores' and optionally other outputs
    """
    pca = PCA(n_components=3)
    pca.fit(X_train)
    scores = pca.transform(X_test)
    
    return {
        'scores': scores,
        'loadings': pca.components_,
        'rmse': compute_reconstruction_error(X_test, pca)
    }
```

---

## Parameters

### reference_output_key: str
Which output key to use as the reference for metric computation.
- **Required** to enable single-fit reference mode
- Must be returned by the function
- Used to compute RMSE: `rmse_per_fold = sqrt(mean((fold_output - reference_output) ** 2))`

### capture_output_keys: list[str]
Which outputs to capture and reconstruct from folds.
- **Optional** (defaults to `config.capture_outputs`)
- Can be same or different from `reference_output_key`
- Supports multiple outputs
- Used for detailed analysis and comparison

---

## Results Structure

### With reference_output_key='scores'

```python
results = {
    # Metrics comparing fold outputs to reference
    'scores_rmse': 1.323,              # Overall RMSE
    'scores_rmse_per_fold': [0.691, 1.526, 1.143, 1.356, 1.900],
    'scores_rmse_std': 0.402,          # Std of per-fold metrics
    
    # Reconstructed outputs
    'scores_cv': array(100, 3),        # Full-size from folds
    'scores_single': array(100, 3),    # Reference from single fit
    
    # Additional captured outputs (if capture_output_keys includes them)
    'loadings_cv': array(3, 20),       # Averaged from folds (if not sample-based)
    'loadings_single': array(3, 20),   # Single fit reference
    
    'n_folds': 5                       # Number of folds
}
```

---

## Output Types

### Sample-Based Outputs (like scores)
- **Shape**: (n_samples, n_features)
- **Reconstruction**: Placed back in original sample positions
- **Result**: Full-size array (100, 3)

### Non-Sample-Based Outputs (like loadings, weights)
- **Shape**: (n_components, n_features) — no sample dimension
- **Reconstruction**: Averaged across folds (since they're model parameters)
- **Result**: Single averaged array

---

## Use Cases

### 1. Model Stability Assessment
```python
# How stable is PCA when trained on different subsets?
results = pipeline.run(
    pca_function,
    X=X,
    reference_output_key='scores',
    capture_output_keys=['scores']
)

# Compare fold scores to reference
fold_diff = results['scores_cv'] - results['scores_single']
instability = np.std(fold_diff, axis=0)  # Per-feature instability
print(f"Features with high instability: {np.argsort(instability)[-3:]}")
```

### 2. Model Robustness Testing
```python
# Which samples are most affected by training set variations?
per_sample_rmse = np.sqrt(np.mean(
    (results['scores_cv'] - results['scores_single']) ** 2,
    axis=1
))

unstable_samples = np.where(per_sample_rmse > threshold)[0]
print(f"Unstable samples: {unstable_samples}")
```

### 3. Component Consistency
```python
# How consistent are PCA components across folds?
loadings_single = results['loadings_single']  # Reference
loadings_cv = results['loadings_cv']         # Averaged

# Cosine similarity
for i in range(n_components):
    similarity = np.dot(
        loadings_single[i], loadings_cv[i]
    ) / (np.linalg.norm(loadings_single[i]) * np.linalg.norm(loadings_cv[i]))
    print(f"Component {i} similarity: {similarity:.4f}")
```

---

## Metric Interpretation

| RMSE Value | Interpretation | Action |
|------------|-----------------|--------|
| < 0.1 | Highly stable | ✅ Confident in results |
| 0.1 - 0.5 | Moderately stable | ⚠️ Monitor for instability |
| > 0.5 | Unstable | ❌ Investigate data quality |

### Per-Fold Variation
```python
# Check variability across folds
cv_std = results['scores_rmse_std']
print(f"RMSE std across folds: {cv_std:.4f}")

# Low std = consistent across folds
# High std = model highly sensitive to training set
```

---

## Advanced: Multiple Outputs with Different Reconstruction

```python
results = pipeline.run(
    pca_function,
    X=X,
    reference_output_key='scores',
    capture_output_keys=['scores', 'loadings', 'explained_variance', 'singular_values']
)

# Sample-based output (reconstructed in original positions)
scores_cv = results['scores_cv']              # (100, 3)
scores_single = results['scores_single']      # (100, 3)

# Non-sample outputs (averaged across folds)
loadings_cv = results['loadings_cv']          # (3, 20) averaged
loadings_single = results['loadings_single']  # (3, 20) from single fit

# Scalar outputs
exp_var_cv = results['explained_variance_cv']      # Scalar (averaged)
exp_var_single = results['explained_variance_single']  # Scalar
```

---

## See Also

- [CV_INTEGRATION_GUIDE.md](CV_INTEGRATION_GUIDE.md) - Basic CV integration
- [OUTPUT_STABILITY_ASSESSMENT.md](OUTPUT_STABILITY_ASSESSMENT.md) - Per-fold output capture
- [tests/test_single_fit_reference.py](../tests/test_single_fit_reference.py) - Working example
