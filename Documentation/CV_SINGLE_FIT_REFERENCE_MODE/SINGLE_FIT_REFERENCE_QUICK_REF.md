# Single-Fit Reference Mode - Quick Reference

## Enable Single-Fit Reference

```python
from chemometrics.cv_pipeline import CVConfig, CVPipeline

cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5
)

pipeline = CVPipeline(cv_config)

# ✅ NEW: Run with reference
results = pipeline.run(
    func,
    X=X,
    reference_output_key='scores',           # Use this output as reference
    capture_output_keys=['scores', 'loadings']  # Capture these outputs
)
```

## Function Must Support Both Modes

```python
def your_func(X_train, X_test, fold=-1):
    # fold=-1: Single fit on all data
    # fold=0-4: CV folds with train/test split
    
    model = fit(X_train)
    output = model.transform(X_test)
    
    return {
        'scores': output,        # Reference output
        'loadings': model.W,     # Additional output
        'rmse': compute_error(X_test, model)
    }
```

## Results Structure

```python
results = {
    # Metrics: How different are folds from reference?
    'scores_rmse': 1.323,                    # Overall
    'scores_rmse_per_fold': [0.69, 1.53, ...],  # Per fold
    'scores_rmse_std': 0.402,                # Variability
    
    # Reconstructed full-size outputs
    'scores_cv': array(100, 3),        # From folds
    'scores_single': array(100, 3),    # From single fit (reference)
    
    # Additional outputs - segregated by fold
    'loadings_cv': FoldSegregatedOutput([...]),  # Index access: [0], [1], etc.
    'loadings_single': array(3, 20),   # From single fit
    
    # Metadata
    'n_folds': 5
}
```

## Access Fold Outputs

### Sample-Based Outputs (Reconstructed)
```python
# Direct array access - full-size arrays
scores_cv = results['scores_cv']  # shape (100, 3)
scores_single = results['scores_single']  # shape (100, 3)
```

### Non-Sample-Based Outputs (Segregated)
```python
# Index-based access - scalable, no hardcoding
loadings_cv = results['loadings_cv']

# Access by index (recommended for loops)
fold_0 = loadings_cv[0]  # shape (3, 20)
fold_1 = loadings_cv[1]  # shape (3, 20)

# Loop through all folds
for i in range(len(loadings_cv)):
    fold_array = loadings_cv[i]
    # Process...

# Stack all folds into array
all_folds = loadings_cv.as_array()  # shape (5, 3, 20)

# Also works with string keys
fold_0_alt = loadings_cv['fold_0']  # Same as loadings_cv[0]
```

## Analysis Examples

### Stability Assessment
```python
# How much do scores vary from reference?
diff = results['scores_cv'] - results['scores_single']
stability = np.std(diff, axis=0)

# Which samples are most unstable?
per_sample_rmse = np.sqrt(np.mean(diff ** 2, axis=1))
unstable_idx = np.argsort(per_sample_rmse)[-5:]  # Top 5
```

### Per-Fold Comparison
```python
# Per-fold stability metrics
per_fold = results['scores_rmse_per_fold']
print(f"Best fold: {np.min(per_fold):.4f}")
print(f"Worst fold: {np.max(per_fold):.4f}")
print(f"Mean: {np.mean(per_fold):.4f}")
```

### Component Analysis
```python
# Compare model components/parameters
loadings_cv = results['loadings_cv']
loadings_single = results['loadings_single']

# Iterate through folds (scalable!)
for i in range(len(loadings_cv)):
    fold_loadings = loadings_cv[i]
    diff = fold_loadings - loadings_single
    rmse = np.sqrt(np.mean(diff ** 2))
    print(f"Fold {i} RMSE: {rmse:.6f}")

# Cosine similarity per component
for i in range(3):
    v1 = loadings_single[i]
    v2 = loadings_cv[i]
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"Component {i} similarity: {sim:.4f}")
```

## Key Parameters

| Parameter | Type | Required | Default | Purpose |
|-----------|------|----------|---------|---------|
| `reference_output_key` | str | No | None | Which output to compare against |
| `capture_output_keys` | list[str] | No | config.capture_outputs | Which outputs to capture |

## Output Reconstruction

| Output Type | Example | Reconstruction |
|-------------|---------|-----------------|
| **Sample-based** | scores (100, 3) | Placed in original positions |
| **Non-sample** | loadings (3, 20) | Averaged across folds |
| **Scalar** | explained_variance | Averaged across folds |

## Stability Interpretation

| RMSE | Stability | Action |
|------|-----------|--------|
| < 0.1 | Excellent | ✅ Use confidently |
| 0.1 - 0.5 | Good | ⚠️ Monitor |
| 0.5 - 1.0 | Fair | ⚠️ Investigate |
| > 1.0 | Poor | ❌ Fix issues |

## Common Patterns

### Pattern 1: Quick Stability Check
```python
results = pipeline.run(func, X=X, reference_output_key='output')
if results['output_rmse'] < 0.1:
    print("✓ Model is stable")
else:
    print("✗ Model is unstable - investigate data")
```

### Pattern 2: Feature Stability
```python
diff = results['scores_cv'] - results['scores_single']
feature_stability = np.std(diff, axis=0)
unstable_features = np.argsort(feature_stability)[-3:]
print(f"Unstable features: {unstable_features}")
```

### Pattern 3: Sample Robustness
```python
rmse_per_sample = np.sqrt(np.mean(
    (results['scores_cv'] - results['scores_single']) ** 2,
    axis=1
))
fragile_samples = np.where(rmse_per_sample > threshold)[0]
print(f"Check samples: {fragile_samples}")
```

## File Reference

- **Source**: `chemometrics/cv_pipeline.py`
- **Test**: `tests/test_single_fit_reference.py`
- **Docs**: `SINGLE_FIT_REFERENCE_MODE.md`

## See Also

- [SINGLE_FIT_REFERENCE_MODE.md](SINGLE_FIT_REFERENCE_MODE.md) - Full reference
- [SINGLE_FIT_REFERENCE_COMPLETE.md](SINGLE_FIT_REFERENCE_COMPLETE.md) - Implementation details
- [CV_INTEGRATION_GUIDE.md](CV_INTEGRATION_GUIDE.md) - Basic CV setup
