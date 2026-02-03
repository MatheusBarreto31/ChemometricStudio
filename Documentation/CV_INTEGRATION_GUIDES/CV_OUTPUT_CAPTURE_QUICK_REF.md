# CV Output Capture - Quick Reference

## Enable Output Capture

```python
from chemometrics.cv_pipeline import CVConfig

cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    output_metrics=['rmse', 'r2'],
    capture_outputs=['scores', 'loadings']  # What to capture
)
```

## Function Returns Dict with Captured Outputs

```python
def my_function(X_train, X_test, fold=0, cv_config=None):
    model = fit_model(X_train)
    scores = model.transform(X_test)
    metric = compute_metric(X_test, model)
    
    return {
        'rmse': metric,              # Metric (aggregated across folds)
        'scores': scores,            # Output 1 (captured per fold)
        'loadings': model.components_  # Output 2 (captured per fold)
    }
```

## Results Structure

```python
results = pipeline.run(my_function, X=X)

# Aggregated metrics
results['rmse_folds']    # [0.88, 0.92, 0.85, 0.87, 0.90]
results['rmse_mean']     # 0.884
results['rmse_std']      # 0.028

# Captured outputs (per fold)
results['scores_fold_0']    # Shape: (20, 3)
results['scores_fold_1']    # Shape: (20, 3)
results['loadings_fold_0']  # Shape: (3, 20)
results['loadings_fold_1']  # Shape: (3, 20)
```

## Extract and Analyze

```python
import numpy as np

# Get all fold scores
scores_list = [results[f'scores_fold_{i}'] for i in range(5)]

# Stack into single array
scores_stacked = np.stack(scores_list, axis=0)  # (5, 20, 3)

# Compute stability (std across folds)
stability = np.std(scores_stacked, axis=0)  # (20, 3)

# Identify unstable samples
unstable = np.max(stability, axis=1) > threshold
print(f"Unstable samples: {np.sum(unstable)}")
```

## Interpretation

| Stability Level | What It Means | Action |
|-----------------|---------------|--------|
| **Low std** (< 0.1) | Outputs consistent across folds | ✅ Robust model |
| **Medium std** (0.1-0.5) | Some variation | ⚠️ Monitor for overfitting |
| **High std** (> 0.5) | Large variations | ❌ Unstable, investigate data/params |

## Common Patterns

### PCA Stability Assessment
```python
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    capture_outputs=['scores']
)

# Function returns PCA scores per fold
# Assess: Do scores cluster similarly across folds?
```

### Calibration Robustness
```python
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    capture_outputs=['coefficients']
)

# Function returns fitted coefficients per fold
# Assess: How much do coefficients change with different training data?
```

### Embedding Stability
```python
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    capture_outputs=['embeddings', 'weights']
)

# Function returns learned embeddings and weights per fold
# Assess: Are embeddings consistent across training subsets?
```

## Key Points

✅ **Metrics are aggregated** (mean, std, min, max)  
✅ **Captured outputs are per-fold** (scores_fold_0, scores_fold_1, ...)  
✅ **Works with multiway data** (3D, 4D, etc.)  
✅ **Backward compatible** (capture_outputs defaults to [])  
✅ **No function changes needed** (just add return dict keys)  

## See Also

- [OUTPUT_STABILITY_ASSESSMENT.md](OUTPUT_STABILITY_ASSESSMENT.md) - Full guide with examples
- [CV_INTEGRATION_GUIDE.md](CV_INTEGRATION_GUIDE.md) - How to add CV to your function
- [tests/test_cv_output_stability.py](../tests/test_cv_output_stability.py) - Working example
