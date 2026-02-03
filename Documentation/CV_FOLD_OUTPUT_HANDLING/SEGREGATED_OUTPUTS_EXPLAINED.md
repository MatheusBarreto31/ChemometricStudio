# Segregated Non-Sample Outputs - Explained

## The Change

When you capture outputs that **don't have the same dimension as samples**, they are now **segregated by fold** instead of being averaged.

### Example

With 5 CV folds and `capture_output_keys=['scores', 'loadings']`:

**Before** (averaged):
- `loadings_cv`: (3, 20) - single averaged array
- `loadings_single`: (3, 20)

**Now** (segregated):
- `loadings_cv_fold_0`: (3, 20)
- `loadings_cv_fold_1`: (3, 20)
- `loadings_cv_fold_2`: (3, 20)
- `loadings_cv_fold_3`: (3, 20)
- `loadings_cv_fold_4`: (3, 20)
- `loadings_single`: (3, 20)

---

## How It Detects Output Type

The system automatically detects whether an output is sample-based or not:

```python
first_output.shape[0] == len(test_set)
```

### Sample-Based (reconstructed full-size)
✅ Outputs where first dimension = number of samples

Examples:
- **Scores**: (20 test samples, 3 components) → (100 full samples, 3 components)
- **Predictions**: (20 test samples,) → (100 full samples,)
- **Embeddings**: (20 test samples, 64 dims) → (100 full samples, 64 dims)

Result: **Full-size array** with test outputs positioned in original locations

```python
results['scores_cv']  # shape (100, 3)
```

### Non-Sample-Based (segregated by fold)
✅ Outputs where first dimension ≠ number of samples

Examples:
- **Loadings**: (3 components, 20 variables) - same for all test sizes
- **Weights**: (20 input features, 5 hidden units) - model parameters
- **Components**: (3, 20) - PCA components
- **Coefficients**: (shape independent of test set)

Result: **Separate arrays per fold** labeled `fold_0`, `fold_1`, etc.

```python
results['loadings_cv_fold_0']  # shape (3, 20)
results['loadings_cv_fold_1']  # shape (3, 20)
results['loadings_cv_fold_2']  # shape (3, 20)
results['loadings_cv_fold_3']  # shape (3, 20)
results['loadings_cv_fold_4']  # shape (3, 20)
```

---

## Why Segregate Instead of Average?

### The Problem with Averaging
When you average model parameters across folds:
- You lose information about **how much the parameters vary**
- You get one "blended" version that never actually existed
- You can't see which folds produced different parameters

### The Benefit of Segregating
When you keep them separate:
- ✅ See parameter variation across folds
- ✅ Compare specific fold parameters to single-fit version
- ✅ Identify which folds are most/least different
- ✅ Compute your own aggregate if desired (mean, std, etc.)

---

## Code Example

```python
from chemometrics.cv_pipeline import CVPipeline

# Your function returns both scores and loadings
def pca_func(X_train, X_test, fold=-1):
    pca = PCA(n_components=3, random_state=42 if fold == -1 else fold)
    pca.fit(X_train)
    scores = pca.transform(X_test)
    loadings = pca.components_  # shape (3, 20) - NOT per-sample
    return {'scores': scores, 'loadings': loadings}

# Run with both outputs captured
pipeline = CVPipeline(...)
results = pipeline.run(
    pca_func,
    X=X,
    reference_output_key='scores',
    capture_output_keys=['scores', 'loadings']
)

# Sample-based output: full reconstructed array
print(results['scores_cv'].shape)        # (100, 3)
print(results['scores_single'].shape)    # (100, 3)

# Non-sample-based outputs: segregated by fold
for fold in range(5):
    key = f'loadings_cv_fold_{fold}'
    print(f"{key}: {results[key].shape}")  # Each is (3, 20)

print(results['loadings_single'].shape)  # (3, 20)
```

---

## Analysis Patterns

### 1. Compare Fold Variation
```python
import numpy as np

# Get all fold loadings
fold_loadings = [results[f'loadings_cv_fold_{i}'] for i in range(5)]
fold_loadings_array = np.stack(fold_loadings)  # (5, 3, 20)

# Compare to single fit
single_loadings = results['loadings_single']

# Compute variation
loadings_std = np.std(fold_loadings_array, axis=0)  # (3, 20)
loadings_diff = fold_loadings_array - single_loadings  # (5, 3, 20)
loadings_diff_mean = np.mean(np.abs(loadings_diff), axis=0)  # (3, 20)

print(f"Max parameter variation: {loadings_std.max():.6f}")
print(f"Mean absolute difference from single fit: {loadings_diff_mean.mean():.6f}")
```

### 2. Identify Unstable Parameters
```python
# Which parameters change most?
unstable_idx = np.argsort(loadings_std.flatten())[-10:]  # Top 10
print(f"Most unstable parameters: {unstable_idx}")

# Which fold is most different?
fold_distances = [np.linalg.norm(fold_loadings[i] - single_loadings) 
                  for i in range(5)]
most_different_fold = np.argmax(fold_distances)
print(f"Fold {most_different_fold} is most different")
```

### 3. Custom Aggregation
```python
# Only aggregate if you want
averaged_loadings = np.mean(fold_loadings_array, axis=0)
median_loadings = np.median(fold_loadings_array, axis=0)

# Or use weighted average based on fold performance
fold_rmses = results['scores_rmse_per_fold']
weights = 1 - np.array(fold_rmses) / max(fold_rmses)  # Better folds get higher weight
weighted_loadings = np.average(fold_loadings_array, axis=0, weights=weights)
```

---

## Test Output Example

```
SEGREGATED OUTPUTS (NON-SAMPLE-BASED):
  loadings_cv_fold_0 shape: (3, 20)
  loadings_cv_fold_1 shape: (3, 20)
  loadings_cv_fold_2 shape: (3, 20)
  loadings_cv_fold_3 shape: (3, 20)
  loadings_cv_fold_4 shape: (3, 20)
  loadings_single shape: (3, 20)
```

Each fold's loadings is independent, showing how the PCA components differ when trained on different subsets of the data.

---

## Summary Table

| Aspect | Sample-Based (e.g., scores) | Non-Sample-Based (e.g., loadings) |
|--------|------|--------|
| **First dimension** | Equals test set size | Different from test set size |
| **Output structure** | Single full-size array | Multiple fold-specific arrays |
| **Key naming** | `output_cv` | `output_cv_fold_0`, `output_cv_fold_1`, etc. |
| **Size** | (n_total_samples, ...) | (fold_output_dim1, fold_output_dim2, ...) |
| **Use case** | Compare predictions | Assess parameter stability |
| **Example** | `scores_cv` (100, 3) | `loadings_cv_fold_0` (3, 20) |

---

## Why This Matters

Different output types have fundamentally different information:

- **Sample-based outputs** tell you how the model predicts for each sample
  - When you reconstruct, you get the actual predictions for all samples
  - Missing positions are the test set positions across folds

- **Non-sample-based outputs** tell you what the model learned
  - When you segregate, you see how model parameters vary
  - This reveals model stability and robustness

Averaging non-sample outputs loses the crucial information about **how much the model itself changes** across different training subsets.
