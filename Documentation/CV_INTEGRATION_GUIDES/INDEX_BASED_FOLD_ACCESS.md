# Index-Based Access for Fold Outputs

## Your Two Questions - Answered

### 1. Sample-based outputs are still reconstructed ✅
Yes, outputs with a sample dimension (like scores, predictions) are still reconstructed into **full-size arrays**:
```python
results['scores_cv']      # (100, 3) - reconstructed full array
results['scores_single']  # (100, 3) - reference from single fit
```

### 2. Index-based access for fold outputs ✅
Non-sample-based outputs (like loadings) are now wrapped in `FoldSegregatedOutput` which allows **scalable index-based access**:

#### Before (hardcoded):
```python
# Had to hardcode specific fold names
results['loadings_cv_fold_0']  # Access fold 0
results['loadings_cv_fold_1']  # Access fold 1
results['loadings_cv_fold_2']  # Access fold 2
```

#### Now (index-based):
```python
# Use index in a loop - scalable!
for i in range(len(results['loadings_cv'])):
    fold_loadings = results['loadings_cv'][i]  # Access by index
    # No need to hardcode fold names
```

---

## FoldSegregatedOutput - Complete Usage

The `FoldSegregatedOutput` class wraps segregated outputs and provides multiple access patterns:

### 1. Index-based access (recommended for scalability)
```python
loadings_cv = results['loadings_cv']

# Access by integer index
fold_0_loadings = loadings_cv[0]
fold_1_loadings = loadings_cv[1]

# Check number of folds
n_folds = len(loadings_cv)
```

### 2. String key access (direct)
```python
# Also works with string keys
fold_0_loadings = loadings_cv['fold_0']
fold_2_loadings = loadings_cv['fold_2']
```

### 3. Iteration (clean for processing all folds)
```python
# Iterate through all folds in order
for fold_array in loadings_cv:
    print(fold_array.shape)
    # Process each fold
```

### 4. Stack into single array (if needed)
```python
# Convert all folds to stacked array
all_folds_array = loadings_cv.as_array()
# Shape: (5, 3, 20) - 5 folds, each (3, 20)
```

### 5. Get underlying dictionary
```python
# Get the raw fold dictionary
fold_dict = loadings_cv.as_dict()
# Returns: {'fold_0': array, 'fold_1': array, ...}
```

---

## Scalable Analysis Patterns

### Pattern 1: Process variable number of folds
```python
loadings_cv = results['loadings_cv']

# Works regardless of fold count
for fold_idx in range(len(loadings_cv)):
    fold_loadings = loadings_cv[fold_idx]
    
    # Compare to single fit
    diff = fold_loadings - results['loadings_single']
    rmse = np.sqrt(np.mean(diff ** 2))
    print(f"Fold {fold_idx} RMSE: {rmse:.6f}")
```

### Pattern 2: Compute per-fold statistics
```python
# No hardcoding needed!
fold_stats = []
for i in range(len(loadings_cv)):
    fold_array = loadings_cv[i]
    fold_stats.append({
        'fold': i,
        'mean': np.mean(fold_array),
        'std': np.std(fold_array),
        'max': np.max(fold_array)
    })

# Or with list comprehension
fold_means = [np.mean(loadings_cv[i]) for i in range(len(loadings_cv))]
fold_stds = [np.std(loadings_cv[i]) for i in range(len(loadings_cv))]
```

### Pattern 3: Compare across folds
```python
# Stack all folds into array
folds_stacked = loadings_cv.as_array()  # (5, 3, 20)

# Variation across folds
std_across_folds = np.std(folds_stacked, axis=0)  # (3, 20)
mean_across_folds = np.mean(folds_stacked, axis=0)  # (3, 20)

# Which parameters vary most?
max_variation = np.max(std_across_folds)
most_unstable = np.unravel_index(np.argmax(std_across_folds), std_across_folds.shape)
```

### Pattern 4: Fold-by-fold comparison to single fit
```python
single_loadings = results['loadings_single']

differences = []
for i in range(len(loadings_cv)):
    fold_loadings = loadings_cv[i]
    diff = np.linalg.norm(fold_loadings - single_loadings)
    differences.append(diff)

most_stable_fold = np.argmin(differences)
least_stable_fold = np.argmax(differences)
print(f"Most stable: Fold {most_stable_fold}")
print(f"Least stable: Fold {least_stable_fold}")
```

---

## Real-World Example

```python
from chemometrics.cv_pipeline import CVPipeline

# Your function returning scores and loadings
def pca_func(X_train, X_test, fold=-1):
    from sklearn.decomposition import PCA
    rs = 42 if fold == -1 else fold
    pca = PCA(n_components=3, random_state=rs)
    pca.fit(X_train)
    return {
        'scores': pca.transform(X_test),
        'loadings': pca.components_
    }

# Run with single-fit reference mode
pipeline = CVPipeline(...)
results = pipeline.run(
    pca_func,
    X=X,
    reference_output_key='scores',
    capture_output_keys=['scores', 'loadings']
)

# ===== Scalable analysis (works for any n_folds) =====

# 1. Sample-based: Access as full array
scores_cv = results['scores_cv']  # (100, 3)
print(f"Reconstructed scores shape: {scores_cv.shape}")

# 2. Non-sample-based: Access by index
loadings_cv = results['loadings_cv']

# Loop through all folds (no hardcoding!)
for fold_id in range(len(loadings_cv)):
    fold_loadings = loadings_cv[fold_id]  # (3, 20)
    
    # Compare to single fit
    diff = fold_loadings - results['loadings_single']
    stability = 1 - (np.linalg.norm(diff) / np.linalg.norm(results['loadings_single']))
    print(f"Fold {fold_id} parameter stability: {stability:.2%}")

# 3. Stack all folds for analysis
all_folds = loadings_cv.as_array()  # (5, 3, 20)
parameter_variation = np.std(all_folds, axis=0)

print(f"\nParameter variation (std across folds): {parameter_variation.mean():.6f}")
```

---

## Why Index-Based Access Matters

### Before (hardcoded fold names):
```python
# ❌ Breaks if you change n_splits to 10
for fold_name in ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']:
    array = results[f'loadings_cv_{fold_name}']
    # Process...
```

### Now (index-based):
```python
# ✅ Works with any n_splits
for i in range(len(results['loadings_cv'])):
    array = results['loadings_cv'][i]
    # Process...
```

---

## Summary Table

| Access Method | Syntax | Use Case | Scalability |
|---|---|---|---|
| **Index** | `loadings_cv[0]` | Loop through folds | ✅ Excellent |
| **String key** | `loadings_cv['fold_0']` | Direct specific fold | ⚠️ Hardcoded |
| **Iterate** | `for arr in loadings_cv:` | Process all folds | ✅ Excellent |
| **Stack** | `loadings_cv.as_array()` | Vectorized analysis | ✅ Excellent |
| **Get dict** | `loadings_cv.as_dict()` | Access raw fold dict | ✓ Available |

---

## Key Benefits

✅ **Scalable**: Works with any number of folds  
✅ **Flexible**: Multiple access patterns for different needs  
✅ **Clean**: No string formatting or hardcoding  
✅ **Compatible**: Both integer and string keys work  
✅ **Informative**: `len()` tells you how many folds  
✅ **Efficient**: Direct indexing without name lookup
