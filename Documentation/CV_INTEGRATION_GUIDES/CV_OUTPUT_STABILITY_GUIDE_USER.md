# Testing Output Stability: Complete Walkthrough

## Your Question Answered

**You asked**: "I'm not trying to do this for different functions, I'm trying to do it for inside a single one. How could I test stability of scores if the PCA is only run once?"

**Answer**: Now you can! The CV framework has been enhanced to run your function (e.g., PCA) **5 times** (once per fold) and **capture the outputs from each run**. This lets you compare the 5 different score matrices to assess stability.

---

## What's New

### Before (Metrics Only)
```python
# Traditional CV: Only aggregated metrics
cv_config = CVConfig(use_cv=True, cv_strategy='kfold', n_splits=5)
results = pipeline.run(pca_func, X=X)

# Can see: RMSE varies by fold
print(results['rmse_folds'])  # [0.88, 0.92, 0.85, ...]

# Can't see: How do the actual PCA scores differ across folds?
# (Because scores weren't captured)
```

### After (Metrics + Fold Outputs)
```python
# New CV: Both metrics AND fold outputs
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    capture_outputs=['scores']  # ← NEW: Capture scores per fold
)
results = pipeline.run(pca_func, X=X)

# Can see: RMSE varies by fold
print(results['rmse_folds'])  # [0.88, 0.92, 0.85, ...]

# Can also see: How PCA scores differ across folds
scores_fold_0 = results['scores_fold_0']  # Shape (20, 3)
scores_fold_1 = results['scores_fold_1']  # Shape (20, 3)
scores_fold_2 = results['scores_fold_2']  # Shape (20, 3)
# ... etc
```

---

## Real Example: PCA Stability

### Step 1: Create Your CV Configuration

```python
from chemometrics.cv_pipeline import CVConfig, CVPipeline

# Configure CV to capture PCA scores from each fold
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',        # 5-fold cross-validation
    n_splits=5,
    output_metrics=['rmse'],    # Metrics to aggregate
    capture_outputs=['scores']  # Outputs to capture per fold
)
```

### Step 2: Create Your Function

Your function must:
1. Accept `X_train` and `X_test` (CVPipeline splits the data)
2. Accept `fold` parameter (fold index: 0, 1, 2, 3, 4)
3. Return a dict with 'rmse' and 'scores'

```python
import numpy as np
from sklearn.decomposition import PCA

def pca_decomposition(X_train, X_test, fold=0):
    """
    Run PCA and return metrics + outputs for capture.
    
    Called 5 times: once for each fold with different training data.
    """
    # Fit PCA on training data (different subset each fold)
    pca = PCA(n_components=3, random_state=fold)
    pca.fit(X_train)
    
    # Project test data
    test_scores = pca.transform(X_test)  # Shape: (20, 3)
    
    # Compute reconstruction error
    X_recon = pca.inverse_transform(test_scores)
    rmse = np.sqrt(np.mean((X_test - X_recon) ** 2))
    
    # Return dict with metrics and outputs
    return {
        'rmse': rmse,           # Metric (aggregated)
        'scores': test_scores   # Output (captured per fold)
    }
```

### Step 3: Run Through CV Pipeline

```python
import numpy as np

# Create sample data
X = np.random.randn(100, 20)  # 100 samples, 20 features

# Run through CV
pipeline = CVPipeline(cv_config)
results = pipeline.run(pca_decomposition, X=X)
```

### Step 4: Examine Results

```python
# Results structure:
# {
#     'rmse_folds': [0.88, 0.92, 0.85, 0.87, 0.90],
#     'rmse_mean': 0.884,
#     'rmse_std': 0.028,
#     'rmse_min': 0.85,
#     'rmse_max': 0.92,
#     'scores_fold_0': array(20, 3),  ← NEW: Fold outputs
#     'scores_fold_1': array(20, 3),
#     'scores_fold_2': array(20, 3),
#     'scores_fold_3': array(20, 3),
#     'scores_fold_4': array(20, 3),
#     'n_folds': 5
# }

print("RMSE per fold:", results['rmse_folds'])
print("RMSE mean:", results['rmse_mean'])
print("RMSE std:", results['rmse_std'])
```

### Step 5: Assess Output Stability

Now you can compare the scores across folds to see how stable they are:

```python
# Extract scores from all folds
scores_fold_0 = results['scores_fold_0']  # (20, 3)
scores_fold_1 = results['scores_fold_1']  # (20, 3)
scores_fold_2 = results['scores_fold_2']  # (20, 3)
scores_fold_3 = results['scores_fold_3']  # (20, 3)
scores_fold_4 = results['scores_fold_4']  # (20, 3)

# Stack them together
scores_list = [scores_fold_0, scores_fold_1, scores_fold_2, scores_fold_3, scores_fold_4]
scores_stacked = np.stack(scores_list, axis=0)  # (5, 20, 3)
# Shape breakdown:
# - Axis 0: 5 folds
# - Axis 1: 20 test samples
# - Axis 2: 3 PCA components

# Compute stability: Standard deviation across folds
score_stability = np.std(scores_stacked, axis=0)  # (20, 3)

# Interpretation:
# - score_stability[i, j] = how much does sample i's component j vary across folds?
# - Low values (e.g., 0.05) = Stable (good!)
# - High values (e.g., 0.8) = Unstable (concerning!)

print("\nScore Stability Analysis:")
print(f"Mean stability: {np.mean(score_stability):.4f}")
print(f"Std of stabilities: {np.std(score_stability):.4f}")
print(f"Max instability: {np.max(score_stability):.4f}")
print(f"Min stability: {np.min(score_stability):.4f}")
```

### Step 6: Interpret Results

```python
# Which samples have the most unstable scores?
max_instability_per_sample = np.max(score_stability, axis=1)
unstable_samples = np.argsort(max_instability_per_sample)[-5:]  # Top 5 most unstable
print(f"\nMost unstable samples: {unstable_samples}")

# Which components are most stable?
component_stability = np.mean(score_stability, axis=0)
print(f"\nStability per component:")
print(f"  Component 0: {component_stability[0]:.4f}")
print(f"  Component 1: {component_stability[1]:.4f}")
print(f"  Component 2: {component_stability[2]:.4f}")
```

---

## What This Tells You

### High Stability (Low std)
✅ **Good!** PCA is **robust** and **reproducible**
- Scores are consistent regardless of training subset
- Model generalizes well
- Can trust the PCA components

### Low Stability (High std)
❌ **Concerning!** PCA is **unstable** and **sensitive**
- Scores change significantly with different training data
- Possible overfitting or noise
- Should investigate data quality or parameters

---

## Advanced: Capture Multiple Outputs

You can capture more than just scores:

```python
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    capture_outputs=['scores', 'loadings', 'explained_variance']
)

def pca_decomposition(X_train, X_test, fold=0):
    pca = PCA(n_components=3)
    pca.fit(X_train)
    
    return {
        'rmse': compute_error(X_test, pca),
        'scores': pca.transform(X_test),
        'loadings': pca.components_,
        'explained_variance': pca.explained_variance_
    }

results = pipeline.run(pca_decomposition, X=X)

# Now you can assess:
# - Score stability
# - Loading changes (component drift)
# - Explained variance consistency
```

---

## Complete Runnable Example

See [tests/test_cv_output_stability.py](../tests/test_cv_output_stability.py) for a complete, runnable example that you can copy and adapt.

To run it:
```bash
python tests/test_cv_output_stability.py
```

---

## Why This Matters

### Scientific Validation
Testing output stability is important for:
- **Validating PCA robustness**: Are components stable?
- **Detecting overfitting**: Do coefficients change wildly?
- **Assessing reproducibility**: Can other labs reproduce results?

### Practical Uses
- **Quality control**: Flag unstable models
- **Feature engineering**: Verify learned features are robust
- **Model comparison**: Which model is more stable?
- **Data diagnostics**: Identify problematic samples

---

## Key Points to Remember

✅ Your function is **called 5 times** (once per fold)  
✅ Each call gets **different training data**  
✅ Each call produces **different outputs** (different PCA model)  
✅ You can **compare outputs** to assess stability  
✅ Metrics are **aggregated** (mean, std, min, max)  
✅ Outputs are **preserved per-fold** (scores_fold_0, etc.)  

---

## Next Steps

1. **Try the example**: Run `tests/test_cv_output_stability.py`
2. **Adapt to your data**: Modify the function to use your actual data
3. **Capture what you need**: Add outputs to `capture_outputs` list
4. **Analyze stability**: Compute std, correlation, or similarity metrics
5. **Interpret results**: Check if stability is acceptable for your use case

---

## See Also

- [OUTPUT_STABILITY_ASSESSMENT.md](OUTPUT_STABILITY_ASSESSMENT.md) - Comprehensive guide
- [CV_OUTPUT_CAPTURE_QUICK_REF.md](CV_OUTPUT_CAPTURE_QUICK_REF.md) - Quick reference
- [CV_INTEGRATION_GUIDE.md](CV_INTEGRATION_GUIDE.md) - How to add CV to functions
- [chemometrics/cv_pipeline.py](../chemometrics/cv_pipeline.py) - Source code
