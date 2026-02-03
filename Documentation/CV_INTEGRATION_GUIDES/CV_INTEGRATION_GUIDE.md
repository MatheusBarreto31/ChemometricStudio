# CV Integration Guide

## Quick Start: Adding CV Support to Your Function

This guide shows how to add cross-validation support to any modeling function in 3 steps.

### Step 1: Accept `cv_config` Parameter

Modify your function signature to accept `cv_config`:

```python
def my_function(X, Y=None, fold=0, cv_config=None, **other_params):
    """
    Your modeling function with CV support.
    
    Args:
        X: Training/test data
        Y: Optional target values
        fold: Current fold index (0, 1, 2, ...). Set by CVPipeline.
        cv_config: CVConfig object from CV Configuration node. Set by router.
        **other_params: Your function's normal parameters
    
    Returns:
        dict with your metric results
    """
    pass
```

### Step 2: Return Dictionary with Metrics

Change your function to return a dictionary:

```python
def my_function(X, Y=None, fold=0, cv_config=None):
    # Do your work
    result = some_calculation(X, Y)
    rmse = compute_error(result, Y)
    
    # Return dict instead of tuple/scalar
    return {
        'rmse': rmse,  # Metric name must be in cv_config.output_metrics
        'r2': compute_r2(result, Y),  # Optional additional metric
    }
```

### Step 3: Add to function_specs.json

Update `function_specs.json` to declare CV support:

```json
{
  "function_specs": {
    "my_function": {
      "return_specs": ["my_output"],
      "input_specs": {
        "X": "array",
        "Y": "array",
        "cv_config": "cv_config"  // NEW: Add this
      },
      "return_specs": {
        "my_output": ["results", "cv_results"],  // Results can be CV or single fit
      }
    }
  }
}
```

That's it! Your function now supports CV through routing.

---

## How It Works: The Routing Pattern

### Without CV (Single Fit)

```
User Input (X, Y) → my_function() → Results
```

### With CV (Via Router)

```
CVConfiguration → [cv_config]
User Input (X, Y) → my_function(X, Y, cv_config) → Results + CV_Results
```

The CV Configuration node **routes its output** to your function. If the router detects `cv_config` is enabled, the CVPipeline automatically:

1. **Splits data** into K folds
2. **Calls your function K times** (once per fold with `fold=0, 1, 2, ...`)
3. **Aggregates metrics** across folds
4. **Returns** both single-fit and CV results

---

## Complete Example: Univariate Calibration

Here's how univariate calibration was enhanced:

```python
from chemometrics.cv_pipeline import CVPipeline, CVConfig

def univariate_calibration(
    X_train,
    X_test,
    Y_train,
    Y_test,
    fold=0,
    cv_config=None,  # NEW: Added parameter
    poly_order=2,
    **kwargs
):
    """
    Univariate polynomial calibration with CV support.
    """
    
    # Single fit logic
    def _fit_model():
        # Your model fitting code
        model = fit_polynomial(X_train, Y_train, poly_order)
        Y_pred = model.predict(X_test)
        rmse = compute_rmse(Y_pred, Y_test)
        r2 = compute_r2(Y_pred, Y_test)
        
        return {
            'rmse': rmse,
            'r2': r2,
            'coefficients': model.coef_,  # Optional: for output capture
        }
    
    # Check if CV is enabled
    if cv_config and cv_config.is_enabled():
        # Let CVPipeline handle the folding
        pipeline = CVPipeline(cv_config)
        return pipeline.run(
            _univariate_calibration_single_fit,
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            fold=fold,
            poly_order=poly_order
        )
    else:
        # Single fit (no CV)
        return _fit_model()


def _univariate_calibration_single_fit(
    X_train, X_test, Y_train, Y_test, fold=0, poly_order=2
):
    """Helper function for single fit logic."""
    # ... same logic as above ...
    return {'rmse': rmse, 'r2': r2}
```

---

## Advanced: Capturing Function Outputs

You can now capture non-metric outputs from each fold to assess **output stability**.

### Setup: Configure Output Capture

```python
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    output_metrics=['rmse', 'r2'],
    capture_outputs=['scores', 'loadings']  # NEW: What to capture
)
```

### Function: Return Outputs to Capture

```python
def pca_decomposition(X_train, X_test, fold=0, cv_config=None):
    pca = PCA(n_components=3)
    pca.fit(X_train)
    test_scores = pca.transform(X_test)
    rmse = compute_reconstruction_error(X_test, pca)
    
    return {
        'rmse': rmse,
        'scores': test_scores,  # Gets captured as 'scores_fold_0', etc.
        'loadings': pca.components_
    }
```

### Results: Access Captured Outputs

```python
results = pipeline.run(pca_decomposition, X=X)

# Metrics (aggregated)
print(results['rmse_mean'])  # 0.88
print(results['rmse_std'])   # 0.03

# Captured outputs (per-fold)
print(results['scores_fold_0'])  # Shape (20, 3)
print(results['scores_fold_1'])  # Shape (20, 3)
print(results['loadings_fold_0'])  # Shape (3, 20)

# Assess stability
scores_list = [results[f'scores_fold_{i}'] for i in range(5)]
stability = np.std(np.stack(scores_list), axis=0)
```

---

## Common Patterns

### Pattern 1: Optional CV

Function works with or without CV configuration:

```python
def my_function(X, Y=None, fold=0, cv_config=None):
    if cv_config and cv_config.is_enabled():
        # CV mode: split data, handle folds
        pipeline = CVPipeline(cv_config)
        return pipeline.run(_my_function_single_fit, X=X, Y=Y)
    else:
        # Single fit mode
        return _my_function_single_fit(X, Y)
```

### Pattern 2: Multiway Data (3D, 4D, etc.)

CV automatically handles multiway data:

```python
# Your function signature is the same
def my_function(X_train, X_test, fold=0, cv_config=None):
    # X_train could be (80, 50, 100) - 3D
    # X_test could be (20, 50, 100)
    # CVPipeline handles splitting on axis 0 (samples)
    pca = PCA(n_components=3)
    scores = pca.fit_transform(X_train.reshape(X_train.shape[0], -1))
    # ... rest of logic
```

### Pattern 3: Stratified CV

For classification with target distribution preservation:

```python
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='stratified_kfold',  # Preserves class ratios
    n_splits=5,
    output_metrics=['accuracy', 'f1']
)
```

### Pattern 4: Time Series CV (Forward Chaining)

For temporal data where future can't leak to past:

```python
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='timeseries',  # Time-aware splits
    n_splits=5
)
```

---

## Testing Your Integration

### Test 1: Verify CV is Optional

```python
# Should work without CV config
result1 = my_function(X, Y)
assert 'rmse' in result1

# Should work with CV disabled
cv_config = CVConfig(use_cv=False)
result2 = my_function(X, Y, cv_config=cv_config)
assert 'rmse' in result2
```

### Test 2: Verify CV Produces Multiple Folds

```python
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    output_metrics=['rmse']
)
result = my_function(X, Y, cv_config=cv_config)

assert 'rmse_folds' in result
assert len(result['rmse_folds']) == 5  # One per fold
assert 'rmse_mean' in result
assert 'rmse_std' in result
```

### Test 3: Verify Output Capture

```python
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    capture_outputs=['scores']
)
result = my_function(X, Y, cv_config=cv_config)

assert 'scores_fold_0' in result
assert 'scores_fold_1' in result
assert 'scores_fold_2' in result
assert 'scores_fold_3' in result
assert 'scores_fold_4' in result
```

---

## Common Mistakes & Fixes

### Mistake 1: Forgetting to Return Dict

❌ **Wrong**:
```python
def my_function(X, Y=None, fold=0, cv_config=None):
    rmse = compute_error(X, Y)
    return rmse  # Scalar, not dict
```

✅ **Correct**:
```python
def my_function(X, Y=None, fold=0, cv_config=None):
    rmse = compute_error(X, Y)
    return {'rmse': rmse}  # Dict with metric
```

### Mistake 2: Forgetting `fold` Parameter

❌ **Wrong**:
```python
def my_function(X, Y=None, cv_config=None):
    # CVPipeline tries to pass fold=0, 1, 2, ... but function doesn't accept it
    pass
```

✅ **Correct**:
```python
def my_function(X, Y=None, fold=0, cv_config=None):
    # Include fold parameter even if you don't use it
    pass
```

### Mistake 3: Not Handling Train/Test Split

❌ **Wrong**:
```python
# CVPipeline splits X into X_train and X_test
# But your function expects just X
def my_function(X, Y=None, fold=0, cv_config=None):
    model = fit(X, Y)  # Which X? There are two!
```

✅ **Correct**:
```python
# Accept both X_train and X_test
def my_function(X_train, X_test, Y_train, Y_test, fold=0, cv_config=None):
    model = fit(X_train, Y_train)
    pred = model.predict(X_test)
```

### Mistake 4: Missing Metric in Output

❌ **Wrong**:
```python
cv_config = CVConfig(output_metrics=['rmse', 'r2'])

def my_function(X, Y=None, fold=0, cv_config=None):
    result = some_calc(X, Y)
    return {
        'accuracy': result,  # But output_metrics asked for 'rmse'!
    }
```

✅ **Correct**:
```python
def my_function(X, Y=None, fold=0, cv_config=None):
    result = some_calc(X, Y)
    return {
        'rmse': compute_rmse(result, Y),
        'r2': compute_r2(result, Y)
    }
```

---

## File Structure

- **[chemometrics/cv_pipeline.py](../chemometrics/cv_pipeline.py)**: Core CV infrastructure
  - `CVConfig`: Configuration dataclass with `capture_outputs` parameter
  - `CVPipeline`: Orchestrator that manages fold splitting and result aggregation
  - `CVSplitter`: Abstract base for split strategies
  - 5 concrete splitters: KFoldSplitter, StratifiedKFoldSplitter, TimeSeriesSplitter, RepeatedKFoldSplitter, ShuffleSplitSplitter

- **`gui_configs/en/cv_configuration_config.json`**: GUI node for CV parameters
  - Allows users to set CV strategy, number of folds, metrics, and output capture

- **`function_specs.json`**: Function metadata
  - Lists `cv_configuration` as available function
  - Declares which functions accept `cv_config` parameter

- **`model.json`**: Default model configuration
  - Contains `cv_configuration` node with default parameters

---

## Next Steps

1. **Add CV to your function**: Follow the 3-step pattern above
2. **Test with small data**: Verify CV works correctly
3. **Add output capture** (optional): If you want stability assessment
4. **Document your metrics**: What does 'rmse' mean in your function?

See also:
- [OUTPUT_STABILITY_ASSESSMENT.md](OUTPUT_STABILITY_ASSESSMENT.md) - Advanced stability testing
- [CV_IMPLEMENTATION_NOTES.md](CV_IMPLEMENTATION_NOTES.md) - Architecture details
- [chemometrics/cv_pipeline.py](../chemometrics/cv_pipeline.py) - Source code
