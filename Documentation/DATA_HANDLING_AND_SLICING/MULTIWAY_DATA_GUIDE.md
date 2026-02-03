# Multiway Data & Cross-Validation Guide

## Overview

CM Studio's CV pipeline is **fully compatible with multiway (3D, 4D, etc.) data**. No special configuration or data reshaping needed—it just works.

## Data Dimensions Supported

| Data Type | Shape | Example | Notes |
|-----------|-------|---------|-------|
| **Vector** | (n_samples,) | 50 absorption spectra | Rare but supported |
| **Univariate (2D)** | (n_samples, n_vars) | 50 samples × 103 wavelengths | Current standard |
| **Multiway (3D)** | (n_samples, d1, d2) | 50 × 103 wavelengths × 35 time points | Spectroscopy + temporal |
| **Multiway (4D)** | (n_samples, d1, d2, d3) | 50 × 103 × 35 × 3 electrodes | Multi-sensor, multi-temporal |
| **Multiway (5D+)** | (n_samples, d1, d2, ..., dN) | Any combination | **No limit** — fully supported |

## How CV Pipeline Splits Multiway Data

**Core rule: Always split on axis 0 (samples), never reshape or flatten.**

The pipeline is **dimension-agnostic and supports any number of dimensions (1D, 2D, 3D, 4D, 5D, 6D, ...)**

```python
import numpy as np
from chemometrics.cv_pipeline import CVPipeline, cv_configuration

# 3D data example: Fluorescence spectra over time
X = np.random.randn(50, 103, 35)  # 50 samples, 103 wavelengths, 35 time points
Y = np.random.randn(50, 2)        # 50 samples, 2 responses

cv_config = cv_configuration(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    output_metrics=['rmse', 'r2']
)['cv_config']

# Your modeling function receives properly split 3D data
pipeline = CVPipeline(cv_config)
cv_results = pipeline.run(
    your_multiway_calibration_function,
    X=X,      # Shape: (50, 103, 35)
    Y=Y       # Shape: (50, 2)
)

# Internally, for each fold:
# X_train shape: (40, 103, 35)  [80% of samples, all wavelengths, all time points]
# X_test shape:  (10, 103, 35)  [20% of samples, all wavelengths, all time points]
# Structure preserved; no flattening
```

## What Your Function Receives

When you implement a multiway-compatible function:

```python
def my_multiway_calibration(X_train, X_test, Y_train, Y_test, fold=0):
    """
    Args:
        X_train: (n_train_samples, 103, 35) - Multiway training data
        X_test:  (n_test_samples, 103, 35)  - Multiway test data
        Y_train: (n_train_samples, 2)       - Training responses
        Y_test:  (n_test_samples, 2)        - Test responses
        fold:    int                         - Which fold (0, 1, 2, 3, 4)
    
    Returns:
        {'rmse': 0.85, 'r2': 0.92}  # Metrics for this fold
    """
    # Build model on training data (preserves 3D structure)
    model = fit_multiway_model(X_train, Y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Return metrics
    return {
        'rmse': np.sqrt(np.mean((Y_test - y_pred) ** 2)),
        'r2': 1 - (np.sum((Y_test - y_pred) ** 2) / np.sum((Y_test - np.mean(Y_test)) ** 2))
    }
```

## Real-World Examples

### Example 1: Time-Resolved Spectroscopy

```
Data: Fluorescence emission spectra measured at different time points
Shape: (30 samples, 256 wavelengths, 50 time points)
Response: (30 samples, 2 analyte concentrations)

CV Setup:
  - K-Fold with 5 splits
  - Each fold: 24 train samples, 6 test samples
  - All 256 wavelengths, all 50 time points preserved
  
Train fold 1: (24, 256, 50)
Test fold 1:  (6, 256, 50)
```

### Example 2: Multi-Electrode Array + Temporal

```
Data: Brain activity from 64 electrodes, sampled at 250 Hz over 10 seconds
Shape: (100 subjects, 64 electrodes, 2500 time samples)
Response: (100 subjects, 1 target variable)

CV Setup:
  - Stratified K-Fold (keep patient groups balanced)
  - 5 splits
  
Train fold 1: (80, 64, 2500)
Test fold 1:  (20, 64, 2500)
→ All electrode channels and all time samples preserved
```

### Example 3: Hyperspectral Imaging

```
Data: Hyperspectral cube from NIR imaging
Shape: (150 image tiles, 224 wavelengths, 256 x 256 spatial pixels)
Response: (150 tiles, 3 soil properties)

CV Setup:
  - Time Series CV (samples ordered by depth)
  - 5 folds with forward-chaining
  
Train fold 1: (30, 224, 256, 256)
Test fold 1:  (30, 224, 256, 256)
→ All spectral and spatial information intact
```

## Implementation Checklist

When building a multiway-compatible function:

- [ ] Accept multiway arrays (don't assume 2D)
- [ ] Don't flatten or reshape input data
- [ ] Handle arbitrary number of dimensions beyond samples
- [ ] Test with 2D, 3D, and 4D data
- [ ] Document expected shapes in docstring
- [ ] Return metrics dict (one entry per metric)

Example template:

```python
def my_calibration(X_train, X_test, Y_train, Y_test, some_param=1, fold=0):
    """
    Multiway-compatible calibration function.
    
    Args:
        X_train: Training data, shape (n_train, ...) where ... can be any dimensions
        X_test:  Test data, shape (n_test, ...)
        Y_train: Training responses, shape (n_train, n_responses)
        Y_test:  Test responses, shape (n_test, n_responses)
        some_param: Your model parameter
        fold: Fold index (passed by CV pipeline)
    
    Returns:
        dict with metrics: {'rmse': float, 'r2': float}
    """
    # Model should handle X's shape gracefully
    # Many sklearn models flatten internally - that's ok
    # Or use multiway-aware models (PARAFAC, Tucker, etc.)
    
    model = build_model(some_param)
    model.fit(X_train.reshape(X_train.shape[0], -1), Y_train)
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    
    return {
        'rmse': compute_rmse(Y_test, y_pred),
        'r2': compute_r2(Y_test, y_pred)
    }
```

## Common Patterns

### Pattern 1: Flatten to 2D for Standard Models

```python
# If using sklearn models (they expect 2D):
def my_pls(X_train, X_test, Y_train, Y_test, n_components=3, fold=0):
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    
    # Flatten all dimensions except first (samples)
    X_train_2d = X_train.reshape(n_train_samples, -1)
    X_test_2d = X_test.reshape(n_test_samples, -1)
    
    # Standard model
    from sklearn.cross_decomposition import PLSRegression
    model = PLSRegression(n_components=n_components)
    model.fit(X_train_2d, Y_train)
    y_pred = model.predict(X_test_2d)
    
    return {'rmse': compute_rmse(Y_test, y_pred)}
```

### Pattern 2: Use Multiway-Aware Models

```python
# If using tensorly, PARAFAC, Tucker, etc.:
def my_parafac(X_train, X_test, Y_train, Y_test, n_components=3, fold=0):
    # These models handle 3D+ data natively
    # Build PARAFAC model on training (preserves structure)
    factors, _ = parafac(X_train, rank=n_components)
    
    # Score/predict on test
    y_pred = score_parafac(X_test, factors)
    
    return {'rmse': compute_rmse(Y_test, y_pred)}
```

## Verification: Test with Multiple Dimensions

```python
import numpy as np
from chemometrics.cv_pipeline import cv_configuration

# Quick test for your new function
def test_multiway_function():
    # Test 2D
    X_2d = np.random.randn(30, 20)
    Y = np.random.randn(30, 2)
    cv_config = cv_configuration(use_cv=True, cv_strategy='kfold', n_splits=3)['cv_config']
    result_2d = your_function(X_2d, Y, cv_config=cv_config)
    assert 'cv_results' in result_2d
    
    # Test 3D
    X_3d = np.random.randn(30, 20, 15)
    result_3d = your_function(X_3d, Y, cv_config=cv_config)
    assert 'cv_results' in result_3d
    
    # Test 4D
    X_4d = np.random.randn(30, 20, 15, 3)
    result_4d = your_function(X_4d, Y, cv_config=cv_config)
    assert 'cv_results' in result_4d
    
    print("All dimension tests passed!")
```

## FAQ

**Q: Do I need to handle `nway_flag`?**
A: The CV pipeline doesn't care about `nway_flag`. If your function needs it (for internal reshaping), pass it as a constant parameter: `cv_pipeline.run(func, ..., nway_flag=3)`.

**Q: What if my data is 5D or 6D?**
A: Works the same way. CV pipeline splits on axis 0, preserves all other dimensions. Your function just needs to handle that shape.

**Q: Should I reshape multiway to 2D before CV?**
A: No. Let the CV pipeline receive multiway data as-is. Reshape inside your function if needed (e.g., for sklearn models).

**Q: Does stratification work with multiway data?**
A: Yes. `StratifiedKFold` uses the `y` vector (responses), which is always 2D. The multiway dimensions don't affect stratification.

**Q: What about temporal data? Should I use TimeSeriesSplit?**
A: If your samples have a temporal ordering (e.g., measurements over days/weeks), yes. `TimeSeriesSplit` respects that order and prevents lookahead bias. The fact that each sample is 3D/4D doesn't matter—splitting still respects time order.
