# Cross-Validation (CV) Pipeline

The CM-Studio CV pipeline provides a **unified, flexible cross-validation framework** that works with any modeling function.

## Key Features

✅ **Unified Reference Framework**: Both input references (Y_cal) and output references (single-fit)  
✅ **Matrix Support**: Works with vector predictions and matrix outputs (PCA scores, multi-response)  
✅ **Flexible Metrics**: RMSE, R², MAE, Bias, SEP - computed automatically from comparisons  
✅ **Output Capture**: Reconstruct predictions or segregate non-sample outputs by fold  
✅ **Multiple Strategies**: K-Fold, Stratified, Time Series, LOOCV, Bootstrap, Moving Window, etc.  
✅ **Multiway Data**: Handles 2D, 3D, 4D+ arrays (splits on samples axis only)  

## Quick Start

### 1. Configure CV Strategy
```python
from chemometrics.cv_pipeline import cv_configuration

cv_cfg = cv_configuration(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5
)
```

### 2. Run Your Model Through CV
```python
from chemometrics.cv_pipeline import CVPipeline

pipeline = CVPipeline(cv_cfg['cv_config'])

results = pipeline.run(
    my_model_function,
    X=X, Y=Y,                           # Your data
    reference_input_key='Y',            # Compare to actual Y
    comparison_output_key='y_pred',     # My model's prediction output
    capture_output_keys=['y_pred']      # What to reconstruct
)
```

### 3. Access Results
```python
# Metrics aggregated across folds
print(f"RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
print(f"R²:   {results['r2_mean']:.4f}")

# Reconstructed predictions (full dataset)
y_pred_full = results['y_pred_cv']  # Shape: (100, n_response)

# Per-fold metrics
print(f"Per-fold RMSE: {results['rmse_folds']}")
```

## Core Concepts

### Reference Modes

**Input Reference** (Standard CV)
- Compares model output against actual input (e.g., Y_cal)
- Each fold: fold_output vs reference[test_idx]
- Use case: Evaluating prediction accuracy

```python
results = pipeline.run(
    model_func, X=X, Y=Y,
    reference_input_key='Y',        # Use Y as ground truth
    comparison_output_key='y_pred'  # Compare predictions
)
```

**Output Reference** (Stability Assessment)
- Runs single fit on full data, compares folds to it
- Each fold: fold_output vs single_fit_output[test_idx]
- Use case: Assessing model stability across data subsets

```python
results = pipeline.run(
    model_func, X=X,
    reference_output_key='scores',  # Use single-fit scores as reference
    # comparison_output_key defaults to 'scores'
)
```

### Output Types

**Sample-Based** (predictions, scores)
- Shape: `(n_test, n_features)`
- Result: Reconstructed to full-size `(n_samples, n_features)`
- Access: `results['y_pred_cv'][i]` works

**Non-Sample-Based** (loadings, weights)
- Shape: `(n_wavelengths, n_components)` - first dim ≠ samples
- Result: Segregated by fold (can't reconstruct order)
- Access: `results['loadings_cv'][fold_idx]`

```python
# Sample-based reconstruction
y_full = results['y_pred_cv']  # (100,) - all samples in order

# Non-sample-based segregation
for fold_idx in range(5):
    loadings = results['loadings_cv'][fold_idx]  # (n_wavelengths, n_comp)
```

### Metrics

Automatically computed from reference comparison:

| Metric | Formula | Use Case |
|--------|---------|----------|
| **rmse** | √(mean((y-ŷ)²)) | Primary accuracy |
| **r2** | 1 - SS_res/SS_tot | Explained variance |
| **mae** | mean(\|y-ŷ\|) | Robust to outliers |
| **mse** | mean((y-ŷ)²) | Penalizes large errors |
| **bias** | mean(y-ŷ) | Systematic over/underestimation |
| **sep** | √(mean((y-ŷ-bias)²)) | Random error |

## Complete Example

```python
import numpy as np
from chemometrics.cv_pipeline import cv_configuration, CVPipeline
from sklearn.linear_model import LinearRegression

# Create data
X = np.random.randn(100, 5)
Y = 2*X[:, 0] + 0.5*X[:, 1] + np.random.randn(100) * 0.1

# Configure CV
cv_cfg = cv_configuration(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5
)

# Define model function (called by pipeline)
def my_regression_model(X_train=None, X_test=None, Y_train=None, Y_test=None, fold=0, **kwargs):
    """
    Model function signature:
    - X_train, X_test, Y_train, Y_test: Split data
    - fold: Current fold index
    - Returns: dict with outputs
    """
    model = LinearRegression()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    return {
        'y_pred': y_pred,           # What to compare/reconstruct
        'coefficients': model.coef_  # Additional output
    }

# Run CV
pipeline = CVPipeline(cv_cfg['cv_config'])
results = pipeline.run(
    my_regression_model,
    X=X, Y=Y,
    reference_input_key='Y',           # Compare to actual Y
    comparison_output_key='y_pred',    # Compare predictions
    output_metrics=['rmse', 'r2', 'mae'],
    capture_output_keys=['y_pred']
)

# Analyze results
print(f"RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
print(f"R²:   {results['r2_mean']:.4f}")
print(f"Per-fold RMSE: {results['rmse_folds']}")

# Full reconstructed predictions
y_pred_all = results['y_pred_cv']  # Shape: (100,)
print(f"Reconstructed predictions: {y_pred_all.shape}")
```

## Supported Strategies

| Strategy | Use Case | Parameters |
|----------|----------|------------|
| **kfold** | General, IID data | n_splits, shuffle |
| **stratified_kfold** | Classification | n_splits, shuffle |
| **timeseries** | Time-dependent data | n_splits |
| **repeated_kfold** | More robust estimates | n_splits, n_repeats |
| **shuffle_split** | Large datasets | n_splits, test_size |
| **venetian_windows** | Continuous signals | n_splits, window_size |
| **moving_window** | Sequential data | n_splits, window_size |
| **loocv** | Small datasets | None (n=n_samples) |
| **bootstrap** | Robust estimation | n_splits |

## Integration with Models

Each model function receives:
- Split data: `X_train`, `X_test`, `Y_train`, `Y_test`
- Metadata: `fold` (fold index)
- Additional kwargs

Should return dict with outputs to compare/capture.

```python
def model_function(X_train=None, X_test=None, fold=0, **kwargs):
    # Pipeline splits data and calls this per fold
    # fold=0, 1, 2, ... 4 (for 5-fold CV)
    # fold=-1 for single fit (if requested)
    
    model = train(X_train)
    predictions = model.predict(X_test)
    
    return {
        'y_pred': predictions,  # To be compared
        'other_output': something  # To be captured
    }
```

## Configuration (GUI-Level)

Only strategy parameters are exposed in GUI:

```json
{
  "use_cv": true,
  "cv_strategy": "kfold",
  "n_splits": 5,
  "random_state": 42,
  "shuffle": true,
  "window_size": null,
  "n_repeats": 10,
  "test_size": 0.2
}
```

## Configuration (Code-Level)

Model specifies comparison logic:

```python
results = pipeline.run(
    my_model,
    X=X, Y=Y,
    reference_input_key='Y',        # Input to use as reference
    comparison_output_key='y_pred', # Output to compare
    output_metrics=['rmse', 'r2'],  # What to compute
    capture_output_keys=['y_pred']  # What to capture
)
```

## Multiway Data Support

Works with any number of dimensions:

```python
# 2D (univariate)
X.shape  # (100, 5) - 100 samples, 5 variables
Y.shape  # (100, 1) - 100 samples, 1 response

# 3D (e.g., spectral time series)
X.shape  # (100, 50, 20) - 100 samples, 50 wavelengths, 20 timepoints
# Splits on axis 0 only: train (80, 50, 20), test (20, 50, 20)

# 4D (e.g., with auxiliary axis)
X.shape  # (100, 50, 20, 3) - samples, wavelength, time, electrode
# Works the same way: preserves 50, 20, 3
```

## Advanced Usage

### Stability Assessment
```python
# Compare PCA scores across folds
results = pipeline.run(
    pca_func,
    X=X,
    reference_output_key='scores',  # Single-fit scores as reference
    capture_output_keys=['scores', 'loadings']
)

# Lower RMSE = more stable scores
print(f"Score stability (RMSE): {results['rmse_mean']:.4f}")

# Access fold-specific loadings
for i in range(5):
    loadings = results['loadings_cv'][i]
    print(f"Fold {i} loadings shape: {loadings.shape}")
```

### Matrix References
```python
# Multi-response Y
Y.shape  # (100, 3) - 3 response variables

results = pipeline.run(
    multi_response_model,
    X=X, Y=Y,
    reference_input_key='Y',  # 3-column matrix
    comparison_output_key='y_pred'  # 3-column predictions
)

# RMSE computed across all 3 responses
print(f"RMSE (across all 3 responses): {results['rmse_mean']:.4f}")
```

### Custom Metrics
Extend by adding metrics to `output_metrics`:

```python
results = pipeline.run(
    model_func, X=X, Y=Y,
    reference_input_key='Y',
    comparison_output_key='y_pred',
    output_metrics=['rmse', 'r2', 'mae', 'bias', 'sep']
)

# All metrics are available
for metric in ['rmse', 'r2', 'mae', 'bias', 'sep']:
    print(f"{metric}_mean: {results[metric + '_mean']:.4f}")
```

## Implementation

**Main Classes**:
- `CVConfig`: Configuration dataclass
- `CVPipeline`: Pipeline orchestrator
- `FoldSegregatedOutput`: Index-based access to fold outputs

**Key Methods**:
- `pipeline.run()`: Execute CV
- `pipeline._compute_fold_metrics()`: Metric computation
- `pipeline._reconstruct_from_folds()`: Smart reconstruction

## Files

- `chemometrics/cv_pipeline.py` - Core implementation
- `chemometrics/univ_calibration.py` - Integration example
- Tests in `tests/` directory

## See Also

- [Univariate Calibration](../UNIVARIATE_CALIBRATION/) - Example integration
- [Data Handling](../DATA_HANDLING_AND_SLICING/) - Multiway data formats
