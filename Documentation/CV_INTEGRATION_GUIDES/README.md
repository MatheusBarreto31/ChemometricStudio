# CV Integration Guide

Quick reference for integrating your model with the CM-Studio CV pipeline.

## The Pattern

Every model function follows this signature:

```python
def my_model(X_train=None, X_test=None, Y_train=None, Y_test=None, fold=0, **kwargs):
    """
    Train on fold's training set, predict on test set.
    
    Args:
        X_train, X_test: Split feature data
        Y_train, Y_test: Split target data
        fold: Current fold index (0, 1, 2, ...)
    
    Returns:
        dict: {'output_name': array, ...}
    """
    model = MyModel()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    
    return {
        'y_pred': predictions,      # Output to compare
        'loadings': model.weights_  # Output to capture
    }
```

## 3-Step Integration

### Step 1: Configure CV
```python
from chemometrics.cv_pipeline import cv_configuration

cv_cfg = cv_configuration(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5
)
```

### Step 2: Define Your Model Function
```python
def my_model_wrapper(X_train=None, X_test=None, Y_train=None, Y_test=None, fold=0, **kwargs):
    return {'y_pred': predictions}
```

### Step 3: Run Through Pipeline
```python
from chemometrics.cv_pipeline import CVPipeline

pipeline = CVPipeline(cv_cfg['cv_config'])
results = pipeline.run(
    my_model_wrapper,
    X=X, Y=Y,
    reference_input_key='Y',
    comparison_output_key='y_pred',
    capture_output_keys=['y_pred']
)

print(f"RMSE: {results['rmse_mean']:.4f}")
print(f"R²:   {results['r2_mean']:.4f}")
```

## Real Examples

### Linear Regression
```python
from sklearn.linear_model import LinearRegression

def linear_model(X_train=None, X_test=None, Y_train=None, Y_test=None, fold=0, **kwargs):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return {'y_pred': model.predict(X_test)}
```

### PCA Stability Assessment
```python
from sklearn.decomposition import PCA

def pca_model(X_train=None, X_test=None, fold=0, **kwargs):
    pca = PCA(n_components=3)
    pca.fit(X_train)
    return {
        'scores': pca.transform(X_test),
        'loadings': pca.components_
    }

# Run with output reference
results = pipeline.run(
    pca_model, X=X,
    reference_output_key='scores',
    capture_output_keys=['scores', 'loadings']
)
```

## See Also

- [Main CV Pipeline Documentation](../CV_PIPELINE.md)
- [Univariate Calibration Example](../UNIVARIATE_CALIBRATION/)
