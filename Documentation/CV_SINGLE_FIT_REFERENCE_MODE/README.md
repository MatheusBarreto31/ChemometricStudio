# Single-Fit Reference Mode

Compare CV folds against a single fit on full data for stability assessment.

## Concept

1. **Single Fit**: Train on all data
2. **CV Folds**: Train on each fold separately
3. **Comparison**: How different are fold outputs from single fit?
4. **Result**: Low RMSE = stable (consistent across data subsets)

## Example: PCA Stability

```python
from sklearn.decomposition import PCA

def pca_model(X_train=None, X_test=None, fold=0, **kwargs):
    pca = PCA(n_components=3)
    pca.fit(X_train)
    return {'scores': pca.transform(X_test)}

# Run CV with output reference
results = pipeline.run(
    pca_model, X=X,
    reference_output_key='scores',  # Use single-fit scores as reference
    comparison_output_key='scores'  # Compare fold scores (defaults to same)
)

# Lower RMSE = more stable scores
print(f"Score stability (RMSE): {results['rmse_mean']:.4f}")
```

## Example: Regression Stability

```python
def regression_model(X_train=None, X_test=None, Y_train=None, Y_test=None, fold=0, **kwargs):
    from sklearn.linear_model import Ridge
    model = Ridge()
    model.fit(X_train, Y_train)
    return {
        'y_pred': model.predict(X_test),
        'coefficients': model.coef_
    }

# Compare fold predictions to single-fit
results = pipeline.run(
    regression_model, X=X, Y=Y,
    reference_output_key='y_pred',  # Single-fit predictions as reference
    capture_output_keys=['y_pred', 'coefficients']
)

print(f"Prediction stability: {results['rmse_mean']:.4f}")
print(f"Coefficient stability across folds: {results['coefficients_cv']}")
```

## Use Cases

✓ **Assess model robustness**: Does model change with different training data?  
✓ **Feature stability**: Do PCA loadings vary across folds?  
✓ **Parameter consistency**: Do coefficients differ significantly by fold?  
✓ **Debug overfitting**: Large RMSE suggests data-dependent behavior  

## Output Reference vs Input Reference

| Mode | Reference | Use Case |
|------|-----------|----------|
| **Output** | Single-fit output | Stability assessment |
| **Input** | Ground truth (Y) | Accuracy evaluation |

```python
# Output reference: fold vs single fit
results = pipeline.run(func, X=X, reference_output_key='scores')

# Input reference: fold vs actual Y
results = pipeline.run(func, X=X, Y=Y, reference_input_key='Y')
```

## See Also

- [Main CV Pipeline Documentation](../CV_PIPELINE.md)
- [CV Integration Guide](../CV_INTEGRATION_GUIDES/)
- [Fold Output Handling](../CV_FOLD_OUTPUT_HANDLING/)
