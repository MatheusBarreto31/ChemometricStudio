# CV Fold Output Handling

When CV captures outputs, they're processed automatically based on data structure.

## Output Types

### Sample-Based Outputs (Standard)
These correspond to samples: `(n_test, n_features)`

**Result**: Reconstructed to full-size array
```python
results['y_pred_cv']  # Shape: (100, n_features) - all samples
```

### Non-Sample-Based Outputs (Multiway)
First dimension ≠ samples: `(n_wavelengths, n_components)`

**Result**: Segregated by fold (cannot reconstruct order)
```python
results['loadings_cv'][0]   # Fold 0 loadings
results['loadings_cv'][1]   # Fold 1 loadings
```

## Example

```python
results = pipeline.run(
    my_model, X=X, Y=Y,
    reference_input_key='Y',
    comparison_output_key='y_pred',
    capture_output_keys=['y_pred', 'loadings']
)

# Sample-based: fully reconstructed
y_pred_full = results['y_pred_cv']  # (100, n_response)

# Non-sample-based: per-fold access
for i in range(5):
    loadings = results['loadings_cv'][i]  # (n_wavelengths, n_comp)
```

## See Also

- [Main CV Pipeline Documentation](../CV_PIPELINE.md)
- [CV Integration Guide](../CV_INTEGRATION_GUIDES/)
