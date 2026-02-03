# Univariate Calibration Documentation

This folder contains documentation for **univariate calibration with single-fit reference mode** - using the single-fit reference feature with polynomial calibration.

## 📖 Files in This Folder

### Main Guide
- **UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md** ⭐ **START HERE**
  - Complete usage guide for univariate calibration
  - Single-fit reference mode setup
  - Results structure and interpretation
  - Analysis examples
  - Advanced patterns
  - (15 minute read)

---

## 🎯 Reading Guide

### "I want to use this feature" (15 min)
1. Read: UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md
2. See "Usage" section for your setup
3. Check "Analysis Examples" for patterns
4. Apply to your data

### "I need specific examples" (10 min)
- See "Analysis Examples" section in the documentation

### "I want advanced patterns" (10 min)
- See "Advanced: Custom Analysis" section

---

## ✨ Feature Highlights

✅ **Run single calibration**: On all data to get reference  
✅ **Run CV folds**: Compare fold results to single fit  
✅ **Per-fold metrics**: See how models vary  
✅ **Identify unstable**: Models that change across subsets  
✅ **Multivariate**: Support for multiple X and Y  

---

## 🚀 Quick Example

```python
from chemometrics.cv_pipeline import CVConfig
from chemometrics.univ_calibration import univariate_calibration

# Configure
config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5
)

# Run with single-fit reference
results = univariate_calibration(
    X_cal, Y_cal,
    cv_config=config,
    reference_output_key='y_cal_pred',
    capture_output_keys=['y_cal_pred', 'metrics']
)

# Analyze
print(f"Stability: {results['cv_results']['y_cal_pred_rmse']:.4f}")
print(f"Per-fold: {results['cv_results']['y_cal_pred_rmse_per_fold']}")

# Sample-based: predictions
preds_cv = results['cv_results']['y_cal_pred_cv']

# Non-sample-based: metrics per fold
metrics_cv = results['cv_results']['metrics_cv']
for i in range(len(metrics_cv)):
    fold_metrics = metrics_cv[i]
    # Analyze...
```

---

## 📊 Results Structure

### Single Fit Results
```python
results['y_cal_pred']      # Predictions on all data
results['models']          # Full model details
results['metrics']         # Metrics from single fit
```

### CV Comparison Results
```python
cv_results = results['cv_results']

# Metrics
cv_results['y_cal_pred_rmse']           # Overall stability
cv_results['y_cal_pred_rmse_per_fold']  # Per-fold metrics
cv_results['y_cal_pred_rmse_std']       # Variability

# Outputs (sample-based)
cv_results['y_cal_pred_cv']      # Predictions from folds
cv_results['y_cal_pred_single']  # Reference predictions

# Outputs (non-sample-based)
cv_results['metrics_cv']         # Metrics per fold
cv_results['metrics_single']     # Single fit metrics
```

---

## 📚 What You Get

### Predictions Comparison
```python
preds_cv = cv_results['y_cal_pred_cv']
preds_single = cv_results['y_cal_pred_single']

# Check stability
diff = preds_cv - preds_single
rmse = np.sqrt(np.mean(diff ** 2))
print(f"Prediction stability RMSE: {rmse:.4f}")
```

### Per-Fold Model Stability
```python
metrics_cv = cv_results['metrics_cv']

for i in range(len(metrics_cv)):
    fold_metrics = metrics_cv[i]
    for model_key, model_stats in fold_metrics.items():
        rmse = model_stats['metrics_cal'].get('RMSE')
        print(f"Fold {i}, {model_key}: RMSE={rmse:.4f}")
```

### Identify Unstable Models
```python
# Find models that vary most across folds
model_variation = {}

for model_key in single_metrics.keys():
    fold_rmses = []
    for i in range(len(metrics_cv)):
        fold_metrics = metrics_cv[i]
        if model_key in fold_metrics:
            rmse = fold_metrics[model_key]['metrics_cal'].get('RMSE')
            fold_rmses.append(rmse)
    
    if fold_rmses:
        cv_value = np.std(fold_rmses) / np.mean(fold_rmses)
        model_variation[model_key] = cv_value

# Most unstable
unstable = sorted(model_variation.items(), 
                 key=lambda x: x[1], reverse=True)
```

---

## 🔄 Analysis Patterns

### Pattern 1: Quick Stability Check
```python
rmse = cv_results['y_cal_pred_rmse']
if rmse < 0.1:
    print("✓ Calibration is stable")
else:
    print("✗ Calibration is unstable")
```

### Pattern 2: Per-Fold Stability
```python
per_fold = cv_results['y_cal_pred_rmse_per_fold']
best = np.min(per_fold)
worst = np.max(per_fold)
mean = np.mean(per_fold)
print(f"Best: {best:.4f}, Worst: {worst:.4f}, Mean: {mean:.4f}")
```

### Pattern 3: Sample Robustness
```python
preds_cv = cv_results['y_cal_pred_cv']
preds_single = cv_results['y_cal_pred_single']

per_sample_error = np.abs(preds_cv - preds_single)
fragile_samples = np.where(per_sample_error > threshold)[0]
print(f"Check samples: {fragile_samples}")
```

---

## 🎯 Interpretation Guide

| Metric | Good | Moderate | Poor |
|--------|------|----------|------|
| RMSE | < 0.1 | 0.1-0.5 | > 0.5 |
| Std across folds | Low | Medium | High |
| CV (std/mean) | < 0.1 | 0.1-0.3 | > 0.3 |

---

## 📚 Related Documentation

- **[../CV_SINGLE_FIT_REFERENCE_MODE/](../CV_SINGLE_FIT_REFERENCE_MODE/)** - Single-fit reference feature
- **[../CV_FOLD_OUTPUT_HANDLING/](../CV_FOLD_OUTPUT_HANDLING/)** - Fold output handling
- **[../CV_INTEGRATION_GUIDES/](../CV_INTEGRATION_GUIDES/)** - Integration and master index

---

## 📂 Implementation Files

- **Module**: `chemometrics/univ_calibration.py`
- **Test**: `tests/test_single_fit_reference.py` - PCA example (same patterns apply)

---

**Status**: ✅ Complete and ready to use
