# Univariate Calibration with Single-Fit Reference Mode

## Overview

The univariate calibration module now supports **single-fit reference mode**, allowing you to:
1. Run a single calibration on all data to get reference predictions
2. Run CV folds independently
3. Compare fold predictions to the reference to assess model stability

This is useful for understanding how calibration coefficients and predictions vary when trained on different data subsets.

---

## Usage

### Enable Single-Fit Reference Mode

```python
from chemometrics.cv_pipeline import CVConfig
from chemometrics.univ_calibration import univariate_calibration

# Configure CV with 5 folds
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    random_state=42
)

# Run with single-fit reference
results = univariate_calibration(
    X_cal, Y_cal,
    X_val=X_val,  # optional validation set
    Y_val=Y_val,
    degree=2,
    intercept=True,
    cv_config=cv_config,
    reference_output_key='y_cal_pred',      # Use calibration predictions as reference
    capture_output_keys=['y_cal_pred', 'metrics']  # Also capture metrics
)
```

### What You Get Back

```python
# Single fit results (on all data)
results['y_cal_pred']  # Predictions from all calibration data
results['models']      # Full model details (coefficients, stats, etc.)
results['metrics']     # Metrics from single fit

# CV comparison results
cv_results = results['cv_results']

# Metrics comparing folds to single fit
cv_results['y_cal_pred_rmse']           # Overall RMSE difference
cv_results['y_cal_pred_rmse_per_fold']  # Per-fold metrics: [0.45, 0.52, 0.48, ...]
cv_results['y_cal_pred_rmse_std']       # Variability across folds

# Reconstructed outputs from folds
cv_results['y_cal_pred_cv']      # Full-size predictions reconstructed from folds
cv_results['y_cal_pred_single']  # Reference predictions from single fit

# Captured metric outputs (segregated by fold)
cv_results['metrics_cv']         # FoldSegregatedOutput with metrics per fold
cv_results['metrics_single']     # Metrics from single fit
```

---

## Results Structure

### Sample-Based Outputs (Reconstructed)

**`y_cal_pred_cv`** and **`y_cal_pred_single`** contain full-size predictions:

```python
cv_preds = cv_results['y_cal_pred_cv']      # shape (100,) or (100, 5)
single_preds = cv_results['y_cal_pred_single']  # same shape

# Direct comparison
diff = cv_preds - single_preds
stability = np.std(diff)
```

### Non-Sample-Based Outputs (Segregated)

**`metrics_cv`** is a `FoldSegregatedOutput` with **index-based access**:

```python
metrics_cv = cv_results['metrics_cv']

# Access by index (scalable!)
fold_0_metrics = metrics_cv[0]
fold_1_metrics = metrics_cv[1]

# Loop through all folds
for i in range(len(metrics_cv)):
    fold_metrics = metrics_cv[i]
    print(f"Fold {i}: RMSE = {fold_metrics[...]['RMSE']}")

# Or iterate
for fold_metrics in metrics_cv:
    # Process each fold...
    pass

# Stack into single array if needed
all_folds = metrics_cv.as_array()
```

---

## Analysis Examples

### 1. Assess Prediction Stability

```python
# How much do fold predictions vary from reference?
cv_preds = cv_results['y_cal_pred_cv']
single_preds = cv_results['y_cal_pred_single']

# Per-sample variation
diff = cv_preds - single_preds
rmse_per_sample = np.sqrt(diff ** 2)
max_error = np.max(rmse_per_sample)
mean_error = np.mean(rmse_per_sample)

print(f"Max prediction difference: {max_error:.4f}")
print(f"Mean prediction difference: {mean_error:.4f}")

# Overall metric from CV
overall_rmse = cv_results['y_cal_pred_rmse']
print(f"Overall RMSE: {overall_rmse:.4f}")
```

### 2. Compare Per-Fold Performance

```python
per_fold = cv_results['y_cal_pred_rmse_per_fold']

# Which folds are most/least stable?
best_fold = np.argmin(per_fold)
worst_fold = np.argmax(per_fold)

print(f"Best fold: {best_fold} (RMSE={per_fold[best_fold]:.4f})")
print(f"Worst fold: {worst_fold} (RMSE={per_fold[worst_fold]:.4f})")
print(f"Mean: {np.mean(per_fold):.4f}")
print(f"Std: {np.std(per_fold):.4f}")
```

### 3. Analyze Metrics Per Fold

```python
metrics_cv = cv_results['metrics_cv']

# Scalable loop (no hardcoding fold names!)
for fold_idx in range(len(metrics_cv)):
    fold_metrics = metrics_cv[fold_idx]
    
    # Each fold_metrics is a dict like:
    # {'X0_Y0': {'metrics_cal': {...}, 'metrics_val': {...}}, ...}
    
    for model_key, model_metrics in fold_metrics.items():
        cal_rmse = model_metrics['metrics_cal'].get('RMSE')
        val_rmse = model_metrics['metrics_val'].get('RMSE') if model_metrics['metrics_val'] else None
        print(f"Fold {fold_idx}, {model_key}: RMSE_cal={cal_rmse:.4f}, RMSE_val={val_rmse}")
```

### 4. Identify Unstable Models

```python
# Get metrics from all folds
metrics_cv = cv_results['metrics_cv']
single_metrics = cv_results['metrics_single']

# Compute per-fold variation
model_variation = {}

for model_key in single_metrics.keys():
    fold_rmses = []
    for i in range(len(metrics_cv)):
        fold_metrics = metrics_cv[i]
        if model_key in fold_metrics:
            rmse = fold_metrics[model_key]['metrics_cal'].get('RMSE')
            if rmse is not None:
                fold_rmses.append(rmse)
    
    if fold_rmses:
        model_variation[model_key] = {
            'mean': np.mean(fold_rmses),
            'std': np.std(fold_rmses),
            'cv': np.std(fold_rmses) / np.mean(fold_rmses)  # Coefficient of variation
        }

# Find most unstable models
unstable = sorted(model_variation.items(), 
                 key=lambda x: x[1]['cv'], 
                 reverse=True)

print("Most unstable models:")
for model_key, stats in unstable[:5]:
    print(f"  {model_key}: CV={stats['cv']:.2%}")
```

---

## Interpreting Results

### What Does Per-Fold RMSE Mean?

`y_cal_pred_rmse_per_fold` = [0.45, 0.52, 0.48, 0.55, 0.50]

- **Overall**: Average of these values ≈ 0.50
- **Per-fold**: Each fold's predictions differed from single fit reference by this much
- **High variation** (e.g., 0.45 vs 0.55): Model is unstable - predictions change significantly based on training data
- **Low variation** (e.g., 0.49, 0.50, 0.51): Model is stable - consistent predictions regardless of training data

### Sample-Based vs Non-Sample-Based

**Sample-Based** (like predictions):
- Full-size reconstructed arrays
- Same samples in same positions
- Compare directly: `cv_preds - single_preds`

**Non-Sample-Based** (like metrics, coefficients):
- Segregated by fold with index access
- Different shape per fold (e.g., metrics are dictionaries)
- Iterate through: `for i in range(len(metrics_cv)): fold_data = metrics_cv[i]`

---

## Advanced: Custom Analysis with Index-Based Access

```python
# Get all fold data
metrics_cv = cv_results['metrics_cv']
n_folds = len(metrics_cv)

# Initialize storage
rmse_matrix = []  # (n_folds, n_models)
models_list = None

# Populate matrix (no hardcoding!)
for fold_idx in range(n_folds):
    fold_metrics = metrics_cv[fold_idx]
    
    if models_list is None:
        models_list = list(fold_metrics.keys())
    
    fold_rmses = []
    for model_key in models_list:
        if model_key in fold_metrics:
            rmse = fold_metrics[model_key]['metrics_cal'].get('RMSE', np.nan)
            fold_rmses.append(rmse)
        else:
            fold_rmses.append(np.nan)
    
    rmse_matrix.append(fold_rmses)

rmse_matrix = np.array(rmse_matrix)  # (n_folds, n_models)

# Analysis
mean_rmse = np.mean(rmse_matrix, axis=0)
std_rmse = np.std(rmse_matrix, axis=0)

print("Model stability:")
for model_key, mean, std in zip(models_list, mean_rmse, std_rmse):
    print(f"  {model_key}: {mean:.4f} ± {std:.4f}")
```

---

## Comparison: Single Fit vs CV Reference

### Traditional CV
```python
results = univariate_calibration(X_cal, Y_cal, cv_config=config)
# Get only: per-fold metrics
# Can't compare folds to single fit
```

### CV with Single-Fit Reference ✨
```python
results = univariate_calibration(
    X_cal, Y_cal, 
    cv_config=config,
    reference_output_key='y_cal_pred',
    capture_output_keys=['y_cal_pred', 'metrics']
)
# Get: per-fold metrics + comparison to single fit
# See how model changes across training subsets
# Identify unstable coefficients or predictions
```

---

## See Also

- [INDEX_BASED_FOLD_ACCESS.md](INDEX_BASED_FOLD_ACCESS.md) - Full guide to `FoldSegregatedOutput`
- [SINGLE_FIT_REFERENCE_MODE.md](SINGLE_FIT_REFERENCE_MODE.md) - Complete CV reference mode guide
- [SEGREGATED_OUTPUTS_EXPLAINED.md](SEGREGATED_OUTPUTS_EXPLAINED.md) - Why non-sample outputs are segregated
