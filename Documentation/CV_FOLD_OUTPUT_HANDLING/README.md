# CV Fold Output Handling Documentation

This folder contains documentation for **fold output handling** - how sample-based and non-sample-based outputs are reconstructed and segregated when running CV with single-fit reference mode.

## 📖 Files in This Folder

### Main Guide
- **INDEX_BASED_FOLD_ACCESS.md** ⭐ **START HERE**
  - Complete guide to `FoldSegregatedOutput` class
  - Index-based access patterns
  - Iteration and stacking examples
  - Real-world analysis patterns
  - (15 minute read)

### Design & Rationale
- **SEGREGATED_OUTPUTS_EXPLAINED.md**
  - Why segregation instead of averaging
  - Sample-based vs non-sample-based outputs
  - Design decisions and benefits
  - Analysis examples
  - (10 minute read)

### Implementation Details
- **INDEX_BASED_FOLD_ACCESS_UPDATE_SUMMARY.md**
  - Summary of changes made
  - FoldSegregatedOutput class details
  - File changes summary
  - (10 minute read)

---

## 🎯 Reading Guide

### "I want to use index-based access" (15 min)
1. Read: INDEX_BASED_FOLD_ACCESS.md
2. See examples section for your use case
3. Apply pattern to your code

### "I want to understand why" (10 min)
- Read: SEGREGATED_OUTPUTS_EXPLAINED.md

### "I want to see what changed" (10 min)
- Read: INDEX_BASED_FOLD_ACCESS_UPDATE_SUMMARY.md

---

## ✨ Key Concepts

### Sample-Based Outputs
Outputs where first dimension = number of samples (e.g., scores, predictions)

```python
# Reconstructed as full-size arrays
results['scores_cv']      # (100, 3) - all samples
results['scores_single']  # (100, 3) - reference
```

### Non-Sample-Based Outputs
Outputs where first dimension ≠ number of samples (e.g., loadings, coefficients)

```python
# Segregated by fold with index access
loadings_cv = results['loadings_cv']  # FoldSegregatedOutput
loadings_cv[0]  # Fold 0
loadings_cv[1]  # Fold 1
```

---

## 🚀 Quick Example: Index-Based Access

```python
# Scalable fold processing (works with any n_splits)
fold_outputs = results['loadings_cv']

for i in range(len(fold_outputs)):
    fold_data = fold_outputs[i]
    # Process fold data...

# Also works
for fold_data in fold_outputs:
    # Process...

# Stack into array
all_folds = fold_outputs.as_array()  # (5, 3, 20)
```

---

## 📚 FoldSegregatedOutput API

```python
# Index access
output[0]           # Get fold 0
output[1]           # Get fold 1

# Length
len(output)         # Get number of folds

# Iteration
for fold in output:
    # Process...

# Stack
output.as_array()   # Convert to array

# String keys (backward compatible)
output['fold_0']    # Same as output[0]
output['fold_1']    # Same as output[1]

# Get dictionary
output.as_dict()    # Get underlying fold dict
```

---

## 📊 Why This Design?

**Problem**: Different output types need different handling
- Predictions must be positioned by sample
- Parameters can't be positioned (they don't correspond to samples)

**Solution**: Automatic detection + smart output handling
- Sample-based → Reconstructed full-size arrays
- Non-sample-based → Segregated by fold (with index access)

**Benefit**: Scalable, efficient, no hardcoding needed

---

## 🔄 Analysis Patterns

### Pattern 1: Loop Through All Folds
```python
for i in range(len(outputs)):
    fold_output = outputs[i]
    # Works with any n_splits!
```

### Pattern 2: Compare To Reference
```python
single_output = results['loadings_single']
for i in range(len(outputs)):
    fold_output = outputs[i]
    diff = fold_output - single_output
```

### Pattern 3: Stack and Analyze
```python
stacked = outputs.as_array()  # (5, 3, 20)
variation = np.std(stacked, axis=0)
```

---

## 📚 Related Documentation

- **[../CV_SINGLE_FIT_REFERENCE_MODE/](../CV_SINGLE_FIT_REFERENCE_MODE/)** - Single-fit reference mode overview
- **[../UNIVARIATE_CALIBRATION/](../UNIVARIATE_CALIBRATION/)** - Univariate calibration examples
- **[../CV_INTEGRATION_GUIDES/](../CV_INTEGRATION_GUIDES/)** - Integration and master index

---

## 📂 Implementation Files

- **Core Class**: `chemometrics/cv_pipeline.py` - `FoldSegregatedOutput` class
- **Test**: `tests/test_single_fit_reference.py` - Working examples

---

**Status**: ✅ Complete and ready to use
