# Single-Fit Reference Mode with Index-Based Fold Access - Complete Documentation Index

**Status**: ✅ **Complete and Fully Implemented**

This index organizes all documentation for the single-fit reference mode feature with scalable index-based fold access.

---

## Quick Navigation

### I Just Want to Use It (5 minutes)
1. **[SINGLE_FIT_REFERENCE_QUICK_REF.md](SINGLE_FIT_REFERENCE_QUICK_REF.md)** - Copy-paste examples
2. Run: `python tests/test_single_fit_reference.py` - See it working
3. Done! Apply pattern to your function

### I Want to Understand It (15 minutes)
1. **[SINGLE_FIT_REFERENCE_COMPLETE_FEATURE.md](SINGLE_FIT_REFERENCE_COMPLETE_FEATURE.md)** - Overview of what you asked for and what we built
2. **[INDEX_BASED_FOLD_ACCESS.md](INDEX_BASED_FOLD_ACCESS.md)** - How the `FoldSegregatedOutput` works

### I Want Everything (30 minutes)
1. **[SINGLE_FIT_REFERENCE_MODE.md](SINGLE_FIT_REFERENCE_MODE.md)** - Complete reference guide
2. **[INDEX_BASED_FOLD_ACCESS.md](INDEX_BASED_FOLD_ACCESS.md)** - Detailed fold access patterns
3. **[SEGREGATED_OUTPUTS_EXPLAINED.md](SEGREGATED_OUTPUTS_EXPLAINED.md)** - Why segregation instead of averaging
4. **[SINGLE_FIT_REFERENCE_SUMMARY.md](SINGLE_FIT_REFERENCE_SUMMARY.md)** - How it works internally

### For Univariate Calibration
**[UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md](UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md)** - Complete usage guide with examples

---

## Documentation Files

### Reference & Guides

| File | Purpose | Audience | Time |
|------|---------|----------|------|
| **SINGLE_FIT_REFERENCE_QUICK_REF.md** | Quick lookup, code snippets, patterns | All users | 5 min |
| **SINGLE_FIT_REFERENCE_COMPLETE_FEATURE.md** | Your request + what we built | Getting started | 10 min |
| **SINGLE_FIT_REFERENCE_MODE.md** | Complete technical reference | Implementers | 20 min |
| **SINGLE_FIT_REFERENCE_SUMMARY.md** | How it works internally | Developers | 15 min |
| **INDEX_BASED_FOLD_ACCESS.md** | FoldSegregatedOutput usage | All users | 10 min |
| **SEGREGATED_OUTPUTS_EXPLAINED.md** | Why segregation vs averaging | Understanding | 10 min |
| **UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md** | Univariate module guide | Chemometrics users | 15 min |

### Implementation Info

| File | Purpose | Audience | Time |
|------|---------|----------|------|
| **SINGLE_FIT_REFERENCE_INDEX.md** | Original feature index | Reference | 5 min |
| **SINGLE_FIT_REFERENCE_COMPLETE.md** | Original implementation summary | Reference | 10 min |
| **INDEX_BASED_FOLD_ACCESS_UPDATE_SUMMARY.md** | Changes & updates made | Developers | 10 min |

---

## Feature Overview

### What You Asked For
> "Run a function's single fit (on all data) to get outputs. Being able to use one of those outputs as the y_test reference for CV evaluation. The CV function will provide the statistics and also the output capture."

### What We Built ✅

1. **Single-Fit Reference Mode**
   - Run function once on all data (fold=-1)
   - Run CV folds normally (fold=0-4)
   - Compare fold outputs to single-fit reference
   - Get metrics showing differences

2. **Sample-Based Output Reconstruction**
   - Scores, predictions, embeddings → Full-size arrays
   - Test positions preserved across folds
   - Direct comparison possible

3. **Non-Sample-Based Output Segregation** (NEW)
   - Loadings, coefficients, components → Fold-specific versions
   - Index-based access: `output[0]`, `output[1]`, etc.
   - Scalable without hardcoding

4. **Applied to Univariate Calibration**
   - `univariate_calibration()` now supports single-fit reference mode
   - Same API: `reference_output_key`, `capture_output_keys`
   - Get per-fold calibration stability

---

## Core Components

### CV Pipeline
**File**: `chemometrics/cv_pipeline.py`

```python
from chemometrics.cv_pipeline import CVPipeline, FoldSegregatedOutput

# FoldSegregatedOutput enables:
# - output[0], output[1], etc. (index access)
# - len(output) (get fold count)
# - for fold in output: (iteration)
# - output.as_array() (stack into array)
# - output['fold_0'] (string keys still work)
```

### Univariate Calibration
**File**: `chemometrics/univ_calibration.py`

```python
from chemometrics.univ_calibration import univariate_calibration

# Now supports:
results = univariate_calibration(
    X_cal, Y_cal,
    cv_config=config,
    reference_output_key='y_cal_pred',
    capture_output_keys=['y_cal_pred', 'metrics']
)
```

### Test Example
**File**: `tests/test_single_fit_reference.py`

- Complete working example with PCA
- Demonstrates all features
- Shows index-based access in action
- Run: `python tests/test_single_fit_reference.py`

---

## Output Structure

### Sample-Based Outputs (Reconstructed)
```python
results['scores_cv']      # Full-size array, all samples
results['scores_single']  # Reference from single fit

# Direct access and comparison
diff = results['scores_cv'] - results['scores_single']
```

### Non-Sample-Based Outputs (Segregated)
```python
loadings_cv = results['loadings_cv']  # FoldSegregatedOutput

# Index access (scalable!)
fold_0 = loadings_cv[0]
fold_1 = loadings_cv[1]

# Loop (works with any n_splits)
for i in range(len(loadings_cv)):
    fold_data = loadings_cv[i]

# Stack if needed
all_folds_array = loadings_cv.as_array()
```

### Metrics
```python
results['scores_rmse']           # Overall: how different?
results['scores_rmse_per_fold']  # Per-fold: [0.69, 1.53, ...]
results['scores_rmse_std']       # Variability across folds
```

---

## Usage Patterns

### Pattern 1: Quick Stability Check
```python
results = pipeline.run(func, X=X, reference_output_key='scores')
print(f"Stability RMSE: {results['scores_rmse']:.4f}")
```

### Pattern 2: Per-Fold Analysis
```python
for rmse in results['scores_rmse_per_fold']:
    print(f"Fold RMSE: {rmse:.4f}")
```

### Pattern 3: Scalable Fold Processing
```python
fold_outputs = results['loadings_cv']
for i in range(len(fold_outputs)):
    fold_data = fold_outputs[i]
    # Works with any number of folds
```

### Pattern 4: Stack and Analyze
```python
loadings_cv = results['loadings_cv']
all_folds = loadings_cv.as_array()  # (5, 3, 20)
param_std = np.std(all_folds, axis=0)  # Variation per parameter
```

---

## Key Features

✅ **Flexible**: Use any function output as reference  
✅ **Smart**: Automatic sample-based vs non-sample detection  
✅ **Scalable**: Index-based access works with any n_splits  
✅ **Complete**: Metrics + full/segregated outputs  
✅ **Applied**: Works with univariate calibration  
✅ **Compatible**: Fully backward compatible  
✅ **Tested**: Comprehensive working example included  

---

## File Map

```
chemometrics/
├── cv_pipeline.py          ← Core implementation + FoldSegregatedOutput
└── univ_calibration.py     ← Enhanced with single-fit reference support

tests/
└── test_single_fit_reference.py  ← Working example

Generated Notes/
├── SINGLE_FIT_REFERENCE_QUICK_REF.md           ← Start here for quick use
├── SINGLE_FIT_REFERENCE_COMPLETE_FEATURE.md    ← Overview of feature
├── SINGLE_FIT_REFERENCE_MODE.md                ← Complete reference
├── SINGLE_FIT_REFERENCE_SUMMARY.md             ← How it works internally
├── INDEX_BASED_FOLD_ACCESS.md                  ← FoldSegregatedOutput guide
├── SEGREGATED_OUTPUTS_EXPLAINED.md             ← Design rationale
├── UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md  ← Univariate examples
├── SINGLE_FIT_REFERENCE_INDEX.md               ← Original index
├── SINGLE_FIT_REFERENCE_COMPLETE.md            ← Original summary
└── INDEX_BASED_FOLD_ACCESS_UPDATE_SUMMARY.md   ← What changed
```

---

## Quick Start Commands

```bash
# See it working
python tests/test_single_fit_reference.py

# Read quick reference
cat "Generated Notes/SINGLE_FIT_REFERENCE_QUICK_REF.md"

# Understand the feature
cat "Generated Notes/SINGLE_FIT_REFERENCE_COMPLETE_FEATURE.md"

# Learn index access
cat "Generated Notes/INDEX_BASED_FOLD_ACCESS.md"
```

---

## Implementation Details

### FoldSegregatedOutput Class
- Location: `chemometrics/cv_pipeline.py`
- Supports integer indexing, iteration, stacking
- Backward compatible with string key access
- Provides length via `len()`

### Reconstruction Logic
- Sample-based: Positions in original indices
- Non-sample-based: Returns segregated dict wrapped in FoldSegregatedOutput

### Metrics Computation
- Per-fold: RMSE between fold output and reference[test_indices]
- Overall: Mean of per-fold metrics
- Std: Standard deviation across folds

---

## Support

**Q: What do I need to do to use this?**
A: Read `SINGLE_FIT_REFERENCE_QUICK_REF.md` and copy the pattern to your function.

**Q: How do I loop through variable numbers of folds?**
A: Use index-based access: `for i in range(len(output)):` - See `INDEX_BASED_FOLD_ACCESS.md`

**Q: Why is my non-sample output segregated instead of averaged?**
A: See `SEGREGATED_OUTPUTS_EXPLAINED.md` - It preserves fold-specific information.

**Q: Can I use this with univariate calibration?**
A: Yes! See `UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md` for examples.

---

## Status: Production Ready ✅

- ✅ Core implementation complete
- ✅ Index-based access implemented
- ✅ Applied to univariate calibration
- ✅ Comprehensive documentation (9 files)
- ✅ Working test example
- ✅ Backward compatible
- ✅ Tested and verified

**Ready to use!**
