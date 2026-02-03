# Single-Fit Reference Mode - Documentation Index

**Status**: ✅ **Complete and Ready**

This feature enables you to run a function's single fit and use its outputs as a reference for CV evaluation, allowing assessment of model stability across different training subsets.

---

## Start Here 👇

### Quick Start (5 minutes)
📖 **[SINGLE_FIT_REFERENCE_QUICK_REF.md](SINGLE_FIT_REFERENCE_QUICK_REF.md)**
- Copy-paste code examples
- Parameter reference
- Common patterns
- Quick interpretation guide

### Complete Understanding (15 minutes)
📖 **[SINGLE_FIT_REFERENCE_SUMMARY.md](SINGLE_FIT_REFERENCE_SUMMARY.md)**
- Your use case explained
- How it works internally
- Results you get
- Examples

### Full Reference (30 minutes)
📖 **[SINGLE_FIT_REFERENCE_MODE.md](SINGLE_FIT_REFERENCE_MODE.md)**
- Comprehensive guide
- Advanced usage
- All parameters explained
- Multiple output types
- Use cases and examples

### Implementation Details (20 minutes)
📖 **[SINGLE_FIT_REFERENCE_COMPLETE.md](SINGLE_FIT_REFERENCE_COMPLETE.md)**
- What was built
- Design decisions
- Technical deep dive
- Test results

### Working Example
💻 **[tests/test_single_fit_reference.py](../tests/test_single_fit_reference.py)**
- Run: `python tests/test_single_fit_reference.py`
- Complete working example
- PCA stability assessment
- Shows all features

---

## Reading Guide

### By Your Need

**"I want to use this feature right now"**
1. Read: [SINGLE_FIT_REFERENCE_QUICK_REF.md](SINGLE_FIT_REFERENCE_QUICK_REF.md)
2. Run: `python tests/test_single_fit_reference.py`
3. Copy code pattern to your function

**"I want to understand what you built"**
1. Read: [SINGLE_FIT_REFERENCE_SUMMARY.md](SINGLE_FIT_REFERENCE_SUMMARY.md)
2. Skim: [SINGLE_FIT_REFERENCE_COMPLETE.md](SINGLE_FIT_REFERENCE_COMPLETE.md)
3. Run: `python tests/test_single_fit_reference.py`

**"I want comprehensive reference"**
1. Read: [SINGLE_FIT_REFERENCE_MODE.md](SINGLE_FIT_REFERENCE_MODE.md)
2. Ref: [SINGLE_FIT_REFERENCE_QUICK_REF.md](SINGLE_FIT_REFERENCE_QUICK_REF.md)
3. Study: [SINGLE_FIT_REFERENCE_COMPLETE.md](SINGLE_FIT_REFERENCE_COMPLETE.md)

**"I want technical details"**
1. Read: [SINGLE_FIT_REFERENCE_COMPLETE.md](SINGLE_FIT_REFERENCE_COMPLETE.md)
2. Study: [chemometrics/cv_pipeline.py](../chemometrics/cv_pipeline.py)
3. Trace: `tests/test_single_fit_reference.py` execution

---

## Quick Reference

### Enable the Feature
```python
results = pipeline.run(
    func,
    X=X,
    reference_output_key='scores',
    capture_output_keys=['scores', 'loadings']
)
```

### What You Get
```python
results['scores_rmse']            # Overall metric
results['scores_rmse_per_fold']   # Per-fold metrics
results['scores_cv']              # Reconstructed from folds
results['scores_single']          # Reference from single fit
```

### Function Requirements
```python
def func(X_train, X_test, fold=-1):
    # fold=-1: Single fit on all data
    # fold=0-4: CV folds
    return {'scores': array, 'loadings': array}
```

---

## Feature Overview

| Aspect | Details |
|--------|---------|
| **What it does** | Runs single fit + CV, compares fold outputs to single-fit reference |
| **Primary use** | Assess model stability across different training subsets |
| **Parameters** | `reference_output_key`, `capture_output_keys` |
| **Metrics computed** | RMSE between fold and reference outputs |
| **Results included** | Per-fold metrics, reconstructed full-size arrays, reference outputs |
| **Backward compatible** | Yes (parameters optional) |
| **Multiway data** | Yes (automatic handling) |

---

## Documentation Files

| File | Purpose | Length | Audience |
|------|---------|--------|----------|
| **SINGLE_FIT_REFERENCE_QUICK_REF.md** | Quick lookup, code snippets | 1 page | Users |
| **SINGLE_FIT_REFERENCE_SUMMARY.md** | Overview, how it works, examples | 3 pages | Everyone |
| **SINGLE_FIT_REFERENCE_MODE.md** | Complete reference guide | 5 pages | Implementers |
| **SINGLE_FIT_REFERENCE_COMPLETE.md** | Implementation details, design | 4 pages | Engineers |
| **test_single_fit_reference.py** | Working code example | - | Everyone |

---

## Feature Highlights

✅ **Flexible reference**: Use any function output as reference  
✅ **Automatic reconstruction**: Full-size arrays from fold tests  
✅ **Per-fold metrics**: See performance variation across folds  
✅ **Multiple outputs**: Capture different metrics simultaneously  
✅ **Smart handling**: Sample-based vs non-sample reconstruction  
✅ **Backward compatible**: Existing CV code unaffected  
✅ **Well tested**: Comprehensive test suite included  

---

## Common Use Cases

### 1. PCA Stability Assessment
```python
results = pipeline.run(
    pca_func,
    X=X,
    reference_output_key='scores'
)
# See how PCA scores vary when trained on different data
```

### 2. Model Robustness Testing
```python
results = pipeline.run(
    calibration_func,
    X=X, Y=Y,
    reference_output_key='coefficients'
)
# See if calibration coefficients change significantly
```

### 3. Feature Extraction Validation
```python
results = pipeline.run(
    embedding_func,
    X=X,
    reference_output_key='embeddings',
    capture_output_keys=['embeddings', 'weights']
)
# Verify embeddings are stable and meaningful
```

---

## Implementation Status

| Item | Status |
|------|--------|
| Core implementation | ✅ Complete |
| Testing | ✅ Passing |
| Documentation | ✅ Comprehensive |
| Examples | ✅ Provided |
| Backward compatibility | ✅ Verified |
| Ready to use | ✅ Yes |

---

## Key Design Decisions

1. **Two separate parameters**:
   - `reference_output_key`: Which output to use for metrics (required for this mode)
   - `capture_output_keys`: Which outputs to preserve (optional, can include more than reference)

2. **Automatic reconstruction**:
   - Sample-based outputs (scores): Placed in original positions → Full-size array
   - Non-sample outputs (loadings): Averaged across folds → Single representative array

3. **Backward compatible**:
   - Both parameters optional
   - If not provided, uses traditional CV mode
   - No breaking changes to existing code

4. **Metric computation**:
   - Per-fold: RMSE(fold_output, reference_output[test_idx])
   - Overall: Mean of per-fold RMSEs
   - Reflects model variability across training subsets

---

## Next Steps

1. **Explore**: Read [SINGLE_FIT_REFERENCE_QUICK_REF.md](SINGLE_FIT_REFERENCE_QUICK_REF.md)
2. **Test**: Run `python tests/test_single_fit_reference.py`
3. **Learn**: Read [SINGLE_FIT_REFERENCE_MODE.md](SINGLE_FIT_REFERENCE_MODE.md)
4. **Apply**: Use in your project

---

## File Locations

- **Implementation**: `chemometrics/cv_pipeline.py` (modified)
- **Test**: `tests/test_single_fit_reference.py` (new)
- **Documentation**: `Generated Notes/SINGLE_FIT_REFERENCE_*.md` (new)

---

**Status**: ✅ Complete and ready to use!

Need help? Start with [SINGLE_FIT_REFERENCE_QUICK_REF.md](SINGLE_FIT_REFERENCE_QUICK_REF.md)
