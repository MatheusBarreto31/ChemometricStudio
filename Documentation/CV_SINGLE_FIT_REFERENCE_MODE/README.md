# CV Single-Fit Reference Mode Documentation

This folder contains all documentation for the **single-fit reference mode** feature - the ability to run a function's single fit as a reference and compare CV fold outputs against it.

## 📖 Files in This Folder

### Quick Start
- **SINGLE_FIT_REFERENCE_QUICK_REF.md** ⭐ **START HERE**
  - Copy-paste code examples
  - Quick reference for parameters
  - Common analysis patterns
  - (5 minute read)

### Complete Understanding
- **SINGLE_FIT_REFERENCE_COMPLETE_FEATURE.md**
  - Your original request + what was built
  - Complete feature overview
  - Real-world examples
  - (10 minute read)

- **SINGLE_FIT_REFERENCE_MODE.md**
  - Full technical reference guide
  - All parameters explained
  - Use cases and advanced usage
  - (20 minute read)

### Technical Details
- **SINGLE_FIT_REFERENCE_SUMMARY.md**
  - How it works internally
  - Architecture and flow
  - Design decisions
  - (15 minute read)

- **SINGLE_FIT_REFERENCE_COMPLETE.md**
  - Implementation summary
  - Technical deep dive
  - (10 minute read)

### Reference
- **SINGLE_FIT_REFERENCE_INDEX.md**
  - Original feature index
  - Navigation guide

---

## 🎯 Reading Guide

### "I just want to use it" (5 min)
1. Read: SINGLE_FIT_REFERENCE_QUICK_REF.md
2. Run: `python tests/test_single_fit_reference.py`
3. Copy pattern to your function

### "I want to understand it" (20 min)
1. Read: SINGLE_FIT_REFERENCE_COMPLETE_FEATURE.md
2. Read: SINGLE_FIT_REFERENCE_MODE.md
3. Check: SINGLE_FIT_REFERENCE_SUMMARY.md

### "I want complete reference" (30 min)
1. Read: SINGLE_FIT_REFERENCE_MODE.md
2. Read: SINGLE_FIT_REFERENCE_SUMMARY.md
3. Skim: SINGLE_FIT_REFERENCE_COMPLETE.md

---

## ✨ Feature Highlights

✅ **Flexible**: Use any function output as reference  
✅ **Sample-based**: Full-size array reconstruction  
✅ **Non-sample-based**: Segregated fold outputs  
✅ **Scalable**: Index-based fold access  
✅ **Compatible**: Fully backward compatible  
✅ **Tested**: Working examples included  

---

## 🚀 Quick Example

```python
from chemometrics.cv_pipeline import CVPipeline

results = pipeline.run(
    pca_function,
    X=X,
    reference_output_key='scores',
    capture_output_keys=['scores', 'loadings']
)

# Results
print(f"Stability: {results['scores_rmse']:.4f}")
print(f"Per-fold: {results['scores_rmse_per_fold']}")

# Sample-based: full array
scores_cv = results['scores_cv']  # (100, 3)

# Non-sample-based: index access
loadings_cv = results['loadings_cv']
for i in range(len(loadings_cv)):
    fold_loadings = loadings_cv[i]
```

---

## 📚 Related Documentation

- **[../CV_FOLD_OUTPUT_HANDLING/](../CV_FOLD_OUTPUT_HANDLING/)** - How fold outputs are handled
- **[../UNIVARIATE_CALIBRATION/](../UNIVARIATE_CALIBRATION/)** - Univariate calibration with single-fit reference
- **[../CV_INTEGRATION_GUIDES/](../CV_INTEGRATION_GUIDES/)** - Integration guides and master index

---

## 📂 Implementation Files

- **Core**: `chemometrics/cv_pipeline.py`
- **Test**: `tests/test_single_fit_reference.py`

---

**Status**: ✅ Complete and ready to use
