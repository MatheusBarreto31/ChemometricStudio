# CV Integration Guides Documentation

This folder contains **integration guides** and **complete implementation summaries** for all CV-related features.

## 📖 Files in This Folder

### Master Index
- **SINGLE_FIT_REFERENCE_WITH_INDEX_ACCESS_INDEX.md** ⭐ **START HERE**
  - Master index of all CV features
  - Navigation by use case
  - Quick start guides
  - File organization
  - (10 minute read)

### Complete Deliverable
- **IMPLEMENTATION_COMPLETE_DELIVERABLE.md**
  - What was accomplished
  - Files modified and created
  - Documentation organization
  - Backward compatibility notes
  - Ready for production checklist
  - (10 minute read)

---

## 🎯 Reading Guide

### "I'm lost - where do I start?" (10 min)
→ Read: SINGLE_FIT_REFERENCE_WITH_INDEX_ACCESS_INDEX.md

### "What exactly was built?" (10 min)
→ Read: IMPLEMENTATION_COMPLETE_DELIVERABLE.md

### "I need a specific feature" (5 min)
→ Check the master index for navigation

---

## 📚 All Available Features

### 1. Single-Fit Reference Mode
Run a single fit as reference, compare CV folds against it.

**See**: [../CV_SINGLE_FIT_REFERENCE_MODE/](../CV_SINGLE_FIT_REFERENCE_MODE/)

Quick example:
```python
results = pipeline.run(
    func, X=X,
    reference_output_key='scores',
    capture_output_keys=['scores', 'loadings']
)
```

### 2. Index-Based Fold Access
Access fold outputs by index for scalability.

**See**: [../CV_FOLD_OUTPUT_HANDLING/](../CV_FOLD_OUTPUT_HANDLING/)

Quick example:
```python
fold_outputs = results['loadings_cv']
for i in range(len(fold_outputs)):
    fold_data = fold_outputs[i]
```

### 3. Univariate Calibration
Use single-fit reference with polynomial calibration.

**See**: [../UNIVARIATE_CALIBRATION/](../UNIVARIATE_CALIBRATION/)

Quick example:
```python
results = univariate_calibration(
    X_cal, Y_cal,
    cv_config=config,
    reference_output_key='y_cal_pred',
    capture_output_keys=['y_cal_pred', 'metrics']
)
```

---

## ✨ What Was Delivered

✅ **Core Implementation**
- Single-fit reference mode in CVPipeline
- FoldSegregatedOutput class for index-based access
- Automatic sample-based vs non-sample-based detection

✅ **Applied Modules**
- Univariate calibration with new parameters
- Backward compatible with existing code

✅ **Documentation** (15 files organized in 4 folders)
- Quick reference guides
- Complete feature documentation
- Implementation details
- Analysis examples

✅ **Testing**
- Working test example
- All tests passing
- Verified functionality

✅ **Production Ready**
- Fully backward compatible
- Comprehensive documentation
- Ready for immediate use

---

## 📂 Documentation Structure

```
Documentation/
├── README.md                                    (Main index)
│
├── CV_SINGLE_FIT_REFERENCE_MODE/               (6 files)
│   ├── README.md
│   ├── SINGLE_FIT_REFERENCE_QUICK_REF.md       ⭐ Quick start
│   ├── SINGLE_FIT_REFERENCE_MODE.md            Complete reference
│   ├── SINGLE_FIT_REFERENCE_SUMMARY.md         How it works
│   ├── SINGLE_FIT_REFERENCE_COMPLETE.md        Implementation
│   └── SINGLE_FIT_REFERENCE_COMPLETE_FEATURE.md Feature overview
│
├── CV_FOLD_OUTPUT_HANDLING/                    (4 files)
│   ├── README.md
│   ├── INDEX_BASED_FOLD_ACCESS.md              ⭐ FoldSegregatedOutput
│   ├── SEGREGATED_OUTPUTS_EXPLAINED.md         Design rationale
│   └── INDEX_BASED_FOLD_ACCESS_UPDATE_SUMMARY.md Changes made
│
├── UNIVARIATE_CALIBRATION/                     (2 files)
│   ├── README.md
│   └── UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md ⭐ Usage
│
├── CV_INTEGRATION_GUIDES/                      (2 files - this folder)
│   ├── README.md
│   ├── SINGLE_FIT_REFERENCE_WITH_INDEX_ACCESS_INDEX.md Master index
│   └── IMPLEMENTATION_COMPLETE_DELIVERABLE.md  Summary
│
└── [Other existing docs]
```

---

## 🚀 Quick Start by Role

### Data Scientist / Analyst
1. Read: [../CV_SINGLE_FIT_REFERENCE_MODE/SINGLE_FIT_REFERENCE_QUICK_REF.md](../CV_SINGLE_FIT_REFERENCE_MODE/SINGLE_FIT_REFERENCE_QUICK_REF.md)
2. Run: `python tests/test_single_fit_reference.py`
3. Apply pattern to your data

### Python Developer
1. Read: [../CV_FOLD_OUTPUT_HANDLING/INDEX_BASED_FOLD_ACCESS.md](../CV_FOLD_OUTPUT_HANDLING/INDEX_BASED_FOLD_ACCESS.md)
2. Check: `chemometrics/cv_pipeline.py` - FoldSegregatedOutput class
3. Use in your modules

### Chemometrics Specialist
1. Read: [../UNIVARIATE_CALIBRATION/README.md](../UNIVARIATE_CALIBRATION/README.md)
2. See: [../UNIVARIATE_CALIBRATION/UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md](../UNIVARIATE_CALIBRATION/UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md)
3. Apply to calibration models

### Project Lead / Manager
1. Read: IMPLEMENTATION_COMPLETE_DELIVERABLE.md
2. Review: Feature summary and status
3. Check: Backward compatibility notes

---

## ✅ Feature Status

| Feature | Status | Documentation | Test |
|---------|--------|---|---|
| Single-Fit Reference Mode | ✅ Complete | 6 docs | ✅ Passing |
| Index-Based Fold Access | ✅ Complete | 4 docs | ✅ Passing |
| Univariate Calibration | ✅ Complete | 2 docs | ✅ Passing |
| Backward Compatibility | ✅ Full | Verified | ✅ Tested |

**Overall Status**: ✅ **PRODUCTION READY**

---

## 🔗 External References

- **Implementation**: `chemometrics/cv_pipeline.py`
- **Test Example**: `tests/test_single_fit_reference.py`
- **Module**: `chemometrics/univ_calibration.py`

---

## 💡 Tips

1. **Not sure where to start?** Read SINGLE_FIT_REFERENCE_WITH_INDEX_ACCESS_INDEX.md
2. **Want quick examples?** Check the QUICK_REF files
3. **Need complete details?** Read the MODE/SUMMARY files
4. **Have questions?** Check the relevant feature folder
5. **Want to contribute?** See the implementation files

---

**Last Updated**: February 3, 2026
**Status**: ✅ All features complete and documented
