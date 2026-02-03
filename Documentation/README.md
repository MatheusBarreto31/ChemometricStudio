# Documentation Index

Welcome to CM-Studio documentation. Start with the feature that interests you.

---

## 🔄 Cross-Validation (CV)

**[CV_PIPELINE.md](CV_PIPELINE.md)** ⭐ **START HERE FOR CV**

Complete CV pipeline documentation covering:
- Quick start (3 steps to integrate your model)
- Reference modes (input and output)
- All 9 CV strategies
- Multiway data support
- Advanced usage

### Quick Example
```python
from chemometrics.cv_pipeline import cv_configuration, CVPipeline

cv_cfg = cv_configuration(use_cv=True, cv_strategy='kfold', n_splits=5)
pipeline = CVPipeline(cv_cfg['cv_config'])
results = pipeline.run(my_model, X=X, Y=Y, 
                      reference_input_key='Y',
                      comparison_output_key='y_pred')
```

### Subfolders

#### [CV_INTEGRATION_GUIDES/](CV_INTEGRATION_GUIDES/)
How to integrate your models with CV pipeline.

#### [CV_FOLD_OUTPUT_HANDLING/](CV_FOLD_OUTPUT_HANDLING/)
How reconstructed outputs are organized (sample-based vs non-sample-based).

#### [CV_SINGLE_FIT_REFERENCE_MODE/](CV_SINGLE_FIT_REFERENCE_MODE/)
Using single-fit outputs as reference for stability assessment.

---

## 📊 Analysis & Calibration

#### [UNIVARIATE_CALIBRATION/](UNIVARIATE_CALIBRATION/)
Univariate calibration module (polynomial fitting with CV support).

### 💻 User Interface

#### [GUI/](GUI/)
Graphical User Interface documentation for CM-Studio.
- GUI feature documentation
- Quick start guide for new users

### 📈 Data Handling

#### [DATA_HANDLING_AND_SLICING/](DATA_HANDLING_AND_SLICING/)
Multi-dimensional data manipulation and table viewing.
- *README.md* - Overview and quick navigation
- *MULTI_DIMENSIONAL_SLICING_GUIDE.md* - Complete guide for multi-dimensional data
- *MULTI_DIMENSIONAL_SLICING_QUICK_REF.md* - Quick reference for slicing operations
- *TABLE_DATA_SLICING_QUICK_REFERENCE.md* - Table data manipulation reference
- *TABLE_VIEWING_IMPLEMENTATION.md* - Table viewing feature details
- *MULTI_DIMENSIONAL_SLICING_IMPLEMENTATION.md* - Implementation details
- *example_4d_slicing_config.json* - Example configuration for 4D slicing

### ⚙️ Configuration & Setup

#### [CONFIGURATION_AND_SETUP/](CONFIGURATION_AND_SETUP/)
System configuration, file structure, and development reference.
- *README.md* - Overview and quick navigation
- *ANALYSIS_CONFIGURATION.md* - Analysis configuration guide
- *ANALYSIS_CONFIG_QUICK_REF.md* - Quick reference for config parameters
- *ANALYSIS_JSON_SCHEMA.md* - JSON schema documentation
- *FILE_STRUCTURE_GUIDE.md* - Project file structure and organization
- *FUNCTION_DEVELOPMENT_REFERENCE.md* - Guide for developing and extending functions

### 🗺️ Navigation & Routing

#### [NAVIGATION_AND_ROUTING/](NAVIGATION_AND_ROUTING/)
Routing system and navigation controls documentation.
- *README.md* - Overview and quick navigation
- *README_ROUTING.md* - Complete routing system guide
- *ROUTING_VISUAL_GUIDE.md* - Visual diagrams and routing concepts
- *NAVIGATION_QUICK_REF.md* - Quick reference for navigation controls
- **[FUNCTION_DEVELOPMENT_REFERENCE.md](FUNCTION_DEVELOPMENT_REFERENCE.md)** - Function development guide

### Navigation & Routing

- **[README_ROUTING.md](README_ROUTING.md)** - Routing documentation
- **[ROUTING_VISUAL_GUIDE.md](ROUTING_VISUAL_GUIDE.md)** - Visual guide for routing

---

## 🚀 Quick Start by Feature

### I want to...

**Assess model stability across CV folds**
→ Start with [CV_SINGLE_FIT_REFERENCE_MODE/QUICK_REF.md](CV_SINGLE_FIT_REFERENCE_MODE/SINGLE_FIT_REFERENCE_QUICK_REF.md)

**Use index-based access for fold outputs**
→ See [CV_FOLD_OUTPUT_HANDLING/INDEX_BASED_FOLD_ACCESS.md](CV_FOLD_OUTPUT_HANDLING/INDEX_BASED_FOLD_ACCESS.md)

**Understand output reconstruction and segregation**
→ Read [CV_FOLD_OUTPUT_HANDLING/SEGREGATED_OUTPUTS_EXPLAINED.md](CV_FOLD_OUTPUT_HANDLING/SEGREGATED_OUTPUTS_EXPLAINED.md)

**Use single-fit reference with univariate calibration**
→ See [UNIVARIATE_CALIBRATION/UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md](UNIVARIATE_CALIBRATION/UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md)

**Get complete reference for single-fit reference mode**
→ Read [CV_SINGLE_FIT_REFERENCE_MODE/SINGLE_FIT_REFERENCE_MODE.md](CV_SINGLE_FIT_REFERENCE_MODE/SINGLE_FIT_REFERENCE_MODE.md)

**See master index of all new features**
→ Check [CV_INTEGRATION_GUIDES/SINGLE_FIT_REFERENCE_WITH_INDEX_ACCESS_INDEX.md](CV_INTEGRATION_GUIDES/SINGLE_FIT_REFERENCE_WITH_INDEX_ACCESS_INDEX.md)

---

## 📋 File Organization

```
Documentation/
├── README.md (this file)
│
├── GUI_DOCUMENTATION.md
├── GUI_QUICKSTART.md
│
├── CV_SINGLE_FIT_REFERENCE_MODE/
│   ├── SINGLE_FIT_REFERENCE_QUICK_REF.md          (Start here!)
│   ├── SINGLE_FIT_REFERENCE_MODE.md               (Complete reference)
│   ├── SINGLE_FIT_REFERENCE_SUMMARY.md            (How it works)
│   ├── SINGLE_FIT_REFERENCE_COMPLETE.md           (Implementation)
│   ├── SINGLE_FIT_REFERENCE_COMPLETE_FEATURE.md   (Feature overview)
│   └── SINGLE_FIT_REFERENCE_INDEX.md              (Original index)
│
├── CV_FOLD_OUTPUT_HANDLING/
│   ├── INDEX_BASED_FOLD_ACCESS.md                 (FoldSegregatedOutput guide)
│   ├── SEGREGATED_OUTPUTS_EXPLAINED.md            (Design rationale)
│   └── INDEX_BASED_FOLD_ACCESS_UPDATE_SUMMARY.md  (What changed)
│
├── UNIVARIATE_CALIBRATION/
│   └── UNIVARIATE_CALIBRATION_SINGLE_FIT_REFERENCE.md  (Usage guide)
│
├── CV_INTEGRATION_GUIDES/
│   ├── SINGLE_FIT_REFERENCE_WITH_INDEX_ACCESS_INDEX.md  (Master index)
│   └── IMPLEMENTATION_COMPLETE_DELIVERABLE.md           (Summary)
│
├── MULTI_DIMENSIONAL_SLICING_GUIDE.md
├── MULTI_DIMENSIONAL_SLICING_QUICK_REF.md
├── TABLE_DATA_SLICING_QUICK_REFERENCE.md
│
├── ANALYSIS_CONFIGURATION.md
├── FILE_STRUCTURE_GUIDE.md
├── FUNCTION_DEVELOPMENT_REFERENCE.md
│
├── README_ROUTING.md
└── ROUTING_VISUAL_GUIDE.md
```

---

## 🔗 Related Files

- **Implementation**: `chemometrics/cv_pipeline.py` - Core CV pipeline with single-fit reference mode
- **Test**: `tests/test_single_fit_reference.py` - Working example
- **Module**: `chemometrics/univ_calibration.py` - Univariate calibration module

---

## 📖 Feature Summary

### Single-Fit Reference Mode ✨

Run a function's single fit on all data as a reference, then run CV folds to compare.

**Benefits**:
- Assess model stability across training subsets
- Identify unstable parameters
- Measure prediction robustness
- No need for separate y_test

**Status**: ✅ Complete and ready to use

**See**: [CV_INTEGRATION_GUIDES/IMPLEMENTATION_COMPLETE_DELIVERABLE.md](CV_INTEGRATION_GUIDES/IMPLEMENTATION_COMPLETE_DELIVERABLE.md)

### Index-Based Fold Access 📇

Access fold outputs by index instead of hardcoding fold names.

**Benefits**:
- Scalable code works with any n_splits
- No string formatting needed
- Clean loops and iterations
- Efficient fold processing

**Status**: ✅ Complete and ready to use

**See**: [CV_FOLD_OUTPUT_HANDLING/INDEX_BASED_FOLD_ACCESS.md](CV_FOLD_OUTPUT_HANDLING/INDEX_BASED_FOLD_ACCESS.md)

---

## 💡 Documentation Tips

1. **New to a feature?** Start with the QUICK_REF file
2. **Need examples?** Check the specific feature folder
3. **Want all details?** Read the complete MODE/SUMMARY files
4. **Lost?** Check the master index in CV_INTEGRATION_GUIDES
5. **Implementation details?** See the COMPLETE files

---

## ✅ All Features Complete

- ✅ Single-fit reference mode
- ✅ Index-based fold access
- ✅ Segregated output handling
- ✅ Univariate calibration integration
- ✅ Comprehensive documentation (15 files organized in 4 folders)
- ✅ Working test examples

**Ready to use!** 🚀

---

## Contact & Support

For questions about specific features, see the relevant documentation folder or check the implementation files in `chemometrics/`.
