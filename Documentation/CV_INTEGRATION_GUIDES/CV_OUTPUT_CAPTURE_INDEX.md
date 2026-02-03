# CV Output Capture Feature - Complete Documentation Index

**Status**: ✅ **Implemented, tested, and ready to use**

This index guides you through the new "Output Capture" feature for cross-validation, which lets you run a function across CV folds and capture the actual outputs (scores, loadings, coefficients) from each fold to assess stability.

---

## Start Here 👇

### For You Right Now
📖 **[CV_OUTPUT_STABILITY_GUIDE_USER.md](CV_OUTPUT_STABILITY_GUIDE_USER.md)**
- **Read this first** - It answers your exact question
- Real PCA example with complete code
- Step-by-step walkthrough
- Interpretation guide
- Time: 10-15 minutes

### To See It In Action
💻 **[tests/test_cv_output_stability.py](../tests/test_cv_output_stability.py)**
- Run: `python tests/test_cv_output_stability.py`
- Shows side-by-side comparison: with/without output capture
- Demonstrates stability assessment workflow
- Real numerical results you can examine
- Time: 2-5 minutes to run

---

## Documentation by Purpose

### I Want To...

**"Understand what was done"**
→ Read: [CV_OUTPUT_CAPTURE_COMPLETE.md](CV_OUTPUT_CAPTURE_COMPLETE.md)
- What you asked for
- What changed in the code
- Key implementation details
- Test results

**"Use this feature with my data"**
→ Read: [CV_OUTPUT_STABILITY_GUIDE_USER.md](CV_OUTPUT_STABILITY_GUIDE_USER.md)
- Step-by-step example with PCA
- Real code you can copy/adapt
- How to interpret results
- Most practical guide

**"See all the details"**
→ Read: [OUTPUT_STABILITY_ASSESSMENT.md](OUTPUT_STABILITY_ASSESSMENT.md)
- Comprehensive reference guide
- Quick example section
- Advanced patterns (multiple outputs)
- Stability metrics and interpretation
- Integration examples
- Memory and performance notes

**"Get a quick reference"**
→ Read: [CV_OUTPUT_CAPTURE_QUICK_REF.md](CV_OUTPUT_CAPTURE_QUICK_REF.md)
- One-page cheatsheet
- Configuration snippets
- Common patterns table
- Stability interpretation table
- Best for quick lookup while coding

**"Add CV support to my own functions"**
→ Read: [CV_INTEGRATION_GUIDE.md](CV_INTEGRATION_GUIDE.md)
- 3-step integration pattern
- How the routing works
- Complete working example
- Common mistakes and fixes
- Testing patterns

**"Understand the technical implementation"**
→ Read: [CV_OUTPUT_CAPTURE_IMPLEMENTATION.md](CV_OUTPUT_CAPTURE_IMPLEMENTATION.md)
- Problem statement
- Solution architecture
- Architecture diagrams
- Design decisions (why it works this way)
- Backward compatibility details
- Performance implications

**"See what changed in the code"**
→ Read: [CV_OUTPUT_CAPTURE_ENHANCEMENT.md](CV_OUTPUT_CAPTURE_ENHANCEMENT.md)
- Summary of changes
- Before/after comparison
- Files modified
- Test results

---

## Documentation Map

```
CV Output Capture Feature
│
├─ START HERE
│  ├─ CV_OUTPUT_STABILITY_GUIDE_USER.md ⭐ (For you!)
│  └─ tests/test_cv_output_stability.py (Run this!)
│
├─ GUIDES BY FUNCTION
│  ├─ For understanding what was done
│  │  └─ CV_OUTPUT_CAPTURE_COMPLETE.md
│  │
│  ├─ For using the feature
│  │  ├─ CV_OUTPUT_STABILITY_GUIDE_USER.md
│  │  ├─ OUTPUT_STABILITY_ASSESSMENT.md (Comprehensive)
│  │  └─ CV_OUTPUT_CAPTURE_QUICK_REF.md (Quick lookup)
│  │
│  ├─ For adding CV to functions
│  │  ├─ CV_INTEGRATION_GUIDE.md
│  │  └─ tests/test_cv_output_stability.py (Example)
│  │
│  └─ For technical details
│     ├─ CV_OUTPUT_CAPTURE_IMPLEMENTATION.md
│     └─ CV_OUTPUT_CAPTURE_ENHANCEMENT.md
│
└─ SOURCE CODE
   ├─ chemometrics/cv_pipeline.py (Modified)
   │  ├─ CVConfig (Added capture_outputs parameter)
   │  └─ CVPipeline.run() (Enhanced to capture outputs)
   └─ tests/test_cv_output_stability.py (New test)
```

---

## Quick Reference Table

| Need | Document | Time |
|------|----------|------|
| Your exact question answered | CV_OUTPUT_STABILITY_GUIDE_USER.md | 10 min |
| See it working | tests/test_cv_output_stability.py | 2 min |
| Code example to copy | OUTPUT_STABILITY_ASSESSMENT.md | 5 min |
| One-page cheatsheet | CV_OUTPUT_CAPTURE_QUICK_REF.md | 2 min |
| Integrate with my function | CV_INTEGRATION_GUIDE.md | 15 min |
| Technical deep dive | CV_OUTPUT_CAPTURE_IMPLEMENTATION.md | 20 min |
| Summary of changes | CV_OUTPUT_CAPTURE_ENHANCEMENT.md | 10 min |

---

## The Feature In One Picture

```
BEFORE (Traditional CV)
├─ Data: 100 samples
├─ Split into 5 folds
├─ Run PCA on each fold
└─ Return: RMSE mean/std only
   ❌ Can't compare actual PCA scores

AFTER (CV with Output Capture)
├─ Data: 100 samples
├─ Split into 5 folds
├─ Run PCA on each fold
├─ Capture PCA scores from each fold
└─ Return: RMSE + 5 different score matrices
   ✅ Can compare scores to assess stability!
```

---

## Core Concept

Your function is **called 5 times** (for 5-fold CV):

```
Fold 0: Train on [1-20, 40-100]     → PCA model A → Test scores A
Fold 1: Train on [1-40, 60-100]     → PCA model B → Test scores B
Fold 2: Train on [1-60, 80-100]     → PCA model C → Test scores C
Fold 3: Train on [1-80, 100]        → PCA model D → Test scores D
Fold 4: Train on [20-80]            → PCA model E → Test scores E

Compare: How different are scores A, B, C, D, E?
         → Measure stability using std, correlation, etc.
```

---

## Configuration Example

```python
from chemometrics.cv_pipeline import CVConfig, CVPipeline

# Configure what to capture
cv_config = CVConfig(
    use_cv=True,
    cv_strategy='kfold',
    n_splits=5,
    output_metrics=['rmse'],           # Metrics to aggregate
    capture_outputs=['scores']         # Outputs to capture per fold
)

# Run PCA through CV
pipeline = CVPipeline(cv_config)
results = pipeline.run(pca_function, X=X)

# Access results
rmse_mean = results['rmse_mean']            # Aggregated metric
scores_0 = results['scores_fold_0']        # Per-fold output
scores_1 = results['scores_fold_1']        # Per-fold output
# ... scores_2, scores_3, scores_4

# Assess stability
scores_stacked = np.stack([results[f'scores_fold_{i}'] for i in range(5)])
stability = np.std(scores_stacked, axis=0)  # Low = stable, High = unstable
```

---

## Files Reference

### Documentation Files
| File | Purpose | Read Time |
|------|---------|-----------|
| CV_OUTPUT_STABILITY_GUIDE_USER.md | Your question answered + walkthrough | 10 min |
| OUTPUT_STABILITY_ASSESSMENT.md | Comprehensive reference | 20 min |
| CV_OUTPUT_CAPTURE_QUICK_REF.md | Quick lookup cheatsheet | 2 min |
| CV_INTEGRATION_GUIDE.md | Add CV support to functions | 15 min |
| CV_OUTPUT_CAPTURE_IMPLEMENTATION.md | Technical details | 20 min |
| CV_OUTPUT_CAPTURE_ENHANCEMENT.md | Summary of changes | 10 min |
| CV_OUTPUT_CAPTURE_COMPLETE.md | Status and next steps | 5 min |

### Code Files
| File | Modified/Created | Purpose |
|------|------------------|---------|
| chemometrics/cv_pipeline.py | Modified | Core implementation |
| tests/test_cv_output_stability.py | Created | Runnable example |

---

## Recommended Reading Order

### Option 1: I Just Want To Use It (15 minutes)
1. Read: [CV_OUTPUT_STABILITY_GUIDE_USER.md](CV_OUTPUT_STABILITY_GUIDE_USER.md)
2. Run: `python tests/test_cv_output_stability.py`
3. Copy pattern from guide and adapt to your data

### Option 2: I Want Full Understanding (45 minutes)
1. Read: [CV_OUTPUT_CAPTURE_COMPLETE.md](CV_OUTPUT_CAPTURE_COMPLETE.md) - Overview
2. Read: [CV_OUTPUT_STABILITY_GUIDE_USER.md](CV_OUTPUT_STABILITY_GUIDE_USER.md) - User guide
3. Run: `python tests/test_cv_output_stability.py` - See it work
4. Read: [OUTPUT_STABILITY_ASSESSMENT.md](OUTPUT_STABILITY_ASSESSMENT.md) - Comprehensive details
5. Read: [CV_OUTPUT_CAPTURE_IMPLEMENTATION.md](CV_OUTPUT_CAPTURE_IMPLEMENTATION.md) - Technical depth

### Option 3: I'm Integrating This Into My Codebase (60 minutes)
1. Read: [CV_INTEGRATION_GUIDE.md](CV_INTEGRATION_GUIDE.md) - Integration patterns
2. Read: [CV_OUTPUT_STABILITY_GUIDE_USER.md](CV_OUTPUT_STABILITY_GUIDE_USER.md) - User guide
3. Study: [tests/test_cv_output_stability.py](../tests/test_cv_output_stability.py) - Example code
4. Read: [OUTPUT_STABILITY_ASSESSMENT.md](OUTPUT_STABILITY_ASSESSMENT.md) - Advanced patterns
5. Skim: [CV_OUTPUT_CAPTURE_QUICK_REF.md](CV_OUTPUT_CAPTURE_QUICK_REF.md) - For reference

---

## Key Points

✅ Function is called **5 times** (once per fold)  
✅ Each call gets **different training data**  
✅ Each call produces **different outputs**  
✅ You **capture outputs** from each call  
✅ You **compare outputs** to assess stability  
✅ **Metrics are aggregated**, **outputs are per-fold**  
✅ **100% backward compatible** (opt-in feature)  

---

## Quick Links

### To Learn
- 📖 [User Guide](CV_OUTPUT_STABILITY_GUIDE_USER.md)
- 📘 [Comprehensive Reference](OUTPUT_STABILITY_ASSESSMENT.md)
- 📋 [Quick Reference](CV_OUTPUT_CAPTURE_QUICK_REF.md)

### To Implement
- 🔧 [Integration Guide](CV_INTEGRATION_GUIDE.md)
- 💻 [Example Code](../tests/test_cv_output_stability.py)

### To Understand
- 🏗️ [Implementation Details](CV_OUTPUT_CAPTURE_IMPLEMENTATION.md)
- 📝 [Change Summary](CV_OUTPUT_CAPTURE_ENHANCEMENT.md)
- ✅ [Status & Next Steps](CV_OUTPUT_CAPTURE_COMPLETE.md)

---

## Status

| Item | Status |
|------|--------|
| Core implementation | ✅ Complete |
| Testing | ✅ Passing |
| Documentation | ✅ Comprehensive |
| Example code | ✅ Available |
| Backward compatibility | ✅ Verified |
| Ready to use | ✅ Yes |

---

**Last Updated**: Current session  
**Version**: 1.0  
**Ready to use**: Yes ✅
