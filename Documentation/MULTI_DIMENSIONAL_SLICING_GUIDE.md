# Multi-Dimensional Data Slicing (4D+) - User Guide

## Overview

The multi-dimensional slicing feature allows you to explore 4D and higher-dimensional data by selecting and navigating through different combinations of dimensions. This is particularly useful for complex datasets like hyperspectral imaging, multi-batch analyses, or time-series with multiple spatial dimensions.

## When to Use

Use multi-dimensional slicing when:
- Your data has **4 or more dimensions**
- You want to explore different **combinations** of dimensions
- Standard per-dimension navigation becomes cumbersome

## How It Works

### Standard Slicing (3D and below)
For 3D data, you typically specify which dimensions to slice:
```json
"data_slicing": [
  {"name": "Sample", "dimension": 0, "show_navigation_menu": true}
]
```

This creates a single navigation control for the sample dimension.

### Multi-Dimensional Slicing (4D+)

For 4D+ data, the system:
1. **Identifies remaining dimensions** - those not specified in `data_slicing`
2. **Computes all possible combinations** of these dimensions
3. **Creates UI controls** to select and navigate combinations

#### Combination Size Logic
- **If dimensions are specified in `data_slicing`**: Combinations of size = `ndim - 1`
- **If no dimensions specified**: Combinations of size = `ndim - 2`

## Configuration

### Required Settings

```json
{
  "type": "graph",
  "config": {
    "graph_type": "line",
    "x_axis": {...},
    "y_axis": {...},
    "title": "My 4D Graph",
    "show_md_menu": true,  // ← REQUIRED for multi-dimensional slicing
    "data_slicing": [...],
    "md_default": {...}     // ← Optional defaults
  }
}
```

### Configuration Options

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `show_md_menu` | boolean | Yes | Enable multi-dimensional slicing UI |
| `md_default` | object | No | Default combination and slice positions |

### md_default Options

```json
"md_default": {
  "combo_index": 0,    // Which combination to show initially (0-based)
  "dim_0": 0,          // Default slice position for dimension 0
  "dim_1": 5,          // Default slice position for dimension 1
  "dim_2": 10          // etc.
}
```

## Examples

### Example 1: 4D Spectroscopic Imaging Data

**Data Shape**: `(20 samples, 100 height, 100 width, 256 wavelengths)`

**Goal**: Navigate through different spatial slices while viewing spectral profiles

```json
{
  "type": "graph",
  "config": {
    "graph_type": "line",
    "x_axis": {"data_source": "axis_n_info", "index": 3, "label": "Wavelength"},
    "y_axis": {"data_source": "X_cal", "label": "Intensity"},
    "title": "Spectral Profile at Point (x,y)",
    "show_md_menu": true,
    "data_slicing": [
      {
        "name": "Sample",
        "dimension": 0,
        "default": 0,
        "show_navigation_menu": true
      }
    ],
    "md_default": {
      "combo_index": 0,
      "dim_1": 50,
      "dim_2": 50
    }
  }
}
```

**What happens:**
- Dimension 0 (samples) has standard navigation
- Remaining dimensions (1, 2, 3) are combined
- Combinations of size 2 (ndim-2): `(1,2)`, `(1,3)`, `(2,3)`
- Dropdown lets you select which pair to slice
- Navigation controls appear for selected dimensions

**UI Display:**
```
Sample: 1/20  [<] 1 [>]

┌─────────────────────────────────────────┐
│ Multi-Dimensional Slicing (4D+)        │
├─────────────────────────────────────────┤
│ Dimension Combination: [Dims: 1, 2  ▼] │
│                                         │
│ Dimension 1: 50/100  [<] 50 [>]        │
│ Dimension 2: 50/100  [<] 50 [>]        │
└─────────────────────────────────────────┘

[Graph shows spectral profile at sample 1, position (50, 50)]
```

### Example 2: 5D Multi-Batch Time Series

**Data Shape**: `(5 batches, 30 samples, 100 timepoints, 50 sensors, 3 features)`

```json
{
  "type": "graph",
  "config": {
    "graph_type": "line",
    "x_axis": {"data_source": "axis_n_info", "index": 2, "label": "Time"},
    "y_axis": {"data_source": "X_cal", "label": "Sensor Value"},
    "title": "Time Series Analysis",
    "show_md_menu": true,
    "data_slicing": [
      {
        "name": "Batch",
        "dimension": 0,
        "default": 0,
        "show_navigation_menu": true
      },
      {
        "name": "Sample",
        "dimension": 1,
        "default": 0,
        "show_navigation_menu": true
      }
    ],
    "md_default": {
      "combo_index": 0,
      "dim_2": 0,
      "dim_3": 25,
      "dim_4": 0
    }
  }
}
```

**What happens:**
- Dimensions 0, 1 have standard navigation
- Remaining dimensions (2, 3, 4) combined
- Only one combination possible: `(2,3,4)` (size = ndim-2 = 3)
- All three dimensions get navigation controls

### Example 3: 4D with No Specified Dimensions

**Data Shape**: `(50, 40, 30, 20)`

```json
{
  "type": "graph",
  "config": {
    "graph_type": "scatter",
    "x_axis": {"data_source": "X_cal", "label": "X"},
    "y_axis": {"data_source": "X_cal", "label": "Y"},
    "title": "4D Exploration",
    "show_md_menu": true,
    "data_slicing": [],  // No dimensions specified
    "md_default": {
      "combo_index": 0,
      "dim_0": 25,
      "dim_1": 20
    }
  }
}
```

**What happens:**
- No dimensions specified in `data_slicing`
- All 4 dimensions available for combinations
- Combinations of size 2 (ndim-2): `(0,1)`, `(0,2)`, `(0,3)`, `(1,2)`, `(1,3)`, `(2,3)`
- Dropdown shows all 6 combinations
- Two navigation controls appear for selected pair

## User Workflow

1. **Initial Display**
   - Graph renders with default combination and slice positions
   - Multi-dimensional slicing UI appears (if enabled)

2. **Change Combination**
   - Select different dimension pair from dropdown
   - Navigation controls update for new dimensions
   - Graph re-renders with new slicing

3. **Navigate Slices**
   - Use Previous/Next buttons for each dimension
   - Graph updates in real-time
   - Index displays show current position (1-based for users)

4. **Standard Navigation**
   - Any dimensions in `data_slicing` have their own controls
   - Work independently from multi-dimensional slicing

## Technical Details

### Combination Algorithm

```python
from itertools import combinations

# Get all dimensions
all_dims = set(range(ndim))

# Remove specified dimensions
specified_dims = {0, 1}  # From data_slicing
remaining_dims = all_dims - specified_dims

# Compute combinations
combo_size = ndim - 1 if specified_dims else ndim - 2
combos = list(combinations(remaining_dims, combo_size))
```

### State Management

The system maintains:
- `md_combo_index`: Currently selected combination
- `md_slice_indices`: Dict mapping dimension → current index
- Both persist across navigation events

### Data Extraction

When rendering:
1. Merge standard indices (`indices`)
2. Merge multi-dimensional indices (`md_slice_indices`)
3. Merge axis-specific indices (`axis_indices`)
4. Apply all to data extraction

## Best Practices

1. **Use meaningful names** in `data_slicing` for clarity
2. **Set sensible defaults** in `md_default` to start at interesting positions
3. **Consider data size** - very high-dimensional data may have many combinations
4. **Test combinations** to ensure they make sense for your use case
5. **Combine with standard slicing** for the most control

## Limitations

- Only works with **4D+ data**
- Requires `show_md_menu: true` to be enabled
- All dimensions must be valid array dimensions
- Combinations are computed at runtime based on data shape

## Backward Compatibility

This feature is **fully backward compatible**:
- If `show_md_menu` is `false` or omitted, standard slicing is used
- Existing 3D data slicing works unchanged
- No changes to existing configurations required

---

**Questions or Issues?**
Refer to `example_4d_slicing_config.json` for complete configuration examples.
