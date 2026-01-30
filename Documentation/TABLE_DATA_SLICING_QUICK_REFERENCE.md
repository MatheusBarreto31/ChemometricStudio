# Table Data Slicing - Quick Reference

## What is Table Data Slicing?

Table data slicing allows users to interactively navigate through multi-dimensional data arrays using Previous/Next buttons. This is useful for 3D+ data where you want to examine one 2D slice at a time.

## Quick Start Example

Add this to your function's JSON configuration in the `analysis` section:

```json
{
  "type": "table",
  "config": {
    "data_source": "X_cal",
    "title": "Spectral Data Slice",
    "data_slicing": [
      {
        "name": "Sample",
        "dimension": 0,
        "default": 0,
        "show_navigation_menu": true
      }
    ]
  }
}
```

## What You'll See

```
Sample: 1/50

[<] 1 [>]

┌─────────────────────┐
│ Col0  Col1  Col2    │
├─────────────────────┤
│ 1.23  2.45  3.67    │
│ 4.56  5.78  6.89    │
│ 7.01  8.12  9.34    │
└─────────────────────┘
```

- **"Sample: 1/50"** - Current slice index (1) out of total (50)
- **< >** - Navigation buttons to move through slices
- **Table below** - Shows the 2D data from the current slice

## Configuration Options

### Essential Properties

| Property | Type | Required | Default | Purpose |
|----------|------|----------|---------|---------|
| `name` | string | ✓ | - | Label for the dimension (e.g., "Sample") |
| `dimension` | integer | ✓ | - | Which dimension to slice (0=rows, 1=columns, etc.) |
| `data_slicing` | array | ✓ | - | Must be present to enable slicing |

### Optional Properties

| Property | Type | Default | Purpose |
|----------|------|---------|---------|
| `default` | integer | 0 | Which slice to show initially |
| `show_navigation_menu` | boolean | true | Whether to show < > buttons |

## Multi-Dimensional Slicing (3D Data)

For data with 3+ dimensions, you can slice multiple dimensions:

```json
{
  "data_slicing": [
    {
      "name": "Sample",
      "dimension": 0,
      "default": 0,
      "show_navigation_menu": true
    },
    {
      "name": "Batch", 
      "dimension": 2,
      "default": 0,
      "show_navigation_menu": true
    }
  ]
}
```

This creates two sets of navigation controls - one for each dimension.

## Real-World Examples

### Example 1: Spectroscopic Data (2D → 1D Slices)

**Data shape:** (50 samples, 1024 wavelengths)

```json
{
  "type": "table",
  "config": {
    "data_source": "spectra",
    "title": "Individual Spectra",
    "data_slicing": [
      {
        "name": "Sample #",
        "dimension": 0,
        "show_navigation_menu": true
      }
    ]
  }
}
```

**Result:** Browse through all 50 individual spectra, one at a time.

### Example 2: Hyperspectral Imaging (3D → 2D Slices)

**Data shape:** (100 lines, 100 pixels, 256 bands)

```json
{
  "type": "table",
  "config": {
    "data_source": "hyperspectral_cube",
    "title": "Hyperspectral Image Analysis",
    "max_rows": 25,
    "max_cols": 25,
    "data_slicing": [
      {
        "name": "Wavelength Band",
        "dimension": 2,
        "show_navigation_menu": true
      }
    ]
  }
}
```

**Result:** View the spatial pattern at each wavelength band.

### Example 3: Multi-Batch Analysis (4D Data)

**Data shape:** (10 batches, 50 samples, 100 variables, 20 replicates)**

```json
{
  "type": "table",
  "config": {
    "data_source": "multi_batch_data",
    "title": "Batch and Sample Analysis",
    "data_slicing": [
      {
        "name": "Batch",
        "dimension": 0,
        "show_navigation_menu": true
      },
      {
        "name": "Sample",
        "dimension": 1,
        "show_navigation_menu": true
      }
    ]
  }
}
```

**Result:** Navigate through batches and samples independently.

## Common Mistakes

❌ **Wrong:** Using string for `dimension`
```json
"dimension": "0"  // Wrong - should be integer
```

✅ **Right:** Using integer
```json
"dimension": 0  // Correct
```

---

❌ **Wrong:** Forgetting the data_slicing array
```json
{
  "type": "table",
  "config": {
    "data_source": "X_cal",
    "title": "My Table"
    // Missing data_slicing!
  }
}
```

✅ **Right:** Always include data_slicing array
```json
{
  "type": "table",
  "config": {
    "data_source": "X_cal",
    "title": "My Table",
    "data_slicing": [...]
  }
}
```

---

❌ **Wrong:** Slicing a dimension that doesn't exist
```json
"data_slicing": [
  {
    "name": "Band",
    "dimension": 5  // If data is only 3D, this won't work!
  }
]
```

✅ **Right:** Check your data shape first
```python
# If data.shape = (50, 100, 256), valid dimensions are 0, 1, 2
"data_slicing": [
  {
    "name": "Band",
    "dimension": 2  // Correct - 0-indexed, within bounds
  }
]
```

## Comparing to Graph Slicing

Tables and graphs both support `data_slicing` using the same configuration format:

| Feature | Tables | Graphs |
|---------|--------|--------|
| Multi-dimensional | ✓ | ✓ |
| Navigation controls | ✓ | ✓ |
| Config format | Same | Same |
| Performance | Fast (native tables) | Slower (matplotlib render) |
| Visual representation | Numeric data | Visual plot |

Use **tables** for numeric exploration, **graphs** for visual patterns.

## FAQ

**Q: Can I hide the navigation buttons?**
A: Yes, set `"show_navigation_menu": false`

**Q: Can I change the initial slice?**
A: Yes, set `"default": 5` to start at slice 5 instead of 0

**Q: What if my data is only 2D?**
A: Slicing still works! Each slice will be a 1D row/column

**Q: Can I slice columns instead of rows?**
A: Yes, set `"dimension": 1` to slice along columns

**Q: Do the navigation buttons wrap around?**
A: No, they stop at first and last indices (0 and max)

**Q: Can I combine slicing with filtering?**
A: Slicing is applied first, then the table is displayed with all other features (export, statistics, etc.)

## See Also

- [TABLE_DATA_SLICING_IMPLEMENTATION.md](TABLE_DATA_SLICING_IMPLEMENTATION.md) - Full technical documentation
- [ANALYSIS_CONFIGURATION.md](Documentation/ANALYSIS_CONFIGURATION.md) - Complete analysis config reference
- [GUI_DOCUMENTATION.md](Documentation/GUI_DOCUMENTATION.md) - General GUI documentation
