# Multi-Dimensional Slicing (4D+) - Quick Reference

## Quick Start

Add to your graph configuration:

```json
{
  "type": "graph",
  "config": {
    "graph_type": "line",
    "x_axis": {...},
    "y_axis": {...},
    "show_md_menu": true,          // ← Enable 4D+ slicing
    "data_slicing": [...],
    "md_default": {                // ← Optional defaults
      "combo_index": 0,
      "dim_1": 0,
      "dim_2": 0
    }
  }
}
```

## Configuration Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `show_md_menu` | boolean | `false` | Enable multi-dimensional slicing UI |
| `md_default.combo_index` | integer | `0` | Initial combination index |
| `md_default.dim_X` | integer | `0` | Default slice position for dimension X |

## How It Works

### For 4D Data

**With 1 dimension specified in data_slicing:**
- Remaining 3 dimensions combined in groups of 2 (ndim-2)
- Possible combinations: `(1,2)`, `(1,3)`, `(2,3)`

**With no dimensions specified:**
- All 4 dimensions combined in groups of 2 (ndim-2)
- Possible combinations: `(0,1)`, `(0,2)`, `(0,3)`, `(1,2)`, `(1,3)`, `(2,3)`

### UI Components

1. **Combination Dropdown** - Select which dimensions to slice
2. **Navigation Controls** - Previous/Next buttons for each dimension in combination
3. **Index Display** - Shows current position (1-based)

## Example: 4D Imaging Data

**Data Shape**: `(samples=20, height=100, width=100, channels=3)`

```json
{
  "show_md_menu": true,
  "data_slicing": [
    {"name": "Sample", "dimension": 0, "show_navigation_menu": true}
  ],
  "md_default": {
    "combo_index": 0,
    "dim_1": 50,
    "dim_2": 50
  }
}
```

**Result:**
- Sample dimension: standard navigation
- Remaining dims (1,2,3): combinatorial navigation
- Dropdown shows: `(1,2)`, `(1,3)`, `(2,3)`
- Default: combination 0 = `(1,2)` at positions (50, 50)

## Combination Size Logic

```
IF dimensions specified in data_slicing:
    combo_size = ndim - 1
ELSE:
    combo_size = ndim - 2
```

## When NOT to Use

- Data has **3 or fewer dimensions** (use standard slicing)
- Want **per-dimension control** without combinations
- `show_md_menu` is `false` or omitted

## Tips

✅ **DO:**
- Use meaningful dimension names in `data_slicing`
- Set sensible defaults in `md_default`
- Test with your actual data shape

❌ **DON'T:**
- Enable for 3D data (not supported)
- Mix too many specified dimensions (limits combinations)
- Forget to set `show_md_menu: true`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Menu doesn't appear | Check `show_md_menu: true` is set |
| No combinations shown | Verify data is 4D+ |
| Wrong dimensions in combo | Review `data_slicing` specification |
| Index out of bounds | Check `md_default` values |

## See Also

- [MULTI_DIMENSIONAL_SLICING_GUIDE.md](MULTI_DIMENSIONAL_SLICING_GUIDE.md) - Full documentation
- [example_4d_slicing_config.json](../example_4d_slicing_config.json) - Configuration examples
- [TABLE_DATA_SLICING_QUICK_REFERENCE.md](TABLE_DATA_SLICING_QUICK_REFERENCE.md) - Standard slicing
