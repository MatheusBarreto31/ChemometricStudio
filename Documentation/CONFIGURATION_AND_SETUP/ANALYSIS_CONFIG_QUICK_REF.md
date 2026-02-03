# Analysis Configuration Persistence - Quick Reference

## Feature Summary

**Save/Load analysis page layouts and configurations in model.json**

- ✅ Automatically persists with model
- ✅ All layout types supported (fp, ns, ew, fd, sd)
- ✅ Section configurations preserved
- ✅ Backward compatible with old models
- ✅ 17 comprehensive tests (100% pass)

## What Gets Saved

```
Analysis Data for Each Function Instance:
├── Pages (array)
│   ├── Title
│   ├── Layout type (fp, ns, ew, fd, sd)
│   └── Sections (array)
│       ├── Type (graph, table, or null)
│       └── Configuration
│           ├── Graph: graph_type, axes, title, navigation
│           └── Table: data_source, decimal_places, max_rows/cols
└── Current page index
```

## Model.json Structure

```json
{
  "metadata": {...},
  "functions": [...],
  "routing": [...],
  "analysis": {
    "instance_alias": {
      "pages": [{
        "title": "Page Title",
        "layout": "fd",
        "sections": [
          {"type": "graph", "config": {...}},
          {"type": "table", "config": {...}},
          {"type": null},
          {"type": null}
        ]
      }],
      "current_page": 0
    }
  }
}
```

## Core Methods

### Serialization
```python
_serialize_analysis_data() -> dict
  └─ Returns analysis config for saving to model.json
  └─ Excludes execution results and runtime state
  └─ Deep copies section configs
```

### Deserialization
```python
_deserialize_analysis_data(analysis_config: dict) -> None
  └─ Restores analysis data from loaded config
  └─ Initializes analysis_data if needed
  └─ Sets defaults for missing fields
```

### Integration Points
```python
_generate_model_json()
  └─ Calls _serialize_analysis_data()
  └─ Merges analysis into model.json

_parse_and_load_model_json()
  └─ Calls _deserialize_analysis_data()
  └─ Restores pages and layouts
```

## Automatic Save/Load

### Saves To
- ✅ .mdcd files (model with calibration data)
- ✅ .mdon files (model method only)
- ✅ .mdfd files (full model)
- ✅ model.json (direct generation)

### Loads From
- ✅ .mdcd, .mdon, .mdfd archives
- ✅ model.json files
- ✅ Auto-restores pages on load

## Layout Types

| Type | Sections | Use Case |
|------|----------|----------|
| `fp` | 1 full page | Single large graph or table |
| `ns` | 2 stacked | Comparison before/after |
| `ew` | 2 side-by-side | Left-right comparison |
| `fd` | 4 grid (2×2) | Multiple data views |
| `sd` | 3 (top 2, bottom 2) | Hierarchical display |

## Section Types

### Graph Section
```python
{'type': 'graph', 'config': {
    'graph_type': 'scatter|line|heatmap|bar|histogram|box',
    'x_axis': {'data_source': 'name', 'label': 'X'},
    'y_axis': {'data_source': 'name', 'label': 'Y'},
    'title': 'Graph Title',
    'navigation_axes': ['Axis1'],
    'slice_info': {'dimension': 0, 'index': 0}
}}
```

### Table Section
```python
{'type': 'table', 'config': {
    'data_source': 'name',
    'decimal_places': 4,
    'max_rows': 50,
    'max_cols': 15,
    'column_headers': ['X', 'Y'],
    'row_headers': ['S1', 'S2']
}}
```

### Empty Section
```python
{'type': None}
```

## Access Patterns

### Check if analysis exists
```python
if hasattr(self, 'analysis_data') and self.analysis_data:
    # Analysis data is present
```

### Get pages for function
```python
pages = self.analysis_data['function_alias']['pages']
current_page = self.analysis_data['function_alias']['current_page']
```

### Create new function analysis
```python
self.analysis_data['new_func'] = {
    'pages': [
        {
            'title': 'Default',
            'layout': 'fp',
            'sections': [{'type': None}]
        }
    ],
    'current_page': 0
}
```

## Data NOT Persisted

❌ Execution results (arrays, outputs)  
❌ Runtime state (graph slices, table filters)  
❌ Temporary data (selections, zoom levels)  

**Reason**: Keep file size small, preserve layout only

## Backward Compatibility

✅ **Old models load without errors**
- Missing analysis section defaults to empty dict
- No breaking changes to APIs
- Old models automatically get empty analysis_data

```python
# Safe default handling
analysis_config = model_data.get('analysis', {})
# Returns {} if not present - no crash
```

## Test Coverage

- ✅ 17 comprehensive tests
- ✅ 100% pass rate (17/17)
- ✅ Covers all major scenarios
- ✅ Round-trip save/load verified

## Error Handling

All operations fail gracefully:

```python
# Serialization errors
except Exception as e:
    print(f"Warning: Failed to serialize: {e}")
    return {}  # Doesn't crash

# Deserialization errors
except Exception as e:
    print(f"Warning: Failed to deserialize: {e}")
    self.analysis_data = {}  # Sets empty, continues
```

## Usage Workflow

```
1. User creates analysis pages
   └─ Adds graphs/tables to sections
   └─ Arranges layouts

2. User saves model
   └─ Analysis config automatically included
   └─ Saved to model.json in archive

3. User loads model later
   └─ Pages automatically restored
   └─ Previous layout displayed

4. User runs "Run to Here"
   └─ Execution results populated
   └─ Sections updated with new data
```

## File Size Impact

**Per function instance with 3 pages:**
- Metadata: ~150 bytes
- Sections: ~200 bytes each
- Total: ~1-2 KB per instance

**5-function pipeline: 15-20 KB total**
- Negligible impact on archive size
- Configuration only (results not saved)

## Common Tasks

### Verify config was saved
```python
# In model.json, should see:
"analysis": {
    "function_alias": {
        "pages": [...],
        "current_page": 0
    }
}
```

### Debug serialization
```python
# Check if data serializes properly
try:
    json_str = json.dumps(self.analysis_data)
    print(f"✓ Serializable: {len(json_str)} bytes")
except Exception as e:
    print(f"✗ Error: {e}")
```

### Clear analysis data
```python
# Reset to defaults
self.analysis_data = {}
# Or for specific instance
self.analysis_data.pop('function_alias', None)
```

## Known Limitations

1. **No execution result caching** - Must "Run to Here" again
2. **No runtime state saving** - Graph slices/filters reset
3. **Layout only** - Data not persisted with pages

## Production Status

✅ **COMPLETE AND TESTED**
- Implementation: Complete
- Testing: 17/17 pass
- Documentation: Complete
- Integration: Automatic
- Backward compatible: Yes
- Production ready: Yes

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Config not loading | Check model.json exists and has "analysis" section |
| Empty analysis_data | Create pages with "Add Page" button |
| Config loss on save | Ensure running latest version with serialize call |
| Old model won't load | Backward compatible - should work automatically |

---

**For detailed information**: See ANALYSIS_CONFIG_PERSISTENCE.md  
**Test file**: test_analysis_config_persistence.py  
**Status**: ✅ Production Ready

