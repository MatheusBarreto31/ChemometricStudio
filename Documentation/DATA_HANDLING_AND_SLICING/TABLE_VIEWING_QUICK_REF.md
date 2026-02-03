# Table Viewing - Quick Reference Guide

## At a Glance

**What**: Data table/spreadsheet viewing with statistics and export  
**Where**: Analysis section (table_type sections)  
**Status**: ✅ Complete and tested (29/29 tests pass)

## Quick Start

### Minimal Configuration

```json
{
  "section_type": "table",
  "data_source": "my_output"
}
```

### Full Configuration

```json
{
  "section_type": "table",
  "title": "Results Table",
  "data_source": "output_data",
  "decimal_places": 4,
  "max_rows": 50,
  "max_cols": 15,
  "column_headers": ["X", "Y", "Z"],
  "row_headers": ["S1", "S2"]
}
```

## Features

| Feature | Support | Notes |
|---------|---------|-------|
| 1D Arrays | ✅ | Single column |
| 2D Arrays | ✅ | Multi-column spreadsheet |
| 3D+ Arrays | ✅ | Flattened view |
| Scrollbars | ✅ | Horizontal & vertical |
| Statistics | ✅ | Min, max, mean, quartiles |
| CSV Export | ✅ | With timestamp |
| Custom Headers | ✅ | Rows and columns |
| Decimal Formatting | ✅ | Configurable |
| Data Types | ✅ | int, float, complex, string |

## Methods

### Primary Method
**`_render_table_section(parent, instance_alias, section_data)`**
- Main table rendering method
- Creates UI and toolbar
- Initializes state

### Helper Methods

| Method | Purpose | Lines |
|--------|---------|-------|
| `_create_table_view()` | Create Treeview widget | 70 |
| `_export_table_to_csv()` | Export to CSV | 30 |
| `_show_table_statistics()` | Statistics popup | 35 |
| `_refresh_table()` | Reset table state | 15 |

## Configuration Options

```python
config = {
    # Display
    'title': str,                      # Default: 'Table: {data_source}'
    'decimal_places': int,             # Default: 4
    
    # Data limits
    'max_rows': int,                   # Default: 50
    'max_cols': int,                   # Default: 15
    
    # Headers
    'column_headers': list[str],       # None = auto-generated
    'row_headers': list[str],          # None = use indices
    
    # Source
    'data_source': str                 # Required: key in outputs dict
}
```

## Toolbar Buttons

### Export Button
```python
Command: _export_table_to_csv(data, title)
Output: CSV file with timestamp
Format: Table_Name_YYYYMMDD_HHMMSS.csv
```

### Statistics Button
```python
Command: _show_table_statistics(data, title)
Output: Popup window with statistics
Shows: Min, max, mean, median, std, var, quartiles, counts
```

### Refresh Button
```python
Command: _refresh_table(instance_alias, section_id)
Action: Reset sort, filter, slice state
Effect: Reinitializes table display
```

## Data Handling

### Input Data
- NumPy arrays (preferred)
- Lists (auto-converted)
- Any array-like objects

### Type Support
```
int32, int64      → "1", "2", "3"
float32, float64  → "1.2500" (formatted)
complex128        → "1+2j"
string            → "value"
bool              → "True", "False"
```

### Array Shapes
```
1D: (n,)          → Single column table
2D: (n, m)        → Multi-column table
3D: (n, m, p)     → Flattened to (n, m*p)
```

## State Management

### Table State
```python
table_state = {
    'sort_column': None,           # Current sort column
    'sort_order': 'ascending',     # Sort direction
    'filter_text': '',             # Filter expression
    'current_slice': 0             # Slice index
}
```

### Per-Section
- Each table has independent state
- Identified by section_id (Python id())
- Multiple tables don't interfere

## Usage Patterns

### Pattern 1: Simple Data Display
```json
{
  "section_type": "table",
  "data_source": "results"
}
```

### Pattern 2: High-Precision Display
```json
{
  "section_type": "table",
  "data_source": "calibration_data",
  "decimal_places": 8,
  "title": "Calibration Matrix"
}
```

### Pattern 3: Large Dataset
```json
{
  "section_type": "table",
  "data_source": "big_data",
  "max_rows": 100,
  "max_cols": 25
}
```

### Pattern 4: Custom Headers
```json
{
  "section_type": "table",
  "data_source": "matrix",
  "title": "Sample Matrix",
  "column_headers": ["Var1", "Var2", "Var3"],
  "row_headers": ["Sample A", "Sample B", "Sample C"]
}
```

## Display Info

The info bar shows:
```
Shape: (1000, 50) | Type: float64 | Min: -5.1 | Max: 8.9 | Mean: 0.3
```

Contains:
- Array shape
- Data type
- Min/max/mean values

## Column Width

**Formula**:
```python
width = max(50, min(150, 800 // num_cols))
```

**Range**: 50-150 pixels
**Dynamic**: Based on number of columns

## CSV Export Format

### 1D Array
```
Table Title

Index,Value
0,1.0
1,2.0
```

### 2D Array
```
Table Title

Col0,Col1,Col2
1.0,2.0,3.0
4.0,5.0,6.0
```

### Naming
```
Table_Name_YYYYMMDD_HHMMSS.csv
Example: Results_20260129_143052.csv
```

## Statistics Popup

**Window**: 500x400 pixels  
**Content**: Read-only text display

**Includes**:
```
Shape: (1000, 50)
Data Type: float64

Basic Statistics:
  Min: -5.123456
  Max: 8.987654
  Mean: 0.345678
  Median: 0.234567
  Std Dev: 1.234567
  Variance: 1.524157

Quartiles:
  Q1 (25%): -0.456789
  Q2 (50%): 0.234567
  Q3 (75%): 0.789012

Count:
  Total Elements: 50000
  Non-zero Elements: 49998
  Zero Elements: 2
  NaN Elements: 0
  Inf Elements: 0
```

## Limits & Defaults

```python
max_rows = 50              # Default row limit
max_cols = 15              # Default column limit
decimal_places = 4         # Default decimal places
col_width_min = 50         # Minimum column width
col_width_max = 150        # Maximum column width
stats_window_size = 500x400 # Statistics popup size
```

## Error Handling

**Handled Errors**:
- ✅ Missing execution results
- ✅ Unsuccessful execution status
- ✅ Missing data source
- ✅ Non-array data types
- ✅ High-dimensional arrays
- ✅ NaN and Inf values
- ✅ Type conversion issues

**Response**: User-friendly error message in red

## Testing

**Test File**: test_table_viewing.py

**Test Categories**:
1. Rendering (9 tests)
2. Statistics (4 tests)
3. CSV Export (4 tests)
4. State Management (4 tests)
5. Configuration (3 tests)
6. Data Types (5 tests)

**Result**: ✅ 29/29 PASS

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Rendering | O(n*m) | Limited by max_rows/cols |
| Statistics | O(n) | Fast even for large arrays |
| CSV Export | O(n*m) | Disk I/O dependent |
| State Reset | O(1) | Instantaneous |

## Common Tasks

### Display Data with 6 Decimals
```json
{
  "decimal_places": 6
}
```

### Show Only First 100 Rows
```json
{
  "max_rows": 100
}
```

### Add Column Labels
```json
{
  "column_headers": ["Time", "Signal", "Reference"]
}
```

### Export Data
- Click "Export to CSV" button
- File created in current directory
- Timestamp added to filename

### View Statistics
- Click "Statistics" button
- Popup window opens
- Shows comprehensive statistics

### Reset Table
- Click "Refresh" button
- Resets all state
- Re-initializes display

## Integration

**Compatible With**:
- ✅ Graph rendering
- ✅ Navigation controls
- ✅ Configuration system
- ✅ State management
- ✅ Execution pipeline

**File Size**: +210 lines to main_gui.py

## Backward Compatibility

✅ **Fully Compatible**
- Old configs work as-is
- Default values provided
- No breaking changes

## Code Locations

| Component | Location |
|-----------|----------|
| Main method | main_gui.py:1765 |
| Create view | main_gui.py:~1860 |
| Export CSV | main_gui.py:~1930 |
| Statistics | main_gui.py:~1960 |
| Refresh | main_gui.py:~1990 |

## Summary

Complete table viewing solution with:
- ✅ Multi-format display (1D, 2D, 3D+)
- ✅ Rich statistics (14+ metrics)
- ✅ CSV export with timestamp
- ✅ Flexible configuration
- ✅ Per-table state management
- ✅ Multiple data type support
- ✅ 29/29 tests passing
- ✅ Production ready

**Status**: ✅ READY FOR USE

---

**Quick Links**:
- [Full Implementation Guide](TABLE_VIEWING_IMPLEMENTATION.md)
- [Test File](test_table_viewing.py)
- [Configuration Examples](#configuration-options)
