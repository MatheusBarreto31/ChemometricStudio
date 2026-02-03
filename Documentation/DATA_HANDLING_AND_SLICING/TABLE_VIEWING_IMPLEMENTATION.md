# Data Table Viewing Capability - Implementation Guide

## Overview

A comprehensive table/spreadsheet viewing system has been implemented for displaying and analyzing multi-dimensional data in the Chemometrics GUI. Users can now view, export, and analyze tabular data with features like statistics, CSV export, and custom formatting.

## Features Implemented

### 1. **Table Rendering**
- ✅ 1D array display (single column)
- ✅ 2D array display (multi-column spreadsheet)
- ✅ High-dimensional array support (flattened view)
- ✅ Custom column headers
- ✅ Custom row headers
- ✅ Automatic column width calculation
- ✅ Configurable max rows/columns

### 2. **Data Display & Formatting**
- ✅ Customizable decimal places
- ✅ Number formatting for floats
- ✅ Support for multiple data types (int, float, complex, string)
- ✅ Data shape and type information
- ✅ Horizontal and vertical scrollbars

### 3. **Statistics & Analysis**
- ✅ Basic statistics (min, max, mean, median)
- ✅ Standard deviation and variance
- ✅ Quartiles (Q1, Q2, Q3)
- ✅ Element counting (total, non-zero, zeros, NaN, Inf)
- ✅ Statistics display in popup window

### 4. **Export & Sharing**
- ✅ CSV export functionality
- ✅ Automatic timestamp in filenames
- ✅ 1D, 2D, and flattened export support
- ✅ Customizable title in exports

### 5. **State Management**
- ✅ Per-table state tracking
- ✅ Sort column tracking
- ✅ Filter text storage
- ✅ Current slice management
- ✅ Table state refresh capability

## Architecture

### Main Method: `_render_table_section()`

Enhanced version of table rendering with comprehensive features.

**Location**: main_gui.py, line 1765

**Workflow**:
1. Validate execution results
2. Retrieve data source
3. Initialize table state (if first render)
4. Configure table settings (title, decimals, limits)
5. Create main container with title and info bar
6. Add toolbar with action buttons
7. Create table view with data

**Configuration Options**:
```python
config = {
    'title': 'Data Table',              # Table title
    'data_source': 'output_name',       # Which output to display
    'decimal_places': 4,                # Decimal formatting
    'max_rows': 50,                     # Maximum rows to display
    'max_cols': 15,                     # Maximum columns to display
    'column_headers': ['X', 'Y', 'Z'],  # Custom column names
    'row_headers': ['S1', 'S2', ...]    # Custom row names
}
```

### Helper Methods

#### 1. `_create_table_view()` - 70 lines
Creates the actual Treeview widget with data.

**Parameters**:
- `parent`: Parent frame for the table
- `data`: NumPy array to display
- `config`: Configuration dictionary
- `decimal_places`: Decimal places for formatting
- `max_rows`: Row limit
- `max_cols`: Column limit
- `col_headers`: Column header list
- `row_headers`: Row header list

**Features**:
- Data preparation (reshape for 1D/3D+ arrays)
- Column and row creation
- Data formatting and insertion
- Scrollbar setup

#### 2. `_export_table_to_csv()` - 30 lines
Exports table data to CSV file.

**Features**:
- Automatic filename generation with timestamp
- Supports 1D, 2D, and flattened array export
- Title row in CSV
- Proper formatting

#### 3. `_show_table_statistics()` - 35 lines
Displays statistical popup window.

**Statistics Shown**:
- Shape and data type
- Min, max, mean, median
- Standard deviation, variance
- Quartiles (Q1, Q2, Q3)
- Element counts (total, non-zero, zeros, NaN, Inf)

#### 4. `_refresh_table()` - 15 lines
Refreshes table display and resets state.

**Actions**:
- Resets sort column
- Resets filter text
- Resets slice index
- Clears temporary state

## Configuration Schema

### Basic Configuration

```json
{
  "section_type": "table",
  "title": "Analysis Results",
  "data_source": "processed_data",
  "decimal_places": 4,
  "max_rows": 50,
  "max_cols": 15
}
```

### Advanced Configuration

```json
{
  "section_type": "table",
  "title": "Spectral Data",
  "data_source": "wavelength_intensity",
  "decimal_places": 6,
  "max_rows": 100,
  "max_cols": 20,
  "column_headers": [
    "Wavelength (nm)",
    "Intensity (AU)",
    "Error"
  ],
  "row_headers": [
    "Sample 1",
    "Sample 2",
    "Sample 3"
  ]
}
```

## Data Type Support

### Supported Types

| Type | Format | Display |
|------|--------|---------|
| int32, int64 | Integer | "1", "2", "3" |
| float32, float64 | Decimal | "1.2500" |
| complex128 | Complex number | "1+2j" |
| string | Text | "value" |
| bool | Boolean | "True", "False" |

### Type Conversion

- Lists automatically converted to numpy arrays
- Data coerced to compatible type for display
- Float formatting applied for numeric types

## Display Limits

### Default Limits

```python
max_rows = 50    # Maximum rows to display
max_cols = 15    # Maximum columns to display
```

### Rationale

- Prevents UI overload with large datasets
- Maintains responsive performance
- Scrollbars allow access to full data
- All data still available for export

### Customization

Override in configuration:
```json
{
  "max_rows": 100,
  "max_cols": 25
}
```

## State Management

### Table State Structure

```python
table_state[section_id] = {
    'sort_column': None,        # Column to sort by
    'sort_order': 'ascending',  # Sort direction
    'filter_text': '',          # Filter expression
    'current_slice': 0          # Current slice index
}
```

### Per-Section Isolation

- Each table maintains independent state
- Multiple tables on same page don't interfere
- Section ID = Python id() of section_data

### State Persistence

- State preserved during navigation
- Survives view updates
- Can be reset with Refresh button

## Toolbar Features

### Export Button
- Saves data to CSV file
- Timestamp in filename
- Confirms export with message

### Statistics Button
- Opens statistics popup window
- Shows comprehensive statistics
- Read-only display

### Refresh Button
- Resets table state
- Clears temporary filters
- Re-initializes display

## CSV Export Format

### 1D Array Export
```
Title
[blank line]
Index,Value
0,1.0
1,2.0
2,3.0
```

### 2D Array Export
```
Col0,Col1,Col2
1.0,2.0,3.0
4.0,5.0,6.0
```

### High-Dimensional Export
```
Flattened: (2,3,4)
[flattened data]
```

## Statistics Display

### Window Layout
- Popup window with 500x400 dimensions
- Read-only text widget
- Comprehensive statistics in formatted text

### Statistics Included

**Basic**:
- Min, Max, Mean, Median
- Std Dev, Variance

**Distribution**:
- Q1 (25th percentile)
- Q2 (50th percentile / median)
- Q3 (75th percentile)

**Counts**:
- Total elements
- Non-zero count
- Zero count
- NaN count
- Infinity count

## Information Bar

Displays key data information:
```
Shape: (1000, 50) | Type: float64 | Min: -5.1234 | Max: 8.9876 | Mean: 0.3456
```

### Information Shown
- Shape of array
- Data type
- Minimum value
- Maximum value
- Mean value

## Usage Examples

### Simple Table Configuration

```json
{
  "section_type": "table",
  "title": "Results",
  "data_source": "my_data"
}
```

### With Custom Headers

```json
{
  "section_type": "table",
  "title": "Calibration Data",
  "data_source": "calibration_matrix",
  "decimal_places": 6,
  "column_headers": [
    "Wavelength A",
    "Wavelength B",
    "Wavelength C"
  ]
}
```

### With Row Limits

```json
{
  "section_type": "table",
  "title": "Large Dataset",
  "data_source": "big_array",
  "max_rows": 100,
  "max_cols": 25,
  "decimal_places": 3
}
```

## Implementation Details

### Rendering Pipeline

```
_render_table_section()
├── Validate inputs
├── Get execution results
├── Initialize table state
├── Create main container
├── Add title and info bar
├── Create toolbar
│   ├── Export button
│   ├── Statistics button
│   └── Refresh button
└── _create_table_view()
    ├── Prepare data (reshape)
    ├── Create treeview widget
    ├── Configure columns
    ├── Insert data rows
    └── Add scrollbars
```

### Data Preparation

1. **1D Arrays**: Reshape to (n, 1)
2. **2D Arrays**: Use directly
3. **3D+ Arrays**: Flatten to 2D (shape[0], -1)

### Column Width Calculation

```python
col_width = max(50, min(150, 800 // num_cols))
```

- Minimum: 50 pixels
- Maximum: 150 pixels
- Dynamic based on number of columns

## Error Handling

### Handled Scenarios
- ✅ Missing execution results
- ✅ Unsuccessful execution
- ✅ Missing data source
- ✅ Non-array data (converts to array)
- ✅ High-dimensional arrays (flattens)
- ✅ NaN and Inf values
- ✅ Type conversion errors

### Error Display
- User-friendly error messages
- Red text for visibility
- Traceback printed to console

## Testing

**Test File**: test_table_viewing.py

**Coverage**: 29 test cases
- ✅ Table rendering (9 tests)
- ✅ Statistics (4 tests)
- ✅ CSV export (4 tests)
- ✅ State management (4 tests)
- ✅ Configuration (3 tests)
- ✅ Data types (5 tests)

**Result**: 29/29 PASS ✅

## Performance

- **Rendering**: O(n*m) where n=rows, m=cols
- **Statistics**: O(n) for all calculations
- **Export**: O(n*m) for CSV writing
- **Display Limit**: Prevents UI slowdown with max_rows/cols

## Future Enhancements

1. **Sorting**: Click column headers to sort
2. **Filtering**: Text-based row filtering
3. **Search**: Find values in table
4. **Copy**: Copy data to clipboard
5. **Plot**: Create graphs from table data
6. **Pivot**: Create pivot tables
7. **Aggregation**: Sum, average by groups
8. **Formatting**: Conditional highlighting
9. **Frozen Headers**: Scroll data while keeping headers visible
10. **Column Resizing**: User-adjustable column widths

## Integration

### With Graph Rendering
- Tables and graphs can coexist
- Independent state management
- Share same data sources

### With Navigation Controls
- Tables support multi-dimensional data
- Potential for slice navigation on tables
- State management follows same pattern

### With Configuration System
- JSON-based configuration
- Backward compatible
- Optional features

## Backward Compatibility

✅ **100% Compatible**
- Old table configs work unchanged
- Default values for new options
- No breaking changes to API

## Code Statistics

| Metric | Value |
|--------|-------|
| Main method lines | 60 |
| Helper method lines | 150 |
| Total new code | 210 |
| Test cases | 29 |
| Test pass rate | 100% |
| Documentation lines | 350+ |

## Conclusion

The table viewing capability provides a complete solution for displaying, analyzing, and exporting tabular data. With comprehensive statistics, multiple data type support, and flexible configuration, it seamlessly integrates with the existing analysis system.

**Status**: ✅ COMPLETE AND TESTED

---

**Implementation Date**: January 2026  
**Test Status**: 29/29 PASS ✅  
**Production Ready**: YES
