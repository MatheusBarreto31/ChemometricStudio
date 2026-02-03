# Table Data Slicing Implementation

## Overview

The data slicing functionality previously available only for graphs in the analysis tab has been successfully implemented for tables. This allows users to interactively navigate through multi-dimensional table data using arrow buttons and index displays.

## Features Implemented

### 1. **Navigation Controls**
- Previous/Next buttons to change slice index
- Current index display (1-based for user convenience)
- Maximum index display to show data bounds
- Multiple dimension support for 3D+ data

### 2. **State Management**
- **table_slices**: New data structure in analysis_data to track table slice state
  - `indices`: Dictionary mapping dimensions to current slice indices
  - `data_slicing`: Configuration from table config
  - `outputs`: Execution results reference
  - `config`: Table configuration

### 3. **Data Extraction**
- New `_extract_sliced_data()` method to slice multi-dimensional arrays
- Support for dict-based dimension mapping
- Proper handling of boundary conditions

### 4. **Interactive Updates**
- `_on_table_navigate_slice()` handles button clicks
- Updates both state and UI labels
- Triggers table re-render with new slice
- Full page refresh for reliable display

## API Methods

### `_extract_sliced_data(data, indices)`
Extracts a slice from multi-dimensional data.

**Parameters:**
- `data`: NumPy array to slice
- `indices`: Dict mapping dimension → index (e.g., {0: 5, 1: 2})

**Returns:** Sliced NumPy array

### `_create_table_navigation_controls(parent_frame, instance_alias, section_id, outputs, config, slice_state)`
Creates navigation UI for table slicing.

**Parameters:**
- `parent_frame`: Parent tkinter frame
- `instance_alias`: Function instance identifier
- `section_id`: Unique section identifier
- `outputs`: Execution results dictionary
- `config`: Table configuration
- `slice_state`: Current slice state object

### `_on_table_navigate_slice(instance_alias, section_id, direction, dimension, axis_name, max_index)`
Handles navigation button clicks.

**Parameters:**
- `direction`: -1 for previous, +1 for next
- `dimension`: Dimension to slice (0-indexed)
- `axis_name`: Display name for UI labels
- `max_index`: Maximum valid index for bounds checking

## Configuration Example

### Basic Single Dimension Slicing

```json
{
  "type": "table",
  "config": {
    "data_source": "X_cal",
    "title": "Single Sample Slice",
    "decimal_places": 4,
    "max_rows": 30,
    "max_cols": 15,
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

### Multi-Dimensional Slicing (3D Data)

```json
{
  "type": "table",
  "config": {
    "data_source": "X_cal",
    "title": "Multi-Dimensional Slice",
    "decimal_places": 4,
    "max_rows": 20,
    "max_cols": 15,
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
}
```

## Configuration Properties

### `data_slicing` Array

Each item in the `data_slicing` array defines one navigable dimension:

- **`name`** (string, required): Display name for the dimension (e.g., "Sample", "Batch")
- **`dimension`** (integer, required): Which array dimension to slice (0-indexed)
- **`default`** (integer, default: 0): Initial index to display
- **`show_navigation_menu`** (boolean, default: true): Whether to show < > navigation buttons

## Implementation Details

### 1. State Initialization
When a table section is first rendered, `_render_table_section()` checks for `data_slicing` config and initializes:
- Creates `table_slices` dictionary in analysis_data if needed
- Sets up indices from default values
- Stores references to outputs and config

### 2. Data Extraction
Before displaying the table:
1. Check if `data_slicing` is configured
2. Extract current indices from slice state
3. Call `_extract_sliced_data()` to get the slice
4. Update `slice_info` in config for display

### 3. Navigation Control Creation
When rendering table with slicing enabled:
1. Create `nav_frame` for controls
2. For each dimension in `data_slicing`:
   - Create axis_frame with label and buttons
   - Get max index from data shape
   - Get current index from slice state
   - Wire buttons to `_on_table_navigate_slice()` callbacks

### 4. Refresh Mechanism
When user clicks navigation buttons:
1. `_on_table_navigate_slice()` updates indices
2. Updates UI labels (index_label and full_label)
3. Calls `_refresh_table()` to re-render
4. `_refresh_table()` clears and re-renders entire analysis page

## File Changes

### main_gui.py
- Added `_extract_sliced_data()` method (line ~2256)
- Added `_create_table_navigation_controls()` method (line ~2283)
- Added `_on_table_navigate_slice()` method (line ~2362)
- Updated `_render_table_section()` to initialize table_slices and add navigation controls (line ~2410)
- Updated `_refresh_table()` to properly re-render table with slicing (line ~2730)
- Fixed existing bugs in graph slicing code:
  - Fixed undefined `indices` → `base_indices` (line 3145)
  - Fixed undefined `current_index` (line 3274)

### gui_configs/en/load_data_config.json
- Added example table with data slicing to "Data Overview" page (line ~197)
- Example shows single-dimension slicing through sample data

### Documentation/ANALYSIS_CONFIGURATION.md
- Expanded "Table Configuration" section with data_slicing examples
- Added new subsection "Data Slicing Configuration"
- Provided single and multi-dimensional usage examples

## Usage Example

1. **In your function's JSON config**, add a table section with data_slicing:
```json
{
  "type": "table",
  "config": {
    "data_source": "output_array",
    "title": "Sliced Output",
    "data_slicing": [
      {"name": "Sample", "dimension": 0, "default": 0, "show_navigation_menu": true}
    ]
  }
}
```

2. **In the GUI**, the Analysis tab will show:
   - Navigation controls with Previous/Next buttons
   - Current index display (e.g., "Sample: 3/50")
   - Table data for the current slice
   - All other table features (export, statistics, etc.)

3. **User interaction**:
   - Click < to go to previous slice
   - Click > to go to next slice
   - Index updates automatically in UI and data
   - Table re-renders with new slice

## Compatibility

### Backward Compatibility
- Tables without `data_slicing` config work exactly as before
- No changes to existing table rendering for non-sliced tables
- All existing table configuration properties still work

### Forward Compatibility
- Supports both old and new `data_slicing` format (string vs dict)
- Can mix sliced and non-sliced tables on same page
- Works with all conditional rendering features

## Testing

The implementation has been tested with:
1. Single-dimension slicing (2D arrays)
2. Multi-dimension slicing (3D+ arrays)
3. Edge cases (first/last indices)
4. Mixed table types on same page
5. Configuration with various defaults

Example test in load_data_config.json shows:
- Slicing 3D spectroscopic data by sample
- Default slice of 0 (first sample)
- Display of 30 rows × 15 columns
- 4 decimal places for floating point

## Troubleshooting

### Navigation buttons not appearing
- Check `show_navigation_menu` is set to `true`
- Verify `data_slicing` array is not empty

### Incorrect data dimensions
- Verify `dimension` index matches data shape
- Check that data source contains multi-dimensional array

### Index out of bounds
- Implementation automatically clamps to valid range
- Check data.shape to verify expected dimensions

## Future Enhancements

Potential improvements:
1. Slider control as alternative to buttons
2. Jump-to-index input field
3. Multiple simultaneous slices (column filtering)
4. Row filtering/searching in sliced view
5. Slice synchronization across multiple tables
