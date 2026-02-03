# Analysis Data JSON Schema

This document defines the structure for storing analysis configuration and results in model.json under the `"analysis"` key.

## Root Analysis Structure

```json
{
  "analysis": {
    "function_instance_alias": {
      "pages": [...],
      "current_page": 0,
      "execution_results": {...}
    }
  }
}
```

## Page Structure

Each page contains layout information and sections.

```json
{
  "pages": [
    {
      "title": "Page Title",
      "layout": "fp",
      "sections": [
        { "section_config": {} }
      ]
    }
  ]
}
```

### Layout Types

- **fp** - Full Page (1 section covering entire space)
- **ns** - North-South (2 sections: top covers full width, bottom covers full width)
- **ew** - East-West (2 sections: left covers full height, right covers full height)
- **fd** - Four Divisions (4 sections in 2x2 grid)
- **sd** - South Division (3 sections: top 2 side-by-side, bottom 2 side-by-side)
- **nd** - North Division (3 sections: top 2 side-by-side, bottom 2 side-by-side)
- **ed** - East Division (3 sections: left 2 stacked, right 2 side-by-side)
- **wd** - West Division (3 sections: left 2 side-by-side, right 2 stacked)

## Section Types and Configurations

### Empty Section
```json
{
  "type": null
}
```

### Graph Section
```json
{
  "type": "graph",
  "config": {
    "title": "Graph Title",
    "graph_type": "scatter|line|bar|heatmap|surface|etc",
    "x_axis": {
      "data_source": "output_key",
      "index": 0,
      "label": "X Axis Label"
    },
    "y_axis": {
      "data_source": "output_key",
      "index": 0,
      "label": "Y Axis Label"
    },
    "z_axis": {
      "data_source": "output_key",
      "index": 0,
      "label": "Z Axis Label"
    },
    "slice_info": {
      "dimension": 0,
      "index": 0,
      "description": "Description of current slice"
    },
    "navigation_axes": ["x", "y"]
  }
}
```

**Graph Types Examples:**
- **scatter** - 2D scatter plot (x vs y)
- **line** - Line plot (x vs y)
- **bar** - Bar chart
- **heatmap** - 2D heatmap
- **surface** - 3D surface plot
- **contour** - Contour plot
- **histogram** - Distribution histogram
- **box** - Box plot

**Data Sources:**
All data sources reference outputs from previous functions in the execution chain.

Example: `"X_cal"` references the X_cal output from a previous function.

**Index Fields:**
For matrix/tensor data, index specifies which slice to display:
- Index 0, 1, 2... for different columns/rows
- Can be array for multi-dimensional indexing: `[0, 1]` for matrix element at row 0, col 1

**Navigation Axes:**
Specifies which axes have navigation buttons:
- `"x"` - Add horizontal navigation buttons for x-axis
- `"y"` - Add vertical navigation buttons for y-axis
- `"z"` - Add depth navigation buttons (for 3D)

### Table Section
```json
{
  "type": "table",
  "config": {
    "title": "Table Title",
    "data_source": "output_key",
    "row_labels": "output_key_for_row_labels",
    "column_labels": "output_key_for_column_labels",
    "slice_info": {
      "dimension": 0,
      "index": 0,
      "description": "Description of current slice"
    },
    "decimal_places": 3,
    "column_width": 80
  }
}
```

## Execution Results Structure

When "Run to here" is executed, results are stored here:

```json
{
  "execution_results": {
    "status": "success|error|pending",
    "timestamp": "2024-01-29T12:00:00",
    "execution_time": 5.23,
    "error_message": null,
    "outputs": {
      "output_key_1": "numpy_array_data",
      "output_key_2": "data_value",
      "output_key_3": [...]
    }
  }
}
```

**Notes on outputs:**
- NumPy arrays are serialized using a custom encoder or stored as lists
- Complex data types (DataFrames, etc.) need special serialization
- References to output names match the function_specs.json return_specs

## Default Analysis Structure

Each function has a default analysis configuration in its GUI config file:

**File:** `gui_configs/[language]/[function_name]_config.json`

```json
{
  "default_analysis": {
    "pages": [
      {
        "title": "Summary",
        "layout": "fp",
        "sections": [
          {
            "type": "graph",
            "config": {
              "title": "Results Visualization",
              "graph_type": "scatter"
            }
          }
        ]
      }
    ]
  }
}
```

When a function is first analyzed, this default is loaded. User modifications override it.

## Complete Example

```json
{
  "analysis": {
    "load_data#1": {
      "pages": [
        {
          "title": "Data Overview",
          "layout": "ns",
          "sections": [
            {
              "type": "table",
              "config": {
                "title": "X Matrix (first 10 rows)",
                "data_source": "X_cal",
                "decimal_places": 2
              }
            },
            {
              "type": "graph",
              "config": {
                "title": "Data Distribution",
                "graph_type": "histogram",
                "x_axis": {
                  "data_source": "X_cal",
                  "index": 0
                }
              }
            },
            {
              "type": "graph",
              "config": {
                "title": "Sample Overview",
                "graph_type": "bar"
              }
            },
            {
              "type": null
            }
          ]
        },
        {
          "title": "Advanced",
          "layout": "fd",
          "sections": [
            {
              "type": "graph",
              "config": {
                "title": "Variable 1 vs 2",
                "graph_type": "scatter",
                "x_axis": { "data_source": "X_cal", "index": 0 },
                "y_axis": { "data_source": "X_cal", "index": 1 },
                "navigation_axes": ["x", "y"]
              }
            },
            { "type": null },
            { "type": null },
            { "type": null }
          ]
        }
      ],
      "current_page": 0,
      "execution_results": {
        "status": "success",
        "timestamp": "2024-01-29T12:00:00",
        "execution_time": 2.5,
        "outputs": {
          "X_cal": [[...data...]],
          "Y_cal": [[...data...]],
          "var_label": ["Var1", "Var2"],
          "smp_cal": ["S1", "S2"]
        }
      }
    }
  }
}
```

## Notes

1. **Data Serialization**: NumPy arrays and other non-JSON-serializable types need custom encoding/decoding
2. **Large Data**: For large arrays, consider storing only the displayed portion or references
3. **Caching**: Execution results should be cached to avoid re-running expensive operations
4. **Inheritance**: Sections inherit data from previous functions in the methodology chain
5. **Defaults**: Function GUI configs should include default analysis pages
