# Analysis Configuration Guide

The `analysis` section in the GUI configuration JSON defines how execution results are visualized and presented to users after running a function step.

## Table of Contents
1. [Basic Structure](#basic-structure)
2. [Layouts](#layouts)
3. [Pages](#pages)
4. [Sections](#sections)
5. [Graph Configuration](#graph-configuration)
6. [Table Configuration](#table-configuration)
7. [Navigation Axes](#navigation-axes)
8. [Conditional Rendering](#conditional-rendering)
9. [Examples](#examples)

---

## Basic Structure

```json
{
  "analysis": {
    "pages": [
      {
        "title": "Page Title",
        "layout": "fp",
        "sections": [
          { "type": "graph", "config": {...} },
          { "type": "table", "config": {...} }
        ]
      }
    ],
    "current_page": 0
  }
}
```

**Properties:**
- `pages` (array): List of analysis pages. Each page can have a different layout and content.
- `current_page` (integer): Index of the page to display by default (0-based).

---

## Layouts

Layouts define how sections are arranged on a page. Choose based on your visualization needs:

### `fp` - Full Page (1 section)
Single full-width section. Best for single visualizations.

```json
{
  "layout": "fp",
  "sections": [
    { "type": "graph", "config": {...} }
  ]
}
```

### `ew` - East-West (2 sections side-by-side)
Splits page horizontally. Left and right containers are resizable with a draggable sash.

```json
{
  "layout": "ew",
  "sections": [
    { "type": "graph", "config": {...} },
    { "type": "table", "config": {...} }
  ]
}
```

### `ns` - North-South (2 sections stacked)
Splits page vertically. Top and bottom containers.

```json
{
  "layout": "ns",
  "sections": [
    { "type": "graph", "config": {...} },
    { "type": "graph", "config": {...} }
  ]
}
```

### `fd` - Four Divided (2×2 grid)
Creates a 2×2 grid of sections.

```json
{
  "layout": "fd",
  "sections": [
    { "type": "graph", "config": {...} },
    { "type": "graph", "config": {...} },
    { "type": "table", "config": {...} },
    { "type": "table", "config": {...} }
  ]
}
```

### `sd` - South Divided (3 sections: 1 top, 2 bottom)
One large section on top, two smaller sections side-by-side on bottom. Uses nested paned windows for resizable division.

```json
{
  "layout": "sd",
  "sections": [
    { "type": "graph", "config": {...} },
    { "type": "table", "config": {...} },
    { "type": "table", "config": {...} }
  ]
}
```

### `nd` - North Divided (3 sections: 2 top, 1 bottom)
Two sections side-by-side on top, one large section on bottom. Inverse of `sd`.

```json
{
  "layout": "nd",
  "sections": [
    { "type": "graph", "config": {...} },
    { "type": "graph", "config": {...} },
    { "type": "table", "config": {...} }
  ]
}
```

### `wd` - West Divided (3 sections: 2 left stacked, 1 right)
Two sections stacked on left, one section on right. Uses nested paned windows.

```json
{
  "layout": "wd",
  "sections": [
    { "type": "graph", "config": {...} },
    { "type": "table", "config": {...} },
    { "type": "graph", "config": {...} }
  ]
}
```

### `ed` - East Divided (3 sections: 1 left, 2 right stacked)
One section on left, two sections stacked on right. Inverse of `wd`.

```json
{
  "layout": "ed",
  "sections": [
    { "type": "graph", "config": {...} },
    { "type": "table", "config": {...} },
    { "type": "graph", "config": {...} }
  ]
}
```

---

## Pages

Pages allow organizing analysis into multiple tabs. Users can switch between pages at the bottom of the analysis tab.

### Basic Page

```json
{
  "title": "Data Overview",
  "layout": "fp",
  "sections": [...]
}
```

### Page with Condition

Pages can be conditionally shown based on execution input parameters:

```json
{
  "title": "3D/Multiway Analysis",
  "layout": "ew",
  "condition": {
    "parameter": "nway_flag",
    "operator": ">",
    "value": 1
  },
  "sections": [...]
}
```

**Condition operators:**
- `==` - equals
- `!=` - not equals
- `>`, `<`, `>=`, `<=` - comparisons
- `in` - value is in a list/array
- `contains` - value contains the expected value

The condition evaluates against execution **inputs** (parameters used when running the function).

---

## Sections

Sections are containers within a page that hold either graphs or tables.

### Graph Section

```json
{
  "type": "graph",
  "config": {
    "graph_type": "scatter",
    "x_axis": {"data_source": "output_name", "label": "X Label"},
    "y_axis": {"data_source": "output_name", "label": "Y Label"},
    "title": "Graph Title",
    "navigation_axes": [...]
  }
}
```

### Table Section

```json
{
  "type": "table",
  "config": {
    "data_source": "output_name",
    "decimal_places": 4,
    "max_rows": 50,
    "max_cols": 15,
    "title": "Table Title"
  }
}
```

### Section with Condition

Sections can also have conditions to conditionally render based on inputs:

```json
{
  "type": "graph",
  "condition": {
    "parameter": "data_type",
    "operator": "==",
    "value": "spectroscopic"
  },
  "config": {...}
}
```

---

## Graph Configuration

### Scatter Plot

```json
{
  "graph_type": "scatter",
  "x_axis": {
    "data_source": "X_data",
    "label": "Variables",
    "index": 0
  },
  "y_axis": {
    "data_source": "Y_data",
    "label": "Values",
    "default_column": 0
  },
  "title": "Scatter Plot",
  "navigation_axes": [
    {"name": "Sample", "dimension": 0}
  ]
}
```

### Line Plot

```json
{
  "graph_type": "line",
  "x_axis": {
    "data_source": "axis_n_info",
    "label": "Variables"
  },
  "y_axis": {
    "data_source": "spectral_data",
    "label": "Intensity"
  },
  "marker": "o",
  "show_legend": false,
  "title": "Spectral Data",
  "navigation_axes": [
    {"name": "Sample", "dimension": 0, "axis": "y"}
  ]
}
```

**Line-specific options:**
- `marker` (string, optional): Marker style ('o', 's', '^', etc.). Omit for line-only plot.
- `show_legend` (boolean, default: false): Show legend for multi-row matrices.

### Bar Plot

```json
{
  "graph_type": "bar",
  "x_axis": {
    "data_source": "categories",
    "label": "Category"
  },
  "y_axis": {
    "data_source": "values",
    "label": "Count"
  },
  "title": "Bar Chart"
}
```

### Histogram

```json
{
  "graph_type": "histogram",
  "y_axis": {
    "data_source": "data",
    "label": "Distribution"
  },
  "title": "Histogram"
}
```

### Heatmap

```json
{
  "graph_type": "heatmap",
  "y_axis": {
    "data_source": "matrix_data",
    "label": "Heatmap"
  },
  "title": "Heatmap Visualization"
}
```

### Contour Plot

```json
{
  "graph_type": "contour",
  "x_axis": {
    "data_source": "x_values",
    "label": "X"
  },
  "y_axis": {
    "data_source": "y_values",
    "label": "Y"
  },
  "z_axis": {
    "data_source": "z_values",
    "label": "Z"
  },
  "title": "Contour Plot"
}
```

### Axis Configuration

Each axis (x, y, z) has its own configuration:

```json
"x_axis": {
  "data_source": "output_key",      // Required: key in execution outputs
  "label": "X Axis Label",          // Required: label displayed on axis
  "index": 0,                       // Optional: static index for multi-dim data
  "default_column": 0               // Optional: default column for axis selection
}
```

**Special data sources:**
- `axis_n_info` - Axis labels/coordinates (automatically extracted from loaded data)
- Any key in the execution outputs

---

## Navigation Axes

Navigation axes allow users to slice through multi-dimensional data using arrow buttons.

### Dimension Slicing

Slice along a dimension (e.g., select different samples):

```json
"navigation_axes": [
  {
    "name": "Sample",
    "dimension": 0,
    "default": 0
  }
]
```

### Axis Selection

Switch between different columns/variables on a specific plot axis:

```json
"navigation_axes": [
  {
    "name": "X Variable",
    "dimension": 1,
    "axis": "x",
    "default": 0
  },
  {
    "name": "Y Variable",
    "dimension": 1,
    "axis": "y",
    "default": 1
  }
]
```

**Navigation properties:**
- `name` (string, required): Label displayed in UI ("Sample", "Column", etc.)
- `dimension` (integer, required): Which dimension to navigate (0=rows, 1=columns, etc.)
- `axis` (string, optional): Which plot axis affected ('x', 'y', 'z'). Omit for dimension slicing.
- `default` (integer, optional): Default starting index (0-based). Validated against bounds.

---

## Table Configuration

```json
{
  "type": "table",
  "config": {
    "data_source": "output_key",
    "title": "Data Table",
    "decimal_places": 4,
    "max_rows": 100,
    "max_cols": 20
  }
}
```

**Properties:**
- `data_source` (string, required): Key in execution outputs to display
- `title` (string, optional): Title shown above table
- `decimal_places` (integer, default: 2): Decimal precision for floating-point numbers
- `max_rows` (integer, default: 50): Maximum rows to display before scrolling
- `max_cols` (integer, default: 15): Maximum columns to display before scrolling

---

## Conditional Rendering

Both pages and sections support conditional rendering based on execution inputs.

### Example: Show 3D Analysis Only for Multiway Data

```json
{
  "pages": [
    {
      "title": "2D Analysis",
      "layout": "ew",
      "condition": {
        "parameter": "nway_flag",
        "operator": "==",
        "value": 1
      },
      "sections": [...]
    },
    {
      "title": "3D/Multiway Analysis",
      "layout": "fd",
      "condition": {
        "parameter": "nway_flag",
        "operator": ">",
        "value": 1
      },
      "sections": [...]
    }
  ]
}
```

### Example: Conditional Section Display

```json
{
  "sections": [
    {
      "type": "graph",
      "condition": {
        "parameter": "data_type",
        "operator": "in",
        "value": ["spectroscopic", "chromatographic"]
      },
      "config": {...}
    }
  ]
}
```

---

## Examples

### Example 1: Simple 2D Spectroscopy Analysis

```json
{
  "analysis": {
    "pages": [
      {
        "title": "Spectral Analysis",
        "layout": "ew",
        "sections": [
          {
            "type": "graph",
            "config": {
              "graph_type": "line",
              "x_axis": {
                "data_source": "axis_n_info",
                "label": "Wavenumber (cm⁻¹)"
              },
              "y_axis": {
                "data_source": "X_cal",
                "label": "Absorbance"
              },
              "title": "Sample Spectra",
              "marker": null,
              "navigation_axes": [
                {
                  "name": "Sample",
                  "dimension": 0,
                  "default": 0
                }
              ]
            }
          },
          {
            "type": "table",
            "config": {
              "data_source": "X_cal",
              "title": "Spectral Data",
              "decimal_places": 4,
              "max_rows": 100,
              "max_cols": 20
            }
          }
        ]
      }
    ],
    "current_page": 0
  }
}
```

### Example 2: Multi-dimensional with Conditional Pages

```json
{
  "analysis": {
    "pages": [
      {
        "title": "2D Analysis",
        "layout": "fp",
        "condition": {
          "parameter": "nway_flag",
          "operator": "==",
          "value": 1
        },
        "sections": [
          {
            "type": "graph",
            "config": {
              "graph_type": "scatter",
              "x_axis": {"data_source": "PC1", "label": "PC1"},
              "y_axis": {"data_source": "PC2", "label": "PC2"},
              "title": "Score Plot"
            }
          }
        ]
      },
      {
        "title": "3D Analysis",
        "layout": "ew",
        "condition": {
          "parameter": "nway_flag",
          "operator": ">",
          "value": 1
        },
        "sections": [
          {
            "type": "graph",
            "config": {
              "graph_type": "scatter",
              "x_axis": {"data_source": "PC1", "label": "PC1"},
              "y_axis": {"data_source": "PC2", "label": "PC2"},
              "z_axis": {"data_source": "PC3", "label": "PC3"},
              "title": "3D Score Plot"
            }
          },
          {
            "type": "graph",
            "config": {
              "graph_type": "line",
              "x_axis": {"data_source": "loadings_x", "label": "Variable"},
              "y_axis": {"data_source": "loadings", "label": "Loading"},
              "title": "Loading Vectors",
              "navigation_axes": [
                {"name": "PC", "dimension": 0, "default": 0}
              ]
            }
          }
        ]
      }
    ],
    "current_page": 0
  }
}
```

### Example 3: Complex Layout with Multiple Graph Types

```json
{
  "analysis": {
    "pages": [
      {
        "title": "Comprehensive Analysis",
        "layout": "sd",
        "sections": [
          {
            "type": "heatmap",
            "config": {
              "graph_type": "heatmap",
              "y_axis": {
                "data_source": "correlation_matrix",
                "label": "Correlation"
              },
              "title": "Correlation Heatmap"
            }
          },
          {
            "type": "graph",
            "config": {
              "graph_type": "histogram",
              "y_axis": {
                "data_source": "residuals",
                "label": "Residuals"
              },
              "title": "Residual Distribution"
            }
          },
          {
            "type": "table",
            "config": {
              "data_source": "statistics",
              "title": "Model Statistics",
              "decimal_places": 6
            }
          }
        ]
      }
    ],
    "current_page": 0
  }
}
```

---

## Tips and Best Practices

1. **Data Sources**: Always reference outputs from your function's `output_aliases` or the actual output keys.

2. **Multi-row Data**: When plotting line graphs with matrix data (multiple rows), each row is plotted as a separate line. Use `show_legend: true` to identify them.

3. **Axis Selection vs Slicing**: 
   - Use `"axis"` field for choosing different columns to plot
   - Omit `"axis"` field to slice through samples/rows

4. **Navigation Limits**: Navigation bounds are automatically determined from data shape. Default values are validated against bounds.

5. **Conditional Logic**: Conditions evaluate against execution **inputs**, not outputs. Use input parameters to decide layout.

6. **Layout Responsiveness**: All layouts are responsive and scale with the window. Paned windows (ew, wd, ed) are resizable.

7. **Special Data**: The `axis_n_info` output is automatically generated from loaded data and should be used for axis labels.

---

## Troubleshooting

**"No data available" message**
- Verify `data_source` keys match exactly with execution outputs
- Ensure function execution completed successfully

**Graph appears empty**
- Check that axis data sources exist in outputs
- Verify x and y data have compatible dimensions
- Look at model_log.txt for execution errors

**Navigation doesn't work**
- Ensure `dimension` matches actual data dimensions
- Check that `default` values are within bounds
- Verify navigation_axes are defined for the right axis

**Layout sections misaligned**
- Verify you have the right number of sections for the layout type
- Empty sections will show a placeholder label

