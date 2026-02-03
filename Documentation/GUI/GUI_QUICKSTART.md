# Chemometric Studio GUI - Quick Start Guide

## Installation

1. Ensure Python 3.8+ is installed
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   This installs: numpy, scipy, scikit-learn

3. No additional GUI packages needed (tkinter is built-in)

## Starting the Application

```bash
# From the project root directory
python launcher.py
```

Or directly:
```bash
python main_gui.py
```

The GUI window will open showing:
- **Left Panel**: Available Functions (organized by category) and Methodology (selected functions)
- **Right Panel**: Configuration workspace with tabs for Setup, Routing, Analysis, and Report

## Workflow

### Step 1: Load Data
1. Click **"Load Data"** in the Functions panel
   - This adds it to the Methodology list on the left
2. Select **"Load Data"** in the Methodology list
3. In the **Setup tab** on the right:
   - **Data Format**: Select "comma", "tabs", or "spaces"
   - **X Data Files**: Click "Browse" to select your data files
   - **Data Dimensionality**: Choose "1", "2", or "3"
   - **Y Data File** (optional): Select Y response data
   - **Variable/Sample Labels** (optional): Select label files
   - **Transpose Data**: Check if data needs transposing
4. Configuration saves automatically
   - **Hover over the ℹ icon** next to parameter names for help text

### Step 2: Create Validation Set (Optional)
1. Click **"Create Validation Set"** in Functions panel
2. Select in Methodology and configure in Setup tab:
   - **Create Validation Set**: Toggle on/off
   - **Creation Method**: Choose random, kennard_stone, or file
   - **Calibration Proportion**: Set 0.0-1.0 (e.g., 0.7)
   - **Selection File**: If using "file" method, select the file

### Step 3: Preprocess Data
Choose preprocessing functions as needed:

#### Baseline Correction
1. Click **"Baseline Correction"** in Functions
2. Configure in Setup tab:
   - **Method**: msc, svn, or moving_average
   - **Window Size**: For moving average (e.g., 5)

#### Smoothing
1. Click **"Smoothing"** in Functions
2. Configure in Setup tab:
   - **Method**: moving_average or savitzky_golay
   - **Window Size**: e.g., 5-9
   - **Polynomial Order**: For Savitzky-Golay (only for Savitzky-Golay method)

#### Center & Normalize
1. Click **"Center & Normalize"** in Functions
2. Configure in Setup tab:
   - **Center**: Toggle on/off
   - **Normalize**: Toggle on/off
   - **Direction**: axis=0 or axis=1

### Step 4: Perform Calibration
1. Click **"Univariate Calibration"** in Functions
2. Configure in Setup tab:
   - **Polynomial Degree**: e.g., 1 for linear
   - **Include Intercept**: Toggle on/off

### Step 5: Define Data Routing
1. Go to **Routing tab**
2. Use the dropdowns to select:
   - **Output**: Function and parameter from a previous step
   - **Input**: Function and parameter for a later step
3. Click **"Add Connection"** to establish the data flow
4. Connections appear in the list below with visual indicators
5. **Click a connection button twice** to remove it
6. Connections are color-coded:
   - **Blue buttons** on the left = outputs (data flowing out)
   - **Red buttons** on the right = inputs (data flowing in)

### Step 6: Execute Pipeline
1. Click **"Analysis"** button (top right) to run your pipeline
2. The pipeline will:
   - Generate `functions.txt` with your function calls
   - Generate `routing.txt` with connections
   - Execute the analysis pipeline
   - Capture output to `model_log.txt`
3. Check `model_log.txt` for results and any errors

## Help System

**Hover over the ℹ icon** next to function titles and input parameters to see:
- **Function-level help**: Overview and description of the entire function
- **Input-level tooltips**: Detailed explanations for each parameter
- **Type information**: Data types and expected formats

**Conditional parameters**: Some inputs only appear when others are configured (e.g., "Polynomial Order" only appears when "Savitzky-Golay" is selected as the smoothing method)

## Key Features

### Function Organization
- **Multiple Instances**: Add the same function multiple times (appears as "Load Data", "Load Data #2", etc.)
- **Auto-Routing**: Connections are automatically created when adding functions (can be customized in Routing tab)
- **Drag to Reorder**: Click and hold on a function in Methodology to change its position
- **Quick Remove**: Click the × button to remove a function

### Configuration
- **Required Fields**: Marked with asterisk (*)
- **Auto-Save**: Values save when you click elsewhere
- **Defaults**: Most parameters have sensible defaults
- **File Paths**: Can be absolute or relative to the project directory
- **Dynamic Visibility**: Some inputs appear/disappear based on other settings

### Analysis Execution
- **Sequential Execution**: Functions run in order from top to bottom
- **Error Handling**: Detailed error messages in `model_log.txt`
- **Progress Tracking**: Check terminal for execution status

## Common Workflows

### Simple Univariate Calibration
```
Load Data → Baseline Correction → Smoothing → 
Center & Normalize → Univariate Calibration
```

### With Validation Set
```
Load Data → Create Validation Set → Baseline Correction → 
Smoothing → Center & Normalize → Univariate Calibration
```

### Multiple Preprocessing Pipelines
```
Load Data (×1)
├→ Baseline Correction → Smoothing → Univariate Calibration
└→ Center & Normalize → Univariate Calibration (×2)
```

## Output Files

After running a pipeline, the following files are generated:

### `functions.txt`
Lists all functions and parameters in execution order.

### `routing.txt`
Defines how data flows between functions.

### `model_log.txt`
Complete execution log including:
- Calibration metrics (R², RMSE, etc.)
- Data shapes and summaries
- Error messages (if any)
- Full traceback (if errors occur)

## Troubleshooting

### GUI Doesn't Start
```bash
# Check Python version (should be 3.8+)
python --version

# Verify tkinter works
python -m tkinter
```

### Functions Not Appearing
- Verify `function_specs.json` exists in project root
- Check that `gui_configs/*.json` files exist for each function
- Look for error messages in the terminal

### Parameters Don't Show Up
- Some parameters are conditional (e.g., Polynomial Order only appears for Savitzky-Golay)
- Hover over the ℹ icon for help text and visibility rules
- Check `model_log.txt` for validation errors

### Model Won't Run
- Check that all required parameters are configured (marked with *)
- Verify data files exist at specified paths
- Check `model_log.txt` for detailed error information
- Ensure file formats match expectations (CSV, tab-separated, etc.)

## Advanced Usage

### Adding New Functions
1. Create function in `chemometrics/` module
2. Add entries to `function_specs.json`:
   - `return_specs`: Output names
   - `input_specs`: Input parameter names
   - `import_map`: Module and function location
   - `gui_listing`: Display info and config path
3. Create `gui_configs/function_name_config.json` with widget layout
4. Functions automatically appear in GUI

### Modifying Function Widgets
Edit `gui_configs/*.json` files to change:
- Widget types (entry, combobox, checkbutton, file_selector)
- Available options and defaults
- Help text and tooltips
- Conditional visibility rules

## Need More Help?

- **GUI Features**: See [GUI_DOCUMENTATION.md](GUI_DOCUMENTATION.md)
- **Data Routing**: See [README_ROUTING.md](README_ROUTING.md)
- **Visual Examples**: See [ROUTING_VISUAL_GUIDE.md](ROUTING_VISUAL_GUIDE.md)
- **Execution Details**: Check `model_log.txt` after running
