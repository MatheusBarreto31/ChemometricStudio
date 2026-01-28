# Chemometric Studio

A comprehensive Tkinter-based GUI application for chemometrics analysis, data preprocessing, and calibration modeling.

## Features

- **Interactive Pipeline Builder**: Function selection with automatic parameter routing
- **Visual Data Flow**: Configure connections between functions with live visualization
- **Rich Parameter Configuration**: JSON-based function configs with tooltips, help text, and conditional visibility
- **Multi-format Data Support**: Load CSV, tab-separated, space-separated data
- **Preprocessing Functions**: Baseline correction, smoothing, center & normalize
- **Calibration Tools**: Univariate and multivariate calibration methods
- **Validation Support**: Automatic or manual validation set creation
- **Help System**: Integrated tooltips and detailed help for all functions

## Install

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Tkinter: On Windows, Tkinter is included with Python installers. On macOS/Linux, install via package manager or use Python from python.org.

## Quick Start

```bash
python launcher.py
```

Or directly:
```bash
python main_gui.py
```

For detailed usage instructions, see [Documentation](Documentation/).

## Project Structure

```
ChemometricsTool/
├── chemometrics/              # Core Python package
│   ├── __init__.py
│   ├── data_input.py
│   ├── data_processing.py
│   ├── reporting.py
│   ├── univ_calibration.py
│   └── validation_data_input.py
├── gui_configs/               # Function-specific GUI configurations (JSON)
├── Documentation/             # User-facing documentation
├── Generated Notes/           # Development notes and summaries
├── tests/                     # Test suite
├── main_gui.py               # Main GUI application
├── launcher.py               # Application launcher
├── function_specs.json       # Metadata registry for functions
├── requirements.txt          # Python dependencies
└── pyproject.toml           # Project metadata
```

## Documentation

- **[GUI Quick Start](Documentation/GUI_QUICKSTART.md)** — Get started in 5 minutes
- **[GUI Documentation](Documentation/GUI_DOCUMENTATION.md)** — Complete feature reference
- **[Routing Guide](Documentation/README_ROUTING.md)** — Data flow and connections
- **[Visual Guide](Documentation/ROUTING_VISUAL_GUIDE.md)** — Diagrams and workflows

## Requirements

- Python 3.8+
- tkinter (included with most Python distributions)
- Dependencies listed in [requirements.txt](requirements.txt):
  - numpy
  - scipy
  - scikit-learn

## Development

Tests are located in the `tests/` directory. Run tests with:

```bash
python -m pytest tests/
```

