# Chemometric Studio

A comprehensive GUI application for chemometric analysis. Chemometric Studio provides an intuitive interface for building data analysis pipelines with automatic parameter routing and visual workflow management.

As of now it is still in early development, with the vast majority of its intended features missing.

## Features

- **Interactive Pipeline Builder**: Modular function selection with automatic parameter routing between steps
- **Visual Data Flow**: Configure connections between analysis functions with live routing visualization
- **Rich Parameter Configuration**: JSON-based function configs with tooltips, help text, and conditional parameter visibility
- **Multi-format & Multi-dimensional Data Support**: Load CSV, tab-separated, space-separated data with support for N-way arrays
- **Preprocessing Functions**: Baseline correction, smoothing, center & normalize with multi-dimensional support
- **Validation Support**: Automatic or manual validation set creation with multiple selection strategies
- **Advanced Data Handling**: Multi-dimensional slicing, sample-wise operations, and flexible data reshaping
- **Help System**: Integrated tooltips and detailed help for all functions
- **Export Capabilities**: Generate reports with analysis results and visualizations

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

### Windows PowerShell (Execution Policy)

If activation is blocked in PowerShell, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

Then activate normally:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows Python 3.11 Setup

Install Python 3.11 with winget (side-by-side):

```powershell
winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
```

Create and populate a dedicated 3.11 environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You can also run the helper script:

```powershell
.\scripts\setup_windows_py311.ps1
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
cm-studio/
├── chemometrics/                   # Core Python package
│   ├── __init__.py
│   ├── data_input.py              # Data loading and parsing
│   ├── data_processing.py         # Preprocessing functions
│   ├── processing.py              # Additional processing utilities
│   ├── reporting.py               # Report generation
│   └── validation_data_input.py   # Validation set handling
├── gui_configs/                    # Function-specific GUI configurations (JSON)
├── languages/                      # Internationalization files
├── Settings/                       # User settings and configuration
├── Graphics/                       # Application icons and graphics
├── Documentation/                  # User-facing documentation
├── main_gui.py                     # Main GUI application
├── launcher.py                     # Application launcher
├── graph_renderer.py              # Graph rendering utilities
├── language_manager.py            # Language/localization management
├── routing_map_window.py          # Routing visualization window
├── analyst.py                     # Analysis utilities
├── user.py                        # User management
├── settings.py                    # Settings management
├── function_specs.json            # Metadata registry for functions
├── about_us.json                  # Application metadata
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
└── README.md                      # This file
```

## Documentation

- **[GUI Quick Start](Documentation/GUI_QUICKSTART.md)** — Get started in 5 minutes
- **[GUI Documentation](Documentation/GUI_DOCUMENTATION.md)** — Complete feature reference
- **[Routing Guide](Documentation/README_ROUTING.md)** — Data flow and connections
- **[Visual Guide](Documentation/ROUTING_VISUAL_GUIDE.md)** — Diagrams and workflows

## Metadata Notes

- `load_data` outputs `cal_metadata` with one entry per sample.
- `validation_data_main` outputs `cal_metadata` and `val_metadata`.
- `sample_index` in metadata is **1-based** (starts at 1), while internal NumPy indexing remains 0-based.
- Validation splitting preserves the original metadata keys and 1-based `sample_index` values.

## Requirements

- Python 3.11+
- tkinter (included with most Python distributions)
- Dependencies listed in [requirements.txt](requirements.txt):
  - numpy 2.3.5
  - scipy 1.15.0
  - scikit-learn 1.5.2
  - matplotlib 3.10.8
  - tensorly 0.9.0
  - pylatex 1.4.2
  - pandas 2.3.3

## Development

For detailed development and contribution guidelines, see the [Documentation](Documentation/) folder which contains:
- Implementation guides
- API references
- Architecture documentation

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

For end-user application use terms in distributed builds, see [EULA.md](EULA.md).

Third-party notices and bundled asset licenses are available in [Licenses/](Licenses/).

