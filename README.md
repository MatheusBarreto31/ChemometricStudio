# Chemometric Studio

Chemometric Studio is a desktop application for chemometric workflow design, execution, interpretation, and reporting.
It combines a visual methodology builder with configurable analysis pages, advanced routing, and PDF reporting.

## Project Status

- Active development. Open beta, still with missing features and known issues. You may encounter unfinished features, UI adjustments, or behavior changes in upcoming versions.
- Still in construction end-user documentation is available in the built-in User Manual under `Manual/` (index.html).

## What the Application Currently Supports

The current feature includes:

- Visual Setup tab for building methodology pipelines (function library, ordered task list, per-function parameter panels).
- Automatic output-to-input routing, with optional manual routing overrides in the Routing tab.
- Execution controls for full runs and partial runs (for example, run to a selected node).
- Analysis tab with method-specific pages/sections (plots, tables, text summaries, and per-section controls).
- Custom Analysis tab for composing a consolidated interpretation workspace from existing analysis sections.
- Report workflow for assembling report content and exporting PDF output.
- Config-driven GUI definitions (`gui_configs/`) with conditional visibility and dynamic section behavior.
- Data input/validation split for calibration and validation workflows.
- Workflow-control functions for loops, parallel branches, and ensembles.

## Function Coverage

- Data Input and Validation:
	- Load Data
	- Validation Data
- Preprocessing:
	- Baseline Correction
	- Smoothing
	- Center and Scale
	- Blank Subtraction
- Calibration and Modeling:
	- First-Order Calibration
	- N-PLS / U-PLS Multiway Regression
- Classification and Novelty Detection:
	- N-class methods (for example Logistic Regression, Random Forest, SVC, KNN, LDA, QDA, PLS-DA)
	- One-class/novelty methods (for example SIMCA, DD-SIMCA, One-Class SVM, Isolation Forest, LOF)
- Exploratory Analysis:
	- PCA Analysis
	- MCR-ALS Analysis
	- PARAFAC Analysis
- Workflow Control:
	- Loop Start/End
	- Parallel Start/Branch/End
	- Ensemble Start/Member/End
- Cross-Validation Configuration:
	- CV Configuration

## User GUI (Guided Use of Saved Models)

Chemometric Studio includes a simplified User GUI flow for guided operation of saved models.

- Source entry point: `python user.py`.
- Packaged app entry point: `ChemometricStudioUser` / `ChemometricStudioUser.exe`.

Saved model formats supported by Load Model:

- `.mdfd` (Full Model): packaged model with calibration/validation-related file paths and data artifacts.
- `.mdcd` (Model With Calibration): packaged model containing calibration context; user is prompted to provide validation data after loading.
- `.mdon` (Method Only): packaged method without embedded file paths; user is prompted to provide calibration and validation data.

Guided behavior in User GUI:

- Start screen focuses on loading a model, then running it.
- For `.mdcd` and `.mdon`, guided setup steps direct the user to required input functions before execution.
- After a successful run, the interface switches to simplified post-run controls (Analysis, Custom Analysis, Report, Save/Load, Run).
- Report action generates and opens PDF directly when report elements are available.

## Supported Runtime

- Python 3.11+
- Tkinter (bundled with standard Python installers on Windows)

## Installation (Source)

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks script execution, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

## Run

Standard GUI:

```bash
python launcher.py
```

Alternative direct entry point:

```bash
python main_gui.py
```

User GUI:

```bash
python user.py
```

## Dependency Set

Pinned in `requirements.txt`:

- numpy==2.4.2
- scipy==1.17.1
- scikit-learn==1.8.0
- matplotlib==3.10.8
- tensorly==0.9.0
- pylatex==1.4.2
- pandas==3.0.1
- pyMCR==0.5.1
- ddsimca==1.0.3
- prcv==1.2.1
- sv-ttk==2.6.1
- Pillow

## User Manual

The end-user manual is available in `Manual/`.
Useful starting pages:

- `Manual/index.html`
- `Manual/getting-started.html`
- `Manual/workflow-guide.html`
- `Manual/analysis-reporting.html`
- `Manual/functions-reference.html`

## Repository Layout

Top-level areas:

- `chemometrics/`: core analysis and processing modules.
- `app_services/`: orchestration and runtime service layers.
- `gui_configs/`: JSON-driven UI definitions for setup and analysis rendering.
- `Manual/`: user manual pages and media assets.
- `scripts/`: development helpers and utility scripts.
- `Graphics/`, `Fonts/`, `languages/`, `Settings/`: UI assets and configuration.

## License

Licensed under Apache-2.0. See `LICENSE`.

Additional distribution and third-party notices:

- `EULA.md`
- `NOTICE`
- `Licenses/`

