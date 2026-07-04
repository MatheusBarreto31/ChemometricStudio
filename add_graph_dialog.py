"""
Add Graph Dialog Module

Provides a comprehensive dialog for adding graphs to the analysis tab with:
- Graph type selection
- Data source configuration
- Axis configuration
- Data slicing/navigation
- Class labels and sample labels
- Preview functionality
- Add to section
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import copy
import json
from pathlib import Path
import platform
from typing import Optional, Dict, List, Tuple, Any
import graph_renderer
from dialog_data_source_utils import (
    append_prefixed_data_sources,
    collect_nested_key_paths,
    get_available_data_sources,
    get_data_source_value,
    get_nested_keys,
    resolve_execution_data_sources,
)
from app_services.analysis_render_service import (
    extract_axis_data,
    normalize_class_data_matrix,
    normalize_class_labels_for_plot,
)


def _set_window_icon(window, base_name: str = "Icon"):
    base_dir = Path(__file__).parent
    graphics_dir = base_dir / "Graphics"
    ico_path = graphics_dir / f"{base_name}.ico"
    png_path = graphics_dir / f"{base_name}.png"
    if not ico_path.exists():
        ico_path = base_dir / f"{base_name}.ico"
    if not png_path.exists():
        png_path = base_dir / f"{base_name}.png"

    if platform.system().lower() == "windows" and ico_path.exists():
        try:
            window.iconbitmap(str(ico_path))
            return
        except tk.TclError:
            pass

    if png_path.exists():
        try:
            icon_photo = tk.PhotoImage(file=str(png_path))
            window.iconphoto(True, icon_photo)
            window._icon_photo = icon_photo
            return
        except tk.TclError:
            pass

    if ico_path.exists():
        try:
            from PIL import Image, ImageTk

            icon_image = Image.open(ico_path)
            icon_photo = ImageTk.PhotoImage(icon_image)
            window.iconphoto(True, icon_photo)
            window._icon_photo = icon_photo
        except Exception:
            pass


class AddGraphDialog:
    """Dialog for adding a new graph to an empty section in the analysis tab."""
    
    def __init__(self, parent, main_gui, instance_alias: str):
        """Initialize the Add Graph dialog.
        
        Args:
            parent: Parent tkinter window
            main_gui: Reference to main ChemometricsGUI instance
            instance_alias: Alias of the current function instance
        """
        self.parent = parent
        self.main_gui = main_gui
        self.instance_alias = instance_alias
        
        # Get execution outputs
        self.outputs = self._get_execution_outputs()
        if not self.outputs:
            self._notify(self._t("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), level="error")
            return
        
        # Find empty sections
        self.empty_sections = self._find_empty_sections()
        if not self.empty_sections:
            self._notify(self._t("ui.messages.no_empty_sections", "No empty sections available. Add a new page or remove existing sections first."), level="warning")
            return
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        _set_window_icon(self.dialog, "Icon")
        self.dialog.title(self._t("ui.dialogs.add_graph", "Add Graph"))
        self.dialog.geometry("1000x552")
        self._center_window(self.dialog, 1000, 552)
        
        # Graph configuration
        self.graph_config = {
            'graph_type': 'scatter',
            'x_axis': {},
            'y_axis': {},
            'z_axis': {},
            'title': 'New Graph',
            'data_slicing': [],
            'legend_show_mode': 'auto',
            'legend_elements': {
                'datasets': True,
                'color': True,
                'marker': False,
                'fill': False,
                'edge': False,
            },
            'legend_position': 'auto',
            'legend_location': 'inside',
        }
        
        # Multi-dataset configuration
        self.datasets_configs = []
        
        # Preview variables
        self.preview_canvas = None
        self.preview_frame = None
        
        # Build UI
        self._build_ui()

    def _notify(self, message: str, level: str = "message"):
        """Show notification using main GUI fading notices, with fallback to messagebox."""
        if hasattr(self.main_gui, '_show_fading_notice'):
            self.main_gui._show_fading_notice(message, level=level)
            return
        if level == "error":
            messagebox.showerror("Error", message)
        elif level == "warning":
            messagebox.showwarning("Warning", message)
        else:
            messagebox.showinfo("Info", message)

    def _t(self, key: str, default: str) -> str:
        """Translate message key through main GUI language manager."""
        if hasattr(self.main_gui, 'language_manager'):
            return self.main_gui.language_manager.translate(key, default)
        return default

    def _normalize_force_integer_mode(self, value: Any, default: str = 'false') -> str:
        """Normalize force_integer UI/config value to false, true, or conditional."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'1', 'true', 'yes', 'on', 'always'}:
                return 'true'
            if normalized in {'conditional', 'auto'}:
                return 'conditional'
            if normalized in {'0', 'false', 'no', 'off', ''}:
                return 'false'
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        elif isinstance(value, (int, float)):
            return 'true' if bool(value) else 'false'

        return default if default in {'false', 'true', 'conditional'} else 'false'

    def _apply_force_integer_mode(self, axis_config: Dict[str, Any], mode_value: Any) -> None:
        """Persist force_integer mode to config while preserving old bool compatibility."""
        mode = self._normalize_force_integer_mode(mode_value)
        if mode == 'true':
            axis_config['force_integer'] = True
        elif mode == 'conditional':
            axis_config['force_integer'] = 'conditional'

    def _center_window(self, window, width: int, height: int):
        """Center a window on the screen with the given size."""
        window.update_idletasks()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        window.geometry(f"{width}x{height}+{x}+{y}")
    
    def _get_execution_outputs(self) -> Optional[Dict]:
        """Get execution data sources (inputs + outputs) from analysis data."""
        if self.instance_alias not in self.main_gui.analysis_data:
            return None
        execution_results = self.main_gui.analysis_data[self.instance_alias].get('execution_results', {})
        if execution_results.get('status') != 'success':
            return None

        runtime_snapshot = None
        if hasattr(self.main_gui, 'build_runtime_state_snapshot'):
            try:
                runtime_snapshot = self.main_gui.build_runtime_state_snapshot()
            except Exception:
                runtime_snapshot = None

        combined_sources = resolve_execution_data_sources(
            execution_results,
            instance_alias=self.instance_alias,
            runtime_snapshot=runtime_snapshot,
            main_gui=self.main_gui,
        )
        return combined_sources if isinstance(combined_sources, dict) else {}

    def _append_prefixed_data_sources(self, combined_sources: Dict[str, Any], execution_results: Dict[str, Any]) -> None:
        """Add explicit in./out. aliases while preserving unprefixed precedence behavior."""
        append_prefixed_data_sources(
            combined_sources,
            execution_results,
            main_gui=self.main_gui,
            instance_alias=self.instance_alias,
        )
    
    def _find_empty_sections(self) -> List[Tuple[int, int, str]]:
        """Find all empty sections in the current analysis pages.
        
        Returns:
            List of tuples: (page_idx, section_idx, description)
        """
        empty = []
        if self.instance_alias not in self.main_gui.analysis_data:
            return empty
        
        pages = self.main_gui.analysis_data[self.instance_alias].get('pages', [])
        for page_idx, page in enumerate(pages):
            page_title = page.get('title', f'Page {page_idx + 1}')
            sections = page.get('sections', [])
            for section_idx, section in enumerate(sections):
                if section.get('type') is None:
                    desc = f"Page '{page_title}' - Section {section_idx + 1}"
                    empty.append((page_idx, section_idx, desc))
        
        return empty
    
    def _get_available_data_sources(self) -> List[str]:
        """Get list of data sources from outputs."""
        return get_available_data_sources(self.outputs)

    def _get_data_source_value(self, data_source: str):
        """Resolve a data source with prefixed/unprefixed compatibility."""
        return get_data_source_value(self.outputs, data_source)
    
    def _get_nested_keys(self, data_source: str) -> List[str]:
        """Get nested key paths if data source is a dictionary."""
        return get_nested_keys(self.outputs, data_source)

    def _collect_nested_key_paths(self, data: dict, prefix: str = "") -> List[str]:
        """Collect nested dictionary paths in dot notation (e.g., 'a.b.c')."""
        return collect_nested_key_paths(data, prefix)

    def _resolve_nested_data(self, data, nested_key: str = None):
        """Resolve nested data using dot notation path."""
        if not nested_key:
            return data

        if isinstance(data, dict):
            current = data
            for part in nested_key.split('.'):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current

        return data
    
    def _get_data_shape_info(self, data_source: str, nested_key: str = None) -> str:
        """Get shape information for a data source."""
        if not data_source or data_source not in self.outputs:
            return "N/A"
        
        data = self.outputs[data_source]
        data = self._resolve_nested_data(data, nested_key)
        if data is None:
            return "Invalid nested key path"
        
        if isinstance(data, np.ndarray):
            return f"Shape: {data.shape}, Type: {data.dtype}"
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
                return f"List of {len(data)} arrays/lists"
            return f"List of {len(data)} items"
        elif isinstance(data, dict):
            return f"Dictionary with {len(data)} keys"
        else:
            return f"Type: {type(data).__name__}"
    
    def _build_ui(self):
        """Build the dialog UI."""
        # Main container with paned window for left (controls) and right (preview)
        paned = ttk.PanedWindow(self.dialog, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Configuration controls
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # Right panel: Preview
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Build left panel
        self._build_config_panel(left_frame)
        
        # Build right panel
        self._build_preview_panel(right_frame)
        
        # Bottom button bar
        self._build_button_bar()
    
    def _build_config_panel(self, parent):
        """Build the configuration panel."""
        # Create notebook for organized tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Basic Configuration
        basic_tab = ttk.Frame(notebook)
        notebook.add(basic_tab, text=self._t("ui.tabs.basic", "Basic"))
        self._build_basic_tab(basic_tab)
        
        # Tab 2: Axes Configuration
        axes_tab = ttk.Frame(notebook)
        notebook.add(axes_tab, text=self._t("ui.tabs.axes", "Axes"))
        self._build_axes_tab(axes_tab)
        
        # Tab 3: Multi-Dataset Configuration
        datasets_tab = ttk.Frame(notebook)
        notebook.add(datasets_tab, text=self._t("ui.tabs.multi_dataset", "Multi-Dataset"))
        self._build_datasets_tab(datasets_tab)
        
        # Tab 4: Advanced Options
        advanced_tab = ttk.Frame(notebook)
        notebook.add(advanced_tab, text=self._t("ui.tabs.advanced", "Advanced"))
        self._build_advanced_tab(advanced_tab)

    def _bind_canvas_mousewheel(self, canvas, scrollable_frame):
        """Enable mouse wheel scrolling when hovering scrollable content."""
        def _block_combobox_wheel(widget):
            if isinstance(widget, ttk.Combobox):
                if not getattr(widget, '_wheel_block_bound', False):
                    widget.bind("<MouseWheel>", lambda e: "break")
                    widget.bind("<Button-4>", lambda e: "break")
                    widget.bind("<Button-5>", lambda e: "break")
                    setattr(widget, '_wheel_block_bound', True)
            for child in widget.winfo_children():
                _block_combobox_wheel(child)

        def _on_mousewheel(event):
            if isinstance(event.widget, ttk.Combobox):
                return "break"

            step = 0
            if event.delta:
                step = int(-1 * (event.delta / 120))
            elif getattr(event, 'num', None) == 4:
                step = -1
            elif getattr(event, 'num', None) == 5:
                step = 1

            if step == 0:
                return "break"

            first, last = canvas.yview()
            if (step < 0 and first <= 0.0) or (step > 0 and last >= 1.0):
                return "break"

            canvas.yview_scroll(step, "units")
            return "break"

        def _bind_wheel(_event):
            _block_combobox_wheel(scrollable_frame)
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(_event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)
        scrollable_frame.bind("<Enter>", _bind_wheel)
        scrollable_frame.bind("<Leave>", _unbind_wheel)

    def _bind_canvas_scrollregion(self, canvas, scrollable_frame):
        """Keep canvas scroll region in sync and avoid stale offset when content is short."""
        def _update_scrollregion(_event=None):
            bbox = canvas.bbox("all")
            if bbox is None:
                return
            canvas.configure(scrollregion=bbox)

            content_height = bbox[3] - bbox[1]
            viewport_height = max(1, canvas.winfo_height())
            if content_height <= viewport_height:
                canvas.yview_moveto(0.0)

        scrollable_frame.bind("<Configure>", _update_scrollregion)
    
    def _build_basic_tab(self, parent):
        """Build basic configuration tab."""
        # Scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        self._bind_canvas_scrollregion(canvas, scrollable_frame)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._bind_canvas_mousewheel(canvas, scrollable_frame)
        
        # Section selection
        section_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.target_section", "Target Section"), padding=10)
        section_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(section_frame, text=self._t("ui.messages.select_empty_section", "Select empty section:")).pack(anchor=tk.W)
        self.section_var = tk.StringVar()
        section_combo = ttk.Combobox(section_frame, textvariable=self.section_var, state="readonly", width=40)
        section_combo['values'] = [desc for _, _, desc in self.empty_sections]
        if self.empty_sections:
            section_combo.current(0)
        section_combo.pack(fill=tk.X, pady=5)
        
        # Graph type selection
        type_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.graph_type", "Graph Type"), padding=10)
        type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        graph_types = [
            ('scatter', 'Scatter Plot (2D/3D)'),
            ('line', 'Line Plot'),
            ('bar', 'Bar Chart'),
            ('histogram', 'Histogram'),
            ('heatmap', 'Heatmap'),
            ('3d_surf', '3D Surface Plot'),
            ('contour', 'Contour Plot')
        ]
        
        self.graph_type_var = tk.StringVar(value='scatter')
        for gtype, desc in graph_types:
            rb = ttk.Radiobutton(type_frame, text=desc, variable=self.graph_type_var, 
                               value=gtype, command=self._on_graph_type_changed)
            rb.pack(anchor=tk.W, pady=2)
        
        # Graph title
        title_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.graph_title", "Graph Title"), padding=10)
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.title_var = tk.StringVar(value=self._t("ui.labels.new_graph", "New Graph"))
        title_entry = ttk.Entry(title_frame, textvariable=self.title_var, width=50)
        title_entry.pack(fill=tk.X)
    
    def _build_axes_tab(self, parent):
        """Build axes configuration tab."""
        # Scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        self._bind_canvas_scrollregion(canvas, scrollable_frame)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._bind_canvas_mousewheel(canvas, scrollable_frame)
        
        # X-Axis configuration
        self._build_axis_config(scrollable_frame, "X-Axis", 'x')
        
        # Y-Axis configuration
        self._build_axis_config(scrollable_frame, "Y-Axis", 'y')
        
        # Z-Axis configuration
        self._build_axis_config(scrollable_frame, "Z-Axis", 'z')
    
    def _build_axis_config(self, parent, label: str, axis_key: str):
        """Build configuration widgets for a single axis."""
        frame = ttk.LabelFrame(parent, text=label, padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Data source
        ttk.Label(frame, text=self._t("ui.labels.input_output_source", "Input/Output Source:")).grid(row=0, column=0, sticky=tk.W, pady=2)
        data_sources = [''] + self._get_available_data_sources() + ['__index__']
        
        axis_var = tk.StringVar()
        setattr(self, f'{axis_key}_data_source_var', axis_var)
        
        data_combo = ttk.Combobox(frame, textvariable=axis_var, values=data_sources, width=30)
        data_combo.grid(row=0, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        data_combo.bind('<<ComboboxSelected>>', lambda e: self._on_data_source_changed(axis_key))
        
        # Nested key (for dict sources)
        ttk.Label(frame, text=self._t("ui.labels.nested_key", "Nested Key:")).grid(row=1, column=0, sticky=tk.W, pady=2)
        nested_var = tk.StringVar()
        setattr(self, f'{axis_key}_nested_key_var', nested_var)
        
        nested_combo = ttk.Combobox(frame, textvariable=nested_var, width=30)
        nested_combo.grid(row=1, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        setattr(self, f'{axis_key}_nested_combo', nested_combo)
        
        # Index (for list/array sources) - removed as it's too simplistic
        # Users should use default column or axis navigation instead
        
        # Axis label
        ttk.Label(frame, text=self._t("ui.labels.axis_label", "Axis Label:")).grid(row=2, column=0, sticky=tk.W, pady=2)
        label_var = tk.StringVar(value=label.replace('-Axis', ''))
        setattr(self, f'{axis_key}_label_var', label_var)
        
        label_entry = ttk.Entry(frame, textvariable=label_var, width=30)
        label_entry.grid(row=2, column=1, sticky=tk.W+tk.E, pady=2, padx=5)

        # Axis type / scale
        ttk.Label(frame, text=self._t("ui.labels.axis_type", "Axis Type:")).grid(row=3, column=0, sticky=tk.W, pady=2)
        axis_type_var = tk.StringVar(value='linear')
        setattr(self, f'{axis_key}_axis_type_var', axis_type_var)
        axis_type_combo = ttk.Combobox(
            frame,
            textvariable=axis_type_var,
            values=['linear', 'log10', 'log2', 'ln'],
            width=27,
            state='readonly'
        )
        axis_type_combo.grid(row=3, column=1, sticky=tk.W+tk.E, pady=2, padx=5)

        # Force integer ticks mode
        force_integer_var = tk.StringVar(value='false')
        setattr(self, f'{axis_key}_force_integer_var', force_integer_var)
        ttk.Label(frame, text=self._t("ui.labels.force_integer_mode", "Force Integer Ticks:")).grid(row=4, column=0, sticky=tk.W, pady=2)
        force_integer_combo = ttk.Combobox(
            frame,
            textvariable=force_integer_var,
            values=['false', 'true', 'conditional'],
            state='readonly',
            width=27
        )
        force_integer_combo.grid(row=4, column=1, sticky=tk.W+tk.E, pady=2, padx=5)

        # Reverse axis direction
        reverse_axis_var = tk.BooleanVar(value=False)
        setattr(self, f'{axis_key}_reverse_axis_var', reverse_axis_var)
        reverse_axis_cb = ttk.Checkbutton(
            frame,
            text=self._t(
                "ui.labels.reverse_axis",
                "Reverse axis"
            ),
            variable=reverse_axis_var
        )
        reverse_axis_cb.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Optional reference settings used by __index__ and other derived axes
        ttk.Label(frame, text=self._t("ui.labels.reference_source_optional", "Reference Source (optional):")).grid(row=6, column=0, sticky=tk.W, pady=2)
        reference_source_var = tk.StringVar()
        setattr(self, f'{axis_key}_reference_source_var', reference_source_var)
        reference_source_combo = ttk.Combobox(
            frame,
            textvariable=reference_source_var,
            values=[''] + self._get_available_data_sources(),
            width=30
        )
        reference_source_combo.grid(row=6, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        reference_source_combo.bind('<<ComboboxSelected>>', lambda _e: self._on_reference_source_changed(axis_key))

        ttk.Label(frame, text=self._t("ui.labels.reference_nested_key_optional", "Reference Nested Key (optional):")).grid(row=7, column=0, sticky=tk.W, pady=2)
        reference_nested_var = tk.StringVar()
        setattr(self, f'{axis_key}_reference_nested_key_var', reference_nested_var)
        reference_nested_combo = ttk.Combobox(frame, textvariable=reference_nested_var, width=30)
        reference_nested_combo.grid(row=7, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        setattr(self, f'{axis_key}_reference_nested_combo', reference_nested_combo)

        ttk.Label(frame, text=self._t("ui.labels.reference_axis_optional", "Reference Axis (optional):")).grid(row=8, column=0, sticky=tk.W, pady=2)
        reference_axis_var = tk.StringVar()
        setattr(self, f'{axis_key}_reference_axis_var', reference_axis_var)
        reference_axis_combo = ttk.Combobox(
            frame,
            textvariable=reference_axis_var,
            values=['', 'first', 'last', '0', '1', '-1'],
            width=30
        )
        reference_axis_combo.grid(row=8, column=1, sticky=tk.W+tk.E, pady=2, padx=5)

        treat_list_var = tk.BooleanVar(value=False)
        setattr(self, f'{axis_key}_treat_as_list_source_var', treat_list_var)
        ttk.Checkbutton(
            frame,
            text=self._t("ui.labels.treat_as_list_source", "Treat nested list as axis-list source"),
            variable=treat_list_var
        ).grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Dynamic axis labels from a data variable (for navigation-synced labels)
        ttk.Label(frame, text=self._t("ui.labels.axis_labels_source", "Dynamic Label Source (optional):")).grid(row=11, column=0, sticky=tk.W, pady=2)
        axis_labels_var = tk.StringVar()
        setattr(self, f'{axis_key}_axis_labels_var', axis_labels_var)
        axis_labels_combo = ttk.Combobox(frame, textvariable=axis_labels_var,
                                         values=[''] + self._get_available_data_sources(), width=30)
        axis_labels_combo.grid(row=11, column=1, sticky=tk.W+tk.E, pady=2, padx=5)

        ttk.Label(frame, text=self._t("ui.labels.axis_labels_dimension", "Label Dimension (for slicing):")).grid(row=12, column=0, sticky=tk.W, pady=2)
        axis_labels_dim_var = tk.StringVar()
        setattr(self, f'{axis_key}_axis_labels_dimension_var', axis_labels_dim_var)
        ttk.Entry(frame, textvariable=axis_labels_dim_var, width=10).grid(row=12, column=1, sticky=tk.W, pady=2, padx=5)

        ttk.Label(frame, text=self._t("ui.labels.label_template", "Label Template (optional):")).grid(row=13, column=0, sticky=tk.W, pady=2)
        label_template_var = tk.StringVar()
        setattr(self, f'{axis_key}_label_template_var', label_template_var)
        ttk.Entry(frame, textvariable=label_template_var, width=30).grid(row=13, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        ttk.Label(
            frame,
            text=self._t("ui.messages.label_template_hint", "Use {label}, {nav_idx} (1-based), {nav_idx0} (0-based)"),
            foreground="gray", font=("Arial", 8)
        ).grid(row=14, column=0, columnspan=2, sticky=tk.W, padx=2)
        
        # Data shape info
        info_var = tk.StringVar(value=self._t("ui.messages.select_data_source", "Select a data source"))
        setattr(self, f'{axis_key}_info_var', info_var)
        
        info_label = ttk.Label(frame, textvariable=info_var, foreground="gray", font=("Arial", 8))
        info_label.grid(row=10, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        frame.columnconfigure(1, weight=1)
    
    def _build_datasets_tab(self, parent):
        """Build multi-dataset configuration tab."""
        # Scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        self._bind_canvas_scrollregion(canvas, scrollable_frame)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._bind_canvas_mousewheel(canvas, scrollable_frame)
        
        # Info label
        info_frame = ttk.Frame(scrollable_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(info_frame, text=self._t("ui.labels.multi_dataset_plots", "Multi-Dataset Plots"), font=("Arial", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(info_frame, 
                 text=self._t("ui.messages.configure_multiple_datasets", "Configure multiple datasets to plot on the same graph (e.g., calibration vs validation)."),
                 wraplength=450, foreground="gray").pack(anchor=tk.W, pady=(2, 5))
        
        ttk.Label(info_frame,
                 text=self._t("ui.messages.datasets_override_single_axes", "Note: If you configure datasets here, the single-dataset axes (Axes tab) will be ignored."),
                 wraplength=450, font=("Arial", 9, "italic"), foreground="orange").pack(anchor=tk.W)
        
        # Datasets list frame
        list_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.datasets", "Datasets"), padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Listbox to show datasets
        listbox_container = ttk.Frame(list_frame)
        listbox_container.pack(fill=tk.BOTH, expand=True)
        
        self.datasets_listbox = tk.Listbox(listbox_container, height=8)
        self.datasets_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        list_scrollbar = ttk.Scrollbar(listbox_container, orient=tk.VERTICAL, 
                                       command=self.datasets_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.datasets_listbox.config(yscrollcommand=list_scrollbar.set)
        
        # Buttons for dataset management
        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="➕ " + self._t("ui.buttons.add_dataset", "Add Dataset"), 
                  command=self._add_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="✏ " + self._t("ui.buttons.edit_selected", "Edit Selected"), 
                  command=self._edit_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🗑 " + self._t("ui.buttons.remove_selected", "Remove Selected"), 
                  command=self._remove_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text=self._t("ui.buttons.clear_all", "Clear All"), 
                  command=self._clear_datasets).pack(side=tk.LEFT, padx=2)
    
    def _add_dataset(self):
        """Add a new dataset configuration."""
        self._show_dataset_config_dialog()
    
    def _edit_dataset(self):
        """Edit the selected dataset."""
        selection = self.datasets_listbox.curselection()
        if not selection:
            self._notify(self._t("ui.messages.select_dataset_edit", "Please select a dataset to edit"), level="warning")
            return
        
        idx = selection[0]
        existing_config = self.datasets_configs[idx] if idx < len(self.datasets_configs) else None
        self._show_dataset_config_dialog(idx, existing_config)
    
    def _remove_dataset(self):
        """Remove the selected dataset."""
        selection = self.datasets_listbox.curselection()
        if not selection:
            self._notify(self._t("ui.messages.select_dataset_remove", "Please select a dataset to remove"), level="warning")
            return
        
        idx = selection[0]
        if idx < len(self.datasets_configs):
            del self.datasets_configs[idx]
            self._refresh_datasets_list()
    
    def _clear_datasets(self):
        """Clear all datasets."""
        if self.datasets_configs and messagebox.askyesno(
            self._t("ui.dialogs.confirm", "Confirm"),
            self._t("ui.messages.clear_all_datasets_confirm", "Clear all datasets?")
        ):
            self.datasets_configs = []
            self._refresh_datasets_list()
    
    def _show_dataset_config_dialog(self, edit_idx=None, existing_config=None):
        """Show dialog to configure a single dataset."""
        dialog = tk.Toplevel(self.dialog)
        _set_window_icon(dialog, "Icon")
        dialog.title(
            self._t("ui.dialogs.configure_dataset", "Configure Dataset")
            if edit_idx is None else
            self._t("ui.dialogs.edit_dataset", "Edit Dataset")
        )
        dialog.geometry("500x600")
        self._center_window(dialog, 500, 600)
        
        # Scrollable frame
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        self._bind_canvas_scrollregion(canvas, scrollable_frame)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._bind_canvas_mousewheel(canvas, scrollable_frame)
        
        # Dataset label
        label_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.dataset_label", "Dataset Label"), padding=10)
        label_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(label_frame, text=self._t("ui.labels.label_for_legend", "Label (for legend):")).pack(anchor=tk.W)
        label_var = tk.StringVar(value=existing_config.get('label', '') if existing_config else '')
        label_entry = ttk.Entry(label_frame, textvariable=label_var, width=40)
        label_entry.pack(fill=tk.X, pady=2)
        
        # Marker
        marker_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.marker_style", "Marker Style"), padding=10)
        marker_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(marker_frame, text=self._t("ui.labels.marker", "Marker:")).pack(anchor=tk.W)
        marker_var = tk.StringVar(value=existing_config.get('marker', 'o') if existing_config else 'o')
        markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', 'H', '+', 'x']
        marker_combo = ttk.Combobox(marker_frame, textvariable=marker_var, values=markers, width=10)
        marker_combo.pack(fill=tk.X, pady=2)
        
        # Color (optional)
        color_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.color_optional", "Color (Optional)"), padding=10)
        color_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(color_frame, text=self._t("ui.labels.color_leave_blank_auto", "Color (leave blank for auto):")).pack(anchor=tk.W)
        color_var = tk.StringVar(value=existing_config.get('color', '') if existing_config else '')
        color_entry = ttk.Entry(color_frame, textvariable=color_var, width=20)
        color_entry.pack(fill=tk.X, pady=2)
        ttk.Label(color_frame, text=self._t("ui.labels.color_examples", "Examples: #1f77b4, red, blue"), 
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        
        # X-axis config
        x_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.x_axis_configuration", "X-Axis Configuration"), padding=10)
        x_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(x_frame, text=self._t("ui.labels.input_output_source_colon", "Input/Output Source:")).pack(anchor=tk.W)
        x_source_var = tk.StringVar(value=existing_config.get('x_axis', {}).get('data_source', '') if existing_config else '')
        x_source_combo = ttk.Combobox(x_frame, textvariable=x_source_var, 
                                      values=[''] + self._get_available_data_sources(), width=40)
        x_source_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(x_frame, text=self._t("ui.labels.nested_key_optional", "Nested Key (optional):")).pack(anchor=tk.W, pady=(5, 0))
        x_nested_var = tk.StringVar(value=existing_config.get('x_axis', {}).get('nested_key', '') if existing_config else '')
        x_nested_combo = ttk.Combobox(x_frame, textvariable=x_nested_var, width=40)
        x_nested_combo.pack(fill=tk.X, pady=2)
        ttk.Label(x_frame, text=self._t("ui.labels.reference_source_optional", "Reference Source (optional):")).pack(anchor=tk.W, pady=(5, 0))
        x_ref_source_var = tk.StringVar(value=existing_config.get('x_axis', {}).get('reference_source', '') if existing_config else '')
        x_ref_source_combo = ttk.Combobox(x_frame, textvariable=x_ref_source_var,
                                          values=[''] + self._get_available_data_sources(), width=40)
        x_ref_source_combo.pack(fill=tk.X, pady=2)
        ttk.Label(x_frame, text=self._t("ui.labels.reference_nested_key_optional", "Reference Nested Key (optional):")).pack(anchor=tk.W, pady=(5, 0))
        x_ref_nested_var = tk.StringVar(value=existing_config.get('x_axis', {}).get('reference_nested_key', '') if existing_config else '')
        x_ref_nested_combo = ttk.Combobox(x_frame, textvariable=x_ref_nested_var, width=40)
        x_ref_nested_combo.pack(fill=tk.X, pady=2)
        ttk.Label(x_frame, text=self._t("ui.labels.reference_axis_optional", "Reference Axis (optional):")).pack(anchor=tk.W, pady=(5, 0))
        x_ref_axis_var = tk.StringVar(value=str(existing_config.get('x_axis', {}).get('reference_axis', '')) if existing_config else '')
        ttk.Combobox(x_frame, textvariable=x_ref_axis_var,
                     values=['', 'first', 'last', '0', '1', '-1'], width=20).pack(anchor=tk.W, pady=2)
        x_treat_list_var = tk.BooleanVar(value=existing_config.get('x_axis', {}).get('treat_as_list_source', False) if existing_config else False)
        ttk.Checkbutton(
            x_frame,
            text=self._t("ui.labels.treat_as_list_source", "Treat nested list as axis-list source"),
            variable=x_treat_list_var
        ).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(x_frame, text=self._t("ui.labels.axis_type", "Axis Type:")).pack(anchor=tk.W, pady=(5, 0))
        x_axis_type_var = tk.StringVar(value=existing_config.get('x_axis', {}).get('axis_type', 'linear') if existing_config else 'linear')
        ttk.Combobox(
            x_frame,
            textvariable=x_axis_type_var,
            values=['linear', 'log10', 'log2', 'ln'],
            state='readonly',
            width=20
        ).pack(anchor=tk.W, pady=2)
        x_force_integer_var = tk.StringVar(
            value=self._normalize_force_integer_mode(existing_config.get('x_axis', {}).get('force_integer', False) if existing_config else False)
        )
        ttk.Label(x_frame, text=self._t("ui.labels.force_integer_mode", "Force Integer Ticks:")).pack(anchor=tk.W, pady=(5, 0))
        ttk.Combobox(
            x_frame,
            textvariable=x_force_integer_var,
            values=['false', 'true', 'conditional'],
            state='readonly',
            width=20
        ).pack(anchor=tk.W, pady=(5, 0))
        x_reverse_axis_var = tk.BooleanVar(value=existing_config.get('x_axis', {}).get('reverse_axis', False) if existing_config else False)
        ttk.Checkbutton(
            x_frame,
            text=self._t("ui.labels.reverse_axis", "Reverse axis"),
            variable=x_reverse_axis_var
        ).pack(anchor=tk.W, pady=(2, 0))
        
        # Y-axis config
        y_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.y_axis_configuration", "Y-Axis Configuration"), padding=10)
        y_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(y_frame, text=self._t("ui.labels.input_output_source_colon", "Input/Output Source:")).pack(anchor=tk.W)
        y_source_var = tk.StringVar(value=existing_config.get('y_axis', {}).get('data_source', '') if existing_config else '')
        y_source_combo = ttk.Combobox(y_frame, textvariable=y_source_var,
                                      values=[''] + self._get_available_data_sources(), width=40)
        y_source_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(y_frame, text=self._t("ui.labels.nested_key_optional", "Nested Key (optional):")).pack(anchor=tk.W, pady=(5, 0))
        y_nested_var = tk.StringVar(value=existing_config.get('y_axis', {}).get('nested_key', '') if existing_config else '')
        y_nested_combo = ttk.Combobox(y_frame, textvariable=y_nested_var, width=40)
        y_nested_combo.pack(fill=tk.X, pady=2)
        ttk.Label(y_frame, text=self._t("ui.labels.reference_source_optional", "Reference Source (optional):")).pack(anchor=tk.W, pady=(5, 0))
        y_ref_source_var = tk.StringVar(value=existing_config.get('y_axis', {}).get('reference_source', '') if existing_config else '')
        y_ref_source_combo = ttk.Combobox(y_frame, textvariable=y_ref_source_var,
                                          values=[''] + self._get_available_data_sources(), width=40)
        y_ref_source_combo.pack(fill=tk.X, pady=2)
        ttk.Label(y_frame, text=self._t("ui.labels.reference_nested_key_optional", "Reference Nested Key (optional):")).pack(anchor=tk.W, pady=(5, 0))
        y_ref_nested_var = tk.StringVar(value=existing_config.get('y_axis', {}).get('reference_nested_key', '') if existing_config else '')
        y_ref_nested_combo = ttk.Combobox(y_frame, textvariable=y_ref_nested_var, width=40)
        y_ref_nested_combo.pack(fill=tk.X, pady=2)
        ttk.Label(y_frame, text=self._t("ui.labels.reference_axis_optional", "Reference Axis (optional):")).pack(anchor=tk.W, pady=(5, 0))
        y_ref_axis_var = tk.StringVar(value=str(existing_config.get('y_axis', {}).get('reference_axis', '')) if existing_config else '')
        ttk.Combobox(y_frame, textvariable=y_ref_axis_var,
                     values=['', 'first', 'last', '0', '1', '-1'], width=20).pack(anchor=tk.W, pady=2)
        y_treat_list_var = tk.BooleanVar(value=existing_config.get('y_axis', {}).get('treat_as_list_source', False) if existing_config else False)
        ttk.Checkbutton(
            y_frame,
            text=self._t("ui.labels.treat_as_list_source", "Treat nested list as axis-list source"),
            variable=y_treat_list_var
        ).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(y_frame, text=self._t("ui.labels.axis_type", "Axis Type:")).pack(anchor=tk.W, pady=(5, 0))
        y_axis_type_var = tk.StringVar(value=existing_config.get('y_axis', {}).get('axis_type', 'linear') if existing_config else 'linear')
        ttk.Combobox(
            y_frame,
            textvariable=y_axis_type_var,
            values=['linear', 'log10', 'log2', 'ln'],
            state='readonly',
            width=20
        ).pack(anchor=tk.W, pady=2)
        y_force_integer_var = tk.StringVar(
            value=self._normalize_force_integer_mode(existing_config.get('y_axis', {}).get('force_integer', False) if existing_config else False)
        )
        ttk.Label(y_frame, text=self._t("ui.labels.force_integer_mode", "Force Integer Ticks:")).pack(anchor=tk.W, pady=(5, 0))
        ttk.Combobox(
            y_frame,
            textvariable=y_force_integer_var,
            values=['false', 'true', 'conditional'],
            state='readonly',
            width=20
        ).pack(anchor=tk.W, pady=(5, 0))
        y_reverse_axis_var = tk.BooleanVar(value=existing_config.get('y_axis', {}).get('reverse_axis', False) if existing_config else False)
        ttk.Checkbutton(
            y_frame,
            text=self._t("ui.labels.reverse_axis", "Reverse axis"),
            variable=y_reverse_axis_var
        ).pack(anchor=tk.W, pady=(2, 0))
        
        # Z-axis config (optional)
        z_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.z_axis_configuration_optional_3d", "Z-Axis Configuration (Optional, for 3D)"), padding=10)
        z_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(z_frame, text=self._t("ui.labels.input_output_source_colon", "Input/Output Source:")).pack(anchor=tk.W)
        z_source_var = tk.StringVar(value=existing_config.get('z_axis', {}).get('data_source', '') if existing_config else '')
        z_source_combo = ttk.Combobox(z_frame, textvariable=z_source_var,
                                      values=[''] + self._get_available_data_sources(), width=40)
        z_source_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(z_frame, text=self._t("ui.labels.nested_key_optional", "Nested Key (optional):")).pack(anchor=tk.W, pady=(5, 0))
        z_nested_var = tk.StringVar(value=existing_config.get('z_axis', {}).get('nested_key', '') if existing_config else '')
        z_nested_combo = ttk.Combobox(z_frame, textvariable=z_nested_var, width=40)
        z_nested_combo.pack(fill=tk.X, pady=2)
        ttk.Label(z_frame, text=self._t("ui.labels.reference_source_optional", "Reference Source (optional):")).pack(anchor=tk.W, pady=(5, 0))
        z_ref_source_var = tk.StringVar(value=existing_config.get('z_axis', {}).get('reference_source', '') if existing_config else '')
        z_ref_source_combo = ttk.Combobox(z_frame, textvariable=z_ref_source_var,
                                          values=[''] + self._get_available_data_sources(), width=40)
        z_ref_source_combo.pack(fill=tk.X, pady=2)
        ttk.Label(z_frame, text=self._t("ui.labels.reference_nested_key_optional", "Reference Nested Key (optional):")).pack(anchor=tk.W, pady=(5, 0))
        z_ref_nested_var = tk.StringVar(value=existing_config.get('z_axis', {}).get('reference_nested_key', '') if existing_config else '')
        z_ref_nested_combo = ttk.Combobox(z_frame, textvariable=z_ref_nested_var, width=40)
        z_ref_nested_combo.pack(fill=tk.X, pady=2)
        ttk.Label(z_frame, text=self._t("ui.labels.reference_axis_optional", "Reference Axis (optional):")).pack(anchor=tk.W, pady=(5, 0))
        z_ref_axis_var = tk.StringVar(value=str(existing_config.get('z_axis', {}).get('reference_axis', '')) if existing_config else '')
        ttk.Combobox(z_frame, textvariable=z_ref_axis_var,
                     values=['', 'first', 'last', '0', '1', '-1'], width=20).pack(anchor=tk.W, pady=2)
        z_treat_list_var = tk.BooleanVar(value=existing_config.get('z_axis', {}).get('treat_as_list_source', False) if existing_config else False)
        ttk.Checkbutton(
            z_frame,
            text=self._t("ui.labels.treat_as_list_source", "Treat nested list as axis-list source"),
            variable=z_treat_list_var
        ).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(z_frame, text=self._t("ui.labels.axis_type", "Axis Type:")).pack(anchor=tk.W, pady=(5, 0))
        z_axis_type_var = tk.StringVar(value=existing_config.get('z_axis', {}).get('axis_type', 'linear') if existing_config else 'linear')
        ttk.Combobox(
            z_frame,
            textvariable=z_axis_type_var,
            values=['linear', 'log10', 'log2', 'ln'],
            state='readonly',
            width=20
        ).pack(anchor=tk.W, pady=2)
        z_force_integer_var = tk.StringVar(
            value=self._normalize_force_integer_mode(existing_config.get('z_axis', {}).get('force_integer', False) if existing_config else False)
        )
        ttk.Label(z_frame, text=self._t("ui.labels.force_integer_mode", "Force Integer Ticks:")).pack(anchor=tk.W, pady=(5, 0))
        ttk.Combobox(
            z_frame,
            textvariable=z_force_integer_var,
            values=['false', 'true', 'conditional'],
            state='readonly',
            width=20
        ).pack(anchor=tk.W, pady=(5, 0))
        z_reverse_axis_var = tk.BooleanVar(value=existing_config.get('z_axis', {}).get('reverse_axis', False) if existing_config else False)
        ttk.Checkbutton(
            z_frame,
            text=self._t("ui.labels.reverse_axis", "Reverse axis"),
            variable=z_reverse_axis_var
        ).pack(anchor=tk.W, pady=(2, 0))

        def _refresh_dataset_nested_keys(source_var: tk.StringVar, nested_combo: ttk.Combobox):
            source = source_var.get().strip()
            nested_keys = self._get_nested_keys(source)
            current = nested_combo.get().strip()
            nested_combo['values'] = [''] + nested_keys
            if current and current not in nested_keys:
                nested_combo.set(current)

        x_source_combo.bind('<<ComboboxSelected>>', lambda _e: _refresh_dataset_nested_keys(x_source_var, x_nested_combo))
        y_source_combo.bind('<<ComboboxSelected>>', lambda _e: _refresh_dataset_nested_keys(y_source_var, y_nested_combo))
        z_source_combo.bind('<<ComboboxSelected>>', lambda _e: _refresh_dataset_nested_keys(z_source_var, z_nested_combo))
        x_ref_source_combo.bind('<<ComboboxSelected>>', lambda _e: _refresh_dataset_nested_keys(x_ref_source_var, x_ref_nested_combo))
        y_ref_source_combo.bind('<<ComboboxSelected>>', lambda _e: _refresh_dataset_nested_keys(y_ref_source_var, y_ref_nested_combo))
        z_ref_source_combo.bind('<<ComboboxSelected>>', lambda _e: _refresh_dataset_nested_keys(z_ref_source_var, z_ref_nested_combo))

        # Initialize nested-key lists for edit mode and pre-filled values
        _refresh_dataset_nested_keys(x_source_var, x_nested_combo)
        _refresh_dataset_nested_keys(y_source_var, y_nested_combo)
        _refresh_dataset_nested_keys(z_source_var, z_nested_combo)
        _refresh_dataset_nested_keys(x_ref_source_var, x_ref_nested_combo)
        _refresh_dataset_nested_keys(y_ref_source_var, y_ref_nested_combo)
        _refresh_dataset_nested_keys(z_ref_source_var, z_ref_nested_combo)
        
        # Class labels (optional)
        class_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.class_labels_optional", "Class Labels (Optional)"), padding=10)
        class_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(class_frame, text=self._t("ui.labels.class_labels_source", "Class Labels Source:")).pack(anchor=tk.W)
        class_var = tk.StringVar(value=existing_config.get('class_labels', '') if existing_config else '')
        class_combo = ttk.Combobox(class_frame, textvariable=class_var,
                                   values=[''] + self._get_available_data_sources(), width=40)
        class_combo.pack(fill=tk.X, pady=2)
        
        # Point labels (optional)
        sample_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.point_labels_optional", "Point Labels (Optional)"), padding=10)
        sample_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sample_frame, text=self._t("ui.labels.point_labels_source", "Point Labels Source:")).pack(anchor=tk.W)
        sample_var = tk.StringVar(value=existing_config.get('point_labels_source', existing_config.get('sample_labels_source', '')) if existing_config else '')
        sample_combo = ttk.Combobox(sample_frame, textvariable=sample_var,
                                    values=[''] + self._get_available_data_sources(), width=40)
        sample_combo.pack(fill=tk.X, pady=2)
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        def save_dataset():
            # Build dataset config
            ds_config = {
                'label': label_var.get() or self._t("ui.labels.dataset", "Dataset"),
                'marker': marker_var.get() or 'o'
            }
            
            if color_var.get():
                ds_config['color'] = color_var.get()
            
            # X-axis
            if x_source_var.get():
                x_axis = {'data_source': x_source_var.get()}
                if x_nested_var.get():
                    x_axis['nested_key'] = x_nested_var.get()
                if x_ref_source_var.get().strip():
                    x_axis['reference_source'] = x_ref_source_var.get().strip()
                if x_ref_nested_var.get().strip():
                    x_axis['reference_nested_key'] = x_ref_nested_var.get().strip()
                if x_ref_axis_var.get().strip():
                    x_axis['reference_axis'] = x_ref_axis_var.get().strip()
                if x_treat_list_var.get():
                    x_axis['treat_as_list_source'] = True
                if x_axis_type_var.get() and x_axis_type_var.get() != 'linear':
                    x_axis['axis_type'] = x_axis_type_var.get()
                self._apply_force_integer_mode(x_axis, x_force_integer_var.get())
                if x_reverse_axis_var.get():
                    x_axis['reverse_axis'] = True
                ds_config['x_axis'] = x_axis
            
            # Y-axis
            if y_source_var.get():
                y_axis = {'data_source': y_source_var.get()}
                if y_nested_var.get():
                    y_axis['nested_key'] = y_nested_var.get()
                if y_ref_source_var.get().strip():
                    y_axis['reference_source'] = y_ref_source_var.get().strip()
                if y_ref_nested_var.get().strip():
                    y_axis['reference_nested_key'] = y_ref_nested_var.get().strip()
                if y_ref_axis_var.get().strip():
                    y_axis['reference_axis'] = y_ref_axis_var.get().strip()
                if y_treat_list_var.get():
                    y_axis['treat_as_list_source'] = True
                if y_axis_type_var.get() and y_axis_type_var.get() != 'linear':
                    y_axis['axis_type'] = y_axis_type_var.get()
                self._apply_force_integer_mode(y_axis, y_force_integer_var.get())
                if y_reverse_axis_var.get():
                    y_axis['reverse_axis'] = True
                ds_config['y_axis'] = y_axis
            
            # Z-axis
            if z_source_var.get():
                z_axis = {'data_source': z_source_var.get()}
                if z_nested_var.get():
                    z_axis['nested_key'] = z_nested_var.get()
                if z_ref_source_var.get().strip():
                    z_axis['reference_source'] = z_ref_source_var.get().strip()
                if z_ref_nested_var.get().strip():
                    z_axis['reference_nested_key'] = z_ref_nested_var.get().strip()
                if z_ref_axis_var.get().strip():
                    z_axis['reference_axis'] = z_ref_axis_var.get().strip()
                if z_treat_list_var.get():
                    z_axis['treat_as_list_source'] = True
                if z_axis_type_var.get() and z_axis_type_var.get() != 'linear':
                    z_axis['axis_type'] = z_axis_type_var.get()
                self._apply_force_integer_mode(z_axis, z_force_integer_var.get())
                if z_reverse_axis_var.get():
                    z_axis['reverse_axis'] = True
                ds_config['z_axis'] = z_axis
            
            # Class labels
            if class_var.get():
                ds_config['class_labels'] = class_var.get()
            
            # Point labels
            if sample_var.get():
                ds_config['point_labels_source'] = sample_var.get()
            
            # Validate
            if not ds_config.get('x_axis') or not ds_config.get('y_axis'):
                self._notify(self._t("ui.messages.x_y_required", "Both X and Y axis data sources are required"), level="warning")
                return
            
            # Add or update
            if edit_idx is not None and edit_idx < len(self.datasets_configs):
                self.datasets_configs[edit_idx] = ds_config
            else:
                self.datasets_configs.append(ds_config)
            
            self._refresh_datasets_list()
            dialog.destroy()
        
        ttk.Button(btn_frame, text="✓ " + self._t("ui.buttons.save", "Save"), command=save_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="✗ " + self._t("ui.buttons.cancel", "Cancel"), command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _refresh_datasets_list(self):
        """Refresh the datasets listbox."""
        self.datasets_listbox.delete(0, tk.END)
        for idx, ds in enumerate(self.datasets_configs):
            label = ds.get('label', f"{self._t('ui.labels.dataset', 'Dataset')} {idx+1}")
            marker = ds.get('marker', 'o')
            x_src = ds.get('x_axis', {}).get('data_source', '?')
            y_src = ds.get('y_axis', {}).get('data_source', '?')
            display = f"{label} ({marker}) - X:{x_src}, Y:{y_src}"
            self.datasets_listbox.insert(tk.END, display)
    
    def _build_advanced_tab(self, parent):
        """Build advanced options tab."""
        # Scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        self._bind_canvas_scrollregion(canvas, scrollable_frame)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._bind_canvas_mousewheel(canvas, scrollable_frame)
        
        # Class labels
        class_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.class_labels_for_coloring", "Class Labels (for coloring)"), padding=10)
        class_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(class_frame, text=self._t("ui.labels.class_data_source", "Class Data Source:")).pack(anchor=tk.W)
        self.class_labels_var = tk.StringVar()
        class_combo = ttk.Combobox(class_frame, textvariable=self.class_labels_var, 
                                   values=[''] + self._get_available_data_sources(), width=40)
        class_combo.pack(fill=tk.X, pady=5)
        
        # Point labels
        sample_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.point_labels_for_scatter", "Point Labels (for scatter)"), padding=10)
        sample_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sample_frame, text=self._t("ui.labels.point_labels_source", "Point Labels Source:")).pack(anchor=tk.W)
        self.sample_labels_var = tk.StringVar()
        sample_combo = ttk.Combobox(sample_frame, textvariable=self.sample_labels_var,
                                    values=[''] + self._get_available_data_sources(), width=40)
        sample_combo.pack(fill=tk.X, pady=5)

        # Display options
        display_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.display_options", "Display Options"), padding=10)
        display_frame.pack(fill=tk.X, padx=5, pady=5)

        self.show_grid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_frame,
            text=self._t("ui.labels.show_grid", "Show grid"),
            variable=self.show_grid_var
        ).pack(anchor=tk.W, pady=2)

        self.show_origin_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_frame,
            text=self._t("ui.labels.show_origin", "Show origin (x=0, y=0)"),
            variable=self.show_origin_var
        ).pack(anchor=tk.W, pady=2)

        self.show_labels_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_frame,
            text=self._t("ui.labels.show_labels", "Show point labels (scatter)"),
            variable=self.show_labels_var
        ).pack(anchor=tk.W, pady=2)

        self.equal_scale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_frame,
            text=self._t("ui.labels.equal_scale", "Equal scale X/Y axes (scatter 2D)"),
            variable=self.equal_scale_var
        ).pack(anchor=tk.W, pady=2)

        self.annotate_heatmap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_frame,
            text=self._t("ui.labels.annotate_heatmap", "Annotate heatmap cells"),
            variable=self.annotate_heatmap_var
        ).pack(anchor=tk.W, pady=2)

        # Confidence ellipse options (scatter)
        conf_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.confidence_ellipses_scatter", "Confidence Ellipses (scatter)"), padding=10)
        conf_frame.pack(fill=tk.X, padx=5, pady=5)

        self.confidence_ellipses_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            conf_frame,
            text=self._t("ui.labels.enable_confidence_ellipses", "Enable confidence ellipses"),
            variable=self.confidence_ellipses_var
        ).pack(anchor=tk.W, pady=2)

        level_row = ttk.Frame(conf_frame)
        level_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(level_row, text=self._t("ui.labels.confidence_level", "Confidence level (%):")).pack(side=tk.LEFT)
        self.confidence_level_var = tk.StringVar(value="95")
        ttk.Entry(level_row, textvariable=self.confidence_level_var, width=10).pack(side=tk.LEFT, padx=(8, 0))

        # Scatter reference lines editor
        lines_frame = ttk.LabelFrame(
            scrollable_frame,
            text=self._t("ui.labels.scatter_reference_lines", "Scatter Reference Lines (JSON)"),
            padding=10
        )
        lines_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        ttk.Label(
            lines_frame,
            text=self._t(
                "ui.messages.scatter_lines_help",
                "Optional. Define vertical/horizontal/path/segment lines for 2D scatter using JSON. "
                "Leave empty to disable."
            ),
            wraplength=450,
            foreground="gray"
        ).pack(anchor=tk.W)

        templates_row = ttk.Frame(lines_frame)
        templates_row.pack(fill=tk.X, pady=(6, 0))

        ttk.Label(
            templates_row,
            text=self._t("ui.labels.quick_templates", "Quick templates:"),
        ).pack(side=tk.LEFT)

        ttk.Button(
            templates_row,
            text=self._t("ui.buttons.insert_vh_template", "Insert V/H"),
            command=lambda: self._insert_scatter_lines_template('vertical_horizontal')
        ).pack(side=tk.LEFT, padx=(8, 4))

        ttk.Button(
            templates_row,
            text=self._t("ui.buttons.insert_path_template", "Insert Path"),
            command=lambda: self._insert_scatter_lines_template('path')
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            templates_row,
            text=self._t("ui.buttons.insert_data_source_path_template", "Insert Data-Source Path"),
            command=lambda: self._insert_scatter_lines_template('data_source_path')
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            templates_row,
            text=self._t("ui.buttons.clear", "Clear"),
            command=lambda: self.scatter_lines_text.delete('1.0', tk.END)
        ).pack(side=tk.RIGHT)

        self.scatter_lines_text = scrolledtext.ScrolledText(lines_frame, height=8, wrap=tk.WORD)
        self.scatter_lines_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        ttk.Label(
            lines_frame,
            text=self._t(
                "ui.messages.scatter_lines_example",
                'Example: [{"orientation":"path","x":{"data_source":"x_path"},"y":{"data_source":"y_path"},"linestyle":"--","linewidth":1.5,"color":"#663300"}]'
            ),
            wraplength=450,
            foreground="gray",
            font=("Arial", 8)
        ).pack(anchor=tk.W, pady=(4, 0))
        
        # Data slicing configuration
        slice_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.data_slicing_navigation", "Data Slicing / Navigation"), padding=10)
        slice_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(slice_frame, text=self._t("ui.messages.enable_nav_multi_dim", "Enable navigation for exploring multi-dimensional data:"), 
                 wraplength=400).pack(anchor=tk.W, pady=5)
        
        # Axis navigation checkboxes with dimension entry
        axes_config = [('x', 'X-axis'), ('y', 'Y-axis'), ('z', 'Z-axis')]
        
        for axis_key, axis_label in axes_config:
            axis_frame = ttk.Frame(slice_frame)
            axis_frame.pack(fill=tk.X, pady=3)
            
            # Checkbox for enabling navigation on this axis
            nav_var = tk.BooleanVar(value=False)
            setattr(self, f'{axis_key}_nav_enabled_var', nav_var)
            cb = ttk.Checkbutton(axis_frame, text=self._t("ui.labels.axis_navigation", "{axis} Navigation").format(axis=axis_label), variable=nav_var)
            cb.pack(side=tk.LEFT)
            
            # Dimension entry
            ttk.Label(axis_frame, text=self._t("ui.labels.dim", "Dim:")).pack(side=tk.LEFT, padx=(10, 2))
            dim_var = tk.StringVar(value="0")
            setattr(self, f'{axis_key}_nav_dim_var', dim_var)
            dim_entry = ttk.Entry(axis_frame, textvariable=dim_var, width=8)
            dim_entry.pack(side=tk.LEFT, padx=2)
            
            # Default value entry
            ttk.Label(axis_frame, text=self._t("ui.labels.default", "Default:")).pack(side=tk.LEFT, padx=(10, 2))
            default_var = tk.StringVar(value="0")
            setattr(self, f'{axis_key}_nav_default_var', default_var)
            
            default_entry = ttk.Entry(axis_frame, textvariable=default_var, width=8)
            default_entry.pack(side=tk.LEFT, padx=2)

            # Optional synchronization group id for linked navigation controls
            ttk.Label(axis_frame, text=self._t("ui.labels.nav_id", "Nav ID:")).pack(side=tk.LEFT, padx=(10, 2))
            nav_id_var = tk.StringVar(value="")
            setattr(self, f'{axis_key}_nav_id_var', nav_id_var)
            nav_id_entry = ttk.Entry(axis_frame, textvariable=nav_id_var, width=14)
            nav_id_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(slice_frame, text=self._t("ui.messages.tip_dim_samples", "Tip: Dim 0=samples, 1=first variable dimension (e.g., PCs)"), 
                 font=("Arial", 8), foreground="gray", wraplength=400).pack(anchor=tk.W, pady=(5, 0))
        
        # Multi-Dimensional Slicing for 4D+ Data
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)
        
        md_slice_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.multi_dim_slicing_4d_data", "Multi-Dimensional Slicing (4D+ Data)"), padding=10)
        md_slice_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(md_slice_frame, text=self._t("ui.messages.configure_4d_heatmap_slicing", "For 4D+ heatmaps/contours, configure dimension combinations and slicing:"), 
                 wraplength=450).pack(anchor=tk.W, pady=(0, 10))
        
        # Enable 4D+ multi-dimensional slicing
        self.enable_md_slicing_var = tk.BooleanVar(value=False)
        md_enable_cb = ttk.Checkbutton(md_slice_frame, text=self._t("ui.labels.enable_4d_multi_dim_slicing", "Enable 4D+ Multi-Dimensional Slicing"), 
                                       variable=self.enable_md_slicing_var, command=self._toggle_md_config)
        md_enable_cb.pack(anchor=tk.W, pady=5)
        
        # Container for MD configuration (hidden by default)
        self.md_config_frame = ttk.Frame(md_slice_frame)
        
        # Dimension combination selector
        combo_frame = ttk.Frame(self.md_config_frame)
        combo_frame.pack(fill=tk.X, pady=5)
        ttk.Label(combo_frame, text=self._t("ui.labels.dimension_combination_index", "Dimension Combination Index:"), width=25).pack(side=tk.LEFT, padx=(0, 5))
        self.md_combo_index_var = tk.StringVar(value="0")
        ttk.Entry(combo_frame, textvariable=self.md_combo_index_var, width=10).pack(side=tk.LEFT)
        ttk.Label(combo_frame, text=self._t("ui.labels.zero_first_combination", "(0 = first combination)"), font=("Arial", 8), 
                 foreground="gray").pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(self.md_config_frame, text=self._t("ui.messages.heatmap_two_dims_xy", "For heatmaps/contours: 2 dims used for X/Y axes, others are navigable."), 
                 font=("Arial", 8), foreground="gray", wraplength=450).pack(anchor=tk.W, pady=(5, 10))
        
        # Default slice indices for navigable dimensions
        ttk.Label(self.md_config_frame, text=self._t("ui.labels.default_slice_indices_navigable", "Default slice indices for navigable dimensions:")).pack(anchor=tk.W)
        ttk.Label(self.md_config_frame, text=self._t("ui.labels.slice_format_dim_index", "Format: dim:index (e.g., '2:0,3:5' means dim 2→index 0, dim 3→index 5)"), 
                 font=("Arial", 8), foreground="gray", wraplength=450).pack(anchor=tk.W, pady=(0, 5))
        
        self.md_slice_indices_var = tk.StringVar()
        md_slice_entry = ttk.Entry(self.md_config_frame, textvariable=self.md_slice_indices_var, width=50)
        md_slice_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.md_config_frame, text=self._t("ui.messages.leave_blank_default_navigable", "Leave blank to use default (0) for all navigable dimensions."), 
                 font=("Arial", 8), foreground="gray", wraplength=450).pack(anchor=tk.W, pady=(0, 5))
    
    def _toggle_md_config(self):
        """Toggle visibility of multi-dimensional slicing configuration."""
        if self.enable_md_slicing_var.get():
            self.md_config_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            self.md_config_frame.pack_forget()
    
    def _build_preview_panel(self, parent):
        """Build the preview panel."""
        # Title
        title = ttk.Label(parent, text=self._t("ui.labels.preview", "Preview"), font=("Arial", 12, "bold"))
        title.pack(pady=10)
        
        # Preview button
        preview_btn = ttk.Button(parent, text="🔄 " + self._t("ui.buttons.update_preview", "Update Preview"), command=self._update_preview)
        preview_btn.pack(pady=5)
        
        # Preview container
        preview_container = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=2)
        preview_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.preview_container = preview_container
        
        # Initial message
        msg = ttk.Label(preview_container, text=self._t("ui.messages.configure_graph_update_preview", "Configure your graph and click 'Update Preview'"),
                       foreground="gray", font=("Arial", 10, "italic"))
        msg.pack(expand=True)
    
    def _build_button_bar(self):
        """Build bottom button bar."""
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        # Add Graph button
        add_btn = ttk.Button(button_frame, text="✓ " + self._t("ui.buttons.add_graph", "Add Graph"), command=self._add_graph)
        add_btn.pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        cancel_btn = ttk.Button(button_frame, text="✗ " + self._t("ui.buttons.cancel", "Cancel"), command=self.dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Spacer
        spacer = ttk.Frame(button_frame)
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Help button
        help_btn = ttk.Button(button_frame, text="? " + self._t("ui.buttons.help", "Help"), command=self._show_help)
        help_btn.pack(side=tk.RIGHT, padx=5)
    
    def _on_graph_type_changed(self):
        """Handle graph type change."""
        # Could add logic here to show/hide relevant options based on graph type
        pass
    
    def _on_data_source_changed(self, axis_key: str):
        """Handle data source change for an axis."""
        data_source_var = getattr(self, f'{axis_key}_data_source_var')
        nested_combo = getattr(self, f'{axis_key}_nested_combo')
        info_var = getattr(self, f'{axis_key}_info_var')
        
        data_source = data_source_var.get()
        
        # Update nested keys combo
        nested_keys = self._get_nested_keys(data_source)
        nested_combo['values'] = [''] + nested_keys
        
        # Update info
        if data_source:
            info = self._get_data_shape_info(data_source)
            info_var.set(info)
        else:
            info_var.set(self._t("ui.messages.select_data_source", "Select a data source"))

    def _on_reference_source_changed(self, axis_key: str):
        """Handle reference source change for an axis."""
        ref_source_var = getattr(self, f'{axis_key}_reference_source_var')
        ref_nested_combo = getattr(self, f'{axis_key}_reference_nested_combo')

        source = ref_source_var.get().strip()
        nested_keys = self._get_nested_keys(source)
        ref_nested_combo['values'] = [''] + nested_keys

    def _parse_scatter_lines_json(self) -> Optional[List[Dict[str, Any]]]:
        """Parse optional scatter reference-lines JSON from advanced tab editor."""
        editor = getattr(self, 'scatter_lines_text', None)
        if editor is None:
            return None

        raw_text = editor.get('1.0', tk.END).strip()
        if not raw_text:
            return None

        try:
            parsed = json.loads(raw_text)
        except Exception as exc:
            raise ValueError(f"Invalid Scatter Reference Lines JSON: {exc}") from exc

        if isinstance(parsed, dict):
            parsed = [parsed]

        if not isinstance(parsed, list):
            raise ValueError("Scatter Reference Lines JSON must be a list (or a single object).")

        normalized: List[Dict[str, Any]] = []
        for idx, entry in enumerate(parsed):
            if not isinstance(entry, dict):
                raise ValueError(f"Scatter Reference Lines entry #{idx + 1} must be an object.")
            normalized.append(entry)

        return normalized if normalized else None

    def _scatter_line_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return built-in scatter-line JSON templates shown in the dialog."""
        return {
            'vertical_horizontal': [
                {
                    "orientation": "vertical",
                    "values": [0.0, 1.0],
                    "labels": ["L1", "L2"],
                    "line_type": "--",
                    "thickness": 1.5,
                    "color": "#d62728",
                },
                {
                    "orientation": "horizontal",
                    "value": 0.5,
                    "label": "H",
                    "line_type": "--",
                    "thickness": 1.5,
                    "color": "#1f77b4",
                },
            ],
            'path': [
                {
                    "orientation": "path",
                    "points": [[0.1, 1.0], [0.5, 0.8], [1.0, 0.3]],
                    "linestyle": "--",
                    "linewidth": 1.5,
                    "color": "#663300",
                }
            ],
            'data_source_path': [
                {
                    "orientation": "vertical",
                    "value": {"data_source": "x_limit_active"},
                    "linestyle": "--",
                    "linewidth": 1.5,
                    "color": "#cc0000",
                    "alpha": 0.85,
                },
                {
                    "orientation": "horizontal",
                    "value": {"data_source": "y_limit_active"},
                    "linestyle": "--",
                    "linewidth": 1.5,
                    "color": "#0055cc",
                    "alpha": 0.85,
                },
                {
                    "orientation": "path",
                    "x": {"data_source": "x_path"},
                    "y": {"data_source": "y_path"},
                    "linestyle": "--",
                    "linewidth": 1.5,
                    "color": "#663300",
                    "alpha": 0.85,
                },
            ],
        }

    def _insert_scatter_lines_template(self, template_key: str) -> None:
        """Insert a scatter-lines template into the JSON editor.

        If valid JSON already exists in the editor, templates are appended.
        Otherwise, the template replaces current text.
        """
        editor = getattr(self, 'scatter_lines_text', None)
        if editor is None:
            return

        templates = self._scatter_line_templates()
        template_lines = copy.deepcopy(templates.get(template_key, []))
        if not template_lines:
            return

        raw_text = editor.get('1.0', tk.END).strip()
        merged_lines: List[Dict[str, Any]] = []

        if raw_text:
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict):
                    parsed = [parsed]
                if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                    merged_lines.extend(parsed)
                else:
                    raise ValueError("Scatter Reference Lines JSON must be an object or list of objects.")
            except Exception:
                # Replace invalid JSON with selected template.
                merged_lines = []

        merged_lines.extend(template_lines)

        editor.delete('1.0', tk.END)
        editor.insert('1.0', json.dumps(merged_lines, indent=2))
    
    def _build_graph_config(self) -> Dict:
        """Build graph configuration dictionary from UI inputs."""
        config = {
            'graph_type': self.graph_type_var.get(),
            'title': self.title_var.get()
        }
        
        # X-axis
        x_source = self.x_data_source_var.get()
        if x_source:
            x_config = {'data_source': x_source}
            
            x_nested = self.x_nested_key_var.get()
            if x_nested:
                x_config['nested_key'] = x_nested

            
            x_label = self.x_label_var.get()
            if x_label:
                x_config['label'] = x_label

            x_axis_type = self.x_axis_type_var.get().strip() if hasattr(self, 'x_axis_type_var') else 'linear'
            if x_axis_type and x_axis_type != 'linear':
                x_config['axis_type'] = x_axis_type

            if getattr(self, 'x_force_integer_var', None):
                self._apply_force_integer_mode(x_config, self.x_force_integer_var.get())
            if getattr(self, 'x_reverse_axis_var', None) and self.x_reverse_axis_var.get():
                x_config['reverse_axis'] = True
            if getattr(self, 'x_reference_source_var', None):
                x_reference_source = self.x_reference_source_var.get().strip()
                if x_reference_source:
                    x_config['reference_source'] = x_reference_source
            if getattr(self, 'x_reference_nested_key_var', None):
                x_reference_nested = self.x_reference_nested_key_var.get().strip()
                if x_reference_nested:
                    x_config['reference_nested_key'] = x_reference_nested
            if getattr(self, 'x_reference_axis_var', None):
                x_reference_axis = self.x_reference_axis_var.get().strip()
                if x_reference_axis:
                    x_config['reference_axis'] = x_reference_axis
            if getattr(self, 'x_treat_as_list_source_var', None) and self.x_treat_as_list_source_var.get():
                x_config['treat_as_list_source'] = True
            if getattr(self, 'x_axis_labels_var', None):
                x_axis_labels = self.x_axis_labels_var.get().strip()
                if x_axis_labels:
                    x_config['axis_labels'] = x_axis_labels
                    dim_str = getattr(self, 'x_axis_labels_dimension_var', None)
                    if dim_str is not None:
                        try:
                            x_config['axis_labels_dimension'] = int(dim_str.get())
                        except (ValueError, AttributeError):
                            pass
                    tmpl = getattr(self, 'x_label_template_var', None)
                    if tmpl is not None and tmpl.get().strip():
                        x_config['label_template'] = tmpl.get().strip()

            config['x_axis'] = x_config
        
        # Y-axis
        y_source = self.y_data_source_var.get()
        if y_source:
            y_config = {'data_source': y_source}
            
            y_nested = self.y_nested_key_var.get()
            if y_nested:
                y_config['nested_key'] = y_nested

            
            y_label = self.y_label_var.get()
            if y_label:
                y_config['label'] = y_label

            y_axis_type = self.y_axis_type_var.get().strip() if hasattr(self, 'y_axis_type_var') else 'linear'
            if y_axis_type and y_axis_type != 'linear':
                y_config['axis_type'] = y_axis_type

            if getattr(self, 'y_force_integer_var', None):
                self._apply_force_integer_mode(y_config, self.y_force_integer_var.get())
            if getattr(self, 'y_reverse_axis_var', None) and self.y_reverse_axis_var.get():
                y_config['reverse_axis'] = True
            if getattr(self, 'y_reference_source_var', None):
                y_reference_source = self.y_reference_source_var.get().strip()
                if y_reference_source:
                    y_config['reference_source'] = y_reference_source
            if getattr(self, 'y_reference_nested_key_var', None):
                y_reference_nested = self.y_reference_nested_key_var.get().strip()
                if y_reference_nested:
                    y_config['reference_nested_key'] = y_reference_nested
            if getattr(self, 'y_reference_axis_var', None):
                y_reference_axis = self.y_reference_axis_var.get().strip()
                if y_reference_axis:
                    y_config['reference_axis'] = y_reference_axis
            if getattr(self, 'y_treat_as_list_source_var', None) and self.y_treat_as_list_source_var.get():
                y_config['treat_as_list_source'] = True
            if getattr(self, 'y_axis_labels_var', None):
                y_axis_labels = self.y_axis_labels_var.get().strip()
                if y_axis_labels:
                    y_config['axis_labels'] = y_axis_labels
                    dim_str = getattr(self, 'y_axis_labels_dimension_var', None)
                    if dim_str is not None:
                        try:
                            y_config['axis_labels_dimension'] = int(dim_str.get())
                        except (ValueError, AttributeError):
                            pass
                    tmpl = getattr(self, 'y_label_template_var', None)
                    if tmpl is not None and tmpl.get().strip():
                        y_config['label_template'] = tmpl.get().strip()

            config['y_axis'] = y_config
        
        # Z-axis
        z_source = self.z_data_source_var.get()
        if z_source:
            z_config = {'data_source': z_source}
            
            z_nested = self.z_nested_key_var.get()
            if z_nested:
                z_config['nested_key'] = z_nested

            
            z_label = self.z_label_var.get()
            if z_label:
                z_config['label'] = z_label

            z_axis_type = self.z_axis_type_var.get().strip() if hasattr(self, 'z_axis_type_var') else 'linear'
            if z_axis_type and z_axis_type != 'linear':
                z_config['axis_type'] = z_axis_type

            if getattr(self, 'z_force_integer_var', None):
                self._apply_force_integer_mode(z_config, self.z_force_integer_var.get())
            if getattr(self, 'z_reverse_axis_var', None) and self.z_reverse_axis_var.get():
                z_config['reverse_axis'] = True
            if getattr(self, 'z_reference_source_var', None):
                z_reference_source = self.z_reference_source_var.get().strip()
                if z_reference_source:
                    z_config['reference_source'] = z_reference_source
            if getattr(self, 'z_reference_nested_key_var', None):
                z_reference_nested = self.z_reference_nested_key_var.get().strip()
                if z_reference_nested:
                    z_config['reference_nested_key'] = z_reference_nested
            if getattr(self, 'z_reference_axis_var', None):
                z_reference_axis = self.z_reference_axis_var.get().strip()
                if z_reference_axis:
                    z_config['reference_axis'] = z_reference_axis
            if getattr(self, 'z_treat_as_list_source_var', None) and self.z_treat_as_list_source_var.get():
                z_config['treat_as_list_source'] = True
            if getattr(self, 'z_axis_labels_var', None):
                z_axis_labels = self.z_axis_labels_var.get().strip()
                if z_axis_labels:
                    z_config['axis_labels'] = z_axis_labels
                    dim_str = getattr(self, 'z_axis_labels_dimension_var', None)
                    if dim_str is not None:
                        try:
                            z_config['axis_labels_dimension'] = int(dim_str.get())
                        except (ValueError, AttributeError):
                            pass
                    tmpl = getattr(self, 'z_label_template_var', None)
                    if tmpl is not None and tmpl.get().strip():
                        z_config['label_template'] = tmpl.get().strip()

            config['z_axis'] = z_config
        
        # Class labels
        class_source = self.class_labels_var.get()
        if class_source:
            config['class_labels'] = class_source
        
        # Point labels
        sample_source = self.sample_labels_var.get()
        if sample_source:
            config['point_labels_source'] = sample_source

        # Display options
        if getattr(self, 'show_grid_var', None) and self.show_grid_var.get():
            config['show_grid'] = True
        if getattr(self, 'show_origin_var', None) and self.show_origin_var.get():
            config['show_origin'] = True
        if getattr(self, 'show_labels_var', None) and self.show_labels_var.get():
            config['show_labels'] = True
        if getattr(self, 'equal_scale_var', None) and self.equal_scale_var.get():
            config['equal_scale'] = True
        if self.graph_type_var.get() == 'heatmap':
            annotate_var = getattr(self, 'annotate_heatmap_var', None)
            config['annotate_heatmap'] = bool(annotate_var.get()) if annotate_var is not None else False
        if getattr(self, 'confidence_ellipses_var', None) and self.confidence_ellipses_var.get():
            config['confidence_ellipses'] = True
            confidence_level = self.confidence_level_var.get().strip() if getattr(self, 'confidence_level_var', None) else ''
            if confidence_level:
                config['confidence_level'] = confidence_level

        scatter_lines = self._parse_scatter_lines_json()
        if config.get('graph_type') == 'scatter' and scatter_lines:
            config['scatter_lines'] = scatter_lines
        
        # Data slicing - build navigation controls
        data_slicing = []
        
        for axis_key, axis_label in [('x', 'X-Axis'), ('y', 'Y-Axis'), ('z', 'Z-Axis')]:
            enable_var = getattr(self, f'{axis_key}_nav_enabled_var', None)
            if enable_var and enable_var.get():
                try:
                    dimension = int(getattr(self, f'{axis_key}_nav_dim_var').get())
                    default = int(getattr(self, f'{axis_key}_nav_default_var').get())
                    nav_id = getattr(self, f'{axis_key}_nav_id_var').get().strip()
                    
                    nav_entry = {
                        'name': axis_label,
                        'dimension': dimension,
                        'axis': axis_key,
                        'default': default,
                        'show_navigation_menu': True
                    }
                    if nav_id:
                        nav_entry['nav_id'] = nav_id

                    data_slicing.append(nav_entry)
                except (ValueError, AttributeError):
                    pass  # Skip if dimension or default is not a valid integer
        
        if data_slicing:
            config['data_slicing'] = data_slicing
        
        # Multi-dimensional slicing for 4D+ data
        if self.enable_md_slicing_var.get():
            config['show_md_menu'] = True
            
            # Parse MD configuration
            md_default = {}
            
            # Get combination index
            try:
                combo_idx = int(self.md_combo_index_var.get())
                md_default['combo_index'] = combo_idx
            except (ValueError, AttributeError):
                md_default['combo_index'] = 0
            
            # Parse slice indices for navigable dimensions
            slice_spec = self.md_slice_indices_var.get().strip()
            if slice_spec:
                try:
                    # Parse format: "2:0,3:5" -> {dim_2: 0, dim_3: 5}
                    for spec in slice_spec.split(','):
                        if ':' in spec:
                            dim_str, idx_str = spec.split(':', 1)
                            dim = int(dim_str.strip())
                            idx = int(idx_str.strip())
                            md_default[f'dim_{dim}'] = idx
                except (ValueError, AttributeError):
                    pass  # Invalid format, skip
            
            if md_default:
                config['md_default'] = md_default
        
        # Multi-dataset configuration
        if hasattr(self, 'datasets_configs') and self.datasets_configs:
            config['datasets'] = self.datasets_configs
        
        return config
    
    def _update_preview(self):
        """Update the graph preview."""
        try:
            # Clear preview container
            for widget in self.preview_container.winfo_children():
                widget.destroy()
            
            # Build config
            config = self._build_graph_config()
            
            # Validate basic requirements
            graph_type = config.get('graph_type', 'scatter')
            
            # Check if using multi-dataset or single dataset
            if config.get('datasets'):
                # Multi-dataset mode
                if not config['datasets']:
                    raise ValueError("No datasets configured. Add datasets in Multi-Dataset tab or use single-dataset mode (Axes tab).")
            else:
                # Single dataset mode - validate axes
                if graph_type == 'histogram':
                    if not config.get('y_axis'):
                        raise ValueError("Y-axis data source required for histogram")
                else:
                    if not config.get('x_axis') or not config.get('y_axis'):
                        raise ValueError("Both X-axis and Y-axis data sources required (configure in Axes tab or use Multi-Dataset tab)")
            
            # Extract data - handle multi-dataset or single dataset
            datasets = None
            sample_labels = None
            sample_labels_by_dataset = None
            x_data = None
            y_data = None
            z_data = None
            
            if config.get('datasets'):
                # Multi-dataset mode: extract data for each dataset
                datasets = []
                sample_labels_by_dataset = {}
                for ds_cfg in config['datasets']:
                    ds_x = None
                    ds_y = None
                    ds_z = None
                    
                    if ds_cfg.get('x_axis'):
                        ds_x = extract_axis_data(self.outputs, ds_cfg['x_axis'], {})
                    if ds_cfg.get('y_axis'):
                        ds_y = extract_axis_data(self.outputs, ds_cfg['y_axis'], {}, ref_data=ds_x)
                    if ds_cfg.get('z_axis'):
                        ds_z = extract_axis_data(self.outputs, ds_cfg['z_axis'], {}, 
                                                 ref_data=ds_x if ds_x is not None else ds_y)
                    
                    if ds_x is None or ds_y is None:
                        continue  # Skip datasets with missing data
                    
                    # Extract class data if specified
                    ds_class_data = None
                    ds_class_layers = None
                    if ds_cfg.get('class_labels'):
                        class_source = ds_cfg['class_labels']
                        if class_source in self.outputs:
                            class_val = self.outputs[class_source]
                            if isinstance(class_val, (list, np.ndarray)):
                                ds_class_layers = normalize_class_data_matrix(class_val)
                                ds_class_data = normalize_class_labels_for_plot(class_val)
                    
                    dataset_entry = {
                        'x_data': ds_x,
                        'y_data': ds_y,
                        'z_data': ds_z,
                        'label': ds_cfg.get('label', 'Dataset'),
                        'marker': ds_cfg.get('marker', 'o'),
                        'class_data': ds_class_data
                    }
                    if ds_class_layers is not None:
                        dataset_entry['class_layers'] = ds_class_layers
                    
                    if ds_cfg.get('color'):
                        dataset_entry['color'] = ds_cfg['color']

                    ds_label_source = ds_cfg.get('point_labels_source', ds_cfg.get('sample_labels_source'))
                    if ds_label_source and ds_label_source in self.outputs:
                        ds_labels = self.outputs[ds_label_source]
                        if isinstance(ds_labels, (list, np.ndarray)):
                            sample_labels_by_dataset[dataset_entry['label']] = [str(lbl) for lbl in ds_labels]
                    
                    datasets.append(dataset_entry)
                
                if not datasets:
                    raise ValueError("No valid datasets found. Check your dataset configurations.")
                if not sample_labels_by_dataset:
                    sample_labels_by_dataset = None
            else:
                # Single dataset mode
                if config.get('x_axis'):
                    x_data = extract_axis_data(self.outputs, config['x_axis'], {})
                point_labels_source = config.get('point_labels_source', config.get('sample_labels_source'))
                if point_labels_source and point_labels_source in self.outputs:
                    labels_data = self.outputs[point_labels_source]
                    if isinstance(labels_data, (list, np.ndarray)):
                        sample_labels = [str(lbl) for lbl in labels_data]
            y_data = None
            z_data = None
            
            if config.get('x_axis'):
                x_data = extract_axis_data(self.outputs, config['x_axis'], {})
            
            if config.get('y_axis'):
                y_data = extract_axis_data(self.outputs, config['y_axis'], {}, ref_data=x_data)
            
            if config.get('z_axis'):
                z_data = extract_axis_data(self.outputs, config['z_axis'], {}, 
                                           ref_data=x_data if x_data is not None else y_data)
            
            # Check for data
            if graph_type != 'histogram' and (x_data is None or y_data is None):
                raise ValueError("Failed to extract axis data. Check your data source configuration.")
            
            if graph_type == 'histogram' and y_data is None:
                raise ValueError("Failed to extract Y-axis data for histogram.")
            
            # Build axis configs for renderer
            x_axis_config = config.get('x_axis', {})
            y_axis_config = config.get('y_axis', {})
            z_axis_config = config.get('z_axis', {})
            
            # Render graph
            fig, ax = graph_renderer.render_graph_figure(
                graph_type, config, x_data, y_data, z_data,
                x_axis_config, y_axis_config,
                default_cmap=self.main_gui.settings_manager.get('colormap', 'viridis'),
                datasets=datasets,
                sample_labels=sample_labels,
                sample_labels_by_dataset=sample_labels_by_dataset
            )
            
            # Embed in preview container
            canvas, canvas_frame = graph_renderer.embed_figure_in_tkinter(fig, self.preview_container)
            
            # Store references
            self.preview_canvas = canvas
            self.preview_frame = canvas_frame
            
        except Exception as e:
            # Show error in preview area
            for widget in self.preview_container.winfo_children():
                widget.destroy()
            
            error_label = ttk.Label(self.preview_container, 
                                   text=f"Error: {str(e)}", 
                                   foreground="red",
                                   wraplength=400)
            error_label.pack(expand=True, pady=20)
            
            self._notify(self._t("ui.messages.preview_generate_failed", "Failed to generate preview:") + f"\n\n{str(e)}", level="error")
    
    def _add_graph(self):
        """Add the configured graph to the selected section."""
        try:
            # Get selected section
            section_selection = self.section_var.get()
            if not section_selection:
                self._notify(self._t("ui.messages.select_target_section", "Please select a target section"), level="warning")
                return
            
            # Find the corresponding section indices
            selected_idx = [desc for _, _, desc in self.empty_sections].index(section_selection)
            page_idx, section_idx, _ = self.empty_sections[selected_idx]
            
            # Build final config
            config = self._build_graph_config()
            
            # Validate
            graph_type = config.get('graph_type', 'scatter')
            
            # For multi-dataset, check datasets config
            if config.get('datasets'):
                if not config['datasets']:
                    self._notify(self._t("ui.messages.no_datasets_configured", "No datasets configured. Add at least one dataset in Multi-Dataset tab."), level="warning")
                    return
                # Datasets are already validated when added
            else:
                # Single dataset mode - validate axes
                if graph_type == 'histogram':
                    if not config.get('y_axis'):
                        self._notify(self._t("ui.messages.y_required_histogram", "Y-axis data source required for histogram"), level="warning")
                        return
                else:
                    if not config.get('x_axis') or not config.get('y_axis'):
                        self._notify(self._t("ui.messages.x_y_required_or_datasets", "Both X-axis and Y-axis data sources required (or configure datasets in Multi-Dataset tab)"), level="warning")
                        return
            
            # Update section in analysis data
            pages = self.main_gui.analysis_data[self.instance_alias]['pages']
            pages[page_idx]['sections'][section_idx] = {
                'type': 'graph',
                'config': config
            }
            
            # Close dialog
            self.dialog.destroy()
            
            # Refresh analysis tab
            self.main_gui._show_analysis_tab()
            
            self._notify(self._t("ui.messages.graph_added_to", "Graph added to") + f" {section_selection}", level="success")
            
        except Exception as e:
            self._notify(self._t("ui.messages.add_graph_failed", "Failed to add graph:") + f"\n\n{str(e)}", level="error")
            import traceback
            traceback.print_exc()
    
    def _show_help(self):
        """Show help dialog."""
        help_text = """Add Graph Dialog Help

This dialog allows you to add a new graph to an empty section in your analysis.

1. BASIC TAB:
   - Select the target empty section
   - Choose the graph type (scatter, line, bar, histogram, heatmap, 3d_surf, contour)
   - Set the graph title

2. AXES TAB:
   - Configure X, Y, and Z axes for single-dataset plots
   - Data Source: Select the output variable to use
   - Nested Key: If the data source is a dictionary, select the key
   - Axis Label: Custom label for the axis
    - Axis Type: linear, log10, log2, or ln scale

    Axis label formatting:
    - Greek letters are supported (e.g., α, β, μ)
    - Unicode superscript/subscript is supported (e.g., cm², H₂O)
    - Matplotlib math-text is supported using $...$ (e.g., CO$_2$, x$^2$, $\alpha$)

   Special data sources:
   - __index__: Auto-generate row indices (1, 2, 3, ...)

3. MULTI-DATASET TAB:
   - Configure multiple datasets to plot together
   - Each dataset can have its own marker, color, and data sources
   - Useful for comparing calibration vs validation, or multiple folds
   - If you configure datasets here, single axes (Axes tab) are ignored

4. ADVANCED TAB:
   - Class Labels: Select a data source for coloring points by class
    - Point Labels Source: Select data source used for point labels and scatter tooltips
    - Show point labels (scatter): draw labels directly on each point
     - Confidence ellipses (scatter): draw per-class ellipses (or one global ellipse if no classes)
         * Confidence level (%): defaults to 95 when omitted
        - Scatter Reference Lines (JSON): define vertical/horizontal/path/segment overlays for scatter plots
            * Supports arrays for multiple lines and label arrays
            * Supports style keys: line_type/linestyle, thickness/linewidth, color/colour
            * Supports path boundaries via x/y arrays or points lists
            * Includes quick templates for vertical+horizontal, path, and generic data-source boundaries
   - Data Slicing/Navigation: Enable navigation controls for multi-dimensional data
     * Enable axis navigation (X/Y/Z) with dimension and default value
     * Dim 0=samples, 1=first variable dimension (e.g., PCs)
   - Multi-Dimensional Slicing: Specify fixed indices for extra dimensions
     * Format: "0:5" or "2:10,3:0" (dimension:index pairs)

5. PREVIEW:
   - Click "Update Preview" to see your graph
   - Make adjustments as needed
   - No popup message - preview updates silently

6. ADD GRAPH:
   - Click "Add Graph" to add it to the selected section
   - The analysis tab will refresh automatically

Tips:
- Run "Run to here" or "Run Model" first to populate available data sources
- For scatter plots with PC data, enable axis navigation to select which PCs to plot
- Use class labels to color points by category
- Multi-dataset plots automatically handle different markers and colors
- Navigation controls appear as dropdown menus above the graph
"""
        
        help_dialog = tk.Toplevel(self.dialog)
        _set_window_icon(help_dialog, "Info")
        help_dialog.title(self._t("ui.buttons.help", "Help"))
        help_dialog.geometry("700x600")
        self._center_window(help_dialog, 700, 600)
        
        text = scrolledtext.ScrolledText(help_dialog, wrap=tk.WORD, font=("Arial", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(1.0, help_text)
        text.config(state=tk.DISABLED)
        
        close_btn = ttk.Button(help_dialog, text=self._t("ui.buttons.close", "Close"), command=help_dialog.destroy)
        close_btn.pack(pady=10)


def show_add_graph_dialog(parent, main_gui, instance_alias: str):
    """Show the Add Graph dialog.
    
    Args:
        parent: Parent tkinter window
        main_gui: Reference to main ChemometricsGUI instance
        instance_alias: Alias of the current function instance
    """
    AddGraphDialog(parent, main_gui, instance_alias)
