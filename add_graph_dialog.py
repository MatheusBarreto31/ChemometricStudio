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
from typing import Optional, Dict, List, Tuple, Any
import graph_renderer


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
            messagebox.showerror("Error", "No data available. Please run 'Run Model' or 'Run to here' first.")
            return
        
        # Find empty sections
        self.empty_sections = self._find_empty_sections()
        if not self.empty_sections:
            messagebox.showerror("Error", "No empty sections available. Add a new page or remove existing sections first.")
            return
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add Graph")
        self.dialog.geometry("900x700")
        
        # Graph configuration
        self.graph_config = {
            'graph_type': 'scatter',
            'x_axis': {},
            'y_axis': {},
            'z_axis': {},
            'title': 'New Graph',
            'data_slicing': []
        }
        
        # Multi-dataset configuration
        self.datasets_configs = []
        
        # Preview variables
        self.preview_canvas = None
        self.preview_frame = None
        
        # Build UI
        self._build_ui()
    
    def _get_execution_outputs(self) -> Optional[Dict]:
        """Get execution outputs from analysis data."""
        if self.instance_alias not in self.main_gui.analysis_data:
            return None
        execution_results = self.main_gui.analysis_data[self.instance_alias].get('execution_results', {})
        if execution_results.get('status') != 'success':
            return None
        return execution_results.get('outputs', {})
    
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
        """Get list of available data sources from outputs."""
        if not self.outputs:
            return []
        return sorted(list(self.outputs.keys()))
    
    def _get_nested_keys(self, data_source: str) -> List[str]:
        """Get nested keys if data source is a dictionary."""
        if not data_source or data_source not in self.outputs:
            return []
        
        data = self.outputs[data_source]
        if isinstance(data, dict):
            return sorted(list(data.keys()))
        return []
    
    def _get_data_shape_info(self, data_source: str, nested_key: str = None) -> str:
        """Get shape information for a data source."""
        if not data_source or data_source not in self.outputs:
            return "N/A"
        
        data = self.outputs[data_source]
        if nested_key and isinstance(data, dict) and nested_key in data:
            data = data[nested_key]
        
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
        notebook.add(basic_tab, text="Basic")
        self._build_basic_tab(basic_tab)
        
        # Tab 2: Axes Configuration
        axes_tab = ttk.Frame(notebook)
        notebook.add(axes_tab, text="Axes")
        self._build_axes_tab(axes_tab)
        
        # Tab 3: Multi-Dataset Configuration
        datasets_tab = ttk.Frame(notebook)
        notebook.add(datasets_tab, text="Multi-Dataset")
        self._build_datasets_tab(datasets_tab)
        
        # Tab 4: Advanced Options
        advanced_tab = ttk.Frame(notebook)
        notebook.add(advanced_tab, text="Advanced")
        self._build_advanced_tab(advanced_tab)
    
    def _build_basic_tab(self, parent):
        """Build basic configuration tab."""
        # Scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Section selection
        section_frame = ttk.LabelFrame(scrollable_frame, text="Target Section", padding=10)
        section_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(section_frame, text="Select empty section:").pack(anchor=tk.W)
        self.section_var = tk.StringVar()
        section_combo = ttk.Combobox(section_frame, textvariable=self.section_var, state="readonly", width=40)
        section_combo['values'] = [desc for _, _, desc in self.empty_sections]
        if self.empty_sections:
            section_combo.current(0)
        section_combo.pack(fill=tk.X, pady=5)
        
        # Graph type selection
        type_frame = ttk.LabelFrame(scrollable_frame, text="Graph Type", padding=10)
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
        title_frame = ttk.LabelFrame(scrollable_frame, text="Graph Title", padding=10)
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.title_var = tk.StringVar(value="New Graph")
        title_entry = ttk.Entry(title_frame, textvariable=self.title_var, width=50)
        title_entry.pack(fill=tk.X)
    
    def _build_axes_tab(self, parent):
        """Build axes configuration tab."""
        # Scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
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
        ttk.Label(frame, text="Data Source:").grid(row=0, column=0, sticky=tk.W, pady=2)
        data_sources = [''] + self._get_available_data_sources() + ['__index__']
        
        axis_var = tk.StringVar()
        setattr(self, f'{axis_key}_data_source_var', axis_var)
        
        data_combo = ttk.Combobox(frame, textvariable=axis_var, values=data_sources, width=30)
        data_combo.grid(row=0, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        data_combo.bind('<<ComboboxSelected>>', lambda e: self._on_data_source_changed(axis_key))
        
        # Nested key (for dict sources)
        ttk.Label(frame, text="Nested Key:").grid(row=1, column=0, sticky=tk.W, pady=2)
        nested_var = tk.StringVar()
        setattr(self, f'{axis_key}_nested_key_var', nested_var)
        
        nested_combo = ttk.Combobox(frame, textvariable=nested_var, width=30)
        nested_combo.grid(row=1, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        setattr(self, f'{axis_key}_nested_combo', nested_combo)
        
        # Index (for list/array sources) - removed as it's too simplistic
        # Users should use default column or axis navigation instead
        
        # Axis label
        ttk.Label(frame, text="Axis Label:").grid(row=2, column=0, sticky=tk.W, pady=2)
        label_var = tk.StringVar(value=label.replace('-Axis', ''))
        setattr(self, f'{axis_key}_label_var', label_var)
        
        label_entry = ttk.Entry(frame, textvariable=label_var, width=30)
        label_entry.grid(row=2, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        
        # Data shape info
        info_var = tk.StringVar(value="Select a data source")
        setattr(self, f'{axis_key}_info_var', info_var)
        
        info_label = ttk.Label(frame, textvariable=info_var, foreground="gray", font=("Arial", 8))
        info_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        frame.columnconfigure(1, weight=1)
    
    def _build_datasets_tab(self, parent):
        """Build multi-dataset configuration tab."""
        # Scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Info label
        info_frame = ttk.Frame(scrollable_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(info_frame, text="Multi-Dataset Plots", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(info_frame, 
                 text="Configure multiple datasets to plot on the same graph (e.g., calibration vs validation).",
                 wraplength=450, foreground="gray").pack(anchor=tk.W, pady=(2, 5))
        
        ttk.Label(info_frame,
                 text="Note: If you configure datasets here, the single-dataset axes (Axes tab) will be ignored.",
                 wraplength=450, font=("Arial", 9, "italic"), foreground="orange").pack(anchor=tk.W)
        
        # Datasets list frame
        list_frame = ttk.LabelFrame(scrollable_frame, text="Datasets", padding=10)
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
        
        ttk.Button(btn_frame, text="➕ Add Dataset", 
                  command=self._add_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="✏ Edit Selected", 
                  command=self._edit_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🗑 Remove Selected", 
                  command=self._remove_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear All", 
                  command=self._clear_datasets).pack(side=tk.LEFT, padx=2)
    
    def _add_dataset(self):
        """Add a new dataset configuration."""
        self._show_dataset_config_dialog()
    
    def _edit_dataset(self):
        """Edit the selected dataset."""
        selection = self.datasets_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a dataset to edit")
            return
        
        idx = selection[0]
        existing_config = self.datasets_configs[idx] if idx < len(self.datasets_configs) else None
        self._show_dataset_config_dialog(idx, existing_config)
    
    def _remove_dataset(self):
        """Remove the selected dataset."""
        selection = self.datasets_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a dataset to remove")
            return
        
        idx = selection[0]
        if idx < len(self.datasets_configs):
            del self.datasets_configs[idx]
            self._refresh_datasets_list()
    
    def _clear_datasets(self):
        """Clear all datasets."""
        if self.datasets_configs and messagebox.askyesno("Confirm", "Clear all datasets?"):
            self.datasets_configs = []
            self._refresh_datasets_list()
    
    def _show_dataset_config_dialog(self, edit_idx=None, existing_config=None):
        """Show dialog to configure a single dataset."""
        dialog = tk.Toplevel(self.dialog)
        dialog.title("Configure Dataset" if edit_idx is None else "Edit Dataset")
        dialog.geometry("500x600")
        
        # Scrollable frame
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Dataset label
        label_frame = ttk.LabelFrame(scrollable_frame, text="Dataset Label", padding=10)
        label_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(label_frame, text="Label (for legend):").pack(anchor=tk.W)
        label_var = tk.StringVar(value=existing_config.get('label', '') if existing_config else '')
        label_entry = ttk.Entry(label_frame, textvariable=label_var, width=40)
        label_entry.pack(fill=tk.X, pady=2)
        
        # Marker
        marker_frame = ttk.LabelFrame(scrollable_frame, text="Marker Style", padding=10)
        marker_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(marker_frame, text="Marker:").pack(anchor=tk.W)
        marker_var = tk.StringVar(value=existing_config.get('marker', 'o') if existing_config else 'o')
        markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', 'H', '+', 'x']
        marker_combo = ttk.Combobox(marker_frame, textvariable=marker_var, values=markers, width=10)
        marker_combo.pack(fill=tk.X, pady=2)
        
        # Color (optional)
        color_frame = ttk.LabelFrame(scrollable_frame, text="Color (Optional)", padding=10)
        color_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(color_frame, text="Color (leave blank for auto):").pack(anchor=tk.W)
        color_var = tk.StringVar(value=existing_config.get('color', '') if existing_config else '')
        color_entry = ttk.Entry(color_frame, textvariable=color_var, width=20)
        color_entry.pack(fill=tk.X, pady=2)
        ttk.Label(color_frame, text="Examples: #1f77b4, red, blue", 
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        
        # X-axis config
        x_frame = ttk.LabelFrame(scrollable_frame, text="X-Axis Configuration", padding=10)
        x_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(x_frame, text="Data Source:").pack(anchor=tk.W)
        x_source_var = tk.StringVar(value=existing_config.get('x_axis', {}).get('data_source', '') if existing_config else '')
        x_source_combo = ttk.Combobox(x_frame, textvariable=x_source_var, 
                                      values=[''] + self._get_available_data_sources(), width=40)
        x_source_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(x_frame, text="Nested Key (optional):").pack(anchor=tk.W, pady=(5, 0))
        x_nested_var = tk.StringVar(value=existing_config.get('x_axis', {}).get('nested_key', '') if existing_config else '')
        x_nested_entry = ttk.Entry(x_frame, textvariable=x_nested_var, width=40)
        x_nested_entry.pack(fill=tk.X, pady=2)
        
        # Y-axis config
        y_frame = ttk.LabelFrame(scrollable_frame, text="Y-Axis Configuration", padding=10)
        y_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(y_frame, text="Data Source:").pack(anchor=tk.W)
        y_source_var = tk.StringVar(value=existing_config.get('y_axis', {}).get('data_source', '') if existing_config else '')
        y_source_combo = ttk.Combobox(y_frame, textvariable=y_source_var,
                                      values=[''] + self._get_available_data_sources(), width=40)
        y_source_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(y_frame, text="Nested Key (optional):").pack(anchor=tk.W, pady=(5, 0))
        y_nested_var = tk.StringVar(value=existing_config.get('y_axis', {}).get('nested_key', '') if existing_config else '')
        y_nested_entry = ttk.Entry(y_frame, textvariable=y_nested_var, width=40)
        y_nested_entry.pack(fill=tk.X, pady=2)
        
        # Z-axis config (optional)
        z_frame = ttk.LabelFrame(scrollable_frame, text="Z-Axis Configuration (Optional, for 3D)", padding=10)
        z_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(z_frame, text="Data Source:").pack(anchor=tk.W)
        z_source_var = tk.StringVar(value=existing_config.get('z_axis', {}).get('data_source', '') if existing_config else '')
        z_source_combo = ttk.Combobox(z_frame, textvariable=z_source_var,
                                      values=[''] + self._get_available_data_sources(), width=40)
        z_source_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(z_frame, text="Nested Key (optional):").pack(anchor=tk.W, pady=(5, 0))
        z_nested_var = tk.StringVar(value=existing_config.get('z_axis', {}).get('nested_key', '') if existing_config else '')
        z_nested_entry = ttk.Entry(z_frame, textvariable=z_nested_var, width=40)
        z_nested_entry.pack(fill=tk.X, pady=2)
        
        # Class labels (optional)
        class_frame = ttk.LabelFrame(scrollable_frame, text="Class Labels (Optional)", padding=10)
        class_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(class_frame, text="Class Labels Source:").pack(anchor=tk.W)
        class_var = tk.StringVar(value=existing_config.get('class_labels', '') if existing_config else '')
        class_combo = ttk.Combobox(class_frame, textvariable=class_var,
                                   values=[''] + self._get_available_data_sources(), width=40)
        class_combo.pack(fill=tk.X, pady=2)
        
        # Sample labels (optional)
        sample_frame = ttk.LabelFrame(scrollable_frame, text="Sample Labels (Optional)", padding=10)
        sample_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sample_frame, text="Sample Labels Source:").pack(anchor=tk.W)
        sample_var = tk.StringVar(value=existing_config.get('sample_labels_source', '') if existing_config else '')
        sample_combo = ttk.Combobox(sample_frame, textvariable=sample_var,
                                    values=[''] + self._get_available_data_sources(), width=40)
        sample_combo.pack(fill=tk.X, pady=2)
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        def save_dataset():
            # Build dataset config
            ds_config = {
                'label': label_var.get() or 'Dataset',
                'marker': marker_var.get() or 'o'
            }
            
            if color_var.get():
                ds_config['color'] = color_var.get()
            
            # X-axis
            if x_source_var.get():
                x_axis = {'data_source': x_source_var.get()}
                if x_nested_var.get():
                    x_axis['nested_key'] = x_nested_var.get()
                ds_config['x_axis'] = x_axis
            
            # Y-axis
            if y_source_var.get():
                y_axis = {'data_source': y_source_var.get()}
                if y_nested_var.get():
                    y_axis['nested_key'] = y_nested_var.get()
                ds_config['y_axis'] = y_axis
            
            # Z-axis
            if z_source_var.get():
                z_axis = {'data_source': z_source_var.get()}
                if z_nested_var.get():
                    z_axis['nested_key'] = z_nested_var.get()
                ds_config['z_axis'] = z_axis
            
            # Class labels
            if class_var.get():
                ds_config['class_labels'] = class_var.get()
            
            # Sample labels
            if sample_var.get():
                ds_config['sample_labels_source'] = sample_var.get()
            
            # Validate
            if not ds_config.get('x_axis') or not ds_config.get('y_axis'):
                messagebox.showerror("Error", "Both X and Y axis data sources are required")
                return
            
            # Add or update
            if edit_idx is not None and edit_idx < len(self.datasets_configs):
                self.datasets_configs[edit_idx] = ds_config
            else:
                self.datasets_configs.append(ds_config)
            
            self._refresh_datasets_list()
            dialog.destroy()
        
        ttk.Button(btn_frame, text="✓ Save", command=save_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="✗ Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _refresh_datasets_list(self):
        """Refresh the datasets listbox."""
        self.datasets_listbox.delete(0, tk.END)
        for idx, ds in enumerate(self.datasets_configs):
            label = ds.get('label', f'Dataset {idx+1}')
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
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Class labels
        class_frame = ttk.LabelFrame(scrollable_frame, text="Class Labels (for coloring)", padding=10)
        class_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(class_frame, text="Class Data Source:").pack(anchor=tk.W)
        self.class_labels_var = tk.StringVar()
        class_combo = ttk.Combobox(class_frame, textvariable=self.class_labels_var, 
                                   values=[''] + self._get_available_data_sources(), width=40)
        class_combo.pack(fill=tk.X, pady=5)
        
        # Sample labels
        sample_frame = ttk.LabelFrame(scrollable_frame, text="Sample Labels (for tooltips)", padding=10)
        sample_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sample_frame, text="Sample Labels Source:").pack(anchor=tk.W)
        self.sample_labels_var = tk.StringVar()
        sample_combo = ttk.Combobox(sample_frame, textvariable=self.sample_labels_var,
                                    values=[''] + self._get_available_data_sources(), width=40)
        sample_combo.pack(fill=tk.X, pady=5)
        
        # Data slicing configuration
        slice_frame = ttk.LabelFrame(scrollable_frame, text="Data Slicing / Navigation", padding=10)
        slice_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(slice_frame, text="Enable navigation for exploring multi-dimensional data:", 
                 wraplength=400).pack(anchor=tk.W, pady=5)
        
        # Axis navigation checkboxes with dimension entry
        axes_config = [('x', 'X-axis'), ('y', 'Y-axis'), ('z', 'Z-axis')]
        
        for axis_key, axis_label in axes_config:
            axis_frame = ttk.Frame(slice_frame)
            axis_frame.pack(fill=tk.X, pady=3)
            
            # Checkbox for enabling navigation on this axis
            nav_var = tk.BooleanVar(value=False)
            setattr(self, f'{axis_key}_nav_enabled_var', nav_var)
            cb = ttk.Checkbutton(axis_frame, text=f"{axis_label} Navigation", variable=nav_var)
            cb.pack(side=tk.LEFT)
            
            # Dimension entry
            ttk.Label(axis_frame, text="Dim:").pack(side=tk.LEFT, padx=(10, 2))
            dim_var = tk.StringVar(value="0")
            setattr(self, f'{axis_key}_nav_dim_var', dim_var)
            dim_entry = ttk.Entry(axis_frame, textvariable=dim_var, width=8)
            dim_entry.pack(side=tk.LEFT, padx=2)
            
            # Default value entry
            ttk.Label(axis_frame, text="Default:").pack(side=tk.LEFT, padx=(10, 2))
            default_var = tk.StringVar(value="0")
            setattr(self, f'{axis_key}_nav_default_var', default_var)
            
            default_entry = ttk.Entry(axis_frame, textvariable=default_var, width=8)
            default_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(slice_frame, text="Tip: Dim 0=samples, 1=first variable dimension (e.g., PCs)", 
                 font=("Arial", 8), foreground="gray", wraplength=400).pack(anchor=tk.W, pady=(5, 0))
        
        # Multi-Dimensional Slicing for 4D+ Data
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)
        
        md_slice_frame = ttk.LabelFrame(scrollable_frame, text="Multi-Dimensional Slicing (4D+ Data)", padding=10)
        md_slice_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(md_slice_frame, text="For 4D+ heatmaps/contours, configure dimension combinations and slicing:", 
                 wraplength=450).pack(anchor=tk.W, pady=(0, 10))
        
        # Enable 4D+ multi-dimensional slicing
        self.enable_md_slicing_var = tk.BooleanVar(value=False)
        md_enable_cb = ttk.Checkbutton(md_slice_frame, text="Enable 4D+ Multi-Dimensional Slicing", 
                                       variable=self.enable_md_slicing_var, command=self._toggle_md_config)
        md_enable_cb.pack(anchor=tk.W, pady=5)
        
        # Container for MD configuration (hidden by default)
        self.md_config_frame = ttk.Frame(md_slice_frame)
        
        # Dimension combination selector
        combo_frame = ttk.Frame(self.md_config_frame)
        combo_frame.pack(fill=tk.X, pady=5)
        ttk.Label(combo_frame, text="Dimension Combination Index:", width=25).pack(side=tk.LEFT, padx=(0, 5))
        self.md_combo_index_var = tk.StringVar(value="0")
        ttk.Entry(combo_frame, textvariable=self.md_combo_index_var, width=10).pack(side=tk.LEFT)
        ttk.Label(combo_frame, text="(0 = first combination)", font=("Arial", 8), 
                 foreground="gray").pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(self.md_config_frame, text="For heatmaps/contours: 2 dims used for X/Y axes, others are navigable.", 
                 font=("Arial", 8), foreground="gray", wraplength=450).pack(anchor=tk.W, pady=(5, 10))
        
        # Default slice indices for navigable dimensions
        ttk.Label(self.md_config_frame, text="Default slice indices for navigable dimensions:").pack(anchor=tk.W)
        ttk.Label(self.md_config_frame, text="Format: dim:index (e.g., '2:0,3:5' means dim 2→index 0, dim 3→index 5)", 
                 font=("Arial", 8), foreground="gray", wraplength=450).pack(anchor=tk.W, pady=(0, 5))
        
        self.md_slice_indices_var = tk.StringVar()
        md_slice_entry = ttk.Entry(self.md_config_frame, textvariable=self.md_slice_indices_var, width=50)
        md_slice_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.md_config_frame, text="Leave blank to use default (0) for all navigable dimensions.", 
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
        title = ttk.Label(parent, text="Preview", font=("Arial", 12, "bold"))
        title.pack(pady=10)
        
        # Preview button
        preview_btn = ttk.Button(parent, text="🔄 Update Preview", command=self._update_preview)
        preview_btn.pack(pady=5)
        
        # Preview container
        preview_container = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=2)
        preview_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.preview_container = preview_container
        
        # Initial message
        msg = ttk.Label(preview_container, text="Configure your graph and click 'Update Preview'",
                       foreground="gray", font=("Arial", 10, "italic"))
        msg.pack(expand=True)
    
    def _build_button_bar(self):
        """Build bottom button bar."""
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        # Add Graph button
        add_btn = ttk.Button(button_frame, text="✓ Add Graph", command=self._add_graph)
        add_btn.pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        cancel_btn = ttk.Button(button_frame, text="✗ Cancel", command=self.dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Spacer
        spacer = ttk.Frame(button_frame)
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Help button
        help_btn = ttk.Button(button_frame, text="? Help", command=self._show_help)
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
            info_var.set("Select a data source")
    
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
            
            config['z_axis'] = z_config
        
        # Class labels
        class_source = self.class_labels_var.get()
        if class_source:
            config['class_labels'] = class_source
        
        # Sample labels
        sample_source = self.sample_labels_var.get()
        if sample_source:
            config['sample_labels_source'] = sample_source
        
        # Data slicing - build navigation controls
        data_slicing = []
        
        for axis_key, axis_label in [('x', 'X-Axis'), ('y', 'Y-Axis'), ('z', 'Z-Axis')]:
            enable_var = getattr(self, f'{axis_key}_nav_enabled_var', None)
            if enable_var and enable_var.get():
                try:
                    dimension = int(getattr(self, f'{axis_key}_nav_dim_var').get())
                    default = int(getattr(self, f'{axis_key}_nav_default_var').get())
                    
                    data_slicing.append({
                        'name': axis_label,
                        'dimension': dimension,
                        'axis': axis_key,
                        'default': default,
                        'show_navigation_menu': True
                    })
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
            x_data = None
            y_data = None
            z_data = None
            
            if config.get('datasets'):
                # Multi-dataset mode: extract data for each dataset
                datasets = []
                for ds_cfg in config['datasets']:
                    ds_x = None
                    ds_y = None
                    ds_z = None
                    
                    if ds_cfg.get('x_axis'):
                        ds_x = self.main_gui._extract_axis_data(self.outputs, ds_cfg['x_axis'], {})
                    if ds_cfg.get('y_axis'):
                        ds_y = self.main_gui._extract_axis_data(self.outputs, ds_cfg['y_axis'], {}, ref_data=ds_x)
                    if ds_cfg.get('z_axis'):
                        ds_z = self.main_gui._extract_axis_data(self.outputs, ds_cfg['z_axis'], {}, 
                                                                ref_data=ds_x if ds_x is not None else ds_y)
                    
                    if ds_x is None or ds_y is None:
                        continue  # Skip datasets with missing data
                    
                    # Extract class data if specified
                    ds_class_data = None
                    if ds_cfg.get('class_labels'):
                        class_source = ds_cfg['class_labels']
                        if class_source in self.outputs:
                            class_val = self.outputs[class_source]
                            if isinstance(class_val, (list, np.ndarray)):
                                ds_class_data = np.array(class_val)
                    
                    dataset_entry = {
                        'x_data': ds_x,
                        'y_data': ds_y,
                        'z_data': ds_z,
                        'label': ds_cfg.get('label', 'Dataset'),
                        'marker': ds_cfg.get('marker', 'o'),
                        'class_data': ds_class_data
                    }
                    
                    if ds_cfg.get('color'):
                        dataset_entry['color'] = ds_cfg['color']
                    
                    datasets.append(dataset_entry)
                
                if not datasets:
                    raise ValueError("No valid datasets found. Check your dataset configurations.")
            else:
                # Single dataset mode
                if config.get('x_axis'):
                    x_data = self.main_gui._extract_axis_data(self.outputs, config['x_axis'], {})
            y_data = None
            z_data = None
            
            if config.get('x_axis'):
                x_data = self.main_gui._extract_axis_data(self.outputs, config['x_axis'], {})
            
            if config.get('y_axis'):
                y_data = self.main_gui._extract_axis_data(self.outputs, config['y_axis'], {}, ref_data=x_data)
            
            if config.get('z_axis'):
                z_data = self.main_gui._extract_axis_data(self.outputs, config['z_axis'], {}, 
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
                datasets=datasets
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
            
            messagebox.showerror("Preview Error", f"Failed to generate preview:\n\n{str(e)}")
    
    def _add_graph(self):
        """Add the configured graph to the selected section."""
        try:
            # Get selected section
            section_selection = self.section_var.get()
            if not section_selection:
                messagebox.showerror("Error", "Please select a target section")
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
                    messagebox.showerror("Error", "No datasets configured. Add at least one dataset in Multi-Dataset tab.")
                    return
                # Datasets are already validated when added
            else:
                # Single dataset mode - validate axes
                if graph_type == 'histogram':
                    if not config.get('y_axis'):
                        messagebox.showerror("Error", "Y-axis data source required for histogram")
                        return
                else:
                    if not config.get('x_axis') or not config.get('y_axis'):
                        messagebox.showerror("Error", "Both X-axis and Y-axis data sources required (or configure datasets in Multi-Dataset tab)")
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
            
            messagebox.showinfo("Success", f"Graph added to {section_selection}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add graph:\n\n{str(e)}")
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

   Special data sources:
   - __index__: Auto-generate row indices (1, 2, 3, ...)

3. MULTI-DATASET TAB:
   - Configure multiple datasets to plot together
   - Each dataset can have its own marker, color, and data sources
   - Useful for comparing calibration vs validation, or multiple folds
   - If you configure datasets here, single axes (Axes tab) are ignored

4. ADVANCED TAB:
   - Class Labels: Select a data source for coloring points by class
   - Sample Labels: Select a data source for sample tooltips
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
        help_dialog.title("Help")
        help_dialog.geometry("700x600")
        
        text = scrolledtext.ScrolledText(help_dialog, wrap=tk.WORD, font=("Arial", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(1.0, help_text)
        text.config(state=tk.DISABLED)
        
        close_btn = ttk.Button(help_dialog, text="Close", command=help_dialog.destroy)
        close_btn.pack(pady=10)


def show_add_graph_dialog(parent, main_gui, instance_alias: str):
    """Show the Add Graph dialog.
    
    Args:
        parent: Parent tkinter window
        main_gui: Reference to main ChemometricsGUI instance
        instance_alias: Alias of the current function instance
    """
    AddGraphDialog(parent, main_gui, instance_alias)
