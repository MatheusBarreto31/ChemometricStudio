"""
Add Table Dialog Module

Provides a dialog for adding tables to the analysis tab with:
- Data source selection
- Table formatting options
- Data slicing/navigation for 3D+ data
- Column and row headers
- Multi-column table support
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from pathlib import Path
import platform
from typing import Optional, Dict, List, Tuple
from dialog_data_source_utils import (
    append_prefixed_data_sources,
    collect_nested_key_paths,
    get_available_data_sources,
    get_data_source_value,
    get_nested_keys,
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


class AddTableDialog:
    """Dialog for adding a new table to an empty section in the analysis tab."""
    
    def __init__(self, parent, main_gui, instance_alias: str):
        """Initialize the Add Table dialog.
        
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
        self.dialog.title(self._t("ui.dialogs.add_table", "Add Table"))
        self.dialog.geometry("800x552")
        self._center_window(self.dialog, 800, 552)
        
        # Multi-column configuration
        self.columns_configs = []
        
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
        if hasattr(self.main_gui, '_get_execution_data_sources'):
            combined_sources = self.main_gui._get_execution_data_sources(execution_results, self.instance_alias)
            if not isinstance(combined_sources, dict):
                combined_sources = {}
            self._append_prefixed_data_sources(combined_sources, execution_results)
            return combined_sources

        inputs = execution_results.get('inputs', {})
        outputs = execution_results.get('outputs', {})
        combined_sources = {}
        if isinstance(inputs, dict):
            combined_sources.update(inputs)
        if isinstance(outputs, dict):
            combined_sources.update(outputs)
        self._append_prefixed_data_sources(combined_sources, execution_results)
        return combined_sources

    def _append_prefixed_data_sources(self, combined_sources: Dict, execution_results: Dict) -> None:
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
        """Get list of available data sources from outputs."""
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
        if not data_source:
            return "N/A"

        data = self._get_data_source_value(data_source)
        if data is None:
            return "N/A"
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
        # Main container with paned window for left (controls) and right (info)
        paned = ttk.PanedWindow(self.dialog, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Configuration controls
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=2)
        
        # Right panel: Data info
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Build panels
        self._build_config_panel(left_frame)
        self._build_info_panel(right_frame)
        self._build_button_bar()
    
    def _build_config_panel(self, parent):
        """Build the configuration panel."""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tabs
        basic_tab = ttk.Frame(notebook)
        multicolumn_tab = ttk.Frame(notebook)
        advanced_tab = ttk.Frame(notebook)
        
        notebook.add(basic_tab, text=self._t("ui.tabs.basic", "Basic"))
        notebook.add(multicolumn_tab, text=self._t("ui.tabs.multi_column", "Multi-Column"))
        notebook.add(advanced_tab, text=self._t("ui.tabs.advanced", "Advanced"))
        
        self._build_basic_tab(basic_tab)
        self._build_multicolumn_tab(multicolumn_tab)
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
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif getattr(event, 'num', None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, 'num', None) == 5:
                canvas.yview_scroll(1, "units")
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
    
    def _build_basic_tab(self, parent):
        """Build the basic configuration tab."""
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
        self._bind_canvas_mousewheel(canvas, scrollable_frame)
        
        # Section selection
        section_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.target_section", "Target Section"), padding=10)
        section_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(section_frame, text=self._t("ui.messages.add_table_to", "Add table to:")).pack(anchor=tk.W, pady=5)
        
        self.section_var = tk.StringVar()
        section_combo = ttk.Combobox(section_frame, textvariable=self.section_var, 
                                     values=[desc for _, _, desc in self.empty_sections],
                                     state='readonly', width=50)
        section_combo.pack(fill=tk.X, pady=5)
        if self.empty_sections:
            section_combo.current(0)
        
        # Table title
        title_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.table_title", "Table Title"), padding=10)
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(title_frame, text=self._t("ui.labels.title_optional", "Title (optional):")).pack(anchor=tk.W, pady=5)
        self.title_var = tk.StringVar(value=self._t("ui.labels.data_table", "Data Table"))
        ttk.Entry(title_frame, textvariable=self.title_var, width=50).pack(fill=tk.X, pady=5)
        
        # Data source selection
        data_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.input_output_source", "Input/Output Source"), padding=10)
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(data_frame, text=self._t("ui.labels.input_output_source_single", "Input/Output Source (single-column mode):")).pack(anchor=tk.W, pady=5)
        ttk.Label(data_frame, text=self._t("ui.messages.use_multicolumn_for_multiple", "Note: Use Multi-Column tab for tables with multiple columns"), 
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W, pady=(0, 5))
        self.data_source_var = tk.StringVar()
        data_combo = ttk.Combobox(data_frame, textvariable=self.data_source_var,
                                  values=self._get_available_data_sources(),
                                  state='readonly', width=50)
        data_combo.pack(fill=tk.X, pady=5)
        data_combo.bind('<<ComboboxSelected>>', self._on_data_source_change)
        
        # Nested key selection (for dictionary data)
        ttk.Label(data_frame, text=self._t("ui.labels.nested_key_if_dict", "Nested Key (if data is dictionary):")).pack(anchor=tk.W, pady=(10, 5))
        self.nested_key_var = tk.StringVar()
        self.nested_key_combo = ttk.Combobox(data_frame, textvariable=self.nested_key_var,
                                            state='readonly', width=50)
        self.nested_key_combo.pack(fill=tk.X, pady=5)
        
        # Formatting options
        format_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.formatting", "Formatting"), padding=10)
        format_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Decimal places
        dec_frame = ttk.Frame(format_frame)
        dec_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dec_frame, text=self._t("ui.labels.decimal_places", "Decimal Places:"), width=20).pack(side=tk.LEFT)
        self.decimal_var = tk.StringVar(value="4")
        ttk.Spinbox(dec_frame, from_=0, to=10, textvariable=self.decimal_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Max rows
        rows_frame = ttk.Frame(format_frame)
        rows_frame.pack(fill=tk.X, pady=5)
        ttk.Label(rows_frame, text=self._t("ui.labels.max_rows", "Max Rows:"), width=20).pack(side=tk.LEFT)
        self.max_rows_var = tk.StringVar(value="50")
        ttk.Spinbox(rows_frame, from_=10, to=1000, textvariable=self.max_rows_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Max columns
        cols_frame = ttk.Frame(format_frame)
        cols_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cols_frame, text=self._t("ui.labels.max_columns", "Max Columns:"), width=20).pack(side=tk.LEFT)
        self.max_cols_var = tk.StringVar(value="15")
        ttk.Spinbox(cols_frame, from_=5, to=100, textvariable=self.max_cols_var, width=10).pack(side=tk.LEFT, padx=5)
    
    def _build_multicolumn_tab(self, parent):
        """Build the multi-column configuration tab."""
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
        self._bind_canvas_mousewheel(canvas, scrollable_frame)
        
        # Info
        info_frame = ttk.Frame(scrollable_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(info_frame, text=self._t("ui.labels.multi_column_table_config", "Multi-Column Table Configuration"), 
                 font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(info_frame, 
                  text=self._t("ui.messages.multi_column_table_help", "Create a table with multiple columns from different data sources.\nEach column can have its own data source and name."),
                 wraplength=600, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 10))
        
        # Column list
        list_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.configured_columns", "Configured Columns"), padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        list_scrollbar = ttk.Scrollbar(list_container)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.columns_listbox = tk.Listbox(list_container, yscrollcommand=list_scrollbar.set, height=10)
        self.columns_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.config(command=self.columns_listbox.yview)
        
        # Buttons
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text=self._t("ui.buttons.add_column", "Add Column"), command=self._add_column_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text=self._t("ui.buttons.edit_column", "Edit Column"), command=self._edit_column_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text=self._t("ui.buttons.remove_column", "Remove Column"), command=self._remove_column).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text=self._t("ui.buttons.clear_all", "Clear All"), command=self._clear_columns).pack(side=tk.LEFT, padx=2)
    
    def _add_column_dialog(self):
        """Show dialog to add a new column."""
        dialog = tk.Toplevel(self.dialog)
        _set_window_icon(dialog, "Icon")
        dialog.title(self._t("ui.dialogs.add_column", "Add Column"))
        dialog.geometry("400x300")
        self._center_window(dialog, 400, 300)
        dialog.transient(self.dialog)
        dialog.grab_set()
        
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Column name
        ttk.Label(frame, text=self._t("ui.labels.column_name", "Column Name:")).pack(anchor=tk.W, pady=(0, 5))
        name_var = tk.StringVar(value=f"{self._t('ui.labels.column', 'Column')} {len(self.columns_configs) + 1}")
        ttk.Entry(frame, textvariable=name_var, width=40).pack(fill=tk.X, pady=(0, 10))
        
        # Data source
        ttk.Label(frame, text=self._t("ui.labels.input_output_source_colon", "Input/Output Source:")).pack(anchor=tk.W, pady=(0, 5))
        source_var = tk.StringVar()
        source_combo = ttk.Combobox(frame, textvariable=source_var,
                                   values=self._get_available_data_sources(),
                                   state='readonly', width=40)
        source_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Nested key
        ttk.Label(frame, text=self._t("ui.labels.nested_key_optional", "Nested Key (optional):")).pack(anchor=tk.W, pady=(0, 5))
        nested_var = tk.StringVar()
        nested_combo = ttk.Combobox(frame, textvariable=nested_var, state='readonly', width=40)
        nested_combo.pack(fill=tk.X, pady=(0, 10))
        
        def update_nested_keys(event=None):
            source = source_var.get()
            if source:
                keys = self._get_nested_keys(source)
                nested_combo.config(values=keys)
                if not keys:
                    nested_combo.set('')
        
        source_combo.bind('<<ComboboxSelected>>', update_nested_keys)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        def add_column():
            name = name_var.get().strip()
            source = source_var.get()
            nested = nested_var.get()
            
            if not name or not source:
                self._notify(self._t("ui.messages.provide_column_name_source", "Please provide column name and input/output source"), level="warning")
                return
            
            col_config = {
                'name': name,
                'data_source': source
            }
            if nested:
                col_config['nested_key'] = nested
            
            self.columns_configs.append(col_config)
            self._update_columns_list()
            dialog.destroy()
        
        ttk.Button(btn_frame, text=self._t("ui.buttons.add", "Add"), command=add_column).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text=self._t("ui.buttons.cancel", "Cancel"), command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _edit_column_dialog(self):
        """Show dialog to edit selected column."""
        selection = self.columns_listbox.curselection()
        if not selection:
            self._notify(self._t("ui.messages.select_column_edit", "Please select a column to edit"), level="warning")
            return
        
        idx = selection[0]
        col_config = self.columns_configs[idx]
        
        dialog = tk.Toplevel(self.dialog)
        _set_window_icon(dialog, "Icon")
        dialog.title(self._t("ui.dialogs.edit_column", "Edit Column"))
        dialog.geometry("400x300")
        self._center_window(dialog, 400, 300)
        dialog.transient(self.dialog)
        dialog.grab_set()
        
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Column name
        ttk.Label(frame, text=self._t("ui.labels.column_name", "Column Name:")).pack(anchor=tk.W, pady=(0, 5))
        name_var = tk.StringVar(value=col_config.get('name', ''))
        ttk.Entry(frame, textvariable=name_var, width=40).pack(fill=tk.X, pady=(0, 10))
        
        # Data source
        ttk.Label(frame, text=self._t("ui.labels.input_output_source_colon", "Input/Output Source:")).pack(anchor=tk.W, pady=(0, 5))
        source_var = tk.StringVar(value=col_config.get('data_source', ''))
        source_combo = ttk.Combobox(frame, textvariable=source_var,
                                   values=self._get_available_data_sources(),
                                   state='readonly', width=40)
        source_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Nested key
        ttk.Label(frame, text=self._t("ui.labels.nested_key_optional", "Nested Key (optional):")).pack(anchor=tk.W, pady=(0, 5))
        nested_var = tk.StringVar(value=col_config.get('nested_key', ''))
        nested_combo = ttk.Combobox(frame, textvariable=nested_var, state='readonly', width=40)
        nested_combo.pack(fill=tk.X, pady=(0, 10))
        
        def update_nested_keys(event=None):
            source = source_var.get()
            if source:
                keys = self._get_nested_keys(source)
                nested_combo.config(values=keys)
        
        source_combo.bind('<<ComboboxSelected>>', update_nested_keys)
        update_nested_keys()  # Initial population
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        def save_column():
            name = name_var.get().strip()
            source = source_var.get()
            nested = nested_var.get()
            
            if not name or not source:
                self._notify(self._t("ui.messages.provide_column_name_source", "Please provide column name and input/output source"), level="warning")
                return
            
            col_config = {
                'name': name,
                'data_source': source
            }
            if nested:
                col_config['nested_key'] = nested
            
            self.columns_configs[idx] = col_config
            self._update_columns_list()
            dialog.destroy()
        
        ttk.Button(btn_frame, text=self._t("ui.buttons.save", "Save"), command=save_column).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text=self._t("ui.buttons.cancel", "Cancel"), command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _remove_column(self):
        """Remove selected column."""
        selection = self.columns_listbox.curselection()
        if not selection:
            self._notify(self._t("ui.messages.select_column_remove", "Please select a column to remove"), level="warning")
            return
        
        idx = selection[0]
        del self.columns_configs[idx]
        self._update_columns_list()
    
    def _clear_columns(self):
        """Clear all columns."""
        if self.columns_configs:
            if messagebox.askyesno(self._t("ui.dialogs.confirm", "Confirm"), self._t("ui.messages.clear_all_columns_confirm", "Clear all columns?")):
                self.columns_configs.clear()
                self._update_columns_list()
    
    def _update_columns_list(self):
        """Update the columns listbox."""
        self.columns_listbox.delete(0, tk.END)
        for col_config in self.columns_configs:
            name = col_config.get('name', 'Unnamed')
            source = col_config.get('data_source', '')
            nested = col_config.get('nested_key', '')
            display = f"{name} ← {source}"
            if nested:
                display += f" [{nested}]"
            self.columns_listbox.insert(tk.END, display)
    
    def _build_advanced_tab(self, parent):
        """Build the advanced configuration tab."""
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
        self._bind_canvas_mousewheel(canvas, scrollable_frame)
        
        # Matrix slicing configuration for 2D data
        matrix_slice_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.matrix_row_column_selection_2d", "Matrix Row/Column Selection (2D data)"), padding=10)
        matrix_slice_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(matrix_slice_frame, text=self._t("ui.messages.select_single_row_column", "Select a single row or column from matrix data:"),
                 wraplength=450).pack(anchor=tk.W, pady=5)

        self.enable_matrix_slice_var = tk.BooleanVar(value=False)
        matrix_slice_cb = ttk.Checkbutton(
            matrix_slice_frame,
            text=self._t("ui.labels.enable_matrix_row_column_selection", "Enable Matrix Row/Column Selection"),
            variable=self.enable_matrix_slice_var,
            command=self._toggle_matrix_slicing_config
        )
        matrix_slice_cb.pack(anchor=tk.W, pady=5)

        self.matrix_slicing_config_frame = ttk.Frame(matrix_slice_frame)

        mode_frame = ttk.Frame(self.matrix_slicing_config_frame)
        mode_frame.pack(fill=tk.X, pady=(5, 2))
        ttk.Label(mode_frame, text=self._t("ui.labels.select", "Select:"), width=15).pack(side=tk.LEFT)
        self.matrix_slice_mode_var = tk.StringVar(value="column")
        ttk.Radiobutton(mode_frame, text=self._t("ui.labels.column", "Column"), variable=self.matrix_slice_mode_var,
                        value="column").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(mode_frame, text=self._t("ui.labels.row", "Row"), variable=self.matrix_slice_mode_var,
                        value="row").pack(side=tk.LEFT)

        index_frame = ttk.Frame(self.matrix_slicing_config_frame)
        index_frame.pack(fill=tk.X, pady=(5, 2))
        ttk.Label(index_frame, text=self._t("ui.labels.index", "Index:"), width=15).pack(side=tk.LEFT)
        self.matrix_slice_index_var = tk.StringVar(value="0")
        ttk.Entry(index_frame, textvariable=self.matrix_slice_index_var, width=10).pack(side=tk.LEFT)

        self.matrix_slice_show_nav_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.matrix_slicing_config_frame,
            text=self._t("ui.labels.show_navigation_prev_next", "Show navigation controls (prev/next)"),
            variable=self.matrix_slice_show_nav_var
        ).pack(anchor=tk.W, pady=(5, 0))

        # Data slicing configuration for 3D+ data
        slice_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.data_slicing_for_3d", "Data Slicing (for 3D+ data)"), padding=10)
        slice_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(slice_frame, text=self._t("ui.messages.enable_navigation_3d", "Enable navigation for 3D+ data:"), 
                 wraplength=450).pack(anchor=tk.W, pady=5)
        
        self.enable_slicing_var = tk.BooleanVar(value=False)
        slice_cb = ttk.Checkbutton(slice_frame, text=self._t("ui.labels.enable_data_slicing", "Enable Data Slicing"), 
                                   variable=self.enable_slicing_var, command=self._toggle_slicing_config)
        slice_cb.pack(anchor=tk.W, pady=5)
        
        # Container for slicing configuration (hidden by default)
        self.slicing_config_frame = ttk.Frame(slice_frame)
        
        ttk.Label(self.slicing_config_frame, text=self._t("ui.labels.dimension", "Dimension:")).pack(anchor=tk.W, pady=(5, 2))
        self.slice_dim_var = tk.StringVar(value="0")
        ttk.Entry(self.slicing_config_frame, textvariable=self.slice_dim_var, width=10).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(self.slicing_config_frame, text=self._t("ui.labels.default_index", "Default Index:")).pack(anchor=tk.W, pady=(5, 2))
        self.slice_default_var = tk.StringVar(value="0")
        ttk.Entry(self.slicing_config_frame, textvariable=self.slice_default_var, width=10).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(self.slicing_config_frame, text=self._t("ui.labels.navigation_label", "Navigation Label:")).pack(anchor=tk.W, pady=(5, 2))
        self.slice_name_var = tk.StringVar(value="Slice")
        ttk.Entry(self.slicing_config_frame, textvariable=self.slice_name_var, width=30).pack(anchor=tk.W, pady=(0, 5))
        
        # Headers configuration
        headers_frame = ttk.LabelFrame(scrollable_frame, text=self._t("ui.labels.headers_optional", "Headers (Optional)"), padding=10)
        headers_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(headers_frame, text=self._t("ui.labels.column_headers_comma", "Column Headers (comma-separated):")).pack(anchor=tk.W, pady=(5, 2))
        self.col_headers_var = tk.StringVar()
        ttk.Entry(headers_frame, textvariable=self.col_headers_var, width=50).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(headers_frame, text=self._t("ui.labels.row_headers_source", "Row Headers Source:")).pack(anchor=tk.W, pady=(5, 2))
        self.row_headers_var = tk.StringVar()
        ttk.Combobox(headers_frame, textvariable=self.row_headers_var,
                    values=[''] + self._get_available_data_sources(),
                    state='readonly', width=50).pack(fill=tk.X, pady=(0, 5))
    
    def _toggle_slicing_config(self):
        """Toggle visibility of slicing configuration."""
        if self.enable_slicing_var.get():
            self.slicing_config_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            self.slicing_config_frame.pack_forget()

    def _toggle_matrix_slicing_config(self):
        """Toggle visibility of matrix slicing configuration."""
        if self.enable_matrix_slice_var.get():
            self.matrix_slicing_config_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            self.matrix_slicing_config_frame.pack_forget()
    
    def _on_data_source_change(self, event=None):
        """Handle data source selection change."""
        data_source = self.data_source_var.get()
        nested_keys = self._get_nested_keys(data_source)
        
        if nested_keys:
            self.nested_key_combo.config(values=nested_keys)
            self.nested_key_combo.set('')
        else:
            self.nested_key_combo.config(values=[])
            self.nested_key_combo.set('')
        
        # Update info panel
        self._update_data_info()
    
    def _build_info_panel(self, parent):
        """Build the data info panel."""
        # Title
        title = ttk.Label(parent, text=self._t("ui.labels.data_information", "Data Information"), font=("Arial", 12, "bold"))
        title.pack(pady=10)
        
        # Info text
        self.info_text = tk.Text(parent, wrap=tk.WORD, width=30, height=20, state='disabled')
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial message
        self.info_text.config(state='normal')
        self.info_text.insert('1.0', self._t("ui.messages.select_input_output_view_info", "Select an input/output source to view information..."))
        self.info_text.config(state='disabled')
    
    def _update_data_info(self):
        """Update the data info panel."""
        data_source = self.data_source_var.get()
        nested_key = self.nested_key_var.get()
        
        if not data_source:
            self.info_text.config(state='normal')
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', self._t("ui.messages.select_input_output_view_info", "Select an input/output source to view information..."))
            self.info_text.config(state='disabled')
            return
        
        shape_info = self._get_data_shape_info(data_source, nested_key if nested_key else None)
        
        # Get actual data for more detailed info
        data = self.outputs.get(data_source)
        data = self._resolve_nested_data(data, nested_key if nested_key else None)
        
        info_lines = [
            f"Input/Output Source: {data_source}",
            f"Nested Key: {nested_key if nested_key else 'None'}",
            "",
            shape_info,
        ]
        
        if isinstance(data, np.ndarray):
            info_lines.extend([
                "",
                f"Min: {np.min(data):.6f}",
                f"Max: {np.max(data):.6f}",
                f"Mean: {np.mean(data):.6f}",
                f"Std: {np.std(data):.6f}",
            ])
            
            if data.ndim > 2:
                info_lines.extend([
                    "",
                    self._t("ui.messages.data_3d_detected", "⚠ 3D+ Data Detected"),
                    self._t("ui.messages.enable_data_slicing_in", "Enable data slicing in the"),
                    self._t("ui.messages.advanced_tab_display_data", "Advanced tab to display this data."),
                ])
        
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', '\n'.join(info_lines))
        self.info_text.config(state='disabled')
    
    def _build_button_bar(self):
        """Build bottom button bar."""
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        # Add Table button
        add_btn = ttk.Button(button_frame, text="✓ " + self._t("ui.buttons.add_table", "Add Table"), command=self._add_table)
        add_btn.pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        cancel_btn = ttk.Button(button_frame, text="✗ " + self._t("ui.buttons.cancel", "Cancel"), command=self.dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def _build_table_config(self) -> Dict:
        """Build the table configuration dictionary."""
        config = {}
        
        # Check if using multi-column mode
        if self.columns_configs:
            # Multi-column mode
            config['columns'] = self.columns_configs
        else:
            # Single column mode
            data_source = self.data_source_var.get()
            if not data_source:
                return None
            
            config['data_source'] = data_source
            
            nested_key = self.nested_key_var.get()
            if nested_key:
                config['nested_key'] = nested_key
        
        title = self.title_var.get()
        if title:
            config['title'] = title
            config['table_title'] = title
        
        # Formatting
        try:
            config['decimal_places'] = int(self.decimal_var.get())
        except ValueError:
            config['decimal_places'] = 4
        
        try:
            config['max_rows'] = int(self.max_rows_var.get())
        except ValueError:
            config['max_rows'] = 50
        
        try:
            config['max_cols'] = int(self.max_cols_var.get())
        except ValueError:
            config['max_cols'] = 15
        
        # Data slicing
        data_slicing = []

        # Matrix row/column slicing (2D)
        if self.enable_matrix_slice_var.get():
            try:
                matrix_index = int(self.matrix_slice_index_var.get())
                mode = self.matrix_slice_mode_var.get()
                is_row = mode == 'row'

                data_slicing.append({
                    'name': 'Row' if is_row else 'Column',
                    'dimension': 0 if is_row else 1,
                    'default': matrix_index,
                    'show_navigation_menu': self.matrix_slice_show_nav_var.get()
                })
            except ValueError:
                pass

        # Generic 3D+ slicing
        if self.enable_slicing_var.get():
            try:
                dimension = int(self.slice_dim_var.get())
                default = int(self.slice_default_var.get())
                name = self.slice_name_var.get() or "Slice"

                data_slicing.append({
                    'name': name,
                    'dimension': dimension,
                    'default': default,
                    'show_navigation_menu': True
                })
            except ValueError:
                pass  # Skip if invalid values

        if data_slicing:
            config['data_slicing'] = data_slicing
        
        # Headers
        col_headers_text = self.col_headers_var.get().strip()
        if col_headers_text:
            headers = [h.strip() for h in col_headers_text.split(',')]
            config['column_headers'] = headers
        
        row_headers_source = self.row_headers_var.get()
        if row_headers_source:
            config['row_headers'] = row_headers_source
        
        return config
    
    def _add_table(self):
        """Add the table to the selected section."""
        try:
            # Get selected section
            selected_section = self.section_var.get().strip()
            if not selected_section:
                self._notify(self._t("ui.messages.select_target_section", "Please select a target section"), level="warning")
                return

            section_idx = next(
                (idx for idx, (_, _, desc) in enumerate(self.empty_sections) if desc == selected_section),
                -1
            )
            if section_idx < 0:
                self._notify(self._t("ui.messages.selected_target_section_invalid", "Selected target section is invalid"), level="error")
                return

            page_idx, sec_idx, _ = self.empty_sections[section_idx]
            
            # Build configuration
            config = self._build_table_config()
            if not config:
                self._notify(self._t("ui.messages.configure_column_or_source", "Please configure at least one column in Multi-Column tab or select a data source in Basic tab"), level="warning")
                return
            
            # Validate data exists
            if 'columns' in config:
                # Multi-column mode - require at least one currently available column.
                # Missing optional columns (e.g., CV outputs) are allowed and skipped at render time.
                available_count = 0
                for col_config in config['columns']:
                    data_source = col_config.get('data_source', '')
                    nested_key = col_config.get('nested_key')
                    if self._get_data_source_value(data_source) is not None:
                        # If nested key is present, ensure it resolves to actual data
                        if nested_key:
                            resolved = self._get_data_source_value(data_source)
                            resolved = self._resolve_nested_data(resolved, nested_key)
                            if resolved is None:
                                continue
                        available_count += 1

                if available_count == 0:
                    self._notify(
                        self._t(
                            "ui.messages.no_configured_columns_available",
                            "None of the configured columns are available in current outputs. Run the corresponding upstream steps first."
                        ),
                        level="error"
                    )
                    return
            else:
                # Single column mode
                data_source = config.get('data_source')
                if not data_source or self._get_data_source_value(data_source) is None:
                    self._notify(self._t("ui.messages.data_source_not_found", "Data source not found:") + f" '{data_source}'", level="error")
                    return
            
            # Update analysis data
            pages = self.main_gui.analysis_data[self.instance_alias]['pages']
            pages[page_idx]['sections'][sec_idx] = {
                'type': 'table',
                'config': config
            }
            
            # Refresh the display
            self.main_gui._show_analysis_tab()
            
            # Close dialog
            self.dialog.destroy()
            
            self._notify(self._t("ui.messages.table_added_to", "Table added to") + f" Page {page_idx + 1}, Section {sec_idx + 1}", level="success")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._notify(self._t("ui.messages.add_table_failed", "Failed to add table:") + f" {str(e)}", level="error")


def show_add_table_dialog(parent, main_gui, instance_alias: str):
    """Show the Add Table dialog.
    
    Args:
        parent: Parent tkinter window
        main_gui: Reference to main ChemometricsGUI instance
        instance_alias: Alias of the current function instance
    """
    AddTableDialog(parent, main_gui, instance_alias)
