"""
Main GUI application for CM Studio using tkinter + Sun-Valley theme.
Provides Setup, Routing, Analysis, and Report tabs for building analysis pipelines.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import subprocess
import sys
from io import StringIO
import shlex
import zipfile
import shutil
from datetime import datetime
import tempfile
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# Import language manager
from language_manager import get_language_manager, _

# Import settings manager
from settings import get_settings_manager

# Import graph renderer module
import graph_renderer

# Import routing map window
from routing_map_window import RoutingMapWindow

# Load function specs
SPECS_PATH = Path(__file__).parent / "function_specs.json"
with open(SPECS_PATH, encoding='utf-8') as f:
    FUNCTION_SPECS = json.load(f)


class Tooltip:
    """Create a tooltip for a given widget."""
    
    def __init__(self, widget, text="", wraplength=250):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        
        # Bind hover events
        self.widget.bind("<Enter>", self.showtip, add=True)
        self.widget.bind("<Leave>", self.hidetip, add=True)
    
    def showtip(self, event=None):
        """Show the tooltip."""
        if self.tipwindow:
            return  # Already showing
        if self.text:
            x = self.widget.winfo_rootx() + 50
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
            
            # Create tooltip window
            self.tipwindow = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            
            label = tk.Label(tw, text=self.text, background="#ffffcc", relief=tk.SOLID, 
                           borderwidth=1, font=("Arial", 9), wraplength=self.wraplength, justify=tk.LEFT)
            label.pack(ipadx=1)
    
    def hidetip(self, event=None):
        """Hide the tooltip."""
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class ChemometricsGUI:
    """Main GUI class for building and executing chemometrics pipelines."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        
        # Initialize settings manager and load saved language
        self.settings_manager = get_settings_manager()
        saved_language = self.settings_manager.get("language", "en")
        
        # Initialize language manager with saved language
        self.language_manager = get_language_manager()
        self.language_manager.set_language(saved_language)
        
        self.root.title(self.language_manager.translate("ui.main_title", "CM Studio"))
        self.root.geometry("1280x720")
        
        # Set up tempfiles folder for loaded models
        self.tempfiles_dir = Path(__file__).parent / "tempfiles"
        self._clean_tempfiles()
        
        # Set up cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Data structures
        self.methodology_list: List[str] = []  # [instance_alias, instance_alias, ...] where instance_alias handles duplicates
        self.function_base_aliases: List[str] = []  # [base_func_alias, base_func_alias, ...] stores the original function alias for each instance
        self.function_configs: Dict[str, Dict[str, Any]] = {}  # {instance_alias: {param: value}}
        self.routing_lines: Dict[Tuple, str] = {}  # {(src_idx, src_output, dst_idx, dst_input): routing_info}
        self.selected_function_idx: Optional[int] = None  # Index in methodology_list
        self.gui_configs: Dict[str, Dict] = {}  # {func_alias: config_data}
        
        # Configure dark theme styles
        style = ttk.Style()
        style.configure("Hidden.TFrame", background="#212121")
        
        # Load GUI configs for all functions
        self._load_gui_configs()
        
        # Build menu bar
        self._build_menu_bar()
        
        # Build UI
        self._build_ui()
        self._load_theme()
    
    def _load_gui_configs(self):
        """Load function-specific GUI configuration files with language support."""
        gui_listing = FUNCTION_SPECS.get("gui_listing", {})
        current_language = get_language_manager().get_language()
        
        for func_alias, func_info in gui_listing.items():
            config_file = func_info.get("config_path")
            if config_file:
                # Parse the config file path
                config_path = Path(config_file)
                config_name = config_path.name
                
                # Try language-specific folder first (gui_configs/[language]/[config_name])
                lang_folder = Path(__file__).parent / "gui_configs" / current_language
                lang_config_path = lang_folder / config_name
                
                if lang_config_path.exists():
                    try:
                        with open(lang_config_path, encoding='utf-8') as f:
                            self.gui_configs[func_alias] = json.load(f)
                        continue
                    except json.JSONDecodeError as e:
                        print(f"ERROR: Invalid JSON in {lang_config_path}: {e}")
                        raise
                
                # Fall back to English folder
                en_folder = Path(__file__).parent / "gui_configs" / "en"
                en_config_path = en_folder / config_name
                
                if en_config_path.exists():
                    try:
                        with open(en_config_path, encoding='utf-8') as f:
                            self.gui_configs[func_alias] = json.load(f)
                        continue
                    except json.JSONDecodeError as e:
                        print(f"ERROR: Invalid JSON in {en_config_path}: {e}")
                        raise
                
                # Final fallback to original path (backward compatibility)
                full_path = Path(__file__).parent / config_file
                if full_path.exists():
                    try:
                        with open(full_path, encoding='utf-8') as f:
                            self.gui_configs[func_alias] = json.load(f)
                    except FileNotFoundError:
                        print(f"Warning: Config file not found: {config_file}")
                    except json.JSONDecodeError as e:
                        print(f"ERROR: Invalid JSON in {config_file}: {e}")
                        raise
                else:
                    print(f"Warning: Config file not found: {config_file}")
    
    def _load_theme(self):
        """Attempt to load Sun-Valley theme if available."""
        theme_path = Path(__file__).parent / "Sun-Valley-ttk-theme" / "sun-valley.tcl"
        if theme_path.exists():
            try:
                self.root.tk.call("source", str(theme_path))
                self.root.tk.call("set_theme", "dark")
            except tk.TclError as e:
                print(f"Could not load theme: {e}")
        
        # Configure custom button styles for routing tab
        style = ttk.Style()
        
        # Output button style (blue)
        style.configure("Output.TButton", 
                       font=("Arial", 9),
                       padding=8)
        
        # Input button style (red)
        style.configure("Input.TButton",
                       font=("Arial", 9),
                       padding=8)
    
    def _build_menu_bar(self):
        """Build the menu bar with File, Settings, and Help menus."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.language_manager.translate("menu.file", "File"), menu=file_menu)
        file_menu.add_command(label=self.language_manager.translate("menu.load_model", "Load Model"), command=self._show_load_model_dialog)
        file_menu.add_command(label=self.language_manager.translate("menu.save_model", "Save Model"), command=self._show_save_model_dialog)
        file_menu.add_separator()
        file_menu.add_command(label=self.language_manager.translate("menu.exit", "Exit"), command=self._on_close)
        
        # Settings Menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.language_manager.translate("menu.settings", "Settings"), menu=settings_menu)
        
        # Language submenu
        lang_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label=self.language_manager.translate("menu.language", "Language"), menu=lang_menu)
        
        for lang_code, lang_name in self.language_manager.SUPPORTED_LANGUAGES.items():
            lang_menu.add_command(label=lang_name, command=lambda code=lang_code: self._change_language(code))
        
        # Colormap submenu
        colormap_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label=self.language_manager.translate("menu.colormap", "Colormap"), menu=colormap_menu)
        
        # Load available colormaps
        try:
            colormaps_path = Path(__file__).parent / "Settings" / "colormaps.json"
            with open(colormaps_path, encoding='utf-8') as f:
                colormaps_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            colormaps_data = {"continuous": {"Perceptually Uniform": ["viridis", "plasma", "inferno"]}, "qualitative": []}
        
        # Continuous colormaps submenu
        continuous_menu = tk.Menu(colormap_menu, tearoff=0)
        colormap_menu.add_cascade(label=self.language_manager.translate("menu.colormap_continuous", "Continuous"), menu=continuous_menu)
        
        # Handle both old flat structure and new nested structure for continuous colormaps
        continuous_data = colormaps_data.get("continuous", {})
        if isinstance(continuous_data, list):
            # Old flat structure - treat as a single category
            for cmap in continuous_data:
                continuous_menu.add_command(label=cmap, command=lambda cm=cmap: self._change_colormap(cm))
        else:
            # New nested structure with subcategories
            for category, cmaps in continuous_data.items():
                category_menu = tk.Menu(continuous_menu, tearoff=0)
                continuous_menu.add_cascade(label=category, menu=category_menu)
                for cmap in cmaps:
                    category_menu.add_command(label=cmap, command=lambda cm=cmap: self._change_colormap(cm))
        
        # Qualitative colormaps submenu
        qualitative_menu = tk.Menu(colormap_menu, tearoff=0)
        colormap_menu.add_cascade(label=self.language_manager.translate("menu.colormap_qualitative", "Qualitative"), menu=qualitative_menu)
        for cmap in colormaps_data.get("qualitative", []):
            qualitative_menu.add_command(label=cmap, command=lambda cm=cmap: self._change_colormap(cm))
        
        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.language_manager.translate("menu.help", "Help"), menu=help_menu)
        help_menu.add_command(label=self.language_manager.translate("menu.about", "About"), command=self._show_about_dialog)
    
    def _change_language(self, language_code: str):
        """Change the application language and save setting."""
        self.language_manager.set_language(language_code)
        self.settings_manager.set("language", language_code)
        self._refresh_ui_text()
    
    def _change_colormap(self, colormap_name: str):
        """Change the default colormap and save setting."""
        self.settings_manager.set("colormap", colormap_name)
        messagebox.showinfo(self.language_manager.translate("ui.dialogs.info", "Information"),
                          f"Colormap changed to '{colormap_name}'.\nThis will be used for new plots.")
    
    def _show_about_dialog(self):
        """Show the About dialog with program information."""
        # Load about info from JSON
        about_file = Path(__file__).parent / "about_us.json"
        try:
            with open(about_file, encoding='utf-8') as f:
                about_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            about_data = {}
        
        current_lang = self.language_manager.get_language()
        lang_info = about_data.get(current_lang, about_data.get("en", {}))
        
        # Create the about window
        about_win = tk.Toplevel(self.root)
        about_win.title(lang_info.get("title", "About"))
        about_win.geometry("550x500")
        about_win.resizable(False, False)
        
        # Center the window on the parent window
        about_win.transient(self.root)
        about_win.grab_set()
        
        # Calculate position to center on parent window
        about_win.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (about_win.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (about_win.winfo_height() // 2)
        about_win.geometry(f"+{x}+{y}")
        
        # Program name and version frame with icon
        info_frame = ttk.Frame(about_win)
        info_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Left side with icon
        try:
            icon_path = Path(__file__).parent / "Icon.ico"
            if icon_path.exists():
                # Load and resize the icon
                icon_image = Image.open(icon_path)
                icon_image = icon_image.resize((80, 80), Image.Resampling.LANCZOS)
                icon_photo = ImageTk.PhotoImage(icon_image)
                
                icon_label = ttk.Label(info_frame, image=icon_photo)
                icon_label.image = icon_photo  # Keep a reference
                icon_label.pack(side=tk.LEFT, padx=(0, 20))
        except Exception as e:
            print(f"Warning: Could not load icon: {e}")
        
        # Right side with text
        text_frame = ttk.Frame(info_frame)
        text_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        program_name = about_data.get("program_name", "CM Studio")
        version = about_data.get("version", "1.0.0")
        
        name_label = ttk.Label(text_frame, text=program_name, font=("Arial", 16, "bold"))
        name_label.pack(anchor=tk.W)
        
        version_label = ttk.Label(text_frame, text=f"Version {version}", font=("Arial", 10))
        version_label.pack(anchor=tk.W)
        
        version_text = lang_info.get("version_text", "")
        if version_text:
            version_text_label = ttk.Label(text_frame, text=version_text, font=("Arial", 9))
            version_text_label.pack(anchor=tk.W)
        
        # Description frame
        desc_frame = ttk.LabelFrame(about_win, text="About", padding=15)
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        # Text widget for description
        text_widget = tk.Text(desc_frame, wrap=tk.WORD, height=12, width=50, bg="#f0f0f0", fg="#000000")
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        description = lang_info.get("description", "")
        text_widget.insert(1.0, description)
        text_widget.config(state=tk.DISABLED)  # Make it read-only
        
        # Hyperlink frame at the bottom
        link_frame = ttk.Frame(about_win)
        link_frame.pack(fill=tk.X, padx=20, pady=10)
        
        website_url = about_data.get("website", "https://github.com")
        website_label = lang_info.get("website_label", "Visit our website")
        
        link_label = tk.Label(link_frame, text=website_label, fg="blue", cursor="hand2", font=("Arial", 9, "underline"))
        link_label.pack()
        
        def open_link(event=None):
            import webbrowser
            webbrowser.open(website_url)
        
        link_label.bind("<Button-1>", open_link)
        
        # Close button
        close_btn = ttk.Button(about_win, text="Close", command=about_win.destroy)
        close_btn.pack(pady=(0, 10))
    

    def _build_ui(self):
        """Build main UI layout with panels."""
        main_frame = ttk.Frame(self.root, width=220)
        main_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=10, pady=10)
        main_frame.pack_propagate(False)
        
        functions_label = ttk.Label(main_frame, text=self.language_manager.translate("ui.panels.available_functions", "Available Functions"), font=("Arial", 10, "bold"))
        functions_label.pack(pady=(0, 5))
        
        self._build_functions_panel(main_frame)
        
        separator1 = ttk.Separator(main_frame, orient=tk.HORIZONTAL)
        separator1.pack(fill=tk.X, pady=10)
        
        methodology_label = ttk.Label(main_frame, text=self.language_manager.translate("ui.panels.methodology", "Methodology"), font=("Arial", 10, "bold"))
        methodology_label.pack(pady=(0, 5))
        
        self._build_methodology_panel(main_frame)
        
        # Workspace area (tabs + content)
        workspace_frame = ttk.Frame(self.root)
        workspace_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self._build_control_bar(workspace_frame)
        
        # Tab content frame
        self.tab_content_frame = ttk.Frame(workspace_frame)
        self.tab_content_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=10)
        
        # Initialize tabs as empty; will be created on demand
        self.current_tab = None
        self._show_setup_tab()
    
    def _build_functions_panel(self, parent: ttk.Frame):
        """Build collapsible list of available functions grouped by category."""
        functions_frame = ttk.LabelFrame(parent, text=self.language_manager.translate("ui.panels.functions_by_category", "Functions by Category"), height=200, width=200)
        functions_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        functions_frame.pack_propagate(False)
        
        # Pack scrollbar first, then canvas so scrollbar is visible
        scrollbar = ttk.Scrollbar(functions_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas = tk.Canvas(functions_frame, highlightthickness=0, bg="#f0f0f0", yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)
        
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mousewheel to canvas for scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
        
        gui_listing = FUNCTION_SPECS.get("gui_listing", {})
        categories = {}
        
        # Group by category
        for func_alias in gui_listing.keys():
            # Get config from loaded gui_configs
            config = self.gui_configs.get(func_alias, {})
            category = config.get("category", "Uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append((func_alias, config))
        
        # Create collapsible categories
        for category in sorted(categories.keys()):
            self._add_collapsible_category(scrollable_frame, category, categories[category], canvas)
    
    def _add_collapsible_category(self, parent: ttk.Frame, category: str, functions: List[Tuple], canvas: tk.Canvas):
        """Create collapsible category with function buttons."""
        category_frame = ttk.Frame(parent)
        category_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Mousewheel binding helper
        def bind_mousewheel(widget):
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            widget.bind("<MouseWheel>", _on_mousewheel)
        
        # Bind to category frame so scrolling works on empty space
        bind_mousewheel(category_frame)
        
        button_frame = ttk.Frame(category_frame)
        button_frame.pack(fill=tk.X)
        bind_mousewheel(button_frame)
        
        collapsed = tk.BooleanVar(value=False)
        
        def toggle():
            collapsed.set(not collapsed.get())
            functions_container.pack_forget() if collapsed.get() else functions_container.pack(fill=tk.X, padx=10, pady=5)
        
        toggle_btn = ttk.Button(button_frame, text=f"▶ {category}", command=toggle, width=30)
        toggle_btn.pack(fill=tk.X)
        bind_mousewheel(toggle_btn)
        
        functions_container = ttk.Frame(category_frame)
        functions_container.pack(fill=tk.X, padx=10, pady=5)
        bind_mousewheel(functions_container)
        
        for func_alias, func_info in functions:
            display_name = func_info.get("display_name", func_alias)
            func_btn = ttk.Button(
                functions_container,
                text=display_name,
                command=lambda alias=func_alias: self._add_to_methodology(alias),
                width=25
            )
            func_btn.pack(fill=tk.X, pady=2)
            bind_mousewheel(func_btn)
    
    def _add_to_methodology(self, func_alias: str):
        """Add function to methodology list (with duplicate handling using function aliasing)."""
        config = self.gui_configs.get(func_alias, {})
        display_name = config.get("display_name", func_alias)
        
        # Count existing instances of this function
        existing_count = self.function_base_aliases.count(func_alias)
        
        # Create unique instance alias for this function
        if existing_count > 0:
            instance_alias = f"{func_alias}#{existing_count + 1}"
            item_name = f"{display_name} #{existing_count + 1}"
        else:
            instance_alias = func_alias
            item_name = display_name
        
        self.methodology_list.append(instance_alias)
        self.function_base_aliases.append(func_alias)
        self.function_configs[instance_alias] = {}  # Initialize config for this instance
        
        new_func_idx = len(self.methodology_list) - 1
        self.methodology_listbox.insert(tk.END, item_name)
        
        # Auto-create routing for inputs that match previous outputs
        self._auto_create_routing(new_func_idx, func_alias)
    
    def _auto_create_routing(self, new_func_idx: int, new_func_alias: str):
        """Automatically create routing connections for parameters with matching names.
        
        Only connects to the immediately previous function with matching output.
        This avoids connecting to distant functions when intermediate functions exist.
        """
        if new_func_idx == 0:
            return  # First function, no previous outputs to route from
        
        return_specs = FUNCTION_SPECS.get("return_specs", {})
        input_specs = FUNCTION_SPECS.get("input_specs", {})
        
        new_func_inputs = input_specs.get(new_func_alias, [])
        
        # Check each input parameter of the new function
        for input_param in new_func_inputs:
            # Find the immediately previous function that outputs this parameter
            for src_idx in range(new_func_idx - 1, -1, -1):  # Check backwards from newest to oldest
                src_base_alias = self.function_base_aliases[src_idx]  # Get the base alias
                src_outputs = return_specs.get(src_base_alias, [])
                
                if input_param in src_outputs:
                    # Found the most recent function with this output
                    # Create automatic routing connection
                    key = (src_idx, input_param, new_func_idx, input_param)
                    
                    # Only add if not already exists
                    if key not in self.routing_lines:
                        self.routing_lines[key] = {
                            "src_idx": src_idx,
                            "src_param_key": input_param,
                            "dst_idx": new_func_idx,
                            "dst_param_key": input_param,
                            "auto_created": True
                        }
                    break  # Stop searching - we found the most recent source
    
    def _build_methodology_panel(self, parent: ttk.Frame):
        """Build methodology list with add/remove buttons."""
        list_frame = ttk.LabelFrame(parent, text=self.language_manager.translate("ui.panels.selected_functions", "Selected Functions"), height=200)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        list_frame.pack_propagate(False)
        
        self.methodology_listbox = tk.Listbox(list_frame, height=10, selectmode=tk.SINGLE)
        self.methodology_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.methodology_listbox.bind("<<ListboxSelect>>", self._on_methodology_select)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.methodology_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.methodology_listbox.config(yscrollcommand=scrollbar.set)
        
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=0, pady=10)
        
        remove_btn = ttk.Button(button_frame, text=self.language_manager.translate("ui.buttons.remove_from_methodology", "Remove Selected"), command=self._remove_from_methodology)
        remove_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(button_frame, text=self.language_manager.translate("ui.buttons.clear_methodology", "Clear All"), command=self._clear_methodology)
        clear_btn.pack(side=tk.LEFT, padx=5)
    
    def _on_methodology_select(self, event=None):
        """Handle methodology list selection."""
        selection = self.methodology_listbox.curselection()
        if selection:
            self.selected_function_idx = selection[0]
            # Only refresh function-specific tabs (Setup and Analysis)
            # Routing and Report are not function-specific, so don't refresh them
            if self.current_tab == "analysis":
                self._show_analysis_tab()
            elif self.current_tab == "setup":
                self._show_setup_tab()
            # Routing and Report tabs are not function-specific, don't refresh them
    
    def _remove_from_methodology(self):
        """Remove selected item from methodology."""
        selection = self.methodology_listbox.curselection()
        if selection:
            idx = selection[0]
            self.methodology_listbox.delete(idx)
            instance_alias = self.methodology_list.pop(idx)
            self.function_base_aliases.pop(idx)
            
            # Remove config for this instance
            if instance_alias in self.function_configs:
                del self.function_configs[instance_alias]
            
            # Remove routing lines involving this index
            keys_to_remove = [key for key in self.routing_lines.keys() 
                            if isinstance(key, tuple) and (key[0] == idx or key[2] == idx)]
            for key in keys_to_remove:
                del self.routing_lines[key]
            
            # Update indices in remaining routing lines
            for key in list(self.routing_lines.keys()):
                if isinstance(key, tuple):
                    src_idx, src_param, dst_idx, dst_param = key
                    # Decrement indices that are > removed idx
                    new_src_idx = src_idx - 1 if src_idx > idx else src_idx
                    new_dst_idx = dst_idx - 1 if dst_idx > idx else dst_idx
                    if new_src_idx != src_idx or new_dst_idx != dst_idx:
                        old_key = key
                        new_key = (new_src_idx, src_param, new_dst_idx, dst_param)
                        self.routing_lines[new_key] = self.routing_lines.pop(old_key)
                        # Update internal indices in the routing info
                        self.routing_lines[new_key]["src_idx"] = new_src_idx
                        self.routing_lines[new_key]["dst_idx"] = new_dst_idx
            
            self.selected_function_idx = None
            self._clear_tab()
    
    def _clear_methodology(self):
        """Clear all methodology items."""
        self.methodology_listbox.delete(0, tk.END)
        self.methodology_list.clear()
        self.function_base_aliases.clear()
        self.function_configs.clear()
        self.routing_lines.clear()
        self.selected_function_idx = None
        self._clear_tab()
    
    def _build_control_bar(self, parent: ttk.Frame):
        """Build tab selection buttons and Run Model button."""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=0, pady=(0, 10))
        
        setup_btn = ttk.Button(control_frame, text=self.language_manager.translate("ui.tabs.setup", "Setup"), command=self._show_setup_tab, width=12)
        setup_btn.pack(side=tk.LEFT, padx=5)
        
        analysis_btn = ttk.Button(control_frame, text=self.language_manager.translate("ui.tabs.analysis", "Analysis"), command=self._show_analysis_tab, width=12)
        analysis_btn.pack(side=tk.LEFT, padx=5)
        
        routing_btn = ttk.Button(control_frame, text=self.language_manager.translate("ui.tabs.routing", "Routing"), command=self._show_routing_tab, width=12)
        routing_btn.pack(side=tk.LEFT, padx=5)
        
        report_btn = ttk.Button(control_frame, text=self.language_manager.translate("ui.tabs.report", "Report"), command=self._show_report_tab, width=12)
        report_btn.pack(side=tk.LEFT, padx=5)
        
        # Add spacer frame to push right-side buttons to the right
        spacer = ttk.Frame(control_frame)
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Save/Load Model buttons
        save_btn = ttk.Button(control_frame, text="💾 " + self.language_manager.translate("ui.buttons.save_model", "Save Model"), command=self._show_save_model_dialog, width=14)
        save_btn.pack(side=tk.RIGHT, padx=5)
        
        load_btn = ttk.Button(control_frame, text="📂 " + self.language_manager.translate("ui.buttons.load_model", "Load Model"), command=self._show_load_model_dialog, width=14)
        load_btn.pack(side=tk.RIGHT, padx=5)
        
        # Run Model button
        run_btn = ttk.Button(control_frame, text="🠊 " + self.language_manager.translate("ui.buttons.run_model", "Run Model"), command=self._run_model, width=12)
        run_btn.pack(side=tk.RIGHT, padx=5)
    
    def _clear_tab(self):
        """Clear current tab content."""
        for widget in self.tab_content_frame.winfo_children():
            widget.destroy()
    
    def _refresh_ui_text(self):
        """Refresh UI text and configs when language changes."""
        # Reload GUI configs with new language
        self._load_gui_configs()
        
        # Update window title
        self.root.title(self.language_manager.translate("ui.main_title", "CM Studio"))
        
        # Save current methodology list display state
        current_methodology_list = self.methodology_list.copy()
        current_base_aliases = self.function_base_aliases.copy()
        current_configs = self.function_configs.copy()
        current_routing = self.routing_lines.copy()
        current_selected_idx = self.selected_function_idx
        
        # Rebuild the entire UI
        self._clear_tab()
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self._build_menu_bar()
        self._build_ui()
        
        # Restore methodology list
        self.methodology_list = current_methodology_list
        self.function_base_aliases = current_base_aliases
        self.function_configs = current_configs
        self.routing_lines = current_routing
        self.selected_function_idx = current_selected_idx
        
        # Rebuild methodology listbox with translated display names
        self.methodology_listbox.delete(0, tk.END)
        for idx, instance_alias in enumerate(self.methodology_list):
            base_alias = self.function_base_aliases[idx]
            config = self.gui_configs.get(base_alias, {})
            display_name = config.get("display_name", base_alias)
            
            # Count previous instances of this base alias for suffix
            existing_count = self.function_base_aliases[:idx].count(base_alias)
            
            if existing_count > 0:
                item_name = f"{display_name} #{existing_count + 1}"
            else:
                item_name = display_name
            
            self.methodology_listbox.insert(tk.END, item_name)
    
    def _show_setup_tab(self):
        """Show Setup tab with function configuration widgets."""
        self._clear_tab()
        self.current_tab = "setup"
        
        if self.selected_function_idx is None:
            label = ttk.Label(self.tab_content_frame, text=self.language_manager.translate("ui.messages.no_methodology", "No functions selected. Please add functions to your methodology."), 
                             font=("Arial", 10, "italic"))
            label.pack(padx=20, pady=20)
            return
        
        instance_alias = self.methodology_list[self.selected_function_idx]
        base_alias = self.function_base_aliases[self.selected_function_idx]
        
        # Get config for this function (use base alias to get UI config)
        config = self.gui_configs.get(base_alias, {})
        display_name = config.get("display_name", base_alias)
        
        # Create title frame with help button
        title_frame = ttk.Frame(self.tab_content_frame)
        title_frame.pack(padx=10, pady=10, fill=tk.X)
        
        title = ttk.Label(
            title_frame,
            text=f"Setup: {display_name}",
            font=("Arial", 11, "bold")
        )
        title.pack(side=tk.LEFT, padx=5)
        
        layout = config.get("setup", {}).get("layout", [])
        
        # Add help button for function description if available
        short_desc = config.get("short_description", "")
        long_desc = config.get("long_description", "")
        if short_desc or long_desc:
            help_btn = ttk.Button(title_frame, text="ℹ", width=2,
                               command=lambda: self._show_help_popup(display_name, short_desc, long_desc))
            help_btn.pack(side=tk.LEFT, padx=5)
            Tooltip(help_btn, short_desc if short_desc else "Click for more information")
        
        if not layout:
            label = ttk.Label(self.tab_content_frame, text=self.language_manager.translate("ui.messages.no_config", "No configuration available for this function"))
            label.pack(padx=20, pady=20)
            return
        
        # Create scrollable form frame
        scroll_container = ttk.Frame(self.tab_content_frame)
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create canvas and scrollbar
        form_canvas = tk.Canvas(scroll_container, bg="#f0f0f0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient=tk.VERTICAL, command=form_canvas.yview)
        scrollable_frame = ttk.Frame(form_canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: form_canvas.configure(scrollregion=form_canvas.bbox("all"))
        )
        form_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        form_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        form_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel to canvas (only when over this specific canvas/frame)
        def _on_mousewheel(event):
            try:
                form_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except tk.TclError:
                # Canvas was destroyed, ignore the scroll event
                pass
        form_canvas.bind("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
        
        form_frame = scrollable_frame
        
        # Initialize function config if needed (use instance_alias as key)
        if instance_alias not in self.function_configs:
            self.function_configs[instance_alias] = {}
        
        func_config = self.function_configs[instance_alias]
        
        # Store widgets for visibility control
        visible_widgets = {}
        
        # Get categories and sort inputs by category
        setup_config = config.get("setup", {})
        categories_list = setup_config.get("categories", [])
        
        # Group inputs by category (keeping original order within each category)
        categorized_inputs = {cat: [] for cat in categories_list}
        uncategorized = []
        
        for widget_spec in layout:
            category = widget_spec.get("category")
            if category and category in categorized_inputs:
                categorized_inputs[category].append(widget_spec)
            else:
                uncategorized.append(widget_spec)
        
        # Create widgets in order: uncategorized first, then by category order
        all_inputs_ordered = uncategorized.copy()
        for cat in categories_list:
            all_inputs_ordered.extend(categorized_inputs[cat])
        
        current_category = None
        category_headers = {}  # Track category header widgets
        row = 0  # Grid row counter
        
        for widget_spec in all_inputs_ordered:
            widget_category = widget_spec.get("category")
            
            # Create category section if category changed
            if widget_category != current_category:
                current_category = widget_category
                if current_category:
                    # Add category header
                    cat_header = ttk.Label(form_frame, text=current_category, font=("Arial", 10, "bold"), 
                                          foreground="#0066cc")
                    cat_header.grid(row=row, column=0, sticky=tk.W, padx=10, pady=(15, 5))
                    category_headers[current_category] = {"widget": cat_header, "grid_params": {"row": row, "column": 0, "sticky": tk.W, "padx": 10, "pady": (15, 5)}}
                    row += 1
            
            name = widget_spec.get("name")
            label_text = widget_spec.get("label", name)
            widget_type = widget_spec.get("widget")
            default = widget_spec.get("default", "")
            required = widget_spec.get("required", False)
            input_tooltip = widget_spec.get("tooltip", "")
            visible_if = widget_spec.get("visible_if", None)
            
            # Create a container frame for each input
            input_container = ttk.Frame(form_frame)
            grid_params = {"row": row, "column": 0, "sticky": tk.W, "padx": 10, "pady": (10, 2)}
            input_container.grid(**grid_params)
            
            # Create label with help button
            label_frame = ttk.Frame(input_container)
            label_frame.pack(anchor=tk.W, fill=tk.X)
            
            label = ttk.Label(label_frame, text=label_text + ("*" if required else ""), font=("Arial", 9))
            label.pack(side=tk.LEFT, padx=0, pady=0)
            
            # Add input-level help with superscript tooltip if available
            if input_tooltip:
                # Create a small superscript-style help indicator
                help_label = tk.Label(label_frame, text="ℹ", font=("Arial", 9), fg="#666666", cursor="question_arrow")
                help_label.pack(side=tk.LEFT, padx=(4, 0), pady=0)
                Tooltip(help_label, input_tooltip)
            
            # Store widget reference with visibility info
            widget_data = {
                "container": input_container,
                "visible_if": visible_if,
                "field_name": name,
                "widget_spec": widget_spec,
                "category": widget_category,
                "grid_params": grid_params
            }
            visible_widgets[name] = widget_data
            row += 1
            
            # Create widget based on type
            if widget_type == "entry":
                entry = ttk.Entry(input_container, width=40)
                value = func_config.get(name, default if default else "")
                if value:
                    entry.insert(0, str(value))
                # Save the value immediately (especially important for defaults)
                if value:
                    self._save_widget_value(instance_alias, name, str(value))
                entry.pack(anchor=tk.W, padx=20, pady=(0, 5))
                
                # Binding for FocusOut: save value and update visibility
                def on_entry_focus_out(event, n=name, e_widget=entry, a=instance_alias, vw=visible_widgets, ch=category_headers):
                    self._save_widget_value(a, n, e_widget.get())
                    self._update_field_visibility(a, vw, ch)
                entry.bind("<FocusOut>", on_entry_focus_out)
                
                # Binding for KeyRelease: update visibility (value will be saved on FocusOut)
                entry.bind("<KeyRelease>", lambda e, a=instance_alias, vw=visible_widgets, ch=category_headers: self._update_field_visibility(a, vw, ch))
                
                widget_data["widget"] = entry
                
            elif widget_type == "combobox":
                values = widget_spec.get("values", [])
                value_aliases = widget_spec.get("value_aliases", values)  # Use values as fallback if no aliases
                
                # Create mapping from alias to actual value
                alias_to_value = dict(zip(value_aliases, values))
                value_to_alias = dict(zip(values, value_aliases))
                
                # Display aliases in the combobox
                combo = ttk.Combobox(input_container, values=value_aliases, width=37, state="readonly")
                
                # Get the current value
                current_value = func_config.get(name, default if default else "")
                if current_value:
                    # Display the alias, but store the actual value
                    display_text = value_to_alias.get(current_value, current_value)
                    combo.set(display_text)
                    self._save_widget_value(instance_alias, name, current_value)
                elif default:
                    # Set default - get the corresponding alias
                    display_text = value_to_alias.get(default, default)
                    combo.set(display_text)
                    self._save_widget_value(instance_alias, name, default)
                
                combo.pack(anchor=tk.W, padx=20, pady=(0, 5))
                
                # Binding for ComboboxSelected: convert alias to actual value, then save and update visibility
                def on_combo_selected(event, n=name, c_widget=combo, a=instance_alias, vw=visible_widgets, a2v=alias_to_value, ch=category_headers):
                    selected_alias = c_widget.get()
                    actual_value = a2v.get(selected_alias, selected_alias)
                    self._save_widget_value(a, n, actual_value)
                    self._update_field_visibility(a, vw, ch)
                combo.bind("<<ComboboxSelected>>", on_combo_selected)
                
                widget_data["widget"] = combo
                widget_data["value_to_alias"] = value_to_alias  # Store for later reference if needed
                
            elif widget_type == "checkbutton":
                default_val = widget_spec.get("default", False)
                var = tk.BooleanVar(value=func_config.get(name, default_val))
                check = ttk.Checkbutton(input_container, text=label_text, variable=var, 
                                       command=lambda n=name, v=var, a=instance_alias, vw=visible_widgets, ch=category_headers: 
                                       (self._save_widget_value(a, n, v.get()), self._update_field_visibility(a, vw, ch)))
                # Save the value immediately
                self._save_widget_value(instance_alias, name, var.get())
                check.pack(anchor=tk.W, padx=20, pady=(0, 5))
                
                widget_data["widget"] = check
                widget_data["variable"] = var
                
            elif widget_type == "file_selector":
                file_frame = ttk.Frame(input_container)
                file_frame.pack(anchor=tk.W, padx=20, pady=(0, 5), fill=tk.X)
                
                # Entry field for file path(s)
                file_entry = ttk.Entry(file_frame, width=40)
                if name in func_config:
                    file_entry.insert(0, str(func_config[name]))
                file_entry.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
                file_entry.bind("<FocusOut>", lambda e, n=name, f_widget=file_entry: self._save_widget_value(instance_alias, n, f_widget.get()))
                
                multiple = widget_spec.get("multiple", False)
                def browse(n, f_widget, is_multiple):
                    if is_multiple:
                        files = filedialog.askopenfilenames(title=f"Select files for {label_text}")
                        if files:
                            f_widget.delete(0, tk.END)
                            # Store as comma-separated or newline-separated list
                            f_widget.insert(0, ";".join(files))
                            self._save_widget_value(instance_alias, n, f_widget.get())
                    else:
                        file = filedialog.askopenfilename(title=f"Select file for {label_text}")
                        if file:
                            f_widget.delete(0, tk.END)
                            f_widget.insert(0, file)
                            self._save_widget_value(instance_alias, n, f_widget.get())
                
                browse_btn = ttk.Button(file_frame, text="Browse", command=lambda n=name, f=file_entry, m=multiple: browse(n, f, m), width=10)
                browse_btn.pack(side=tk.LEFT)
                
                widget_data["widget"] = file_entry
            
            elif widget_type == "combobox_list":
                # Dynamic list of comboboxes based on count_source parameter
                count_source = widget_spec.get("count_source", "nway_flag")
                count = int(func_config.get(count_source, 1))
                values = widget_spec.get("values", [])
                value_aliases = widget_spec.get("value_aliases", values)
                default_value = widget_spec.get("default", values[0] if values else "")
                
                # Create mapping from alias to actual value
                alias_to_value = dict(zip(value_aliases, values))
                value_to_alias = dict(zip(values, value_aliases))
                
                # Container frame for all comboboxes
                list_frame = ttk.Frame(input_container)
                list_frame.pack(anchor=tk.W, padx=20, pady=(0, 5), fill=tk.X)
                
                combo_widgets = []
                current_values = func_config.get(name, [])
                if isinstance(current_values, str):
                    current_values = [v.strip() for v in current_values.split(',') if v.strip()]
                
                for i in range(count):
                    item_frame = ttk.Frame(list_frame)
                    item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
                    
                    item_label = ttk.Label(item_frame, text=f"  [{i+1}]:", width=6)
                    item_label.pack(side=tk.LEFT)
                    
                    combo = ttk.Combobox(item_frame, values=value_aliases, width=30, state="readonly")
                    
                    # Set current or default value
                    if i < len(current_values) and current_values[i]:
                        display_text = value_to_alias.get(current_values[i], current_values[i])
                        combo.set(display_text)
                    else:
                        display_text = value_to_alias.get(default_value, default_value)
                        combo.set(display_text)
                    
                    combo.pack(side=tk.LEFT, padx=(5, 0))
                    combo_widgets.append(combo)
                    
                    # Binding to save all combobox values as a list
                    def on_combo_list_selected(event, widgets=combo_widgets, n=name, a=instance_alias, a2v=alias_to_value):
                        values_list = []
                        for w in widgets:
                            selected_alias = w.get()
                            actual_value = a2v.get(selected_alias, selected_alias)
                            values_list.append(actual_value)
                        self._save_widget_value(a, n, values_list)
                    combo.bind("<<ComboboxSelected>>", on_combo_list_selected)
                
                # Save initial values
                initial_values = []
                for w in combo_widgets:
                    selected_alias = w.get()
                    actual_value = alias_to_value.get(selected_alias, selected_alias)
                    initial_values.append(actual_value)
                self._save_widget_value(instance_alias, name, initial_values)
                
                widget_data["widget"] = combo_widgets
                widget_data["list_frame"] = list_frame
                widget_data["count_source"] = count_source
                widget_data["is_dynamic_list"] = True
            
            elif widget_type == "file_selector_list":
                # Dynamic list of file selectors based on count_source parameter
                count_source = widget_spec.get("count_source", "nway_flag")
                count = int(func_config.get(count_source, 1))
                
                # Container frame for all file selectors
                list_frame = ttk.Frame(input_container)
                list_frame.pack(anchor=tk.W, padx=20, pady=(0, 5), fill=tk.X)
                
                file_widgets = []
                current_values = func_config.get(name, [])
                if isinstance(current_values, str):
                    current_values = [v.strip() for v in current_values.split(';') if v.strip()]
                
                for i in range(count):
                    item_frame = ttk.Frame(list_frame)
                    item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
                    
                    item_label = ttk.Label(item_frame, text=f"  [{i+1}]:", width=6)
                    item_label.pack(side=tk.LEFT)
                    
                    file_entry = ttk.Entry(item_frame, width=34)
                    if i < len(current_values) and current_values[i]:
                        file_entry.insert(0, current_values[i])
                    file_entry.pack(side=tk.LEFT, padx=(5, 5))
                    file_widgets.append(file_entry)
                    
                    # Save on focus out
                    def on_file_focus_out(event, widgets=file_widgets, n=name, a=instance_alias):
                        values_list = [w.get() for w in widgets]
                        self._save_widget_value(a, n, values_list)
                    file_entry.bind("<FocusOut>", on_file_focus_out)
                    
                    # Browse button for this entry
                    def browse_single(idx, f_widget, widgets=file_widgets, n=name, a=instance_alias, lbl=label_text):
                        file = filedialog.askopenfilename(title=f"Select file for {lbl} [{idx+1}]")
                        if file:
                            f_widget.delete(0, tk.END)
                            f_widget.insert(0, file)
                            values_list = [w.get() for w in widgets]
                            self._save_widget_value(a, n, values_list)
                    
                    browse_btn = ttk.Button(item_frame, text="Browse", 
                                           command=lambda idx=i, fw=file_entry: browse_single(idx, fw), width=8)
                    browse_btn.pack(side=tk.LEFT)
                
                # Save initial values
                initial_values = [w.get() for w in file_widgets]
                self._save_widget_value(instance_alias, name, initial_values)
                
                widget_data["widget"] = file_widgets
                widget_data["list_frame"] = list_frame
                widget_data["count_source"] = count_source
                widget_data["is_dynamic_list"] = True
            
            elif widget_type == "entry_list":
                # Dynamic list of entry fields based on count_source parameter
                count_source = widget_spec.get("count_source", "nway_flag")
                count = int(func_config.get(count_source, 1))
                default_value = widget_spec.get("default", "")
                
                # Container frame for all entries
                list_frame = ttk.Frame(input_container)
                list_frame.pack(anchor=tk.W, padx=20, pady=(0, 5), fill=tk.X)
                
                entry_widgets = []
                current_values = func_config.get(name, [])
                if isinstance(current_values, str):
                    current_values = [v.strip() for v in current_values.split(';') if v.strip()]
                
                for i in range(count):
                    item_frame = ttk.Frame(list_frame)
                    item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
                    
                    item_label = ttk.Label(item_frame, text=f"  [{i+1}]:", width=6)
                    item_label.pack(side=tk.LEFT)
                    
                    entry = ttk.Entry(item_frame, width=38)
                    if i < len(current_values) and current_values[i]:
                        entry.insert(0, current_values[i])
                    elif default_value:
                        entry.insert(0, default_value)
                    entry.pack(side=tk.LEFT, padx=(5, 0))
                    entry_widgets.append(entry)
                    
                    # Save on focus out
                    def on_entry_focus_out(event, widgets=entry_widgets, n=name, a=instance_alias):
                        values_list = [w.get() for w in widgets]
                        self._save_widget_value(a, n, values_list)
                    entry.bind("<FocusOut>", on_entry_focus_out)
                
                # Save initial values
                initial_values = [w.get() for w in entry_widgets]
                self._save_widget_value(instance_alias, name, initial_values)
                
                widget_data["widget"] = entry_widgets
                widget_data["list_frame"] = list_frame
                widget_data["count_source"] = count_source
                widget_data["is_dynamic_list"] = True
            
            elif widget_type == "sample_paths_list":
                # Dynamic list of multi-file selectors - each sample can have multiple files
                count_source = widget_spec.get("count_source", "num_samples")
                count = int(func_config.get(count_source, 1))
                
                # Container frame for all sample entries
                list_frame = ttk.Frame(input_container)
                list_frame.pack(anchor=tk.W, padx=20, pady=(0, 5), fill=tk.X)
                
                sample_widgets = []  # List of (entry_widget, files_list) tuples
                current_values = func_config.get(name, [])
                if isinstance(current_values, str):
                    # Parse semicolon-separated samples, where each sample has comma-separated files
                    current_values = []
                
                for i in range(count):
                    item_frame = ttk.Frame(list_frame)
                    item_frame.pack(anchor=tk.W, pady=(0, 4), fill=tk.X)
                    
                    item_label = ttk.Label(item_frame, text=f"  Sample {i+1}:", width=10)
                    item_label.pack(side=tk.LEFT)
                    
                    # Entry to display selected files (read-only display)
                    file_entry = ttk.Entry(item_frame, width=40)
                    if i < len(current_values) and current_values[i]:
                        # current_values[i] should be a list of file paths
                        if isinstance(current_values[i], list):
                            file_entry.insert(0, "; ".join(current_values[i]))
                        else:
                            file_entry.insert(0, str(current_values[i]))
                    file_entry.pack(side=tk.LEFT, padx=(5, 5))
                    
                    # Store the files list separately
                    files_list = current_values[i] if i < len(current_values) and isinstance(current_values[i], list) else []
                    sample_widgets.append({"entry": file_entry, "files": files_list})
                    
                    # Browse button for multiple files
                    def browse_multiple_files(idx, f_widget, widgets=sample_widgets, n=name, a=instance_alias, lbl=label_text):
                        files = filedialog.askopenfilenames(title=f"Select files for Sample {idx+1}")
                        if files:
                            # Update the entry display
                            f_widget.delete(0, tk.END)
                            f_widget.insert(0, "; ".join(files))
                            # Update the stored files list
                            widgets[idx]["files"] = list(files)
                            # Save all sample paths
                            values_list = [w["files"] for w in widgets]
                            self._save_widget_value(a, n, values_list)
                    
                    browse_btn = ttk.Button(item_frame, text="Browse...", 
                                           command=lambda idx=i, fw=file_entry: browse_multiple_files(idx, fw), width=10)
                    browse_btn.pack(side=tk.LEFT)
                    
                    # Focus out handler to parse manually entered paths
                    def on_sample_focus_out(event, idx=i, f_widget=file_entry, widgets=sample_widgets, n=name, a=instance_alias):
                        # Parse the entry text as semicolon-separated paths
                        text = f_widget.get()
                        if text.strip():
                            parsed_files = [f.strip() for f in text.split(';') if f.strip()]
                            widgets[idx]["files"] = parsed_files
                        else:
                            widgets[idx]["files"] = []
                        values_list = [w["files"] for w in widgets]
                        self._save_widget_value(a, n, values_list)
                    file_entry.bind("<FocusOut>", on_sample_focus_out)
                
                # Save initial values
                initial_values = [w["files"] for w in sample_widgets]
                self._save_widget_value(instance_alias, name, initial_values)
                
                widget_data["widget"] = sample_widgets
                widget_data["list_frame"] = list_frame
                widget_data["count_source"] = count_source
                widget_data["is_dynamic_list"] = True
        
        # Initial visibility update
        self._update_field_visibility(instance_alias, visible_widgets, category_headers)
    
    def _save_widget_value(self, func_alias: str, param_name: str, value: Any):
        """Save widget value to function config."""
        if func_alias not in self.function_configs:
            self.function_configs[func_alias] = {}
        self.function_configs[func_alias][param_name] = value
    
    def _show_help_popup(self, title: str, short_desc: str, long_desc: str):
        """Show a popup window with function help information."""
        popup = tk.Toplevel(self.root)
        popup.title(f"Help: {title}")
        popup.geometry("600x400")
        
        # Set the window icon to Info.ico
        try:
            info_icon_path = Path(__file__).parent / "Info.ico"
            if info_icon_path.exists():
                popup.iconbitmap(str(info_icon_path))
        except Exception as e:
            print(f"Warning: Could not set help window icon: {e}")
        
        # Create scrollable text area
        text_frame = ttk.Frame(popup)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, font=("Arial", 10))
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Display content
        content = ""
        if short_desc:
            summary_label = self.language_manager.translate("ui.labels.summary", "Summary:")
            content += f"{summary_label}\n{short_desc}\n\n"
        if long_desc:
            details_label = self.language_manager.translate("ui.labels.details", "Details:")
            content += f"{details_label}\n{long_desc}"
        
        text_widget.insert(1.0, content)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Close button
        close_btn = ttk.Button(popup, text="Close", command=popup.destroy)
        close_btn.pack(pady=10)
    
    def _update_field_visibility(self, func_alias: str, visible_widgets: Dict, category_headers: Dict = None):
        """Update visibility of fields based on visible_if conditions and hide empty categories."""
        func_config = self.function_configs.get(func_alias, {})
        
        # First, check and rebuild any dynamic list widgets whose count has changed
        self._update_dynamic_list_widgets(func_alias, visible_widgets, func_config)
        
        # Track which categories have visible content
        visible_categories = set()
        
        for field_name, widget_data in visible_widgets.items():
            visible_if = widget_data.get("visible_if")
            container = widget_data.get("container")
            widget_spec = widget_data.get("widget_spec", {})
            grid_params = widget_data.get("grid_params", {})
            
            # Default to showing the field
            should_show = True
            
            # Handle explicit false (for always-hidden fields)
            if visible_if is False:
                should_show = False
            elif visible_if:
                # visible_if is a dict like {"method": "moving_average"} or {"input_type": "user"}
                # Also supports operator format: {"nway_flag": {"operator": ">=", "value": 3}}
                # Check if ALL conditions are met
                for condition_field, condition_value in visible_if.items():
                    # Special handling for input_type meta-condition
                    if condition_field == "input_type":
                        actual_input_type = widget_spec.get("input_type", "user")
                        if actual_input_type != condition_value:
                            should_show = False
                            break
                    else:
                        # Normal field condition
                        current_value = func_config.get(condition_field)
                        
                        # Handle boolean checkbuttons
                        widget_data_for_condition = visible_widgets.get(condition_field, {})
                        if "variable" in widget_data_for_condition:
                            # It's a checkbutton
                            current_value = widget_data_for_condition["variable"].get()
                        
                        # Check if condition_value is an operator dict
                        if isinstance(condition_value, dict) and "operator" in condition_value:
                            operator = condition_value.get("operator", "==")
                            expected_value = condition_value.get("value")
                            
                            # Try to convert to numeric for comparison
                            try:
                                current_val = int(current_value) if current_value is not None else 0
                                expected_val = int(expected_value) if expected_value is not None else 0
                            except (ValueError, TypeError):
                                current_val = str(current_value) if current_value is not None else ""
                                expected_val = str(expected_value) if expected_value is not None else ""
                            
                            # Evaluate based on operator
                            if operator == "==":
                                condition_met = current_val == expected_val
                            elif operator == "!=":
                                condition_met = current_val != expected_val
                            elif operator == ">":
                                condition_met = current_val > expected_val
                            elif operator == "<":
                                condition_met = current_val < expected_val
                            elif operator == ">=":
                                condition_met = current_val >= expected_val
                            elif operator == "<=":
                                condition_met = current_val <= expected_val
                            else:
                                condition_met = True
                            
                            if not condition_met:
                                should_show = False
                                break
                        else:
                            # Simple equality comparison
                            if current_value != condition_value:
                                should_show = False
                                break
            
            # Show or hide the container using grid with stored parameters
            if should_show:
                container.grid(**grid_params)  # Re-show with original grid parameters
                # Track that this category has visible content
                category = widget_data.get("category")
                if category:
                    visible_categories.add(category)
            else:
                container.grid_remove()  # Hide but keep grid position
        
        # Update category header visibility
        if category_headers:
            for cat_name, cat_header_info in category_headers.items():
                cat_header = cat_header_info.get("widget")
                cat_grid_params = cat_header_info.get("grid_params", {})
                if cat_name in visible_categories:
                    cat_header.grid(**cat_grid_params)  # Re-show with original grid parameters
                else:
                    cat_header.grid_remove()  # Hide but keep grid position
    
    def _update_dynamic_list_widgets(self, func_alias: str, visible_widgets: Dict, func_config: Dict):
        """Rebuild dynamic list widgets (combobox_list, file_selector_list) when their count source changes."""
        for field_name, widget_data in visible_widgets.items():
            if not widget_data.get("is_dynamic_list"):
                continue
            
            count_source = widget_data.get("count_source")
            if not count_source:
                continue
            
            # Get current count from config
            new_count = int(func_config.get(count_source, 1))
            
            # Get current widget count
            current_widgets = widget_data.get("widget", [])
            current_count = len(current_widgets) if isinstance(current_widgets, list) else 0
            
            # If count hasn't changed, skip rebuild
            if new_count == current_count:
                continue
            
            # Rebuild the widget
            widget_spec = widget_data.get("widget_spec", {})
            widget_type = widget_spec.get("widget")
            list_frame = widget_data.get("list_frame")
            container = widget_data.get("container")
            
            if not list_frame:
                continue
            
            # Preserve current values
            current_values = []
            if isinstance(current_widgets, list):
                for w in current_widgets:
                    try:
                        # For sample_paths_list, widgets are dicts with "files" key
                        if isinstance(w, dict) and "files" in w:
                            current_values.append(w["files"])
                        else:
                            current_values.append(w.get())
                    except:
                        current_values.append("")
            
            # Clear the list frame
            for child in list_frame.winfo_children():
                child.destroy()
            
            # Rebuild based on widget type
            if widget_type == "combobox_list":
                self._rebuild_combobox_list(func_alias, field_name, widget_data, widget_spec, 
                                           list_frame, new_count, current_values, visible_widgets)
            elif widget_type == "file_selector_list":
                self._rebuild_file_selector_list(func_alias, field_name, widget_data, widget_spec,
                                                 list_frame, new_count, current_values, visible_widgets)
            elif widget_type == "entry_list":
                self._rebuild_entry_list(func_alias, field_name, widget_data, widget_spec,
                                        list_frame, new_count, current_values, visible_widgets)
            elif widget_type == "sample_paths_list":
                self._rebuild_sample_paths_list(func_alias, field_name, widget_data, widget_spec,
                                               list_frame, new_count, current_values, visible_widgets)
    
    def _rebuild_combobox_list(self, func_alias: str, field_name: str, widget_data: Dict, 
                                widget_spec: Dict, list_frame, count: int, current_values: list,
                                visible_widgets: Dict):
        """Rebuild a combobox_list widget with new count."""
        values = widget_spec.get("values", [])
        value_aliases = widget_spec.get("value_aliases", values)
        default_value = widget_spec.get("default", values[0] if values else "")
        label_text = widget_spec.get("label", field_name)
        
        alias_to_value = dict(zip(value_aliases, values))
        value_to_alias = dict(zip(values, value_aliases))
        
        combo_widgets = []
        
        for i in range(count):
            item_frame = ttk.Frame(list_frame)
            item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
            
            item_label = ttk.Label(item_frame, text=f"  [{i+1}]:", width=6)
            item_label.pack(side=tk.LEFT)
            
            combo = ttk.Combobox(item_frame, values=value_aliases, width=30, state="readonly")
            
            # Set current or default value
            if i < len(current_values) and current_values[i]:
                display_text = value_to_alias.get(current_values[i], current_values[i])
                combo.set(display_text)
            else:
                display_text = value_to_alias.get(default_value, default_value)
                combo.set(display_text)
            
            combo.pack(side=tk.LEFT, padx=(5, 0))
            combo_widgets.append(combo)
            
            # Binding to save all combobox values as a list
            def on_combo_list_selected(event, widgets=combo_widgets, n=field_name, a=func_alias, a2v=alias_to_value):
                values_list = []
                for w in widgets:
                    selected_alias = w.get()
                    actual_value = a2v.get(selected_alias, selected_alias)
                    values_list.append(actual_value)
                self._save_widget_value(a, n, values_list)
            combo.bind("<<ComboboxSelected>>", on_combo_list_selected)
        
        # Save values
        new_values = []
        for w in combo_widgets:
            selected_alias = w.get()
            actual_value = alias_to_value.get(selected_alias, selected_alias)
            new_values.append(actual_value)
        self._save_widget_value(func_alias, field_name, new_values)
        
        # Update widget_data
        widget_data["widget"] = combo_widgets
    
    def _rebuild_file_selector_list(self, func_alias: str, field_name: str, widget_data: Dict,
                                     widget_spec: Dict, list_frame, count: int, current_values: list,
                                     visible_widgets: Dict):
        """Rebuild a file_selector_list widget with new count."""
        label_text = widget_spec.get("label", field_name)
        
        file_widgets = []
        
        for i in range(count):
            item_frame = ttk.Frame(list_frame)
            item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
            
            item_label = ttk.Label(item_frame, text=f"  [{i+1}]:", width=6)
            item_label.pack(side=tk.LEFT)
            
            file_entry = ttk.Entry(item_frame, width=34)
            if i < len(current_values) and current_values[i]:
                file_entry.insert(0, current_values[i])
            file_entry.pack(side=tk.LEFT, padx=(5, 5))
            file_widgets.append(file_entry)
            
            # Save on focus out
            def on_file_focus_out(event, widgets=file_widgets, n=field_name, a=func_alias):
                values_list = [w.get() for w in widgets]
                self._save_widget_value(a, n, values_list)
            file_entry.bind("<FocusOut>", on_file_focus_out)
            
            # Browse button for this entry
            def browse_single(idx, f_widget, widgets=file_widgets, n=field_name, a=func_alias, lbl=label_text):
                file = filedialog.askopenfilename(title=f"Select file for {lbl} [{idx+1}]")
                if file:
                    f_widget.delete(0, tk.END)
                    f_widget.insert(0, file)
                    values_list = [w.get() for w in widgets]
                    self._save_widget_value(a, n, values_list)
            
            browse_btn = ttk.Button(item_frame, text="Browse", 
                                   command=lambda idx=i, fw=file_entry: browse_single(idx, fw), width=8)
            browse_btn.pack(side=tk.LEFT)
        
        # Save initial values
        initial_values = [w.get() for w in file_widgets]
        self._save_widget_value(func_alias, field_name, initial_values)
        
        # Update widget_data
        widget_data["widget"] = file_widgets

    def _rebuild_entry_list(self, func_alias: str, field_name: str, widget_data: Dict,
                            widget_spec: Dict, list_frame, count: int, current_values: list,
                            visible_widgets: Dict):
        """Rebuild an entry_list widget with new count."""
        default_value = widget_spec.get("default", "")
        
        entry_widgets = []
        
        for i in range(count):
            item_frame = ttk.Frame(list_frame)
            item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
            
            item_label = ttk.Label(item_frame, text=f"  [{i+1}]:", width=6)
            item_label.pack(side=tk.LEFT)
            
            entry = ttk.Entry(item_frame, width=38)
            if i < len(current_values) and current_values[i]:
                entry.insert(0, current_values[i])
            elif default_value:
                entry.insert(0, default_value)
            entry.pack(side=tk.LEFT, padx=(5, 0))
            entry_widgets.append(entry)
            
            # Save on focus out
            def on_entry_focus_out(event, widgets=entry_widgets, n=field_name, a=func_alias):
                values_list = [w.get() for w in widgets]
                self._save_widget_value(a, n, values_list)
            entry.bind("<FocusOut>", on_entry_focus_out)
        
        # Save initial values
        initial_values = [w.get() for w in entry_widgets]
        self._save_widget_value(func_alias, field_name, initial_values)
        
        # Update widget_data
        widget_data["widget"] = entry_widgets

    def _rebuild_sample_paths_list(self, func_alias: str, field_name: str, widget_data: Dict,
                                    widget_spec: Dict, list_frame, count: int, current_values: list,
                                    visible_widgets: Dict):
        """Rebuild a sample_paths_list widget with new count."""
        label_text = widget_spec.get("label", field_name)
        
        sample_widgets = []
        
        for i in range(count):
            item_frame = ttk.Frame(list_frame)
            item_frame.pack(anchor=tk.W, pady=(0, 4), fill=tk.X)
            
            item_label = ttk.Label(item_frame, text=f"  Sample {i+1}:", width=10)
            item_label.pack(side=tk.LEFT)
            
            # Entry to display selected files
            file_entry = ttk.Entry(item_frame, width=40)
            if i < len(current_values) and current_values[i]:
                if isinstance(current_values[i], list):
                    file_entry.insert(0, "; ".join(current_values[i]))
                else:
                    file_entry.insert(0, str(current_values[i]))
            file_entry.pack(side=tk.LEFT, padx=(5, 5))
            
            # Store the files list separately
            files_list = current_values[i] if i < len(current_values) and isinstance(current_values[i], list) else []
            sample_widgets.append({"entry": file_entry, "files": files_list})
            
            # Browse button for multiple files
            def browse_multiple_files(idx, f_widget, widgets=sample_widgets, n=field_name, a=func_alias):
                files = filedialog.askopenfilenames(title=f"Select files for Sample {idx+1}")
                if files:
                    f_widget.delete(0, tk.END)
                    f_widget.insert(0, "; ".join(files))
                    widgets[idx]["files"] = list(files)
                    values_list = [w["files"] for w in widgets]
                    self._save_widget_value(a, n, values_list)
            
            browse_btn = ttk.Button(item_frame, text="Browse...", 
                                   command=lambda idx=i, fw=file_entry: browse_multiple_files(idx, fw), width=10)
            browse_btn.pack(side=tk.LEFT)
            
            # Focus out handler
            def on_sample_focus_out(event, idx=i, f_widget=file_entry, widgets=sample_widgets, n=field_name, a=func_alias):
                text = f_widget.get()
                if text.strip():
                    parsed_files = [f.strip() for f in text.split(';') if f.strip()]
                    widgets[idx]["files"] = parsed_files
                else:
                    widgets[idx]["files"] = []
                values_list = [w["files"] for w in widgets]
                self._save_widget_value(a, n, values_list)
            file_entry.bind("<FocusOut>", on_sample_focus_out)
        
        # Save initial values
        initial_values = [w["files"] for w in sample_widgets]
        self._save_widget_value(func_alias, field_name, initial_values)
        
        # Update widget_data
        widget_data["widget"] = sample_widgets

    def _show_routing_tab(self):
        """Show Routing tab with visual connection drawing using canvas."""
        self._clear_tab()
        self.current_tab = "routing"
        
        if not self.methodology_list:
            label = ttk.Label(self.tab_content_frame, text=self.language_manager.translate("ui.messages.empty_methodology", "Add functions to Methodology first"), 
                             font=("Arial", 10, "italic"))
            label.pack(padx=20, pady=20)
            return
        
        title = ttk.Label(self.tab_content_frame, text=self.language_manager.translate("ui.messages.routing_help", "Routing: Connect Function Outputs to Inputs"), 
                         font=("Arial", 11, "bold"))
        title.pack(padx=10, pady=10)
        
        # Main routing frame
        routing_frame = ttk.Frame(self.tab_content_frame)
        routing_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Function selection row (dropdowns side by side)
        selection_frame = ttk.Frame(routing_frame)
        selection_frame.pack(fill=tk.X, pady=10)
        
        # Left side - Output function selection
        left_frame = ttk.LabelFrame(selection_frame, text=self.language_manager.translate("ui.messages.output", "Output"), padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.output_func_var = tk.StringVar(value="--")
        self.output_func_combo = ttk.Combobox(left_frame, textvariable=self.output_func_var, 
                                              state="readonly", width=30)
        self.output_func_combo.pack(fill=tk.X)
        self.output_func_combo.bind("<<ComboboxSelected>>", lambda e: self._on_output_func_selected())
        
        # Right side - Input function selection
        right_frame = ttk.LabelFrame(selection_frame, text=self.language_manager.translate("ui.messages.input", "Input"), padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        self.input_func_var = tk.StringVar(value="--")
        self.input_func_combo = ttk.Combobox(right_frame, textvariable=self.input_func_var, 
                                             state="readonly", width=30)
        self.input_func_combo.pack(fill=tk.X)
        self.input_func_combo.bind("<<ComboboxSelected>>", lambda e: self._on_input_func_selected())
        
        # Populate function dropdowns
        self._populate_function_dropdowns()
        
        # Button row for Routing Map
        button_frame = ttk.Frame(routing_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        routing_map_btn = ttk.Button(
            button_frame,
            text=self.language_manager.translate("ui.buttons.routing_map", "Routing Map"),
            command=self._open_routing_map_window
        )
        routing_map_btn.pack(side=tk.LEFT, padx=5)
        
        Tooltip(routing_map_btn, self.language_manager.translate("ui.tooltips.routing_map", 
                "Open full routing map showing all functions and connections"))
        
        # Canvas area for visual connections
        canvas_frame = ttk.LabelFrame(routing_frame, text=self.language_manager.translate("ui.messages.connections", "Connections"), padding=10)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Canvas with scrollbar
        self.routing_canvas = tk.Canvas(canvas_frame, bg="white", height=400)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.routing_canvas.yview)
        self.routing_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.routing_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel to canvas for vertical scrolling
        def _on_mousewheel(event):
            self.routing_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.routing_canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Initialize UI-specific state variables (but preserve routing_lines data)
        self.routing_buttons = {}  # Map button widget to (function_idx, is_output, param_key, param_name)
        self.selected_button = None  # Currently selected button
        self.button_coordinates = {}  # Map button to (x, y) for line drawing
        
        # Make sure routing_lines dict exists (initialized in __init__)
        if not hasattr(self, 'routing_lines'):
            self.routing_lines = {}
        
        # Draw initial connections from saved routing data
        self._refresh_routing_display()
    
    def _populate_function_dropdowns(self):
        """Populate the function dropdowns for output/input selection."""
        # Build list of functions with their display names
        function_options = ["--"]  # Start with empty selection
        self._function_mapping = {"--": None}  # Map display to function index
        
        for idx, instance_alias in enumerate(self.methodology_list):
            base_alias = self.function_base_aliases[idx]
            func_config = self.gui_configs.get(base_alias, {})
            display_name = func_config.get("display_name", base_alias)
            
            # Show index for duplicate functions
            if self.function_base_aliases.count(base_alias) > 1:
                display = f"[{idx}] {display_name}"
            else:
                display = display_name
            
            function_options.append(display)
            self._function_mapping[display] = idx
        
        self.output_func_combo.config(values=function_options)
        self.input_func_combo.config(values=function_options)
    
    def _on_output_func_selected(self):
        """Handle output function selection - redraw outputs."""
        display = self.output_func_var.get()
        func_idx = self._function_mapping.get(display)
        
        if func_idx is None:  # Empty selection
            return
        
        self._draw_canvas()
    
    def _on_input_func_selected(self):
        """Handle input function selection - redraw inputs."""
        display = self.input_func_var.get()
        func_idx = self._function_mapping.get(display)
        
        if func_idx is None:  # Empty selection
            return
        
        # Check if same function selected on both sides
        output_idx = self._function_mapping.get(self.output_func_var.get())
        input_idx = self._function_mapping.get(self.input_func_var.get())
        
        if output_idx is not None and input_idx is not None and output_idx == input_idx:
            messagebox.showwarning(self.language_manager.translate("ui.dialogs.warning", "Warning"), 
                                 self.language_manager.translate("ui.messages.same_function_error", "Cannot select the same function on both sides"))
            self.input_func_var.set("--")
            return
        
        self._draw_canvas()
    
    def _draw_canvas(self):
        """Draw the routing canvas with outputs and inputs."""
        self.routing_canvas.delete("all")
        self.routing_buttons = {}
        self.button_coordinates = {}
        
        output_display = self.output_func_var.get()
        input_display = self.input_func_var.get()
        
        output_idx = self._function_mapping.get(output_display)
        input_idx = self._function_mapping.get(input_display)
        
        # Calculate layout
        canvas_width = self.routing_canvas.winfo_width()
        if canvas_width < 2:
            canvas_width = 800
        
        left_x = 100  # Left side button column
        right_x = canvas_width - 100  # Right side button column
        start_y = 50
        button_height = 30
        spacing = 28  # Reduced spacing to prevent blending and overlapping
        max_y = start_y  # Track maximum y coordinate for scroll region
        
        # Draw left side (outputs)
        current_y = start_y
        if output_idx is not None:
            instance_alias = self.methodology_list[output_idx]
            base_alias = self.function_base_aliases[output_idx]
            func_config = self.gui_configs.get(base_alias, {})
            display_name = func_config.get("display_name", base_alias)
            outputs = func_config.get("output_aliases", {})
            
            # Draw label centered above output buttons
            self.routing_canvas.create_text(left_x + 65, current_y, text=self.language_manager.translate("ui.messages.outputs", "Outputs"), font=("Arial", 11, "bold"), anchor="center")
            current_y += 25
            
            # Draw output buttons
            for param_key, param_name in outputs.items():
                # Check if this button is selected
                is_selected = self.selected_button is not None and \
                             self.selected_button == (output_idx, True, param_key, param_name)
                
                # Create button with smaller height for better layout
                button = tk.Button(
                    self.routing_canvas,
                    text=f"{param_name} ►",
                    font=("Arial", 9, "bold"),
                    width=16,
                    height=1,
                    bg="#a82434" if is_selected else "#dc2626",
                    fg="white",
                    activebackground="#7f1d1d" if is_selected else "#a82434",
                    activeforeground="white",
                    relief=tk.FLAT,
                    bd=0,
                    padx=8,
                    pady=2,
                    cursor="hand2",
                    anchor="e", #delete to center it
                    command=lambda pk=param_key, pn=param_name, idx=output_idx: 
                        self._on_button_clicked(idx, True, pk, pn)
                )
                window = self.routing_canvas.create_window(left_x, current_y, window=button, anchor="w")
                self.routing_buttons[button] = (output_idx, True, param_key, param_name)
                # Position line origin at right edge of output button (anchor is "w" so add button width offset)
                self.button_coordinates[button] = (left_x + 130, current_y)
                current_y += spacing
                max_y = max(max_y, current_y)
        
        # Draw right side (inputs)
        current_y = start_y
        if input_idx is not None:
            instance_alias = self.methodology_list[input_idx]
            base_alias = self.function_base_aliases[input_idx]
            func_config = self.gui_configs.get(base_alias, {})
            display_name = func_config.get("display_name", base_alias)
            inputs = func_config.get("input_aliases", {})
            
            # Draw label centered above input buttons
            self.routing_canvas.create_text(right_x - 65, current_y, text=self.language_manager.translate("ui.messages.inputs", "Inputs"), font=("Arial", 11, "bold"), anchor="center")
            current_y += 25
            
            # Draw input buttons
            for param_key, param_name in inputs.items():
                # Check if this button is selected
                is_selected = self.selected_button is not None and \
                             self.selected_button == (input_idx, False, param_key, param_name)
                
                # Create button with smaller height for better layout
                button = tk.Button(
                    self.routing_canvas,
                    text=f"► {param_name}",
                    font=("Arial", 9, "bold"),
                    width=16,
                    height=1,
                    bg="#1f6aa5" if is_selected else "#3b82d6",
                    fg="white",
                    activebackground="#1e40af" if is_selected else "#1f6aa5",
                    activeforeground="white",
                    relief=tk.FLAT,
                    bd=0,
                    padx=8,
                    pady=2,
                    cursor="hand2",
                    anchor="w", #delete to center it
                    command=lambda pk=param_key, pn=param_name, idx=input_idx: 
                        self._on_button_clicked(idx, False, pk, pn)
                )
                window = self.routing_canvas.create_window(right_x, current_y, window=button, anchor="e")
                self.routing_buttons[button] = (input_idx, False, param_key, param_name)
                # Position line destination at left edge of input button (anchor is "e" so subtract button width offset)
                self.button_coordinates[button] = (right_x - 130, current_y)
                current_y += spacing
                max_y = max(max_y, current_y)
        
        # Set scroll region to encompass all content
        self.routing_canvas.configure(scrollregion=self.routing_canvas.bbox("all"))
        
        # Redraw existing connections
        self._redraw_existing_lines()
    
    def _on_button_clicked(self, func_idx, is_output, param_key, param_name):
        """Handle button click for output/input selection."""
        current_selection = (func_idx, is_output, param_key, param_name)
        
        # If clicking same button twice, remove connections between currently selected functions
        if self.selected_button == current_selection:
            self.selected_button = None
            
            # Get currently selected functions from comboboxes
            output_display = self.output_func_var.get()
            input_display = self.input_func_var.get()
            selected_output_idx = self._function_mapping.get(output_display)
            selected_input_idx = self._function_mapping.get(input_display)
            
            # Remove routing connections that involve this parameter AND match the selected functions
            keys_to_remove = []
            for key in self.routing_lines.keys():
                src_idx, src_param_key, dst_idx, dst_param_key = key
                if is_output:
                    # Removing output: find connections from selected output function to selected input function
                    if (src_idx == func_idx and src_param_key == param_key and 
                        dst_idx == selected_input_idx and selected_input_idx is not None):
                        keys_to_remove.append(key)
                else:
                    # Removing input: find connections from selected output function to selected input function
                    if (dst_idx == func_idx and dst_param_key == param_key and 
                        src_idx == selected_output_idx and selected_output_idx is not None):
                        keys_to_remove.append(key)
            
            # Remove the identified connections
            if keys_to_remove:
                for key in keys_to_remove:
                    del self.routing_lines[key]
            
            self._draw_canvas()
            return
        
        # If clicking on same side (output or input), deselect previous and select new
        if self.selected_button is not None and self.selected_button[1] == is_output:
            self.selected_button = current_selection
            self._draw_canvas()
            return
        
        # If no selection yet, select this button
        if self.selected_button is None:
            self.selected_button = current_selection
            self._draw_canvas()
            return
        
        # If clicking on other side, try to create connection
        src_idx, src_is_output, src_param_key, src_param_name = self.selected_button
        
        # Determine source and destination
        if src_is_output:
            # Selected output, now clicking input
            dst_idx = func_idx
            dst_param_key = param_key
            dst_param_name = param_name
        else:
            # Selected input, now clicking output - swap them
            src_idx, dst_idx = func_idx, src_idx
            src_param_key, dst_param_key = param_key, src_param_key
            src_param_name, dst_param_name = param_name, src_param_name
        
        # Validate connection (output source before input dest)
        if src_idx >= dst_idx:
            self.selected_button = None
            self._draw_canvas()
            return
        
        # Create or remove connection
        key = (src_idx, src_param_key, dst_idx, dst_param_key)
        
        if key in self.routing_lines:
            # Remove existing connection
            del self.routing_lines[key]
        else:
            # Add new connection
            src_instance = self.methodology_list[src_idx]
            dst_instance = self.methodology_list[dst_idx]
            src_base = self.function_base_aliases[src_idx]
            dst_base = self.function_base_aliases[dst_idx]
            src_func_config = self.gui_configs.get(src_base, {})
            dst_func_config = self.gui_configs.get(dst_base, {})
            src_display = src_func_config.get("display_name", src_base)
            dst_display = dst_func_config.get("display_name", dst_base)
            
            self.routing_lines[key] = {
                "src_idx": src_idx,
                "src_param_key": src_param_key,
                "src_param_name": src_param_name,
                "dst_idx": dst_idx,
                "dst_param_key": dst_param_key,
                "dst_param_name": dst_param_name,
                "src_display": src_display,
                "dst_display": dst_display
            }
        
        self.selected_button = None
        self._draw_canvas()
    
    def _redraw_existing_lines(self):
        """Draw lines for existing routing connections on canvas."""
        # Only draw lines if both output and input functions are selected and visible
        if not self.button_coordinates:
            return
        
        for key, conn_info in self.routing_lines.items():
            src_idx = conn_info.get("src_idx", key[0])
            src_param_key = conn_info.get("src_param_key", key[1])
            dst_idx = conn_info.get("dst_idx", key[2])
            dst_param_key = conn_info.get("dst_param_key", key[3])
            
            # Find button coordinates - both must exist on current canvas
            src_coords = None
            dst_coords = None
            
            for button, (func_idx, is_output, param_key, param_name) in self.routing_buttons.items():
                # Check if this button matches the source (output)
                if func_idx == src_idx and is_output and param_key == src_param_key:
                    if button in self.button_coordinates:
                        src_coords = self.button_coordinates[button]
                # Check if this button matches the destination (input)
                elif func_idx == dst_idx and not is_output and param_key == dst_param_key:
                    if button in self.button_coordinates:
                        dst_coords = self.button_coordinates[button]
            
            # Draw smooth curved line if both endpoints found
            if src_coords and dst_coords:
                # Create smooth S-curve with anti-aliasing-like effect using high spline steps
                x1, y1 = src_coords
                x2, y2 = dst_coords
                
                # Calculate horizontal distance for better curve proportions
                horizontal_distance = abs(x2 - x1)
                
                # Control offset creates proportional curves regardless of distance
                # Larger offset = more pronounced S-curve shape
                ctrl_offset = min(180, horizontal_distance * 0.55)
                
                # Create smooth spline with many steps for near-perfect smoothness
                # High splinesteps count simulates anti-aliasing effect
                self.routing_canvas.create_line(
                    x1, y1,
                    x1 + ctrl_offset, y1,      # Exit output smoothly (horizontal)
                    x2 - ctrl_offset, y2,      # Approach input smoothly (horizontal)
                    x2, y2,
                    fill="#6F8370",  #5a9d5a          # Lighter green for semi-transparent appearance
                    width=6,
                    smooth=True,
                    splinesteps=60
                )
    
    def _refresh_routing_display(self):
        """Refresh routing display (called when opening tab)."""
        # Just draw the initial canvas
        self._draw_canvas()

    def _open_routing_map_window(self):
        """Open the full routing map window."""
        if not self.methodology_list:
            messagebox.showinfo(
                self.language_manager.translate("ui.dialogs.info", "Info"),
                self.language_manager.translate("ui.messages.empty_methodology", "Add functions to Methodology first")
            )
            return
        
        # Open routing map window
        RoutingMapWindow(
            self.root,
            self.methodology_list,
            self.function_base_aliases,
            self.routing_lines,
            self.gui_configs,
            FUNCTION_SPECS
        )

    def _position_paned_sash(self, paned):
        """Position the PanedWindow sash to the middle."""
        try:
            # Force window to update first
            paned.update_idletasks()
            # Get the current paned window width
            parent_width = paned.winfo_width()
            if parent_width > 1:  # Only set if window has been rendered
                sash_pos = parent_width // 2
                paned.sashpos(0, sash_pos)
        except Exception as e:
            pass  # Silently fail if sash positioning isn't available

    def _position_fd_sashes(self, main_paned, top_paned, bottom_paned):
        """Position the sashes for fd (Four Divided) layout."""
        try:
            # Force window to update first
            main_paned.update_idletasks()
            
            # Get dimensions
            main_height = main_paned.winfo_height()
            top_width = top_paned.winfo_width()
            bottom_width = bottom_paned.winfo_width()
            
            # Position vertical sash in main paned window (center vertically)
            if main_height > 1:
                vertical_sash_pos = main_height // 2
                main_paned.sashpos(0, vertical_sash_pos)
            
            # Position horizontal sash in top paned window (center horizontally)
            if top_width > 1:
                horizontal_sash_pos = top_width // 2
                top_paned.sashpos(0, horizontal_sash_pos)
            
            # Position horizontal sash in bottom paned window (center horizontally)
            if bottom_width > 1:
                horizontal_sash_pos = bottom_width // 2
                bottom_paned.sashpos(0, horizontal_sash_pos)
        except Exception as e:
            pass  # Silently fail if sash positioning isn't available

    
    def _show_analysis_tab(self):
        """Show Analysis tab with analysis and visualization tools."""
        self._clear_tab()
        self.current_tab = "analysis"
        
        if self.selected_function_idx is None:
            label = ttk.Label(self.tab_content_frame, text=self.language_manager.translate("ui.messages.no_methodology", "No functions selected. Please add functions to your methodology."), 
                             font=("Arial", 10, "italic"))
            label.pack(padx=20, pady=20)
            return
        
        instance_alias = self.methodology_list[self.selected_function_idx]
        base_alias = self.function_base_aliases[self.selected_function_idx]
        
        # Initialize analysis data in model if not present
        if not hasattr(self, 'analysis_data'):
            self.analysis_data = {}
        
        # Get or create analysis entry for this function
        if instance_alias not in self.analysis_data:
            # Load analysis configuration from function's gui_config if available
            analysis_config = None
            if base_alias in self.gui_configs:
                analysis_config = self.gui_configs[base_alias].get('analysis')
            
            if analysis_config:
                # Use analysis config from function's JSON
                # Deep copy pages to prevent modifications from affecting gui_configs
                self.analysis_data[instance_alias] = {
                    'pages': copy.deepcopy(analysis_config.get('pages', [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}])),
                    'current_page': analysis_config.get('current_page', 0)
                }
            else:
                # Fallback to default structure
                self.analysis_data[instance_alias] = {
                    'pages': [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}],
                    'current_page': 0
                }
        
        analysis_info = self.analysis_data[instance_alias]
        
        # Create top control bar
        control_frame = ttk.Frame(self.tab_content_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Get function display name
        config = self.gui_configs.get(base_alias, {})
        display_name = config.get("display_name", base_alias)
        
        title = ttk.Label(control_frame, text=f"Analysis: {display_name}", font=("Arial", 11, "bold"))
        title.pack(side=tk.LEFT, padx=5)
        
        # Run to here button
        run_btn = ttk.Button(control_frame, text="🠊 Run to here", 
                            command=lambda: self._run_analysis_to_function(instance_alias))
        run_btn.pack(side=tk.LEFT, padx=5)
        
        # Add graph button
        add_graph_btn = ttk.Button(control_frame, text="Add graph", 
                                   command=lambda: messagebox.showinfo("Info", "Add Graph feature is still in development"))
        add_graph_btn.pack(side=tk.LEFT, padx=5)
        
        # Add table button
        add_table_btn = ttk.Button(control_frame, text="Add table", 
                                   command=lambda: messagebox.showinfo("Info", "Add Table feature is still in development"))
        add_table_btn.pack(side=tk.LEFT, padx=5)
        
        # Remove section button
        remove_section_btn = ttk.Button(control_frame, text="Remove section", 
                                       command=lambda: self._show_remove_section_dialog(instance_alias))
        remove_section_btn.pack(side=tk.LEFT, padx=5)
        
        # Spacer
        spacer = ttk.Frame(control_frame)
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add page button
        add_page_btn = ttk.Button(control_frame, text="Add Page", 
                                 command=lambda: self._show_add_page_dialog(instance_alias))
        add_page_btn.pack(side=tk.RIGHT, padx=5)
        
        # Remove page button
        remove_page_btn = ttk.Button(control_frame, text="Remove Page", 
                                    command=lambda: self._remove_current_page(instance_alias))
        remove_page_btn.pack(side=tk.RIGHT, padx=5)
        
        # Build list of visible pages (pages that pass conditions)
        pages = analysis_info.get('pages', [])
        visible_pages = []  # List of (idx, page_data) tuples for pages that are visible
        
        # Only filter by condition if we have execution results with inputs
        has_execution_results = 'execution_results' in analysis_info and analysis_info['execution_results'].get('inputs')
        
        for idx, page in enumerate(pages):
            # Check if page has a condition
            if page.get('condition'):
                # Only apply condition if we have execution results; otherwise show all pages
                if has_execution_results:
                    if not self._evaluate_condition(instance_alias, page.get('condition')):
                        continue  # Skip pages that don't meet condition
            # Add all pages that don't have conditions, or pages with conditions that pass
            visible_pages.append((idx, page))
        
        # Get current page, ensuring it's a valid visible page
        current_page = analysis_info.get('current_page', 0)
        # Find the position of current_page in visible_pages
        visible_page_idx = 0
        for i, (idx, page) in enumerate(visible_pages):
            if idx == current_page:
                visible_page_idx = i
                break
        
        # Page navigation frame (bottom) - PACK THIS FIRST so it doesn't get pushed off screen
        nav_frame = ttk.Frame(self.tab_content_frame)
        nav_frame.pack(fill=tk.X, padx=10, pady=(0, 10), side=tk.BOTTOM)
        
        # Page display label (first)
        if visible_pages:
            current_idx, current_page_data = visible_pages[visible_page_idx]
            page_title = current_page_data.get('title', f'Page {current_idx + 1}')
            page_info = f"Page {visible_page_idx + 1}/{len(visible_pages)}: {page_title}"
        else:
            page_info = "No pages available"
        
        page_label = ttk.Label(nav_frame, text=page_info, font=("Arial", 9))
        page_label.pack(side=tk.LEFT, padx=10)
        
        # Previous page button
        prev_btn = ttk.Button(nav_frame, text="← Previous", width=10,
                             command=lambda: self._switch_analysis_page_relative(instance_alias, -1))
        prev_btn.pack(side=tk.LEFT, padx=2)
        
        # Next page button
        next_btn = ttk.Button(nav_frame, text="Next →", width=10,
                             command=lambda: self._switch_analysis_page_relative(instance_alias, 1))
        next_btn.pack(side=tk.LEFT, padx=2)
        
        # Main content area - PACK THIS AFTER so it expands into remaining space
        content_frame = ttk.Frame(self.tab_content_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, side=tk.TOP)
        
        # Display current page
        if visible_pages:
            current_idx, page_data = visible_pages[visible_page_idx]
            self._render_analysis_page(content_frame, instance_alias, page_data)
    
    def _switch_analysis_page(self, instance_alias: str, page_idx: int):
        """Switch to a different analysis page."""
        if instance_alias in self.analysis_data:
            self.analysis_data[instance_alias]['current_page'] = page_idx
            self._show_analysis_tab()
    
    def _switch_analysis_page_relative(self, instance_alias: str, direction: int):
        """Switch to the previous or next visible analysis page.
        
        Args:
            instance_alias: The function instance alias
            direction: -1 for previous, 1 for next
        """
        if instance_alias not in self.analysis_data:
            return
        
        analysis_info = self.analysis_data[instance_alias]
        pages = analysis_info.get('pages', [])
        current_page = analysis_info.get('current_page', 0)
        
        # Build list of visible pages
        visible_pages = []
        for idx, page in enumerate(pages):
            if page.get('condition'):
                if not self._evaluate_condition(instance_alias, page.get('condition')):
                    continue
            visible_pages.append((idx, page))
        
        if not visible_pages:
            return
        
        # Find current page position in visible pages
        current_visible_idx = 0
        for i, (idx, page) in enumerate(visible_pages):
            if idx == current_page:
                current_visible_idx = i
                break
        
        # Calculate new position
        new_visible_idx = current_visible_idx + direction
        
        # Clamp to valid range
        if new_visible_idx < 0:
            new_visible_idx = 0
        elif new_visible_idx >= len(visible_pages):
            new_visible_idx = len(visible_pages) - 1
        
        # Switch to the new page
        new_page_idx = visible_pages[new_visible_idx][0]
        self.analysis_data[instance_alias]['current_page'] = new_page_idx
        self._show_analysis_tab()
    
    def _evaluate_condition(self, instance_alias: str, condition: dict) -> bool:
        """Evaluate a condition against execution inputs.
        
        Args:
            instance_alias: The function instance alias
            condition: Dict with 'parameter', 'operator', and 'value' keys
                      Example: {"parameter": "nway_flag", "operator": ">", "value": "1"}
        
        Returns:
            True if condition is met, False otherwise
        """
        if not condition or not isinstance(condition, dict):
            return True
        
        parameter = condition.get('parameter')
        operator = condition.get('operator', '==')
        expected_value = condition.get('value')
        
        if not parameter:
            return True
        
        # Get the actual value from execution inputs
        if instance_alias not in self.analysis_data:
            return True
        
        execution_results = self.analysis_data[instance_alias].get('execution_results', {})
        inputs = execution_results.get('inputs', {})
        actual_value = inputs.get(parameter)
        
        if actual_value is None:
            # If parameter not found in inputs, default to showing the page
            return True
        
        # Convert both values to appropriate types for comparison
        try:
            # Try to convert to int/float for numeric comparisons
            if operator in ['>', '<', '>=', '<=']:
                try:
                    actual_value = int(actual_value) if isinstance(actual_value, str) else actual_value
                    expected_value = int(expected_value) if isinstance(expected_value, str) else expected_value
                except (ValueError, TypeError):
                    # If conversion fails, convert to strings for comparison
                    actual_value = str(actual_value)
                    expected_value = str(expected_value)
            else:
                # For == and != operators, convert both to strings for consistent comparison
                actual_value = str(actual_value)
                expected_value = str(expected_value)
        except Exception:
            pass
        
        # Evaluate based on operator
        try:
            if operator == '==':
                return actual_value == expected_value
            elif operator == '!=':
                return actual_value != expected_value
            elif operator == '>':
                return actual_value > expected_value
            elif operator == '<':
                return actual_value < expected_value
            elif operator == '>=':
                return actual_value >= expected_value
            elif operator == '<=':
                return actual_value <= expected_value
            elif operator == 'in':
                return actual_value in expected_value
            elif operator == 'contains':
                return expected_value in actual_value
            else:
                return True
        except (TypeError, ValueError):
            return False
    
    def _render_analysis_page(self, parent: ttk.Frame, instance_alias: str, page_data: dict):
        """Render the current analysis page with the specified layout."""
        layout_type = page_data.get('layout', 'fp')
        sections = page_data.get('sections', [])
        
        # Create layout containers
        containers = self._create_layout_containers(parent, layout_type)
        
        # Populate sections, filtering by condition if present
        section_idx = 0
        for container in containers:
            # Find next section that passes condition check
            while section_idx < len(sections):
                section_data = sections[section_idx]
                section_idx += 1
                
                # Skip removed sections (type is None)
                if section_data.get('type') is None:
                    continue  # Skip this section, try next
                
                # Check if section has a condition
                if section_data.get('condition'):
                    if not self._evaluate_condition(instance_alias, section_data.get('condition')):
                        continue  # Skip this section, try next
                
                # Render the section
                self._render_section(container, instance_alias, section_data, section_idx - 1)
                break
            else:
                # No more sections with passing conditions
                placeholder = ttk.Label(container, text="[Empty Section]", foreground="gray")
                placeholder.pack(expand=True)
    
    def _create_layout_containers(self, parent: ttk.Frame, layout_type: str) -> list:
        """Create layout containers based on layout type."""
        containers = []
        # Use consistent padding for all section containers to prevent border clipping
        section_padding = 8
        
        if layout_type == 'fd':  # Four sections (2x2 grid)
            # Use nested paned windows to ensure equal space distribution
            main_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
            main_paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Top row with horizontal paned window for 2 side-by-side containers
            top_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
            main_paned.add(top_paned, weight=1)
            for j in range(2):
                container = ttk.LabelFrame(top_paned, text=f"Section", padding=section_padding)
                top_paned.add(container, weight=1)
                containers.append(container)
            
            # Bottom row with horizontal paned window for 2 side-by-side containers
            bottom_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
            main_paned.add(bottom_paned, weight=1)
            for j in range(2):
                container = ttk.LabelFrame(bottom_paned, text=f"Section", padding=section_padding)
                bottom_paned.add(container, weight=1)
                containers.append(container)
            
            # Position sashes after rendering
            parent.after_idle(lambda: self._position_fd_sashes(main_paned, top_paned, bottom_paned))
        
        elif layout_type == 'fp':  # Full page (1 section)
            container = ttk.LabelFrame(parent, text="Section", padding=section_padding)
            container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            containers.append(container)
        
        elif layout_type == 'ns':  # North-South (2 sections: top, bottom)
            top_frame = ttk.LabelFrame(parent, text="Section", padding=section_padding)
            top_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            containers.append(top_frame)
            
            bottom_frame = ttk.LabelFrame(parent, text="Section", padding=section_padding)
            bottom_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            containers.append(bottom_frame)
        
        elif layout_type == 'ew':  # East-West (2 sections: left, right)
            paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
            paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            left_container = ttk.LabelFrame(paned, text="Section", padding=section_padding)
            paned.add(left_container, weight=1)
            containers.append(left_container)
            
            right_container = ttk.LabelFrame(paned, text="Section", padding=section_padding)
            paned.add(right_container, weight=1)
            containers.append(right_container)
            
            parent.after_idle(lambda: self._position_paned_sash(paned))
        
        elif layout_type == 'sd':  # South Divided (3 sections: 1 top, 2 bottom)
            # Use vertical paned window for top/bottom division
            main_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
            main_paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Top section
            top_frame = ttk.LabelFrame(main_paned, text="Section", padding=section_padding)
            main_paned.add(top_frame, weight=1)
            containers.append(top_frame)
            
            # Bottom side with horizontal paned window for 2 side-by-side containers
            bottom_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
            main_paned.add(bottom_paned, weight=1)
            for j in range(2):
                container = ttk.LabelFrame(bottom_paned, text=f"Section", padding=section_padding)
                bottom_paned.add(container, weight=1)
                containers.append(container)
        
        elif layout_type == 'nd':  # North Divided (3 sections: 2 top, 1 bottom)
            # Use vertical paned window for top/bottom division
            main_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
            main_paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Top side with horizontal paned window for 2 side-by-side containers
            top_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
            main_paned.add(top_paned, weight=1)
            for j in range(2):
                container = ttk.LabelFrame(top_paned, text=f"Section", padding=section_padding)
                top_paned.add(container, weight=1)
                containers.append(container)
            
            # Bottom section
            bottom_frame = ttk.LabelFrame(main_paned, text="Section", padding=section_padding)
            main_paned.add(bottom_frame, weight=1)
            containers.append(bottom_frame)
        
        elif layout_type == 'ed':  # East Divided (3 sections: 1 left, 2 right stacked)
            paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
            paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Left side (single container)
            left_container = ttk.LabelFrame(paned, text="Section", padding=section_padding)
            paned.add(left_container, weight=1)
            containers.append(left_container)
            
            # Right side with vertical paned window for 2 stacked containers
            right_paned = ttk.PanedWindow(paned, orient=tk.VERTICAL)
            paned.add(right_paned, weight=1)
            
            for i in range(2):
                container = ttk.LabelFrame(right_paned, text=f"Section", padding=section_padding)
                right_paned.add(container, weight=1)
                containers.append(container)
            
            parent.after_idle(lambda: self._position_paned_sash(paned))
        
        elif layout_type == 'wd':  # West Divided (3 sections: 2 left stacked, 1 right)
            paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
            paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Left side with vertical paned window for 2 stacked containers
            left_paned = ttk.PanedWindow(paned, orient=tk.VERTICAL)
            paned.add(left_paned, weight=1)
            
            for i in range(2):
                container = ttk.LabelFrame(left_paned, text=f"Section", padding=section_padding)
                left_paned.add(container, weight=1)
                containers.append(container)
            
            # Right side (single container)
            right_container = ttk.LabelFrame(paned, text="Section", padding=section_padding)
            paned.add(right_container, weight=1)
            containers.append(right_container)
            
            parent.after_idle(lambda: self._position_paned_sash(paned))
        
        else:
            # Default to full page for unknown layouts
            container = ttk.LabelFrame(parent, text="Section", padding=section_padding)
            container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            containers.append(container)
        
        return containers
    
    def _render_section(self, parent: ttk.Frame, instance_alias: str, section_data: dict, section_idx: int = 0, is_popup: bool = False):
        """Render a section (either graph or table)."""
        section_type = section_data.get('type')
        
        # Set the section frame title from config's 'title' field if parent is a LabelFrame
        config = section_data.get('config', {})
        section_title = config.get('title', 'Section')
        if hasattr(parent, 'configure'):
            try:
                parent.configure(text=section_title)
            except tk.TclError:
                pass  # Parent may not be a LabelFrame
        
        if section_type == 'graph':
            self._render_graph_section(parent, instance_alias, section_data, section_idx)
        elif section_type == 'table':
            self._render_table_section(parent, instance_alias, section_data, section_idx)
        else:
            # Empty section
            label = ttk.Label(parent, text="[Empty]", foreground="gray")
            label.pack(expand=True)
        
        # Add popup button AFTER content is rendered (only if not already in a popup)
        if not is_popup:
            self._create_section_popup_button(parent, instance_alias, section_idx, section_data)
    
    def _resolve_axis_label(self, axis_config: dict, outputs: dict) -> str:
        """Resolve an axis label from axis configuration.
        
        The label configuration supports:
        - Direct string: "label": "My Axis"
        - Variable reference: "label": "variable_name"
        - Variable with index: "label": "variable_name", "l_index": 0
        
        If the variable is a list and l_index is provided, returns the value at that index.
        Otherwise, returns the variable value as a string.
        
        Args:
            axis_config: Axis configuration dict with 'label' field
            outputs: Dictionary of execution outputs
            
        Returns:
            Resolved label as string, or empty string if not found
        """
        if not axis_config:
            return ""
        
        label_config = axis_config.get('label')
        if not label_config:
            return ""
        
        # First try to resolve as variable reference
        if isinstance(label_config, str):
            if label_config in outputs:
                # It's a variable name
                data = outputs[label_config]
                l_index = axis_config.get('l_index')
                
                if l_index is not None and isinstance(data, (list, np.ndarray)):
                    # Use index to select from list
                    try:
                        result = str(data[l_index])
                        return result
                    except (IndexError, TypeError):
                        return str(data) if not isinstance(data, (list, np.ndarray)) else label_config
                elif isinstance(data, str):
                    return data
                elif isinstance(data, (int, float)):
                    return str(data)
                elif isinstance(data, (list, np.ndarray)) and len(data) > 0:
                    # If no l_index but it's a list, return first element or joined string
                    return str(data[0]) if len(data) == 1 else label_config
                else:
                    return str(data)
            else:
                # It's a literal string label
                return label_config
        else:
            return str(label_config)
    
    def _get_variable_label(self, outputs: dict, var_name: str, dimension: int, index: int, fallback: bool = True) -> Optional[str]:
        """Get a label from a multi-dimensional variable labels configuration.
        
        Variable should be a list of lists (one list per dimension).
        For dimension D, position P, returns the label from lists[D][P].
        
        Args:
            outputs: Dictionary of execution outputs
            var_name: Name of variable containing lists of labels
            dimension: Which dimension/list to access
            index: Index within that dimension's list
            fallback: If True, return "V{index+1}" when data is missing/empty
            
        Returns:
            Label string, fallback "V{index+1}" if fallback=True, or None
        """
        fallback_label = f"V{index + 1}" if fallback else None
        
        if var_name not in outputs:
            return fallback_label
        
        data = outputs[var_name]
        if isinstance(data, (list, np.ndarray)):
            try:
                # Check if dimension index is valid
                if dimension >= len(data):
                    return fallback_label
                    
                dim_list = data[dimension]
                
                # Check if dimension list is empty or None
                if dim_list is None or (isinstance(dim_list, (list, np.ndarray)) and len(dim_list) == 0):
                    return fallback_label
                
                if isinstance(dim_list, (list, np.ndarray)):
                    if index >= len(dim_list):
                        return fallback_label
                    value = dim_list[index]
                    return str(value) if value is not None else fallback_label
                else:
                    # Single dimension - data itself is the list
                    if index >= len(data):
                        return fallback_label
                    value = data[index]
                    return str(value) if value is not None else fallback_label
            except (IndexError, TypeError):
                return fallback_label
        return fallback_label
    
    def _get_dimension_labels(self, outputs: dict, config: dict) -> dict:
        """Get dimension labels from config or outputs.
        
        Returns a dict mapping dimension index to label string.
        Checks 'dimension_labels' in config which can be:
        - A variable name referencing a list in outputs
        - A direct list of labels
        
        Args:
            outputs: Dictionary of execution outputs
            config: Graph configuration dict
            
        Returns:
            Dict mapping dimension index to label string
        """
        dim_labels = {}
        labels_config = config.get('dimension_labels')
        
        if labels_config is None:
            return dim_labels
        
        if isinstance(labels_config, str) and labels_config in outputs:
            # Variable reference
            data = outputs[labels_config]
            if isinstance(data, (list, np.ndarray)):
                for i, label in enumerate(data):
                    if label:
                        dim_labels[i] = str(label)
        elif isinstance(labels_config, (list, np.ndarray)):
            # Direct list
            for i, label in enumerate(labels_config):
                if label:
                    dim_labels[i] = str(label)
        elif isinstance(labels_config, dict):
            # Direct mapping
            for k, v in labels_config.items():
                try:
                    dim_labels[int(k)] = str(v)
                except (ValueError, TypeError):
                    pass
        
        return dim_labels
    
    def _render_graph_section(self, parent: ttk.Frame, instance_alias: str, section_data: dict, section_idx: int = 0):
        """Render a graph using matplotlib with optional navigation controls."""
        try:
            config = section_data.get('config', {})
            graph_type = config.get('graph_type', 'scatter')
            
            # Get execution results
            if instance_alias not in self.analysis_data:
                label = ttk.Label(parent, text="No data available - Please run 'Run Model' or 'Run to here' first", foreground="gray")
                label.pack(expand=True)
                return
            
            execution_results = self.analysis_data[instance_alias].get('execution_results', {})
            if not execution_results:
                label = ttk.Label(parent, text="No data available - Please run 'Run Model' or 'Run to here' first", foreground="gray")
                label.pack(expand=True)
                return
            
            if execution_results.get('status') != 'success':
                label = ttk.Label(parent, text="Execution failed - Check model_log.txt for details", foreground="red")
                label.pack(expand=True)
                return
            
            outputs = execution_results.get('outputs', {})
            
            # Initialize slice state if needed
            # Use section_idx as the key to ensure each graph (even in same section) has unique state
            section_id = section_idx
            if instance_alias not in self.analysis_data:
                self.analysis_data[instance_alias] = {}
            if 'graph_slices' not in self.analysis_data[instance_alias]:
                self.analysis_data[instance_alias]['graph_slices'] = {}
            
            # Get current slices for this section
            slice_state = self.analysis_data[instance_alias]['graph_slices']
            if section_id not in slice_state:
                # Initialize slices from config or defaults
                slice_info = config.get('slice_info', {})
                nav_axes = config.get('data_slicing', [])
                
                # Build indices dict for multi-dimensional navigation
                # Support both old format (list of strings) and new format (list of dicts)
                indices = {}
                for nav_item in nav_axes:
                    if isinstance(nav_item, dict):
                        # New format: {"name": "Samples", "dimension": 0}
                        dim = nav_item.get('dimension', 0)
                        indices[dim] = slice_info.get(f'index_{dim}', 0)
                    else:
                        # Old format: just a string, assume dimension index matches position
                        dim = len(indices)  # Position in the list
                        indices[dim] = slice_info.get(f'index_{dim}', 0)
                
                slice_state[section_id] = {
                    'indices': indices,  # Now a dict: {dimension: index}
                    'data_slicing': nav_axes,
                    'outputs': outputs,
                    'config': config,
                    'graph_type': graph_type
                }
            
            current_slice = slice_state[section_id]
            
            # Initialize axis indices and dimension slicing indices with defaults BEFORE extraction
            nav_axes = config.get('data_slicing', [])
            
            if 'axis_indices' not in current_slice:
                current_slice['axis_indices'] = {}
                # Only initialize axis_indices for axes that appear in data_slicing with "axis" field
                for nav_item in nav_axes:
                    if isinstance(nav_item, dict):
                        target_axis = nav_item.get('axis')
                        if target_axis:  # Only for axis selection items
                            # Only set axis index if dimension is explicitly specified
                            if 'dimension' in nav_item:
                                axis_indices_dict = {}
                                dimension = nav_item['dimension']
                                # Get default from nav_item or axis config
                                default_val = nav_item.get('default', None)
                                if default_val is None:
                                    default_val = config.get(f'{target_axis}_axis', {}).get('default_column', 0)
                                axis_indices_dict[dimension] = default_val
                                current_slice['axis_indices'][target_axis] = axis_indices_dict
            
            if 'indices' not in current_slice:
                current_slice['indices'] = {}
                # Set default indices based on navigation_axes config for non-axis slicing
                for nav_item in nav_axes:
                    if isinstance(nav_item, dict):
                        # Only process if dimension is explicitly specified
                        # If dimension is missing, it will be handled by md_slice_indices
                        if 'dimension' in nav_item:
                            dimension = nav_item['dimension']
                            target_axis = nav_item.get('axis')
                            # Only use default for non-axis items (slicing mode)
                            if not target_axis:
                                default_idx = nav_item.get('default', 0)
                                current_slice['indices'][dimension] = default_idx
            
            # Extract axis data with current slices
            # For each axis, merge its per-axis indices with the shared indices
            base_indices = current_slice.get('indices', {})
            axis_indices = current_slice.get('axis_indices', {})
            md_slice_indices = current_slice.get('md_slice_indices', {})
            
            # For 4D+ data with aux_axis, ensure md_slice_indices is initialized
            if 'aux_axis' in config:
                # Get data source to check if we need 4D+ initialization
                data_source_check = config.get('z_axis', {}).get('data_source')
                if data_source_check and data_source_check in outputs:
                    data_check = outputs[data_source_check]
                    if isinstance(data_check, np.ndarray) and data_check.ndim >= 4:
                        # Initialize md_combo_index if not present
                        if 'md_combo_index' not in current_slice:
                            md_defaults = config.get('md_default', {})
                            current_slice['md_combo_index'] = md_defaults.get('combo_index', 0)
                        
                        # Initialize md_slice_indices if not present
                        if 'md_slice_indices' not in current_slice:
                            nav_axes_init = config.get('data_slicing', [])
                            specified_dims_init = set()
                            for nav_item in nav_axes_init:
                                if isinstance(nav_item, dict):
                                    dim = nav_item.get('dimension')
                                    if dim is not None:
                                        specified_dims_init.add(dim)
                            
                            # Compute combinations
                            combo_size_init = 2
                            md_combinations_init = self._compute_dimension_combinations(data_check.shape, specified_dims_init, combo_size_init)
                            
                            if md_combinations_init:
                                current_slice['md_slice_indices'] = {}
                                combo_idx_init = current_slice.get('md_combo_index', 0)
                                if combo_idx_init < len(md_combinations_init):
                                    current_combo_init = md_combinations_init[combo_idx_init]
                                    all_dims_init = set(range(len(data_check.shape)))
                                    navigable_dims_init = all_dims_init - set(current_combo_init) - specified_dims_init
                                    
                                    md_defaults = config.get('md_default', {})
                                    for dim in navigable_dims_init:
                                        default_val = md_defaults.get(f'dim_{dim}', 0)
                                        max_idx = data_check.shape[dim] - 1
                                        if default_val < 0 or default_val > max_idx:
                                            default_val = 0
                                        current_slice['md_slice_indices'][dim] = default_val
                        
                        # Update md_slice_indices reference
                        md_slice_indices = current_slice.get('md_slice_indices', {})
            
            # For 4D+ data, get the active dimension combination
            md_active_dims = None
            if md_slice_indices:
                # Get the active combination from the computed combinations
                nav_axes_temp = config.get('data_slicing', [])
                specified_dims = set()
                for nav_item in nav_axes_temp:
                    if isinstance(nav_item, dict):
                        dim = nav_item.get('dimension')
                        if dim is not None:
                            specified_dims.add(dim)
                
                # Get data to compute combinations
                # For aux_axis configs, use z_axis as it points to actual data
                data_source_temp = None
                if 'aux_axis' in config:
                    data_source_temp = config.get('z_axis', {}).get('data_source')
                else:
                    axis_config_temp = config.get('y_axis', {}) or config.get('x_axis', {})
                    data_source_temp = axis_config_temp.get('data_source')
                
                if data_source_temp and data_source_temp in outputs:
                    data_temp = outputs[data_source_temp]
                    if isinstance(data_temp, np.ndarray):
                        # For heatmaps, we display 2 dimensions (x, y) and navigate through others
                        # Combo size is always 2 regardless of how many dimensions are specified
                        combo_size = 2
                        md_combinations = self._compute_dimension_combinations(data_temp.shape, specified_dims, combo_size)
                        
                        combo_idx = current_slice.get('md_combo_index', 0)
                        if combo_idx < len(md_combinations):
                            md_active_dims = list(md_combinations[combo_idx])
            
            # For 4D+ data with aux_axis config, create axis configs automatically
            if 'aux_axis' in config and md_active_dims:
                # aux_axis contains data_source and labels for all dimensions
                aux_axis_config = config['aux_axis']
                data_source = aux_axis_config.get('data_source')
                labels_config = aux_axis_config.get('labels', [])
                
                # Resolve labels - can be array or variable name
                labels = []
                if isinstance(labels_config, str):
                    # labels is a variable name - look it up in outputs
                    if labels_config in outputs:
                        label_data = outputs[labels_config]
                        if isinstance(label_data, (list, np.ndarray)):
                            labels = [str(lbl) for lbl in label_data]
                elif isinstance(labels_config, list):
                    # labels is a direct array
                    labels = labels_config
                
                # x-axis uses first dimension in active combination
                x_axis_config = None
                if len(md_active_dims) > 0:
                    dim_idx = md_active_dims[0]
                    x_label = labels[dim_idx] if dim_idx < len(labels) else f"Axis {dim_idx}"
                    x_axis_config = {'data_source': data_source, 'index': dim_idx, 'label': x_label}
                
                # y-axis uses second dimension in active combination
                y_axis_config = None
                if len(md_active_dims) > 1:
                    dim_idx = md_active_dims[1]
                    y_label = labels[dim_idx] if dim_idx < len(labels) else f"Axis {dim_idx}"
                    y_axis_config = {'data_source': data_source, 'index': dim_idx, 'label': y_label}
            else:
                # Traditional x_axis/y_axis config
                x_axis_config = config.get('x_axis', {})
                y_axis_config = config.get('y_axis', {})
            
            # Resolve axis labels from variables if needed
            # Make copies to avoid modifying original configs
            x_axis_config = x_axis_config.copy() if x_axis_config else {}
            y_axis_config = y_axis_config.copy() if y_axis_config else {}
            
            resolved_x_label = self._resolve_axis_label(x_axis_config, outputs)
            if resolved_x_label:
                x_axis_config['label'] = resolved_x_label
            
            resolved_y_label = self._resolve_axis_label(y_axis_config, outputs)
            if resolved_y_label:
                y_axis_config['label'] = resolved_y_label
            
            # Also resolve z-axis label if present
            z_axis_config = config.get('z_axis', {})
            if z_axis_config:
                z_axis_config = z_axis_config.copy()
                resolved_z_label = self._resolve_axis_label(z_axis_config, outputs)
                if resolved_z_label:
                    z_axis_config['label'] = resolved_z_label
            
            # Merge axis indices for x (base + md + axis-specific)
            x_indices = base_indices.copy()
            x_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'x' in axis_indices:
                x_indices.update(axis_indices['x'])
            x_data = self._extract_axis_data(outputs, x_axis_config, x_indices)
            
            # Merge axis indices for y (base + md + axis-specific)
            y_indices = base_indices.copy()
            y_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'y' in axis_indices:
                y_indices.update(axis_indices['y'])
            y_data = self._extract_axis_data(outputs, y_axis_config, y_indices)
            
            # Merge axis indices for z (base + md + axis-specific)
            z_indices = base_indices.copy()
            z_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'z' in axis_indices:
                z_indices.update(axis_indices['z'])
            # For 4D+ multi-dimensional display, exclude dimensions that are being displayed
            # (md_active_dims) from z_indices AFTER all merges to avoid slicing them away
            if md_active_dims:
                for dim in md_active_dims:
                    z_indices.pop(dim, None)  # Remove if present
            z_data = self._extract_axis_data(outputs, config.get('z_axis', {}), z_indices)
            
            # Create container with navigation controls on top
            control_frame = ttk.Frame(parent)
            control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Store control frame reference for later updates (e.g., when combo changes)
            if 'graph_control_frames' not in self.analysis_data[instance_alias]:
                self.analysis_data[instance_alias]['graph_control_frames'] = {}
            self.analysis_data[instance_alias]['graph_control_frames'][section_id] = control_frame
            
            # Add navigation controls if axes are navigable
            nav_axes = config.get('data_slicing', [])
            if nav_axes:
                self._create_navigation_controls(control_frame, instance_alias, section_id, 
                                                 outputs, config, current_slice)
            
            # Create a copy of config with resolved axis labels for rendering
            render_config = config.copy()
            render_config['x_axis'] = x_axis_config
            render_config['y_axis'] = y_axis_config
            if z_axis_config:
                render_config['z_axis'] = z_axis_config
            
            # Render graph using graph_renderer module
            fig, ax = graph_renderer.render_graph_figure(
                graph_type, render_config, x_data, y_data, z_data, x_axis_config, y_axis_config,
                default_cmap=self.settings_manager.get('colormap', 'viridis')
            )
            
            # Embed figure in tkinter within a managed frame
            canvas, canvas_frame = graph_renderer.embed_figure_in_tkinter(fig, parent)
            
            # Store canvas reference for updates
            if 'graph_canvases' not in self.analysis_data[instance_alias]:
                self.analysis_data[instance_alias]['graph_canvases'] = {}
            self.analysis_data[instance_alias]['graph_canvases'][section_id] = (canvas, canvas_frame)
            
        except Exception as e:
            label = ttk.Label(parent, text=f"Error rendering graph: {str(e)}", foreground="red")
            label.pack(expand=True)
    
    def _extract_axis_data(self, outputs: dict, axis_config: dict, indices: dict = None) -> Optional[np.ndarray]:
        """Extract data for an axis from execution outputs.
        
        Args:
            outputs: Dictionary of output data
            axis_config: Config for this axis (including data_source)
            indices: Dictionary mapping dimension to index for slicing (e.g., {0: 5, 1: 2})
                    or an integer for backward compatibility
        """
        if not axis_config:
            return None
        
        data_source = axis_config.get('data_source')
        if not data_source:
            return None
            
        # Check if data source exists
        if data_source not in outputs:
            return None
            
        data = outputs[data_source]
        
        # Handle None values
        if data is None:
            return None
        
        # Track if this is a list source (coordinates/axes) vs array source (data)
        is_list_source = isinstance(data, list) and len(data) > 0
        
        # Handle list data (list of arrays like axis vectors)
        if is_list_source:
            # Get index from config
            list_index = axis_config.get('index', 0)
            if list_index < len(data):
                data = data[list_index] if isinstance(data[list_index], np.ndarray) else np.array(data[list_index])
            else:
                data = data[0] if isinstance(data[0], np.ndarray) else np.array(data[0])
        
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except (ValueError, TypeError):
                return None
        
        # Handle indexing for multi-dimensional array data (not list sources)
        # Apply config index if specified
        if not is_list_source:
            config_index = axis_config.get('index')
            if config_index is not None:
                if isinstance(config_index, int) and data.ndim > 1:
                    try:
                        data = data[config_index] if config_index < data.shape[0] else data
                    except (IndexError, TypeError):
                        pass
                elif isinstance(config_index, list) and data.ndim > 1:
                    try:
                        for idx in config_index:
                            data = data[idx]
                    except (IndexError, TypeError):
                        pass
        
        # Apply dimension-based slicing (only for array data, not list sources like axes)
        # List sources are coordinate arrays and should not be sliced by data dimensions
        if not is_list_source and isinstance(indices, dict) and indices:
            # Build proper indexing tuple for multi-dimensional slicing
            # Create a list of slice objects for each dimension
            index_list = []
            for dim in range(data.ndim):
                if dim in indices:
                    # Clip index to valid range for this dimension
                    idx = indices[dim]
                    max_idx = data.shape[dim] - 1
                    if idx > max_idx:
                        idx = max_idx
                    elif idx < 0:
                        idx = 0
                    index_list.append(idx)
                else:
                    index_list.append(slice(None))
            # Apply the indexing
            try:
                data = data[tuple(index_list)]
            except (IndexError, TypeError):
                pass
        elif not is_list_source and isinstance(indices, int) and data.ndim > 1:
            # Backward compatibility: indices is a single integer
            try:
                if indices >= 0 and indices < data.shape[0]:
                    data = data[indices]
            except (IndexError, TypeError):
                pass
        
        return data
    
    def _extract_sliced_data(self, data: np.ndarray, indices: dict) -> np.ndarray:
        """Extract sliced data from multi-dimensional array using indices dictionary.
        
        Args:
            data: NumPy array to slice
            indices: Dictionary mapping dimension to index (e.g., {0: 5, 1: 2})
        
        Returns:
            Sliced data array
        """
        if not isinstance(data, np.ndarray):
            return data
        
        result = data.copy()
        
        # Apply slicing for each dimension in REVERSE sorted order
        # This way we slice from higher dimensions first, so dimension indices don't shift
        if isinstance(indices, dict):
            for dim in sorted(indices.keys(), reverse=True):
                idx = indices[dim]
                try:
                    if dim < len(result.shape) and idx >= 0 and idx < result.shape[dim]:
                        # Use take with axis to properly handle multi-dimensional slicing
                        result = np.take(result, idx, axis=dim)
                except (IndexError, TypeError):
                    pass
        
        return result
    
    def _compute_dimension_combinations(self, data_shape: tuple, specified_dims: set, 
                                       ndim: int) -> List[Tuple[int, ...]]:
        """Compute all combinations of remaining dimensions for multi-dimensional slicing.
        
        For 4D+ data, computes all possible combinations of dimensions not specified in
        the "dimension" field of data_slicing config.
        
        Args:
            data_shape: Shape of the data array
            specified_dims: Set of dimensions already specified in data_slicing
            ndim: Target dimensionality for combinations (ndim-1 or ndim-2)
        
        Returns:
            List of tuples, each representing a combination of dimension indices
        """
        from itertools import combinations
        
        # Get all dimensions
        all_dims = set(range(len(data_shape)))
        
        # Get remaining (unspecified) dimensions
        remaining_dims = sorted(all_dims - specified_dims)
        
        # If no remaining dimensions, return empty list
        if not remaining_dims:
            return []
        
        # Compute combinations of size ndim
        if ndim <= 0 or ndim > len(remaining_dims):
            return []
        
        combos = list(combinations(remaining_dims, ndim))
        return combos
    
    def _create_table_navigation_controls(self, parent_frame: ttk.Frame, instance_alias: str,
                                         section_id: int, outputs: dict, config: dict,
                                         slice_state: dict) -> None:
        """Create navigation controls for table data slicing.
        
        Similar to graph navigation but for table dimensions.
        """
        try:
            nav_axes = config.get('data_slicing', [])
            if not nav_axes:
                return
            
            # Get the data to determine shape/bounds
            data_source = config.get('data_source')
            if not data_source or data_source not in outputs:
                return
            
            data = outputs[data_source]
            if not isinstance(data, np.ndarray):
                try:
                    data = np.array(data)
                except (ValueError, TypeError):
                    return
            
            # Initialize indices if not present
            if 'indices' not in slice_state:
                slice_state['indices'] = {}
                for nav_item in nav_axes:
                    if isinstance(nav_item, dict):
                        dim = nav_item.get('dimension', 0)
                        slice_state['indices'][dim] = nav_item.get('default', 0)
            
            # Create navigation frame
            nav_frame = ttk.Frame(parent_frame)
            nav_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # For each navigable axis, create controls
            for nav_item in nav_axes:
                # Parse navigation item
                if isinstance(nav_item, dict):
                    axis_name = nav_item.get('name', 'Dimension')
                    dimension = nav_item.get('dimension', 0)
                    show_nav = nav_item.get('show_navigation_menu', True)
                else:
                    # Old format: just a string
                    axis_name = nav_item
                    dimension = nav_axes.index(nav_item)
                    show_nav = True
                
                # Skip if navigation is disabled
                if not show_nav:
                    continue
                
                axis_frame = ttk.Frame(nav_frame)
                axis_frame.pack(fill=tk.X, padx=5, pady=2)
                
                # Get max index from data shape
                max_index = data.shape[dimension] - 1 if dimension < len(data.shape) else 0
                
                # Get current index
                indices = slice_state.get('indices', {})
                current_index = indices.get(dimension, 0)
                
                # Axis label
                label_text = f"{axis_name}: {current_index + 1}/{max_index + 1}"
                label = ttk.Label(axis_frame, text=label_text, width=15)
                label.pack(side=tk.LEFT, padx=5)
                
                # Previous button
                prev_btn = ttk.Button(
                    axis_frame,
                    text="<",
                    width=3,
                    command=lambda d=dimension, an=axis_name: self._on_table_navigate_slice(
                        instance_alias, section_id, -1, d, an, max_index
                    )
                )
                prev_btn.pack(side=tk.LEFT, padx=2)
                
                # Index display
                index_label = ttk.Label(axis_frame, text=str(current_index + 1), width=3)
                index_label.pack(side=tk.LEFT, padx=2)
                
                # Next button
                next_btn = ttk.Button(
                    axis_frame,
                    text=">",
                    width=3,
                    command=lambda d=dimension, an=axis_name: self._on_table_navigate_slice(
                        instance_alias, section_id, 1, d, an, max_index
                    )
                )
                next_btn.pack(side=tk.LEFT, padx=2)
                
                # Store reference for updates
                if not hasattr(self, '_table_nav_labels'):
                    self._table_nav_labels = {}
                label_key = (instance_alias, section_id, dimension, axis_name)
                self._table_nav_labels[label_key] = (index_label, label)
        
        except Exception as e:
            print(f"Error creating table navigation controls: {str(e)}")
    
    def _on_table_navigate_slice(self, instance_alias: str, section_id: int, direction: int,
                                 dimension: int, axis_name: str, max_index: int) -> None:
        """Handle table navigation button click to change slice index."""
        try:
            if instance_alias not in self.analysis_data:
                return
            
            if 'table_slices' not in self.analysis_data[instance_alias]:
                return
            
            slice_state = self.analysis_data[instance_alias]['table_slices']
            if section_id not in slice_state:
                return
            
            current_state = slice_state[section_id]
            indices = current_state.get('indices', {})
            
            # Update index
            current_idx = indices.get(dimension, 0)
            new_idx = current_idx + direction
            
            # Clamp to valid range
            new_idx = max(0, min(new_idx, max_index))
            
            # Store updated index
            indices[dimension] = new_idx
            
            # Update label if it exists
            if hasattr(self, '_table_nav_labels'):
                label_key = (instance_alias, section_id, dimension, axis_name)
                if label_key in self._table_nav_labels:
                    index_label, full_label = self._table_nav_labels[label_key]
                    index_label.config(text=str(new_idx + 1))
                    full_label.config(text=f"{axis_name}: {new_idx + 1}/{max_index + 1}")
            
            # Refresh table display
            self._refresh_table(instance_alias, section_id)
        
        except Exception as e:
            print(f"Error navigating table slice: {str(e)}")
    
    def _render_table_section(self, parent: ttk.Frame, instance_alias: str, section_data: dict, section_idx: int = 0):
        """Render a comprehensive data table with sorting, filtering, and formatting."""
        try:
            config = section_data.get('config', {})
            
            # Get execution results
            if instance_alias not in self.analysis_data:
                label = ttk.Label(parent, text="No data available - Please run 'Run Model' or 'Run to here' first", foreground="gray")
                label.pack(expand=True)
                return
            
            execution_results = self.analysis_data[instance_alias].get('execution_results', {})
            if not execution_results:
                label = ttk.Label(parent, text="No data available - Please run 'Run Model' or 'Run to here' first", foreground="gray")
                label.pack(expand=True)
                return
            
            if execution_results.get('status') != 'success':
                label = ttk.Label(parent, text="Execution failed - Check model_log.txt for details", foreground="red")
                label.pack(expand=True)
                return
            
            outputs = execution_results.get('outputs', {})
            data_source = config.get('data_source')
            
            if not data_source or data_source not in outputs:
                label = ttk.Label(parent, text="Data source not found", foreground="red")
                label.pack(expand=True)
                return
            
            data = outputs[data_source]
            
            # Convert to numpy array if needed
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Initialize table slices state for data slicing support
            # Use data_source as stable section ID instead of id(section_data) which changes each call
            section_id = f"{instance_alias}_{data_source}_{section_idx}"
            if 'table_slices' not in self.analysis_data[instance_alias]:
                self.analysis_data[instance_alias]['table_slices'] = {}
            
            slice_state = self.analysis_data[instance_alias]['table_slices']
            if section_id not in slice_state:
                # Initialize slices from config or defaults
                nav_axes = config.get('data_slicing', [])
                
                # Build indices dict for multi-dimensional slicing
                indices = {}
                for nav_item in nav_axes:
                    if isinstance(nav_item, dict):
                        dim = nav_item.get('dimension', 0)
                        indices[dim] = nav_item.get('default', 0)
                    else:
                        # Old format: just a string
                        dim = len(indices)
                        indices[dim] = 0
                
                slice_state[section_id] = {
                    'indices': indices,  # Dict: {dimension: index}
                    'data_slicing': nav_axes,
                    'outputs': outputs,
                    'config': config
                }
            
            current_slice = slice_state[section_id]
            
            # Extract sliced data if data_slicing is configured
            nav_axes = config.get('data_slicing', [])
            if nav_axes:
                indices = current_slice.get('indices', {})
                data = self._extract_sliced_data(data, indices)
                # Update slice info in config for display
                current_index = list(indices.values())[0] if indices else 0
                config['slice_info'] = {
                    'description': list(nav_axes)[0].get('name', '') if isinstance(list(nav_axes)[0], dict) else list(nav_axes)[0],
                    'index': current_index
                }
            else:
                config['slice_info'] = {}
            
            # NOW check for 3D+ data without data_slicing configuration (AFTER extraction)
            if data.ndim > 2:
                error_msg = f"Cannot display {data.ndim}D data ({data.shape}) in table without data slicing configuration.\n\n"
                error_msg += "Add 'data_slicing' to your table config:\n"
                error_msg += '{\n  "data_slicing": [\n'
                error_msg += '    {\n      "name": "Dimension",\n'
                error_msg += '      "dimension": 0,\n      "show_navigation_menu": true\n    }\n  ]\n}'
                label = ttk.Label(parent, text=error_msg, foreground="red", justify=tk.LEFT)
                label.pack(expand=True, padx=10, pady=10)
                return
            
            # Get table configuration
            title = config.get('title', 'Table')  # Section title as fallback
            table_title = config.get('table_title')  # Optional table title displayed on table
            decimal_places = config.get('decimal_places', 4)
            max_rows = config.get('max_rows', 50)
            max_cols = config.get('max_cols', 15)
            col_headers = config.get('column_headers', None)
            row_headers = config.get('row_headers', None)
            
            # Initialize table state if needed
            # Note: section_id is already defined above as stable identifier
            if 'table_state' not in self.analysis_data[instance_alias]:
                self.analysis_data[instance_alias]['table_state'] = {}
            
            table_state = self.analysis_data[instance_alias]['table_state']
            if section_id not in table_state:
                table_state[section_id] = {
                    'sort_column': None,
                    'sort_order': 'ascending',
                    'filter_text': '',
                    'current_slice': 0
                }
            
            # Create main container
            main_frame = ttk.Frame(parent)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Add title only if table_title is provided
            if table_title:
                title_label = ttk.Label(main_frame, text=table_title, font=('Arial', 10, 'bold'))
                title_label.pack(anchor='w', pady=(0, 5))
            
            # Add info bar (shape, stats)
            info_text = f"Shape: {data.shape} | Type: {data.dtype} | Min: {np.min(data):.4f} | Max: {np.max(data):.4f} | Mean: {np.mean(data):.4f}"
            info_label = ttk.Label(main_frame, text=info_text, font=('Arial', 8), foreground='gray')
            info_label.pack(anchor='w', pady=(0, 5))
            
            # Add navigation controls if data_slicing is configured
            nav_axes = config.get('data_slicing', [])
            if nav_axes:
                self._create_table_navigation_controls(main_frame, instance_alias, section_id,
                                                      outputs, config, current_slice)
            
            # Create toolbar for table controls
            toolbar = ttk.Frame(main_frame)
            toolbar.pack(fill=tk.X, pady=(0, 5))
            
            # Export button - use table_title > section title > data_source
            export_title = table_title or title or data_source
            export_btn = ttk.Button(toolbar, text='Export to CSV', 
                                   command=lambda: self._export_table_to_csv(data, export_title))
            export_btn.pack(side=tk.LEFT, padx=2)
            
            # Statistics button - use table_title > section title > data_source
            stats_title = table_title or title or data_source
            stats_btn = ttk.Button(toolbar, text='Show Statistics',
                                  command=lambda: self._show_table_statistics(data, stats_title))
            stats_btn.pack(side=tk.LEFT, padx=2)
            
            # Refresh button
            refresh_btn = ttk.Button(toolbar, text='Refresh',
                                    command=lambda: self._refresh_table(instance_alias, section_id))
            refresh_btn.pack(side=tk.LEFT, padx=2)
            
            # Create table view
            self._create_table_view(main_frame, data, config, decimal_places, 
                                   max_rows, max_cols, col_headers, row_headers)
            
        except Exception as e:
            import traceback
            label = ttk.Label(parent, text=f"Error rendering table: {str(e)}", foreground="red")
            label.pack(expand=True)
            traceback.print_exc()
    
    def _create_table_view(self, parent: ttk.Frame, data: np.ndarray, config: dict,
                          decimal_places: int, max_rows: int, max_cols: int,
                          col_headers: list = None, row_headers: list = None) -> None:
        """Create the actual table view with scrollbars and formatting."""
        try:
            # Check if data is still 3D+ (shouldn't happen if slicing is configured, but safety check)
            if data.ndim > 2:
                error_msg = f"Cannot display {data.ndim}D data in table view.\n"
                error_msg += f"Data shape: {data.shape}\n\n"
                error_msg += "This should have been caught earlier. Ensure data_slicing is configured."
                label = ttk.Label(parent, text=error_msg, foreground="red", justify=tk.LEFT)
                label.pack(expand=True, padx=10, pady=10)
                return
            
            # Prepare data for display
            if data.ndim == 1:
                display_data = data.reshape(-1, 1)
            elif data.ndim == 2:
                display_data = data
            else:
                # Should not reach here due to check above
                display_data = data.reshape(data.shape[0], -1)
            
            num_rows, num_cols = display_data.shape
            num_rows = min(num_rows, max_rows)
            num_cols = min(num_cols, max_cols)
            
            # Create frame for table
            tree_frame = ttk.Frame(parent)
            tree_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create columns
            if col_headers is not None:
                columns = tuple(str(h) for h in col_headers[:num_cols])
            else:
                columns = tuple(f'Col{i}' for i in range(num_cols))
            
            # Create treeview
            tree = ttk.Treeview(tree_frame, columns=columns)
            
            # Configure row header column
            tree.column('#0', width=50, anchor='center')
            tree.heading('#0', text='Row')
            
            # Configure data columns
            col_width = max(50, min(150, 800 // num_cols))
            for col_idx, col_name in enumerate(columns):
                tree.column(col_name, width=col_width, anchor='center')
                tree.heading(col_name, text=col_name)
            
            # Insert data rows
            for row_idx in range(num_rows):
                if row_headers is not None and row_idx < len(row_headers):
                    row_label = str(row_headers[row_idx])
                else:
                    row_label = str(row_idx)
                
                # Format values
                values = []
                for col_idx in range(num_cols):
                    val = display_data[row_idx, col_idx]
                    if isinstance(val, float):
                        formatted = f"{val:.{decimal_places}f}"
                    else:
                        formatted = str(val)
                    values.append(formatted)
                
                tree.insert('', 'end', text=row_label, values=tuple(values))
            
            # Add scrollbars
            scroll_x = ttk.Scrollbar(tree_frame, orient='horizontal', command=tree.xview)
            scroll_y = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
            tree.configure(xscroll=scroll_x.set, yscroll=scroll_y.set)
            
            tree.grid(row=0, column=0, sticky='nsew')
            scroll_x.grid(row=1, column=0, sticky='ew')
            scroll_y.grid(row=0, column=1, sticky='ns')
            
            tree_frame.grid_rowconfigure(0, weight=1)
            tree_frame.grid_columnconfigure(0, weight=1)
            
        except Exception as e:
            label = ttk.Label(parent, text=f"Error creating table view: {str(e)}", foreground="red")
            label.pack(expand=True)
    
    def _export_table_to_csv(self, data: np.ndarray, title: str = 'export') -> None:
        """Export table data to CSV file with file save dialog."""
        try:
            from tkinter import filedialog
            import csv
            from datetime import datetime
            
            # Create default filename
            default_filename = f"{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Open file save dialog
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile=default_filename,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Table to CSV"
            )
            
            # User cancelled the dialog
            if not filepath:
                return
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write title
                writer.writerow([title])
                writer.writerow([])
                
                # Write data
                if data.ndim == 1:
                    writer.writerow(['Index', 'Value'])
                    for i, val in enumerate(data):
                        writer.writerow([i, val])
                elif data.ndim == 2:
                    writer.writerow([f'Col{i}' for i in range(data.shape[1])])
                    for row in data:
                        writer.writerow(row)
                else:
                    writer.writerow([f'Flattened: {data.shape}'])
                    flat_data = data.reshape(data.shape[0], -1)
                    for row in flat_data:
                        writer.writerow(row)
            
            messagebox.showinfo("Success", f"✅ Table exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"❌ Error exporting table: {str(e)}")
    
    def _show_table_statistics(self, data: np.ndarray, title: str = 'Statistics') -> None:
        """Display statistical summary of table data."""
        try:
            stats_window = tk.Toplevel(self.root)
            stats_window.title(f"{title} - Statistics")
            stats_window.geometry("500x400")
            
            # Calculate statistics
            stats_text = f"""
DATA STATISTICS
===============

Shape: {data.shape}
Data Type: {data.dtype}

Basic Statistics:
  Min: {np.min(data):.6f}
  Max: {np.max(data):.6f}
  Mean: {np.mean(data):.6f}
  Median: {np.median(data):.6f}
  Std Dev: {np.std(data):.6f}
  Variance: {np.var(data):.6f}

Quartiles:
  Q1 (25%): {np.percentile(data, 25):.6f}
  Q2 (50%): {np.percentile(data, 50):.6f}
  Q3 (75%): {np.percentile(data, 75):.6f}

Count:
  Total Elements: {data.size}
  Non-zero Elements: {np.count_nonzero(data)}
  Zero Elements: {np.sum(data == 0)}
  NaN Elements: {np.isnan(data).sum()}
  Inf Elements: {np.isinf(data).sum()}
            """
            
            text_widget = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.insert('1.0', stats_text)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"Error showing statistics: {str(e)}")
    
    def _refresh_table(self, instance_alias: str, section_id: int) -> None:
        """Refresh the table display with current slicing."""
        try:
            if instance_alias not in self.analysis_data:
                return
            
            if 'execution_results' not in self.analysis_data[instance_alias]:
                return
            
            # Get all analysis pages and sections to find and update the matching table
            analysis_info = self.analysis_data[instance_alias]
            pages = analysis_info.get('pages', [])
            current_page_idx = analysis_info.get('current_page', 0)
            
            if current_page_idx >= len(pages):
                return
            
            page_data = pages[current_page_idx]
            sections = page_data.get('sections', [])
            
            # Find the matching section and re-render it
            for idx, section in enumerate(sections):
                if id(section) == section_id:
                    # Found matching section - get its parent frame and clear it
                    # We need to find the actual frame widget for this section
                    # This is a bit tricky since we need to locate the frame by content
                    
                    # For now, trigger a full page re-render which is safer
                    # Get the current tab frame
                    if hasattr(self, 'tab_content_frame'):
                        # Clear the tab content
                        for widget in self.tab_content_frame.winfo_children():
                            widget.destroy()
                        
                        # Re-render the current page
                        self._render_analysis_page(self.tab_content_frame, instance_alias, page_data)
                    
                    print(f"✅ Table refreshed")
                    return
            
        except Exception as e:
            print(f"Error refreshing table: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _show_section_popup(self, instance_alias: str, section_idx: int, section_data: dict):
        """Show a popup window with the same content as a section."""
        try:
            # Create a new popup window
            popup = tk.Toplevel(self.root)
            popup.title(f"Section: {section_data.get('config', {}).get('title', 'Section')}")
            popup.geometry("900x700")
            
            # Add button frame at top for action buttons
            button_frame = ttk.Frame(popup)
            button_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
            
            # Add "Save as Image" button if this is a graph section
            if section_data.get('type') == 'graph':
                save_btn = ttk.Button(
                    button_frame,
                    text="💾 Save as Image",
                    command=lambda: self._save_section_graph_as_image(instance_alias, section_idx)
                )
                save_btn.pack(side=tk.LEFT, padx=5)
            
            # Add close button on the right
            close_btn = ttk.Button(button_frame, text="Close", command=popup.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)
            
            # Create a frame for the content
            content_frame = ttk.Frame(popup)
            content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Render the section in the popup (with is_popup=True to skip popup button)
            self._render_section(content_frame, instance_alias, section_data, section_idx, is_popup=True)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open section popup: {str(e)}")
    
    def _save_section_graph_as_image(self, instance_alias: str, section_idx: int):
        """Save a graph section as an image file (PNG, JPEG, or TIFF)."""
        try:
            # Get the canvas and figure from stored references
            if instance_alias not in self.analysis_data:
                messagebox.showerror("Error", "Analysis data not found")
                return
            
            graph_canvases = self.analysis_data[instance_alias].get('graph_canvases', {})
            if section_idx not in graph_canvases:
                messagebox.showerror("Error", "Graph not found for this section")
                return
            
            canvas, canvas_frame = graph_canvases[section_idx]
            fig = canvas.figure
            
            # Open file save dialog
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG Image", "*.png"),
                    ("JPEG Image", "*.jpg"),
                    ("TIFF Image", "*.tiff"),
                    ("All Files", "*.*")
                ]
            )
            
            if file_path:
                # Save the figure
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Graph saved successfully to:\n{file_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save graph: {str(e)}")
    
    def _create_section_popup_button(self, parent: ttk.Frame, instance_alias: str, section_idx: int, section_data: dict):
        """Create a small floating button in the upper right corner of a section (hovering over content)."""
        try:
            # Defer button creation to allow content to render first
            def create_button():
                # Create a small popup button using ttk.Button for consistent styling
                popup_btn = ttk.Button(
                    parent, 
                    text="🗗",  # Icon to indicate opening in a new window
                    width=2,
                    command=lambda: self._show_section_popup(instance_alias, section_idx, section_data)
                )
                # Position in upper right corner using place()
                popup_btn.place(relx=1.0, rely=0.0, anchor="ne", x=5, y=-10)
                popup_btn.lift()
            
            # Schedule button creation for after all packing/rendering is done
            parent.after(100, create_button)
        
        except Exception as e:
            print(f"Error creating section popup button: {str(e)}")
    
    def _create_navigation_controls(self, parent_frame: ttk.Frame, instance_alias: str, 
                                   section_id: int, outputs: dict, config: dict, 
                                   slice_state: dict) -> None:
        """Create navigation controls (arrow buttons) for multi-dimensional data slicing and axis selection.
        
        Supports both slicing and axis selection:
        Slicing: "data_slicing": [{"name": "Samples", "dimension": 0}]
        Axis selection: "data_slicing": [{"name": "X-Axis", "dimension": 1, "axis": "x"}, {"name": "Y-Axis", "dimension": 1, "axis": "y"}]
        """
        try:
            nav_axes = config.get('data_slicing', [])
            if not nav_axes:
                return
            
            # Get the data to determine shape/bounds
            # For aux_axis configs, use z_axis as it points to actual data (e.g., X_cal)
            # aux_axis points to axis vectors (e.g., axis_n_info) which is a list
            # For 3D graph types (heatmap, contour, 3d_surf), use z_axis as it contains the actual data
            data_source = None
            graph_type = config.get('graph_type', '')
            if 'aux_axis' in config:
                # Use z_axis to get the actual data array
                data_source = config.get('z_axis', {}).get('data_source')
            elif graph_type in ('heatmap', 'contour', '3d_surf'):
                # For 3D visualization types, z_axis contains the actual multi-dimensional data
                data_source = config.get('z_axis', {}).get('data_source')
            else:
                # Traditional configs - use y_axis or x_axis
                axis_config = config.get('y_axis', {}) or config.get('x_axis', {})
                data_source = axis_config.get('data_source')
            
            if not data_source or data_source not in outputs:
                return
            
            data = outputs[data_source]
            if not isinstance(data, np.ndarray):
                try:
                    data = np.array(data)
                except (ValueError, TypeError):
                    return
            
            # Initialize axis indices if not present (each axis has its own indices dict)
            if 'axis_indices' not in slice_state:
                slice_state['axis_indices'] = {}
                # Only initialize axis_indices for axes that appear in navigation_axes with "axis" field
                for nav_item in nav_axes:
                    if isinstance(nav_item, dict):
                        target_axis = nav_item.get('axis')
                        if target_axis:  # Only for axis selection items
                            # Only set axis index if dimension is explicitly specified
                            if 'dimension' in nav_item:
                                axis_indices_dict = {}
                                dimension = nav_item['dimension']
                                # Get default from nav_item or axis config
                                default_val = nav_item.get('default', None)
                                if default_val is None:
                                    default_val = config.get(f'{target_axis}_axis', {}).get('default_column', 0)
                                # Validate default is in bounds
                                max_idx = data.shape[dimension] - 1 if dimension < len(data.shape) else 0
                                if default_val < 0 or default_val > max_idx:
                                    default_val = 0
                                axis_indices_dict[dimension] = default_val
                                slice_state['axis_indices'][target_axis] = axis_indices_dict
            
            # Initialize shared dimension slicing indices if not present
            if 'indices' not in slice_state:
                slice_state['indices'] = {}
                # Set default indices based on navigation_axes config for non-axis slicing
                for nav_item in nav_axes:
                    if isinstance(nav_item, dict):
                        # Only process if dimension is explicitly specified
                        if 'dimension' in nav_item:
                            dimension = nav_item['dimension']
                            target_axis = nav_item.get('axis')
                            # Only use default for non-axis items (slicing mode)
                            if not target_axis:
                                default_idx = nav_item.get('default', 0)
                                # Validate default is in bounds
                                max_idx = data.shape[dimension] - 1 if dimension < len(data.shape) else 0
                                if default_idx < 0 or default_idx > max_idx:
                                    default_idx = 0
                                slice_state['indices'][dimension] = default_idx
            
            # ===== Multi-Dimensional Slicing for 4D+ Data =====
            # For 4D+ data, always compute combinations and initialize state
            # show_md_menu only controls UI visibility
            show_md_menu = config.get('show_md_menu', False)
            md_combinations = []
            
            if data.ndim >= 4:
                # Collect dimensions already specified in data_slicing
                specified_dims = set()
                for nav_item in nav_axes:
                    if isinstance(nav_item, dict):
                        dim = nav_item.get('dimension')
                        if dim is not None:
                            specified_dims.add(dim)
                
                # Determine combination size: always 2 for heatmap (x and y axes)
                # The remaining dimensions (not in combination, not specified) will be navigable
                combo_size = 2
                
                # Compute all possible combinations of remaining dimensions
                md_combinations = self._compute_dimension_combinations(data.shape, specified_dims, combo_size)
                
                if md_combinations:
                    # Initialize multi-dimensional state if needed
                    if 'md_combo_index' not in slice_state:
                        # Get default combination index from config, default to 0
                        md_defaults = config.get('md_default', {})
                        slice_state['md_combo_index'] = md_defaults.get('combo_index', 0)
                    
                    if 'md_slice_indices' not in slice_state:
                        # Initialize slice indices for dimensions NOT in the current combination
                        # These are the dimensions we navigate through (not displayed on x/y axes)
                        slice_state['md_slice_indices'] = {}
                        md_defaults = config.get('md_default', {})
                        current_combo_idx = slice_state.get('md_combo_index', 0)
                        if current_combo_idx < len(md_combinations):
                            current_combo = md_combinations[current_combo_idx]
                            # Find all dimensions that are NOT in the combination and NOT specified
                            all_dims = set(range(len(data.shape)))
                            navigable_dims = all_dims - set(current_combo) - specified_dims
                            
                            for dim in navigable_dims:
                                # Get default from md_default config or use 0
                                default_val = md_defaults.get(f'dim_{dim}', 0)
                                max_idx = data.shape[dim] - 1 if dim < len(data.shape) else 0
                                if default_val < 0 or default_val > max_idx:
                                    default_val = 0
                                slice_state['md_slice_indices'][dim] = default_val
                    
                    # Create UI for multi-dimensional slicing ONLY if show_md_menu is True
                    if show_md_menu:
                        md_frame = ttk.LabelFrame(parent_frame, text="Multi-Dimensional Slicing (4D+)", padding=5)
                        md_frame.pack(fill=tk.X, padx=5, pady=5)
                        
                        # Combination selector
                        combo_select_frame = ttk.Frame(md_frame)
                        combo_select_frame.pack(fill=tk.X, padx=5, pady=2)
                        
                        ttk.Label(combo_select_frame, text="Dimension Combination:", width=20).pack(side=tk.LEFT, padx=5)
                        
                        # Get dimension labels from config
                        dim_labels = self._get_dimension_labels(outputs, config)
                        
                        # Create combo box with dimension combinations (use labels if available)
                        combo_options = []
                        for combo in md_combinations:
                            # Use dimension labels if available, otherwise use dimension index
                            combo_parts = []
                            for d in combo:
                                if d in dim_labels:
                                    combo_parts.append(dim_labels[d])
                                else:
                                    combo_parts.append(str(d))
                            combo_str = f"Dims: {', '.join(combo_parts)}"
                            combo_options.append(combo_str)
                        
                        current_combo_idx = slice_state.get('md_combo_index', 0)
                        
                        combo_dropdown = ttk.Combobox(combo_select_frame, 
                                                     values=combo_options, state='readonly', width=30)
                        combo_dropdown.current(current_combo_idx if current_combo_idx < len(combo_options) else 0)
                        combo_dropdown.pack(side=tk.LEFT, padx=5)
                        combo_dropdown.bind('<<ComboboxSelected>>', 
                                           lambda e: self._on_md_combo_changed(instance_alias, section_id, 
                                                                               combo_dropdown.current(), md_combinations))
                        
                        # Create slicing controls for dimensions NOT in current combination
                        # These are the dimensions we navigate through (not displayed on x/y axes)
                        if current_combo_idx < len(md_combinations):
                            current_combo = md_combinations[current_combo_idx]
                            # Find all dimensions that are NOT in the combination and NOT specified
                            all_dims = set(range(len(data.shape)))
                            navigable_dims = sorted(all_dims - set(current_combo) - specified_dims)
                            
                            for dim in navigable_dims:
                                dim_frame = ttk.Frame(md_frame)
                                dim_frame.pack(fill=tk.X, padx=5, pady=2)
                                
                                # Get max index for this dimension
                                max_index = data.shape[dim] - 1 if dim < len(data.shape) else 0
                                
                                # Get current index
                                current_index = slice_state.get('md_slice_indices', {}).get(dim, 0)
                                
                                # Dimension label - use dimension labels if available
                                dim_name = dim_labels.get(dim, f"Dimension {dim}")
                                label_text = f"{dim_name}: {current_index + 1}/{max_index + 1}"
                                label = ttk.Label(dim_frame, text=label_text, width=20)
                                label.pack(side=tk.LEFT, padx=5)
                                
                                # Previous button - capture max_index by value with m=max_index
                                prev_btn = ttk.Button(
                                    dim_frame,
                                    text="<",
                                    width=3,
                                    command=lambda d=dim, m=max_index: self._on_md_navigate(
                                        instance_alias, section_id, -1, d, m
                                    )
                                )
                                prev_btn.pack(side=tk.LEFT, padx=2)
                                
                                # Index display
                                index_label = ttk.Label(dim_frame, text=str(current_index + 1), width=3)
                                index_label.pack(side=tk.LEFT, padx=2)
                                
                                # Next button - capture max_index by value with m=max_index
                                next_btn = ttk.Button(
                                    dim_frame,
                                    text=">",
                                    width=3,
                                    command=lambda d=dim, m=max_index: self._on_md_navigate(
                                        instance_alias, section_id, 1, d, m
                                    )
                                )
                                next_btn.pack(side=tk.LEFT, padx=2)
                                
                                # Variable labels - show current value label between buttons if configured
                                var_labels_config = config.get('variable_labels')
                                if var_labels_config:
                                    var_label_text = self._get_variable_label(outputs, var_labels_config, dim, current_index)
                                    if var_label_text:
                                        var_label = ttk.Label(dim_frame, text=f"[{var_label_text}]", foreground="gray")
                                        var_label.pack(side=tk.LEFT, padx=5)
                                        # Store reference for updates
                                        if not hasattr(self, '_var_labels'):
                                            self._var_labels = {}
                                        var_label_key = (instance_alias, section_id, dim, 'md')
                                        self._var_labels[var_label_key] = (var_label, var_labels_config, dim)
                                
                                # Store reference for updates
                                if not hasattr(self, '_md_nav_labels'):
                                    self._md_nav_labels = {}
                                label_key = (instance_alias, section_id, dim)
                                self._md_nav_labels[label_key] = (index_label, label, dim_labels.get(dim))
            
            # For each navigable axis, create controls if enabled for that item
            for nav_item in nav_axes:
                # Parse navigation item - support both old and new formats
                if isinstance(nav_item, dict):
                    # New format: {"name": "Samples", "dimension": 0} or {"name": "X-Axis", "dimension": 1, "axis": "x"}
                    axis_name = nav_item.get('name', 'Axis')
                    dimension = nav_item.get('dimension', 0)
                    target_axis = nav_item.get('axis')  # 'x', 'y', 'z', or None for slicing
                    # Check if this item should show navigation (default to False - must be explicitly enabled)
                    show_nav = nav_item.get('show_navigation_menu', False)
                else:
                    # Old format: just a string "Samples"
                    axis_name = nav_item
                    dimension = nav_axes.index(nav_item)  # Position in the list
                    target_axis = None
                    show_nav = True  # Show by default for old format
                
                # Skip this item if navigation menu is disabled
                if not show_nav:
                    continue
                
                axis_frame = ttk.Frame(parent_frame)
                axis_frame.pack(fill=tk.X, padx=5, pady=2)
                
                # Get max index from data shape
                max_index = data.shape[dimension] - 1 if dimension < len(data.shape) else 0
                
                # Determine current index based on whether this is axis selection or slicing
                if target_axis:
                    # Axis selection mode - get from that axis's indices dict
                    axis_indices_dict = slice_state.get('axis_indices', {}).get(target_axis, {})
                    # Get default from nav_item or axis config
                    default_col = nav_item.get('default', None)
                    if default_col is None:
                        default_col = config.get(f'{target_axis}_axis', {}).get('default_column',
                                                 0 if target_axis == 'x' else 1 if target_axis == 'y' else 2)
                    # Validate default is in bounds
                    if default_col < 0 or default_col > max_index:
                        default_col = 0
                    # Use stored value for this dimension, or default
                    current_index = axis_indices_dict.get(dimension, default_col)
                else:
                    # Slicing mode - get from shared indices dict
                    indices = slice_state.get('indices', {})
                    # Get default from nav_item
                    default_idx = nav_item.get('default', 0) if isinstance(nav_item, dict) else 0
                    # Validate default is in bounds
                    if default_idx < 0 or default_idx > max_index:
                        default_idx = 0
                    # Use stored value for this dimension, or default
                    current_index = indices.get(dimension, default_idx)
                
                # Axis label - display 1-based for user (current_index + 1 and max_index + 1)
                label_text = f"{axis_name}: {current_index + 1}/{max_index + 1}"
                label = ttk.Label(axis_frame, text=label_text, width=15)
                label.pack(side=tk.LEFT, padx=5)
                
                # Previous button - capture max_index by value with m=max_index
                prev_btn = ttk.Button(
                    axis_frame,
                    text="<",
                    width=3,
                    command=lambda an=axis_name, d=dimension, ax=target_axis, m=max_index: self._on_navigate_slice(
                        instance_alias, section_id, -1, d, an, m, ax
                    )
                )
                prev_btn.pack(side=tk.LEFT, padx=2)
                
                # Index display - show 1-based for user
                index_label = ttk.Label(axis_frame, text=str(current_index + 1), width=3)
                index_label.pack(side=tk.LEFT, padx=2)
                
                # Next button - capture max_index by value with m=max_index
                next_btn = ttk.Button(
                    axis_frame,
                    text=">",
                    width=3,
                    command=lambda an=axis_name, d=dimension, ax=target_axis, m=max_index: self._on_navigate_slice(
                        instance_alias, section_id, 1, d, an, m, ax
                    )
                )
                next_btn.pack(side=tk.LEFT, padx=2)
                
                # Variable labels - show current value label after buttons if configured
                var_labels_config = config.get('variable_labels')
                if var_labels_config:
                    var_label_text = self._get_variable_label(outputs, var_labels_config, dimension, current_index)
                    if var_label_text:
                        var_label = ttk.Label(axis_frame, text=f"[{var_label_text}]", foreground="gray")
                        var_label.pack(side=tk.LEFT, padx=5)
                        # Store reference for updates
                        if not hasattr(self, '_var_labels'):
                            self._var_labels = {}
                        var_label_key = (instance_alias, section_id, dimension, axis_name, target_axis)
                        self._var_labels[var_label_key] = (var_label, var_labels_config, dimension)
                
                # Store reference to index label for updates
                if not hasattr(self, '_nav_labels'):
                    self._nav_labels = {}
                # Use a tuple that includes axis name and target_axis to ensure uniqueness
                label_key = (instance_alias, section_id, dimension, axis_name, target_axis)
                self._nav_labels[label_key] = (index_label, label)
        
        except Exception as e:
            print(f"Error creating navigation controls: {str(e)}")
    
    def _on_navigate_slice(self, instance_alias: str, section_id: int, direction: int,
                          dimension: int, axis_name: str, max_index: int, target_axis: str = None) -> None:
        """Handle navigation button click to change slice index or axis selection.
        
        Args:
            target_axis: 'x', 'y', 'z' for axis selection, or None for dimension slicing
        """
        try:
            if instance_alias not in self.analysis_data:
                return
            
            if 'graph_slices' not in self.analysis_data[instance_alias]:
                return
            
            slice_state = self.analysis_data[instance_alias]['graph_slices']
            if section_id not in slice_state:
                return
            
            current_state = slice_state[section_id]
            
            # Handle axis selection or dimension slicing
            if target_axis:
                # Axis selection mode - get from that axis's indices dict
                axis_indices_dict = current_state.get('axis_indices', {}).get(target_axis, {})
                current_index = axis_indices_dict.get(dimension, 0)
            else:
                # Dimension slicing mode
                indices = current_state.get('indices', {})
                current_index = indices.get(dimension, 0)
            
            # Calculate new index
            new_index = current_index + direction
            
            # Bounds checking
            if new_index < 0:
                new_index = 0
            elif new_index > max_index:
                new_index = max_index
            
            # Update state only if index changed
            if new_index != current_index:
                if target_axis:
                    # Update axis's indices dict for this dimension
                    if 'axis_indices' not in current_state:
                        current_state['axis_indices'] = {}
                    if target_axis not in current_state['axis_indices']:
                        current_state['axis_indices'][target_axis] = {}
                    current_state['axis_indices'][target_axis][dimension] = new_index
                else:
                    # Update dimension slicing indices dict
                    indices = current_state.get('indices', {})
                    indices[dimension] = new_index
                    current_state['indices'] = indices
                
                # Update label display if it exists - show 1-based for user
                if hasattr(self, '_nav_labels'):
                    # Use the same unique key as when storing
                    key = (instance_alias, section_id, dimension, axis_name, target_axis)
                    if key in self._nav_labels:
                        index_label, full_label = self._nav_labels[key]
                        index_label.config(text=str(new_index + 1))
                        full_label.config(text=f"{axis_name}: {new_index + 1}/{max_index + 1}")
                
                # Update variable label if it exists
                if hasattr(self, '_var_labels'):
                    var_key = (instance_alias, section_id, dimension, axis_name, target_axis)
                    if var_key in self._var_labels:
                        var_label, var_labels_config, dim = self._var_labels[var_key]
                        # Get outputs to resolve variable label
                        outputs = self.analysis_data[instance_alias].get('execution_results', {}).get('outputs', {})
                        var_label_text = self._get_variable_label(outputs, var_labels_config, dim, new_index)
                        if var_label_text:
                            var_label.config(text=f"[{var_label_text}]")
                        else:
                            var_label.config(text="")
                
                # Refresh the graph with new slice/axis
                self._update_graph_with_slice(instance_alias, section_id, dimension)
        
        except Exception as e:
            print(f"Error navigating slice: {str(e)}")
    
    def _on_md_combo_changed(self, instance_alias: str, section_id: int, 
                            combo_index: int, combinations: List[Tuple[int, ...]]) -> None:
        """Handle multi-dimensional combination dropdown change.
        
        Args:
            instance_alias: Function instance identifier
            section_id: Section identifier
            combo_index: Selected combination index
            combinations: List of all dimension combinations
        """
        try:
            if instance_alias not in self.analysis_data:
                return
            
            if 'graph_slices' not in self.analysis_data[instance_alias]:
                return
            
            slice_state = self.analysis_data[instance_alias]['graph_slices']
            if section_id not in slice_state:
                return
            
            current_state = slice_state[section_id]
            
            # Update the selected combination index
            current_state['md_combo_index'] = combo_index
            
            # Reset slice indices for dimensions NOT in the new combination
            # These are the navigable dimensions
            if combo_index < len(combinations):
                current_combo = combinations[combo_index]
                current_state['md_slice_indices'] = {}
                
                # Get data to validate bounds and find specified dimensions
                outputs = current_state.get('outputs', {})
                config = current_state.get('config', {})
                
                # Get data source - support both aux_axis and traditional configs
                data_source = None
                if 'aux_axis' in config:
                    data_source = config.get('z_axis', {}).get('data_source')
                else:
                    axis_config = config.get('y_axis', {}) or config.get('x_axis', {})
                    data_source = axis_config.get('data_source')
                
                if data_source and data_source in outputs:
                    data = outputs[data_source]
                    if isinstance(data, np.ndarray):
                        # Find specified dimensions
                        specified_dims = set()
                        nav_axes = config.get('data_slicing', [])
                        for nav_item in nav_axes:
                            if isinstance(nav_item, dict):
                                dim = nav_item.get('dimension')
                                if dim is not None:
                                    specified_dims.add(dim)
                        
                        # Find navigable dimensions (not in combo, not specified)
                        all_dims = set(range(len(data.shape)))
                        navigable_dims = all_dims - set(current_combo) - specified_dims
                        
                        # Clear old md_slice_indices and rebuild for new combination
                        current_state['md_slice_indices'] = {}
                        for dim in navigable_dims:
                            # Initialize to 0 for new combination
                            max_idx = data.shape[dim] - 1 if dim < len(data.shape) else 0
                            current_state['md_slice_indices'][dim] = 0
            
            # Rebuild navigation controls to reflect new navigable dimensions
            config = current_state.get('config', {})
            outputs = current_state.get('outputs', {})
            
            # Get stored control frame and rebuild controls
            if 'graph_control_frames' in self.analysis_data[instance_alias]:
                control_frame = self.analysis_data[instance_alias]['graph_control_frames'].get(section_id)
                if control_frame:
                    # Clear existing controls
                    for widget in control_frame.winfo_children():
                        widget.destroy()
                    
                    # Clear stored label references for this section
                    if hasattr(self, '_md_nav_labels'):
                        keys_to_remove = [k for k in self._md_nav_labels.keys() 
                                         if k[0] == instance_alias and k[1] == section_id]
                        for k in keys_to_remove:
                            del self._md_nav_labels[k]
                    
                    # Rebuild navigation controls
                    nav_axes = config.get('data_slicing', [])
                    if nav_axes:
                        self._create_navigation_controls(control_frame, instance_alias, section_id,
                                                        outputs, config, current_state)
            
            # Update the graph with the new slice
            self._update_graph_with_slice(instance_alias, section_id, 0)
            self._update_graph_with_slice(instance_alias, section_id, 0)
        
        except Exception as e:
            print(f"Error changing MD combination: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _on_md_navigate(self, instance_alias: str, section_id: int, direction: int,
                       dimension: int, max_index: int) -> None:
        """Handle multi-dimensional slice navigation button click.
        
        Args:
            instance_alias: Function instance identifier
            section_id: Section identifier
            direction: Navigation direction (-1 or 1)
            dimension: Dimension being sliced
            max_index: Maximum valid index for this dimension
        """
        try:
            if instance_alias not in self.analysis_data:
                return
            
            if 'graph_slices' not in self.analysis_data[instance_alias]:
                return
            
            slice_state = self.analysis_data[instance_alias]['graph_slices']
            if section_id not in slice_state:
                return
            
            current_state = slice_state[section_id]
            md_slice_indices = current_state.get('md_slice_indices', {})
            
            # Validate dimension is still navigable (controls might be stale after combo change)
            if dimension not in md_slice_indices:
                print(f"WARNING: Dimension {dimension} is not in current navigable dimensions {list(md_slice_indices.keys())}")
                return
            
            # Get current index
            current_index = md_slice_indices.get(dimension, 0)
            
            # Calculate new index
            new_index = current_index + direction
            
            # Bounds checking
            if new_index < 0:
                new_index = 0
            elif new_index > max_index:
                new_index = max_index
            
            # Update state only if index changed
            if new_index != current_index:
                md_slice_indices[dimension] = new_index
                current_state['md_slice_indices'] = md_slice_indices
                
                # Update label display if it exists
                if hasattr(self, '_md_nav_labels'):
                    key = (instance_alias, section_id, dimension)
                    if key in self._md_nav_labels:
                        stored = self._md_nav_labels[key]
                        index_label, full_label = stored[0], stored[1]
                        dim_name = stored[2] if len(stored) > 2 else None
                        index_label.config(text=str(new_index + 1))
                        if dim_name:
                            full_label.config(text=f"{dim_name}: {new_index + 1}/{max_index + 1}")
                        else:
                            full_label.config(text=f"Dimension {dimension}: {new_index + 1}/{max_index + 1}")
                
                # Update variable label if it exists
                if hasattr(self, '_var_labels'):
                    var_key = (instance_alias, section_id, dimension, 'md')
                    if var_key in self._var_labels:
                        var_label, var_labels_config, dim = self._var_labels[var_key]
                        # Get outputs to resolve variable label
                        outputs = self.analysis_data[instance_alias].get('execution_results', {}).get('outputs', {})
                        var_label_text = self._get_variable_label(outputs, var_labels_config, dim, new_index)
                        if var_label_text:
                            var_label.config(text=f"[{var_label_text}]")
                        else:
                            var_label.config(text="")
                
                # Refresh the graph with new multi-dimensional slice
                self._update_graph_with_slice(instance_alias, section_id, dimension)
        
        except Exception as e:
            print(f"Error navigating MD slice: {str(e)}")
    
    def _update_graph_with_slice(self, instance_alias: str, section_id: int, 
                                dimension: int) -> None:
        """Update the graph display with the new slice index for a specific dimension."""
        try:
            if instance_alias not in self.analysis_data:
                return
            
            if 'graph_slices' not in self.analysis_data[instance_alias]:
                return
            
            slice_state = self.analysis_data[instance_alias]['graph_slices']
            if section_id not in slice_state:
                return
            
            current_state = slice_state[section_id]
            outputs = current_state.get('outputs', {})
            config = current_state.get('config', {})
            graph_type = current_state.get('graph_type', 'scatter')
            
            # Extract data for each axis, merging per-axis indices with shared indices
            base_indices = current_state.get('indices', {})
            axis_indices = current_state.get('axis_indices', {})
            
            # Also merge multi-dimensional slice indices if present
            md_slice_indices = current_state.get('md_slice_indices', {})
            
            # For 4D+ data, get the active dimension combination
            md_active_dims = None
            if md_slice_indices:
                # Get the active combination from the computed combinations
                nav_axes_temp = config.get('data_slicing', [])
                specified_dims = set()
                for nav_item in nav_axes_temp:
                    if isinstance(nav_item, dict):
                        dim = nav_item.get('dimension')
                        if dim is not None:
                            specified_dims.add(dim)
                
                # Get data to compute combinations
                # For aux_axis configs, use z_axis as it points to actual data
                data_source_temp = None
                if 'aux_axis' in config:
                    data_source_temp = config.get('z_axis', {}).get('data_source')
                else:
                    axis_config_temp = config.get('y_axis', {}) or config.get('x_axis', {})
                    data_source_temp = axis_config_temp.get('data_source')
                
                if data_source_temp and data_source_temp in outputs:
                    data_temp = outputs[data_source_temp]
                    if isinstance(data_temp, np.ndarray):
                        # For heatmaps, we display 2 dimensions (x, y) and navigate through others
                        combo_size = 2
                        md_combinations = self._compute_dimension_combinations(data_temp.shape, specified_dims, combo_size)
                        
                        combo_idx = current_state.get('md_combo_index', 0)
                        if combo_idx < len(md_combinations):
                            md_active_dims = list(md_combinations[combo_idx])
            
            # For 4D+ data with aux_axis config, create axis configs automatically
            if 'aux_axis' in config and md_active_dims:
                # aux_axis contains data_source and labels for all dimensions
                aux_axis_config = config['aux_axis']
                data_source = aux_axis_config.get('data_source')
                labels_config = aux_axis_config.get('labels', [])
                
                # Resolve labels - can be array or variable name
                labels = []
                if isinstance(labels_config, str):
                    # labels is a variable name - look it up in outputs
                    if labels_config in outputs:
                        label_data = outputs[labels_config]
                        if isinstance(label_data, (list, np.ndarray)):
                            labels = [str(lbl) for lbl in label_data]
                elif isinstance(labels_config, list):
                    # labels is a direct array
                    labels = labels_config
                
                # x-axis uses first dimension in active combination
                x_axis_config = None
                if len(md_active_dims) > 0:
                    dim_idx = md_active_dims[0]
                    x_label = labels[dim_idx] if dim_idx < len(labels) else f"Axis {dim_idx}"
                    x_axis_config = {'data_source': data_source, 'index': dim_idx, 'label': x_label}
                
                # y-axis uses second dimension in active combination
                y_axis_config = None
                if len(md_active_dims) > 1:
                    dim_idx = md_active_dims[1]
                    y_label = labels[dim_idx] if dim_idx < len(labels) else f"Axis {dim_idx}"
                    y_axis_config = {'data_source': data_source, 'index': dim_idx, 'label': y_label}
            else:
                # Traditional x_axis/y_axis config
                x_axis_config = config.get('x_axis', {})
                y_axis_config = config.get('y_axis', {})
            
            # Resolve axis labels from variables if needed
            # Make copies to avoid modifying original configs
            x_axis_config = x_axis_config.copy() if x_axis_config else {}
            y_axis_config = y_axis_config.copy() if y_axis_config else {}
            
            resolved_x_label = self._resolve_axis_label(x_axis_config, outputs)
            if resolved_x_label:
                x_axis_config['label'] = resolved_x_label
            
            resolved_y_label = self._resolve_axis_label(y_axis_config, outputs)
            if resolved_y_label:
                y_axis_config['label'] = resolved_y_label
            
            # Also resolve z-axis label if present
            z_axis_config = config.get('z_axis', {})
            if z_axis_config:
                z_axis_config = z_axis_config.copy()
                resolved_z_label = self._resolve_axis_label(z_axis_config, outputs)
                if resolved_z_label:
                    z_axis_config['label'] = resolved_z_label
            
            # Merge axis indices for x (base + md + axis-specific)
            x_indices = base_indices.copy()
            x_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'x' in axis_indices:
                x_indices.update(axis_indices['x'])
            x_data = self._extract_axis_data(outputs, x_axis_config, x_indices)
            
            # Merge axis indices for y (base + md + axis-specific)
            y_indices = base_indices.copy()
            y_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'y' in axis_indices:
                y_indices.update(axis_indices['y'])
            y_data = self._extract_axis_data(outputs, y_axis_config, y_indices)
            
            # Merge axis indices for z (base + md + axis-specific)
            z_indices = base_indices.copy()
            z_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'z' in axis_indices:
                z_indices.update(axis_indices['z'])
            # For 4D+ multi-dimensional display, exclude dimensions that are being displayed
            # (md_active_dims) from z_indices AFTER all merges to avoid slicing them away
            if md_active_dims:
                for dim in md_active_dims:
                    z_indices.pop(dim, None)  # Remove if present
            z_data = self._extract_axis_data(outputs, config.get('z_axis', {}), z_indices)
            
            # Create a copy of config with resolved axis labels for rendering
            render_config = config.copy()
            render_config['x_axis'] = x_axis_config
            render_config['y_axis'] = y_axis_config
            if z_axis_config:
                render_config['z_axis'] = z_axis_config
            
            # Render graph using graph_renderer module
            fig, ax = graph_renderer.render_graph_figure(
                graph_type, render_config, x_data, y_data, z_data, x_axis_config, y_axis_config,
                default_cmap=self.settings_manager.get('colormap', 'viridis')
            )
            
            # Update existing canvas with new figure
            graph_renderer.update_embedded_figure(fig, instance_alias, section_id, 
                                                 self.analysis_data, None)
            
        except Exception as e:
            print(f"Error updating graph with slice: {str(e)}")
    
    def _show_remove_section_dialog(self, instance_alias: str):
        """Show dialog to remove a section."""
        if instance_alias not in self.analysis_data:
            return
        
        current_page = self.analysis_data[instance_alias]['current_page']
        pages = self.analysis_data[instance_alias].get('pages', [])
        
        if current_page >= len(pages):
            return
        
        page_data = pages[current_page]
        sections = page_data.get('sections', [])
        
        if not sections:
            messagebox.showinfo("Info", "No sections to remove")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Remove Section")
        dialog.geometry("350x250")
        dialog.resizable(False, False)
        
        label = ttk.Label(dialog, text="Select section to remove:", font=("Arial", 10))
        label.pack(padx=10, pady=10)
        
        # Section list frame
        listbox_frame = ttk.Frame(dialog)
        listbox_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        listbox = tk.Listbox(listbox_frame, height=6)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=scrollbar.set)
        
        for idx, section in enumerate(sections):
            section_type = section.get('type', 'Empty')
            config = section.get('config', {})
            section_title = config.get('title', f'Section {idx + 1}')
            listbox.insert(tk.END, f"{section_title} ({section_type})")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def remove_selected():
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                sections[idx]['type'] = None  # Clear the section
                dialog.destroy()
                self._show_analysis_tab()
        
        ok_btn = ttk.Button(button_frame, text="Remove", command=remove_selected)
        ok_btn.pack(padx=5)
        
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack(padx=5)
    
    def _show_add_page_dialog(self, instance_alias: str):
        """Show dialog to add a new page."""
        if instance_alias not in self.analysis_data:
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Page")
        dialog.geometry("350x250")
        dialog.resizable(False, False)
        
        # Page title
        title_label = ttk.Label(dialog, text="Page Title:", font=("Arial", 10))
        title_label.pack(padx=10, pady=(10, 5))
        
        title_entry = ttk.Entry(dialog, width=30)
        title_entry.pack(padx=10, pady=(0, 10))
        title_entry.insert(0, f"Page {len(self.analysis_data[instance_alias]['pages']) + 1}")
        
        # Layout selection
        layout_label = ttk.Label(dialog, text="Layout:", font=("Arial", 10))
        layout_label.pack(padx=10, pady=(10, 5))
        
        layouts = [
            ('fp', 'Full Page (1 section)'),
            ('ns', 'North-South (2 sections)'),
            ('ew', 'East-West (2 sections)'),
            ('fd', 'Four Divisions (4 sections)'),
            ('sd', 'South Division (3 sections)'),
        ]
        
        layout_var = tk.StringVar(value='fp')
        for layout_code, layout_desc in layouts:
            radio = ttk.Radiobutton(dialog, text=layout_desc, variable=layout_var, value=layout_code)
            radio.pack(anchor=tk.W, padx=30, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def add_page():
            title = title_entry.get() or "New Page"
            layout = layout_var.get()
            
            # Create sections based on layout
            num_sections = {
                'fp': 1, 'ns': 2, 'ew': 2, 'fd': 4, 'sd': 3, 'nd': 3, 'ed': 3, 'wd': 4
            }.get(layout, 1)
            
            new_page = {
                'title': title,
                'layout': layout,
                'sections': [{'type': None} for _ in range(num_sections)]
            }
            
            self.analysis_data[instance_alias]['pages'].append(new_page)
            self.analysis_data[instance_alias]['current_page'] = len(self.analysis_data[instance_alias]['pages']) - 1
            
            dialog.destroy()
            self._show_analysis_tab()
        
        ok_btn = ttk.Button(button_frame, text="Add", command=add_page)
        ok_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def _remove_current_page(self, instance_alias: str):
        """Remove the current page."""
        if instance_alias not in self.analysis_data:
            return
        
        pages = self.analysis_data[instance_alias]['pages']
        current_page = self.analysis_data[instance_alias]['current_page']
        
        if len(pages) <= 1:
            messagebox.showwarning("Warning", "Cannot remove the last page")
            return
        
        if messagebox.askyesno("Confirm", f"Remove page '{pages[current_page]['title']}'?"):
            pages.pop(current_page)
            self.analysis_data[instance_alias]['current_page'] = min(current_page, len(pages) - 1)
            self._show_analysis_tab()
    
    def _run_analysis_to_function(self, instance_alias: str):
        """Run the model up to the specified function and populate analysis data."""
        try:
            # Clear any cached execution results and graphs to ensure fresh data
            self._clear_execution_cache()
            
            # Find the index of this function in the methodology
            if instance_alias not in self.methodology_list:
                messagebox.showerror("Error", "Function not found in methodology")
                return
            
            stop_at_idx = self.methodology_list.index(instance_alias)
            
            # Generate model.json first
            if not self._generate_model_json():
                messagebox.showerror("Error", "Failed to generate model.json")
                return
            
            # Run analyst in partial mode
            from analyst import analyst_main
            
            # Capture output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            output_buffer = StringIO()
            sys.stdout = output_buffer
            sys.stderr = output_buffer
            
            try:
                # Run analyst with stop_at_function_idx parameter
                outputs = analyst_main(stop_at_function_idx=stop_at_idx)
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            # Store execution results in analysis data
            # Extract base function alias to handle both "load_data" and "load_data#2" cases
            base_alias = instance_alias.split('#')[0] if '#' in instance_alias else instance_alias
            
            # Load analysis configuration from function's gui_config if available
            analysis_config = None
            if base_alias in self.gui_configs:
                analysis_config = self.gui_configs[base_alias].get('analysis')
            
            if instance_alias not in self.analysis_data:
                if analysis_config:
                    # Use analysis config from function's JSON
                    self.analysis_data[instance_alias] = {
                        'pages': copy.deepcopy(analysis_config.get('pages', [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}])),
                        'current_page': analysis_config.get('current_page', 0)
                    }
                else:
                    # Fallback to default structure
                    self.analysis_data[instance_alias] = {
                        'pages': [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}],
                        'current_page': 0
                    }
            # Note: When analysis_data already exists, we preserve the user's modifications
            # (like removed sections). The condition evaluation will use fresh inputs from
            # execution_results which is updated below.
            
            # Get input parameters from function_configs
            input_parameters = self.function_configs.get(instance_alias, {}).copy()
            
            # Get outputs for this function instance
            function_outputs = outputs.get(instance_alias, {}) if outputs else {}
            
            # Store the outputs from the execution
            self.analysis_data[instance_alias]['execution_results'] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'execution_time': 0,  # TODO: measure execution time
                'outputs': function_outputs,
                'inputs': input_parameters  # Store the input parameters for condition evaluation
            }
            
            # Note: graph_canvases, graph_slices, and table_slices are already cleared
            # by _clear_execution_cache() at the start of this method
            
            # Show success message
            messagebox.showinfo("Success", f"Model executed up to {instance_alias}\n\nResults loaded for analysis.")
            
            # Refresh the analysis tab to show results
            self._show_analysis_tab()
            
        except Exception as e:
            error_msg = f"Failed to run model: {str(e)}"
            messagebox.showerror("Error", error_msg)
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()

    
    def _show_report_tab(self):
        """Show Report tab (placeholder)."""
        self._clear_tab()
        self.current_tab = "report"
        
        label = ttk.Label(self.tab_content_frame, text=self.language_manager.translate("ui.messages.report_tab", "Report Tab (Under Development)"), 
                         font=("Arial", 12, "bold"))
        label.pack(padx=20, pady=20)
    
    def _generate_model_json(self) -> bool:
        """Generate model.json from current configuration."""
        try:
            from datetime import datetime
            
            # Load function specs to get parameter types
            specs_path = Path(__file__).parent / "function_specs.json"
            with open(specs_path, 'r', encoding='utf-8') as f:
                specs_data = json.load(f)
            parameter_types = specs_data.get("parameter_types", {})
            
            # First pass: build parameters for each function, including inherited parameters
            all_function_params = {}  # {instance_alias: {key: value}}
            
            for idx, instance_alias in enumerate(self.methodology_list):
                params = self.function_configs.get(instance_alias, {}).copy()
                base_func_alias = self.function_base_aliases[idx] if idx < len(self.function_base_aliases) else instance_alias
                
                # Get the GUI config for this function to check for inheritance
                func_config = self.gui_configs.get(base_func_alias, {})
                layout = func_config.get("setup", {}).get("layout", [])
                
                # Check for fields that should be inherited from previous functions
                for field_info in layout:
                    field_name = field_info.get("name", "")
                    if field_info.get("input_type") == "inherited":
                        # This field should be inherited from upstream
                        # Look for a previous function that has this field marked with input_type: "inherited"
                        for prev_idx in range(idx - 1, -1, -1):
                            prev_alias = self.methodology_list[prev_idx]
                            prev_base_alias = self.function_base_aliases[prev_idx] if prev_idx < len(self.function_base_aliases) else prev_alias
                            prev_config = self.gui_configs.get(prev_base_alias, {})
                            prev_layout = prev_config.get("setup", {}).get("layout", [])
                            
                            # Check if previous function has this field to inherit from
                            for prev_field_info in prev_layout:
                                if prev_field_info.get("name") == field_name and prev_field_info.get("input_type") != "inherited":
                                    # Found the upstream function with this field (not inherited itself)
                                    prev_params = all_function_params.get(prev_alias, {})
                                    if field_name in prev_params:
                                        params[field_name] = prev_params[field_name]
                                        break
                            if field_name in params:
                                break
                
                all_function_params[instance_alias] = params
            
            # Build functions array
            functions_array = []
            for idx, instance_alias in enumerate(self.methodology_list):
                base_alias = self.function_base_aliases[idx]
                params = all_function_params.get(instance_alias, {})
                func_config = self.gui_configs.get(base_alias, {})
                display_name = func_config.get("display_name", base_alias)
                
                # Process parameters: handle file path normalization
                processed_params = {}
                params_with_types = {}  # Store parameter types alongside values
                
                # Get type information for this function
                func_types = parameter_types.get(base_alias, {})
                
                for key, value in params.items():
                    # Skip None values and empty strings, but keep False (for checkbuttons)
                    if value is None or (isinstance(value, str) and not value):
                        continue
                    
                    # Special handling for file paths
                    if key in ("data_path", "var_path", "smp_path", "y_path", "y_val_path", "X_val_path", "Y_val_path"):
                        # Check if value is already a list
                        if isinstance(value, list):
                            # Already a list - normalize the paths inside
                            normalized_list = [p.replace("\\", "/") for p in value]
                            processed_params[key] = normalized_list
                        # Check if this is a multi-file path (contains semicolons)
                        elif ";" in str(value):
                            # Multiple files - convert to list
                            files = [f.strip().replace("\\", "/") for f in str(value).split(";")]
                            processed_params[key] = files
                        elif key == "data_path":
                            # Even single file should be a list for data_path
                            if str(value).startswith("["):
                                # String representation of list - parse it
                                try:
                                    path_list = json.loads(str(value))
                                    normalized_list = [p.replace("\\", "/") for p in path_list]
                                    processed_params[key] = normalized_list
                                except:
                                    processed_params[key] = [str(value).replace("\\", "/")]
                            else:
                                # Single path - normalize and wrap in list
                                normalized_path = str(value).replace("\\", "/")
                                processed_params[key] = [normalized_path]
                        else:
                            # Optional paths - normalize to forward slashes
                            processed_params[key] = str(value).replace("\\", "/")
                    else:
                        processed_params[key] = value
                    
                    # Store the type information for this parameter
                    param_type = func_types.get(key, "str")
                    params_with_types[key] = param_type
                
                functions_array.append({
                    "instance_alias": instance_alias,
                    "base_alias": base_alias,
                    "display_name": display_name,
                    "parameters": processed_params,
                    "parameter_types": params_with_types
                })
            
            # Build routing array
            routing_array = []
            for key, conn_info in self.routing_lines.items():
                if isinstance(conn_info, dict):
                    src_idx = conn_info.get("src_idx", key[0] if isinstance(key, tuple) else 0)
                    dst_idx = conn_info.get("dst_idx", key[2] if isinstance(key, tuple) and len(key) > 2 else 1)
                    src_param_key = conn_info.get("src_param_key", key[1] if isinstance(key, tuple) else "")
                    dst_param_key = conn_info.get("dst_param_key", key[3] if isinstance(key, tuple) and len(key) > 3 else "")
                    src_param_name = conn_info.get("src_param_name", src_param_key)
                    dst_param_name = conn_info.get("dst_param_name", dst_param_key)
                    auto_created = conn_info.get("auto_created", False)
                    
                    # Get function instance aliases from methodology list
                    src_instance_alias = self.methodology_list[src_idx] if src_idx < len(self.methodology_list) else ""
                    dst_instance_alias = self.methodology_list[dst_idx] if dst_idx < len(self.methodology_list) else ""
                    
                    if src_instance_alias and dst_instance_alias:
                        routing_array.append({
                            "source": {
                                "instance_alias": src_instance_alias,
                                "param_key": src_param_key,
                                "param_name": src_param_name
                            },
                            "destination": {
                                "instance_alias": dst_instance_alias,
                                "param_key": dst_param_key,
                                "param_name": dst_param_name
                            },
                            "auto_created": auto_created
                        })
            
            # Build complete model JSON
            model_data = {
                "metadata": {
                    "version": "1.0",
                    "created": datetime.now().isoformat(),
                    "description": "CM Studio Model Configuration"
                },
                "functions": functions_array,
                "routing": routing_array
            }
            
            # Add analysis config if present
            if hasattr(self, 'analysis_data') and self.analysis_data:
                model_data['analysis'] = self._serialize_analysis_data()
            
            # Write model.json
            model_path = Path(__file__).parent / "model.json"
            with open(model_path, "w", encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate model.json: {e}")
            return False
    
    def _serialize_analysis_data(self) -> dict:
        """Serialize analysis_data structure for saving to model.json."""
        try:
            analysis_config = {}
            
            for instance_alias, analysis_info in self.analysis_data.items():
                # Only serialize persistent data, skip execution results
                analysis_config[instance_alias] = {
                    'pages': analysis_info.get('pages', []),
                    'current_page': analysis_info.get('current_page', 0)
                }
                
                # For each page, preserve section configurations
                for page in analysis_config[instance_alias].get('pages', []):
                    for section in page.get('sections', []):
                        # Only keep config data, remove runtime references
                        if 'config' in section:
                            # Deep copy config to avoid reference issues
                            section['config'] = section['config'].copy()
            
            return analysis_config
        except Exception as e:
            print(f"Warning: Failed to serialize analysis data: {e}")
            return {}
    
    def _deserialize_analysis_data(self, analysis_config: dict):
        """Deserialize analysis_data from model.json."""
        try:
            if not analysis_config:
                return
            
            # Initialize analysis_data if needed
            if not hasattr(self, 'analysis_data'):
                self.analysis_data = {}
            
            for instance_alias, config_data in analysis_config.items():
                self.analysis_data[instance_alias] = {
                    'pages': config_data.get('pages', []),
                    'current_page': config_data.get('current_page', 0),
                    'execution_results': {}  # Will be populated on demand
                }
        except Exception as e:
            print(f"Warning: Failed to deserialize analysis data: {e}")
            self.analysis_data = {}
    
    def _clean_tempfiles(self):
        """Clean the tempfiles directory."""
        if self.tempfiles_dir.exists():
            try:
                shutil.rmtree(self.tempfiles_dir)
            except Exception as e:
                print(f"Warning: Could not clean tempfiles: {e}")
        self.tempfiles_dir.mkdir(exist_ok=True)
    
    def _on_close(self):
        """Clean up tempfiles and close the application."""
        self._clean_tempfiles()
        self.root.destroy()
    
    def _show_save_model_dialog(self):
        """Show dialog to choose between saving with or without data."""
        dialog = tk.Toplevel(self.root)
        dialog.title(self.language_manager.translate("ui.buttons.save_model", "Save Model"))
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog on the screen
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        label = ttk.Label(dialog, text=self.language_manager.translate("ui.messages.choose_save_option", "Choose save option:"), font=("Arial", 11))
        label.pack(pady=15)
        
        full_model_btn = ttk.Button(
            dialog,
            text=self.language_manager.translate("ui.messages.save_full_model", "Save Full Model (.mdfd)"),
            command=lambda: self._save_full_model(dialog)
        )
        full_model_btn.pack(pady=5, padx=20, fill=tk.X)
        
        with_data_btn = ttk.Button(
            dialog,
            text=self.language_manager.translate("ui.messages.save_with_calibration", "Save With Calibration (.mdcd)"),
            command=lambda: self._save_model_with_data(dialog)
        )
        with_data_btn.pack(pady=5, padx=20, fill=tk.X)
        
        method_only_btn = ttk.Button(
            dialog,
            text=self.language_manager.translate("ui.messages.save_method_only", "Save Method Only (.mdon)"),
            command=lambda: self._save_model_method_only(dialog)
        )
        method_only_btn.pack(pady=5, padx=20, fill=tk.X)
    
    def _get_path_parameters(self, func_alias: str) -> List[str]:
        """Get list of parameter names marked with ispath:true for a function."""
        path_params = []
        
        # Try to get from already-loaded gui_configs first (language-aware)
        if func_alias in self.gui_configs:
            config = self.gui_configs[func_alias]
        else:
            # Fall back to loading from FUNCTION_SPECS path
            gui_listing = FUNCTION_SPECS.get("gui_listing", {})
            if func_alias not in gui_listing:
                return path_params
            
            func_info = gui_listing[func_alias]
            config_file = func_info.get("config_path")
            
            if not config_file:
                return path_params
            
            # Load the config file
            config_path = Path(__file__).parent / config_file
            
            if not config_path.exists():
                return path_params
            
            try:
                with open(config_path, encoding='utf-8') as f:
                    config = json.load(f)
            except Exception:
                return path_params
        
        # Check setup.layout for ispath parameters
        for param in config.get("setup", {}).get("layout", []):
            if param.get("ispath", False):
                path_params.append(param.get("name"))
        
        return path_params
    
    def _save_model_with_data(self, dialog):
        """Save model with calibration data to .mdcd file."""
        dialog.destroy()
        
        # Open file dialog to choose save location
        save_path = filedialog.asksaveasfilename(
            defaultextension=".mdcd",
            filetypes=[("Model With Calibration", "*.mdcd"), ("All Files", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            # Generate model.json
            if not self._generate_model_json():
                return
            
            # Create temporary directory for packaging
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Read and modify model.json
                model_path = Path(__file__).parent / "model.json"
                with open(model_path, encoding='utf-8') as f:
                    model_data = json.load(f)
                
                # Process model.json: collect all file paths and copy them
                files_to_copy = []
                
                # Process each function's parameters
                for func_entry in model_data.get('functions', []):
                    instance_alias = func_entry.get('instance_alias', '')
                    base_alias = func_entry.get('base_alias', '')
                    params = func_entry.get('parameters', {})
                    
                    # Get path parameters for this function using ispath flag
                    path_params = self._get_path_parameters(base_alias)
                    
                    # For validation_data functions, remove all file paths
                    if 'validation_data' in base_alias:
                        for param_name in path_params:
                            if param_name in params:
                                del params[param_name]
                        continue
                    
                    # Replace paths with tempfiles references
                    for param_name in path_params:
                        if param_name in params:
                            param_value = params[param_name]
                            # Handle nested lists, simple lists, and string values
                            if isinstance(param_value, list):
                                new_paths = []
                                for item in param_value:
                                    if isinstance(item, list):
                                        # Nested list (e.g., sample_paths: list of lists)
                                        nested_paths = []
                                        for file_path in item:
                                            if isinstance(file_path, str) and file_path:
                                                src_file = Path(file_path)
                                                if src_file.exists():
                                                    files_to_copy.append(src_file)
                                                    nested_paths.append(f"tempfiles/{src_file.name}")
                                                else:
                                                    nested_paths.append(file_path)
                                            else:
                                                nested_paths.append(file_path)
                                        new_paths.append(nested_paths)
                                    elif isinstance(item, str) and item:
                                        src_file = Path(item)
                                        if src_file.exists():
                                            files_to_copy.append(src_file)
                                            new_paths.append(f"tempfiles/{src_file.name}")
                                        else:
                                            new_paths.append(item)
                                    else:
                                        new_paths.append(item)
                                params[param_name] = new_paths
                            elif isinstance(param_value, str) and param_value:
                                src_file = Path(param_value)
                                if src_file.exists():
                                    files_to_copy.append(src_file)
                                    params[param_name] = f"tempfiles/{src_file.name}"
                
                # Write modified model.json
                model_tmpfile = tmpdir_path / "model.json"
                with open(model_tmpfile, "w", encoding='utf-8') as f:
                    json.dump(model_data, f, indent=2, ensure_ascii=False)
                
                # Create files subdirectory and copy data files
                files_dir = tmpdir_path / "files"
                files_dir.mkdir(exist_ok=True)
                for src_file in files_to_copy:
                    shutil.copy(src_file, files_dir / src_file.name)
                
                # Create zip file with .mdcd extension
                zip_path = save_path[:-5] + ".zip" if save_path.endswith(".mdcd") else save_path + ".zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file_path in tmpdir_path.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(tmpdir_path)
                            zf.write(file_path, arcname)
                
                # Rename to .mdcd
                mdcd_path = save_path if save_path.endswith(".mdcd") else save_path + ".mdcd"
                if Path(zip_path).exists():
                    Path(zip_path).rename(mdcd_path)
                
                messagebox.showinfo(self.language_manager.translate("ui.dialogs.success", "Success"), 
                                  self.language_manager.translate("ui.messages.model_saved", "Model saved to:") + f"\n{mdcd_path}")
        
        except Exception as e:
            messagebox.showerror(self.language_manager.translate("ui.dialogs.error", "Error"), 
                               self.language_manager.translate("ui.messages.save_failed", "Failed to save model:") + f" {e}")
    
    def _save_model_method_only(self, dialog):
        """Save model method only (no data) to .mdon file (removes only file paths, keeps validation inputs)."""
        dialog.destroy()
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".mdon",
            filetypes=[(self.language_manager.translate("ui.dialogs.model_method_only", "Model Method Only"), "*.mdon"), 
                      (self.language_manager.translate("ui.dialogs.file_filter_all", "All Files"), "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            # Generate model.json
            if not self._generate_model_json():
                return
            
            # Create temporary directory for packaging
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Read and modify model.json - remove only file paths
                model_path = Path(__file__).parent / "model.json"
                with open(model_path, encoding='utf-8') as f:
                    model_data = json.load(f)
                
                # Process model.json: remove only file paths
                # Process each function's parameters
                for func_entry in model_data.get('functions', []):
                    instance_alias = func_entry.get('instance_alias', '')
                    base_alias = func_entry.get('base_alias', '')
                    params = func_entry.get('parameters', {})
                    
                    # Get path parameters for this function using ispath flag
                    path_params = self._get_path_parameters(base_alias)
                    
                    # Remove only file paths marked with ispath:true
                    for param_name in path_params:
                        if param_name in params:
                            del params[param_name]
                
                # Write modified model.json
                model_tmpfile = tmpdir_path / "model.json"
                with open(model_tmpfile, "w", encoding='utf-8') as f:
                    json.dump(model_data, f, indent=2, ensure_ascii=False)
                
                # Create zip file with .mdon extension
                zip_path = save_path[:-5] + ".zip" if save_path.endswith(".mdon") else save_path + ".zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file_path in tmpdir_path.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(tmpdir_path)
                            zf.write(file_path, arcname)
                
                # Rename to .mdon
                mdon_path = save_path if save_path.endswith(".mdon") else save_path + ".mdon"
                if Path(zip_path).exists():
                    Path(zip_path).rename(mdon_path)
                
                messagebox.showinfo(self.language_manager.translate("ui.dialogs.success", "Success"), 
                                  self.language_manager.translate("ui.messages.model_saved", "Model saved to:") + f"\n{mdon_path}")
        
        except Exception as e:
            messagebox.showerror(self.language_manager.translate("ui.dialogs.error", "Error"), 
                               self.language_manager.translate("ui.messages.save_failed", "Failed to save model:") + f" {e}")
    
    def _save_full_model(self, dialog):
        """Save full model to .mdfd file (includes all data and validation inputs)."""
        dialog.destroy()
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".mdfd",
            filetypes=[(self.language_manager.translate("ui.dialogs.model_full", "Model Full"), "*.mdfd"), 
                      (self.language_manager.translate("ui.dialogs.file_filter_all", "All Files"), "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            # Generate model.json
            if not self._generate_model_json():
                return
            
            # Create temporary directory for packaging
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Read model.json
                model_path = Path(__file__).parent / "model.json"
                with open(model_path, encoding='utf-8') as f:
                    model_data = json.load(f)
                
                files_to_copy = []
                
                # Process each function's parameters
                for func_entry in model_data.get('functions', []):
                    instance_alias = func_entry.get('instance_alias', '')
                    base_alias = func_entry.get('base_alias', '')
                    params = func_entry.get('parameters', {})
                    
                    # Get path parameters for this function using ispath flag
                    path_params = self._get_path_parameters(base_alias)
                    
                    # Process each path parameter (copy files, rename paths to tempfiles/)
                    for param_name in path_params:
                        if param_name in params:
                            param_value = params[param_name]
                            # Handle nested lists, simple lists, and string values
                            if isinstance(param_value, list):
                                new_paths = []
                                for item in param_value:
                                    if isinstance(item, list):
                                        # Nested list (e.g., sample_paths: list of lists)
                                        nested_paths = []
                                        for file_path in item:
                                            if isinstance(file_path, str) and file_path:
                                                src_file = Path(file_path)
                                                if src_file.exists():
                                                    files_to_copy.append(src_file)
                                                    nested_paths.append(f"tempfiles/{src_file.name}")
                                                else:
                                                    nested_paths.append(file_path)
                                            else:
                                                nested_paths.append(file_path)
                                        new_paths.append(nested_paths)
                                    elif isinstance(item, str) and item:
                                        src_file = Path(item)
                                        if src_file.exists():
                                            files_to_copy.append(src_file)
                                            new_paths.append(f"tempfiles/{src_file.name}")
                                        else:
                                            new_paths.append(item)
                                    else:
                                        new_paths.append(item)
                                params[param_name] = new_paths
                            elif isinstance(param_value, str) and param_value:
                                src_file = Path(param_value)
                                if src_file.exists():
                                    files_to_copy.append(src_file)
                                    params[param_name] = f"tempfiles/{src_file.name}"
                
                # Write modified model.json
                model_tmpfile = tmpdir_path / "model.json"
                with open(model_tmpfile, "w", encoding='utf-8') as f:
                    json.dump(model_data, f, indent=2, ensure_ascii=False)
                
                # Create files subdirectory and copy data files
                files_dir = tmpdir_path / "files"
                files_dir.mkdir(exist_ok=True)
                for src_file in files_to_copy:
                    shutil.copy(src_file, files_dir / src_file.name)
                
                # Create zip file with .mdfd extension
                zip_path = save_path[:-5] + ".zip" if save_path.endswith(".mdfd") else save_path + ".zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file_path in tmpdir_path.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(tmpdir_path)
                            zf.write(file_path, arcname)
                
                # Rename to .mdfd
                mdfd_path = save_path if save_path.endswith(".mdfd") else save_path + ".mdfd"
                if Path(zip_path).exists():
                    Path(zip_path).rename(mdfd_path)
                
                messagebox.showinfo(self.language_manager.translate("ui.dialogs.success", "Success"), 
                                  self.language_manager.translate("ui.messages.model_saved", "Model saved to:") + f"\n{mdfd_path}")
        
        except Exception as e:
            messagebox.showerror(self.language_manager.translate("ui.dialogs.error", "Error"), 
                               self.language_manager.translate("ui.messages.save_failed", "Failed to save model:") + f" {e}")
    
    def _show_load_model_dialog(self):
        """Show dialog to load a model from .mdcd, .mdon, or .mdfd file."""
        file_path = filedialog.askopenfilename(
            filetypes=[(self.language_manager.translate("ui.dialogs.model_files", "Model Files"), "*.mdcd *.mdon *.mdfd"), 
                      (self.language_manager.translate("ui.dialogs.file_filter_all", "All Files"), "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self._load_model(file_path)
        except Exception as e:
            messagebox.showerror(self.language_manager.translate("ui.dialogs.error", "Error"), 
                               self.language_manager.translate("ui.messages.load_failed", "Failed to load model:") + f" {e}")
    
    def _load_model(self, file_path: str):
        """Load a model from .mdcd, .mdon, or .mdfd file and update GUI."""
        file_path = Path(file_path)
        is_with_data = file_path.suffix in (".mdcd", ".mdfd") or str(file_path).endswith((".mdcd", ".mdfd"))
        
        # Clear any cached execution results and graphs to ensure fresh data
        self._clear_execution_cache()
        
        # Reload GUI configs to ensure fresh state (prevents stale modifications)
        self._load_gui_configs()
        
        # Clean tempfiles before loading
        self._clean_tempfiles()
        
        # Create a temporary extraction directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Rename to .zip and extract
            zip_path = tmpdir_path / "model.zip"
            shutil.copy(file_path, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmpdir_path)
            
            # Extract files folder to tempfiles if .mdwd
            if is_with_data:
                files_folder = tmpdir_path / "files"
                if files_folder.exists():
                    for src_file in files_folder.iterdir():
                        if src_file.is_file():
                            shutil.copy(src_file, self.tempfiles_dir / src_file.name)
            
            # Load functions.txt
            functions_file = tmpdir_path / "functions.txt"
            # Load model.json
            model_file = tmpdir_path / "model.json"
            if model_file.exists():
                model_path = Path(__file__).parent / "model.json"
                with open(model_file, encoding='utf-8') as f:
                    content = f.read()
                with open(model_path, "w", encoding='utf-8') as f:
                    f.write(content)
        
        # Parse and load configuration from model.json
        self._parse_and_load_model_json()
        
        # Refresh GUI
        self._refresh_gui_from_config()
        
        messagebox.showinfo(self.language_manager.translate("ui.dialogs.success", "Success"), 
                          self.language_manager.translate("ui.messages.model_loaded", "Model loaded from:") + f"\n{file_path}")
    
    def _parse_and_load_model_json(self):
        """Parse model.json and load configuration."""
        model_path = Path(__file__).parent / "model.json"
        if not model_path.exists():
            return
        
        self.methodology_list = []
        self.function_base_aliases = []
        self.function_configs = {}
        self.routing_lines = {}
        # Reset analysis_data completely when loading a new model
        # This ensures previous model's analysis changes don't persist
        self.analysis_data = {}
        
        try:
            with open(model_path, encoding='utf-8') as f:
                model_data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model.json: {e}")
            return
        
        # Load functions
        for func_entry in model_data.get('functions', []):
            instance_alias = func_entry.get('instance_alias', '')
            base_alias = func_entry.get('base_alias', '')
            params = func_entry.get('parameters', {}).copy()
            
            self.methodology_list.append(instance_alias)
            self.function_base_aliases.append(base_alias)
            
            # Convert tempfiles references to absolute paths
            for key, value in params.items():
                if isinstance(value, list):
                    converted_list = []
                    for item in value:
                        if isinstance(item, str) and item.startswith('tempfiles/'):
                            abs_path = str(self.tempfiles_dir / item.replace('tempfiles/', ''))
                            converted_list.append(abs_path)
                        else:
                            converted_list.append(item)
                    params[key] = converted_list
                elif isinstance(value, str) and value.startswith('tempfiles/'):
                    params[key] = str(self.tempfiles_dir / value.replace('tempfiles/', ''))
            
            self.function_configs[instance_alias] = params
        
        # Load routing
        for route_entry in model_data.get('routing', []):
            src_info = route_entry.get('source', {})
            dst_info = route_entry.get('destination', {})
            auto_created = route_entry.get('auto_created', False)
            
            src_alias = src_info.get('instance_alias', '')
            dst_alias = dst_info.get('instance_alias', '')
            src_param = src_info.get('param_key', '')
            dst_param = dst_info.get('param_key', '')
            src_param_name = src_info.get('param_name', src_param)
            dst_param_name = dst_info.get('param_name', dst_param)
            
            # Find indices
            try:
                src_idx = self.methodology_list.index(src_alias)
                dst_idx = self.methodology_list.index(dst_alias)
                
                key = (src_idx, src_param, dst_idx, dst_param)
                self.routing_lines[key] = {
                    "src_idx": src_idx,
                    "src_param_key": src_param,
                    "src_param_name": src_param_name,
                    "dst_idx": dst_idx,
                    "dst_param_key": dst_param,
                    "dst_param_name": dst_param_name,
                    "auto_created": auto_created
                }
            except ValueError:
                print(f"Warning: Could not find routing source or destination: {src_alias} -> {dst_alias}")
        
        # Load analysis configuration if present
        analysis_config = model_data.get('analysis', {})
        if analysis_config:
            self._deserialize_analysis_data(analysis_config)
    
    def _clear_execution_cache(self):
        """Clear all cached execution results and graphs, preserving analysis structure.
        
        This ensures that when a model is run, loaded, or a function is executed,
        the analysis displays fresh results without stale cached values.
        Preserves the analysis_data structure (pages, current_page) while clearing execution_results.
        Also clears graph_slices and table_slices to prevent stale slice indices from causing errors when
        data dimensions change (e.g., nway_flag changes).
        """
        # Clear execution_results and slice states while preserving analysis structure
        if hasattr(self, 'analysis_data') and self.analysis_data:
            for instance_alias in self.analysis_data:
                if 'execution_results' in self.analysis_data[instance_alias]:
                    del self.analysis_data[instance_alias]['execution_results']
                # Clear graph_slices to prevent stale dimension indices
                if 'graph_slices' in self.analysis_data[instance_alias]:
                    del self.analysis_data[instance_alias]['graph_slices']
                # Clear table_slices to prevent stale dimension indices for tables
                if 'table_slices' in self.analysis_data[instance_alias]:
                    del self.analysis_data[instance_alias]['table_slices']
                # Clear cached graph canvases as well
                if 'graph_canvases' in self.analysis_data[instance_alias]:
                    del self.analysis_data[instance_alias]['graph_canvases']
                # Clear control frame references
                if 'graph_control_frames' in self.analysis_data[instance_alias]:
                    del self.analysis_data[instance_alias]['graph_control_frames']
    
    def _refresh_gui_from_config(self):
        """Refresh GUI to reflect loaded configuration."""
        # Rebuild methodology listbox
        self.methodology_listbox.delete(0, tk.END)
        
        # Re-add methodology items with translated display names
        for idx, instance_alias in enumerate(self.methodology_list):
            base_alias = self.function_base_aliases[idx]
            config = self.gui_configs.get(base_alias, {})
            display_name = config.get("display_name", base_alias)
            
            # Count previous instances of this base alias for suffix
            existing_count = self.function_base_aliases[:idx].count(base_alias)
            
            if existing_count > 0:
                item_name = f"{display_name} #{existing_count + 1}"
            else:
                item_name = display_name
            
            self.methodology_listbox.insert(tk.END, item_name)
        
        # Clear setup tab
        self._clear_tab()
        self.selected_function_idx = None
        label = ttk.Label(self.tab_content_frame, text=self.language_manager.translate("ui.messages.model_loaded_select", "Model loaded. Select a function to configure."), 
                         font=("Arial", 10, "italic"))
        label.pack(padx=20, pady=20)
    
    def _run_model(self):
        """Execute model and capture output."""
        if not self.methodology_list:
            messagebox.showwarning(self.language_manager.translate("ui.dialogs.warning", "Warning"), 
                                 self.language_manager.translate("ui.messages.empty_methodology", "Add functions to Methodology first"))
            return
        
        # Clear any cached execution results and graphs to ensure fresh data
        self._clear_execution_cache()
        
        if not self._generate_model_json():
            return
        
        try:
            # Import and run analyst_main
            from analyst import analyst_main
            
            # Capture output
            output_buffer = StringIO()
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            try:
                sys.stdout = output_buffer
                sys.stderr = output_buffer
                
                # Run the full model and capture outputs
                outputs = analyst_main()
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            # Write output to model_log.txt
            log_path = Path(__file__).parent / "model_log.txt"
            with open(log_path, "w") as f:
                f.write(output_buffer.getvalue())
            
            # Load results into analysis_data for each function
            if not hasattr(self, 'analysis_data'):
                self.analysis_data = {}
            
            for idx, instance_alias in enumerate(self.methodology_list):
                base_alias = self.function_base_aliases[idx]
                
                # Initialize analysis data structure if not already present
                # If analysis_data already exists (from a loaded model), preserve it
                if instance_alias not in self.analysis_data:
                    # Load analysis configuration from function's gui_config if available
                    analysis_config = None
                    if base_alias in self.gui_configs:
                        analysis_config = self.gui_configs[base_alias].get('analysis')
                    
                    if analysis_config:
                        # Use analysis config from function's JSON
                        self.analysis_data[instance_alias] = {
                            'pages': copy.deepcopy(analysis_config.get('pages', [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}])),
                            'current_page': analysis_config.get('current_page', 0)
                        }
                    else:
                        # Fallback to default structure
                        self.analysis_data[instance_alias] = {
                            'pages': [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}],
                            'current_page': 0
                        }
                
                # Get input parameters from function_configs for condition evaluation
                input_parameters = self.function_configs.get(instance_alias, {}).copy()
                
                # Store the outputs from the execution
                self.analysis_data[instance_alias]['execution_results'] = {
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': 0,
                    'outputs': outputs.get(instance_alias, {}) if outputs else {},
                    'inputs': input_parameters  # Store the input parameters for condition evaluation
                }
            
            messagebox.showinfo(self.language_manager.translate("ui.dialogs.success", "Success"), 
                              self.language_manager.translate("ui.messages.model_executed", "Model executed successfully. Results loaded for analysis."))
            
            # Switch to analysis tab to show results
            if self.selected_function_idx is not None:
                self._show_analysis_tab()
            
        except Exception as e:
            error_log_path = Path(__file__).parent / "model_log.txt"
            with open(error_log_path, "w") as f:
                f.write(f"ERROR: {str(e)}\n\n")
                f.write(output_buffer.getvalue() if 'output_buffer' in locals() else "")
            
            messagebox.showerror(self.language_manager.translate("ui.dialogs.error", "Error"), 
                               self.language_manager.translate("ui.messages.execution_failed", "Model execution failed:") + f" {e}\n" + 
                               self.language_manager.translate("ui.messages.check_log", "Check model_log.txt for details"))


def main():
    """Main entry point for the GUI."""
    root = tk.Tk()
    root.iconbitmap("Icon.ico")
    app = ChemometricsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
