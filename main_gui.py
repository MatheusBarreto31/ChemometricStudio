"""
Main GUI application for CM Studio using tkinter + Sun-Valley theme.
Provides Setup, Routing, Analysis, and Report tabs for building analysis pipelines.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
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

# Import language manager
from language_manager import get_language_manager, _

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
        self.language_manager = get_language_manager()
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
        
        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.language_manager.translate("menu.help", "Help"), menu=help_menu)
        help_menu.add_command(label=self.language_manager.translate("menu.about", "About"), command=self._show_about_dialog)
    
    def _change_language(self, language_code: str):
        """Change the application language."""
        self.language_manager.set_language(language_code)
        self._refresh_ui_text()
    
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
            self._show_setup_tab()
    
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
        run_btn = ttk.Button(control_frame, text="► " + self.language_manager.translate("ui.buttons.run_model", "Run Model"), command=self._run_model, width=12)
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
        
        for widget_spec in layout:
            name = widget_spec.get("name")
            label_text = widget_spec.get("label", name)
            widget_type = widget_spec.get("widget")
            default = widget_spec.get("default", "")
            required = widget_spec.get("required", False)
            input_tooltip = widget_spec.get("tooltip", "")
            visible_if = widget_spec.get("visible_if", None)
            
            # Create a container frame for each input
            input_container = ttk.Frame(form_frame)
            input_container.pack(anchor=tk.W, padx=10, pady=(10, 2), fill=tk.X)
            
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
                "widget_spec": widget_spec
            }
            visible_widgets[name] = widget_data
            
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
                def on_entry_focus_out(event, n=name, e_widget=entry, a=instance_alias, vw=visible_widgets):
                    self._save_widget_value(a, n, e_widget.get())
                    self._update_field_visibility(a, vw)
                entry.bind("<FocusOut>", on_entry_focus_out)
                
                # Binding for KeyRelease: update visibility (value will be saved on FocusOut)
                entry.bind("<KeyRelease>", lambda e, a=instance_alias, vw=visible_widgets: self._update_field_visibility(a, vw))
                
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
                def on_combo_selected(event, n=name, c_widget=combo, a=instance_alias, vw=visible_widgets, a2v=alias_to_value):
                    selected_alias = c_widget.get()
                    actual_value = a2v.get(selected_alias, selected_alias)
                    self._save_widget_value(a, n, actual_value)
                    self._update_field_visibility(a, vw)
                combo.bind("<<ComboboxSelected>>", on_combo_selected)
                
                widget_data["widget"] = combo
                widget_data["value_to_alias"] = value_to_alias  # Store for later reference if needed
                
            elif widget_type == "checkbutton":
                default_val = widget_spec.get("default", False)
                var = tk.BooleanVar(value=func_config.get(name, default_val))
                check = ttk.Checkbutton(input_container, text=label_text, variable=var, 
                                       command=lambda n=name, v=var, a=instance_alias, vw=visible_widgets: 
                                       (self._save_widget_value(a, n, v.get()), self._update_field_visibility(a, vw)))
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
        
        # Initial visibility update
        self._update_field_visibility(instance_alias, visible_widgets)
    
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
    
    def _update_field_visibility(self, func_alias: str, visible_widgets: Dict):
        """Update visibility of fields based on visible_if conditions."""
        func_config = self.function_configs.get(func_alias, {})
        
        for field_name, widget_data in visible_widgets.items():
            visible_if = widget_data.get("visible_if")
            container = widget_data.get("container")
            widget_spec = widget_data.get("widget_spec", {})
            
            # Default to showing the field
            should_show = True
            
            # Handle explicit false (for always-hidden fields)
            if visible_if is False:
                should_show = False
            elif visible_if:
                # visible_if is a dict like {"method": "moving_average"} or {"input_type": "user"}
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
                        
                        # Compare values
                        if current_value != condition_value:
                            should_show = False
                            break
            
            # Show or hide the container
            if should_show:
                container.pack(anchor=tk.W, padx=10, pady=(10, 2), fill=tk.X)
            else:
                container.pack_forget()
    
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

    
    def _show_analysis_tab(self):
        """Show Analysis tab (placeholder)."""
        self._clear_tab()
        self.current_tab = "analysis"
        
        label = ttk.Label(self.tab_content_frame, text=self.language_manager.translate("ui.messages.analysis_tab", "Analysis Tab (Under Development)"), 
                         font=("Arial", 12, "bold"))
        label.pack(padx=20, pady=20)
    
    def _show_report_tab(self):
        """Show Report tab (placeholder)."""
        self._clear_tab()
        self.current_tab = "report"
        
        label = ttk.Label(self.tab_content_frame, text=self.language_manager.translate("ui.messages.report_tab", "Report Tab (Under Development)"), 
                         font=("Arial", 12, "bold"))
        label.pack(padx=20, pady=20)
    
    def _generate_config_files(self) -> bool:
        """Generate functions.txt and routing.txt from current configuration."""
        try:
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
            
            # Second pass: generate functions.txt with assembled parameters
            functions_content = []
            for instance_alias in self.methodology_list:
                params = all_function_params.get(instance_alias, {})
                
                # Build parameter string
                param_parts = []
                for key, value in params.items():
                    if value or key.startswith("d_specs"):  # Include d_specs even if empty for assembly
                        # Special handling for d_specs: assemble from individual parts
                        if key == "d_specs_separator" and value:
                            # Assemble d_specs list from individual components
                            d_specs_sep = params.get("d_specs_separator", "tabs")
                            d_specs_headlines = params.get("d_specs_headlines", "0")
                            d_specs_type = params.get("d_specs_type", "x_matrix")
                            d_specs_dims = params.get("d_specs_dimensions", "")
                            d_specs_value = f'["{d_specs_sep}","{d_specs_headlines}","{d_specs_type}","{d_specs_dims}"]'
                            param_parts.append(f"d_specs:{d_specs_value}")
                            # Skip the individual d_specs parts in the loop
                            continue
                        elif key.startswith("d_specs_") and key != "d_specs_separator":
                            # Skip individual d_specs parts (already assembled above)
                            continue
                        elif value:
                            # Special handling for file paths: convert semicolon-separated to JSON list
                            if key in ("data_path", "var_path", "smp_path", "y_path", "X_val_path", "y_val_path"):
                                # Check if this is a multi-file path (contains semicolons)
                                if ";" in str(value):
                                    # Multiple files - convert to list
                                    files = [f.strip().replace("\\", "/") for f in str(value).split(";")]
                                    formatted_value = json.dumps(files)  # Use json.dumps for proper formatting
                                elif key == "data_path":
                                    # Even single file should be a list for data_path
                                    # Make sure value is not already a list representation
                                    if str(value).startswith("["):
                                        # Already a list - normalize the paths inside
                                        try:
                                            path_list = json.loads(str(value))
                                            normalized_list = [p.replace("\\", "/") for p in path_list]
                                            formatted_value = json.dumps(normalized_list)
                                        except:
                                            formatted_value = value
                                    else:
                                        # Single path - normalize and wrap in list
                                        normalized_path = str(value).replace("\\", "/")
                                        formatted_value = json.dumps([normalized_path])
                                else:
                                    # Optional paths - normalize to forward slashes
                                    formatted_value = str(value).replace("\\", "/")
                            else:
                                formatted_value = value
                            
                            param_parts.append(f"{key}:{formatted_value}")
                
                param_str = " ".join(param_parts)
                functions_content.append(f"{instance_alias} {param_str}".strip())
            
            functions_path = Path(__file__).parent / "functions.txt"
            with open(functions_path, "w") as f:
                f.write("\n".join(functions_content))
            
            # Generate routing.txt
            routing_content = []
            for key, conn_info in self.routing_lines.items():
                if isinstance(conn_info, dict):
                    # New format with full information
                    src_idx = conn_info.get("src_idx", key[0] if isinstance(key, tuple) else 0)
                    dst_idx = conn_info.get("dst_idx", key[2] if isinstance(key, tuple) and len(key) > 2 else 1)
                    src_param_key = conn_info.get("src_param_key", key[1] if isinstance(key, tuple) else "")
                    dst_param_key = conn_info.get("dst_param_key", key[3] if isinstance(key, tuple) and len(key) > 3 else "")
                    
                    # Get function instance aliases from methodology list
                    src_instance_alias = self.methodology_list[src_idx] if src_idx < len(self.methodology_list) else ""
                    dst_instance_alias = self.methodology_list[dst_idx] if dst_idx < len(self.methodology_list) else ""
                    
                    # Format: src_instance.src_param -> dst_instance.dst_param
                    if src_instance_alias and dst_instance_alias:
                        routing_content.append(f"{src_instance_alias}.{src_param_key} -> {dst_instance_alias}.{dst_param_key}")
                else:
                    # Legacy format - just output the display
                    routing_content.append(str(conn_info))
            
            routing_path = Path(__file__).parent / "routing.txt"
            with open(routing_path, "w") as f:
                f.write("\n".join(routing_content))
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate config files: {e}")
            return False
    
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
            # Generate functions.txt and routing.txt
            if not self._generate_config_files():
                return
            
            # Create temporary directory for packaging
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Read and modify functions.txt
                functions_path = Path(__file__).parent / "functions.txt"
                with open(functions_path) as f:
                    functions_content = f.read()
                
                # Process functions: collect all file paths and copy them
                files_to_copy = []
                modified_functions = []
                
                for line in functions_content.split('\n'):
                    if not line.strip():
                        modified_functions.append(line)
                        continue
                    
                    # Extract function instance alias (first word)
                    parts = line.split()
                    instance_alias = parts[0] if parts else ""
                    
                    # Get base alias to look up path parameters
                    base_alias = None
                    try:
                        idx = self.methodology_list.index(instance_alias)
                        base_alias = self.function_base_aliases[idx]
                    except (ValueError, IndexError):
                        # If not found, try to extract from instance alias
                        if '#' in instance_alias:
                            base_alias = instance_alias.split('#')[0]
                        else:
                            base_alias = instance_alias
                    
                    # Get path parameters for this function using ispath flag
                    path_params = self._get_path_parameters(base_alias)
                    
                    modified_line = line
                    
                    # For validation_data functions, remove all file paths
                    if 'validation_data' in base_alias:
                        for param_name in path_params:
                            import re
                            modified_line = re.sub(rf'{param_name}:[^\s]*', '', modified_line)
                        # Clean up multiple spaces
                        modified_line = ' '.join(modified_line.split())
                        modified_functions.append(modified_line)
                        continue
                    
                    # Replace paths with tempfiles references
                    for param_name in path_params:
                        param_prefix = f'{param_name}:'
                        if param_prefix in modified_line:
                            # Find start position
                            start_pos = modified_line.find(param_prefix)
                            if start_pos != -1:
                                start_pos += len(param_prefix)
                                rest_of_line = modified_line[start_pos:]
                                
                                if rest_of_line.startswith('['):
                                    # It's a JSON list - find the closing bracket
                                    end_pos = rest_of_line.find(']')
                                    if end_pos != -1:
                                        value_str = rest_of_line[:end_pos+1]
                                        try:
                                            import json
                                            paths_list = json.loads(value_str)
                                            if isinstance(paths_list, list):
                                                new_paths = []
                                                for file_path in paths_list:
                                                    src_file = Path(file_path)
                                                    if src_file.exists():
                                                        files_to_copy.append(src_file)
                                                        new_path = f"tempfiles/{src_file.name}"
                                                        new_paths.append(new_path)
                                                    else:
                                                        new_paths.append(file_path)
                                                new_value_str = json.dumps(new_paths)
                                                old_value = value_str
                                                modified_line = modified_line[:start_pos] + new_value_str + modified_line[start_pos+len(old_value):]
                                        except Exception as e:
                                            pass
                                else:
                                    # It's a single value - find the next space or end of line
                                    space_pos = rest_of_line.find(' ')
                                    if space_pos == -1:
                                        value_str = rest_of_line
                                        end_in_rest = len(rest_of_line)
                                    else:
                                        value_str = rest_of_line[:space_pos]
                                        end_in_rest = space_pos
                                    
                                    # Check if file exists and copy it
                                    src_file = Path(value_str)
                                    if src_file.exists():
                                        files_to_copy.append(src_file)
                                        new_path = f"tempfiles/{src_file.name}"
                                        modified_line = modified_line[:start_pos] + new_path + modified_line[start_pos+end_in_rest:]
                    
                    modified_functions.append(modified_line)
                
                # Write modified functions.txt
                functions_tmpfile = tmpdir_path / "functions.txt"
                with open(functions_tmpfile, "w") as f:
                    f.write('\n'.join(modified_functions))
                
                # Copy routing.txt
                routing_path = Path(__file__).parent / "routing.txt"
                if routing_path.exists():
                    shutil.copy(routing_path, tmpdir_path / "routing.txt")
                
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
            # Generate functions.txt and routing.txt
            if not self._generate_config_files():
                return
            
            # Create temporary directory for packaging
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Read and modify functions.txt - remove only file paths
                functions_path = Path(__file__).parent / "functions.txt"
                with open(functions_path) as f:
                    functions_content = f.read()
                
                modified_functions = []
                
                for line in functions_content.split('\n'):
                    if not line.strip():
                        modified_functions.append(line)
                        continue
                    
                    # Extract function instance alias (first word)
                    parts = line.split()
                    instance_alias = parts[0] if parts else ""
                    
                    # Get base alias to look up path parameters
                    base_alias = None
                    try:
                        idx = self.methodology_list.index(instance_alias)
                        base_alias = self.function_base_aliases[idx]
                    except (ValueError, IndexError):
                        # If not found, try to extract from instance alias
                        if '#' in instance_alias:
                            base_alias = instance_alias.split('#')[0]
                        else:
                            base_alias = instance_alias
                    
                    # Get path parameters for this function using ispath flag
                    path_params = self._get_path_parameters(base_alias)
                    
                    # Remove only file paths marked with ispath:true (keep other validation inputs)
                    modified_line = line
                    for param_name in path_params:
                        # Look for param_name:value pattern and remove it
                        import re
                        # Remove the specific param:value part
                        param_pattern = rf'{param_name}:[^\s]*'
                        modified_line = re.sub(param_pattern, '', modified_line)
                    
                    # Clean up multiple spaces
                    modified_line = ' '.join(modified_line.split())
                    modified_functions.append(modified_line)
                
                # Write modified functions.txt
                functions_tmpfile = tmpdir_path / "functions.txt"
                with open(functions_tmpfile, "w") as f:
                    f.write('\n'.join(modified_functions))
                
                # Copy routing.txt
                routing_path = Path(__file__).parent / "routing.txt"
                if routing_path.exists():
                    shutil.copy(routing_path, tmpdir_path / "routing.txt")
                
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
            # Generate functions.txt and routing.txt
            if not self._generate_config_files():
                return
            
            # Create temporary directory for packaging
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Read functions.txt
                functions_path = Path(__file__).parent / "functions.txt"
                with open(functions_path) as f:
                    functions_content = f.read()
                
                modified_functions = []
                files_to_copy = []
                
                for line in functions_content.split('\n'):
                    if not line.strip():
                        modified_functions.append(line)
                        continue
                    
                    # Extract function instance alias (first word)
                    parts = line.split()
                    instance_alias = parts[0] if parts else ""
                    
                    # Get base alias to look up path parameters
                    base_alias = None
                    try:
                        idx = self.methodology_list.index(instance_alias)
                        base_alias = self.function_base_aliases[idx]
                    except (ValueError, IndexError):
                        # If not found, try to extract from instance alias
                        if '#' in instance_alias:
                            base_alias = instance_alias.split('#')[0]
                        else:
                            base_alias = instance_alias
                    
                    # Get path parameters for this function using ispath flag
                    path_params = self._get_path_parameters(base_alias)
                    
                    # Process each path parameter (copy files, rename paths to tempfiles/)
                    modified_line = line
                    for param_name in path_params:
                        # Look for param_name:[...] or param_name:value pattern
                        import re
                        
                        # Find the parameter in the line
                        param_pattern = rf'{param_name}:'
                        if param_pattern in modified_line:
                            start_pos = modified_line.find(param_pattern) + len(param_pattern)
                            rest_of_line = modified_line[start_pos:]
                            
                            # Check if it's a list [...]
                            if rest_of_line.startswith('['):
                                # Find matching ]
                                bracket_count = 0
                                end_idx = -1
                                for i, char in enumerate(rest_of_line):
                                    if char == '[':
                                        bracket_count += 1
                                    elif char == ']':
                                        bracket_count -= 1
                                        if bracket_count == 0:
                                            end_idx = i
                                            break
                                
                                if end_idx != -1:
                                    value_str = rest_of_line[:end_idx+1]
                                    try:
                                        # Parse the list
                                        paths_list = json.loads(value_str)
                                        new_paths = []
                                        for file_path in paths_list:
                                            src_file = Path(file_path)
                                            if src_file.exists():
                                                files_to_copy.append(src_file)
                                                new_path = f"tempfiles/{src_file.name}"
                                                new_paths.append(new_path)
                                            else:
                                                new_paths.append(file_path)
                                        new_value_str = json.dumps(new_paths)
                                        old_value = value_str
                                        modified_line = modified_line[:start_pos] + new_value_str + modified_line[start_pos+len(old_value):]
                                    except Exception as e:
                                        pass
                            else:
                                # It's a single value - find the next space or end of line
                                space_pos = rest_of_line.find(' ')
                                if space_pos == -1:
                                    value_str = rest_of_line
                                    end_in_rest = len(rest_of_line)
                                else:
                                    value_str = rest_of_line[:space_pos]
                                    end_in_rest = space_pos
                                
                                # Check if file exists and copy it
                                src_file = Path(value_str)
                                if src_file.exists():
                                    files_to_copy.append(src_file)
                                    new_path = f"tempfiles/{src_file.name}"
                                    modified_line = modified_line[:start_pos] + new_path + modified_line[start_pos+end_in_rest:]
                    
                    modified_functions.append(modified_line)
                
                # Write modified functions.txt
                functions_tmpfile = tmpdir_path / "functions.txt"
                with open(functions_tmpfile, "w") as f:
                    f.write('\n'.join(modified_functions))
                
                # Copy routing.txt
                routing_path = Path(__file__).parent / "routing.txt"
                if routing_path.exists():
                    shutil.copy(routing_path, tmpdir_path / "routing.txt")
                
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
            if functions_file.exists():
                functions_path = Path(__file__).parent / "functions.txt"
                with open(functions_file) as f:
                    content = f.read()
                with open(functions_path, "w") as f:
                    f.write(content)
            
            # Load routing.txt
            routing_file = tmpdir_path / "routing.txt"
            if routing_file.exists():
                routing_path = Path(__file__).parent / "routing.txt"
                with open(routing_file) as f:
                    content = f.read()
                with open(routing_path, "w") as f:
                    f.write(content)
        
        # Parse and load configuration from functions.txt
        self._parse_and_load_functions_txt()
        
        # Parse and load routing configuration
        self._parse_and_load_routing_txt()
        
        # Refresh GUI
        self._refresh_gui_from_config()
        
        messagebox.showinfo(self.language_manager.translate("ui.dialogs.success", "Success"), 
                          self.language_manager.translate("ui.messages.model_loaded", "Model loaded from:") + f"\n{file_path}")
    
    def _parse_and_load_functions_txt(self):
        """Parse functions.txt and load configuration."""
        functions_path = Path(__file__).parent / "functions.txt"
        if not functions_path.exists():
            return
        
        self.methodology_list = []
        self.function_base_aliases = []
        self.function_configs = {}
        
        with open(functions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                instance_alias = parts[0]
                # Extract base alias (remove #N suffix if present)
                if '#' in instance_alias:
                    base_alias = instance_alias.split('#')[0]
                else:
                    base_alias = instance_alias
                
                self.methodology_list.append(instance_alias)
                self.function_base_aliases.append(base_alias)
                
                # Parse parameters
                params = {}
                for part in parts[1:]:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        # Handle JSON list values
                        if value.startswith('[') and value.endswith(']'):
                            import json
                            try:
                                params[key] = json.loads(value)
                                # For tempfiles references, convert to absolute paths so they can be found
                                if isinstance(params[key], list):
                                    converted_list = []
                                    for item in params[key]:
                                        if isinstance(item, str) and item.startswith('tempfiles/'):
                                            # Convert to absolute path for file existence checking
                                            abs_path = str(self.tempfiles_dir / item.replace('tempfiles/', ''))
                                            converted_list.append(abs_path)
                                        else:
                                            converted_list.append(item)
                                    params[key] = converted_list
                            except:
                                params[key] = value
                        else:
                            # Single value - check if it's a tempfiles reference
                            if value.startswith('tempfiles/'):
                                # Convert to absolute path for file existence checking
                                params[key] = str(self.tempfiles_dir / value.replace('tempfiles/', ''))
                            else:
                                # Normalize validation_mode values from old format to new format
                                if key == 'validation_mode' and base_alias == 'validation_data':
                                    # Old format: "Load External Validation Set" or "Create Validation Set"
                                    # New format: "Load" or "Create"
                                    if value in ("Load External Validation Set", "load"):
                                        params[key] = "Load"
                                    elif value in ("Create Validation Set", "create"):
                                        params[key] = "Create"
                                    else:
                                        params[key] = value
                                else:
                                    params[key] = value
                
                self.function_configs[instance_alias] = params
    
    def _parse_and_load_routing_txt(self):
        """Parse routing.txt and load routing configuration."""
        routing_path = Path(__file__).parent / "routing.txt"
        if not routing_path.exists():
            return
        
        self.routing_lines = {}
        
        with open(routing_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: src_instance_alias.src_param -> dst_instance_alias.dst_param
                if '->' in line:
                    src_part, dst_part = line.split('->')
                    src_instance_alias, src_param = src_part.strip().split('.')
                    dst_instance_alias, dst_param = dst_part.strip().split('.')
                    
                    # Find indices in methodology list using instance aliases
                    try:
                        src_idx = self.methodology_list.index(src_instance_alias)
                        dst_idx = self.methodology_list.index(dst_instance_alias)
                        
                        key = (src_idx, src_param, dst_idx, dst_param)
                        self.routing_lines[key] = {
                            "src_idx": src_idx,
                            "src_param_key": src_param,
                            "dst_idx": dst_idx,
                            "dst_param_key": dst_param,
                            "auto_created": False
                        }
                    except ValueError:
                        # Instance alias not found in methodology list
                        print(f"Warning: Could not find routing source or destination function instance in methodology")
    
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
        
        if not self._generate_config_files():
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
                
                analyst_main()
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            # Write output to model_log.txt
            log_path = Path(__file__).parent / "model_log.txt"
            with open(log_path, "w") as f:
                f.write(output_buffer.getvalue())
            
            messagebox.showinfo(self.language_manager.translate("ui.dialogs.success", "Success"), 
                              self.language_manager.translate("ui.messages.model_executed", "Model executed successfully. Output saved to model_log.txt"))
            
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
