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
import platform
from datetime import datetime
import tempfile
import threading
import time
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

# Import add graph dialog
from add_graph_dialog import show_add_graph_dialog

# Import add table dialog
from add_table_dialog import show_add_table_dialog

# Import routing map window
from routing_map_window import RoutingMapWindow
from chemometrics.reporting import build_latex_document, compile_latex_to_pdf

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
        self._disable_combobox_mousewheel()
        
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
        self.workflow_control_aliases = {
            "workflow_loop_start",
            "workflow_loop_end",
            "workflow_parallel_start",
            "workflow_parallel_branch",
            "workflow_parallel_end"
        }
        self.gui_configs: Dict[str, Dict] = {}  # {func_alias: config_data}
        self.notification_color_schemes: Dict[str, Dict[str, str]] = {
            "message": {"bg": "#D9EDF7", "fg": "#0288d1"},
            "success": {"bg": "#DFF2BF", "fg": "#2F7D32"},
            "warning": {"bg": "#FEEFB3", "fg": "#9F6000"},
            "error": {"bg": "#FFBABA", "fg": "#D8000C"}
        }
        self.notification_default_durations: Dict[str, int] = {
            "message": 2200,
            "success": 1800,
            "warning": 2800,
            "error": 3400
        }
        self.execution_progress_frame = None
        self.execution_progress_popup = None
        self.execution_progress_status_label = None
        self.execution_progress_percent_label = None
        self.execution_progress_bar = None
        self.execution_progress_mode = ""
        self.execution_progress_hide_after_id = None
        self._execution_progress_root_bind_set = False
        self.latest_timing_report: Optional[Dict[str, Any]] = None
        self.tools_menu = None
        self.timing_report_menu_index: Optional[int] = None
        self.report_data: Dict[str, Any] = {
            'elements': [],
            'selected_index': None
        }
        
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

    def _disable_combobox_mousewheel(self):
        """Prevent mouse wheel from changing ttk.Combobox selections."""
        self.root.bind_class("TCombobox", "<MouseWheel>", lambda e: "break", add="+")
        self.root.bind_class("TCombobox", "<Button-4>", lambda e: "break", add="+")
        self.root.bind_class("TCombobox", "<Button-5>", lambda e: "break", add="+")

    def _show_fading_notice(self, message: str, level: str = "message", duration_ms: Optional[int] = None):
        """Show a non-blocking fading notification.

        Color scheme is configured in self.notification_color_schemes.
        Supported levels: message, success, warning, error.
        """
        if not message:
            return

        if level not in self.notification_color_schemes:
            level = "message"

        colors = self.notification_color_schemes[level]
        if duration_ms is None:
            duration_ms = self.notification_default_durations.get(level, 2200)

        existing_toast = getattr(self, "_notification_toast", None)
        if existing_toast is not None and existing_toast.winfo_exists():
            existing_toast.destroy()

        toast = tk.Toplevel(self.root)
        self._notification_toast = toast
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)

        try:
            toast.attributes("-alpha", 0.96)
        except tk.TclError:
            pass

        container = tk.Frame(toast, bg=colors["bg"], bd=1, relief=tk.SOLID)
        container.pack(fill=tk.BOTH, expand=True)

        label = tk.Label(
            container,
            text=message,
            bg=colors["bg"],
            fg=colors["fg"],
            font=("Arial", 10),
            justify=tk.LEFT,
            anchor="w",
            wraplength=520,
            padx=12,
            pady=8
        )
        label.pack(fill=tk.BOTH, expand=True)

        self.root.update_idletasks()
        toast.update_idletasks()
        x = self.root.winfo_rootx() + self.root.winfo_width() - toast.winfo_reqwidth() - 24
        y = self.root.winfo_rooty() + 58
        toast.geometry(f"+{max(0, x)}+{max(0, y)}")

        fade_steps = 12
        fade_step_ms = 85

        def fade_out(step=fade_steps):
            if not toast.winfo_exists():
                return
            if step <= 0:
                toast.destroy()
                return
            try:
                toast.attributes("-alpha", max(0.0, step / fade_steps))
            except tk.TclError:
                if step <= 1 and toast.winfo_exists():
                    toast.destroy()
                return
            toast.after(fade_step_ms, lambda: fade_out(step - 1))

        toast.after(max(400, duration_ms), fade_out)

    def _show_fading_warning(self, message: str, duration_ms: Optional[int] = None):
        """Backward-compatible warning notification helper."""
        self._show_fading_notice(message, level="warning", duration_ms=duration_ms)

    def _show_fading_error(self, message: str, duration_ms: Optional[int] = None):
        """Show an error-style fading notification."""
        self._show_fading_notice(message, level="error", duration_ms=duration_ms)

    def _show_fading_message(self, message: str, duration_ms: Optional[int] = None):
        """Show a neutral info-style fading notification."""
        self._show_fading_notice(message, level="message", duration_ms=duration_ms)

    def _show_fading_success(self, message: str, duration_ms: Optional[int] = None):
        """Show a success-style fading notification."""
        self._show_fading_notice(message, level="success", duration_ms=duration_ms)
    
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

        # Modern progress bar style for model execution feedback
        style.configure(
            "Execution.Horizontal.TProgressbar",
            troughcolor="#2f2f2f",
            background="#2f9fff",
            bordercolor="#2f2f2f",
            lightcolor="#2f9fff",
            darkcolor="#2f9fff",
            thickness=12
        )
    
    def _build_menu_bar(self):
        """Build the menu bar with File, Tools, Settings, and Help menus."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.language_manager.translate("menu.file", "File"), menu=file_menu)
        file_menu.add_command(label=self.language_manager.translate("menu.load_model", "Load Model"), command=self._show_load_model_dialog)
        file_menu.add_command(label=self.language_manager.translate("menu.save_model", "Save Model"), command=self._show_save_model_dialog)
        file_menu.add_separator()
        file_menu.add_command(label=self.language_manager.translate("menu.exit", "Exit"), command=self._on_close)

        # Tools Menu
        self.tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.language_manager.translate("menu.tools", "Tools"), menu=self.tools_menu)
        self.tools_menu.add_command(
            label=self.language_manager.translate("menu.timing_report", "Timing Report"),
            command=self._show_timing_report_popup,
            state=tk.DISABLED
        )
        self.timing_report_menu_index = self.tools_menu.index("end")
        self._set_timing_report_menu_state(self.latest_timing_report is not None)
        
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
            qualitative_menu.add_command(label=cmap, command=lambda cm=cmap: self._change_qualitative_colormap(cm))
        
        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.language_manager.translate("menu.help", "Help"), menu=help_menu)
        help_menu.add_command(label=self.language_manager.translate("menu.about", "About"), command=self._show_about_dialog)

    def _set_timing_report_menu_state(self, enabled: bool):
        """Enable or disable the Timing Report menu item."""
        if self.tools_menu is None or self.timing_report_menu_index is None:
            return

        state = tk.NORMAL if enabled else tk.DISABLED
        try:
            self.tools_menu.entryconfig(self.timing_report_menu_index, state=state)
        except tk.TclError:
            pass

    def _store_timing_report(self, run_type_label: str, timing_report: Optional[Dict[str, Any]], stop_at_function_alias: Optional[str] = None):
        """Persist latest timing report and enable menu access."""
        if not timing_report:
            self.latest_timing_report = None
            self._set_timing_report_menu_state(False)
            return

        self.latest_timing_report = {
            'run_type': run_type_label,
            'stop_at_function_alias': stop_at_function_alias,
            'total_execution_time': timing_report.get('total_execution_time', 0.0),
            'function_timings': timing_report.get('function_timings', []),
            'executed_function_count': timing_report.get('executed_function_count', 0),
            'partial_run': timing_report.get('partial_run', False),
            'timestamp': datetime.now().isoformat()
        }
        self._set_timing_report_menu_state(True)

    def _format_execution_seconds(self, seconds: Any) -> str:
        """Format execution duration consistently for reports."""
        try:
            return f"{float(seconds):.3f} s"
        except (TypeError, ValueError):
            return "0.000 s"

    def _show_timing_report_popup(self):
        """Show timing report popup for latest model run."""
        report = self.latest_timing_report
        if not report:
            self._show_fading_warning(
                self.language_manager.translate("ui.messages.no_timing_report", "No timing report available. Run the model first.")
            )
            return

        report_win = tk.Toplevel(self.root)
        report_win.title(self.language_manager.translate("ui.dialogs.timing_report", "Timing Report"))
        report_win.geometry("640x460")
        report_win.transient(self.root)
        report_win.grab_set()

        container = ttk.Frame(report_win, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        text_frame = ttk.Frame(container)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        report_text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 10),
            padx=8,
            pady=8
        )
        report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=report_text.yview)

        lines = [
            self.language_manager.translate("ui.timing_report.title", "Execution Timing Report"),
            "",
            f"{self.language_manager.translate('ui.timing_report.run_type', 'Run type')}: {report.get('run_type', 'N/A')}",
            f"{self.language_manager.translate('ui.timing_report.total_time', 'Total run time')}: {self._format_execution_seconds(report.get('total_execution_time', 0.0))}",
            f"{self.language_manager.translate('ui.timing_report.executed_count', 'Executed functions')}: {report.get('executed_function_count', 0)}",
        ]

        if report.get('partial_run'):
            stop_alias = report.get('stop_at_function_alias') or self.language_manager.translate("ui.timing_report.unknown", "Unknown")
            if stop_alias != self.language_manager.translate("ui.timing_report.unknown", "Unknown"):
                # Look up display name for the stop target function
                if stop_alias in self.methodology_list:
                    idx = self.methodology_list.index(stop_alias)
                    base_alias = self.function_base_aliases[idx] if idx < len(self.function_base_aliases) else stop_alias
                    config = self.gui_configs.get(base_alias, {})
                    config_display_name = config.get("display_name", base_alias)
                    existing_count = self.methodology_list[:idx].count(base_alias)
                    if existing_count > 0:
                        stop_alias_display = f"{config_display_name} #{existing_count + 1}"
                    else:
                        stop_alias_display = config_display_name
                else:
                    stop_alias_display = stop_alias
            else:
                stop_alias_display = stop_alias
            lines.append(
                f"{self.language_manager.translate('ui.timing_report.run_to_here_target', 'Run to here target')}: {stop_alias_display}"
            )

        lines.append("")
        lines.append(self.language_manager.translate("ui.timing_report.per_function", "Per-function run time:"))

        function_timings = report.get('function_timings', [])
        if function_timings:
            for idx, function_entry in enumerate(function_timings, start=1):
                instance_alias = function_entry.get('instance_alias', '')
                base_alias = function_entry.get('base_alias', '')
                
                # Look up configured display name for this function
                config = self.gui_configs.get(base_alias, {})
                config_display_name = config.get("display_name", base_alias)
                
                # Count previous instances of this base alias for suffix
                existing_count = self.methodology_list[:self.methodology_list.index(instance_alias)].count(base_alias) if instance_alias in self.methodology_list else 0
                if existing_count > 0:
                    display_name = f"{config_display_name} #{existing_count + 1}"
                else:
                    display_name = config_display_name
                
                lines.append(f"{idx:>2}. {display_name}: {self._format_execution_seconds(function_entry.get('execution_time', 0.0))}")
        else:
            lines.append(self.language_manager.translate("ui.timing_report.no_functions", "No function timing data recorded."))

        report_text.insert(tk.END, "\n".join(lines))
        report_text.config(state=tk.DISABLED)

        close_btn = ttk.Button(
            container,
            text=self.language_manager.translate("ui.buttons.close", "Close"),
            command=report_win.destroy
        )
        close_btn.pack(anchor="e", pady=(10, 0))
    
    def _change_language(self, language_code: str):
        """Change the application language and save setting."""
        self.language_manager.set_language(language_code)
        self.settings_manager.set("language", language_code)
        self._refresh_ui_text()
    
    def _change_colormap(self, colormap_name: str):
        """Change the default continuous colormap and save setting."""
        self.settings_manager.set("colormap", colormap_name)
        self._show_fading_message(
            self.language_manager.translate("ui.messages.colormap_changed_to", "Colormap changed to") +
            f" '{colormap_name}'.\n" +
            self.language_manager.translate("ui.messages.colormap_used_new_plots", "This will be used for new plots.")
        )
    
    def _change_qualitative_colormap(self, colormap_name: str):
        """Change the qualitative colormap and save setting."""
        self.settings_manager.set("qualitative_colormap", colormap_name)
        self._show_fading_message(
            self.language_manager.translate("ui.messages.qual_colormap_changed_to", "Qualitative colormap changed to") +
            f" '{colormap_name}'.\n" +
            self.language_manager.translate("ui.messages.colormap_used_new_plots", "This will be used for new plots.")
        )
    
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
        desc_frame = ttk.LabelFrame(about_win, text=self.language_manager.translate("menu.about", "About"), padding=15)
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
        close_btn = ttk.Button(about_win, text=self.language_manager.translate("ui.buttons.close", "Close"), command=about_win.destroy)
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
        self.workspace_frame = workspace_frame

        # Keep progress popup centered over workspace while window/layout changes
        if not self._execution_progress_root_bind_set:
            self.root.bind("<Configure>", self._on_execution_progress_anchor_configure, add="+")
            self._execution_progress_root_bind_set = True
        workspace_frame.bind("<Configure>", self._on_execution_progress_anchor_configure, add="+")
        
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
        self._refresh_methodology_listbox(selected_idx=new_func_idx)
        
        # Auto-create routing for inputs that match previous outputs
        self._auto_create_routing(new_func_idx, func_alias)

    def _is_workflow_control(self, base_alias: str) -> bool:
        return base_alias in self.workflow_control_aliases

    def _get_workflow_scope_signature(self, target_idx: int) -> Tuple:
        """Return active workflow scope signature before target index.

        Signature includes loop/parallel nesting and current parallel branch so auto-routing
        can be constrained to the same branch context.
        """
        active_stack = []
        loop_counter = 0
        parallel_counter = 0

        for idx in range(max(0, target_idx)):
            base_alias = self.function_base_aliases[idx] if idx < len(self.function_base_aliases) else ""
            if base_alias == "workflow_loop_start":
                loop_counter += 1
                active_stack.append(("loop", loop_counter))
            elif base_alias == "workflow_loop_end":
                for stack_idx in range(len(active_stack) - 1, -1, -1):
                    if active_stack[stack_idx][0] == "loop":
                        active_stack.pop(stack_idx)
                        break
            elif base_alias == "workflow_parallel_start":
                parallel_counter += 1
                active_stack.append(("parallel", parallel_counter, 1))
            elif base_alias == "workflow_parallel_branch":
                for stack_idx in range(len(active_stack) - 1, -1, -1):
                    if active_stack[stack_idx][0] == "parallel":
                        p_type, p_id, p_branch = active_stack[stack_idx]
                        active_stack[stack_idx] = (p_type, p_id, p_branch + 1)
                        break
            elif base_alias == "workflow_parallel_end":
                for stack_idx in range(len(active_stack) - 1, -1, -1):
                    if active_stack[stack_idx][0] == "parallel":
                        active_stack.pop(stack_idx)
                        break

        return tuple(active_stack)

    def _can_auto_route_between(self, src_idx: int, dst_idx: int) -> bool:
        if src_idx < 0 or dst_idx < 0 or src_idx >= len(self.function_base_aliases) or dst_idx >= len(self.function_base_aliases):
            return False
        src_base = self.function_base_aliases[src_idx]
        dst_base = self.function_base_aliases[dst_idx]
        if self._is_workflow_control(src_base) or self._is_workflow_control(dst_base):
            return False
        src_scope = self._get_workflow_scope_signature(src_idx)
        dst_scope = self._get_workflow_scope_signature(dst_idx)

        # Same scope always allowed
        if src_scope == dst_scope:
            return True

        # Allow routing from outer (upstream) scopes into nested inner scopes
        # so functions inside loop/parallel blocks can receive inputs from previous functions.
        if len(src_scope) <= len(dst_scope) and dst_scope[:len(src_scope)] == src_scope:
            return True

        # Reject sibling/cross-branch auto-routing by default.
        return False

    def _get_output_spec_keys(self, func_alias: str) -> List[str]:
        return_specs = FUNCTION_SPECS.get("return_specs", {})
        outputs = return_specs.get(func_alias, [])
        keys = []
        for entry in outputs:
            if isinstance(entry, str):
                keys.append(entry)
            elif isinstance(entry, dict):
                key = entry.get("key")
                if key:
                    keys.append(key)
        return keys

    def _get_input_spec_candidates(self, func_alias: str) -> List[Tuple[str, List[str]]]:
        input_specs = FUNCTION_SPECS.get("input_specs", {})
        inputs = input_specs.get(func_alias, [])
        candidates: List[Tuple[str, List[str]]] = []
        for entry in inputs:
            if isinstance(entry, str):
                candidates.append((entry, [entry]))
            elif isinstance(entry, dict):
                key = entry.get("key")
                if not key:
                    continue
                alias_candidates = [key]
                aliases = entry.get("aliases", [])
                if isinstance(aliases, list):
                    alias_candidates.extend([alias for alias in aliases if isinstance(alias, str) and alias])
                candidates.append((key, alias_candidates))
        return candidates

    def _get_methodology_item_display(self, idx: int, depth: int) -> str:
        instance_alias = self.methodology_list[idx]
        base_alias = self.function_base_aliases[idx]
        config = self.gui_configs.get(base_alias, {})
        display_name = config.get("display_name", base_alias)
        existing_count = self.function_base_aliases[:idx].count(base_alias)
        if existing_count > 0:
            display_name = f"{display_name} #{existing_count + 1}"

        if base_alias == "workflow_loop_start":
            text = f"┌ {display_name}"
        elif base_alias == "workflow_loop_end":
            text = f"└ {display_name}"
        elif base_alias == "workflow_parallel_start":
            text = f"┌ {display_name}"
        elif base_alias == "workflow_parallel_branch":
            text = f"├ {display_name}"
        elif base_alias == "workflow_parallel_end":
            text = f"└ {display_name}"
        else:
            text = display_name

        indent = "    " * max(0, depth)
        return f"{indent}{text}"

    def _refresh_methodology_listbox(self, selected_idx: Optional[int] = None):
        """Rebuild methodology list display with workflow indentation."""
        if not hasattr(self, "methodology_listbox"):
            return

        self.methodology_listbox.delete(0, tk.END)
        depth = 0

        for idx, base_alias in enumerate(self.function_base_aliases):
            if base_alias in ("workflow_loop_end", "workflow_parallel_end"):
                depth = max(0, depth - 1)

            if base_alias == "workflow_parallel_branch":
                item_depth = max(0, depth - 1)
            else:
                item_depth = depth

            self.methodology_listbox.insert(tk.END, self._get_methodology_item_display(idx, item_depth))

            if base_alias in ("workflow_loop_start", "workflow_parallel_start"):
                depth += 1

        if selected_idx is not None and 0 <= selected_idx < len(self.methodology_list):
            self.methodology_listbox.selection_clear(0, tk.END)
            self.methodology_listbox.selection_set(selected_idx)
            self.methodology_listbox.activate(selected_idx)
            self.methodology_listbox.see(selected_idx)
    
    def _auto_create_routing(self, new_func_idx: int, new_func_alias: str):
        """Automatically create routing connections for parameters with matching names.
        
        Only connects to the immediately previous function with matching output.
        This avoids connecting to distant functions when intermediate functions exist.
        """
        if new_func_idx == 0:
            return  # First function, no previous outputs to route from
        if self._is_workflow_control(new_func_alias):
            return
        
        new_func_inputs = self._get_input_spec_candidates(new_func_alias)
        
        # Check each input parameter of the new function
        for dst_param_key, input_candidates in new_func_inputs:
            # Find the immediately previous function that outputs this parameter
            for src_idx in range(new_func_idx - 1, -1, -1):  # Check backwards from newest to oldest
                src_base_alias = self.function_base_aliases[src_idx]  # Get the base alias
                if not self._can_auto_route_between(src_idx, new_func_idx):
                    continue
                src_outputs = self._get_output_spec_keys(src_base_alias)
                matched_src_param = None
                for candidate in input_candidates:
                    if candidate in src_outputs:
                        matched_src_param = candidate
                        break
                
                if matched_src_param:
                    # Found the most recent function with this output
                    # Create automatic routing connection
                    key = (src_idx, matched_src_param, new_func_idx, dst_param_key)
                    
                    # Only add if not already exists
                    if key not in self.routing_lines:
                        self.routing_lines[key] = {
                            "src_idx": src_idx,
                            "src_param_key": matched_src_param,
                            "dst_idx": new_func_idx,
                            "dst_param_key": dst_param_key,
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
            self._refresh_methodology_listbox()
            self._clear_tab()
    
    def _clear_methodology(self):
        """Clear all methodology items."""
        self.methodology_list.clear()
        self.function_base_aliases.clear()
        self.function_configs.clear()
        self.routing_lines.clear()
        self.selected_function_idx = None
        self._refresh_methodology_listbox()
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

    def _ensure_execution_progress_popup(self):
        """Create floating execution progress popup with main GUI color scheme if needed."""
        popup = self.execution_progress_popup
        if popup is not None:
            try:
                if popup.winfo_exists():
                    return
            except Exception:
                pass

        popup = tk.Toplevel(self.root)
        popup.overrideredirect(True)
        popup.attributes("-topmost", True)
        popup.withdraw()

        # Use active ttk theme colors so popup matches light/dark mode
        style = ttk.Style()
        bg_color = style.lookup("TFrame", "background") or self.root.cget("bg") or "#f0f0f0"
        fg_color = style.lookup("TLabel", "foreground") or "#202020"
        border_color = style.lookup("TLabelframe", "bordercolor") or "#b5b5b5"

        # Outer border frame
        border_frame = tk.Frame(popup, bg=border_color, bd=0)
        border_frame.pack(fill=tk.BOTH, expand=True)

        # Inner card frame with padding
        card = tk.Frame(border_frame, bg=bg_color, bd=0)
        card.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        header_frame = tk.Frame(card, bg=bg_color)
        header_frame.pack(fill=tk.X, padx=14, pady=(10, 6))

        status_label = tk.Label(
            header_frame,
            text=self.language_manager.translate("ui.messages.ready_to_run", "Ready to run"),
            bg=bg_color,
            fg=fg_color,
            font=("Arial", 10, "bold"),
            anchor="w"
        )
        status_label.pack(side=tk.LEFT)

        percent_label = tk.Label(
            header_frame,
            text="0%",
            bg=bg_color,
            fg=fg_color,
            font=("Arial", 10),
            anchor="e"
        )
        percent_label.pack(side=tk.RIGHT)

        progress_bar = ttk.Progressbar(
            card,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=1,
            value=0,
            style="Execution.Horizontal.TProgressbar"
        )
        progress_bar.pack(fill=tk.X, padx=14, pady=(0, 12))

        self.execution_progress_popup = popup
        self.execution_progress_status_label = status_label
        self.execution_progress_percent_label = percent_label
        self.execution_progress_bar = progress_bar

    def _position_execution_progress_popup(self):
        """Position execution popup centered on the workspace/tabs pane."""
        popup = self.execution_progress_popup
        if popup is None or not popup.winfo_exists():
            return

        self.root.update_idletasks()
        popup.update_idletasks()

        anchor_widget = getattr(self, "workspace_frame", None)
        if anchor_widget is not None and anchor_widget.winfo_exists():
            anchor_widget.update_idletasks()
            anchor_x = anchor_widget.winfo_rootx()
            anchor_y = anchor_widget.winfo_rooty()
            anchor_width = anchor_widget.winfo_width()
            anchor_height = anchor_widget.winfo_height()
        else:
            anchor_x = self.root.winfo_rootx()
            anchor_y = self.root.winfo_rooty()
            anchor_width = self.root.winfo_width()
            anchor_height = self.root.winfo_height()

        width = max(420, min(560, anchor_width - 80))
        height = 80

        x = anchor_x + max(16, (anchor_width - width) // 2)
        y = anchor_y + max(16, (anchor_height - height) // 2)

        popup.geometry(f"{width}x{height}+{x}+{y}")

    def _on_execution_progress_anchor_configure(self, event=None):
        """Reposition execution popup when root/workspace geometry changes."""
        popup = self.execution_progress_popup
        if popup is None:
            return
        try:
            if popup.winfo_exists() and popup.winfo_ismapped():
                self._position_execution_progress_popup()
        except tk.TclError:
            pass

    def _begin_execution_progress(self, total_steps: int, mode_label: str):
        """Initialize and show execution progress bar."""
        self._ensure_execution_progress_popup()
        if not self.execution_progress_bar:
            return

        if self.execution_progress_hide_after_id is not None:
            try:
                self.root.after_cancel(self.execution_progress_hide_after_id)
            except Exception:
                pass
            self.execution_progress_hide_after_id = None

        total = max(1, int(total_steps))
        self.execution_progress_mode = mode_label
        self.execution_progress_bar.configure(maximum=total)
        self.execution_progress_bar.configure(value=0)

        if self.execution_progress_status_label:
            self.execution_progress_status_label.configure(text=f"{mode_label}: 0/{total}")
        if self.execution_progress_percent_label:
            self.execution_progress_percent_label.configure(text="0%")

        popup = self.execution_progress_popup
        if popup is not None and popup.winfo_exists():
            self._position_execution_progress_popup()
            popup.deiconify()
            popup.lift()
        self.root.update()

    def _show_execution_progress_message(self, status_text: str):
        """Show centered execution popup with an indeterminate progress message."""
        self._ensure_execution_progress_popup()
        if not self.execution_progress_bar:
            return

        if self.execution_progress_hide_after_id is not None:
            try:
                self.root.after_cancel(self.execution_progress_hide_after_id)
            except Exception:
                pass
            self.execution_progress_hide_after_id = None

        self.execution_progress_mode = ""
        self.execution_progress_bar.stop()
        self.execution_progress_bar.configure(mode="indeterminate")
        self.execution_progress_bar.configure(maximum=100)
        self.execution_progress_bar.configure(value=0)
        self.execution_progress_bar.start(12)

        if self.execution_progress_status_label:
            self.execution_progress_status_label.configure(text=status_text)
        if self.execution_progress_percent_label:
            self.execution_progress_percent_label.configure(text="")

        popup = self.execution_progress_popup
        if popup is not None and popup.winfo_exists():
            self._position_execution_progress_popup()
            popup.deiconify()
            popup.lift()
        self.root.update_idletasks()

    def _stop_execution_progress_message(self):
        """Stop indeterminate popup progress state and hide popup immediately."""
        if self.execution_progress_bar:
            try:
                self.execution_progress_bar.stop()
            except Exception:
                pass
            self.execution_progress_bar.configure(mode="determinate")
            self.execution_progress_bar.configure(value=0)
        self._hide_execution_progress()

    def _update_execution_progress(self, completed_steps: int, total_steps: int, instance_alias: str = "", base_alias: str = ""):
        """Update execution progress bar and status text."""
        if not self.execution_progress_bar:
            return

        total = max(1, int(total_steps))
        completed = min(max(0, int(completed_steps)), total)
        percent = int((completed / total) * 100)

        self.execution_progress_bar.configure(maximum=total)
        self.execution_progress_bar.configure(value=completed)

        current_func_label = ""
        if instance_alias or base_alias:
            display_name = ""

            if instance_alias and instance_alias in self.methodology_list:
                idx = self.methodology_list.index(instance_alias)
                resolved_base_alias = self.function_base_aliases[idx] if idx < len(self.function_base_aliases) else base_alias
                config = self.gui_configs.get(resolved_base_alias, {})
                display_name = config.get("display_name", resolved_base_alias or instance_alias)

                # Match methodology naming for duplicates
                duplicate_count = self.function_base_aliases[:idx].count(resolved_base_alias)
                if duplicate_count > 0:
                    display_name = f"{display_name} #{duplicate_count + 1}"
            else:
                resolved_base_alias = base_alias
                config = self.gui_configs.get(resolved_base_alias, {}) if resolved_base_alias else {}
                display_name = config.get("display_name", resolved_base_alias or instance_alias)

            if display_name:
                current_func_label = f" • {display_name}"

        if self.execution_progress_status_label:
            mode = self.execution_progress_mode or self.language_manager.translate("ui.buttons.run_model", "Run Model")
            self.execution_progress_status_label.configure(text=f"{mode}: {completed}/{total}{current_func_label}")
        if self.execution_progress_percent_label:
            self.execution_progress_percent_label.configure(text=f"{percent}%")

        self._position_execution_progress_popup()
        self.root.update_idletasks()

    def _finish_execution_progress(self, success: bool = True):
        """Finalize execution progress bar and auto-hide after a short delay."""
        popup = self.execution_progress_popup
        if popup is None or not popup.winfo_exists():
            return

        if self.execution_progress_status_label:
            if success:
                self.execution_progress_status_label.configure(
                    text=self.language_manager.translate("ui.messages.execution_complete", "Execution complete")
                )
            else:
                self.execution_progress_status_label.configure(
                    text=self.language_manager.translate("ui.messages.execution_failed", "Model execution failed")
                )

        if self.execution_progress_hide_after_id is not None:
            try:
                self.root.after_cancel(self.execution_progress_hide_after_id)
            except Exception:
                pass

        hide_delay_ms = 1800 if success else 3600
        self.execution_progress_hide_after_id = self.root.after(hide_delay_ms, self._hide_execution_progress)

    def _hide_execution_progress(self):
        """Hide execution progress area."""
        self.execution_progress_hide_after_id = None
        popup = self.execution_progress_popup
        if popup is not None and popup.winfo_exists():
            popup.withdraw()
    
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

        self._refresh_methodology_listbox(selected_idx=current_selected_idx)
    
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
            text=f"{self.language_manager.translate('ui.tabs.setup', 'Setup')}: {display_name}",
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
        locked_params = self._get_swept_param_locks_for_index(self.selected_function_idx)
        
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

                if name in locked_params:
                    entry.configure(state="disabled")
                    lock_label = ttk.Label(input_container, text="Swept by loop", font=("Arial", 8, "italic"))
                    lock_label.pack(anchor=tk.W, padx=20, pady=(0, 2))
                
                widget_data["widget"] = entry
                
            elif widget_type == "combobox":
                values = widget_spec.get("values", [])
                value_aliases = widget_spec.get("value_aliases", values)  # Use values as fallback if no aliases
                editable_combo = bool(widget_spec.get("editable", False))
                
                # Create mapping from alias to actual value
                alias_to_value = dict(zip(value_aliases, values))
                value_to_alias = dict(zip(values, value_aliases))
                
                # Display aliases in the combobox
                combo_state = "normal" if editable_combo else "readonly"
                combo = ttk.Combobox(input_container, values=value_aliases, width=37, state=combo_state)
                
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
                def on_combo_selected(event, n=name, c_widget=combo, a=instance_alias, vw=visible_widgets, ch=category_headers):
                    selected_alias = c_widget.get()
                    field_data = vw.get(n, {})
                    a2v = field_data.get("alias_to_value", alias_to_value)
                    if "," in selected_alias:
                        mapped_parts = [a2v.get(part.strip(), part.strip()) for part in selected_alias.split(',') if part.strip()]
                        actual_value = ",".join(str(part) for part in mapped_parts)
                    else:
                        actual_value = a2v.get(selected_alias, selected_alias)
                    self._save_widget_value(a, n, actual_value)
                    self._update_field_visibility(a, vw, ch)
                    current_base_alias = self.function_base_aliases[self.selected_function_idx] if self.selected_function_idx is not None else ""
                    if current_base_alias == "workflow_loop_start" and n in ("mode", "sweep_target", "benchmark_source"):
                        self._refresh_workflow_setup_dynamic_options(a, current_base_alias, vw)
                combo.bind("<<ComboboxSelected>>", on_combo_selected)

                def on_combo_focus_out(event, n=name, c_widget=combo, a=instance_alias, vw=visible_widgets, ch=category_headers):
                    if not editable_combo:
                        return
                    selected_alias = c_widget.get().strip()
                    field_data = vw.get(n, {})
                    a2v = field_data.get("alias_to_value", alias_to_value)
                    if "," in selected_alias:
                        mapped_parts = [a2v.get(part.strip(), part.strip()) for part in selected_alias.split(',') if part.strip()]
                        actual_value = ",".join(str(part) for part in mapped_parts)
                    else:
                        actual_value = a2v.get(selected_alias, selected_alias)
                    self._save_widget_value(a, n, actual_value)
                    self._update_field_visibility(a, vw, ch)
                combo.bind("<FocusOut>", on_combo_focus_out)

                if name in locked_params:
                    combo.configure(state="disabled")
                    lock_label = ttk.Label(input_container, text="Swept by loop", font=("Arial", 8, "italic"))
                    lock_label.pack(anchor=tk.W, padx=20, pady=(0, 2))
                
                widget_data["widget"] = combo
                widget_data["alias_to_value"] = alias_to_value
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

                if name in locked_params:
                    check.configure(state="disabled")
                    lock_label = ttk.Label(input_container, text="Swept by loop", font=("Arial", 8, "italic"))
                    lock_label.pack(anchor=tk.W, padx=20, pady=(0, 2))
                
                widget_data["widget"] = check
                widget_data["variable"] = var

            elif widget_type == "checklist":
                checklist_frame = ttk.Frame(input_container)
                checklist_frame.pack(anchor=tk.W, padx=20, pady=(0, 5), fill=tk.X)

                values = [str(v) for v in widget_spec.get("values", [])]
                value_aliases = [str(v) for v in widget_spec.get("value_aliases", values)]
                save_to = widget_spec.get("save_to", name)
                selected_values = func_config.get(save_to, [])
                if isinstance(selected_values, str):
                    selected_values = [segment.strip() for segment in selected_values.split(',') if segment.strip()]
                if not isinstance(selected_values, list):
                    selected_values = []
                selected_set = {str(v) for v in selected_values}

                check_vars = []
                for actual_value, display_value in zip(values, value_aliases):
                    var = tk.BooleanVar(value=actual_value in selected_set)
                    chk = ttk.Checkbutton(checklist_frame, text=display_value, variable=var)
                    chk.pack(anchor=tk.W, pady=(0, 2))
                    check_vars.append((var, actual_value, chk))

                def on_checklist_change(*_args, vars_with_values=check_vars, a=instance_alias, field=name, save_field=save_to):
                    selected = [actual for var, actual, _widget in vars_with_values if var.get()]
                    self._save_widget_value(a, field, selected)
                    self._save_widget_value(a, save_field, ",".join(selected))

                for var, _actual, _widget in check_vars:
                    var.trace_add("write", on_checklist_change)

                if name in locked_params:
                    for _var, _actual, chk_widget in check_vars:
                        chk_widget.configure(state="disabled")
                    lock_label = ttk.Label(input_container, text="Swept by loop", font=("Arial", 8, "italic"))
                    lock_label.pack(anchor=tk.W, padx=20, pady=(0, 2))

                on_checklist_change()
                widget_data["widget"] = check_vars
                widget_data["save_to"] = save_to
                widget_data["is_checklist"] = True
                widget_data["checklist_frame"] = checklist_frame
                
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
                        files = filedialog.askopenfilenames(
                            title=self.language_manager.translate("ui.dialogs.select_files_for", "Select files for") + f" {label_text}"
                        )
                        if files:
                            f_widget.delete(0, tk.END)
                            # Store as comma-separated or newline-separated list
                            f_widget.insert(0, ";".join(files))
                            self._save_widget_value(instance_alias, n, f_widget.get())
                    else:
                        file = filedialog.askopenfilename(
                            title=self.language_manager.translate("ui.dialogs.select_file_for", "Select file for") + f" {label_text}"
                        )
                        if file:
                            f_widget.delete(0, tk.END)
                            f_widget.insert(0, file)
                            self._save_widget_value(instance_alias, n, f_widget.get())
                
                browse_btn = ttk.Button(
                    file_frame,
                    text=self.language_manager.translate("ui.buttons.browse", "Browse"),
                    command=lambda n=name, f=file_entry, m=multiple: browse(n, f, m),
                    width=10
                )
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
                        file = filedialog.askopenfilename(
                            title=self.language_manager.translate("ui.dialogs.select_file_for", "Select file for") + f" {lbl} [{idx+1}]"
                        )
                        if file:
                            f_widget.delete(0, tk.END)
                            f_widget.insert(0, file)
                            values_list = [w.get() for w in widgets]
                            self._save_widget_value(a, n, values_list)
                    
                    browse_btn = ttk.Button(item_frame, text=self.language_manager.translate("ui.buttons.browse", "Browse"), 
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
                        files = filedialog.askopenfilenames(
                            title=self.language_manager.translate("ui.dialogs.select_files_for", "Select files for") + f" Sample {idx+1}"
                        )
                        if files:
                            # Update the entry display
                            f_widget.delete(0, tk.END)
                            f_widget.insert(0, "; ".join(files))
                            # Update the stored files list
                            widgets[idx]["files"] = list(files)
                            # Save all sample paths
                            values_list = [w["files"] for w in widgets]
                            self._save_widget_value(a, n, values_list)
                    
                    browse_btn = ttk.Button(item_frame, text=self.language_manager.translate("ui.buttons.browse_more", "Browse..."), 
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
        self._refresh_workflow_setup_dynamic_options(instance_alias, base_alias, visible_widgets)

    def _find_matching_control_end(self, start_idx: int, start_alias: str, end_alias: str) -> int:
        depth = 0
        for idx in range(start_idx, len(self.function_base_aliases)):
            base_alias = self.function_base_aliases[idx]
            if base_alias == start_alias:
                depth += 1
            elif base_alias == end_alias:
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    def _get_loop_body_indices(self, loop_start_idx: int) -> List[int]:
        loop_end_idx = self._find_matching_control_end(loop_start_idx, "workflow_loop_start", "workflow_loop_end")
        if loop_end_idx < 0:
            return []
        return list(range(loop_start_idx + 1, loop_end_idx))

    def _get_swept_param_locks_for_index(self, target_idx: Optional[int]) -> set:
        """Return parameter names controlled by enclosing loop sweep for given function index."""
        locked = set()
        if target_idx is None or target_idx < 0 or target_idx >= len(self.methodology_list):
            return locked

        target_instance = self.methodology_list[target_idx]
        loop_stack: List[int] = []

        for idx in range(target_idx + 1):
            base_alias = self.function_base_aliases[idx]
            if base_alias == "workflow_loop_start":
                loop_stack.append(idx)
            elif base_alias == "workflow_loop_end" and loop_stack:
                loop_stack.pop()

        for loop_idx in loop_stack:
            loop_instance = self.methodology_list[loop_idx]
            loop_cfg = self.function_configs.get(loop_instance, {})
            loop_mode = str(loop_cfg.get("mode", ""))
            if loop_mode not in ("sweep_numeric", "sweep_choice"):
                continue

            sweep_target = str(loop_cfg.get("sweep_target", "") or "").strip()
            if not sweep_target or "." not in sweep_target:
                continue
            target_alias, target_param = sweep_target.split('.', 1)
            if target_alias == target_instance and target_param:
                locked.add(target_param)

        return locked

    def _get_widget_spec_by_name(self, base_alias: str, field_name: str) -> Optional[Dict[str, Any]]:
        config = self.gui_configs.get(base_alias, {})
        layout = config.get("setup", {}).get("layout", [])
        for field in layout:
            if field.get("name") == field_name:
                return field
        return None

    def _set_setup_combobox_options(self, widget_data: Dict[str, Any], actual_values: List[str], display_values: List[str], selected_actual: Optional[str] = None):
        combo = widget_data.get("widget")
        if combo is None:
            return
        alias_to_value = dict(zip(display_values, actual_values))
        value_to_alias = dict(zip(actual_values, display_values))
        combo.configure(values=display_values)
        widget_data["alias_to_value"] = alias_to_value
        widget_data["value_to_alias"] = value_to_alias

        if selected_actual:
            combo.set(value_to_alias.get(selected_actual, selected_actual))
        elif combo.get().strip() and combo.get().strip() in alias_to_value:
            pass
        else:
            combo.set("")

    def _set_setup_checklist_options(self, widget_data: Dict[str, Any], actual_values: List[str], display_values: List[str], selected_actual_values: List[str], instance_alias: str, field_name: str):
        if not widget_data.get("is_checklist"):
            return

        checklist_frame = widget_data.get("checklist_frame")
        if checklist_frame is None:
            return

        for child in checklist_frame.winfo_children():
            child.destroy()

        save_to = widget_data.get("save_to", field_name)
        selected_set = {str(v) for v in (selected_actual_values or [])}
        check_vars = []

        for actual_value, display_value in zip(actual_values, display_values):
            actual_str = str(actual_value)
            var = tk.BooleanVar(value=actual_str in selected_set)
            chk = ttk.Checkbutton(checklist_frame, text=str(display_value), variable=var)
            chk.pack(anchor=tk.W, pady=(0, 2))
            check_vars.append((var, actual_str, chk))

        def on_checklist_change(*_args, vars_with_values=check_vars, a=instance_alias, field=field_name, save_field=save_to):
            selected = [actual for var, actual, _widget in vars_with_values if var.get()]
            self._save_widget_value(a, field, selected)
            self._save_widget_value(a, save_field, ",".join(selected))

        for var, _actual, _widget in check_vars:
            var.trace_add("write", on_checklist_change)

        on_checklist_change()
        widget_data["widget"] = check_vars

    def _refresh_workflow_setup_dynamic_options(self, instance_alias: str, base_alias: str, visible_widgets: Dict[str, Dict[str, Any]]):
        """Populate dynamic combobox options for loop workflow controls."""
        if base_alias != "workflow_loop_start":
            return
        if instance_alias not in self.methodology_list:
            return

        loop_start_idx = self.methodology_list.index(instance_alias)
        body_indices = self._get_loop_body_indices(loop_start_idx)
        if not body_indices:
            return

        current_mode = str(self.function_configs.get(instance_alias, {}).get("mode", "repeat"))

        sweep_targets_actual: List[str] = []
        sweep_targets_display: List[str] = []
        target_meta: Dict[str, Dict[str, Any]] = {}

        benchmark_actual: List[str] = []
        benchmark_display: List[str] = []

        parameter_types = FUNCTION_SPECS.get("parameter_types", {})
        return_specs = FUNCTION_SPECS.get("return_specs", {})

        for idx in body_indices:
            body_instance = self.methodology_list[idx]
            body_base = self.function_base_aliases[idx]
            if self._is_workflow_control(body_base):
                continue

            body_config = self.gui_configs.get(body_base, {})
            body_display = body_config.get("display_name", body_base)
            layout = body_config.get("setup", {}).get("layout", [])
            body_param_types = parameter_types.get(body_base, {})

            for field in layout:
                field_name = field.get("name")
                field_widget = field.get("widget")
                if not field_name:
                    continue
                if field.get("input_type", "user") != "user":
                    continue

                type_name = str(body_param_types.get(field_name, field.get("type", "str"))).lower()
                is_numeric = type_name in ("int", "float") or field.get("type") in ("int", "float")
                is_choice = field_widget == "combobox"

                if current_mode == "sweep_numeric" and not is_numeric:
                    continue
                if current_mode == "sweep_choice" and not is_choice:
                    continue
                if current_mode not in ("sweep_numeric", "sweep_choice"):
                    continue

                target_value = f"{body_instance}.{field_name}"
                target_label = field.get("label", field_name)
                target_alias = f"{body_display} [{body_instance}] · {target_label}"

                sweep_targets_actual.append(target_value)
                sweep_targets_display.append(target_alias)

                choice_values = field.get("values", []) if is_choice else []
                choice_aliases = field.get("value_aliases", choice_values) if is_choice else []
                target_meta[target_value] = {
                    "is_choice": is_choice,
                    "choice_values": [str(v) for v in choice_values],
                    "choice_aliases": [str(v) for v in choice_aliases],
                    "is_numeric": is_numeric
                }

            output_keys = return_specs.get(body_base, [])
            for output_key in output_keys:
                if isinstance(output_key, dict):
                    output_key = output_key.get("key", "")
                if not output_key:
                    continue
                actual = f"{body_instance}.{output_key}"
                display = f"{body_display} [{body_instance}] · {output_key}"
                benchmark_actual.append(actual)
                benchmark_display.append(display)

        sweep_target_data = visible_widgets.get("sweep_target")
        benchmark_data = visible_widgets.get("benchmark_source")
        benchmark_nested_data = visible_widgets.get("benchmark_nested_key")
        sweep_values_data = visible_widgets.get("sweep_values")
        sweep_choice_data = visible_widgets.get("sweep_choice_values")

        selected_target_actual = self.function_configs.get(instance_alias, {}).get("sweep_target", "")

        if sweep_target_data:
            sweep_target_data["target_meta"] = target_meta
            self._set_setup_combobox_options(
                sweep_target_data,
                sweep_targets_actual,
                sweep_targets_display,
                selected_actual=selected_target_actual
            )

        if benchmark_data:
            selected_benchmark = self.function_configs.get(instance_alias, {}).get("benchmark_source", "")
            self._set_setup_combobox_options(
                benchmark_data,
                benchmark_actual,
                benchmark_display,
                selected_actual=selected_benchmark
            )

        if benchmark_nested_data:
            selected_benchmark_source = self.function_configs.get(instance_alias, {}).get("benchmark_source", "")
            selected_nested_key = self.function_configs.get(instance_alias, {}).get("benchmark_nested_key", "")
            nested_actual: List[str] = []
            nested_display: List[str] = []

            if isinstance(selected_benchmark_source, str) and "." in selected_benchmark_source:
                source_instance, source_output = selected_benchmark_source.split(".", 1)

                source_value = None
                source_analysis = self.analysis_data.get(source_instance, {}) if hasattr(self, 'analysis_data') else {}
                source_history = source_analysis.get('execution_history', []) if isinstance(source_analysis, dict) else []
                if source_history and isinstance(source_history, list):
                    latest_snapshot = source_history[-1]
                    if isinstance(latest_snapshot, dict):
                        latest_outputs = latest_snapshot.get('outputs', {})
                        if isinstance(latest_outputs, dict):
                            source_value = latest_outputs.get(source_output)

                if source_value is None and isinstance(source_analysis, dict):
                    fallback_outputs = source_analysis.get('execution_results', {}).get('outputs', {})
                    if isinstance(fallback_outputs, dict):
                        source_value = fallback_outputs.get(source_output)

                def _collect_dict_paths(value, prefix=""):
                    paths = []
                    if isinstance(value, dict):
                        for key, child in value.items():
                            child_prefix = f"{prefix}.{key}" if prefix else str(key)
                            paths.append(child_prefix)
                            paths.extend(_collect_dict_paths(child, child_prefix))
                    return paths

                if isinstance(source_value, dict):
                    nested_actual = _collect_dict_paths(source_value)
                    nested_display = nested_actual.copy()

            self._set_setup_combobox_options(
                benchmark_nested_data,
                nested_actual,
                nested_display,
                selected_actual=str(selected_nested_key) if selected_nested_key is not None else ""
            )

            hint_label = benchmark_nested_data.get("hint_label")
            if not nested_actual:
                if hint_label is None:
                    hint_label = ttk.Label(
                        benchmark_nested_data.get("container"),
                        text=self.language_manager.translate(
                            "ui.messages.run_to_discover_nested_keys",
                            "Run the model once to discover nested keys automatically."
                        ),
                        font=("Arial", 8, "italic")
                    )
                    benchmark_nested_data["hint_label"] = hint_label
                if not hint_label.winfo_ismapped():
                    hint_label.pack(anchor=tk.W, padx=20, pady=(0, 2))
            elif hint_label is not None and hint_label.winfo_exists() and hint_label.winfo_ismapped():
                hint_label.pack_forget()

        if sweep_values_data:
            entry_widget = sweep_values_data.get("widget")
            if entry_widget is not None and hasattr(entry_widget, "configure"):
                entry_widget.configure(state="normal")

        if sweep_choice_data:
            selected_target_actual = self.function_configs.get(instance_alias, {}).get("sweep_target", "")
            target_info = target_meta.get(selected_target_actual, {})
            choice_values = target_info.get("choice_values", []) if target_info.get("is_choice") else []
            choice_aliases = target_info.get("choice_aliases", choice_values) if target_info.get("is_choice") else []
            current_sweep_raw = self.function_configs.get(instance_alias, {}).get("sweep_values", "")
            if isinstance(current_sweep_raw, str):
                selected_choice_values = [segment.strip() for segment in current_sweep_raw.split(',') if segment.strip()]
            elif isinstance(current_sweep_raw, list):
                selected_choice_values = [str(v) for v in current_sweep_raw]
            else:
                selected_choice_values = []

            self._set_setup_checklist_options(
                sweep_choice_data,
                [str(v) for v in choice_values],
                [str(v) for v in choice_aliases],
                selected_choice_values,
                instance_alias,
                "sweep_choice_values"
            )
    
    def _save_widget_value(self, func_alias: str, param_name: str, value: Any):
        """Save widget value to function config."""
        if func_alias not in self.function_configs:
            self.function_configs[func_alias] = {}
        self.function_configs[func_alias][param_name] = value
    
    def _show_help_popup(self, title: str, short_desc: str, long_desc: str):
        """Show a popup window with function help information."""
        popup = tk.Toplevel(self.root)
        popup.title(f"{self.language_manager.translate('ui.dialogs.help_for', 'Help:')} {title}")
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
        close_btn = ttk.Button(popup, text=self.language_manager.translate("ui.buttons.close", "Close"), command=popup.destroy)
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
                            # Simple equality or list membership comparison
                            if isinstance(condition_value, list):
                                # Check if current_value is in the list
                                if current_value not in condition_value:
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

        nested_frame = ttk.Frame(routing_frame)
        nested_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(
            nested_frame,
            text=self.language_manager.translate("ui.messages.routing_nested_key", "Optional source nested key (e.g., metrics.rmse):")
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.routing_nested_key_var = tk.StringVar(value="")
        ttk.Entry(nested_frame, textvariable=self.routing_nested_key_var, width=40).pack(side=tk.LEFT)
        
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
            if self._is_workflow_control(base_alias):
                continue
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
            self._show_fading_warning(
                self.language_manager.translate("ui.messages.same_function_error", "Cannot select the same function on both sides")
            )
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
                "dst_display": dst_display,
                "src_nested_key": self.routing_nested_key_var.get().strip() if hasattr(self, "routing_nested_key_var") else ""
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
            self._show_fading_warning(
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
        self._ensure_analysis_result_selection(instance_alias)
        analysis_info = self.analysis_data[instance_alias]
        
        # Create top control bar
        control_frame = ttk.Frame(self.tab_content_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Get function display name
        config = self.gui_configs.get(base_alias, {})
        display_name = config.get("display_name", base_alias)
        
        title = ttk.Label(
            control_frame,
            text=f"{self.language_manager.translate('ui.tabs.analysis', 'Analysis')}: {display_name}",
            font=("Arial", 11, "bold")
        )
        title.pack(side=tk.LEFT, padx=5)

        history_entries_for_hint = analysis_info.get('execution_history', [])
        has_loop_context = any(
            isinstance(entry, dict) and bool((entry.get('history_context', {}) or {}).get('loop_path'))
            for entry in history_entries_for_hint
        )
        has_parallel_context = any(
            isinstance(entry, dict) and bool((entry.get('history_context', {}) or {}).get('parallel_path'))
            for entry in history_entries_for_hint
        )
        if has_loop_context or has_parallel_context:
            hint_parts = []
            if has_loop_context:
                hint_parts.append("⟳ Loop Context")
            if has_parallel_context:
                hint_parts.append("⎇ Branch Context")
            context_hint = ttk.Label(
                control_frame,
                text="  •  " + " | ".join(hint_parts),
                font=("Arial", 9, "italic")
            )
            context_hint.pack(side=tk.LEFT, padx=3)
        
        # Run to here button
        run_btn = ttk.Button(control_frame, text="🠊 " + self.language_manager.translate("ui.buttons.run_to_here", "Run to here"), 
                            command=lambda: self._run_analysis_to_function(instance_alias))
        run_btn.pack(side=tk.LEFT, padx=5)
        
        # Add graph button
        add_graph_btn = ttk.Button(control_frame, text=self.language_manager.translate("ui.buttons.add_graph", "Add graph"), 
                                   command=lambda: self._show_add_graph_dialog(instance_alias))
        add_graph_btn.pack(side=tk.LEFT, padx=5)
        
        # Add table button
        add_table_btn = ttk.Button(control_frame, text=self.language_manager.translate("ui.buttons.add_table", "Add table"), 
                                   command=lambda: self._show_add_table_dialog(instance_alias))
        add_table_btn.pack(side=tk.LEFT, padx=5)
        
        # Remove section button
        remove_section_btn = ttk.Button(control_frame, text=self.language_manager.translate("ui.buttons.remove_section", "Remove section"), 
                                       command=lambda: self._show_remove_section_dialog(instance_alias))
        remove_section_btn.pack(side=tk.LEFT, padx=5)
        
        # Spacer
        spacer = ttk.Frame(control_frame)
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add page button
        add_page_btn = ttk.Button(control_frame, text=self.language_manager.translate("ui.buttons.add_page", "Add Page"), 
                                 command=lambda: self._show_add_page_dialog(instance_alias))
        add_page_btn.pack(side=tk.RIGHT, padx=5)
        
        # Remove page button
        remove_page_btn = ttk.Button(control_frame, text=self.language_manager.translate("ui.buttons.remove_page", "Remove Page"), 
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
        
        # Navigation frame (bottom): page navigation row + result navigation row
        nav_frame = ttk.Frame(self.tab_content_frame)
        nav_frame.pack(fill=tk.X, padx=10, pady=(0, 10), side=tk.BOTTOM)
        page_nav_row = ttk.Frame(nav_frame)
        page_nav_row.pack(fill=tk.X)
        
        # Page display label (first)
        if visible_pages:
            current_idx, current_page_data = visible_pages[visible_page_idx]
            page_title = current_page_data.get('title', f'Page {current_idx + 1}')
            page_info = f"Page {visible_page_idx + 1}/{len(visible_pages)}: {page_title}"
        else:
            page_info = self.language_manager.translate("ui.messages.no_pages_available", "No pages available")
        
        # Previous page button
        prev_btn = ttk.Button(page_nav_row, text="← " + self.language_manager.translate("ui.buttons.previous", "Previous"), width=10,
                     command=lambda: self._switch_analysis_page_relative(instance_alias, -1))
        prev_btn.pack(side=tk.LEFT, padx=2)
        
        # Next page button
        next_btn = ttk.Button(page_nav_row, text=self.language_manager.translate("ui.buttons.next", "Next") + " →", width=10,
                             command=lambda: self._switch_analysis_page_relative(instance_alias, 1))
        next_btn.pack(side=tk.LEFT, padx=2)
        
        page_label = ttk.Label(page_nav_row, text=page_info, font=("Arial", 9))
        page_label.pack(side=tk.LEFT, padx=10)

        # Result/cycle/branch navigation (under page controls)
        history_entries = analysis_info.get('execution_history', [])
        has_contextual_history = any(
            bool((entry.get('history_context', {}) or {}).get('loop_path')) or
            bool((entry.get('history_context', {}) or {}).get('parallel_path'))
            for entry in history_entries
            if isinstance(entry, dict)
        )
        if history_entries and has_contextual_history:
            result_nav_row = ttk.Frame(nav_frame)
            result_nav_row.pack(fill=tk.X, pady=(6, 0))

            result_prev_btn = ttk.Button(
                result_nav_row,
                text="← " + self.language_manager.translate("ui.buttons.previous", "Previous"),
                width=10,
                command=lambda: self._switch_analysis_result_relative(instance_alias, -1)
            )
            result_prev_btn.pack(side=tk.LEFT, padx=2)

            result_next_btn = ttk.Button(
                result_nav_row,
                text=self.language_manager.translate("ui.buttons.next", "Next") + " →",
                width=10,
                command=lambda: self._switch_analysis_result_relative(instance_alias, 1)
            )
            result_next_btn.pack(side=tk.LEFT, padx=2)

            ttk.Label(
                result_nav_row,
                text=self.language_manager.translate("ui.messages.result_selector", "Cycle/Branch:"),
                font=("Arial", 9)
            ).pack(side=tk.LEFT, padx=(10, 6))

            result_options = [
                self._build_analysis_result_label(entry, idx, len(history_entries))
                for idx, entry in enumerate(history_entries)
            ]
            current_result_idx = analysis_info.get('current_result_idx', 0)
            current_result_idx = max(0, min(current_result_idx, len(history_entries) - 1))
            result_combo = ttk.Combobox(
                result_nav_row,
                state="readonly",
                values=result_options,
                width=52
            )
            result_combo.current(current_result_idx)
            result_combo.bind("<<ComboboxSelected>>", lambda e, ia=instance_alias, cb=result_combo: self._on_analysis_result_selected(ia, cb))
            result_combo.pack(side=tk.LEFT, padx=4)
        
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

    def _ensure_analysis_result_selection(self, instance_alias: str):
        """Ensure execution_results reflects the selected history snapshot for this instance."""
        if instance_alias not in self.analysis_data:
            return

        analysis_info = self.analysis_data[instance_alias]
        history_entries = analysis_info.get('execution_history', [])

        if not history_entries:
            if 'current_result_idx' in analysis_info:
                del analysis_info['current_result_idx']
            return

        current_idx = analysis_info.get('current_result_idx', 0)
        if current_idx < 0:
            current_idx = 0
        if current_idx >= len(history_entries):
            current_idx = len(history_entries) - 1

        analysis_info['current_result_idx'] = current_idx
        selected_snapshot = history_entries[current_idx]
        if isinstance(selected_snapshot, dict):
            analysis_info['execution_results'] = selected_snapshot.copy()

    def _get_sweep_value_display_alias(self, sweep_target: str, sweep_value: Any) -> str:
        """Map a sweep value to the target field alias (value_aliases) when available."""
        if sweep_value is None:
            return ""

        raw_text = str(sweep_value)
        if not isinstance(sweep_target, str) or "." not in sweep_target:
            return raw_text

        target_instance, target_param = sweep_target.split('.', 1)
        if not target_instance or not target_param:
            return raw_text

        try:
            if target_instance not in self.methodology_list:
                return raw_text
            target_idx = self.methodology_list.index(target_instance)
            if target_idx < 0 or target_idx >= len(self.function_base_aliases):
                return raw_text

            target_base_alias = self.function_base_aliases[target_idx]
            field_spec = self._get_widget_spec_by_name(target_base_alias, target_param)
            if not isinstance(field_spec, dict):
                return raw_text

            values = field_spec.get("values", [])
            value_aliases = field_spec.get("value_aliases", values)
            if not isinstance(values, list) or not isinstance(value_aliases, list):
                return raw_text

            value_to_alias = {str(v): str(a) for v, a in zip(values, value_aliases)}
            return value_to_alias.get(raw_text, raw_text)
        except Exception:
            return raw_text

    def _build_analysis_result_label(self, history_entry: dict, position: int, total: int) -> str:
        context = history_entry.get('history_context', {}) if isinstance(history_entry, dict) else {}
        loop_path = context.get('loop_path', []) if isinstance(context, dict) else []
        parallel_path = context.get('parallel_path', []) if isinstance(context, dict) else []

        context_parts = []
        if loop_path:
            loop_segments = []
            for entry in loop_path:
                base_text = f"⟳{entry.get('loop_id', '?')}:C{entry.get('iteration', '?')}"
                sweep_value = entry.get('sweep_value')
                if sweep_value is not None and str(sweep_value) != "":
                    sweep_target = str(entry.get('sweep_target', '') or '')
                    sweep_display = self._get_sweep_value_display_alias(sweep_target, sweep_value)
                    base_text += f" ({sweep_display})"
                loop_segments.append(base_text)
            loop_desc = ", ".join(loop_segments)
            context_parts.append(loop_desc)
        if parallel_path:
            parallel_desc = ", ".join([
                f"⎇{entry.get('parallel_id', '?')}:B{entry.get('branch', '?')}"
                for entry in parallel_path
            ])
            context_parts.append(parallel_desc)

        context_text = " | ".join(context_parts) if context_parts else "Base"
        return f"R{position + 1} — {context_text}"

    def _clear_instance_analysis_runtime_cache(self, instance_alias: str):
        if instance_alias not in self.analysis_data:
            return
        runtime_keys = [
            'graph_slices',
            'table_slices',
            'graph_canvases',
            'graph_control_frames',
            'graph_data_metadata',
            'table_state'
        ]
        for key in runtime_keys:
            if key in self.analysis_data[instance_alias]:
                del self.analysis_data[instance_alias][key]

    def _set_analysis_result_index(self, instance_alias: str, result_idx: int):
        if instance_alias not in self.analysis_data:
            return

        analysis_info = self.analysis_data[instance_alias]
        history_entries = analysis_info.get('execution_history', [])
        if not history_entries:
            return

        result_idx = max(0, min(result_idx, len(history_entries) - 1))
        analysis_info['current_result_idx'] = result_idx
        analysis_info['execution_results'] = history_entries[result_idx].copy()
        self._clear_instance_analysis_runtime_cache(instance_alias)
        self._show_analysis_tab()

    def _switch_analysis_result_relative(self, instance_alias: str, direction: int):
        if instance_alias not in self.analysis_data:
            return

        analysis_info = self.analysis_data[instance_alias]
        history_entries = analysis_info.get('execution_history', [])
        if not history_entries:
            return

        current_idx = analysis_info.get('current_result_idx', 0)
        target_idx = max(0, min(current_idx + direction, len(history_entries) - 1))
        self._set_analysis_result_index(instance_alias, target_idx)

    def _on_analysis_result_selected(self, instance_alias: str, combo_widget: ttk.Combobox):
        try:
            selected_idx = combo_widget.current()
        except Exception:
            selected_idx = -1
        if selected_idx is None or selected_idx < 0:
            return
        self._set_analysis_result_index(instance_alias, selected_idx)
    
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
                placeholder = ttk.Label(container, text=self.language_manager.translate("ui.messages.empty_section", "[Empty Section]"), foreground="gray")
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
                container = ttk.LabelFrame(top_paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
                top_paned.add(container, weight=1)
                containers.append(container)
            
            # Bottom row with horizontal paned window for 2 side-by-side containers
            bottom_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
            main_paned.add(bottom_paned, weight=1)
            for j in range(2):
                container = ttk.LabelFrame(bottom_paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
                bottom_paned.add(container, weight=1)
                containers.append(container)
            
            # Position sashes after rendering
            parent.after_idle(lambda: self._position_fd_sashes(main_paned, top_paned, bottom_paned))
        
        elif layout_type == 'fp':  # Full page (1 section)
            container = ttk.LabelFrame(parent, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
            container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            containers.append(container)
        
        elif layout_type == 'ns':  # North-South (2 sections: top, bottom)
            top_frame = ttk.LabelFrame(parent, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
            top_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            containers.append(top_frame)
            
            bottom_frame = ttk.LabelFrame(parent, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
            bottom_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            containers.append(bottom_frame)
        
        elif layout_type == 'ew':  # East-West (2 sections: left, right)
            paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
            paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            left_container = ttk.LabelFrame(paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
            paned.add(left_container, weight=1)
            containers.append(left_container)
            
            right_container = ttk.LabelFrame(paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
            paned.add(right_container, weight=1)
            containers.append(right_container)
            
            parent.after_idle(lambda: self._position_paned_sash(paned))
        
        elif layout_type == 'sd':  # South Divided (3 sections: 1 top, 2 bottom)
            # Use vertical paned window for top/bottom division
            main_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
            main_paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Top section
            top_frame = ttk.LabelFrame(main_paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
            main_paned.add(top_frame, weight=1)
            containers.append(top_frame)
            
            # Bottom side with horizontal paned window for 2 side-by-side containers
            bottom_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
            main_paned.add(bottom_paned, weight=1)
            for j in range(2):
                container = ttk.LabelFrame(bottom_paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
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
                container = ttk.LabelFrame(top_paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
                top_paned.add(container, weight=1)
                containers.append(container)
            
            # Bottom section
            bottom_frame = ttk.LabelFrame(main_paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
            main_paned.add(bottom_frame, weight=1)
            containers.append(bottom_frame)
        
        elif layout_type == 'ed':  # East Divided (3 sections: 1 left, 2 right stacked)
            paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
            paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Left side (single container)
            left_container = ttk.LabelFrame(paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
            paned.add(left_container, weight=1)
            containers.append(left_container)
            
            # Right side with vertical paned window for 2 stacked containers
            right_paned = ttk.PanedWindow(paned, orient=tk.VERTICAL)
            paned.add(right_paned, weight=1)
            
            for i in range(2):
                container = ttk.LabelFrame(right_paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
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
                container = ttk.LabelFrame(left_paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
                left_paned.add(container, weight=1)
                containers.append(container)
            
            # Right side (single container)
            right_container = ttk.LabelFrame(paned, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
            paned.add(right_container, weight=1)
            containers.append(right_container)
            
            parent.after_idle(lambda: self._position_paned_sash(paned))
        
        else:
            # Default to full page for unknown layouts
            container = ttk.LabelFrame(parent, text=self.language_manager.translate("ui.labels.section", "Section"), padding=section_padding)
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
            label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.empty", "[Empty]"), foreground="gray")
            label.pack(expand=True)
        
        # Add popup button AFTER content is rendered (only if not already in a popup)
        if not is_popup:
            self._create_section_popup_button(parent, instance_alias, section_idx, section_data)
    
    def _resolve_axis_label(self, axis_config: dict, outputs: dict, axis_index: Optional[int] = None) -> str:
        """Resolve an axis label from axis configuration.
        
        The label configuration supports:
        - Direct string: "label": "My Axis"
        - Variable reference: "label": "variable_name"
        - Variable with index: "label": "variable_name", "l_index": 0
        - Dynamic labels from array: "axis_labels": "pc_labels" (uses axis_index if provided)
        
        If the variable is a list and l_index is provided, returns the value at that index.
        For axis_labels with dynamic indexing, uses the axis_index parameter if provided.
        Otherwise, returns the variable value as a string.
        
        Args:
            axis_config: Axis configuration dict with 'label' or 'axis_labels' field
            outputs: Dictionary of execution outputs
            axis_index: Optional current axis index for dynamic label selection
            
        Returns:
            Resolved label as string, or empty string if not found
        """
        if not axis_config:
            return ""
        
        # Check for axis_labels (dynamic labels based on current index)
        axis_labels_config = axis_config.get('axis_labels')
        axis_labels_nested = axis_config.get('axis_labels_nested')
        if axis_labels_config and axis_index is not None:
            # axis_labels points to a variable containing an array of labels
            # Use helper to support nested dictionary access
            labels_data = self._get_data_from_source(outputs, axis_labels_config, axis_labels_nested)
            if labels_data is not None:
                if isinstance(labels_data, (list, np.ndarray)):
                    try:
                        return str(labels_data[axis_index])
                    except (IndexError, TypeError):
                        pass
        
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
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), foreground="gray")
                label.pack(expand=True)
                return
            
            execution_results = self.analysis_data[instance_alias].get('execution_results', {})
            if not execution_results:
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), foreground="gray")
                label.pack(expand=True)
                return
            
            if execution_results.get('status') != 'success':
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.execution_failed_check_log", "Execution failed. Check model_log.txt for details."), foreground="red")
                label.pack(expand=True)
                return
            
            outputs = self._get_execution_data_sources(execution_results, instance_alias)
            
            # Initialize slice state if needed
            # Use (page_index, section_idx) tuple as key to ensure each graph has unique state per page
            current_page = self.analysis_data[instance_alias].get('current_page', 0)
            section_id = (current_page, section_idx)
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
                data_check = self._get_data_from_source(outputs, data_source_check) if data_source_check else None
                if data_check is not None:
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
                
                data_temp = self._get_data_from_source(outputs, data_source_temp) if data_source_temp else None
                if data_temp is not None:
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
                    label_data = self._get_data_from_source(outputs, labels_config)
                    if label_data is not None:
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
                
                # For multi-dataset scatter plots with no top-level axes, get them from datasets
                datasets_config = config.get('datasets')
                if datasets_config and isinstance(datasets_config, list) and len(datasets_config) > 0:
                    if not x_axis_config and not y_axis_config:
                        # Get axis configs from first dataset that has them
                        for dataset_cfg in datasets_config:
                            if not x_axis_config and 'x_axis' in dataset_cfg:
                                x_axis_config = dataset_cfg['x_axis']
                            if not y_axis_config and 'y_axis' in dataset_cfg:
                                y_axis_config = dataset_cfg['y_axis']
            
            # Resolve axis labels from variables if needed
            # Make copies to avoid modifying original configs
            x_axis_config = x_axis_config.copy() if x_axis_config else {}
            y_axis_config = y_axis_config.copy() if y_axis_config else {}
            
            # For axis selection mode, get the current axis indices
            # axis_indices structure: {'x': {dimension: index}, 'y': {dimension: index}}
            # We need to extract the actual index value for each axis
            x_axis_idx = None
            y_axis_idx = None
            z_axis_idx = None
            
            # Find dimension for each axis from data_slicing config
            nav_axes = config.get('data_slicing', [])
            x_dimension = None
            y_dimension = None
            z_dimension = None
            
            for nav_item in nav_axes:
                if isinstance(nav_item, dict):
                    target_axis = nav_item.get('axis')
                    dimension = nav_item.get('dimension')
                    if target_axis == 'x' and dimension is not None:
                        x_dimension = dimension
                    elif target_axis == 'y' and dimension is not None:
                        y_dimension = dimension
                    elif target_axis == 'z' and dimension is not None:
                        z_dimension = dimension
            
            # Extract the actual index values from axis_indices
            if 'x' in axis_indices and x_dimension is not None:
                x_axis_idx = axis_indices['x'].get(x_dimension)
            if 'y' in axis_indices and y_dimension is not None:
                y_axis_idx = axis_indices['y'].get(y_dimension)
            if 'z' in axis_indices and z_dimension is not None:
                z_axis_idx = axis_indices['z'].get(z_dimension)
            
            resolved_x_label = self._resolve_axis_label(x_axis_config, outputs, axis_index=x_axis_idx)
            if resolved_x_label:
                x_axis_config['label'] = resolved_x_label
            
            resolved_y_label = self._resolve_axis_label(y_axis_config, outputs, axis_index=y_axis_idx)
            if resolved_y_label:
                y_axis_config['label'] = resolved_y_label
            
            # Also resolve z-axis label if present
            z_axis_config = config.get('z_axis', {})
            if z_axis_config:
                z_axis_config = z_axis_config.copy()
                resolved_z_label = self._resolve_axis_label(z_axis_config, outputs, axis_index=z_axis_idx)
                if resolved_z_label:
                    z_axis_config['label'] = resolved_z_label
            
            # Merge axis indices for y (base + md + axis-specific)
            # Extract y FIRST so __index__ on x-axis can use y_data as reference
            y_indices = base_indices.copy()
            y_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'y' in axis_indices:
                y_indices.update(axis_indices['y'])
            y_data = self._extract_axis_data(outputs, y_axis_config, y_indices)
            
            # Merge axis indices for x (base + md + axis-specific)
            x_indices = base_indices.copy()
            x_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'x' in axis_indices:
                x_indices.update(axis_indices['x'])
            # Pass y_data as reference for row index generation if needed
            x_data = self._extract_axis_data(outputs, x_axis_config, x_indices, ref_data=y_data)
            
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
            # Pass x_data or y_data as reference for row index generation if needed
            z_data = self._extract_axis_data(outputs, config.get('z_axis', {}), z_indices, ref_data=x_data if x_data is not None else y_data)
            
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
            
            # Handle multiple datasets if configured
            datasets_config = config.get('datasets')
            extracted_datasets = None
            if datasets_config and isinstance(datasets_config, list):
                extracted_datasets = []
                for dataset_idx, dataset_cfg in enumerate(datasets_config):
                    dataset_label = dataset_cfg.get('label', f'Dataset {dataset_idx + 1}')
                    
                    # Check dataset condition if provided (optional filtering)
                    if 'condition' in dataset_cfg:
                        condition = dataset_cfg['condition']
                        param_name = condition.get('parameter')
                        operator = condition.get('operator')
                        expected_value = condition.get('value')
                        
                        # Evaluate condition against execution inputs
                        exec_inputs = execution_results.get('inputs', {})
                        if param_name and param_name in exec_inputs:
                            actual_value = exec_inputs[param_name]
                            include_dataset = self._evaluate_condition(actual_value, operator, expected_value)
                            if not include_dataset:
                                # Dataset filtered by condition - skip it
                                continue
                    
                    # Extract data for this dataset using same indices as main axes
                    ds_x_axis = dataset_cfg.get('x_axis', {})
                    ds_y_axis = dataset_cfg.get('y_axis', {})
                    ds_z_axis = dataset_cfg.get('z_axis', {})
                    
                    # Use same indices for consistency with main plot
                    ds_x_data = self._extract_axis_data(outputs, ds_x_axis, x_indices)
                    # Pass ds_x_data as reference for row index generation if needed
                    ds_y_data = self._extract_axis_data(outputs, ds_y_axis, y_indices, ref_data=ds_x_data)
                    # Pass ds_x_data or ds_y_data as reference for row index generation if needed
                    ds_z_data = self._extract_axis_data(outputs, ds_z_axis, z_indices, ref_data=ds_x_data if ds_x_data is not None else ds_y_data) if ds_z_axis else None
                    
                    # Skip if required data sources don't exist (normal for optional datasets)
                    if ds_x_data is None or ds_y_data is None:
                        continue
                    
                    # Extract class labels if specified
                    ds_class_data = None
                    if 'class_labels' in dataset_cfg:
                        class_source = dataset_cfg['class_labels']
                        class_val = self._get_data_from_source(outputs, class_source)
                        if class_val is not None:
                            if isinstance(class_val, (list, np.ndarray)):
                                ds_class_data = np.array(class_val)
                    
                    # Dataset is valid and will be rendered
                    dataset_entry = {
                        'x_data': ds_x_data,
                        'y_data': ds_y_data,
                        'label': dataset_label,
                        'marker': dataset_cfg.get('marker', 'o'),
                        'x_axis': ds_x_axis,  # Preserve axis config for label extraction
                        'y_axis': ds_y_axis   # Preserve axis config for label extraction
                    }
                    if ds_z_data is not None:
                        dataset_entry['z_data'] = ds_z_data
                    if ds_class_data is not None:
                        dataset_entry['class_data'] = ds_class_data
                    # Include color if specified (used as fallback when no class_data)
                    if 'color' in dataset_cfg:
                        dataset_entry['color'] = dataset_cfg['color']
                    # Include sample_labels_source if specified per-dataset
                    if 'sample_labels_source' in dataset_cfg:
                        dataset_entry['sample_labels_source'] = dataset_cfg['sample_labels_source']
                    extracted_datasets.append(dataset_entry)
            
            # If main plot has class_labels config, treat it as a dataset for proper class coloring with qualitative colormap
            if 'class_labels' in config and graph_type == 'scatter' and x_data is not None and y_data is not None:
                class_source = config['class_labels']
                class_val = self._get_data_from_source(outputs, class_source)
                if class_val is not None:
                    if isinstance(class_val, (list, np.ndarray)):
                        main_class_data = np.array(class_val)
                        
                        # Create extracted_datasets if it doesn't exist
                        if extracted_datasets is None:
                            extracted_datasets = []
                        
                        # Build main dataset with class data
                        main_dataset = {
                            'x_data': x_data,
                            'y_data': y_data,
                            'label': 'Main Dataset',
                            'marker': 'o',
                            'class_data': main_class_data
                        }
                        if z_data is not None:
                            main_dataset['z_data'] = z_data
                        # Include sample_labels_source if specified at config level
                        if 'sample_labels_source' in config:
                            main_dataset['sample_labels_source'] = config['sample_labels_source']
                        
                        # Add main dataset at the beginning of the list
                        extracted_datasets.insert(0, main_dataset)
                        
                        # Clear x/y/z data so they won't conflict with multi-dataset rendering
                        x_data = None
                        y_data = None
                        z_data = None
            
            # Extract sample labels for tooltip display from individual datasets
            sample_labels = None
            sample_labels_by_dataset = None
            
            # If we have extracted datasets, collect their sample_labels_source
            if extracted_datasets and len(extracted_datasets) > 0:
                sample_labels_by_dataset = {}
                for dataset_entry in extracted_datasets:
                    ds_label = dataset_entry.get('label')
                    ds_source = dataset_entry.get('sample_labels_source')
                    if ds_label and ds_source and ds_source in outputs:
                        labels_data = outputs[ds_source]
                        if isinstance(labels_data, (list, np.ndarray)):
                            sample_labels_by_dataset[ds_label] = [str(lbl) for lbl in labels_data]
            else:
                # For single dataset, check for sample_labels_source in config
                sample_labels_source = config.get('sample_labels_source')
                if isinstance(sample_labels_source, str):
                    # Single sample labels source
                    if sample_labels_source in outputs:
                        labels_data = outputs[sample_labels_source]
                        if isinstance(labels_data, (list, np.ndarray)):
                            sample_labels = [str(lbl) for lbl in labels_data]
            
            # Fall back to default names if no explicit source found
            if not sample_labels and not sample_labels_by_dataset:
                for label_key in ['smp_cal', 'sample_labels', 'smp_path', 'sample_names']:
                    if label_key in outputs:
                        labels_data = outputs[label_key]
                        if isinstance(labels_data, (list, np.ndarray)):
                            sample_labels = [str(lbl) for lbl in labels_data]
                            break
            
            # Render graph using graph_renderer module
            fig, ax = graph_renderer.render_graph_figure(
                graph_type, render_config, x_data, y_data, z_data, x_axis_config, y_axis_config,
                default_cmap=self.settings_manager.get('colormap', 'viridis'),
                datasets=extracted_datasets,
                qualitative_cmap=self.settings_manager.get('qualitative_colormap', 'tab10'),
                sample_labels=sample_labels,
                sample_labels_by_dataset=sample_labels_by_dataset
            )
            
            # Embed figure in tkinter within a managed frame
            canvas, canvas_frame = graph_renderer.embed_figure_in_tkinter(fig, parent)
            
            # Store canvas reference for updates
            if 'graph_canvases' not in self.analysis_data[instance_alias]:
                self.analysis_data[instance_alias]['graph_canvases'] = {}
            self.analysis_data[instance_alias]['graph_canvases'][section_id] = (canvas, canvas_frame)
            
            # Store graph data metadata for CSV export
            if 'graph_data_metadata' not in self.analysis_data[instance_alias]:
                self.analysis_data[instance_alias]['graph_data_metadata'] = {}
            
            # For aux_axis configs, store the data source name so we can re-extract during export
            z_data_source = None
            if 'aux_axis' in config:
                z_data_source = config.get('z_axis', {}).get('data_source')
            elif config.get('graph_type') in ('heatmap', '3d_surf', 'contour'):
                z_data_source = config.get('z_axis', {}).get('data_source')
            
            self.analysis_data[instance_alias]['graph_data_metadata'][section_id] = {
                'x_data': x_data.copy() if isinstance(x_data, np.ndarray) else x_data,
                'y_data': y_data.copy() if isinstance(y_data, np.ndarray) else y_data,
                'z_data': z_data.copy() if isinstance(z_data, np.ndarray) else z_data,
                'z_data_source': z_data_source,  # Store source name for re-extraction during export
                'x_axis_config': x_axis_config.copy() if x_axis_config else {},
                'y_axis_config': y_axis_config.copy() if y_axis_config else {},
                'z_axis_config': z_axis_config.copy() if z_axis_config else {},
                'extracted_datasets': extracted_datasets,  # Could contain references but will be used as-is
                'graph_type': graph_type,
                'graph_title': config.get('graph_title', config.get('title', 'Graph')),
                'outputs': outputs,  # Store outputs for accessing any additional data sources
                'config': config  # Store config for accessing class_labels and other metadata
            }
            
        except Exception as e:
            label = ttk.Label(
                parent,
                text=self.language_manager.translate("ui.messages.error_rendering_graph", "Error rendering graph:") + f" {str(e)}",
                foreground="red"
            )
            label.pack(expand=True)
    
    def _get_data_from_source(self, outputs: dict, data_source: str, nested_key: str = None) -> Any:
        """Extract data from a source, supporting nested dictionary access and special markers.
        
        Args:
            outputs: Dictionary of execution outputs
            data_source: Key to access in outputs (e.g., 'metrics', 'model_results')
                        Can also be special markers:
                        - "__index__": Auto-generated row indices (requires reference_source)
                        - "row_index": Alias for __index__
            nested_key: Optional key for nested dictionary access (e.g., 'pct_variance_explained')
                       Can be a single key or dot-separated path (e.g., 'stats.mean')
        
        Returns:
            The extracted data, or None if not found
        """
        # Handle special index markers
        if data_source in ("__index__", "row_index"):
            # This will be handled specially - return marker to caller
            return "__index__"
        
        # Get the base data from outputs with prefixed/unprefixed compatibility
        if data_source not in outputs:
            source_key = str(data_source) if data_source is not None else ""
            fallback_keys = []
            if source_key.startswith('in.') or source_key.startswith('out.'):
                fallback_keys.append(source_key.split('.', 1)[1])
            elif source_key:
                fallback_keys.extend([f"out.{source_key}", f"in.{source_key}"])

            data = None
            found = False
            for key in fallback_keys:
                if key in outputs:
                    data = outputs[key]
                    found = True
                    break
            if not found:
                return None
        else:
            data = outputs[data_source]
        
        # If no nested key, return the base data
        if not nested_key:
            return data
        
        # Handle nested access for dictionaries
        if isinstance(data, dict):
            # Support dot notation for nested paths (e.g., "stats.mean")
            keys_path = nested_key.split('.') if '.' in nested_key else [nested_key]
            
            try:
                for key in keys_path:
                    if isinstance(data, dict) and key in data:
                        data = data[key]
                    else:
                        return None
                return data
            except (KeyError, TypeError, AttributeError):
                return None
        
        # If data isn't a dict but nested_key was specified, try list/array indexing
        if isinstance(nested_key, str) and nested_key.isdigit():
            try:
                idx = int(nested_key)
                if isinstance(data, (list, np.ndarray)) and 0 <= idx < len(data):
                    return data[idx]
            except (ValueError, TypeError, IndexError):
                pass
        
        return None

    def _get_execution_data_sources(self, execution_results: dict, instance_alias: str = None) -> dict:
        """Get combined execution data sources (inputs + routed inputs + outputs).

        Precedence on key collisions: direct inputs < routed inputs < outputs.
        """
        combined_sources = {}
        try:
            if not isinstance(execution_results, dict):
                return combined_sources

            inputs = execution_results.get('inputs', {})
            outputs = execution_results.get('outputs', {})
            inherited_inputs = {}
            if instance_alias:
                try:
                    inherited_inputs = self._resolve_inherited_upstream_outputs(instance_alias, execution_results)
                except Exception:
                    inherited_inputs = {}

            def _add_prefixed_aliases() -> None:
                if isinstance(inputs, dict):
                    for key, value in inputs.items():
                        combined_sources[f"in.{key}"] = value
                    if instance_alias and instance_alias in inputs and isinstance(inputs[instance_alias], dict):
                        for key, value in inputs[instance_alias].items():
                            combined_sources[f"in.{key}"] = value

                if instance_alias:
                    try:
                        routed_inputs = self._resolve_routed_inputs(instance_alias)
                        if isinstance(routed_inputs, dict):
                            for key, value in routed_inputs.items():
                                combined_sources[f"in.{key}"] = value
                    except Exception:
                        pass

                if isinstance(inherited_inputs, dict):
                    for key, value in inherited_inputs.items():
                        combined_sources.setdefault(f"in.{key}", value)

                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        combined_sources[f"out.{key}"] = value
                    if instance_alias and instance_alias in outputs and isinstance(outputs[instance_alias], dict):
                        for key, value in outputs[instance_alias].items():
                            combined_sources[f"out.{key}"] = value

                for key, value in list(combined_sources.items()):
                    if not isinstance(key, str):
                        continue
                    if key.startswith('in.') or key.startswith('out.'):
                        continue
                    combined_sources.setdefault(f"in.{key}", value)

            if isinstance(inputs, dict):
                combined_sources.update(inputs)
                if instance_alias and instance_alias in inputs and isinstance(inputs[instance_alias], dict):
                    combined_sources.update(inputs[instance_alias])

            if instance_alias:
                try:
                    combined_sources.update(self._resolve_routed_inputs(instance_alias))
                except Exception:
                    pass

            if isinstance(inherited_inputs, dict):
                for key, value in inherited_inputs.items():
                    combined_sources.setdefault(key, value)

            if isinstance(outputs, dict):
                combined_sources.update(outputs)
                if instance_alias and instance_alias in outputs and isinstance(outputs[instance_alias], dict):
                    combined_sources.update(outputs[instance_alias])

            _add_prefixed_aliases()

            return combined_sources
        except Exception:
            return combined_sources

    def _resolve_routed_inputs(self, instance_alias: str) -> dict:
        """Resolve routed input values for a function instance from upstream execution outputs."""
        resolved_inputs = {}

        if instance_alias not in self.methodology_list:
            return resolved_inputs

        dst_idx = self.methodology_list.index(instance_alias)

        for key, routing_info in self.routing_lines.items():
            try:
                src_idx = None
                src_param_key = None
                dst_idx_key = None
                dst_param_key = None

                if isinstance(key, tuple) and len(key) >= 4:
                    src_idx, src_param_key, dst_idx_key, dst_param_key = key[:4]
                elif isinstance(routing_info, dict):
                    src_idx = routing_info.get('src_idx')
                    src_param_key = routing_info.get('src_param_key')
                    dst_idx_key = routing_info.get('dst_idx')
                    dst_param_key = routing_info.get('dst_param_key')

                try:
                    src_idx = int(src_idx) if src_idx is not None else None
                    dst_idx_key = int(dst_idx_key) if dst_idx_key is not None else None
                except (TypeError, ValueError):
                    continue

                if dst_idx_key != dst_idx or src_idx is None or src_param_key is None or dst_param_key is None:
                    continue

                if src_idx < 0 or src_idx >= len(self.methodology_list):
                    continue

                src_instance_alias = self.methodology_list[src_idx]
                src_exec = self.analysis_data.get(src_instance_alias, {}).get('execution_results', {})
                if src_exec.get('status') != 'success':
                    continue

                src_outputs = src_exec.get('outputs', {})
                src_inputs = src_exec.get('inputs', {})
                src_nested_key = routing_info.get('src_nested_key', '') if isinstance(routing_info, dict) else ''

                def _extract_nested(value, nested_key: str):
                    if not nested_key:
                        return value
                    current = value
                    for part in str(nested_key).split('.'):
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            return None
                    return current

                if isinstance(src_outputs, dict) and src_param_key in src_outputs:
                    extracted_value = _extract_nested(src_outputs[src_param_key], src_nested_key)
                    if extracted_value is not None:
                        resolved_inputs[dst_param_key] = extracted_value
                elif isinstance(src_inputs, dict) and src_param_key in src_inputs:
                    extracted_value = _extract_nested(src_inputs[src_param_key], src_nested_key)
                    if extracted_value is not None:
                        resolved_inputs[dst_param_key] = extracted_value
            except Exception:
                continue

        return resolved_inputs

    def _resolve_inherited_upstream_outputs(self, instance_alias: str, target_execution_results: Optional[dict] = None) -> dict:
        """Resolve inherited upstream outputs for contextual analysis rendering.

        Uses the best matching upstream execution snapshot for the target history context.
        """
        inherited = {}

        if instance_alias not in self.methodology_list:
            return inherited

        dst_idx = self.methodology_list.index(instance_alias)
        target_context = {}
        if isinstance(target_execution_results, dict):
            target_context = target_execution_results.get('history_context', {}) or {}

        def _pick_contextual_execution(src_instance_alias: str) -> dict:
            analysis_info = self.analysis_data.get(src_instance_alias, {})
            history_entries = analysis_info.get('execution_history', [])
            if not isinstance(history_entries, list) or not history_entries:
                exec_result = analysis_info.get('execution_results', {})
                return exec_result if isinstance(exec_result, dict) else {}

            target_loop = target_context.get('loop_path', []) if isinstance(target_context, dict) else []
            target_parallel = target_context.get('parallel_path', []) if isinstance(target_context, dict) else []

            def _score(entry: dict) -> int:
                if not isinstance(entry, dict):
                    return -1
                ctx = entry.get('history_context', {}) or {}
                loop_path = ctx.get('loop_path', []) if isinstance(ctx, dict) else []
                parallel_path = ctx.get('parallel_path', []) if isinstance(ctx, dict) else []

                if loop_path == target_loop and parallel_path == target_parallel:
                    return 4
                if (
                    isinstance(loop_path, list)
                    and isinstance(target_loop, list)
                    and len(loop_path) <= len(target_loop)
                    and target_loop[:len(loop_path)] == loop_path
                    and parallel_path == target_parallel
                ):
                    return 3
                if (
                    isinstance(parallel_path, list)
                    and isinstance(target_parallel, list)
                    and len(parallel_path) <= len(target_parallel)
                    and target_parallel[:len(parallel_path)] == parallel_path
                ):
                    return 2
                if not loop_path and not parallel_path:
                    return 1
                return 0

            best_entry = None
            best_score = -1
            for entry in history_entries:
                score = _score(entry)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if isinstance(best_entry, dict):
                return best_entry

            exec_result = analysis_info.get('execution_results', {})
            return exec_result if isinstance(exec_result, dict) else {}

        for src_idx in range(dst_idx):
            try:
                if hasattr(self, '_can_auto_route_between') and not self._can_auto_route_between(src_idx, dst_idx):
                    continue

                src_instance_alias = self.methodology_list[src_idx]
                src_exec = _pick_contextual_execution(src_instance_alias)
                if src_exec.get('status') != 'success':
                    continue

                src_outputs = src_exec.get('outputs', {})
                if isinstance(src_outputs, dict):
                    inherited.update(src_outputs)
            except Exception:
                continue

        return inherited
    
    def _extract_axis_data(self, outputs: dict, axis_config: dict, indices: dict = None, ref_data: np.ndarray = None) -> Optional[np.ndarray]:
        """Extract data for an axis from execution outputs, supporting nested dictionary access and row indices.
        
        Args:
            outputs: Dictionary of output data
            axis_config: Config for this axis (including data_source and optional nested_key)
                        Special data_source values:
                        - "__index__": Auto-generate row indices (0, 1, 2, ...)
                        - "row_index": Alias for __index__
            indices: Dictionary mapping dimension to index for slicing (e.g., {0: 5, 1: 2})
                    or an integer for backward compatibility
            ref_data: Optional reference data to infer length from (used for __index__ if no reference_source)
        """
        if not axis_config:
            return None
        
        data_source = axis_config.get('data_source')
        if not data_source:
            return None
        
        # Handle special row index marker
        if data_source in ("__index__", "row_index"):
            # Get length from reference_data, or reference_source, or explicit reference_length
            length = None
            
            # Try reference_data first (passed from calling function)
            if ref_data is not None:
                # Convert to numpy array if needed to safely get length
                if not isinstance(ref_data, np.ndarray):
                    try:
                        ref_data = np.array(ref_data)
                    except (ValueError, TypeError):
                        ref_data = None
                
                # Get length from ref_data if it's a valid array
                if ref_data is not None and hasattr(ref_data, 'shape'):
                    if ref_data.ndim > 0:  # Must be at least 1D
                        length = ref_data.shape[0]
            
            # If still no length, try reference_source from config
            if length is None:
                reference_source = axis_config.get('reference_source')
                if reference_source:
                    reference_nested = axis_config.get('reference_nested_key')
                    ref_data = self._get_data_from_source(outputs, reference_source, reference_nested)
                    if ref_data is not None:
                        if not isinstance(ref_data, np.ndarray):
                            ref_data = np.array(ref_data)
                        length = ref_data.shape[0]
                else:
                    # Try to get length from axis_config reference_length field
                    if 'reference_length' in axis_config:
                        length = axis_config.get('reference_length')
            
            if length is None:
                # No length information provided
                return None
            
            # Generate row indices: [1, 2, 3, ..., length] (1-based for user-friendliness)
            data = np.arange(1, length + 1)
        else:
            # Use helper to get data, supporting nested_key for dict access
            nested_key = axis_config.get('nested_key')
            data = self._get_data_from_source(outputs, data_source, nested_key)
            
            if data is None:
                return None
        
        # Track if this is a list source (coordinates/axes) vs array source (data)
        # A list source contains multiple arrays (one per axis/dimension)
        # A data source is a flat list or array of values
        is_list_source = False
        if isinstance(data, list) and len(data) > 0:
            # Check if first element is itself an array or list (list of arrays)
            first_elem = data[0]
            is_list_source = isinstance(first_elem, (list, np.ndarray))
        
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
    
    def _on_table_navigate_slice(self, instance_alias: str, section_id: tuple, direction: int,
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
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), foreground="gray")
                label.pack(expand=True)
                return
            
            execution_results = self.analysis_data[instance_alias].get('execution_results', {})
            if not execution_results:
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), foreground="gray")
                label.pack(expand=True)
                return
            
            if execution_results.get('status') != 'success':
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.execution_failed_check_log", "Execution failed. Check model_log.txt for details."), foreground="red")
                label.pack(expand=True)
                return
            
            outputs = self._get_execution_data_sources(execution_results, instance_alias)
            
            # Initialize col_headers - will be set based on configuration
            col_headers = None
            
            # Check if using multi-column configuration (new feature)
            columns_config = config.get('columns')
            if columns_config:
                # Extract multiple columns from different sources/keys
                data_columns = []
                col_headers = []
                
                for col_spec in columns_config:
                    col_data_source = col_spec.get('data_source')
                    col_nested_key = col_spec.get('nested_key')
                    col_name = col_spec.get('name', col_data_source)
                    
                    # Extract column data
                    col_data = self._get_data_from_source(outputs, col_data_source, col_nested_key)
                    
                    if col_data is None:
                        label = ttk.Label(
                            parent,
                            text=self.language_manager.translate("ui.messages.column_data_source_not_found", "Column data source not found:") + f" '{col_data_source}'",
                            foreground="red"
                        )
                        label.pack(expand=True)
                        return
                    
                    # Convert to numpy array if needed
                    if not isinstance(col_data, np.ndarray):
                        col_data = np.array(col_data)
                    
                    # Ensure 1D
                    col_data = col_data.flatten()
                    
                    data_columns.append(col_data)
                    col_headers.append(col_name)
                
                # Stack columns into a 2D array
                try:
                    data = np.column_stack(data_columns)
                except ValueError:
                    label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.column_lengths_mismatch", "Column lengths do not match. All columns must have the same length."), foreground="red")
                    label.pack(expand=True)
                    return
            else:
                # Original single-column extraction
                data_source = config.get('data_source')
                nested_key = config.get('nested_key')
                
                # Use helper to extract data, supporting nested dictionary access
                data = self._get_data_from_source(outputs, data_source, nested_key)
                
                if data is None:
                    label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.data_source_not_found_generic", "Data source not found"), foreground="red")
                    label.pack(expand=True)
                    return
                
                # Convert to numpy array if needed
                if not isinstance(data, np.ndarray):
                    data = np.array(data)
                
                # Get col_headers from config if provided
                col_headers = config.get('column_headers')
            
            # Initialize table slices state for data slicing support
            # Use (page, section_idx) tuple as section ID to ensure unique state per page
            current_page = self.analysis_data[instance_alias].get('current_page', 0)
            section_id = (current_page, section_idx)
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
            export_btn = ttk.Button(toolbar, text=self.language_manager.translate("ui.buttons.export_csv", "Export to CSV"), 
                                   command=lambda: self._export_table_to_csv(data, export_title))
            export_btn.pack(side=tk.LEFT, padx=2)
            
            # Statistics button - use table_title > section title > data_source
            stats_title = table_title or title or data_source
            stats_btn = ttk.Button(toolbar, text=self.language_manager.translate("ui.buttons.show_statistics", "Show Statistics"),
                                  command=lambda: self._show_table_statistics(data, stats_title))
            stats_btn.pack(side=tk.LEFT, padx=2)
            
            # Refresh button
            refresh_btn = ttk.Button(toolbar, text=self.language_manager.translate("ui.buttons.refresh", "Refresh"),
                                    command=lambda: self._refresh_table(instance_alias, section_id))
            refresh_btn.pack(side=tk.LEFT, padx=2)
            
            # Create table view
            self._create_table_view(main_frame, data, config, decimal_places, 
                                   max_rows, max_cols, col_headers, row_headers)
            
        except Exception as e:
            import traceback
            label = ttk.Label(
                parent,
                text=self.language_manager.translate("ui.messages.error_rendering_table", "Error rendering table:") + f" {str(e)}",
                foreground="red"
            )
            label.pack(expand=True)
            traceback.print_exc()
    
    def _create_table_view(self, parent: ttk.Frame, data: np.ndarray, config: dict,
                          decimal_places: int, max_rows: int, max_cols: int,
                          col_headers: list = None, row_headers: list = None) -> None:
        """Create the actual table view with scrollbars and formatting."""
        try:
            # Check if data is still 3D+ (shouldn't happen if slicing is configured, but safety check)
            if data.ndim > 2:
                error_msg = self.language_manager.translate("ui.messages.cannot_display_nd_table", "Cannot display {ndim}D data in table view.").format(ndim=data.ndim) + "\n"
                error_msg += self.language_manager.translate("ui.messages.data_shape", "Data shape:") + f" {data.shape}\n\n"
                error_msg += self.language_manager.translate("ui.messages.ensure_data_slicing", "This should have been caught earlier. Ensure data_slicing is configured.")
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
            
            # Configure row header column with configurable label
            row_label_header = config.get('row_label', 'Row')  # Default to 'Row' if not specified
            tree.column('#0', width=50, anchor='center')
            tree.heading('#0', text=row_label_header)
            
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
                    row_label = str(row_idx + 1)  # Display 1-based index for user
                
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
            label = ttk.Label(
                parent,
                text=self.language_manager.translate("ui.messages.error_creating_table_view", "Error creating table view:") + f" {str(e)}",
                foreground="red"
            )
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
                title=self.language_manager.translate("ui.dialogs.export_table_csv", "Export Table to CSV")
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
            
            self._show_fading_success(
                "✅ " + self.language_manager.translate("ui.messages.table_exported_to", "Table exported to:") + f"\n{filepath}"
            )
        except Exception as e:
            self._show_fading_error(
                "❌ " + self.language_manager.translate("ui.messages.table_export_error", "Error exporting table:") + f" {str(e)}"
            )
    
    def _show_table_statistics(self, data: np.ndarray, title: str = 'Statistics') -> None:
        """Display statistical summary of table data."""
        try:
            stats_window = tk.Toplevel(self.root)
            stats_window.title(f"{title} - {self.language_manager.translate('ui.labels.statistics', 'Statistics')}")
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
            popup.title(
                self.language_manager.translate("ui.labels.section_prefix", "Section:") +
                f" {section_data.get('config', {}).get('title', self.language_manager.translate('ui.labels.section', 'Section'))}"
            )
            popup.geometry("900x700")
            
            # Add button frame at top for action buttons
            button_frame = ttk.Frame(popup)
            button_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
            
            # Add "Save as Image" button if this is a graph section
            if section_data.get('type') == 'graph':
                save_img_btn = ttk.Button(
                    button_frame,
                    text="💾 " + self.language_manager.translate("ui.buttons.save_as_image", "Save as Image"),
                    command=lambda: self._save_section_graph_as_image(instance_alias, section_idx)
                )
                save_img_btn.pack(side=tk.LEFT, padx=5)
                
                # Add "Save data" button
                save_data_btn = ttk.Button(
                    button_frame,
                    text="💾 " + self.language_manager.translate("ui.buttons.save_data", "Save Data"),
                    command=lambda: self._save_section_graph_as_csv(instance_alias, section_idx)
                )
                save_data_btn.pack(side=tk.LEFT, padx=5)
            
            # Add close button on the right
            close_btn = ttk.Button(button_frame, text=self.language_manager.translate("ui.buttons.close", "Close"), command=popup.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)
            
            # Create a frame for the content
            content_frame = ttk.Frame(popup)
            content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Render the section in the popup (with is_popup=True to skip popup button)
            self._render_section(content_frame, instance_alias, section_data, section_idx, is_popup=True)
        
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.section_popup_open_failed", "Failed to open section popup:") + f" {str(e)}"
            )
    
    def _save_section_graph_as_image(self, instance_alias: str, section_idx: int):
        """Save a graph section as an image file (PNG, JPEG, or TIFF)."""
        try:
            # Get the canvas and figure from stored references
            if instance_alias not in self.analysis_data:
                self._show_fading_error(
                    self.language_manager.translate("ui.messages.analysis_data_not_found", "Analysis data not found")
                )
                return
            
            # Get current page to build the correct section_id tuple
            current_page = self.analysis_data[instance_alias].get('current_page', 0)
            section_id = (current_page, section_idx)
            
            graph_canvases = self.analysis_data[instance_alias].get('graph_canvases', {})
            if section_id not in graph_canvases:
                self._show_fading_error(
                    self.language_manager.translate("ui.messages.graph_not_found_section", "Graph not found for this section")
                )
                return
            
            canvas, canvas_frame = graph_canvases[section_id]
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
                self._show_fading_success(
                    self.language_manager.translate("ui.messages.graph_saved_to", "Graph saved successfully to:") + f"\n{file_path}"
                )
        
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.graph_save_failed", "Failed to save graph:") + f" {str(e)}"
            )
    
    def _save_section_graph_as_csv(self, instance_alias: str, section_idx: int):
        """Save all data used in a graph section as CSV file(s)."""
        import pandas as pd
        
        try:
            # Get the metadata stored during graph rendering
            if instance_alias not in self.analysis_data:
                self._show_fading_error(
                    self.language_manager.translate("ui.messages.analysis_data_not_found", "Analysis data not found")
                )
                return
            
            # Get current page to build the correct section_id tuple
            current_page = self.analysis_data[instance_alias].get('current_page', 0)
            section_id = (current_page, section_idx)
            
            graph_data_metadata = self.analysis_data[instance_alias].get('graph_data_metadata', {})
            if section_id not in graph_data_metadata:
                self._show_fading_error(
                    self.language_manager.translate("ui.messages.graph_metadata_not_found", "Graph metadata not found for this section")
                )
                return
            
            metadata = graph_data_metadata[section_id]
            
            # Open directory save dialog
            dir_path = filedialog.askdirectory(
                title="Select directory to save graph data",
                mustexist=True
            )
            
            if not dir_path:
                return
            
            # Get graph title for file naming
            graph_title = metadata.get('graph_title', 'graph_data')
            # Sanitize title for use in filename
            safe_title = "".join(c for c in graph_title if c.isalnum() or c in ('-', '_', ' ')).rstrip()
            safe_title = safe_title.replace(' ', '_') if safe_title else 'graph_data'
            
            # Counter for files created
            files_created = []
            
            # Extract and save data
            x_data = metadata.get('x_data')
            y_data = metadata.get('y_data')
            z_data = metadata.get('z_data')
            z_data_source = metadata.get('z_data_source')
            x_axis_config = metadata.get('x_axis_config', {})
            y_axis_config = metadata.get('y_axis_config', {})
            z_axis_config = metadata.get('z_axis_config', {})
            extracted_datasets = metadata.get('extracted_datasets')
            graph_type = metadata.get('graph_type', 'unknown')
            outputs = metadata.get('outputs', {})
            config = metadata.get('config', {})
            
            # For aux_axis configs, re-extract z_data from the raw source to ensure we have full dimensional data
            z_data_raw = self._get_data_from_source(outputs, z_data_source) if ('aux_axis' in config and z_data_source) else None
            if z_data_raw is not None:
                if isinstance(z_data_raw, np.ndarray):
                    z_data = z_data_raw  # Use the full raw data instead of the sliced version
            
            # For aux_axis configs, recompute x_data and y_data based on current dimension combination
            if 'aux_axis' in config and z_data is not None and isinstance(z_data, np.ndarray) and z_data.ndim >= 4:
                # Get current slice state to get md_combo_index
                slice_state = None
                if instance_alias in self.analysis_data:
                    graph_slices = self.analysis_data[instance_alias].get('graph_slices', {})
                    slice_state = graph_slices.get(section_id)
                
                if slice_state:
                    # Compute which dimensions are active based on md_combo_index
                    nav_axes = config.get('data_slicing', [])
                    specified_dims = set()
                    for nav_item in nav_axes:
                        if isinstance(nav_item, dict):
                            dim = nav_item.get('dimension')
                            if dim is not None:
                                specified_dims.add(dim)
                    
                    # Compute dimension combinations
                    from itertools import combinations
                    all_dims = set(range(z_data.ndim))
                    remaining_dims = sorted(all_dims - specified_dims)
                    combo_size = 2
                    
                    if remaining_dims and len(remaining_dims) >= combo_size:
                        combos = list(combinations(remaining_dims, combo_size))
                        md_combo_index = slice_state.get('md_combo_index', 0)
                        if md_combo_index < len(combos):
                            md_active_dims = list(combos[md_combo_index])
                            
                            # Now extract x_data and y_data from aux_axis based on active dims
                            aux_axis_config = config['aux_axis']
                            data_source = aux_axis_config.get('data_source')
                            labels_config = aux_axis_config.get('labels', [])
                            
                            # Resolve labels
                            labels = []
                            if isinstance(labels_config, str):
                                # labels is a variable name
                                label_data = self._get_data_from_source(outputs, labels_config)
                                if label_data is not None:
                                    if isinstance(label_data, (list, np.ndarray)):
                                        labels = [str(lbl) for lbl in label_data]
                            elif isinstance(labels_config, list):
                                labels = labels_config
                            
                            # Extract axis data from data_source
                            axis_data_full = self._get_data_from_source(outputs, data_source) if data_source else None
                            if axis_data_full is not None:
                                if isinstance(axis_data_full, list):
                                    # x-axis uses first dimension in active combination
                                    if len(md_active_dims) > 0:
                                        dim_idx = md_active_dims[0]
                                        if dim_idx < len(axis_data_full):
                                            x_data = np.array(axis_data_full[dim_idx])
                                            x_label = labels[dim_idx] if dim_idx < len(labels) else f"Axis {dim_idx}"
                                            x_axis_config = {'label': x_label}
                                    
                                    # y-axis uses second dimension in active combination
                                    if len(md_active_dims) > 1:
                                        dim_idx = md_active_dims[1]
                                        if dim_idx < len(axis_data_full):
                                            y_data = np.array(axis_data_full[dim_idx])
                                            y_label = labels[dim_idx] if dim_idx < len(labels) else f"Axis {dim_idx}"
                                            y_axis_config = {'label': y_label}
            
            # Get axis labels
            x_label = x_axis_config.get('label', 'X')
            y_label = y_axis_config.get('label', 'Y')
            z_label = z_axis_config.get('label', 'Z')
            
            # Handle different graph types
            if extracted_datasets:
                # Multi-dataset scatter plots
                for dataset_idx, dataset in enumerate(extracted_datasets):
                    dataset_label = dataset.get('label', f'Dataset_{dataset_idx}')
                    safe_dataset_label = "".join(c for c in dataset_label if c.isalnum() or c in ('-', '_', ' ')).rstrip()
                    safe_dataset_label = safe_dataset_label.replace(' ', '_') if safe_dataset_label else f'dataset_{dataset_idx}'
                    
                    ds_x_data = dataset.get('x_data')
                    ds_y_data = dataset.get('y_data')
                    ds_z_data = dataset.get('z_data')
                    ds_class_data = dataset.get('class_data')
                    
                    # Get axis info from dataset
                    ds_x_axis = dataset.get('x_axis', {})
                    ds_y_axis = dataset.get('y_axis', {})
                    ds_x_label = ds_x_axis.get('label', 'X')
                    ds_y_label = ds_y_axis.get('label', 'Y')
                    
                    # Build dataframe
                    data_dict = {}
                    if ds_x_data is not None:
                        data_dict[ds_x_label] = ds_x_data.flatten() if isinstance(ds_x_data, np.ndarray) else ds_x_data
                    if ds_y_data is not None:
                        data_dict[ds_y_label] = ds_y_data.flatten() if isinstance(ds_y_data, np.ndarray) else ds_y_data
                    if ds_z_data is not None:
                        z_label_ds = 'Z'
                        data_dict[z_label_ds] = ds_z_data.flatten() if isinstance(ds_z_data, np.ndarray) else ds_z_data
                    if ds_class_data is not None:
                        data_dict['Class'] = ds_class_data.flatten() if isinstance(ds_class_data, np.ndarray) else ds_class_data
                    
                    if data_dict:
                        df = pd.DataFrame(data_dict)
                        filename = f"{safe_title}_{safe_dataset_label}.csv"
                        file_path = Path(dir_path) / filename
                        df.to_csv(file_path, index=False)
                        files_created.append(filename)
            else:
                # Single dataset graph
                data_dict = {}
                
                if x_data is not None:
                    data_dict[x_label] = x_data.flatten() if isinstance(x_data, np.ndarray) else x_data
                if y_data is not None:
                    data_dict[y_label] = y_data.flatten() if isinstance(y_data, np.ndarray) else y_data
                if z_data is not None:
                    data_dict[z_label] = z_data.flatten() if isinstance(z_data, np.ndarray) else z_data
                
                # Try to get class data if not already in extracted_datasets
                outputs = metadata.get('outputs', {})
                config = metadata.get('config', {})
                
                # Check if config has class_labels field
                class_labels_source = config.get('class_labels')
                class_val = self._get_data_from_source(outputs, class_labels_source) if class_labels_source else None
                if class_val is not None:
                    if isinstance(class_val, (list, np.ndarray)):
                        class_data = np.array(class_val)
                        if class_data.ndim > 1:
                            class_data = class_data.flatten()
                        data_dict['Class'] = class_data
                else:
                    # Try common class label sources
                    for class_source in ['class_data_cal', 'class_data_val', 'class_labels', 'class_data']:
                        class_val = self._get_data_from_source(outputs, class_source)
                        if class_val is not None:
                            if isinstance(class_val, (list, np.ndarray)):
                                class_data = np.array(class_val)
                                if class_data.ndim > 1:
                                    class_data = class_data.flatten()
                                data_dict['Class'] = class_data
                                break
                
                if data_dict:
                    # For multi-dimensional data (heatmap, 3D), include shape info
                    if graph_type in ('heatmap', '3d_surf', 'contour') and z_data is not None:
                        if isinstance(z_data, np.ndarray) and z_data.ndim > 2:
                            # For 3D+ data, need to get the currently displayed 2D slice
                            # Get the slice state from analysis_data
                            slice_state = None
                            if instance_alias in self.analysis_data:
                                graph_slices = self.analysis_data[instance_alias].get('graph_slices', {})
                                slice_state = graph_slices.get(section_id)
                            
                            if slice_state:
                                # Check if this is an aux_axis config (4D+ with dimension combo)
                                md_active_dims = None
                                has_aux_axis = 'aux_axis' in config
                                
                                if has_aux_axis:
                                    # For aux_axis configs, compute which dimensions are actively displayed
                                    nav_axes = config.get('data_slicing', [])
                                    specified_dims = set()
                                    for nav_item in nav_axes:
                                        if isinstance(nav_item, dict):
                                            dim = nav_item.get('dimension')
                                            if dim is not None:
                                                specified_dims.add(dim)
                                    
                                    # Compute dimension combinations
                                    from itertools import combinations
                                    all_dims = set(range(z_data.ndim))
                                    remaining_dims = sorted(all_dims - specified_dims)
                                    combo_size = 2
                                    
                                    if remaining_dims and len(remaining_dims) >= combo_size:
                                        combos = list(combinations(remaining_dims, combo_size))
                                        md_combo_index = slice_state.get('md_combo_index', 0)
                                        if md_combo_index < len(combos):
                                            md_active_dims = list(combos[md_combo_index])
                                
                                # Extract the currently displayed 2D slice using the stored indices
                                indices = slice_state.get('indices', {})
                                md_slice_indices = slice_state.get('md_slice_indices', {})
                                
                                # Merge indices for slicing
                                all_indices = indices.copy()
                                all_indices.update(md_slice_indices)
                                
                                # For aux_axis configs, exclude active dimensions from slicing
                                if md_active_dims:
                                    for dim in md_active_dims:
                                        all_indices.pop(dim, None)
                                
                                # Build proper indexing tuple
                                index_list = []
                                for dim in range(z_data.ndim):
                                    if dim in all_indices:
                                        idx = all_indices[dim]
                                        max_idx = z_data.shape[dim] - 1
                                        if idx > max_idx:
                                            idx = max_idx
                                        elif idx < 0:
                                            idx = 0
                                        index_list.append(idx)
                                    else:
                                        index_list.append(slice(None))
                                
                                # Apply slicing to get 2D data
                                try:
                                    z_data_2d = z_data[tuple(index_list)]
                                    # If still not 2D, try to reshape
                                    if z_data_2d.ndim > 2:
                                        z_data_2d = z_data_2d.flatten().reshape(z_data.shape[0], -1)
                                    df = pd.DataFrame(z_data_2d)
                                    # Add row/col labels only if dimensions match exactly
                                    if x_data is not None and y_data is not None:
                                        try:
                                            x_array = np.asarray(x_data).flatten() if hasattr(x_data, '__len__') else x_data
                                            y_array = np.asarray(y_data).flatten() if hasattr(y_data, '__len__') else y_data
                                            # Only apply labels if dimensions match
                                            if len(x_array) == z_data_2d.shape[1]:
                                                df.columns = [f"{x_label}_{i}_{val}" for i, val in enumerate(x_array)]
                                            if len(y_array) == z_data_2d.shape[0]:
                                                df.index = [f"{y_label}_{i}_{val}" for i, val in enumerate(y_array)]
                                        except Exception:
                                            pass  # Keep default integer indices if something fails
                                except (IndexError, ValueError):
                                    # Fallback: just use first 2D slice
                                    z_data_2d = z_data[0] if z_data.ndim > 2 else z_data
                                    df = pd.DataFrame(z_data_2d)
                            else:
                                # No slice state, use first 2D slice
                                z_data_2d = z_data[0] if z_data.ndim > 2 else z_data
                                df = pd.DataFrame(z_data_2d)
                        elif isinstance(z_data, np.ndarray) and z_data.ndim == 2:
                            # For 2D data, save as-is with row/col headers
                            df = pd.DataFrame(z_data)
                            # Add row/col indices only if x and y data dimensions match
                            if x_data is not None and y_data is not None:
                                try:
                                    x_array = np.asarray(x_data).flatten() if hasattr(x_data, '__len__') else x_data
                                    y_array = np.asarray(y_data).flatten() if hasattr(y_data, '__len__') else y_data
                                    # Only apply labels if dimensions match
                                    if len(x_array) == z_data.shape[1]:
                                        df.columns = [f"{x_label}_{i}_{val}" for i, val in enumerate(x_array)]
                                    if len(y_array) == z_data.shape[0]:
                                        df.index = [f"{y_label}_{i}_{val}" for i, val in enumerate(y_array)]
                                except Exception:
                                    pass  # Keep default integer indices if something fails
                        else:
                            df = pd.DataFrame(data_dict)
                    else:
                        df = pd.DataFrame(data_dict)
                    
                    filename = f"{safe_title}.csv"
                    file_path = Path(dir_path) / filename
                    # For heatmaps, export just the data matrix without headers or index
                    if graph_type in ('heatmap', 'contour', '3d_surf'):
                        df.to_csv(file_path, index=False, header=False)
                    else:
                        df.to_csv(file_path, index=False)
                    files_created.append(filename)
            
            # Show success message
            if files_created:
                message = (
                    self.language_manager.translate("ui.messages.graph_data_saved", "Data saved successfully!") +
                    "\n\n" +
                    self.language_manager.translate("ui.messages.files_created", "Files created:") +
                    "\n"
                )
                for fname in files_created:
                    message += f"  • {fname}\n"
                message += f"\n{self.language_manager.translate('ui.messages.location', 'Location')}: {dir_path}"
                self._show_fading_success(message)
            else:
                self._show_fading_warning(
                    self.language_manager.translate("ui.messages.no_data_found_save_graph", "No data found to save for this graph.")
                )
        
        except ImportError:
            self._show_fading_error(
                self.language_manager.translate(
                    "ui.messages.pandas_required_csv",
                    "pandas library is required to save data as CSV.\n\nPlease install it using: pip install pandas"
                )
            )
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.graph_data_save_failed", "Failed to save graph data:") + f" {str(e)}"
            )
            import traceback
            traceback.print_exc()
    
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
                                   section_id: tuple, outputs: dict, config: dict, 
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
            nested_key = None  # Initialize to None for all code paths
            graph_type = config.get('graph_type', '')
            if 'aux_axis' in config:
                # Use z_axis to get the actual data array
                data_source = config.get('z_axis', {}).get('data_source')
                nested_key = config.get('z_axis', {}).get('nested_key')
            elif graph_type in ('heatmap', 'contour', '3d_surf'):
                # For 3D visualization types, z_axis contains the actual multi-dimensional data
                data_source = config.get('z_axis', {}).get('data_source')
                nested_key = config.get('z_axis', {}).get('nested_key')
            elif 'datasets' in config:
                # Multi-dataset scatter plots: use first available dataset's data source
                datasets_cfg = config.get('datasets', [])
                for ds_cfg in datasets_cfg:
                    ds_y_source = ds_cfg.get('y_axis', {}).get('data_source')
                    if ds_y_source and ds_y_source in outputs:
                        data_source = ds_y_source
                        nested_key = ds_cfg.get('y_axis', {}).get('nested_key')
                        break
                    ds_x_source = ds_cfg.get('x_axis', {}).get('data_source')
                    if ds_x_source and ds_x_source in outputs:
                        data_source = ds_x_source
                        nested_key = ds_cfg.get('x_axis', {}).get('nested_key')
                        break
            else:
                # Traditional configs - use y_axis or x_axis
                axis_config = config.get('y_axis', {}) or config.get('x_axis', {})
                data_source = axis_config.get('data_source')
                nested_key = axis_config.get('nested_key')
            
            if not data_source:
                return
            
            # Use helper to support nested dictionary access
            data = self._get_data_from_source(outputs, data_source, nested_key)
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
                        md_frame = ttk.LabelFrame(parent_frame, text=self.language_manager.translate("ui.labels.multi_dim_slicing", "Multi-Dimensional Slicing (4D+)"), padding=5)
                        md_frame.pack(fill=tk.X, padx=5, pady=5)
                        
                        # Combination selector
                        combo_select_frame = ttk.Frame(md_frame)
                        combo_select_frame.pack(fill=tk.X, padx=5, pady=2)
                        
                        ttk.Label(combo_select_frame, text=self.language_manager.translate("ui.labels.dimension_combination", "Dimension Combination:"), width=20).pack(side=tk.LEFT, padx=5)
                        
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
    
    def _on_navigate_slice(self, instance_alias: str, section_id: tuple, direction: int,
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
                        exec_results = self.analysis_data[instance_alias].get('execution_results', {})
                        outputs = self._get_execution_data_sources(exec_results, instance_alias)
                        var_label_text = self._get_variable_label(outputs, var_labels_config, dim, new_index)
                        if var_label_text:
                            var_label.config(text=f"[{var_label_text}]")
                        else:
                            var_label.config(text="")
                
                # Refresh the graph with new slice/axis
                self._update_graph_with_slice(instance_alias, section_id, dimension)
        
        except Exception as e:
            print(f"Error navigating slice: {str(e)}")
    
    def _on_md_combo_changed(self, instance_alias: str, section_id: tuple, 
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
                nested_key = None
                if 'aux_axis' in config:
                    data_source = config.get('z_axis', {}).get('data_source')
                    nested_key = config.get('z_axis', {}).get('nested_key')
                else:
                    axis_config = config.get('y_axis', {}) or config.get('x_axis', {})
                    data_source = axis_config.get('data_source')
                    nested_key = axis_config.get('nested_key')
                
                if data_source:
                    # Use helper to support nested dictionary access
                    data = self._get_data_from_source(outputs, data_source, nested_key)
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
    
    def _on_md_navigate(self, instance_alias: str, section_id: tuple, direction: int,
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
                        exec_results = self.analysis_data[instance_alias].get('execution_results', {})
                        outputs = self._get_execution_data_sources(exec_results, instance_alias)
                        var_label_text = self._get_variable_label(outputs, var_labels_config, dim, new_index)
                        if var_label_text:
                            var_label.config(text=f"[{var_label_text}]")
                        else:
                            var_label.config(text="")
                
                # Refresh the graph with new multi-dimensional slice
                self._update_graph_with_slice(instance_alias, section_id, dimension)
        
        except Exception as e:
            print(f"Error navigating MD slice: {str(e)}")
    
    def _update_graph_with_slice(self, instance_alias: str, section_id: tuple, 
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
            # Get fresh outputs from execution_results, not from stale cached version
            execution_results = self.analysis_data[instance_alias].get('execution_results', {})
            outputs = self._get_execution_data_sources(execution_results, instance_alias)
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
                
                data_temp = self._get_data_from_source(outputs, data_source_temp) if data_source_temp else None
                if data_temp is not None:
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
                    label_data = self._get_data_from_source(outputs, labels_config)
                    if label_data is not None:
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
            
            # For axis selection mode, get the current axis indices
            # axis_indices structure: {'x': {dimension: index}, 'y': {dimension: index}}
            # We need to extract the actual index value for each axis
            x_axis_idx = None
            y_axis_idx = None
            z_axis_idx = None
            
            # Find dimension for each axis from data_slicing config
            nav_axes = config.get('data_slicing', [])
            x_dimension = None
            y_dimension = None
            z_dimension = None
            
            for nav_item in nav_axes:
                if isinstance(nav_item, dict):
                    target_axis = nav_item.get('axis')
                    dimension = nav_item.get('dimension')
                    if target_axis == 'x' and dimension is not None:
                        x_dimension = dimension
                    elif target_axis == 'y' and dimension is not None:
                        y_dimension = dimension
                    elif target_axis == 'z' and dimension is not None:
                        z_dimension = dimension
            
            # Extract the actual index values from axis_indices
            if 'x' in axis_indices and x_dimension is not None:
                x_axis_idx = axis_indices['x'].get(x_dimension)
            if 'y' in axis_indices and y_dimension is not None:
                y_axis_idx = axis_indices['y'].get(y_dimension)
            if 'z' in axis_indices and z_dimension is not None:
                z_axis_idx = axis_indices['z'].get(z_dimension)
            
            resolved_x_label = self._resolve_axis_label(x_axis_config, outputs, axis_index=x_axis_idx)
            if resolved_x_label:
                x_axis_config['label'] = resolved_x_label
            
            resolved_y_label = self._resolve_axis_label(y_axis_config, outputs, axis_index=y_axis_idx)
            if resolved_y_label:
                y_axis_config['label'] = resolved_y_label
            
            # Also resolve z-axis label if present
            z_axis_config = config.get('z_axis', {})
            if z_axis_config:
                z_axis_config = z_axis_config.copy()
                resolved_z_label = self._resolve_axis_label(z_axis_config, outputs, axis_index=z_axis_idx)
                if resolved_z_label:
                    z_axis_config['label'] = resolved_z_label
            
            # Merge axis indices for y (base + md + axis-specific)
            # Extract y FIRST so __index__ on x-axis can use y_data as reference
            y_indices = base_indices.copy()
            y_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'y' in axis_indices:
                y_indices.update(axis_indices['y'])
            y_data = self._extract_axis_data(outputs, y_axis_config, y_indices)
            
            # Merge axis indices for x (base + md + axis-specific)
            x_indices = base_indices.copy()
            x_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'x' in axis_indices:
                x_indices.update(axis_indices['x'])
            # Pass y_data as reference for row index generation if needed
            x_data = self._extract_axis_data(outputs, x_axis_config, x_indices, ref_data=y_data)
            
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
            # Pass x_data or y_data as reference for row index generation if needed
            z_data = self._extract_axis_data(outputs, config.get('z_axis', {}), z_indices, ref_data=x_data if x_data is not None else y_data)
            
            # Create a copy of config with resolved axis labels for rendering
            render_config = config.copy()
            render_config['x_axis'] = x_axis_config
            render_config['y_axis'] = y_axis_config
            if z_axis_config:
                render_config['z_axis'] = z_axis_config
            
            # Handle multiple datasets if configured
            datasets_config = config.get('datasets')
            
            # For multi-dataset scatter plots, if config has datasets but no top-level x/y axes,
            # resolve the axis labels from the first dataset's axes (all datasets typically share same axis labels)
            if datasets_config and isinstance(datasets_config, list) and len(datasets_config) > 0:
                # Only override if main config doesn't have x_axis/y_axis
                if not config.get('x_axis') or not config.get('y_axis'):
                    # Get axis configs from first dataset that has them
                    for dataset_cfg in datasets_config:
                        if not config.get('x_axis') and 'x_axis' in dataset_cfg:
                            temp_x_axis = dataset_cfg['x_axis'].copy()
                            resolved_x_label = self._resolve_axis_label(temp_x_axis, outputs, axis_index=x_axis_idx)
                            if resolved_x_label:
                                temp_x_axis['label'] = resolved_x_label
                            render_config['x_axis'] = temp_x_axis
                            break
                    
                    for dataset_cfg in datasets_config:
                        if not config.get('y_axis') and 'y_axis' in dataset_cfg:
                            temp_y_axis = dataset_cfg['y_axis'].copy()
                            resolved_y_label = self._resolve_axis_label(temp_y_axis, outputs, axis_index=y_axis_idx)
                            if resolved_y_label:
                                temp_y_axis['label'] = resolved_y_label
                            render_config['y_axis'] = temp_y_axis
                            break
            
            extracted_datasets = None
            if datasets_config and isinstance(datasets_config, list):
                extracted_datasets = []
                for dataset_idx, dataset_cfg in enumerate(datasets_config):
                    dataset_label = dataset_cfg.get('label', f'Dataset {dataset_idx + 1}')
                    
                    # Check dataset condition if provided (optional filtering)
                    if 'condition' in dataset_cfg:
                        condition = dataset_cfg['condition']
                        param_name = condition.get('parameter')
                        operator = condition.get('operator')
                        expected_value = condition.get('value')
                        
                        # Evaluate condition against execution inputs
                        exec_inputs = execution_results.get('inputs', {})
                        if param_name and param_name in exec_inputs:
                            actual_value = exec_inputs[param_name]
                            include_dataset = self._evaluate_condition(actual_value, operator, expected_value)
                            if not include_dataset:
                                # Dataset filtered by condition - skip it
                                continue
                    
                    # Extract data for this dataset using same indices as main axes
                    ds_x_axis = dataset_cfg.get('x_axis', {})
                    ds_y_axis = dataset_cfg.get('y_axis', {})
                    ds_z_axis = dataset_cfg.get('z_axis', {})
                    
                    # Use same indices for consistency with main plot
                    ds_x_data = self._extract_axis_data(outputs, ds_x_axis, x_indices)
                    # Pass ds_x_data as reference for row index generation if needed
                    ds_y_data = self._extract_axis_data(outputs, ds_y_axis, y_indices, ref_data=ds_x_data)
                    # Pass ds_x_data or ds_y_data as reference for row index generation if needed
                    ds_z_data = self._extract_axis_data(outputs, ds_z_axis, z_indices, ref_data=ds_x_data if ds_x_data is not None else ds_y_data) if ds_z_axis else None
                    
                    # Skip if required data sources don't exist (normal for optional datasets)
                    if ds_x_data is None or ds_y_data is None:
                        continue
                    
                    # Extract class labels if specified
                    ds_class_data = None
                    if 'class_labels' in dataset_cfg:
                        class_source = dataset_cfg['class_labels']
                        class_val = self._get_data_from_source(outputs, class_source)
                        if class_val is not None:
                            if isinstance(class_val, (list, np.ndarray)):
                                ds_class_data = np.array(class_val)
                    
                    # Dataset is valid and will be rendered
                    dataset_entry = {
                        'x_data': ds_x_data,
                        'y_data': ds_y_data,
                        'label': dataset_label,
                        'marker': dataset_cfg.get('marker', 'o'),
                        'x_axis': ds_x_axis,  # Preserve axis config for label extraction
                        'y_axis': ds_y_axis   # Preserve axis config for label extraction
                    }
                    if ds_z_data is not None:
                        dataset_entry['z_data'] = ds_z_data
                    if ds_class_data is not None:
                        dataset_entry['class_data'] = ds_class_data
                    # Include color if specified (used as fallback when no class_data)
                    if 'color' in dataset_cfg:
                        dataset_entry['color'] = dataset_cfg['color']
                    # Include sample_labels_source if specified per-dataset
                    if 'sample_labels_source' in dataset_cfg:
                        dataset_entry['sample_labels_source'] = dataset_cfg['sample_labels_source']
                    extracted_datasets.append(dataset_entry)
            
            # If main plot has class_labels config, treat it as a dataset for proper class coloring with qualitative colormap
            if 'class_labels' in config and graph_type == 'scatter' and x_data is not None and y_data is not None:
                class_source = config['class_labels']
                class_val = self._get_data_from_source(outputs, class_source)
                if class_val is not None:
                    if isinstance(class_val, (list, np.ndarray)):
                        main_class_data = np.array(class_val)
                        
                        # Create extracted_datasets if it doesn't exist
                        if extracted_datasets is None:
                            extracted_datasets = []
                        
                        # Build main dataset with class data
                        main_dataset = {
                            'x_data': x_data,
                            'y_data': y_data,
                            'label': 'Main Dataset',
                            'marker': 'o',
                            'class_data': main_class_data
                        }
                        if z_data is not None:
                            main_dataset['z_data'] = z_data
                        # Include sample_labels_source if specified at config level
                        if 'sample_labels_source' in config:
                            main_dataset['sample_labels_source'] = config['sample_labels_source']
                        
                        # Add main dataset at the beginning of the list
                        extracted_datasets.insert(0, main_dataset)
                        
                        # Clear x/y/z data so they won't conflict with multi-dataset rendering
                        x_data = None
                        y_data = None
                        z_data = None
            
            # Extract sample labels for tooltip display from individual datasets
            sample_labels = None
            sample_labels_by_dataset = None
            
            # If we have extracted datasets, collect their sample_labels_source
            if extracted_datasets and len(extracted_datasets) > 0:
                sample_labels_by_dataset = {}
                for dataset_entry in extracted_datasets:
                    ds_label = dataset_entry.get('label')
                    ds_source = dataset_entry.get('sample_labels_source')
                    if ds_label and ds_source and ds_source in outputs:
                        labels_data = outputs[ds_source]
                        if isinstance(labels_data, (list, np.ndarray)):
                            sample_labels_by_dataset[ds_label] = [str(lbl) for lbl in labels_data]
            else:
                # For single dataset, check for sample_labels_source in config
                sample_labels_source = config.get('sample_labels_source')
                if isinstance(sample_labels_source, str):
                    # Single sample labels source
                    if sample_labels_source in outputs:
                        labels_data = outputs[sample_labels_source]
                        if isinstance(labels_data, (list, np.ndarray)):
                            sample_labels = [str(lbl) for lbl in labels_data]
            
            # Fall back to default names if no explicit source found
            if not sample_labels and not sample_labels_by_dataset:
                for label_key in ['smp_cal', 'sample_labels', 'smp_path', 'sample_names']:
                    if label_key in outputs:
                        labels_data = outputs[label_key]
                        if isinstance(labels_data, (list, np.ndarray)):
                            sample_labels = [str(lbl) for lbl in labels_data]
                            break
            
            # Render graph using graph_renderer module
            fig, ax = graph_renderer.render_graph_figure(
                graph_type, render_config, x_data, y_data, z_data, x_axis_config, y_axis_config,
                default_cmap=self.settings_manager.get('colormap', 'viridis'),
                datasets=extracted_datasets,
                qualitative_cmap=self.settings_manager.get('qualitative_colormap', 'tab10'),
                sample_labels=sample_labels,
                sample_labels_by_dataset=sample_labels_by_dataset
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
            self._show_fading_message(
                self.language_manager.translate("ui.messages.no_sections_remove", "No sections to remove")
            )
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(self.language_manager.translate("ui.dialogs.remove_section", "Remove Section"))
        dialog.geometry("350x250")
        dialog.resizable(False, False)
        
        label = ttk.Label(dialog, text=self.language_manager.translate("ui.messages.select_section_remove", "Select section to remove:"), font=("Arial", 10))
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
            section_type = section.get('type', self.language_manager.translate("ui.labels.empty", "Empty"))
            config = section.get('config', {})
            section_title = config.get('title', f"{self.language_manager.translate('ui.labels.section', 'Section')} {idx + 1}")
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
        
        ok_btn = ttk.Button(button_frame, text=self.language_manager.translate("ui.buttons.remove", "Remove"), command=remove_selected)
        ok_btn.pack(padx=5)
        
        cancel_btn = ttk.Button(button_frame, text=self.language_manager.translate("ui.buttons.cancel", "Cancel"), command=dialog.destroy)
        cancel_btn.pack(padx=5)
    
    def _show_add_graph_dialog(self, instance_alias: str):
        """Show the Add Graph dialog."""
        show_add_graph_dialog(self.root, self, instance_alias)
    
    def _show_add_table_dialog(self, instance_alias: str):
        """Show the Add Table dialog."""
        show_add_table_dialog(self.root, self, instance_alias)
    
    def _show_add_page_dialog(self, instance_alias: str):
        """Show dialog to add a new page."""
        if instance_alias not in self.analysis_data:
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title(self.language_manager.translate("ui.dialogs.add_page", "Add Page"))
        dialog.geometry("500x520")
        dialog.resizable(False, False)
        
        # Main content frame with scrolling
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Page title
        title_label = ttk.Label(main_frame, text=self.language_manager.translate("ui.labels.page_title", "Page Title:"), font=("Arial", 10))
        title_label.pack(anchor=tk.W, pady=(0, 5))
        
        title_entry = ttk.Entry(main_frame, width=30)
        title_entry.pack(anchor=tk.W, pady=(0, 10))
        title_entry.insert(0, f"Page {len(self.analysis_data[instance_alias]['pages']) + 1}")
        
        # Layout selection with visual buttons
        layout_label = ttk.Label(main_frame, text=self.language_manager.translate("ui.labels.layout", "Layout:"), font=("Arial", 11, "bold"))
        layout_label.pack(anchor=tk.W, pady=(10, 10))
        
        layouts = [
            ('fp', 'Full\nPage'),
            ('ns', 'North-\nSouth'),
            ('ew', 'East-\nWest'),
            ('fd', 'Four\nSections'),
            ('sd', 'South\nDivision'),
            ('nd', 'North\nDivision'),
            ('ed', 'East\nDivision'),
            ('wd', 'West\nDivision'),
        ]
        
        layout_var = tk.StringVar(value='fp')
        button_map = {}
        
        def draw_layout_visualization(canvas, layout_code):
            """Draw layout visualization on canvas."""
            canvas.create_rectangle(2, 2, 78, 78, outline='black', fill='white', width=1)
            
            if layout_code == 'fp':
                pass  # Just rectangle
            elif layout_code == 'ns':
                canvas.create_line(2, 39, 78, 39, width=2)  # Horizontal
            elif layout_code == 'ew':
                canvas.create_line(39, 2, 39, 78, width=2)  # Vertical
            elif layout_code == 'fd':
                canvas.create_line(39, 2, 39, 78, width=2)  # Vertical
                canvas.create_line(2, 39, 78, 39, width=2)  # Horizontal
            elif layout_code == 'sd':
                canvas.create_line(2, 39, 78, 39, width=2)  # Horizontal
                canvas.create_line(39, 39, 39, 78, width=2)  # Vertical in bottom
            elif layout_code == 'nd':
                canvas.create_line(2, 39, 78, 39, width=2)  # Horizontal
                canvas.create_line(39, 2, 39, 39, width=2)  # Vertical in top
            elif layout_code == 'ed':
                canvas.create_line(39, 2, 39, 78, width=2)  # Vertical
                canvas.create_line(39, 39, 78, 39, width=2)  # Horizontal in right
            elif layout_code == 'wd':
                canvas.create_line(39, 2, 39, 78, width=2)  # Vertical
                canvas.create_line(2, 39, 39, 39, width=2)  # Horizontal in left
        
        def on_layout_select(layout_code, btn_frame):
            """Handle layout button selection."""
            layout_var.set(layout_code)
            for btn in button_map.values():
                btn.config(relief=tk.RAISED, bg='SystemButtonFace')
            btn_frame.config(relief=tk.SUNKEN, bg='#D0E8FF')
        
        # Create layout buttons in grid (4 columns, 2 rows)
        layouts_grid_frame = ttk.Frame(main_frame)
        layouts_grid_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        for row in range(2):
            row_frame = ttk.Frame(layouts_grid_frame)
            row_frame.pack(fill=tk.BOTH, expand=True, pady=3, padx=2)
            
            for col in range(4):
                idx = row * 4 + col
                if idx < len(layouts):
                    layout_code, layout_desc = layouts[idx]
                    
                    # Create button frame
                    btn_frame = tk.Frame(row_frame, relief=tk.RAISED, borderwidth=2, width=95, height=125, bg='SystemButtonFace')
                    btn_frame.pack_propagate(False)
                    btn_frame.pack(side=tk.LEFT, padx=5, pady=0)
                    
                    # Canvas with layout visualization
                    canvas = tk.Canvas(btn_frame, width=80, height=80, bg='white', highlightthickness=1, highlightbackground='#999999')
                    canvas.pack(padx=3, pady=(3, 0))
                    draw_layout_visualization(canvas, layout_code)
                    
                    # Label
                    label = tk.Label(btn_frame, text=layout_desc, font=("Arial", 7), bg='SystemButtonFace', justify=tk.CENTER)
                    label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
                    
                    # Store button reference
                    button_map[layout_code] = btn_frame
                    
                    # Bind click events
                    def make_click_handler(code, frame):
                        def handler(event=None):
                            on_layout_select(code, frame)
                        return handler
                    
                    btn_frame.bind("<Button-1>", make_click_handler(layout_code, btn_frame))
                    canvas.bind("<Button-1>", make_click_handler(layout_code, btn_frame))
                    label.bind("<Button-1>", make_click_handler(layout_code, btn_frame))
        
        # Pre-select first button
        first_btn = button_map['fp']
        first_btn.config(relief=tk.SUNKEN, bg='#D0E8FF')
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10, side=tk.BOTTOM)
        
        def add_page():
            title = title_entry.get() or self.language_manager.translate("ui.labels.new_page", "New Page")
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
        
        ok_btn = ttk.Button(button_frame, text=self.language_manager.translate("ui.buttons.add", "Add"), command=add_page)
        ok_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(button_frame, text=self.language_manager.translate("ui.buttons.cancel", "Cancel"), command=dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def _remove_current_page(self, instance_alias: str):
        """Remove the current page."""
        if instance_alias not in self.analysis_data:
            return
        
        pages = self.analysis_data[instance_alias]['pages']
        current_page = self.analysis_data[instance_alias]['current_page']
        
        if len(pages) <= 1:
            self._show_fading_warning(
                self.language_manager.translate("ui.messages.cannot_remove_last_page", "Cannot remove the last page")
            )
            return
        
        if messagebox.askyesno(
            self.language_manager.translate("ui.dialogs.confirm", "Confirm"),
            self.language_manager.translate("ui.messages.remove_page_confirm", "Remove page '{title}'?").format(title=pages[current_page]['title'])
        ):
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
                self._show_fading_error(
                    self.language_manager.translate("ui.messages.function_not_found_methodology", "Function not found in methodology")
                )
                return
            
            stop_at_idx = self.methodology_list.index(instance_alias)

            self._begin_execution_progress(
                total_steps=stop_at_idx + 1,
                mode_label=self.language_manager.translate("ui.buttons.run_to_here", "Run to here")
            )
            
            # Generate model.json first
            if not self._generate_model_json():
                self._show_fading_error(
                    self.language_manager.translate("ui.messages.generate_model_json_failed", "Failed to generate model.json")
                )
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
                def _progress_callback(completed_steps: int, total_steps: int, current_instance: str, current_base: str):
                    self._update_execution_progress(completed_steps, total_steps, current_instance, current_base)

                # Run analyst with stop_at_function_idx parameter
                outputs, timing_report = analyst_main(
                    stop_at_function_idx=stop_at_idx,
                    progress_callback=_progress_callback,
                    return_timing=True
                )
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            execution_time_by_instance = {
                entry.get('instance_alias'): entry.get('execution_time', 0.0)
                for entry in (timing_report.get('function_timings', []) if timing_report else [])
            }
            execution_history_by_instance = timing_report.get('execution_history_by_instance', {}) if timing_report else {}

            # Store execution results in analysis data for ALL executed functions
            # (not just the selected one), so functions before it can also display their results
            for idx in range(stop_at_idx + 1):
                func_instance_alias = self.methodology_list[idx]
                func_base_alias = self.function_base_aliases[idx] if idx < len(self.function_base_aliases) else func_instance_alias
                
                # Load analysis configuration from function's gui_config if available
                analysis_config = None
                if func_base_alias in self.gui_configs:
                    analysis_config = self.gui_configs[func_base_alias].get('analysis')
                
                if func_instance_alias not in self.analysis_data:
                    if analysis_config:
                        # Use analysis config from function's JSON
                        self.analysis_data[func_instance_alias] = {
                            'pages': copy.deepcopy(analysis_config.get('pages', [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}])),
                            'current_page': analysis_config.get('current_page', 0)
                        }
                    else:
                        # Fallback to default structure
                        self.analysis_data[func_instance_alias] = {
                            'pages': [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}],
                            'current_page': 0
                        }
                # Note: When analysis_data already exists, we preserve the user's modifications
                # (like removed sections). The condition evaluation will use fresh inputs from
                # execution_results which is updated below.
                
                # Get input parameters from function_configs
                input_parameters = self.function_configs.get(func_instance_alias, {}).copy()
                
                # Get outputs for this function instance
                function_outputs = outputs.get(func_instance_alias, {}) if outputs else {}
                history_entries = copy.deepcopy(execution_history_by_instance.get(func_instance_alias, []))
                
                if history_entries:
                    self.analysis_data[func_instance_alias]['execution_history'] = history_entries
                    selected_history_idx = 0
                    self.analysis_data[func_instance_alias]['current_result_idx'] = selected_history_idx
                    self.analysis_data[func_instance_alias]['execution_results'] = history_entries[selected_history_idx].copy()
                else:
                    self.analysis_data[func_instance_alias]['execution_results'] = {
                        'status': 'success',
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': execution_time_by_instance.get(func_instance_alias, 0.0),
                        'outputs': function_outputs,
                        'inputs': input_parameters
                    }
                    self.analysis_data[func_instance_alias]['execution_history'] = [
                        self.analysis_data[func_instance_alias]['execution_results'].copy()
                    ]
                    self.analysis_data[func_instance_alias]['current_result_idx'] = 0

            self._store_timing_report(
                run_type_label=self.language_manager.translate("ui.buttons.run_to_here", "Run to here"),
                timing_report=timing_report,
                stop_at_function_alias=instance_alias
            )
            
            # Note: graph_canvases, graph_slices, and table_slices are already cleared
            # by _clear_execution_cache() at the start of this method
            
            self._show_fading_success(
                self.language_manager.translate("ui.messages.model_executed_up_to", "Model executed up to") +
                f" {instance_alias}\n\n" +
                self.language_manager.translate("ui.messages.results_loaded_analysis", "Results loaded for analysis.")
            )
            self._finish_execution_progress(success=True)
            
            # Refresh the analysis tab to show results
            self._show_analysis_tab()
            
        except Exception as e:
            self._finish_execution_progress(success=False)
            error_msg = self.language_manager.translate("ui.messages.run_model_failed", "Failed to run model:") + f" {str(e)}"
            self._show_fading_error(error_msg)
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()

    
    def _show_report_tab(self):
        """Show Report tab with report composition controls and preview."""
        self._clear_tab()
        self.current_tab = "report"

        if not hasattr(self, 'report_data') or not isinstance(self.report_data, dict):
            self.report_data = {'elements': [], 'selected_index': None}
        if 'elements' not in self.report_data:
            self.report_data['elements'] = []
        if 'selected_index' not in self.report_data:
            self.report_data['selected_index'] = None

        main_paned = ttk.PanedWindow(self.tab_content_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        sidebar_frame = ttk.Frame(main_paned, width=360)
        sidebar_frame.pack_propagate(False)
        main_paned.add(sidebar_frame, weight=0)

        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        preview_save_frame = ttk.LabelFrame(sidebar_frame, text=self.language_manager.translate("ui.labels.preview_save", "Preview / Save"), padding=8)
        preview_save_frame.pack(fill=tk.X, padx=4, pady=(4, 8))

        preview_save_buttons = ttk.Frame(preview_save_frame)
        preview_save_buttons.pack(fill=tk.X)

        preview_btn = ttk.Button(
            preview_save_buttons,
            text=self.language_manager.translate("ui.buttons.preview_report", "Preview"),
            command=self._generate_report_preview,
            width=14
        )
        preview_btn.pack(side=tk.LEFT, padx=(0, 4))

        save_btn = ttk.Button(
            preview_save_buttons,
            text=self.language_manager.translate("ui.buttons.save_pdf_report", "Save PDF"),
            command=self._save_pdf_report,
            width=14
        )
        save_btn.pack(side=tk.LEFT, padx=(4, 0))

        middle_frame = ttk.Frame(sidebar_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        vertical_split = ttk.PanedWindow(middle_frame, orient=tk.VERTICAL)
        vertical_split.pack(fill=tk.BOTH, expand=True)

        elements_frame = ttk.LabelFrame(vertical_split, text=self.language_manager.translate("ui.labels.elements", "Elements"), padding=8)
        vertical_split.add(elements_frame, weight=1)

        element_definitions = [
            ("report_header", self._report_element_label("report_header")),
            ("title", self._report_element_label("title")),
            ("section", self._report_element_label("section")),
            ("subsection", self._report_element_label("subsection")),
            ("subsubsection", self._report_element_label("subsubsection")),
            ("sample_name_cal", self._report_element_label("sample_name_cal")),
            ("sample_name_val", self._report_element_label("sample_name_val")),
            ("sample_metadata_cal", self._report_element_label("sample_metadata_cal")),
            ("sample_metadata_val", self._report_element_label("sample_metadata_val")),
            ("text", self._report_element_label("text")),
            ("graph", self._report_element_label("graph")),
            ("table", self._report_element_label("table")),
            ("page_break", self._report_element_label("page_break")),
        ]

        elements_scroll_container = ttk.Frame(elements_frame)
        elements_scroll_container.pack(fill=tk.BOTH, expand=True)

        elements_scrollbar = tk.Scrollbar(elements_scroll_container, orient=tk.VERTICAL)
        elements_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        elements_canvas = tk.Canvas(elements_scroll_container, highlightthickness=0, bg="#f0f0f0", yscrollcommand=elements_scrollbar.set)
        elements_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        elements_scrollbar.configure(command=elements_canvas.yview)

        elements_inner = ttk.Frame(elements_canvas)
        elements_canvas.create_window((0, 0), window=elements_inner, anchor="nw")
        elements_inner.bind("<Configure>", lambda e: elements_canvas.configure(scrollregion=elements_canvas.bbox("all")))

        def _elements_mousewheel(event):
            try:
                elements_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass

        elements_canvas.bind("<MouseWheel>", _elements_mousewheel)
        elements_inner.bind("<MouseWheel>", _elements_mousewheel)

        for element_type, label in element_definitions:
            btn = ttk.Button(
                elements_inner,
                text=label,
                command=lambda et=element_type: self._add_report_element(et)
            )
            btn.pack(fill=tk.X, pady=2)

        structure_frame = ttk.LabelFrame(vertical_split, text=self.language_manager.translate("ui.labels.structure", "Structure"), padding=8)
        vertical_split.add(structure_frame, weight=1)

        structure_list_frame = ttk.Frame(structure_frame)
        structure_list_frame.pack(fill=tk.BOTH, expand=True)

        self.report_structure_listbox = tk.Listbox(structure_list_frame, height=12, selectmode=tk.SINGLE)
        self.report_structure_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.report_structure_listbox.bind("<<ListboxSelect>>", self._on_report_structure_select)

        structure_scroll = ttk.Scrollbar(structure_list_frame, orient=tk.VERTICAL, command=self.report_structure_listbox.yview)
        structure_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_structure_listbox.configure(yscrollcommand=structure_scroll.set)

        structure_btn_frame = ttk.Frame(structure_frame)
        structure_btn_frame.pack(fill=tk.X, pady=(8, 0))

        remove_btn = ttk.Button(
            structure_btn_frame,
            text=self.language_manager.translate("ui.buttons.remove_selected", "Remove Selected"),
            command=self._remove_selected_report_element
        )
        remove_btn.pack(side=tk.LEFT, padx=(0, 4))

        clear_btn = ttk.Button(
            structure_btn_frame,
            text=self.language_manager.translate("ui.buttons.clear_all", "Clear All"),
            command=self._clear_report_elements
        )
        clear_btn.pack(side=tk.LEFT, padx=(4, 0))

        reorder_btn_frame = ttk.Frame(structure_frame)
        reorder_btn_frame.pack(fill=tk.X, pady=(6, 0))

        up_btn = ttk.Button(
            reorder_btn_frame,
            text="↑",
            width=4,
            command=lambda: self._move_report_element(-1)
        )
        up_btn.pack(side=tk.LEFT, padx=2)

        down_btn = ttk.Button(
            reorder_btn_frame,
            text="↓",
            width=4,
            command=lambda: self._move_report_element(1)
        )
        down_btn.pack(side=tk.LEFT, padx=2)

        editor_frame = ttk.LabelFrame(right_frame, text=self.language_manager.translate("ui.labels.element_configuration", "Element Configuration"), padding=10)
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 4))
        self.report_editor_frame = editor_frame

        preview_frame = ttk.LabelFrame(right_frame, text=self.language_manager.translate("ui.labels.report_preview", "Report Preview (LaTeX)"), padding=8)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        preview_text = tk.Text(preview_frame, wrap=tk.WORD, font=("Consolas", 9), height=14)
        preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=preview_text.yview)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        preview_text.configure(yscrollcommand=preview_scroll.set)
        self.report_preview_text = preview_text

        status_label = ttk.Label(right_frame, text=self.language_manager.translate("ui.messages.report_ready", "Report editor ready."), font=("Arial", 9, "italic"))
        status_label.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.report_status_label = status_label

        self._update_report_structure_list()

        selected_index = self.report_data.get('selected_index')
        if isinstance(selected_index, int) and 0 <= selected_index < len(self.report_data.get('elements', [])):
            self.report_structure_listbox.selection_clear(0, tk.END)
            self.report_structure_listbox.selection_set(selected_index)
            self.report_structure_listbox.see(selected_index)
            self._show_report_element_editor(selected_index)
        else:
            self._show_report_element_editor(None)

    def _add_report_element(self, element_type: str):
        """Add a report element to the structure list."""
        default_settings = {
            'report_header': {'text': self.language_manager.translate("ui.messages.default_report_header", "Chemometric Studio Report"), 'font_size': 16, 'align': 'center', 'bold': True, 'italic': False, 'underline': False},
            'title': {'text': self.language_manager.translate("ui.messages.default_title", "Title"), 'font_size': 16, 'align': 'left', 'bold': False, 'italic': False, 'underline': False},
            'section': {'text': self.language_manager.translate("ui.messages.default_section", "Section"), 'font_size': 14, 'align': 'left', 'bold': False, 'italic': False, 'underline': False},
            'subsection': {'text': self.language_manager.translate("ui.messages.default_subsection", "Subsection"), 'font_size': 13, 'align': 'left', 'bold': False, 'italic': False, 'underline': False},
            'subsubsection': {'text': self.language_manager.translate("ui.messages.default_subsubsection", "Subsubsection"), 'font_size': 12, 'align': 'left', 'bold': False, 'italic': False, 'underline': False},
            'sample_name_cal': {},
            'sample_name_val': {},
            'sample_metadata_cal': {},
            'sample_metadata_val': {},
            'text': {'text': '', 'font_size': 11, 'align': 'left', 'bold': False, 'italic': False, 'underline': False},
            'graph': {'graph_ref': None, 'title': self.language_manager.translate("ui.messages.default_graph", "Graph"), 'width': 0.9},
            'table': {
                'table_ref': None,
                'title': self.language_manager.translate("ui.messages.default_table", "Table"),
                'max_rows': 50,
                'omit_row_column': False,
                'row_column_header': self.language_manager.translate('ui.labels.row', 'Row')
            },
            'page_break': {},
        }

        new_element = {
            'type': element_type,
            'settings': copy.deepcopy(default_settings.get(element_type, {'text': ''}))
        }

        self.report_data.setdefault('elements', []).append(new_element)
        new_index = len(self.report_data['elements']) - 1
        self.report_data['selected_index'] = new_index
        self._update_report_structure_list()
        if hasattr(self, 'report_structure_listbox'):
            self.report_structure_listbox.selection_clear(0, tk.END)
            self.report_structure_listbox.selection_set(new_index)
            self.report_structure_listbox.see(new_index)
        self._show_report_element_editor(new_index)

    def _update_report_structure_list(self):
        """Refresh report structure listbox from report_data."""
        if not hasattr(self, 'report_structure_listbox'):
            return

        self.report_structure_listbox.delete(0, tk.END)
        elements = self.report_data.get('elements', [])

        for idx, element in enumerate(elements):
            element_type = element.get('type', 'text')
            settings = element.get('settings', {}) or {}
            title_text = settings.get('text') or settings.get('title') or ''
            title_text = str(title_text).strip()
            base_label = self._report_element_label(element_type)
            if title_text:
                display = f"{idx + 1}. {base_label}: {title_text}"
            else:
                display = f"{idx + 1}. {base_label}"
            self.report_structure_listbox.insert(tk.END, display)

    def _report_element_label(self, element_type: str) -> str:
        """Return localized display label for a report element type."""
        key = f"ui.report_elements.{element_type}"
        fallback_map = {
            'report_header': 'Report Header',
            'title': 'Title',
            'section': 'Section',
            'subsection': 'Subsection',
            'subsubsection': 'Subsubsection',
            'sample_name': 'Samples',
            'sample_metadata': 'Metadata',
            'sample_name_cal': 'Samples (C)',
            'sample_name_val': 'Samples (V/T)',
            'sample_metadata_cal': 'Metadata (C)',
            'sample_metadata_val': 'Metadata (V/T)',
            'text': 'Text',
            'graph': 'Graph',
            'table': 'Table',
            'page_break': 'Page Break',
        }
        return self.language_manager.translate(key, fallback_map.get(element_type, element_type))

    def _report_alignment_options(self) -> List[Tuple[str, str]]:
        """Return localized alignment options as (value, label)."""
        return [
            ('left', self.language_manager.translate('ui.align.left', 'Left')),
            ('center', self.language_manager.translate('ui.align.center', 'Center')),
            ('right', self.language_manager.translate('ui.align.right', 'Right')),
        ]

    def _report_alignment_label(self, value: str) -> str:
        """Convert stored alignment value to localized label."""
        for option_value, option_label in self._report_alignment_options():
            if option_value == value:
                return option_label
        return self.language_manager.translate('ui.align.left', 'Left')

    def _report_alignment_value_from_label(self, label: str) -> str:
        """Convert localized alignment label to stored value."""
        for option_value, option_label in self._report_alignment_options():
            if option_label == label:
                return option_value
        return 'left'

    def _report_has_successful_execution(self) -> bool:
        """Return True when at least one analysis instance has successful execution results."""
        if not hasattr(self, 'analysis_data') or not isinstance(self.analysis_data, dict):
            return False
        for analysis_info in self.analysis_data.values():
            execution_results = analysis_info.get('execution_results', {}) if isinstance(analysis_info, dict) else {}
            if execution_results.get('status') == 'success':
                return True
        return False

    def _is_report_page_visible(self, instance_alias: str, page: dict) -> bool:
        """Determine whether a report page is currently visible based on analysis conditions."""
        if not isinstance(page, dict):
            return False
        condition = page.get('condition')
        if not condition:
            return True
        return self._evaluate_condition(instance_alias, condition)

    def _is_report_section_visible(self, instance_alias: str, section: dict) -> bool:
        """Determine whether a report section is currently visible based on analysis conditions."""
        if not isinstance(section, dict):
            return False
        if section.get('type') is None:
            return False
        condition = section.get('condition')
        if not condition:
            return True
        return self._evaluate_condition(instance_alias, condition)

    def _normalize_report_vector(self, value: Any) -> List[str]:
        """Normalize scalar/list/array values into a flat string list."""
        if value is None:
            return []
        try:
            if isinstance(value, np.ndarray):
                arr = value
            elif isinstance(value, (list, tuple)):
                arr = np.array(value, dtype=object)
            else:
                return [str(value)]

            if arr.ndim == 0:
                return [str(arr.item())]
            if arr.ndim == 1:
                return [str(v) for v in arr.tolist()]

            reshaped = arr.reshape(arr.shape[0], -1)
            if reshaped.shape[1] == 1:
                return [str(v) for v in reshaped[:, 0].tolist()]
            return [", ".join(str(v) for v in row) for row in reshaped.tolist()]
        except Exception:
            return []

    def _latest_report_outputs_map(self) -> Dict[str, Any]:
        """Collect latest available output/input values in methodology order."""
        merged: Dict[str, Any] = {}
        if not hasattr(self, 'analysis_data'):
            return merged

        for instance_alias in self.methodology_list:
            analysis_info = self.analysis_data.get(instance_alias, {})
            execution_results = analysis_info.get('execution_results', {})
            if execution_results.get('status') != 'success':
                continue
            sources = self._get_execution_data_sources(execution_results, instance_alias)
            if isinstance(sources, dict):
                merged.update(sources)
        return merged

    def _flatten_report_metadata_entry(self, value: Any, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested metadata entry dictionaries to dotted keys."""
        if not isinstance(value, dict):
            return {prefix or 'value': value}
        flat: Dict[str, Any] = {}
        for key, child in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, dict):
                flat.update(self._flatten_report_metadata_entry(child, child_key))
            else:
                flat[child_key] = child
        return flat

    def _build_report_sample_element(self, mode: str) -> Dict[str, Any]:
        """Build a sample table element for calibration ('cal') or validation/test ('val')."""
        outputs = self._latest_report_outputs_map()

        def _first_existing(keys: List[str]) -> Any:
            for key in keys:
                if key in outputs:
                    return outputs.get(key)
            return None

        sample_col_name = self.language_manager.translate('ui.labels.sample_name_column', 'Sample Name')
        y_col_name = self.language_manager.translate('ui.labels.y_value_column', 'Y')
        class_col_name = self.language_manager.translate('ui.labels.class_column', 'Class')
        sample_row_header = self.language_manager.translate('ui.labels.sample', 'Sample')

        if mode == 'cal':
            samples = self._normalize_report_vector(_first_existing(['smp_cal']))
            y_vals = self._normalize_report_vector(_first_existing(['Y_cal', 'y_cal']))
            class_vals = self._normalize_report_vector(_first_existing(['class_data_cal']))
            table_title = self.language_manager.translate('ui.messages.samples_c_title', 'Samples (C)')
        else:
            samples = self._normalize_report_vector(_first_existing(['smp_val']))
            y_vals = self._normalize_report_vector(_first_existing(['Y_val', 'Yval', 'y_val']))
            class_vals = self._normalize_report_vector(_first_existing(['class_data_val']))
            table_title = self.language_manager.translate('ui.messages.samples_vt_title', 'Samples (V/T)')

        active_headers: List[str] = [sample_col_name]
        active_columns: List[List[str]] = [samples]
        if len(y_vals) > 0:
            active_headers.append(y_col_name)
            active_columns.append(y_vals)
        if len(class_vals) > 0:
            active_headers.append(class_col_name)
            active_columns.append(class_vals)

        n_rows = max([len(col) for col in active_columns], default=0)
        rows: List[List[str]] = []
        row_labels: List[str] = []
        for i in range(n_rows):
            row = []
            for col in active_columns:
                row.append(col[i] if i < len(col) else '')
            rows.append(row)
            row_labels.append(str(i + 1))

        return {
            'type': 'table',
            'settings': {
                'title': table_title,
                'headers': active_headers,
                'rows': rows,
                'row_labels': row_labels,
                'omit_row_column': False,
                'row_column_header': sample_row_header,
            }
        }

    def _build_report_metadata_element(self, mode: str) -> Dict[str, Any]:
        """Build a metadata table element for calibration ('cal') or validation/test ('val')."""
        outputs = self._latest_report_outputs_map()
        sample_row_header = self.language_manager.translate('ui.labels.sample', 'Sample')

        def _build_table_from_metadata(meta_value: Any, title: str) -> Dict[str, Any]:
            metadata = meta_value if isinstance(meta_value, dict) else {}

            redundant_tail_keys = {
                'file_name',
                'filename',
                'file_extension',
                'extension',
                'ext',
            }

            preferred_columns = [
                ('index', 'Index'),
                ('label', 'Label'),
                ('source', 'Source'),
                ('file', 'File'),
                ('file_size_bytes', 'File Size (Bytes)'),
                ('created', 'Created'),
                ('modified', 'Modified'),
                ('row_index', 'Row Index'),
            ]

            alias_map = {
                'index': ['index', 'idx', 'sample_index'],
                'label': ['label', 'name', 'sample_label', 'sample_name'],
                'source': ['source', 'source_mode', 'origin', 'dataset', 'group', 'class'],
                'file': ['file', 'path', 'filepath', 'file_path', 'source_file', 'filename', 'file_name'],
                'file_size_bytes': ['file_size_bytes', 'size_bytes', 'filesize', 'file_size', 'size', 'bytes'],
                'created': ['created', 'created_time', 'created_at', 'creation_date', 'date_created'],
                'modified': ['modified', 'modified_time', 'modified_at', 'last_modified', 'date_modified', 'updated_at'],
                'row_index': ['row_index', 'row', 'row_id', 'line', 'line_index'],
            }

            def _tail_key(key: str) -> str:
                return str(key).split('.')[-1].strip().lower().replace(' ', '_')

            def _find_value_by_alias(flat_item: Dict[str, Any], aliases: List[str]) -> Any:
                for alias in aliases:
                    for existing_key, existing_value in flat_item.items():
                        if _tail_key(existing_key) == alias:
                            return existing_value
                return ''

            def _is_redundant_metadata_header(header_key: str) -> bool:
                tail = str(header_key).split('.')[-1].strip().lower().replace(' ', '_')
                return tail in redundant_tail_keys

            def _normalize_metadata_value(value: Any) -> str:
                text = '' if value is None else str(value)
                if ('\\' in text) or ('/' in text):
                    candidate = os.path.basename(text.replace('\\', '/'))
                    if candidate:
                        return candidate
                return text

            def _strip_fractional_seconds(value: Any) -> str:
                text = '' if value is None else str(value)
                if '.' in text:
                    head, tail = text.split('.', 1)
                    suffix = ''
                    if 'Z' in tail:
                        suffix = 'Z'
                        tail = tail.replace('Z', '', 1)
                    if '+' in tail:
                        tz = tail[tail.index('+'):]
                        suffix = tz
                    elif '-' in tail and 'T' in head:
                        minus_pos = tail.find('-')
                        if minus_pos > 0:
                            suffix = tail[minus_pos:]
                    text = head + suffix
                return text.replace('T', ' ')

            flattened_entries: List[Dict[str, Any]] = []
            for _key, entry in metadata.items():
                flattened_entries.append(self._flatten_report_metadata_entry(entry))

            normalized_rows: List[Dict[str, str]] = []
            for entry_idx, item in enumerate(flattened_entries):
                normalized_item: Dict[str, str] = {}
                for canonical_key, _display in preferred_columns:
                    raw_value = _find_value_by_alias(item, alias_map.get(canonical_key, []))
                    if canonical_key == 'index' and (raw_value is None or str(raw_value).strip() == ''):
                        raw_value = item.get('sample_index', '')
                    if canonical_key == 'label' and (raw_value is None or str(raw_value).strip() == ''):
                        raw_value = item.get('sample_label', '')
                    if canonical_key == 'row_index' and (raw_value is None or str(raw_value).strip() == ''):
                        fallback_index = item.get('sample_index', '')
                        raw_value = fallback_index if str(fallback_index).strip() != '' else (entry_idx + 1)
                    if canonical_key == 'file':
                        normalized_item[canonical_key] = _normalize_metadata_value(raw_value)
                    elif canonical_key in ('created', 'modified'):
                        normalized_item[canonical_key] = _strip_fractional_seconds(raw_value)
                    else:
                        normalized_item[canonical_key] = '' if raw_value is None else str(raw_value)
                normalized_rows.append(normalized_item)

            all_headers: List[str] = [display_name for _canonical_key, display_name in preferred_columns]

            rows: List[List[str]] = []
            row_labels: List[str] = []
            for idx, normalized_item in enumerate(normalized_rows):
                rows.append([normalized_item.get(canonical_key, '') for canonical_key, _display_name in preferred_columns])
                row_labels.append(str(idx + 1))

            return {
                'type': 'table',
                'settings': {
                    'title': title,
                    'headers': all_headers,
                    'rows': rows,
                    'row_labels': row_labels,
                    'omit_row_column': True,
                    'row_column_header': sample_row_header,
                    'force_landscape': True,
                    'is_metadata': True,
                }
            }

        if mode == 'cal':
            return _build_table_from_metadata(outputs.get('cal_metadata'), self.language_manager.translate('ui.messages.metadata_c_title', 'Metadata (C)'))
        return _build_table_from_metadata(outputs.get('val_metadata'), self.language_manager.translate('ui.messages.metadata_vt_title', 'Metadata (V/T)'))

    def _on_report_structure_select(self, event=None):
        """Handle report structure selection and show editor for selected element."""
        if not hasattr(self, 'report_structure_listbox'):
            return
        selection = self.report_structure_listbox.curselection()
        if not selection:
            self.report_data['selected_index'] = None
            self._show_report_element_editor(None)
            return
        selected_index = selection[0]
        self.report_data['selected_index'] = selected_index
        self._show_report_element_editor(selected_index)

    def _show_report_element_editor(self, index: Optional[int]):
        """Render dynamic configuration controls for a report element."""
        if not hasattr(self, 'report_editor_frame'):
            return

        for child in self.report_editor_frame.winfo_children():
            child.destroy()

        elements = self.report_data.get('elements', [])
        if index is None or index < 0 or index >= len(elements):
            info = ttk.Label(
                self.report_editor_frame,
                text=self.language_manager.translate("ui.messages.select_or_add_element", "Select an item in Structure or add a new element."),
                font=("Arial", 10, "italic")
            )
            info.pack(anchor='w', pady=4)
            return

        element = elements[index]
        element_type = element.get('type', 'text')
        settings = element.setdefault('settings', {})

        header = ttk.Label(
            self.report_editor_frame,
            text=f"{self.language_manager.translate('ui.labels.editing', 'Editing')}: {self._report_element_label(element_type)}",
            font=("Arial", 10, "bold")
        )
        header.pack(anchor='w', pady=(0, 8))

        if element_type in ('report_header', 'title', 'section', 'subsection', 'subsubsection', 'text'):
            ttk.Label(self.report_editor_frame, text=self.language_manager.translate("ui.labels.text_content", "Text")).pack(anchor='w', pady=(0, 2))
            text_widget = tk.Text(self.report_editor_frame, height=8, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
            text_widget.insert('1.0', settings.get('text', ''))

            def on_text_change(event=None, idx=index):
                if not text_widget.winfo_exists():
                    return
                self._update_report_element_setting(idx, 'text', text_widget.get('1.0', tk.END).strip())

            text_widget.bind('<KeyRelease>', on_text_change)

            options_frame = ttk.Frame(self.report_editor_frame)
            options_frame.pack(fill=tk.X)

            ttk.Label(options_frame, text=self.language_manager.translate("ui.labels.font_size", "Font size")).grid(row=0, column=0, sticky='w', padx=(0, 6), pady=2)
            font_var = tk.IntVar(value=int(settings.get('font_size', 11)))
            font_spin = ttk.Spinbox(options_frame, from_=8, to=28, textvariable=font_var, width=8)
            font_spin.grid(row=0, column=1, sticky='w', pady=2)
            font_spin.configure(command=lambda idx=index: self._update_report_element_setting(idx, 'font_size', int(font_var.get())))

            ttk.Label(options_frame, text=self.language_manager.translate("ui.labels.alignment", "Alignment")).grid(row=1, column=0, sticky='w', padx=(0, 6), pady=2)
            align_var = tk.StringVar(value=self._report_alignment_label(settings.get('align', 'left')))
            align_labels = [label for _, label in self._report_alignment_options()]
            align_combo = ttk.Combobox(options_frame, textvariable=align_var, state='readonly', values=align_labels, width=10)
            align_combo.grid(row=1, column=1, sticky='w', pady=2)
            align_combo.bind('<<ComboboxSelected>>', lambda e, idx=index: self._update_report_element_setting(idx, 'align', self._report_alignment_value_from_label(align_var.get())))

            font_spin.bind('<KeyRelease>', lambda e, idx=index: self._update_report_element_setting(idx, 'font_size', int(font_var.get()) if str(font_var.get()).isdigit() else 11))

            style_frame = ttk.Frame(self.report_editor_frame)
            style_frame.pack(fill=tk.X, pady=(8, 0))

            bold_var = tk.BooleanVar(value=bool(settings.get('bold', False)))
            italic_var = tk.BooleanVar(value=bool(settings.get('italic', False)))
            underline_var = tk.BooleanVar(value=bool(settings.get('underline', False)))

            bold_btn = ttk.Checkbutton(style_frame, text='B', variable=bold_var,
                                       command=lambda idx=index: self._update_report_element_setting(idx, 'bold', bool(bold_var.get())))
            bold_btn.pack(side=tk.LEFT, padx=(0, 8))

            italic_btn = ttk.Checkbutton(style_frame, text='I', variable=italic_var,
                                         command=lambda idx=index: self._update_report_element_setting(idx, 'italic', bool(italic_var.get())))
            italic_btn.pack(side=tk.LEFT, padx=(0, 8))

            underline_btn = ttk.Checkbutton(style_frame, text='U', variable=underline_var,
                                            command=lambda idx=index: self._update_report_element_setting(idx, 'underline', bool(underline_var.get())))
            underline_btn.pack(side=tk.LEFT)

        elif element_type in ('sample_name', 'sample_metadata', 'sample_name_cal', 'sample_name_val', 'sample_metadata_cal', 'sample_metadata_val'):
            if element_type in ('sample_name', 'sample_name_cal', 'sample_name_val'):
                message_key = 'ui.messages.samples_element_info'
                fallback = 'This element auto-generates sample tables from latest smp_*/Y_*/class_data_* values.'
            else:
                message_key = 'ui.messages.metadata_element_info'
                fallback = 'This element auto-generates metadata tables from latest cal_metadata/val_metadata values.'
            ttk.Label(
                self.report_editor_frame,
                text=self.language_manager.translate(message_key, fallback),
                justify=tk.LEFT,
                wraplength=520
            ).pack(anchor='w')

        elif element_type == 'graph':
            graph_options = self._collect_report_graph_options()
            ttk.Label(self.report_editor_frame, text=self.language_manager.translate("ui.labels.graph_source", "Graph source")).pack(anchor='w', pady=(0, 2))

            option_labels = [entry['label'] for entry in graph_options]
            selected_ref = settings.get('graph_ref')
            initial_label = ''
            for entry in graph_options:
                if entry['ref'] == selected_ref:
                    initial_label = entry['label']
                    break
            if not initial_label and option_labels:
                initial_label = option_labels[0]
                self._update_report_element_setting(index, 'graph_ref', graph_options[0]['ref'])

            source_var = tk.StringVar(value=initial_label)
            source_combo = ttk.Combobox(self.report_editor_frame, textvariable=source_var, values=option_labels, state='readonly')
            source_combo.pack(fill=tk.X, pady=(0, 8))
            if not option_labels:
                source_combo.configure(state='disabled')

            def on_graph_source_change(event=None, idx=index):
                label = source_var.get()
                for entry in graph_options:
                    if entry['label'] == label:
                        self._update_report_element_setting(idx, 'graph_ref', entry['ref'])
                        break

            source_combo.bind('<<ComboboxSelected>>', on_graph_source_change)

            ttk.Label(self.report_editor_frame, text=self.language_manager.translate("ui.labels.title", "Title")).pack(anchor='w', pady=(0, 2))
            title_var = tk.StringVar(value=settings.get('title', self.language_manager.translate("ui.messages.default_graph", "Graph")))
            title_entry = ttk.Entry(self.report_editor_frame, textvariable=title_var)
            title_entry.pack(fill=tk.X, pady=(0, 8))
            title_entry.bind('<KeyRelease>', lambda e, idx=index: self._update_report_element_setting(idx, 'title', title_var.get()))

            width_frame = ttk.Frame(self.report_editor_frame)
            width_frame.pack(fill=tk.X)
            ttk.Label(width_frame, text=self.language_manager.translate("ui.labels.width_ratio", "Width (0.3 - 1.0)")).pack(side=tk.LEFT)
            width_var = tk.DoubleVar(value=float(settings.get('width', 0.9)))
            width_spin = ttk.Spinbox(width_frame, from_=0.3, to=1.0, increment=0.05, textvariable=width_var, width=8)
            width_spin.pack(side=tk.LEFT, padx=8)
            width_spin.configure(command=lambda idx=index: self._update_report_element_setting(idx, 'width', float(width_var.get())))
            width_spin.bind('<KeyRelease>', lambda e, idx=index: self._update_report_element_setting(idx, 'width', float(width_var.get()) if str(width_var.get()) else 0.9))

            if not option_labels:
                ttk.Label(
                    self.report_editor_frame,
                    text=self.language_manager.translate("ui.messages.no_graphs_available", "No analysis graphs available. Run analysis and configure graph sections first."),
                    foreground='gray'
                ).pack(anchor='w', pady=(8, 0))

        elif element_type == 'table':
            table_options = self._collect_report_table_options()
            ttk.Label(self.report_editor_frame, text=self.language_manager.translate("ui.labels.table_source", "Table source")).pack(anchor='w', pady=(0, 2))

            option_labels = [entry['label'] for entry in table_options]
            selected_ref = settings.get('table_ref')
            initial_label = ''
            for entry in table_options:
                if entry['ref'] == selected_ref:
                    initial_label = entry['label']
                    break
            if not initial_label and option_labels:
                initial_label = option_labels[0]
                self._update_report_element_setting(index, 'table_ref', table_options[0]['ref'])

            source_var = tk.StringVar(value=initial_label)
            source_combo = ttk.Combobox(self.report_editor_frame, textvariable=source_var, values=option_labels, state='readonly')
            source_combo.pack(fill=tk.X, pady=(0, 8))
            if not option_labels:
                source_combo.configure(state='disabled')

            def on_table_source_change(event=None, idx=index):
                label = source_var.get()
                for entry in table_options:
                    if entry['label'] == label:
                        self._update_report_element_setting(idx, 'table_ref', entry['ref'])
                        break

            source_combo.bind('<<ComboboxSelected>>', on_table_source_change)

            ttk.Label(self.report_editor_frame, text=self.language_manager.translate("ui.labels.title", "Title")).pack(anchor='w', pady=(0, 2))
            title_var = tk.StringVar(value=settings.get('title', self.language_manager.translate("ui.messages.default_table", "Table")))
            title_entry = ttk.Entry(self.report_editor_frame, textvariable=title_var)
            title_entry.pack(fill=tk.X, pady=(0, 8))
            title_entry.bind('<KeyRelease>', lambda e, idx=index: self._update_report_element_setting(idx, 'title', title_var.get()))

            max_rows_frame = ttk.Frame(self.report_editor_frame)
            max_rows_frame.pack(fill=tk.X)
            ttk.Label(max_rows_frame, text=self.language_manager.translate("ui.labels.max_rows", "Max rows")).pack(side=tk.LEFT)
            max_rows_var = tk.IntVar(value=int(settings.get('max_rows', 50)))
            max_rows_spin = ttk.Spinbox(max_rows_frame, from_=5, to=500, increment=5, textvariable=max_rows_var, width=8)
            max_rows_spin.pack(side=tk.LEFT, padx=8)
            max_rows_spin.configure(command=lambda idx=index: self._update_report_element_setting(idx, 'max_rows', int(max_rows_var.get())))
            max_rows_spin.bind('<KeyRelease>', lambda e, idx=index: self._update_report_element_setting(idx, 'max_rows', int(max_rows_var.get()) if str(max_rows_var.get()).isdigit() else 50))

            row_options_frame = ttk.Frame(self.report_editor_frame)
            row_options_frame.pack(fill=tk.X, pady=(8, 0))

            omit_row_var = tk.BooleanVar(value=bool(settings.get('omit_row_column', False)))
            omit_row_check = ttk.Checkbutton(
                row_options_frame,
                text=self.language_manager.translate('ui.labels.omit_row_column', 'Omit row column'),
                variable=omit_row_var,
                command=lambda idx=index: self._update_report_element_setting(idx, 'omit_row_column', bool(omit_row_var.get()))
            )
            omit_row_check.pack(anchor='w')

            row_header_frame = ttk.Frame(self.report_editor_frame)
            row_header_frame.pack(fill=tk.X, pady=(4, 0))
            ttk.Label(row_header_frame, text=self.language_manager.translate('ui.labels.row_column_header', 'Row column header')).pack(side=tk.LEFT)
            row_header_var = tk.StringVar(value=settings.get('row_column_header', self.language_manager.translate('ui.labels.row', 'Row')))
            row_header_entry = ttk.Entry(row_header_frame, textvariable=row_header_var)
            row_header_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
            row_header_entry.bind('<KeyRelease>', lambda e, idx=index: self._update_report_element_setting(idx, 'row_column_header', row_header_var.get()))

            if not option_labels:
                ttk.Label(
                    self.report_editor_frame,
                    text=self.language_manager.translate("ui.messages.no_tables_available", "No analysis tables available. Run analysis and configure table sections first."),
                    foreground='gray'
                ).pack(anchor='w', pady=(8, 0))

        elif element_type == 'page_break':
            ttk.Label(
                self.report_editor_frame,
                text=self.language_manager.translate("ui.messages.page_break_description", "This element inserts a page break in the report."),
                font=("Arial", 10, "italic")
            ).pack(anchor='w')

    def _update_report_element_setting(self, index: int, key: str, value: Any):
        """Update one setting key of a report element and refresh structure labels."""
        elements = self.report_data.get('elements', [])
        if not (0 <= index < len(elements)):
            return

        settings = elements[index].setdefault('settings', {})
        settings[key] = value
        if key == 'row_column_header':
            settings['row_column_header_custom'] = True
        self._update_report_structure_list()
        if hasattr(self, 'report_structure_listbox'):
            self.report_structure_listbox.selection_clear(0, tk.END)
            self.report_structure_listbox.selection_set(index)

    def _remove_selected_report_element(self):
        """Remove selected element from report structure."""
        selected_index = self.report_data.get('selected_index')
        elements = self.report_data.get('elements', [])
        if selected_index is None or not (0 <= selected_index < len(elements)):
            return

        del elements[selected_index]
        if not elements:
            self.report_data['selected_index'] = None
            self._update_report_structure_list()
            self._show_report_element_editor(None)
            return

        new_index = min(selected_index, len(elements) - 1)
        self.report_data['selected_index'] = new_index
        self._update_report_structure_list()
        if hasattr(self, 'report_structure_listbox'):
            self.report_structure_listbox.selection_clear(0, tk.END)
            self.report_structure_listbox.selection_set(new_index)
            self.report_structure_listbox.see(new_index)
        self._show_report_element_editor(new_index)

    def _clear_report_elements(self):
        """Remove all report elements from the structure."""
        self.report_data['elements'] = []
        self.report_data['selected_index'] = None
        self._update_report_structure_list()
        self._show_report_element_editor(None)

    def _move_report_element(self, direction: int):
        """Move selected report element up or down in structure."""
        selected_index = self.report_data.get('selected_index')
        elements = self.report_data.get('elements', [])
        if selected_index is None or not (0 <= selected_index < len(elements)):
            return

        new_index = selected_index + direction
        if new_index < 0 or new_index >= len(elements):
            return

        elements[selected_index], elements[new_index] = elements[new_index], elements[selected_index]
        self.report_data['selected_index'] = new_index
        self._update_report_structure_list()
        if hasattr(self, 'report_structure_listbox'):
            self.report_structure_listbox.selection_clear(0, tk.END)
            self.report_structure_listbox.selection_set(new_index)
            self.report_structure_listbox.see(new_index)
        self._show_report_element_editor(new_index)

    def _collect_report_graph_options(self) -> List[Dict[str, Any]]:
        """Collect graph sections from all analysis tabs for report selection."""
        options: List[Dict[str, Any]] = []
        if not hasattr(self, 'analysis_data'):
            return options

        for instance_alias in self.methodology_list:
            if instance_alias not in self.analysis_data:
                continue
            analysis_info = self.analysis_data.get(instance_alias, {})
            execution_results = analysis_info.get('execution_results', {})
            if execution_results.get('status') != 'success':
                continue
            pages = analysis_info.get('pages', [])
            for page_idx, page in enumerate(pages):
                if not self._is_report_page_visible(instance_alias, page):
                    continue
                page_title = page.get('title', f"{self.language_manager.translate('ui.messages.page_label', 'Page')} {page_idx + 1}")
                sections = page.get('sections', [])
                for section_idx, section in enumerate(sections):
                    if section.get('type') != 'graph':
                        continue
                    if not self._is_report_section_visible(instance_alias, section):
                        continue
                    config = section.get('config', {})
                    section_title = config.get('graph_title') or config.get('title') or f"{self.language_manager.translate('ui.report_elements.graph', 'Graph')} {section_idx + 1}"
                    label = f"{instance_alias} > {page_title} > {section_title}"
                    options.append({
                        'label': label,
                        'ref': {
                            'instance_alias': instance_alias,
                            'page_idx': page_idx,
                            'section_idx': section_idx
                        }
                    })
        return options

    def _collect_report_table_options(self) -> List[Dict[str, Any]]:
        """Collect table sections from all analysis tabs for report selection."""
        options: List[Dict[str, Any]] = []
        if not hasattr(self, 'analysis_data'):
            return options

        for instance_alias, analysis_info in self.analysis_data.items():
            pages = analysis_info.get('pages', [])
            for page_idx, page in enumerate(pages):
                page_title = page.get('title', f"{self.language_manager.translate('ui.messages.page_label', 'Page')} {page_idx + 1}")
                sections = page.get('sections', [])
                for section_idx, section in enumerate(sections):
                    if section.get('type') != 'table':
                        continue
                    config = section.get('config', {})
                    section_title = config.get('table_title') or config.get('title') or f"{self.language_manager.translate('ui.report_elements.table', 'Table')} {section_idx + 1}"
                    label = f"{instance_alias} > {page_title} > {section_title}"
                    options.append({
                        'label': label,
                        'ref': {
                            'instance_alias': instance_alias,
                            'page_idx': page_idx,
                            'section_idx': section_idx
                        }
                    })
        return options

    def _get_report_section(self, ref: Dict[str, Any]) -> Tuple[Optional[str], Optional[dict], Optional[dict]]:
        """Resolve report reference to (instance_alias, section_data, analysis_info)."""
        if not ref or not isinstance(ref, dict):
            return None, None, None

        instance_alias = ref.get('instance_alias')
        page_idx = ref.get('page_idx')
        section_idx = ref.get('section_idx')

        if instance_alias is None or page_idx is None or section_idx is None:
            return None, None, None
        if not hasattr(self, 'analysis_data') or instance_alias not in self.analysis_data:
            return None, None, None

        analysis_info = self.analysis_data[instance_alias]
        pages = analysis_info.get('pages', [])
        if page_idx < 0 or page_idx >= len(pages):
            return None, None, None
        sections = pages[page_idx].get('sections', [])
        if section_idx < 0 or section_idx >= len(sections):
            return None, None, None

        return instance_alias, sections[section_idx], analysis_info

    def _build_report_footer_text(self) -> str:
        """Build mandatory report footer text with version, OS, and timestamp."""
        version = "0.0"
        try:
            about_path = Path(__file__).parent / "about_us.json"
            if about_path.exists():
                with open(about_path, encoding='utf-8') as f:
                    about_data = json.load(f)
                version = about_data.get('version', version)
        except Exception:
            pass

        def _build_os_name_with_version() -> str:
            unknown_os = self.language_manager.translate('ui.messages.unknown_os', 'Unknown OS')
            system_name = (platform.system() or '').strip()
            release = (platform.release() or '').strip()
            version_info = (platform.version() or '').strip()
            machine = (platform.machine() or '').strip()
            platform_line = (platform.platform() or '').strip()

            if not system_name:
                return unknown_os

            if system_name == 'Windows':
                parts = [f"Windows {release}".strip()]
                if version_info:
                    parts.append(f"version {version_info}")
                if machine:
                    parts.append(machine)
                return " | ".join([p for p in parts if p])

            if system_name == 'Darwin':
                mac_ver = (platform.mac_ver()[0] or '').strip()
                base = f"macOS {mac_ver}".strip() if mac_ver else 'macOS'
                extras = []
                if release:
                    extras.append(f"kernel {release}")
                if machine:
                    extras.append(machine)
                return " | ".join([base] + extras) if extras else base

            lower_platform = platform_line.lower()
            lower_version = version_info.lower()
            lower_release = release.lower()
            is_android = ('android' in lower_platform) or ('android' in lower_version) or ('android' in lower_release)
            if is_android:
                base = 'Android'
                extras = []
                if release:
                    extras.append(f"kernel {release}")
                if machine:
                    extras.append(machine)
                return " | ".join([base] + extras) if extras else base

            if system_name == 'Linux':
                libc_name, libc_version = platform.libc_ver()
                base = f"Linux {release}".strip() if release else 'Linux'
                extras = []
                if libc_name and libc_version:
                    extras.append(f"{libc_name} {libc_version}")
                if machine:
                    extras.append(machine)
                return " | ".join([base] + extras) if extras else base

            parts = [system_name]
            if release:
                parts.append(release)
            if version_info:
                parts.append(f"version {version_info}")
            if machine:
                parts.append(machine)
            return " | ".join(parts)

        os_name = _build_os_name_with_version()
        timestamp = datetime.now().strftime("%I:%M %p - %d/%m/%Y")
        return self.language_manager.translate(
            'ui.messages.report_footer_template',
            'Model created in Chemometric Studio v{version} on a {os_name} system at {timestamp}'
        ).format(version=version, os_name=os_name, timestamp=timestamp)

    def _resolve_report_graph_image(self, graph_ref: Dict[str, Any], assets_dir: Path) -> Optional[str]:
        """Resolve graph reference to an image path for LaTeX embedding."""
        instance_alias, section_data, analysis_info = self._get_report_section(graph_ref)
        if not instance_alias or not section_data:
            return None

        page_idx = graph_ref.get('page_idx')
        section_idx = graph_ref.get('section_idx')
        section_id = (page_idx, section_idx)

        assets_dir.mkdir(parents=True, exist_ok=True)
        image_path = assets_dir / f"graph_{instance_alias}_{page_idx}_{section_idx}.png"

        try:
            graph_canvases = analysis_info.get('graph_canvases', {})
            if section_id in graph_canvases:
                canvas, _frame = graph_canvases[section_id]
                fig = canvas.figure
                fig.savefig(str(image_path), dpi=220, bbox_inches='tight')
                return str(image_path).replace('\\', '/')
        except Exception:
            pass

        try:
            metadata_map = analysis_info.get('graph_data_metadata', {})
            metadata = metadata_map.get(section_id)
            if not metadata:
                metadata = None
                pages = analysis_info.get('pages', [])
                section_data = None
                if 0 <= page_idx < len(pages):
                    sections = pages[page_idx].get('sections', [])
                    if 0 <= section_idx < len(sections):
                        section_data = sections[section_idx]

                if section_data and section_data.get('type') == 'graph':
                    original_page = analysis_info.get('current_page', 0)
                    temp_window = None
                    temp_frame = None
                    try:
                        analysis_info['current_page'] = page_idx
                        temp_window = tk.Toplevel(self.root)
                        temp_window.withdraw()
                        temp_frame = ttk.Frame(temp_window)
                        temp_frame.pack(fill=tk.BOTH, expand=True)

                        self._render_graph_section(temp_frame, instance_alias, section_data, section_idx)
                        self.root.update_idletasks()

                        graph_canvases = analysis_info.get('graph_canvases', {})
                        if section_id in graph_canvases:
                            canvas, _frame = graph_canvases[section_id]
                            fig = canvas.figure
                            fig.savefig(str(image_path), dpi=220, bbox_inches='tight')
                            return str(image_path).replace('\\', '/')

                        metadata_map = analysis_info.get('graph_data_metadata', {})
                        metadata = metadata_map.get(section_id)
                    finally:
                        analysis_info['current_page'] = original_page
                        if temp_window is not None and temp_window.winfo_exists():
                            temp_window.destroy()

                if not metadata:
                    return None

            config = metadata.get('config', {}).copy()
            graph_type = metadata.get('graph_type', config.get('graph_type', 'scatter'))
            x_data = metadata.get('x_data')
            y_data = metadata.get('y_data')
            z_data = metadata.get('z_data')
            x_axis = metadata.get('x_axis_config', config.get('x_axis', {}))
            y_axis = metadata.get('y_axis_config', config.get('y_axis', {}))
            datasets = metadata.get('extracted_datasets')

            fig, _ax = graph_renderer.render_graph_figure(
                graph_type,
                config,
                x_data,
                y_data,
                z_data,
                x_axis,
                y_axis,
                default_cmap=self.settings_manager.get('colormap', 'viridis'),
                datasets=datasets,
                qualitative_cmap=self.settings_manager.get('qualitative_colormap', 'tab10')
            )
            fig.savefig(str(image_path), dpi=220, bbox_inches='tight')
            plt.close(fig)
            return str(image_path).replace('\\', '/')
        except Exception:
            return None

    def _resolve_report_table_data(self, table_ref: Dict[str, Any], max_rows: int = 50) -> Tuple[List[str], List[List[str]], List[str], str]:
        """Resolve table reference to headers, row values, row labels, and row-header title."""
        instance_alias, section_data, analysis_info = self._get_report_section(table_ref)
        if not instance_alias or not section_data:
            return [], [], [], self.language_manager.translate('ui.labels.row', 'Row')

        config = section_data.get('config', {})
        execution_results = analysis_info.get('execution_results', {})
        if not execution_results or execution_results.get('status') != 'success':
            return [], [], [], self.language_manager.translate('ui.labels.row', 'Row')

        outputs = self._get_execution_data_sources(execution_results, instance_alias)

        data = None
        headers = config.get('column_headers') or []
        row_header_title = str(config.get('row_label', self.language_manager.translate('ui.labels.row', 'Row')))
        row_headers_config = config.get('row_headers')

        columns_config = config.get('columns')
        if columns_config:
            data_columns = []
            local_headers = []
            for col_spec in columns_config:
                source = col_spec.get('data_source')
                nested_key = col_spec.get('nested_key')
                col_name = col_spec.get('name', source)
                column_data = self._get_data_from_source(outputs, source, nested_key)
                if column_data is None:
                    continue
                column_arr = np.array(column_data).flatten()
                data_columns.append(column_arr)
                local_headers.append(col_name)
            if data_columns:
                min_len = min(len(col) for col in data_columns)
                trimmed = [col[:min_len] for col in data_columns]
                data = np.column_stack(trimmed)
                headers = local_headers
        else:
            data_source = config.get('data_source')
            nested_key = config.get('nested_key')
            raw_data = self._get_data_from_source(outputs, data_source, nested_key)
            if raw_data is not None:
                data = np.array(raw_data)

        if data is None:
            return headers, [], [], row_header_title

        nav_axes = config.get('data_slicing', [])
        if nav_axes and data.ndim > 2:
            indices = {}
            for nav_item in nav_axes:
                if isinstance(nav_item, dict):
                    dim = nav_item.get('dimension', 0)
                    indices[dim] = nav_item.get('default', 0)
            if indices:
                data = self._extract_sliced_data(data, indices)

        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)

        row_limit = max(1, int(max_rows))
        data = data[:row_limit]

        if not headers:
            headers = [f"{self.language_manager.translate('ui.labels.column_name_plain', 'Column')} {i + 1}" for i in range(data.shape[1])]
        else:
            headers = [str(h) for h in headers[:data.shape[1]]]
            if len(headers) < data.shape[1]:
                headers.extend([f"{self.language_manager.translate('ui.labels.column_name_plain', 'Column')} {i + 1}" for i in range(len(headers), data.shape[1])])

        decimal_places = int(config.get('decimal_places', 4))
        rows: List[List[str]] = []
        for row in data:
            row_values: List[str] = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    row_values.append(f"{value:.{decimal_places}f}")
                else:
                    row_values.append(str(value))
            rows.append(row_values)

        row_labels: List[str] = []
        if isinstance(row_headers_config, (list, np.ndarray)):
            row_labels = [str(v) for v in list(row_headers_config)[:len(rows)]]
        else:
            row_labels = [str(i + 1) for i in range(len(rows))]

        return headers, rows, row_labels, row_header_title

    def _build_resolved_report_elements(self, assets_dir: Path) -> List[Dict[str, Any]]:
        """Build report elements with resolved graph images and table data."""
        resolved: List[Dict[str, Any]] = []
        for element in self.report_data.get('elements', []):
            element_type = element.get('type', 'text')
            settings = copy.deepcopy(element.get('settings', {}))

            if element_type == 'graph':
                graph_ref = settings.get('graph_ref')
                settings['image_path'] = self._resolve_report_graph_image(graph_ref, assets_dir) if graph_ref else None
                resolved.append({'type': element_type, 'settings': settings})
            elif element_type == 'table':
                table_ref = settings.get('table_ref')
                max_rows = int(settings.get('max_rows', 50))
                headers, rows, row_labels, row_header_title = self._resolve_report_table_data(table_ref, max_rows=max_rows) if table_ref else ([], [], [], self.language_manager.translate('ui.labels.row', 'Row'))
                settings['headers'] = headers
                settings['rows'] = rows
                settings['row_labels'] = row_labels
                default_row_label = self.language_manager.translate('ui.labels.row', 'Row')
                existing_row_header = settings.get('row_column_header', '')
                is_custom_row_header = bool(settings.get('row_column_header_custom', False))
                if is_custom_row_header:
                    settings['row_column_header'] = existing_row_header
                else:
                    if not existing_row_header or str(existing_row_header).strip() in ('', default_row_label):
                        settings['row_column_header'] = row_header_title
                    else:
                        settings['row_column_header'] = row_header_title
                settings['omit_row_column'] = bool(settings.get('omit_row_column', False))
                resolved.append({'type': element_type, 'settings': settings})
            elif element_type == 'sample_name':
                resolved.append(self._build_report_sample_element('cal'))
                resolved.append(self._build_report_sample_element('val'))
            elif element_type == 'sample_metadata':
                resolved.append(self._build_report_metadata_element('cal'))
                resolved.append(self._build_report_metadata_element('val'))
            elif element_type == 'sample_name_cal':
                resolved.append(self._build_report_sample_element('cal'))
            elif element_type == 'sample_name_val':
                resolved.append(self._build_report_sample_element('val'))
            elif element_type == 'sample_metadata_cal':
                resolved.append(self._build_report_metadata_element('cal'))
            elif element_type == 'sample_metadata_val':
                resolved.append(self._build_report_metadata_element('val'))
            else:
                resolved.append({'type': element_type, 'settings': settings})

        return resolved

    def _generate_report_preview(self):
        """Generate LaTeX preview on the right panel."""
        if not hasattr(self, 'report_preview_text'):
            return

        if not self._report_has_successful_execution():
            self._show_fading_warning(
                self.language_manager.translate('ui.messages.run_model_before_report', 'Run the model before previewing or saving the report.')
            )
            return

        if not self.report_data.get('elements'):
            self.report_preview_text.delete('1.0', tk.END)
            self.report_preview_text.insert('1.0', self.language_manager.translate("ui.messages.add_elements_for_preview", "Add report elements to generate preview."))
            return

        with tempfile.TemporaryDirectory(prefix='cm_report_preview_') as temp_dir:
            assets_dir = Path(temp_dir) / 'assets'
            elements = self._build_resolved_report_elements(assets_dir)
            latex_source = build_latex_document(
                elements,
                self.language_manager.get_language(),
                self._build_report_footer_text()
            )

        self.report_preview_text.delete('1.0', tk.END)
        self.report_preview_text.insert('1.0', latex_source)
        if hasattr(self, 'report_status_label'):
            self.report_status_label.configure(text=self.language_manager.translate("ui.messages.preview_updated", "Preview updated."))

    def _save_pdf_report(self):
        """Save report as PDF (and always emit TeX source)."""
        if not self._report_has_successful_execution():
            self._show_fading_warning(
                self.language_manager.translate('ui.messages.run_model_before_report', 'Run the model before previewing or saving the report.')
            )
            return

        if not self.report_data.get('elements'):
            self._show_fading_warning(
                self.language_manager.translate("ui.messages.no_report_elements", "Add at least one report element before saving.")
            )
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"CMStudio_report_{timestamp}.pdf"
        file_path = filedialog.asksaveasfilename(
            title=self.language_manager.translate("ui.dialogs.save_pdf_report", "Save PDF Report"),
            defaultextension='.pdf',
            initialfile=default_name,
            filetypes=[(self.language_manager.translate('ui.dialogs.file_filter_pdf', 'PDF Files (*.pdf)'), '*.pdf')]
        )
        if not file_path:
            return

        output_pdf = Path(file_path)
        assets_dir = output_pdf.parent / f"{output_pdf.stem}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        elements = self._build_resolved_report_elements(assets_dir)
        latex_source = build_latex_document(
            elements,
            self.language_manager.get_language(),
            self._build_report_footer_text()
        )

        self._show_execution_progress_message(
            self.language_manager.translate(
                "ui.messages.generating_pdf_wait",
                "PDF is being generated. Please wait..."
            )
        )

        compile_result_holder: Dict[str, Any] = {}
        compile_exception_holder: Dict[str, str] = {}
        compile_done = threading.Event()

        def _compile_worker():
            try:
                compile_result_holder['result'] = compile_latex_to_pdf(latex_source, str(output_pdf))
            except Exception as exc:
                compile_exception_holder['error'] = str(exc)
            finally:
                compile_done.set()

        compile_thread = threading.Thread(target=_compile_worker, daemon=True)
        compile_thread.start()

        try:
            while not compile_done.is_set():
                self.root.update()
                time.sleep(0.03)

            if compile_exception_holder.get('error'):
                compile_result = {
                    'success': False,
                    'pdf_path': None,
                    'tex_path': str(output_pdf.with_suffix('.tex')),
                    'error': compile_exception_holder['error'],
                }
            else:
                compile_result = compile_result_holder.get('result') or {
                    'success': False,
                    'pdf_path': None,
                    'tex_path': str(output_pdf.with_suffix('.tex')),
                    'error': self.language_manager.translate("ui.messages.report_compile_failed", "Report compilation failed."),
                }
        finally:
            self._stop_execution_progress_message()

        if compile_result.get('success'):
            for suffix in ('.tex', '.aux', '.log', '.out', '.toc'):
                artifact = output_pdf.with_suffix(suffix)
                if artifact.exists():
                    try:
                        artifact.unlink()
                    except Exception:
                        pass

            if assets_dir.exists():
                try:
                    shutil.rmtree(assets_dir)
                except Exception:
                    pass

            self._show_fading_success(
                self.language_manager.translate("ui.messages.report_saved_pdf", "Report saved successfully:") + f"\n{compile_result.get('pdf_path')}"
            )
            if hasattr(self, 'report_status_label'):
                self.report_status_label.configure(text=self.language_manager.translate("ui.messages.report_saved", "PDF report saved."))
            return

        tex_path = compile_result.get('tex_path')
        error_message = compile_result.get('error') or self.language_manager.translate("ui.messages.report_compile_failed", "Report compilation failed.")
        self._show_fading_warning(
            self.language_manager.translate("ui.messages.report_tex_saved", "LaTeX source saved, but PDF compilation failed.") +
            f"\n{tex_path}\n\n{error_message}"
        )
        if hasattr(self, 'report_status_label'):
            self.report_status_label.configure(text=self.language_manager.translate("ui.messages.report_tex_only", "Saved .tex source (PDF compilation failed)."))
    
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
                    src_nested_key = conn_info.get("src_nested_key", "")
                    auto_created = conn_info.get("auto_created", False)
                    
                    # Get function instance aliases from methodology list
                    src_instance_alias = self.methodology_list[src_idx] if src_idx < len(self.methodology_list) else ""
                    dst_instance_alias = self.methodology_list[dst_idx] if dst_idx < len(self.methodology_list) else ""
                    
                    if src_instance_alias and dst_instance_alias:
                        routing_array.append({
                            "source": {
                                "instance_alias": src_instance_alias,
                                "param_key": src_param_key,
                                "param_name": src_param_name,
                                "nested_key": src_nested_key
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

            # Add report config if present
            if hasattr(self, 'report_data') and isinstance(self.report_data, dict):
                report_elements = self.report_data.get('elements', [])
                if report_elements:
                    model_data['report'] = {
                        'elements': copy.deepcopy(report_elements)
                    }
            
            # Write model.json
            model_path = Path(__file__).parent / "model.json"
            with open(model_path, "w", encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.generate_model_json_failed", "Failed to generate model.json") + f": {e}"
            )
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
                    'execution_results': {},
                    'execution_history': [],
                    'current_result_idx': 0
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
                
                self._show_fading_success(
                    self.language_manager.translate("ui.messages.model_saved", "Model saved to:") + f"\n{mdcd_path}"
                )
        
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.save_failed", "Failed to save model:") + f" {e}"
            )
    
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
                
                self._show_fading_success(
                    self.language_manager.translate("ui.messages.model_saved", "Model saved to:") + f"\n{mdon_path}"
                )
        
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.save_failed", "Failed to save model:") + f" {e}"
            )
    
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
                
                self._show_fading_success(
                    self.language_manager.translate("ui.messages.model_saved", "Model saved to:") + f"\n{mdfd_path}"
                )
        
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.save_failed", "Failed to save model:") + f" {e}"
            )
    
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
            self._show_fading_error(
                self.language_manager.translate("ui.messages.load_failed", "Failed to load model:") + f" {e}"
            )
    
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
        
        self._show_fading_success(
            self.language_manager.translate("ui.messages.model_loaded", "Model loaded from:") + f"\n{file_path}"
        )
    
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
        self.report_data = {'elements': [], 'selected_index': None}
        
        try:
            with open(model_path, encoding='utf-8') as f:
                model_data = json.load(f)
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.load_model_json_failed", "Failed to load model.json:") + f" {e}"
            )
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
            src_nested_key = src_info.get('nested_key', '')
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
                    "src_nested_key": src_nested_key,
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

        # Load report configuration if present
        report_config = model_data.get('report', {})
        if isinstance(report_config, dict):
            self.report_data = {
                'elements': copy.deepcopy(report_config.get('elements', [])),
                'selected_index': None
            }
        elif not hasattr(self, 'report_data'):
            self.report_data = {'elements': [], 'selected_index': None}
    
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
                if 'execution_history' in self.analysis_data[instance_alias]:
                    del self.analysis_data[instance_alias]['execution_history']
                if 'current_result_idx' in self.analysis_data[instance_alias]:
                    del self.analysis_data[instance_alias]['current_result_idx']
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

            # Timing report is valid only for the latest successful run
            self.latest_timing_report = None
            self._set_timing_report_menu_state(False)
    
    def _refresh_gui_from_config(self):
        """Refresh GUI to reflect loaded configuration."""
        self._refresh_methodology_listbox()
        
        # Clear setup tab
        self._clear_tab()
        self.selected_function_idx = None
        label = ttk.Label(self.tab_content_frame, text=self.language_manager.translate("ui.messages.model_loaded_select", "Model loaded. Select a function to configure."), 
                         font=("Arial", 10, "italic"))
        label.pack(padx=20, pady=20)
    
    def _run_model(self):
        """Execute model and capture output."""
        if not self.methodology_list:
            self._show_fading_warning(
                self.language_manager.translate("ui.messages.empty_methodology", "Add functions to Methodology first")
            )
            return
        
        # Clear any cached execution results and graphs to ensure fresh data
        self._clear_execution_cache()
        
        if not self._generate_model_json():
            return

        self._begin_execution_progress(
            total_steps=len(self.methodology_list),
            mode_label=self.language_manager.translate("ui.buttons.run_model", "Run Model")
        )
        
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

                def _progress_callback(completed_steps: int, total_steps: int, current_instance: str, current_base: str):
                    self._update_execution_progress(completed_steps, total_steps, current_instance, current_base)
                
                # Run the full model and capture outputs
                outputs, timing_report = analyst_main(progress_callback=_progress_callback, return_timing=True)
                
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
            
            execution_time_by_instance = {
                entry.get('instance_alias'): entry.get('execution_time', 0.0)
                for entry in (timing_report.get('function_timings', []) if timing_report else [])
            }
            execution_history_by_instance = timing_report.get('execution_history_by_instance', {}) if timing_report else {}

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
                history_entries = copy.deepcopy(execution_history_by_instance.get(instance_alias, []))
                
                if history_entries:
                    self.analysis_data[instance_alias]['execution_history'] = history_entries
                    selected_history_idx = 0
                    self.analysis_data[instance_alias]['current_result_idx'] = selected_history_idx
                    self.analysis_data[instance_alias]['execution_results'] = history_entries[selected_history_idx].copy()
                else:
                    self.analysis_data[instance_alias]['execution_results'] = {
                        'status': 'success',
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': execution_time_by_instance.get(instance_alias, 0.0),
                        'outputs': outputs.get(instance_alias, {}) if outputs else {},
                        'inputs': input_parameters
                    }
                    self.analysis_data[instance_alias]['execution_history'] = [
                        self.analysis_data[instance_alias]['execution_results'].copy()
                    ]
                    self.analysis_data[instance_alias]['current_result_idx'] = 0

            self._store_timing_report(
                run_type_label=self.language_manager.translate("ui.buttons.run_model", "Run Model"),
                timing_report=timing_report,
                stop_at_function_alias=None
            )
            
            self._show_fading_success(
                self.language_manager.translate("ui.messages.model_executed", "Model executed successfully. Results loaded for analysis.")
            )
            self._finish_execution_progress(success=True)
            
            # Switch to analysis tab to show results
            if self.selected_function_idx is not None:
                self._show_analysis_tab()
            
        except Exception as e:
            self._finish_execution_progress(success=False)
            error_log_path = Path(__file__).parent / "model_log.txt"
            with open(error_log_path, "w") as f:
                f.write(f"ERROR: {str(e)}\n\n")
                f.write(output_buffer.getvalue() if 'output_buffer' in locals() else "")
            
            self._show_fading_error(
                self.language_manager.translate("ui.messages.execution_failed", "Model execution failed:") + f" {e}\n" +
                self.language_manager.translate("ui.messages.check_log", "Check model_log.txt for details")
            )


def main():
    """Main entry point for the GUI."""
    root = tk.Tk()
    root.iconbitmap("Icon.ico")
    app = ChemometricsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
