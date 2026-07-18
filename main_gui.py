"""
Main GUI application for Chemometric Studio using tkinter + Sun-Valley theme.
Provides Setup, Routing, Analysis, and Report tabs for building analysis pipelines.
"""

import copy
import importlib
import json
import os
import platform
import re
from functools import lru_cache
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
import tempfile
import threading
import time
import traceback
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, filedialog, simpledialog
import webbrowser
import zipfile

import numpy as np

from addon_manager import load_combined_function_specs, normalize_required_addons
from app_services.model_service import build_model_payload, write_model_payload
from app_services.execution_service import consume_last_execution_report_safe
from app_services.run_feedback_service import (
    build_full_run_exception_feedback,
)
from app_services.run_coordinator_service import (
    build_runtime_error_log_contents,
)
from app_services.run_exception_service import (
    build_run_to_here_exception_message,
)
from app_services.passforward_service import (
    get_active_passforward_output_keys as _svc_get_active_passforward_output_keys,
    get_passforward_config as _svc_get_passforward_config,
    get_passforward_output_aliases as _svc_get_passforward_output_aliases,
    is_passforward_compatible as _svc_is_passforward_compatible,
    is_passforward_enabled as _svc_is_passforward_enabled,
)
from app_services.routing_context_service import (
    can_auto_route_between as _svc_can_auto_route_between,
    get_workflow_scope_signature as _svc_get_workflow_scope_signature,
)
from app_services.data_resolution_service import (
    build_execution_data_sources as _svc_build_execution_data_sources,
    resolve_inherited_upstream_outputs as _svc_resolve_inherited_upstream_outputs,
    resolve_routed_inputs as _svc_resolve_routed_inputs,
)
from app_services.methodology_routing_service import (
    filter_manual_routing_lines as _svc_filter_manual_routing_lines,
    remap_manual_routing_lines as _svc_remap_manual_routing_lines,
)
from app_services.analysis_render_service import (
    compute_dimension_combinations as _svc_compute_dimension_combinations,
    extract_sliced_data as _svc_extract_sliced_data,
    extract_axis_data as _svc_extract_axis_data,
    get_data_from_source as _svc_get_data_from_source,
    normalize_class_data_matrix as _svc_normalize_class_data_matrix,
    normalize_class_labels_for_plot as _svc_normalize_class_labels_for_plot,
    resolve_text_section_content as _svc_resolve_text_section_content,
)
from app_services.condition_eval_service import (
    evaluate_condition as _svc_evaluate_condition,
)
from app_services.run_controller import RunController
from app_services.run_orchestrator import orchestrate_run_execution
from language_manager import get_language_manager, _
from runtime_paths import (
    get_runtime_root_dir as _shared_runtime_root_dir,
    get_tempfiles_dir as _shared_tempfiles_dir,
    get_runtime_model_json_path as _shared_runtime_model_json_path,
    get_runtime_model_log_path as _shared_runtime_model_log_path,
)
from settings import get_settings_manager

# Load function specs (core + optional add-ons)
SPECS_PATH = Path(__file__).parent / "function_specs.json"
_combined_specs_payload = load_combined_function_specs(Path(__file__).parent, language="en")
FUNCTION_SPECS = _combined_specs_payload["specs"]
ADDON_REGISTRY = _combined_specs_payload["addon_registry"]

BASE_DIR = Path(__file__).parent
GRAPHICS_DIR = BASE_DIR / "Graphics"
FONTS_DIR = BASE_DIR / "Fonts"
LICENSES_DIR = BASE_DIR / "Licenses"
PROJECT_LICENSE_PATH = BASE_DIR / "LICENSE"
EULA_PATH = BASE_DIR / "EULA.md"
PYPROJECT_PATH = BASE_DIR / "pyproject.toml"
MANUAL_INDEX_PATH = BASE_DIR / "Manual" / "index.html"
SELAWIK_TTF_PATH = FONTS_DIR / "Selawik" / "selawk.ttf"
SPLASH_VERSION_FONT_SIZE = 14
SPLASH_SUBTITLE_FONT_SIZE = 13
SPLASH_TEXT_LINE_SPACING = 6
SPLASH_VERSION_RELATIVE_POS = (0.07, 0.5)
SPLASH_VERSION_MARGIN_RATIO = 0.02


def _get_ui_font_family(preferred: str = "Selawik", fallback: str = "Arial") -> str:
    """Resolve a UI font family available on this system."""
    try:
        families = set(tkfont.families())
        if preferred in families:
            return preferred
    except Exception:
        pass
    return fallback


def _get_runtime_root_dir() -> Path:
    """Return a writable runtime directory for app-managed state."""
    return _shared_runtime_root_dir()


def _get_tempfiles_dir() -> Path:
    """Return the runtime tempfiles directory used by packaged workflows."""
    return _shared_tempfiles_dir()


def _get_runtime_model_json_path() -> Path:
    """Return runtime model.json path in a user-writable location."""
    return _shared_runtime_model_json_path()


def _get_runtime_model_log_path() -> Path:
    """Return runtime model log path in a user-writable location."""
    return _shared_runtime_model_log_path()


def _ui_symbol(name: str) -> str:
    """Return a UI symbol with Linux-safe fallbacks for sparse font setups."""
    default_symbols = {
        "collapsed": "▶",
        "expanded": "▼",
        "prev": "←",
        "next": "→",
        "up": "↑",
        "down": "↓",
        "expand": "🗗",
    }
    linux_symbols = {
        "collapsed": ">",
        "expanded": "v",
        "prev": "<",
        "next": ">",
        "up": "^",
        "down": "v",
        "expand": "[]",
    }
    if platform.system() == "Linux":
        return linux_symbols.get(name, default_symbols.get(name, name))
    return default_symbols.get(name, name)


@lru_cache(maxsize=1)
def _get_application_version(default: str = "0.0") -> str:
    """Read application version from pyproject.toml [project]."""
    if not PYPROJECT_PATH.exists():
        return default

    try:
        import tomllib  # Python 3.11+

        with open(PYPROJECT_PATH, "rb") as f:
            data = tomllib.load(f)
        version = str(data.get("project", {}).get("version", "")).strip()
        if version:
            return version
    except Exception:
        pass

    try:
        in_project_section = False
        with open(PYPROJECT_PATH, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    in_project_section = line == "[project]"
                    continue
                if in_project_section and line.startswith("version") and "=" in line:
                    _, value = line.split("=", 1)
                    cleaned = value.strip().strip('"').strip("'")
                    if cleaned:
                        return cleaned
    except Exception:
        pass

    return default


def _version_compare_key(version: str) -> Tuple[List[int], int]:
    """Build a lightweight comparison key for version strings.

    Returns (numeric_parts, stage_rank) where stage_rank is 0 for stable
    releases and -1 for pre-release labels such as "alpha"/"beta".
    """
    text = str(version or "").strip().lower()
    if text.startswith("v"):
        text = text[1:].strip()

    # Split first numeric block from optional textual suffix.
    match = re.match(r"^(\d+(?:\.\d+)*)\s*(.*)$", text)
    if not match:
        numbers = [int(part) for part in re.findall(r"\d+", text)]
        if not numbers:
            numbers = [0]
        return numbers, -1

    numbers = [int(part) for part in match.group(1).split(".")]
    suffix = match.group(2).strip()
    stage_rank = -1 if suffix else 0
    return numbers, stage_rank


def _compare_versions(left: str, right: str) -> int:
    """Compare version strings; returns -1, 0, or 1."""
    left_nums, left_stage = _version_compare_key(left)
    right_nums, right_stage = _version_compare_key(right)

    max_len = max(len(left_nums), len(right_nums))
    left_padded = left_nums + [0] * (max_len - len(left_nums))
    right_padded = right_nums + [0] * (max_len - len(right_nums))

    if left_padded < right_padded:
        return -1
    if left_padded > right_padded:
        return 1

    if left_stage < right_stage:
        return -1
    if left_stage > right_stage:
        return 1
    return 0


def _normalize_bool_setting(value: Any, default: bool = False) -> bool:
    """Normalize persisted bool-like values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_force_integer_mode_setting(value: Any, default: str = 'false') -> str:
    """Normalize force_integer setting to 'false', 'true', or 'conditional'."""
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


def _set_window_icon(window, base_name: str = "Icon") -> None:
    base_dir = Path(__file__).parent
    graphics_dir = base_dir / "Graphics"
    ico_path = graphics_dir / f"{base_name}.ico"
    png_path = graphics_dir / f"{base_name}.png"
    if not ico_path.exists():
        ico_path = base_dir / f"{base_name}.ico"
    if not png_path.exists():
        png_path = base_dir / f"{base_name}.png"
    current_system = platform.system().lower()

    if current_system == "windows" and ico_path.exists():
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
            x = self.widget.winfo_rootx() + 24
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
            
            # Create tooltip window
            self.tipwindow = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            
            label = tk.Label(tw, text=self.text, background="#ffffcc", relief=tk.SOLID, 
                           borderwidth=1, font=("Arial", 9), wraplength=self.wraplength, justify=tk.LEFT)
            label.pack(ipadx=1)

            # Keep tooltip within screen bounds.
            tw.update_idletasks()
            tip_w = tw.winfo_reqwidth()
            tip_h = tw.winfo_reqheight()
            screen_w = tw.winfo_screenwidth()
            screen_h = tw.winfo_screenheight()

            if x + tip_w > screen_w - 8:
                x = max(8, screen_w - tip_w - 8)
            if y + tip_h > screen_h - 8:
                y = max(8, self.widget.winfo_rooty() - tip_h - 8)

            tw.wm_geometry(f"+{x}+{y}")
    
    def hidetip(self, event=None):
        """Hide the tooltip."""
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class ToggleSwitch(tk.Canvas):
    """Simple switch-style control with on/off knob animation styling."""

    def __init__(self, parent, variable: tk.BooleanVar, command: Optional[Callable[[], None]] = None, **kwargs):
        try:
            parent_bg = str(parent.cget("background"))
        except Exception:
            parent_bg = "#f0f0f0"

        self.track_on_color = kwargs.pop("track_on_color", None)
        self.track_off_color = kwargs.pop("track_off_color", None)
        self.knob_fill_color = kwargs.pop("knob_fill_color", None)
        self.knob_outline_color = kwargs.pop("knob_outline_color", None)

        super().__init__(
            parent,
            width=46,
            height=24,
            highlightthickness=0,
            bd=0,
            bg=kwargs.pop("bg", parent_bg),
            **kwargs,
        )
        self.variable = variable
        self.command = command
        self.configure(cursor="hand2")

        self.bind("<Button-1>", self._on_click)
        self.variable.trace_add("write", lambda *_args: self._draw())
        self._draw()

    def _on_click(self, _event=None):
        self.variable.set(not bool(self.variable.get()))
        if callable(self.command):
            self.command()

    def _draw(self):
        self.delete("all")
        enabled = bool(self.variable.get())

        style = ttk.Style(self)
        bg_color = (
            style.lookup("TFrame", "background")
            or style.lookup(".", "background")
            or self.cget("bg")
            or "#f0f0f0"
        )
        self.configure(bg=bg_color)

        style_track_on = (
            style.lookup(".", "selectbackground")
            or style.lookup(".", "focuscolor")
            or "#22c55e"
        )
        style_track_off = (
            style.lookup("TCheckbutton", "indicatorbackground")
            or style.lookup(".", "bordercolor")
            or "#b8b8b8"
        )
        style_knob_fill = (
            style.lookup("TEntry", "fieldbackground")
            or style.lookup(".", "fieldbackground")
            or "#ffffff"
        )
        style_knob_outline = (
            style.lookup("TCheckbutton", "indicatorforeground")
            or style.lookup(".", "foreground")
            or "#dddddd"
        )

        track_on = self.track_on_color or style_track_on
        track_off = self.track_off_color or style_track_off
        knob_fill = self.knob_fill_color or style_knob_fill
        knob_outline = self.knob_outline_color or style_knob_outline

        track_color = track_on if enabled else track_off

        # Rounded track (rectangle + two circles)
        self.create_oval(2, 2, 22, 22, fill=track_color, outline=track_color)
        self.create_rectangle(12, 2, 34, 22, fill=track_color, outline=track_color)
        self.create_oval(24, 2, 44, 22, fill=track_color, outline=track_color)

        knob_x = 24 if enabled else 4
        self.create_oval(knob_x, 4, knob_x + 16, 20, fill=knob_fill, outline=knob_outline)


class ChemometricsGUI:
    """Main GUI class for building and executing chemometrics pipelines."""

    GRAPH_FONT_SCALE_DEFAULT = 1.0
    GRAPH_FONT_SCALE_OPTIONS = (0.8, 0.9, 1.0, 1.1, 1.25)
    GRAPH_FONT_SCALE_LABEL_KEYS = {
        0.8: "menu.font_scale_very_small",
        0.9: "menu.font_scale_small",
        1.0: "menu.font_scale_normal",
        1.1: "menu.font_scale_large",
        1.25: "menu.font_scale_very_large"
    }
    THEME_DEFAULT = "sv_light"
    THEME_LABELS = {
        "sv_light": "Sun Valley (Light)",
        "sv_dark": "Sun Valley (Dark)",
        "clam": "Tk Clam",
        "alt": "Tk Alt",
        "default": "Tk Default",
        "vista": "Tk Vista",
        "xpnative": "Tk XP Native",
        "aqua": "Tk Aqua",
    }
    CUSTOM_ANALYSIS_ALIAS = "__custom_analysis__"
    
    def __init__(self, root: tk.Tk):
        self.root = root

        # Load Sun Valley colour variant definitions FIRST so that saved-theme
        # normalisation (below) can accept variant IDs without falling back to default.
        self._sv_variant_themes: Dict[str, Dict] = self._load_sv_variant_themes()
        
        # Initialize settings manager and load saved language
        self.settings_manager = get_settings_manager()
        saved_language = self.settings_manager.get("language", "en")
        
        # Initialize language manager with saved language
        self.language_manager = get_language_manager()
        self.language_manager.set_language(saved_language)

        self.language_var = tk.StringVar(value=saved_language)
        self.colormap_var = tk.StringVar(value=self.settings_manager.get("colormap", "jet"))
        self.qualitative_colormap_var = tk.StringVar(value=self.settings_manager.get("qualitative_colormap", "tab10"))

        saved_import_loading_mode = self.settings_manager.get("import_loading_mode", "lazy")
        self.import_loading_mode = self._normalize_import_loading_mode(saved_import_loading_mode)
        if self.import_loading_mode != saved_import_loading_mode:
            self.settings_manager.set("import_loading_mode", self.import_loading_mode)

        saved_display_splashscreen = self.settings_manager.get("display_splashscreen", True)
        self.display_splashscreen = _normalize_bool_setting(saved_display_splashscreen, True)
        if self.display_splashscreen != saved_display_splashscreen:
            self.settings_manager.set("display_splashscreen", self.display_splashscreen)

        saved_categories_start_collapsed = self.settings_manager.get("categories_start_collapsed", False)
        self.categories_start_collapsed = _normalize_bool_setting(saved_categories_start_collapsed, False)
        if self.categories_start_collapsed != saved_categories_start_collapsed:
            self.settings_manager.set("categories_start_collapsed", self.categories_start_collapsed)

        saved_graph_font_scale = self.settings_manager.get("graph_font_scale", self.GRAPH_FONT_SCALE_DEFAULT)
        self.graph_font_scale = self._normalize_graph_font_scale(saved_graph_font_scale)
        self.graph_font_scale_var = tk.StringVar(value=self._format_graph_font_scale_value(self.graph_font_scale))
        if self.graph_font_scale != saved_graph_font_scale:
            self.settings_manager.set("graph_font_scale", self.graph_font_scale)

        saved_theme = self.settings_manager.get("theme", self.THEME_DEFAULT)
        self.selected_theme = self._normalize_theme_id(saved_theme)
        self.theme_var = tk.StringVar(value=self.selected_theme)
        if self.selected_theme != saved_theme:
            self.settings_manager.set("theme", self.selected_theme)

        self.import_loading_mode_var = tk.StringVar(value=self.import_loading_mode)
        self.display_splashscreen_var = tk.BooleanVar(value=self.display_splashscreen)
        self.categories_start_collapsed_var = tk.BooleanVar(value=self.categories_start_collapsed)
        self._graph_renderer = None
        self._reporting_funcs = None
        self._add_graph_dialog_fn = None
        self._add_table_dialog_fn = None
        self._add_text_dialog_fn = None
        self._routing_map_window_cls = None
        self._matplotlib_pyplot = None
        self.run_controller = RunController()

        if self.import_loading_mode == "eager":
            self._preload_heavy_dependencies()
        
        self.root.title(self.language_manager.translate("ui.main_title", "Chemometric Studio"))
        self.root.geometry("1280x720")
        self._disable_combobox_mousewheel()
        
        # Use a user-writable runtime location instead of the install directory.
        self.tempfiles_dir = _get_tempfiles_dir()
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
            "workflow_parallel_end",
            "workflow_ensemble_start",
            "workflow_ensemble_member",
            "workflow_ensemble_end"
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
        self.notification_toasts: List[tk.Toplevel] = []
        self.notification_stack_spacing = 8
        self.notification_stack_margin_top = 58
        self.notification_stack_margin_right = 24
        self._notification_root_bind_set = False
        self.execution_progress_frame = None
        self.execution_progress_popup = None
        self.execution_progress_status_label = None
        self.execution_progress_percent_label = None
        self.execution_progress_bar = None
        self.execution_progress_mode = ""
        self.execution_progress_hide_after_id = None
        self._pending_language_refresh_id = None
        self._execution_progress_root_bind_set = False
        self.latest_timing_report: Optional[Dict[str, Any]] = None
        self.latest_execution_report: Optional[Dict[str, Any]] = None
        self.methodology_highlight_palette: Dict[str, Dict[str, str]] = {
            "warning": {"bg": "#FFE699", "fg": "#000000"},
            "error": {"bg": "#FFB3B3", "fg": "#000000"},
        }
        self.methodology_row_highlights: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.tools_menu = None
        self.timing_report_menu_index: Optional[int] = None
        self.execution_report_menu_index: Optional[int] = None
        self.model_log_menu_index: Optional[int] = None
        self.report_data: Dict[str, Any] = {
            'elements': [],
            'selected_index': None
        }
        self.custom_analysis_data: Dict[str, Any] = {
            'pages': [],
            'current_page': 0,
            'active_sections': {}
        }
        self._globally_pinned_point_labels: Set[str] = set()
        
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

    def _normalize_import_loading_mode(self, mode: Any) -> str:
        """Normalize persisted import-loading mode."""
        normalized = str(mode).strip().lower() if mode is not None else "lazy"
        return normalized if normalized in {"lazy", "eager"} else "lazy"

    def _normalize_graph_font_scale(self, value: Any) -> float:
        """Normalize persisted graph relative font scale to a supported option."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return self.GRAPH_FONT_SCALE_DEFAULT

        for option in self.GRAPH_FONT_SCALE_OPTIONS:
            if abs(numeric - option) < 1e-9:
                return option
        return self.GRAPH_FONT_SCALE_DEFAULT

    def _normalize_theme_id(self, theme_id: Any) -> str:
        """Normalize persisted theme identifier to a supported id."""
        normalized = str(theme_id).strip().lower() if theme_id is not None else self.THEME_DEFAULT
        aliases = {
            "light": "sv_light",
            "dark": "sv_dark",
            "sun-valley-light": "sv_light",
            "sun-valley-dark": "sv_dark",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in self.THEME_LABELS:
            return normalized
        # Accept any loaded Sun Valley colour variant
        if hasattr(self, "_sv_variant_themes") and normalized in self._sv_variant_themes:
            return normalized
        return self.THEME_DEFAULT

    def _load_sv_variant_themes(self) -> Dict[str, Dict]:
        """Discover and parse Sun Valley colour variant JSON files from themes/sun-valley-variants/."""
        variants: Dict[str, Dict] = {}
        variants_dir = Path(__file__).parent / "themes" / "sun-valley-variants"
        if not variants_dir.is_dir():
            return variants
        for json_path in sorted(variants_dir.glob("*.json")):
            try:
                with open(json_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                theme_id = str(data.get("id", "")).strip()
                label = str(data.get("label", json_path.stem)).strip()
                base = str(data.get("base", "light")).strip().lower()
                if base not in ("light", "dark"):
                    base = "light"
                if not theme_id:
                    continue
                tcl_theme_name = str(data.get("tcl_theme_name", "")).strip()
                tcl_file_rel = str(data.get("tcl_file", "")).strip()
                tcl_path: Optional[Path] = None
                if tcl_file_rel:
                    tcl_path = variants_dir / tcl_file_rel
                    if not tcl_path.exists():
                        tcl_path = None
                variants[theme_id] = {
                    "id": theme_id,
                    "label": label,
                    "base": base,
                    "tcl_theme_name": tcl_theme_name,
                    "tcl_path": tcl_path,
                    "palette": data.get("palette", {}),
                    "styles": data.get("styles", {}),
                }
            except Exception as exc:
                print(f"[theme] Could not load variant '{json_path.name}': {exc}")
        return variants

    def _apply_sv_variant_overrides(self, variant: Dict) -> None:
        """Apply per-style colour overrides defined in a Sun Valley colour variant."""
        style = ttk.Style(self.root)

        for style_name, options in variant.get("styles", {}).items():
            if not isinstance(options, dict):
                continue
            try:
                style.configure(style_name, **{k: v for k, v in options.items() if v not in ("", None)})
            except tk.TclError:
                pass

        palette = variant.get("palette", {})
        if palette:
            try:
                self.root.tk_setPalette(**{k: v for k, v in palette.items() if v not in ("", None)})
                root_bg = palette.get("background")
                if root_bg:
                    self.root.configure(bg=root_bg)
            except tk.TclError:
                pass

    def _get_available_theme_options(self) -> List[Tuple[str, str]]:
        """Return available theme ids and labels for the current runtime."""
        options: List[Tuple[str, str]] = []

        sv_available = False
        # Add Sun Valley themes when package is available.
        try:
            importlib.import_module("sv_ttk")
            sv_available = True
            options.append(("sv_light", self.THEME_LABELS["sv_light"]))
            options.append(("sv_dark", self.THEME_LABELS["sv_dark"]))
        except Exception:
            pass

        # Add Sun Valley colour variants (only when sv_ttk is available)
        if sv_available and hasattr(self, "_sv_variant_themes"):
            for variant_id, variant_data in self._sv_variant_themes.items():
                options.append((variant_id, variant_data["label"]))

        style = ttk.Style(self.root)
        available_native = set(style.theme_names())
        for theme_id in ("clam", "alt", "default", "vista", "xpnative", "aqua"):
            if theme_id in available_native:
                options.append((theme_id, self.THEME_LABELS[theme_id]))

        seen_ids = set()
        deduped: List[Tuple[str, str]] = []
        for theme_id, label in options:
            if theme_id in seen_ids:
                continue
            seen_ids.add(theme_id)
            deduped.append((theme_id, label))
        return deduped

    def _capture_theme_control_baseline(self, theme_name: str) -> Dict[str, Dict[str, Any]]:
        """Capture core ttk control font/padding options from a specific theme."""
        style = ttk.Style(self.root)
        available = set(style.theme_names())
        if theme_name not in available:
            return {}

        current_theme = style.theme_use()
        target_options: Dict[str, Tuple[str, ...]] = {
            "TButton": ("font", "padding"),
            "Toolbutton": ("font", "padding"),
            "TMenubutton": ("font", "padding"),
            "TCheckbutton": ("font", "padding", "indicatorsize"),
            "TRadiobutton": ("font", "padding", "indicatorsize"),
            "TEntry": ("font", "padding"),
            "TCombobox": ("font", "padding"),
            "TSpinbox": ("font", "padding"),
            "Treeview": ("font", "rowheight", "padding"),
            "Heading": ("font", "padding"),
            "Treeview.Heading": ("font", "padding"),
            "TNotebook": ("padding", "tabmargins"),
            "TNotebook.Tab": ("font", "padding"),
        }
        baseline: Dict[str, Dict[str, Any]] = {}

        try:
            if current_theme != theme_name:
                style.theme_use(theme_name)

            for style_name, option_names in target_options.items():
                opts: Dict[str, Any] = {}
                configured = style.configure(style_name) or {}
                for option_name in option_names:
                    value = configured.get(option_name)
                    if value in ("", None):
                        value = style.lookup(style_name, option_name)
                    if value not in ("", None):
                        opts[option_name] = value
                baseline[style_name] = opts
        finally:
            if style.theme_use() != current_theme:
                style.theme_use(current_theme)

        return baseline

    def _get_control_geometry_baseline(self) -> Dict[str, Dict[str, Any]]:
        """Return cached baseline for cross-theme geometry normalization.

        On Windows, prefer Vista metrics when available to match the original
        development baseline. On other platforms (or older Windows without Vista
        theme), fall back through native ttk themes.
        """
        cached = getattr(self, "_control_geometry_baseline", None)
        if isinstance(cached, dict):
            return cached

        style = ttk.Style(self.root)
        available = set(style.theme_names())
        candidate_order: List[str] = []
        if platform.system() == "Windows":
            candidate_order.extend(["vista", "xpnative"])
        elif platform.system() == "Darwin":
            candidate_order.append("aqua")
        candidate_order.extend(["clam", "default", "alt"])

        baseline: Dict[str, Dict[str, Any]] = {}
        for candidate in candidate_order:
            if candidate not in available:
                continue
            baseline = self._capture_theme_control_baseline(candidate)
            if baseline:
                break

        self._control_geometry_baseline = baseline
        return baseline

    def _safe_style_configure(self, style: ttk.Style, style_name: str, options: Dict[str, Any]) -> None:
        """Configure ttk style options safely, skipping unsupported keys per backend."""
        for option_name, value in options.items():
            if value in ("", None):
                continue
            try:
                style.configure(style_name, **{option_name: value})
            except tk.TclError:
                continue

    def _configure_custom_ttk_styles(self):
        """Apply app-specific ttk style overrides after setting a base theme."""
        style = ttk.Style()

        # Keep control geometry aligned to native baseline metrics.
        baseline = self._get_control_geometry_baseline()
        for style_name in (
            "TButton",
            "Toolbutton",
            "TMenubutton",
            "TCheckbutton",
            "TRadiobutton",
            "TEntry",
            "TCombobox",
            "TSpinbox",
            "Treeview",
            "Heading",
            "Treeview.Heading",
            "TNotebook",
            "TNotebook.Tab",
        ):
            opts = baseline.get(style_name, {})
            if opts:
                self._safe_style_configure(style, style_name, opts)

        button_opts = baseline.get("TButton", {})
        button_font = button_opts.get("font")
        button_padding = button_opts.get("padding")

        button_style_opts: Dict[str, Any] = {}
        if button_font:
            button_style_opts["font"] = button_font
        if button_padding not in ("", None):
            button_style_opts["padding"] = button_padding

        if button_style_opts:
            self._safe_style_configure(style, "Toggle.TButton", button_style_opts)
            self._safe_style_configure(style, "Accent.TButton", button_style_opts)

        # Output button style (blue)
        if button_style_opts:
            self._safe_style_configure(style, "Output.TButton", button_style_opts)

        # Input button style (red)
        if button_style_opts:
            self._safe_style_configure(style, "Input.TButton", button_style_opts)

        style.configure(
            "FunctionsPanel.TButton",
            anchor="w",
            justify="left"
        )

        style.configure(
            "Execution.Horizontal.TProgressbar",
            troughcolor="#2f2f2f",
            background="#2f9fff",
            bordercolor="#2f2f2f",
            lightcolor="#2f9fff",
            darkcolor="#2f9fff",
            thickness=12
        )

    def _sync_tk_palette_to_theme(self):
        """Sync classic Tk palette colors to the currently active ttk theme."""
        style = ttk.Style(self.root)

        background = style.lookup(".", "background") or self.root.cget("bg") or "#f0f0f0"
        foreground = style.lookup(".", "foreground") or "#000000"
        variant_palette = {}
        if hasattr(self, "_sv_variant_themes"):
            variant_data = self._sv_variant_themes.get(getattr(self, "selected_theme", ""), {})
            if isinstance(variant_data, dict):
                variant_palette = variant_data.get("palette", {}) or {}
        select_bg = (
            variant_palette.get("selectBackground")
            or style.lookup(".", "selectbackground")
            or style.lookup(".", "focuscolor")
            or background
        )
        select_fg = variant_palette.get("selectForeground") or style.lookup(".", "selectforeground") or foreground
        active_bg = style.lookup(".", "activebackground") or background
        active_fg = style.lookup(".", "activeforeground") or foreground

        try:
            self.root.tk_setPalette(
                background=background,
                foreground=foreground,
                activeBackground=active_bg,
                activeForeground=active_fg,
                selectBackground=select_bg,
                selectForeground=select_fg,
            )
            self.root.configure(bg=background)
        except tk.TclError:
            # Some platforms/themes may reject a subset of palette keys.
            pass

    def _get_theme_background_color(self) -> str:
        """Resolve a neutral background color from the active ttk theme."""
        style = ttk.Style(self.root)
        return (
            style.lookup("TFrame", "background")
            or style.lookup(".", "background")
            or self.root.cget("bg")
            or "#f0f0f0"
        )

    def _get_theme_foreground_color(self) -> str:
        """Resolve a readable foreground color from the active ttk theme."""
        style = ttk.Style(self.root)
        return (
            style.lookup("TLabel", "foreground")
            or style.lookup(".", "foreground")
            or "#000000"
        )

    def _get_methodology_theme_colors(self) -> Dict[str, str]:
        """Resolve the listbox colors used by the methodology panel."""
        style = ttk.Style(self.root)
        background = self._get_theme_background_color()
        foreground = self._get_theme_foreground_color()
        variant_palette = {}
        if hasattr(self, "_sv_variant_themes"):
            variant_data = self._sv_variant_themes.get(getattr(self, "selected_theme", ""), {})
            if isinstance(variant_data, dict):
                variant_palette = variant_data.get("palette", {}) or {}
        select_background = (
            variant_palette.get("selectBackground")
            or style.lookup(".", "selectbackground")
            or style.lookup(".", "focuscolor")
            or background
        )
        select_foreground = variant_palette.get("selectForeground") or style.lookup(".", "selectforeground") or foreground
        return {
            "bg": background,
            "fg": foreground,
            "select_bg": select_background,
            "select_fg": select_foreground,
        }

    def _configure_methodology_listbox_theme(self) -> None:
        """Apply active theme colors to the methodology listbox."""
        if not hasattr(self, "methodology_listbox"):
            return

        colors = self._get_methodology_theme_colors()
        try:
            self.methodology_listbox.configure(
                bg=colors["bg"],
                fg=colors["fg"],
                selectbackground=colors["select_bg"],
                selectforeground=colors["select_fg"],
                highlightbackground=colors["bg"],
                highlightcolor=colors["select_bg"],
            )
        except tk.TclError:
            pass

    def _apply_vista_variant_font_tuning(self, theme_id: str) -> None:
        """Apply Tk Vista-like font metrics for the Vista variant only."""
        if theme_id != "sv_variant_vista":
            return

        style = ttk.Style(self.root)
        body_font = ("Segoe UI", 9)
        heading_font = ("Segoe UI", 9, "bold")
        for style_name in (
            ".",
            "TLabel",
            "TButton",
            "Toolbutton",
            "TCheckbutton",
            "TRadiobutton",
            "TMenubutton",
            "TEntry",
            "TCombobox",
            "TSpinbox",
            "TNotebook.Tab",
            "TLabelframe.Label",
            "Treeview",
        ):
            self._safe_style_configure(style, style_name, {"font": body_font})
        self._safe_style_configure(style, "Heading", {"font": heading_font})
        self._safe_style_configure(style, "Treeview.Heading", {"font": heading_font})

    def _refresh_current_tab_view(self):
        """Re-render the currently active tab so existing widgets pick up new theme colors."""
        current_tab = getattr(self, "current_tab", None)
        if current_tab == "analysis":
            self._show_analysis_tab()
        elif current_tab == "custom_analysis":
            self._show_custom_analysis_tab()
        elif current_tab == "setup":
            self._show_setup_tab()
        elif current_tab == "routing":
            self._show_routing_tab()
        elif current_tab == "report":
            self._show_report_tab()

    def _normalize_point_label_token(self, label: Any) -> Optional[str]:
        """Normalize sample/point label text used for global pin tracking."""
        text = str(label).strip() if label is not None else ""
        return text if text else None

    def _sync_global_point_label_pins_in_place(self) -> None:
        """Apply the global pinned-label set to all live graph canvases without re-rendering pages."""
        graph_renderer = self._get_graph_renderer()

        for analysis_info in self.analysis_data.values():
            if not isinstance(analysis_info, dict):
                continue
            graph_canvases = analysis_info.get('graph_canvases', {})
            if not isinstance(graph_canvases, dict):
                continue

            for canvas_data in list(graph_canvases.values()):
                if not isinstance(canvas_data, tuple) or not canvas_data:
                    continue
                canvas = canvas_data[0]
                try:
                    graph_renderer.sync_canvas_pinned_labels(
                        canvas,
                        self._globally_pinned_point_labels,
                        on_label_pin_toggled=self._on_global_point_label_pin_toggled,
                    )
                except Exception:
                    pass

    def _on_global_point_label_pin_toggled(self, label: str, is_pinned: bool) -> None:
        """Track globally pinned point labels and refresh visible graph sections."""
        normalized = self._normalize_point_label_token(label)
        if not normalized:
            return

        changed = False
        if is_pinned:
            if normalized not in self._globally_pinned_point_labels:
                self._globally_pinned_point_labels.add(normalized)
                changed = True
        else:
            if normalized in self._globally_pinned_point_labels:
                self._globally_pinned_point_labels.discard(normalized)
                changed = True

        if changed:
            self._sync_global_point_label_pins_in_place()

    def _apply_selected_theme(self, theme_id: str, persist: bool = False, notify: bool = False):
        """Apply selected UI theme, with fallback and optional persistence."""
        normalized = self._normalize_theme_id(theme_id)
        applied_theme = normalized
        applied = False
        pending_variant: Optional[Dict] = None

        if normalized.startswith("sv_"):
            try:
                sv_ttk = importlib.import_module("sv_ttk")
                # Resolve the base theme for colour variants
                if hasattr(self, "_sv_variant_themes") and normalized in self._sv_variant_themes:
                    pending_variant = self._sv_variant_themes[normalized]
                    base = pending_variant.get("base", "light")
                    tcl_path = pending_variant.get("tcl_path")
                    tcl_theme_name = pending_variant.get("tcl_theme_name", "")
                    if tcl_path and tcl_theme_name:
                        # Ensure sv_ttk internals (fonts, etc.) are loaded first
                        style = ttk.Style(self.root)
                        sv_ttk._load_theme(style)
                        # Source the TCL only if this theme isn't registered yet
                        if tcl_theme_name not in style.theme_names():
                            self.root.tk.call("source", str(tcl_path))
                        style.theme_use(tcl_theme_name)
                        self._normalize_sun_valley_fonts()
                        pending_variant = None   # TCL handles all colours; no Python overrides needed
                    else:
                        # Fallback: use base sv_ttk theme + Python style overrides
                        sv_ttk.set_theme(base, root=self.root)
                        self._normalize_sun_valley_fonts()
                else:
                    base = "dark" if normalized == "sv_dark" else "light"
                    sv_ttk.set_theme(base, root=self.root)
                    self._normalize_sun_valley_fonts()
                applied = True
            except Exception as e:
                print(f"Could not load sv_ttk theme '{normalized}': {e}")

        if not applied:
            style = ttk.Style(self.root)
            available_native = set(style.theme_names())
            fallback_candidates = [normalized, "clam", "default"]
            chosen = next((name for name in fallback_candidates if name in available_native), None)
            if chosen is None and available_native:
                chosen = next(iter(available_native))
            if chosen:
                style.theme_use(chosen)
                applied_theme = chosen

        self.selected_theme = applied_theme
        if self.theme_var.get() != applied_theme:
            self.theme_var.set(applied_theme)
        if persist:
            self.settings_manager.set("theme", applied_theme)

        self._configure_custom_ttk_styles()
        # Apply colour-variant overrides after _configure_custom_ttk_styles so variant
        # colours (e.g. progressbar) are not overwritten by the standard defaults.
        if pending_variant is not None:
            self._apply_sv_variant_overrides(pending_variant)
        self._apply_vista_variant_font_tuning(normalized)
        self._sync_tk_palette_to_theme()
        self._configure_methodology_listbox_theme()
        self._refresh_methodology_listbox(selected_idx=self.selected_function_idx)
        self._refresh_current_tab_view()

        if notify:
            variant_data = (
                self._sv_variant_themes.get(applied_theme)
                if hasattr(self, "_sv_variant_themes") else None
            )
            applied_label = (
                variant_data["label"] if variant_data is not None
                else self.THEME_LABELS.get(applied_theme, applied_theme)
            )
            self._show_fading_message(
                self.language_manager.translate("ui.messages.theme_changed_to", "Theme changed to") + f" {applied_label}."
            )

    def _set_theme(self, theme_id: str):
        """Persist and apply UI theme preference."""
        normalized = self._normalize_theme_id(theme_id)
        if self.theme_var.get() != normalized:
            self.theme_var.set(normalized)
        if normalized == self.selected_theme:
            return
        self._apply_selected_theme(normalized, persist=True, notify=True)

    def _format_graph_font_scale_value(self, scale: float) -> str:
        """Format graph font scale for Tk menu variable values."""
        return f"{float(scale):.2f}"

    def _graph_font_scale_label(self, scale: float) -> str:
        """Build language-aware menu label for graph font scale options."""
        key = self.GRAPH_FONT_SCALE_LABEL_KEYS.get(scale, "menu.font_scale_normal")
        fallback_map = {
            0.8: "Very Small",
            0.9: "Small",
            1.0: "Normal",
            1.1: "Large",
            1.25: "Very Large"
        }
        base_label = self.language_manager.translate(key, fallback_map.get(scale, "Normal"))
        percent_text = f"{int(round(scale * 100))}%"
        default_suffix = ""
        if abs(scale - self.GRAPH_FONT_SCALE_DEFAULT) < 1e-9:
            default_suffix = " " + self.language_manager.translate("menu.default_tag", "(default)")
        return f"{base_label} ({percent_text}){default_suffix}"

    def _populate_graph_font_scale_menu(self, menu: tk.Menu, variable: tk.StringVar, on_select) -> None:
        """Populate a Tk menu with graph font scale radiobutton options."""
        for scale in self.GRAPH_FONT_SCALE_OPTIONS:
            menu.add_radiobutton(
                label=self._graph_font_scale_label(scale),
                variable=variable,
                value=self._format_graph_font_scale_value(scale),
                command=lambda val=scale: on_select(val)
            )

    def _load_colormaps_catalog(self) -> dict:
        """Load colormap catalog from Settings/colormaps.json with safe fallback."""
        try:
            colormaps_path = Path(__file__).parent / "Settings" / "colormaps.json"
            with open(colormaps_path, encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

        return {
            "continuous": {"Perceptually Uniform": ["viridis", "plasma", "inferno"]},
            "qualitative": []
        }

    def _get_continuous_colormap_options(self) -> List[str]:
        """Return flattened list of available continuous colormap names."""
        colormaps_data = self._load_colormaps_catalog()
        continuous_data = colormaps_data.get("continuous", {})

        options: List[str] = []
        if isinstance(continuous_data, list):
            options.extend(str(name) for name in continuous_data)
        elif isinstance(continuous_data, dict):
            for _category, cmaps in continuous_data.items():
                if isinstance(cmaps, list):
                    options.extend(str(name) for name in cmaps)

        deduped: List[str] = []
        seen = set()
        for name in options:
            if name and name not in seen:
                deduped.append(name)
                seen.add(name)
        return deduped

    def _is_graph_option_supported(self, graph_type: str, option_key: str, config: Optional[dict] = None) -> bool:
        """Return whether a context-menu option applies to a graph type/config."""
        cfg = config or {}
        normalized_type = str(graph_type or "").strip().lower()
        has_z_axis = isinstance(cfg.get('z_axis'), dict) and bool(cfg.get('z_axis'))
        is_scatter_3d = normalized_type == 'scatter' and has_z_axis

        if option_key == 'show_grid':
            return normalized_type in {'line', 'scatter'}
        if option_key == 'show_origin':
            return normalized_type in {'line', 'scatter'} and not is_scatter_3d
        if option_key == 'show_labels':
            return normalized_type == 'scatter'
        if option_key == 'confidence_ellipses':
            return normalized_type == 'scatter' and not is_scatter_3d
        if option_key == 'confidence_level':
            return normalized_type == 'scatter' and not is_scatter_3d
        if option_key == 'equal_scale':
            return normalized_type == 'scatter' and not is_scatter_3d
        if option_key == 'cmap':
            return normalized_type in {'line', 'scatter', 'heatmap', '3d_surf', 'contour'}
        if option_key == 'use_wireframe':
            return normalized_type == '3d_surf'
        if option_key == 'contour_filled':
            return normalized_type == 'contour'
        if option_key == 'flip_xy':
            return normalized_type in {'scatter', 'line', 'bar', 'histogram', 'heatmap', '3d_surf', 'contour'}
        if option_key == 'annotate_heatmap':
            return normalized_type == 'heatmap'
        if option_key in {'x_axis_type', 'y_axis_type'}:
            return normalized_type in {'line', 'scatter', 'heatmap', 'contour', '3d_surf', 'bar', 'histogram'}
        if option_key == 'z_axis_type':
            return normalized_type in {'scatter', '3d_surf'} and has_z_axis
        if option_key in {'x_force_integer', 'y_force_integer'}:
            return normalized_type in {'line', 'scatter', 'heatmap'}
        if option_key == 'z_force_integer':
            return normalized_type == 'scatter' and has_z_axis
        if option_key in {'x_reverse_axis', 'y_reverse_axis'}:
            return normalized_type in {'line', 'scatter', 'heatmap', 'contour', '3d_surf', 'bar', 'histogram'}
        if option_key == 'z_reverse_axis':
            return normalized_type in {'scatter', '3d_surf'} and has_z_axis
        return False

    def _get_rendered_dataset_visibility_entries(self, instance_alias: str, section_id: Tuple[int, int]) -> List[Dict[str, str]]:
        """Return dataset entries used by visibility controls when a section has multiple datasets."""
        entries_by_key: Dict[str, str] = {}

        analysis_info = self.analysis_data.get(instance_alias, {})
        pages = analysis_info.get('pages', []) if isinstance(analysis_info, dict) else []
        page_idx, section_idx = section_id
        section_cfg: Dict[str, Any] = {}
        if 0 <= page_idx < len(pages):
            page = pages[page_idx]
            sections = page.get('sections', []) if isinstance(page, dict) else []
            if 0 <= section_idx < len(sections):
                section = sections[section_idx]
                cfg = section.get('config', {}) if isinstance(section, dict) else {}
                if isinstance(cfg, dict):
                    section_cfg = cfg

        datasets_cfg = section_cfg.get('datasets') if isinstance(section_cfg.get('datasets'), list) else []
        for idx, dataset_cfg in enumerate(datasets_cfg):
            if not isinstance(dataset_cfg, dict):
                continue
            key = f'cfg:{idx}'
            label = str(dataset_cfg.get('label', f'Dataset {idx + 1}')).strip() or f'Dataset {idx + 1}'
            entries_by_key[key] = label

        metadata = analysis_info.get('graph_data_metadata', {}).get(section_id, {}) if isinstance(analysis_info, dict) else {}
        extracted = metadata.get('extracted_datasets') if isinstance(metadata, dict) else None
        if isinstance(extracted, list):
            for idx, dataset in enumerate(extracted):
                if not isinstance(dataset, dict):
                    continue
                key = str(dataset.get('visibility_key', f'idx:{idx}')).strip() or f'idx:{idx}'
                label = str(dataset.get('label', f'Dataset {idx + 1}')).strip() or f'Dataset {idx + 1}'
                if key not in entries_by_key:
                    entries_by_key[key] = label

        entries = [{'key': key, 'label': label} for key, label in entries_by_key.items()]
        return entries if len(entries) > 1 else []

    def _apply_dataset_visibility_filter(self, config: dict, extracted_datasets: Optional[List[dict]]) -> Optional[List[dict]]:
        """Filter extracted datasets with config['dataset_visibility'] while ensuring at least one remains."""
        if not isinstance(extracted_datasets, list) or len(extracted_datasets) <= 1:
            return extracted_datasets

        visibility_cfg = config.get('dataset_visibility') if isinstance(config, dict) else None
        if not isinstance(visibility_cfg, dict):
            return extracted_datasets

        filtered: List[dict] = []
        for idx, dataset in enumerate(extracted_datasets):
            if not isinstance(dataset, dict):
                continue
            visibility_key = str(dataset.get('visibility_key', f'idx:{idx}')).strip() or f'idx:{idx}'
            if bool(visibility_cfg.get(visibility_key, True)):
                filtered.append(dataset)

        return filtered if filtered else extracted_datasets

    def _assign_dataset_style_slots(self, extracted_datasets: Optional[List[dict]]) -> Optional[List[dict]]:
        """Assign stable style slots from active datasets before UI visibility filtering."""
        if not isinstance(extracted_datasets, list) or not extracted_datasets:
            return extracted_datasets

        total_datasets = len(extracted_datasets)
        total_lines = 0
        per_dataset_line_counts: List[int] = []

        for dataset in extracted_datasets:
            if not isinstance(dataset, dict):
                per_dataset_line_counts.append(0)
                continue

            y_data = dataset.get('y_data')
            line_count = 1
            try:
                y_arr = np.asarray(y_data)
                if y_arr.ndim > 1:
                    line_count = int(max(1, y_arr.shape[0]))
            except Exception:
                line_count = 1

            per_dataset_line_counts.append(line_count)
            total_lines += line_count

        running_line_start = 0
        for idx, dataset in enumerate(extracted_datasets):
            if not isinstance(dataset, dict):
                continue
            dataset['style_slot'] = idx
            dataset['style_total_datasets'] = total_datasets
            dataset['style_line_start'] = running_line_start
            dataset['style_total_lines'] = max(1, total_lines)
            running_line_start += per_dataset_line_counts[idx]

        return extracted_datasets

    def _update_graph_axis_config_option(self, instance_alias: str, section_id: Tuple[int, int],
                                         axis_key: str, option_key: str, option_value: Any,
                                         popup_refresh_callback: Optional[Callable[[], None]] = None,
                                         refresh_analysis: bool = True) -> None:
        """Persist a graph axis option under config[axis_key] and refresh graph view."""
        try:
            analysis_info = self.analysis_data.get(instance_alias)
            if not isinstance(analysis_info, dict):
                return

            page_idx, section_idx = section_id
            pages = analysis_info.get('pages', [])
            if page_idx < 0 or page_idx >= len(pages):
                return

            sections = pages[page_idx].get('sections', []) if isinstance(pages[page_idx], dict) else []
            if section_idx < 0 or section_idx >= len(sections):
                return

            section = sections[section_idx]
            config = section.setdefault('config', {}) if isinstance(section, dict) else {}
            if not isinstance(config, dict):
                return

            axis_config = config.get(axis_key)
            if not isinstance(axis_config, dict):
                axis_config = {}
                config[axis_key] = axis_config

            axis_config[option_key] = option_value

            graph_slices = analysis_info.get('graph_slices', {})
            if section_id in graph_slices and isinstance(graph_slices[section_id], dict):
                graph_slices[section_id]['config'] = config

            self._generate_model_json()

            if refresh_analysis:
                has_slice_state = section_id in analysis_info.get('graph_slices', {})
                has_canvas = section_id in analysis_info.get('graph_canvases', {})
                if has_slice_state and has_canvas:
                    self._update_graph_with_slice(instance_alias, section_id, -1)
                else:
                    self._show_analysis_tab()

            if popup_refresh_callback is not None:
                popup_refresh_callback()
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.generate_model_json_failed", "Failed to generate model.json") + f": {e}"
            )

    def _normalize_confidence_level_percent(self, value: Any) -> str:
        """Normalize confidence level to percent string (e.g., '95')."""
        raw_value = value if value is not None else '95'
        try:
            parsed = float(str(raw_value).strip().replace('%', ''))
        except (TypeError, ValueError):
            parsed = 95.0

        if parsed <= 0:
            parsed = 95.0
        if parsed <= 1.0:
            parsed *= 100.0

        if abs(parsed - round(parsed)) < 1e-9:
            return str(int(round(parsed)))
        return f"{parsed:.2f}".rstrip('0').rstrip('.')

    @staticmethod
    def _raw_config_has_custom_ellipsis(config: dict) -> bool:
        """Return True if any entry in the raw scatter_lines config has orientation 'custom_ellipsis'."""
        source_candidates = (
            config.get('scatter_lines'),
            config.get('reference_lines'),
            config.get('guide_lines'),
        )
        for candidate in source_candidates:
            if candidate is None:
                continue
            if isinstance(candidate, dict):
                candidate = [candidate]
            if isinstance(candidate, list):
                for entry in candidate:
                    if not isinstance(entry, dict):
                        continue
                    token = str(
                        entry.get('orientation', entry.get('type', entry.get('mode', '')))
                    ).strip().lower()
                    if token in ('custom_ellipsis', 'ellipsis', 'ellipse', 'ejcr'):
                        return True
        return False

    @staticmethod
    def _get_raw_config_custom_ellipsis_confidence_levels(config: dict) -> List[str]:
        """Return ordered unique confidence_level strings from custom_ellipsis entries in raw config."""
        source_candidates = (
            config.get('scatter_lines'),
            config.get('reference_lines'),
            config.get('guide_lines'),
        )
        levels: List[str] = []
        seen: set = set()
        for candidate in source_candidates:
            if candidate is None:
                continue
            if isinstance(candidate, dict):
                candidate = [candidate]
            if isinstance(candidate, list):
                for entry in candidate:
                    if not isinstance(entry, dict):
                        continue
                    token = str(
                        entry.get('orientation', entry.get('type', entry.get('mode', '')))
                    ).strip().lower()
                    if token not in ('custom_ellipsis', 'ellipsis', 'ellipse', 'ejcr'):
                        continue
                    cl = entry.get('confidence_level')
                    if cl is None:
                        continue
                    cl_str = str(cl).strip().replace('%', '')
                    if cl_str and cl_str not in seen:
                        levels.append(cl_str)
                        seen.add(cl_str)
        return levels

    def _get_custom_ellipsis_confidence_levels(self, config: dict, instance_alias: str) -> List[str]:
        """Return ordered confidence levels for custom ellipses from raw and resolved configs.

        Supports packed ellipse payloads provided through reference-line data sources.
        """
        levels: List[str] = []
        seen: set = set()

        def _add_level(value: Any) -> None:
            if value is None:
                return
            level_str = str(value).strip().replace('%', '')
            if level_str and level_str not in seen:
                levels.append(level_str)
                seen.add(level_str)

        for lv in self._get_raw_config_custom_ellipsis_confidence_levels(config):
            _add_level(lv)

        execution_results = self.analysis_data.get(instance_alias, {}).get('execution_results', {})
        if execution_results.get('status') != 'success':
            return levels

        outputs = self._get_execution_data_sources(execution_results, instance_alias)
        resolved_lines = self._resolve_scatter_reference_lines(config, outputs, slice_indices=None)
        for entry in resolved_lines:
            if not isinstance(entry, dict):
                continue
            _add_level(entry.get('confidence_level'))

            packed = entry.get('ellipses_data', entry.get('packed_ellipses'))
            if not isinstance(packed, dict):
                continue
            packed_levels = packed.get('confidence_levels')
            if isinstance(packed_levels, np.ndarray):
                packed_levels = packed_levels.tolist()
            if isinstance(packed_levels, (list, tuple)):
                for lv in packed_levels:
                    _add_level(lv)

        return levels

    def _normalize_class_data_matrix(self, value: Any) -> Optional[np.ndarray]:
        """Normalize class labels into a 2D object array: rows=samples, cols=class layers."""
        return _svc_normalize_class_data_matrix(value)

    def _is_numeric_class_layer_continuous(self, values: np.ndarray) -> Optional[bool]:
        """Heuristic: returns True for continuous numeric class-like values, False for discrete, None if non-numeric."""
        if values is None:
            return None

        cleaned: List[str] = []
        has_decimal = False
        for raw in np.asarray(values, dtype=object).tolist():
            text = str(raw).strip()
            if text == "" or text.lower() in {"nan", "none"}:
                continue
            cleaned.append(text)
            if any(ch in text for ch in ('.', 'e', 'E')):
                has_decimal = True

        if not cleaned:
            return None

        numeric_values: List[float] = []
        for text in cleaned:
            try:
                numeric_values.append(float(text))
            except Exception:
                return None

        if has_decimal:
            return True

        unique_vals = np.unique(np.asarray(numeric_values, dtype=float))
        if unique_vals.size <= 1:
            return False

        span = float(unique_vals.max() - unique_vals.min())
        diffs = np.diff(np.sort(unique_vals))
        median_diff = float(np.median(diffs)) if diffs.size > 0 else 0.0

        if span <= max(3.0, float(unique_vals.size) * 3.0) and median_diff <= 2.0:
            return False
        return True

    def _compute_scatter_class_layer_state(
        self,
        config: dict,
        datasets_config: Optional[List[dict]],
        outputs: Dict[str, Any],
        execution_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build effective class-layer configuration (available layers, natures, order, mapping)."""
        source_values: List[np.ndarray] = []

        def _dataset_axes_resolvable(ds_cfg: dict) -> bool:
            if not isinstance(ds_cfg, dict):
                return False
            x_axis_cfg = ds_cfg.get('x_axis', {}) if isinstance(ds_cfg.get('x_axis', {}), dict) else {}
            y_axis_cfg = ds_cfg.get('y_axis', {}) if isinstance(ds_cfg.get('y_axis', {}), dict) else {}
            x_source = x_axis_cfg.get('data_source')
            y_source = y_axis_cfg.get('data_source')
            if not isinstance(x_source, str) or not x_source.strip():
                return False
            if not isinstance(y_source, str) or not y_source.strip():
                return False

            x_nested = x_axis_cfg.get('nested_key') if isinstance(x_axis_cfg.get('nested_key'), str) else None
            y_nested = y_axis_cfg.get('nested_key') if isinstance(y_axis_cfg.get('nested_key'), str) else None
            x_val = self._get_data_from_source(outputs, x_source, x_nested)
            y_val = self._get_data_from_source(outputs, y_source, y_nested)
            return x_val is not None and y_val is not None

        top_source = config.get('class_labels')
        if isinstance(top_source, str) and top_source:
            top_val = self._get_data_from_source(outputs, top_source)
            top_matrix = self._normalize_class_data_matrix(top_val)
            if top_matrix is not None and top_matrix.size > 0:
                source_values.append(top_matrix)

        if isinstance(datasets_config, list):
            exec_inputs = execution_inputs if isinstance(execution_inputs, dict) else {}
            for ds_cfg in datasets_config:
                if not isinstance(ds_cfg, dict):
                    continue
                include_dataset = True
                condition = ds_cfg.get('condition')
                if isinstance(condition, dict):
                    param_name = condition.get('parameter')
                    operator = condition.get('operator', '==')
                    expected_value = condition.get('value')
                    if param_name and param_name in exec_inputs:
                        actual_value = exec_inputs[param_name]
                        include_dataset = _condition_matches(actual_value, operator, expected_value)
                if not include_dataset or not _dataset_axes_resolvable(ds_cfg):
                    continue

                class_source = ds_cfg.get('class_labels')
                if not isinstance(class_source, str) or not class_source:
                    continue
                class_val = self._get_data_from_source(outputs, class_source)
                class_matrix = self._normalize_class_data_matrix(class_val)
                if class_matrix is not None and class_matrix.size > 0:
                    source_values.append(class_matrix)

        layer_count = 0
        for matrix in source_values:
            if matrix.ndim >= 2:
                layer_count = max(layer_count, int(matrix.shape[1]))

        def _condition_matches(actual_value: Any, operator: str, expected_value: Any) -> bool:
            op = str(operator or '==').strip()
            lhs = actual_value
            rhs = expected_value

            try:
                if op in ['>', '<', '>=', '<=']:
                    try:
                        lhs = int(lhs) if isinstance(lhs, str) else lhs
                        rhs = int(rhs) if isinstance(rhs, str) else rhs
                    except Exception:
                        lhs = str(lhs)
                        rhs = str(rhs)
                else:
                    lhs = str(lhs)
                    rhs = str(rhs)
            except Exception:
                pass

            try:
                if op == '==':
                    return lhs == rhs
                if op == '!=':
                    return lhs != rhs
                if op == '>':
                    return lhs > rhs
                if op == '<':
                    return lhs < rhs
                if op == '>=':
                    return lhs >= rhs
                if op == '<=':
                    return lhs <= rhs
                if op == 'in':
                    return lhs in rhs
                if op == 'contains':
                    return rhs in lhs
                return True
            except Exception:
                return False

        dataset_count = len(datasets_config) if isinstance(datasets_config, list) else 0
        active_dataset_count = dataset_count
        if isinstance(datasets_config, list):
            active_dataset_count = 0
            exec_inputs = execution_inputs if isinstance(execution_inputs, dict) else {}
            for ds_cfg in datasets_config:
                if not isinstance(ds_cfg, dict):
                    continue
                include_dataset = True
                condition = ds_cfg.get('condition')
                if isinstance(condition, dict):
                    param_name = condition.get('parameter')
                    operator = condition.get('operator', '==')
                    expected_value = condition.get('value')
                    if param_name and param_name in exec_inputs:
                        actual_value = exec_inputs[param_name]
                        include_dataset = _condition_matches(actual_value, operator, expected_value)
                if include_dataset and _dataset_axes_resolvable(ds_cfg):
                    active_dataset_count += 1

        is_multi_dataset = active_dataset_count > 1

        all_aspects = ['color', 'marker', 'fill', 'edge']
        available_aspects = ['color', 'marker', 'edge'] if is_multi_dataset else all_aspects
        max_active = len(available_aspects) if layer_count > 0 else 0

        has_user_flag = bool(config.get('class_layer_user_defined', False))

        raw_order = config.get('class_layer_order', [])
        configured_order: List[str] = []
        if isinstance(raw_order, str):
            chunks = [tok.strip().lower() for tok in re.split(r'[;,\s]+', raw_order) if tok and tok.strip()]
            configured_order = [tok for tok in chunks if tok in available_aspects]
        elif isinstance(raw_order, (list, tuple)):
            configured_order = [str(a).strip().lower() for a in raw_order if str(a).strip().lower() in available_aspects]

        raw_map = config.get('class_layer_map', {})
        configured_map: Dict[str, int] = {}
        if isinstance(raw_map, dict):
            for k, v in raw_map.items():
                try:
                    norm_k = str(k).strip().lower()
                    if norm_k in available_aspects:
                        configured_map[norm_k] = int(v)
                except Exception:
                    continue
        elif isinstance(raw_map, str):
            parsed_order, parsed_map = self._parse_class_layer_mapping_description(raw_map)
            for key in parsed_order:
                if key in available_aspects and key in parsed_map:
                    try:
                        configured_map[key] = int(parsed_map[key])
                    except Exception:
                        continue

        has_nonempty_order = bool(configured_order)
        has_nonempty_map = bool(configured_map)
        user_defined_mapping = has_user_flag or has_nonempty_order or has_nonempty_map

        effective_order: List[str] = []
        for aspect in configured_order:
            if aspect not in effective_order:
                effective_order.append(aspect)
            if len(effective_order) >= max_active:
                break

        if not effective_order and max_active > 0 and not user_defined_mapping:
            auto_count = min(len(available_aspects), max(0, layer_count))
            effective_order = available_aspects[:auto_count]

        effective_map: Dict[str, int] = {}
        filtered_order: List[str] = []
        for idx, aspect in enumerate(effective_order):
            raw_layer = configured_map.get(aspect, idx + 1)
            try:
                layer_idx = int(raw_layer)
            except Exception:
                layer_idx = idx + 1

            if layer_idx <= 0:
                continue

            if layer_count > 0:
                layer_idx = max(1, min(layer_idx, layer_count))
            else:
                layer_idx = 1
            filtered_order.append(aspect)
            effective_map[aspect] = layer_idx

        effective_order = filtered_order

        layer_nature: Dict[int, str] = {}
        numeric_layers: List[int] = []
        for layer_idx in range(1, layer_count + 1):
            continuous_votes = 0
            discrete_votes = 0
            numeric_seen = False
            for matrix in source_values:
                if matrix.ndim < 2 or matrix.shape[1] < layer_idx:
                    continue
                state = self._is_numeric_class_layer_continuous(matrix[:, layer_idx - 1])
                if state is None:
                    continue
                numeric_seen = True
                if state:
                    continuous_votes += 1
                else:
                    discrete_votes += 1

            if numeric_seen:
                numeric_layers.append(layer_idx)
                layer_nature[layer_idx] = 'continuous' if continuous_votes >= discrete_votes else 'discrete'
            else:
                layer_nature[layer_idx] = 'discrete'

        overrides = config.get('class_layer_nature', {})
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                try:
                    key_idx = int(key)
                except Exception:
                    continue
                norm = str(value).strip().lower()
                if key_idx >= 1 and key_idx <= layer_count and norm in {'continuous', 'discrete'}:
                    layer_nature[key_idx] = norm

        return {
            'layer_count': layer_count,
            'dataset_count': dataset_count,
            'active_dataset_count': active_dataset_count,
            'is_multi_dataset': is_multi_dataset,
            'available_aspects': available_aspects,
            'max_active': max_active,
            'effective_order': effective_order,
            'effective_map': effective_map,
            'layer_nature': layer_nature,
            'numeric_layers': numeric_layers,
            'user_defined_mapping': user_defined_mapping,
        }

    def _compute_line_class_layer_state(
        self,
        config: dict,
        datasets_config: Optional[List[dict]],
        outputs: Dict[str, Any],
        execution_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build effective class-layer configuration for line plots.

        Line aspects:
          - color (always available)
          - linestyle (single-dataset only; reserved for dataset identity on multi-dataset)
          - marker (disabled when marker is explicitly provided in JSON config)
        """
        base_state = self._compute_scatter_class_layer_state(
            config,
            datasets_config,
            outputs,
            execution_inputs,
        )

        layer_count = int(base_state.get('layer_count', 0))
        is_multi_dataset = bool(base_state.get('is_multi_dataset', False))

        marker_explicit = False
        if 'marker' in config:
            marker_explicit = True
        if isinstance(datasets_config, list):
            for ds_cfg in datasets_config:
                if isinstance(ds_cfg, dict) and 'marker' in ds_cfg:
                    marker_explicit = True
                    break

        available_aspects: List[str] = ['color']
        if not is_multi_dataset:
            available_aspects.append('linestyle')
        if not marker_explicit:
            available_aspects.append('marker')

        max_active = len(available_aspects) if layer_count > 0 else 0

        has_user_flag = bool(config.get('class_layer_user_defined', False))

        raw_order = config.get('class_layer_order', [])
        configured_order: List[str] = []
        if isinstance(raw_order, str):
            chunks = [tok.strip().lower() for tok in re.split(r'[;,\s]+', raw_order) if tok and tok.strip()]
            configured_order = [tok for tok in chunks if tok in available_aspects]
        elif isinstance(raw_order, (list, tuple)):
            configured_order = [str(a).strip().lower() for a in raw_order if str(a).strip().lower() in available_aspects]

        raw_map = config.get('class_layer_map', {})
        configured_map: Dict[str, int] = {}
        if isinstance(raw_map, dict):
            for k, v in raw_map.items():
                try:
                    norm_k = str(k).strip().lower()
                    if norm_k in available_aspects:
                        configured_map[norm_k] = int(v)
                except Exception:
                    continue
        elif isinstance(raw_map, str):
            parsed_order, parsed_map = self._parse_class_layer_mapping_description(raw_map)
            for key in parsed_order:
                if key in available_aspects and key in parsed_map:
                    try:
                        configured_map[key] = int(parsed_map[key])
                    except Exception:
                        continue

        has_nonempty_order = bool(configured_order)
        has_nonempty_map = bool(configured_map)
        user_defined_mapping = has_user_flag or has_nonempty_order or has_nonempty_map

        effective_order: List[str] = []
        for aspect in configured_order:
            if aspect not in effective_order:
                effective_order.append(aspect)
            if len(effective_order) >= max_active:
                break

        if not effective_order and max_active > 0 and not user_defined_mapping:
            auto_count = min(len(available_aspects), max(0, layer_count))
            effective_order = available_aspects[:auto_count]

        effective_map: Dict[str, int] = {}
        filtered_order: List[str] = []
        for idx, aspect in enumerate(effective_order):
            raw_layer = configured_map.get(aspect, idx + 1)
            try:
                layer_idx = int(raw_layer)
            except Exception:
                layer_idx = idx + 1

            if layer_idx <= 0:
                continue

            if layer_count > 0:
                layer_idx = max(1, min(layer_idx, layer_count))
            else:
                layer_idx = 1
            filtered_order.append(aspect)
            effective_map[aspect] = layer_idx

        result = dict(base_state)
        result.update({
            'available_aspects': available_aspects,
            'max_active': max_active,
            'effective_order': filtered_order,
            'effective_map': effective_map,
            'marker_explicit': marker_explicit,
            'user_defined_mapping': user_defined_mapping,
        })
        return result

    def _collect_class_value_order_from_outputs(self, outputs: Dict[str, Any]) -> List[str]:
        """Collect a stable first-seen class order from common class sources in outputs."""
        if not isinstance(outputs, dict):
            return []

        ordered: List[str] = []
        seen: Set[str] = set()

        def _append_values(raw_value: Any) -> None:
            normalized = self._normalize_class_labels_for_plot(raw_value)
            if normalized is None:
                return
            for raw in np.asarray(normalized, dtype=object).reshape(-1).tolist():
                text = str(raw).strip()
                if not text or text.lower() in {'nan', 'none'}:
                    continue
                if text in seen:
                    continue
                seen.add(text)
                ordered.append(text)

        for source_name in ['class_data_cal', 'class_data_val', 'class_labels', 'class_data']:
            value = self._get_data_from_source(outputs, source_name)
            _append_values(value)

        return ordered

    def _build_class_value_order_effective(self, outputs: Dict[str, Any], model_payload: Dict[str, Any]) -> Optional[List[str]]:
        """Build preferred class-value order for stable qualitative color mapping."""
        if not isinstance(model_payload, dict):
            return None

        model_labels_raw = model_payload.get('class_labels')
        model_labels: List[str] = []
        if isinstance(model_labels_raw, (list, tuple, np.ndarray)):
            try:
                labels_arr = np.asarray(model_labels_raw, dtype=object).reshape(-1)
                model_labels = [str(item).strip() for item in labels_arr.tolist() if str(item).strip() != '']
            except Exception:
                model_labels = []

        ref_label = str(model_payload.get('reference_class', '')).strip()
        model_type_label = str(model_payload.get('model_type', '')).strip().lower()

        one_class_model_types = {
            'simca',
            'dd_simca',
            'one_class_svm',
            'isolation_forest',
            'elliptic_envelope',
            'lof',
        }

        is_one_class_payload = bool(
            ref_label
            and model_type_label in one_class_model_types
        )

        if is_one_class_payload:
            global_order = self._collect_class_value_order_from_outputs(outputs)
            if not global_order and model_labels:
                # Fallback when no dataset class labels are available.
                global_order = list(model_labels)

            if ref_label and ref_label not in global_order:
                global_order.insert(0, ref_label)

            # In one-class models, append unknown/additional labels after known dataset classes
            # so existing class colors remain stable across functions.
            unknown_like = [lbl for lbl in model_labels if lbl and lbl != ref_label]
            for lbl in unknown_like:
                if lbl not in global_order:
                    global_order.append(lbl)

            return global_order if global_order else None

        ordered_labels = list(model_labels)
        if ref_label:
            ordered_labels = [ref_label] + [lbl for lbl in ordered_labels if lbl != ref_label]
        return ordered_labels if ordered_labels else None

    def _parse_class_layer_mapping_description(self, text: str) -> Tuple[List[str], Dict[str, int]]:
        """Parse free-text class-layer mapping like 'marker:1, color:2, fill:3, edge:4'."""
        parsed_order: List[str] = []
        parsed_map: Dict[str, int] = {}
        if not text:
            return parsed_order, parsed_map

        alias_map = {
            'marker': 'marker',
            'linestyle': 'linestyle',
            'line_style': 'linestyle',
            'line-type': 'linestyle',
            'linetype': 'linestyle',
            'colour': 'color',
            'color': 'color',
            'fill': 'fill',
            'filltype': 'fill',
            'fill_type': 'fill',
            'edge': 'edge',
            'edgecolour': 'edge',
            'edgecolor': 'edge',
            'edge_color': 'edge',
        }

        for chunk in str(text).split(','):
            if ':' not in chunk:
                continue
            left, right = chunk.split(':', 1)
            key = re.sub(r'[^A-Za-z_]', '', left).strip().lower()
            key = alias_map.get(key)
            if key is None:
                continue
            try:
                layer_idx = int(str(right).strip())
            except Exception:
                continue
            if layer_idx < 1:
                continue
            if key not in parsed_order:
                parsed_order.append(key)
            parsed_map[key] = layer_idx

        return parsed_order, parsed_map

    def _show_graph_warning_once(self, instance_alias: str, section_id: Tuple[int, int], warning_code: str, message: str) -> None:
        """Show a warning message once per graph section and warning code."""
        try:
            analysis_info = self.analysis_data.setdefault(instance_alias, {})
            warning_state = analysis_info.setdefault('graph_warning_state', {})
            if not isinstance(warning_state, dict):
                return
            section_state = warning_state.setdefault(section_id, set())
            if not isinstance(section_state, set):
                section_state = set(section_state) if isinstance(section_state, (list, tuple)) else set()
                warning_state[section_id] = section_state
            if warning_code in section_state:
                return
            section_state.add(warning_code)
            self._show_fading_warning(message)
        except Exception:
            pass

    def _update_graph_section_config_option(self, instance_alias: str, section_id: Tuple[int, int],
                                            option_key: str, option_value: Any,
                                            popup_refresh_callback: Optional[Callable[[], None]] = None,
                                            refresh_analysis: bool = True) -> None:
        """Persist a graph option into analysis config and regenerate model.json."""
        try:
            analysis_info = self.analysis_data.get(instance_alias)
            if not isinstance(analysis_info, dict):
                return

            page_idx, section_idx = section_id
            pages = analysis_info.get('pages', [])
            if page_idx < 0 or page_idx >= len(pages):
                return

            sections = pages[page_idx].get('sections', []) if isinstance(pages[page_idx], dict) else []
            if section_idx < 0 or section_idx >= len(sections):
                return

            section = sections[section_idx]
            config = section.setdefault('config', {}) if isinstance(section, dict) else {}
            if not isinstance(config, dict):
                return

            config[option_key] = option_value

            graph_slices = analysis_info.get('graph_slices', {})
            if section_id in graph_slices and isinstance(graph_slices[section_id], dict):
                graph_slices[section_id]['config'] = config

            self._generate_model_json()

            if refresh_analysis:
                has_slice_state = section_id in analysis_info.get('graph_slices', {})
                has_canvas = section_id in analysis_info.get('graph_canvases', {})
                if has_slice_state and has_canvas:
                    self._update_graph_with_slice(instance_alias, section_id, -1)
                else:
                    self._show_analysis_tab()

            if popup_refresh_callback is not None:
                popup_refresh_callback()
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.generate_model_json_failed", "Failed to generate model.json") + f": {e}"
            )

    def _remove_graph_section_config_option(self, instance_alias: str, section_id: Tuple[int, int],
                                            option_key: str,
                                            popup_refresh_callback: Optional[Callable[[], None]] = None,
                                            refresh_analysis: bool = True) -> None:
        """Remove a graph option from analysis config and regenerate model.json."""
        try:
            analysis_info = self.analysis_data.get(instance_alias)
            if not isinstance(analysis_info, dict):
                return

            page_idx, section_idx = section_id
            pages = analysis_info.get('pages', [])
            if page_idx < 0 or page_idx >= len(pages):
                return

            sections = pages[page_idx].get('sections', []) if isinstance(pages[page_idx], dict) else []
            if section_idx < 0 or section_idx >= len(sections):
                return

            section = sections[section_idx]
            config = section.setdefault('config', {}) if isinstance(section, dict) else {}
            if not isinstance(config, dict):
                return

            config.pop(option_key, None)

            graph_slices = analysis_info.get('graph_slices', {})
            if section_id in graph_slices and isinstance(graph_slices[section_id], dict):
                graph_slices[section_id]['config'] = config

            self._generate_model_json()

            if refresh_analysis:
                has_slice_state = section_id in analysis_info.get('graph_slices', {})
                has_canvas = section_id in analysis_info.get('graph_canvases', {})
                if has_slice_state and has_canvas:
                    self._update_graph_with_slice(instance_alias, section_id, -1)
                else:
                    self._show_analysis_tab()

            if popup_refresh_callback is not None:
                popup_refresh_callback()
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.generate_model_json_failed", "Failed to generate model.json") + f": {e}"
            )

    def _build_graph_context_menu(self, graph_type: str, config: dict, instance_alias: str,
                                  section_id: Tuple[int, int],
                                  popup_refresh_callback: Optional[Callable[[], None]] = None) -> Optional[tk.Menu]:
        """Build graph-specific right-click menu. Returns None when no options apply."""
        menu = tk.Menu(self.root, tearoff=0)
        menu._var_refs = []
        item_count = 0
        normalized_graph_type = str(graph_type).strip().lower()
        line_state_for_menu: Optional[Dict[str, Any]] = None
        scatter_state_for_menu: Optional[Dict[str, Any]] = None

        execution_results = self.analysis_data.get(instance_alias, {}).get('execution_results', {})
        outputs = self._get_execution_data_sources(execution_results, instance_alias) if isinstance(execution_results, dict) else {}
        datasets_cfg = config.get('datasets') if isinstance(config.get('datasets'), list) else None

        if normalized_graph_type == 'line':
            try:
                line_state_for_menu = self._compute_line_class_layer_state(
                    config,
                    datasets_cfg,
                    outputs,
                    execution_results.get('inputs', {}) if isinstance(execution_results, dict) else {},
                )
            except Exception:
                line_state_for_menu = None
        elif normalized_graph_type == 'scatter':
            try:
                scatter_state_for_menu = self._compute_scatter_class_layer_state(
                    config,
                    datasets_cfg,
                    outputs,
                    execution_results.get('inputs', {}) if isinstance(execution_results, dict) else {},
                )
            except Exception:
                scatter_state_for_menu = None

        def _keep_var_ref(var_obj: Any) -> None:
            try:
                menu._var_refs.append(var_obj)
            except Exception:
                pass

        def _add_toggle(option_key: str, label_key: str, fallback: str) -> None:
            nonlocal item_count
            if not self._is_graph_option_supported(graph_type, option_key, config):
                return

            current = bool(config.get(option_key, False))
            state_var = tk.BooleanVar(value=current)
            _keep_var_ref(state_var)

            def _on_toggle() -> None:
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    option_key,
                    bool(state_var.get()),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            menu.add_checkbutton(
                label=self.language_manager.translate(label_key, fallback),
                variable=state_var,
                onvalue=True,
                offvalue=False,
                command=_on_toggle
            )
            item_count += 1

        _add_toggle('show_grid', 'menu.graph_context.grid', 'Grid')
        _add_toggle('show_origin', 'menu.graph_context.origin_axis', 'Origin Axis')
        _add_toggle('show_labels', 'menu.graph_context.labels', 'Labels')

        # Ellipses toggle — greyed out when custom_ellipsis entries are present
        _has_custom_ellipsis = self._raw_config_has_custom_ellipsis(config)
        if self._is_graph_option_supported(graph_type, 'confidence_ellipses', config):
            if _has_custom_ellipsis:
                menu.add_checkbutton(
                    label=self.language_manager.translate('menu.graph_context.ellipses', 'Ellipses'),
                    state=tk.DISABLED,
                )
                item_count += 1
            else:
                _add_toggle('confidence_ellipses', 'menu.graph_context.ellipses', 'Ellipses')

        _add_toggle('equal_scale', 'menu.graph_context.equal_scale', 'Equal Scale')
        _add_toggle('use_wireframe', 'menu.graph_context.use_wireframe', 'Wireframe')
        _add_toggle('annotate_heatmap', 'menu.graph_context.annotate_heatmap', 'Annotate Heatmap')

        if self._is_graph_option_supported(graph_type, 'flip_xy', config):
            flip_default = False
            flip_current = _normalize_bool_setting(config.get('flip_xy'), flip_default)
            flip_var = tk.BooleanVar(value=flip_current)
            _keep_var_ref(flip_var)

            def _set_flip_xy() -> None:
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'flip_xy',
                    bool(flip_var.get()),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            menu.add_checkbutton(
                label=self.language_manager.translate('menu.graph_context.flip_xy', 'Flip X/Y Axes'),
                variable=flip_var,
                onvalue=True,
                offvalue=False,
                command=_set_flip_xy
            )
            item_count += 1

        if self._is_graph_option_supported(graph_type, 'contour_filled', config):
            current_contour_type = str(config.get('contour_type', 'contourf')).strip().lower()
            filled_var = tk.BooleanVar(value=(current_contour_type == 'contourf'))

            def _set_contour_filled() -> None:
                contour_type = 'contourf' if bool(filled_var.get()) else 'contour'
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'contour_type',
                    contour_type,
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            menu.add_checkbutton(
                label=self.language_manager.translate('menu.graph_context.filled_contour', 'Filled Contour'),
                variable=filled_var,
                onvalue=True,
                offvalue=False,
                command=_set_contour_filled
            )
            item_count += 1

        if self._is_graph_option_supported(graph_type, 'confidence_level', config):
            if _has_custom_ellipsis:
                custom_ellipsis_levels = self._get_custom_ellipsis_confidence_levels(config, instance_alias)
                if custom_ellipsis_levels:
                    # Replace the preset submenu with per-level visibility toggles
                    confidence_menu = tk.Menu(menu, tearoff=0)
                    hidden_levels = [
                        str(h).strip().replace('%', '')
                        for h in config.get('hidden_confidence_levels', [])
                    ]
                    for level_str in custom_ellipsis_levels:
                        level_var = tk.BooleanVar(value=(level_str not in hidden_levels))
                        _keep_var_ref(level_var)

                        def _toggle_ellipsis_level(
                            _level=level_str, _var=level_var
                        ) -> None:
                            current_hidden = [
                                str(h).strip().replace('%', '')
                                for h in config.get('hidden_confidence_levels', [])
                            ]
                            if _var.get():
                                current_hidden = [v for v in current_hidden if v != _level]
                            else:
                                if _level not in current_hidden:
                                    current_hidden.append(_level)
                            self._update_graph_section_config_option(
                                instance_alias,
                                section_id,
                                'hidden_confidence_levels',
                                current_hidden,
                                popup_refresh_callback=popup_refresh_callback,
                                refresh_analysis=True,
                            )

                        confidence_menu.add_checkbutton(
                            label=f"{level_str}%",
                            variable=level_var,
                            onvalue=True,
                            offvalue=False,
                            command=_toggle_ellipsis_level,
                        )

                    menu.add_cascade(
                        label=self.language_manager.translate(
                            'menu.graph_context.confidence_level', 'Confidence Level'
                        ),
                        menu=confidence_menu,
                    )
                    item_count += 1
                else:
                    # Custom ellipsis present but none have confidence_level flags — grey out
                    menu.add_cascade(
                        label=self.language_manager.translate(
                            'menu.graph_context.confidence_level', 'Confidence Level'
                        ),
                        menu=tk.Menu(menu, tearoff=0),
                        state=tk.DISABLED,
                    )
                    item_count += 1
            else:
                # Standard auto-ellipse confidence level submenu
                confidence_menu = tk.Menu(menu, tearoff=0)
                confidence_var = tk.StringVar(value=self._normalize_confidence_level_percent(config.get('confidence_level', '95')))
                _keep_var_ref(confidence_var)

                current_prefix = self.language_manager.translate('menu.graph_context.current_value', 'Current')
                confidence_menu.add_command(
                    label=f"{current_prefix}: {confidence_var.get()}%",
                    state=tk.DISABLED
                )
                confidence_menu.add_separator()

                def _set_confidence_level(value: str) -> None:
                    self._update_graph_section_config_option(
                        instance_alias,
                        section_id,
                        'confidence_level',
                        str(value),
                        popup_refresh_callback=popup_refresh_callback,
                        refresh_analysis=True
                    )

                for preset in ('90', '95', '99'):
                    confidence_menu.add_radiobutton(
                        label=f"{preset}%",
                        variable=confidence_var,
                        value=preset,
                        command=lambda v=preset, setter=_set_confidence_level: setter(v)
                    )

                confidence_menu.add_separator()

                def _set_custom_confidence() -> None:
                    title = self.language_manager.translate('menu.graph_context.confidence_level', 'Confidence Level')
                    prompt = self.language_manager.translate(
                        'menu.graph_context.custom_confidence_prompt',
                        'Type confidence level (%) between 0 and 100:'
                    )
                    default_value = self._normalize_confidence_level_percent(config.get('confidence_level', '95'))
                    user_value = simpledialog.askstring(title=title, prompt=prompt, initialvalue=default_value, parent=self.root)
                    if user_value is None:
                        return

                    normalized = self._normalize_confidence_level_percent(user_value)
                    numeric = float(normalized)
                    if numeric <= 0.0 or numeric >= 100.0:
                        return

                    _set_confidence_level(normalized)

                confidence_menu.add_command(
                    label=self.language_manager.translate('menu.graph_context.custom', 'Custom'),
                    command=_set_custom_confidence
                )

                menu.add_cascade(
                    label=self.language_manager.translate('menu.graph_context.confidence_level', 'Confidence Level'),
                    menu=confidence_menu
                )
                item_count += 1

        line_has_active_classes = bool(
            line_state_for_menu is not None and int(line_state_for_menu.get('layer_count', 0)) > 0
        )
        scatter_has_active_classes = bool(
            scatter_state_for_menu is not None and int(scatter_state_for_menu.get('layer_count', 0)) > 0
        )
        if self._is_graph_option_supported(graph_type, 'cmap', config) and not (
            (normalized_graph_type == 'line' and line_has_active_classes) or
            (normalized_graph_type == 'scatter' and scatter_has_active_classes)
        ):
            colormaps_data = self._load_colormaps_catalog()
            continuous_data = colormaps_data.get("continuous", {})
            qualitative_data = colormaps_data.get("qualitative", [])
            cmap_menu = tk.Menu(menu, tearoff=0)
            normalized_cmap_graph = str(graph_type or '').strip().lower()
            use_qualitative_cmaps = normalized_cmap_graph in {'line', 'scatter'}
            default_token = '__default__'
            current_cmap = str(config.get('cmap', default_token))
            cmap_var = tk.StringVar(value=current_cmap)
            _keep_var_ref(cmap_var)

            def _set_cmap(value: str) -> None:
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'cmap',
                    str(value),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            cmap_menu.add_radiobutton(
                label=self.language_manager.translate('menu.graph_context.use_default_colormap', 'Default (settings)'),
                variable=cmap_var,
                value=default_token,
                command=lambda: self._remove_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'cmap',
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )
            )
            cmap_menu.add_separator()

            has_any_cmap = False

            if use_qualitative_cmaps:
                if isinstance(qualitative_data, list):
                    for cmap_name in qualitative_data:
                        cmap_name = str(cmap_name)
                        cmap_menu.add_radiobutton(
                            label=cmap_name,
                            variable=cmap_var,
                            value=cmap_name,
                            command=lambda v=cmap_name, setter=_set_cmap: setter(v)
                        )
                        has_any_cmap = True
            else:
                if isinstance(continuous_data, list):
                    for cmap_name in continuous_data:
                        cmap_menu.add_radiobutton(
                            label=cmap_name,
                            variable=cmap_var,
                            value=cmap_name,
                            command=lambda v=cmap_name, setter=_set_cmap: setter(v)
                        )
                        has_any_cmap = True
                elif isinstance(continuous_data, dict):
                    for category, cmaps in continuous_data.items():
                        if not isinstance(cmaps, list) or not cmaps:
                            continue
                        category_menu = tk.Menu(cmap_menu, tearoff=0)
                        for cmap_name in cmaps:
                            category_menu.add_radiobutton(
                                label=cmap_name,
                                variable=cmap_var,
                                value=cmap_name,
                                command=lambda v=cmap_name, setter=_set_cmap: setter(v)
                            )
                        cmap_menu.add_cascade(label=category, menu=category_menu)
                        has_any_cmap = True

            if has_any_cmap:
                menu.add_cascade(
                    label=self.language_manager.translate('menu.graph_context.colormap', 'Colormap'),
                    menu=cmap_menu
                )
                item_count += 1

        if str(graph_type).strip().lower() == 'scatter':
            legend_menu = tk.Menu(menu, tearoff=0)

            default_suffix = " " + self.language_manager.translate('menu.default_tag', '(default)')

            raw_show_mode = str(config.get('legend_show_mode', '')).strip().lower()
            if raw_show_mode not in {'auto', 'yes', 'no'}:
                legacy_show_legend = config.get('show_legend')
                if isinstance(legacy_show_legend, bool):
                    raw_show_mode = 'yes' if legacy_show_legend else 'no'
                else:
                    raw_show_mode = 'auto'

            show_mode_var = tk.StringVar(value=raw_show_mode)
            _keep_var_ref(show_mode_var)

            def _set_legend_show_mode(value: str) -> None:
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'legend_show_mode',
                    str(value),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            show_menu = tk.Menu(legend_menu, tearoff=0)
            show_menu.add_radiobutton(
                label=self.language_manager.translate('menu.graph_context.legend_show_auto', 'Auto') + default_suffix,
                variable=show_mode_var,
                value='auto',
                command=lambda: _set_legend_show_mode('auto')
            )
            show_menu.add_radiobutton(
                label=self.language_manager.translate('menu.graph_context.legend_show_yes', 'Yes'),
                variable=show_mode_var,
                value='yes',
                command=lambda: _set_legend_show_mode('yes')
            )
            show_menu.add_radiobutton(
                label=self.language_manager.translate('menu.graph_context.legend_show_no', 'No'),
                variable=show_mode_var,
                value='no',
                command=lambda: _set_legend_show_mode('no')
            )
            legend_menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend_show', 'Show Legend'),
                menu=show_menu
            )

            legend_elements_cfg = config.get('legend_elements', {}) if isinstance(config.get('legend_elements', {}), dict) else {}
            legend_elements_state = dict(legend_elements_cfg)

            def _legend_element_default(key: str) -> bool:
                return key in {'datasets', 'color'}

            def _set_legend_element(element_key: str, enabled: bool) -> None:
                legend_elements_state[element_key] = bool(enabled)
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'legend_elements',
                    dict(legend_elements_state),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            legend_elements_menu = tk.Menu(legend_menu, tearoff=0)
            legend_element_specs = [
                ('datasets', 'menu.graph_context.legend_element_datasets', 'Datasets'),
                ('color', 'menu.graph_context.legend_element_color', 'Color'),
                ('marker', 'menu.graph_context.legend_element_marker', 'Marker'),
                ('fill', 'menu.graph_context.legend_element_fill', 'Fill Type'),
                ('edge', 'menu.graph_context.legend_element_edge', 'Edge Color'),
            ]

            for element_key, label_key, fallback in legend_element_specs:
                current_enabled = bool(legend_elements_cfg.get(element_key, _legend_element_default(element_key)))
                element_var = tk.BooleanVar(value=current_enabled)
                _keep_var_ref(element_var)
                legend_elements_menu.add_checkbutton(
                    label=self.language_manager.translate(label_key, fallback),
                    variable=element_var,
                    onvalue=True,
                    offvalue=False,
                    command=lambda k=element_key, v=element_var: _set_legend_element(k, bool(v.get()))
                )

            legend_menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend_elements', 'Elements'),
                menu=legend_elements_menu
            )

            raw_position = str(config.get('legend_position', 'auto')).strip().lower()
            if raw_position not in {'auto', 'nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'}:
                raw_position = 'auto'

            raw_location = str(config.get('legend_location', 'inside')).strip().lower()
            if raw_location not in {'inside', 'outside'}:
                raw_location = 'inside'

            legend_position_root_menu = tk.Menu(legend_menu, tearoff=0)
            legend_position_menu = tk.Menu(legend_position_root_menu, tearoff=0)
            legend_location_menu = tk.Menu(legend_position_root_menu, tearoff=0)

            legend_position_var = tk.StringVar(value=raw_position)
            legend_location_var = tk.StringVar(value=raw_location)
            _keep_var_ref(legend_position_var)
            _keep_var_ref(legend_location_var)

            def _set_legend_position(value: str) -> None:
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'legend_position',
                    str(value),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            def _set_legend_location(value: str) -> None:
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'legend_location',
                    str(value),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            legend_position_specs = [
                ('auto', self.language_manager.translate('menu.graph_context.legend_show_auto', 'Auto') + default_suffix),
                ('nw', 'NW'),
                ('n', 'N'),
                ('ne', 'NE'),
                ('e', 'E'),
                ('se', 'SE'),
                ('s', 'S'),
                ('sw', 'SW'),
                ('w', 'W'),
            ]

            for position_value, position_label in legend_position_specs:
                legend_position_menu.add_radiobutton(
                    label=position_label,
                    variable=legend_position_var,
                    value=position_value,
                    command=lambda v=position_value: _set_legend_position(v)
                )

            legend_location_menu.add_radiobutton(
                label=self.language_manager.translate('menu.graph_context.legend_location_inside', 'Inside') + default_suffix,
                variable=legend_location_var,
                value='inside',
                command=lambda: _set_legend_location('inside')
            )
            legend_location_menu.add_radiobutton(
                label=self.language_manager.translate('menu.graph_context.legend_location_outside', 'Outside'),
                variable=legend_location_var,
                value='outside',
                command=lambda: _set_legend_location('outside')
            )

            legend_position_root_menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend_position_value', 'Position'),
                menu=legend_position_menu
            )
            legend_position_root_menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend_location', 'Location'),
                menu=legend_location_menu
            )

            legend_menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend_position_root', 'Position'),
                menu=legend_position_root_menu
            )

            menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend', 'Legend'),
                menu=legend_menu
            )
            item_count += 1

            execution_results = self.analysis_data.get(instance_alias, {}).get('execution_results', {})
            outputs = self._get_execution_data_sources(execution_results, instance_alias) if isinstance(execution_results, dict) else {}
            datasets_cfg = config.get('datasets') if isinstance(config.get('datasets'), list) else None
            scatter_state = self._compute_scatter_class_layer_state(
                config,
                datasets_cfg,
                outputs,
                execution_results.get('inputs', {}) if isinstance(execution_results, dict) else {},
            )
            layer_count = int(scatter_state.get('layer_count', 0))

            if layer_count > 0:
                aspects_display = {
                    'marker': self.language_manager.translate('menu.graph_context.class_marker', 'Marker'),
                    'color': self.language_manager.translate('menu.graph_context.class_color', 'Color'),
                    'fill': self.language_manager.translate('menu.graph_context.class_fill', 'Fill Type'),
                    'edge': self.language_manager.translate('menu.graph_context.class_edge', 'Edge Color')
                }

                available_aspects = list(scatter_state.get('available_aspects', []))
                max_active = int(scatter_state.get('max_active', len(available_aspects)))
                effective_order = list(scatter_state.get('effective_order', []))
                effective_map = dict(scatter_state.get('effective_map', {}))

                class_layers_menu = tk.Menu(menu, tearoff=0)

                def _persist_layer_settings(new_order: List[str], new_map: Dict[str, int]) -> None:
                    self._update_graph_section_config_option(
                        instance_alias,
                        section_id,
                        'class_layer_user_defined',
                        True,
                        popup_refresh_callback=popup_refresh_callback,
                        refresh_analysis=False
                    )
                    self._update_graph_section_config_option(
                        instance_alias,
                        section_id,
                        'class_layer_order',
                        list(new_order),
                        popup_refresh_callback=popup_refresh_callback,
                        refresh_analysis=False
                    )
                    self._update_graph_section_config_option(
                        instance_alias,
                        section_id,
                        'class_layer_map',
                        dict(new_map),
                        popup_refresh_callback=popup_refresh_callback,
                        refresh_analysis=True
                    )

                for aspect in available_aspects:
                    aspect_layer_menu = tk.Menu(class_layers_menu, tearoff=0)
                    current_layer = int(effective_map.get(aspect, 0) if aspect in effective_order else 0)
                    layer_var = tk.IntVar(value=current_layer)
                    _keep_var_ref(layer_var)

                    def _set_aspect_layer(aspect_name: str, selected_layer: int) -> None:
                        current_order = [str(a).strip().lower() for a in effective_order if str(a).strip().lower() in available_aspects]
                        normalized_map: Dict[str, int] = {}
                        for key, value in effective_map.items():
                            try:
                                norm_key = str(key).strip().lower()
                                if norm_key in available_aspects:
                                    normalized_map[norm_key] = int(value)
                            except Exception:
                                continue

                        if not current_order:
                            raw_order_cfg = config.get('class_layer_order', [])
                            if isinstance(raw_order_cfg, str):
                                tokens = [tok.strip().lower() for tok in re.split(r'[;,\s]+', raw_order_cfg) if tok and tok.strip()]
                                current_order = [tok for tok in tokens if tok in available_aspects]
                            elif isinstance(raw_order_cfg, list):
                                current_order = [str(a).strip().lower() for a in raw_order_cfg if str(a).strip().lower() in available_aspects]

                            raw_map_cfg = config.get('class_layer_map', {})
                            if isinstance(raw_map_cfg, dict):
                                for k, v in raw_map_cfg.items():
                                    try:
                                        norm_k = str(k).strip().lower()
                                        if norm_k in available_aspects:
                                            normalized_map[norm_k] = int(v)
                                    except Exception:
                                        continue
                            elif isinstance(raw_map_cfg, str):
                                parsed_order, parsed_map = self._parse_class_layer_mapping_description(raw_map_cfg)
                                for key in parsed_order:
                                    if key in available_aspects and key in parsed_map:
                                        try:
                                            normalized_map[key] = int(parsed_map[key])
                                        except Exception:
                                            continue

                        if int(selected_layer) <= 0:
                            current_order = [a for a in current_order if a != aspect_name]
                            normalized_map.pop(aspect_name, None)
                            _persist_layer_settings(current_order, normalized_map)
                            return

                        if aspect_name not in current_order:
                            if len(current_order) >= max_active:
                                return
                            current_order.append(aspect_name)
                        normalized_map[aspect_name] = max(1, min(int(selected_layer), layer_count))
                        _persist_layer_settings(current_order, normalized_map)

                    aspect_layer_menu.add_radiobutton(
                        label=self.language_manager.translate('menu.graph_context.none', 'None'),
                        variable=layer_var,
                        value=0,
                        command=lambda a=aspect: _set_aspect_layer(a, 0)
                    )

                    aspect_layer_menu.add_separator()

                    for layer_idx in range(1, layer_count + 1):
                        aspect_layer_menu.add_radiobutton(
                            label=f"Layer {layer_idx}",
                            variable=layer_var,
                            value=layer_idx,
                            command=lambda a=aspect, l=layer_idx: _set_aspect_layer(a, l)
                        )

                    class_layers_menu.add_cascade(
                        label=f"{aspects_display.get(aspect, aspect.title())} Layer",
                        menu=aspect_layer_menu
                    )

                class_layers_menu.add_separator()

                def _edit_mapping_text() -> None:
                    current_desc = ", ".join(f"{k}:{effective_map.get(k, i+1)}" for i, k in enumerate(effective_order))
                    user_text = simpledialog.askstring(
                        title=self.language_manager.translate('menu.graph_context.class_layer_mapping', 'Class Layer Mapping'),
                        prompt=self.language_manager.translate(
                            'menu.graph_context.class_layer_mapping_prompt',
                            "Describe mapping as 'marker:1, color:2, fill:3, edge:4'"
                        ),
                        initialvalue=current_desc,
                        parent=self.root
                    )
                    if user_text is None:
                        return
                    parsed_order, parsed_map = self._parse_class_layer_mapping_description(user_text)
                    parsed_order = [a for a in parsed_order if a in available_aspects]
                    parsed_order = parsed_order[:max_active]
                    if not parsed_order and available_aspects:
                        parsed_order = [available_aspects[0]]
                    bounded_map = {a: max(1, min(int(parsed_map.get(a, 1)), layer_count)) for a in parsed_order}
                    _persist_layer_settings(parsed_order, bounded_map)

                class_layers_menu.add_command(
                    label=self.language_manager.translate('menu.graph_context.class_layer_mapping_entry', 'Layer Mapping Description...'),
                    command=_edit_mapping_text
                )

                menu.add_cascade(
                    label=self.language_manager.translate('menu.graph_context.class_layers', 'Class Layers'),
                    menu=class_layers_menu
                )
                item_count += 1

                numeric_layers = list(scatter_state.get('numeric_layers', []))
                if numeric_layers:
                    class_nature_menu = tk.Menu(menu, tearoff=0)
                    current_nature = scatter_state.get('layer_nature', {})

                    for layer_idx in numeric_layers:
                        layer_menu = tk.Menu(class_nature_menu, tearoff=0)
                        nature_var = tk.StringVar(value=str(current_nature.get(layer_idx, 'discrete')))
                        _keep_var_ref(nature_var)

                        def _set_nature(idx: int, value: str) -> None:
                            overrides = config.get('class_layer_nature', {}) if isinstance(config.get('class_layer_nature', {}), dict) else {}
                            updated = dict(overrides)
                            updated[str(idx)] = value
                            self._update_graph_section_config_option(
                                instance_alias,
                                section_id,
                                'class_layer_nature',
                                updated,
                                popup_refresh_callback=popup_refresh_callback,
                                refresh_analysis=True
                            )

                        layer_menu.add_radiobutton(
                            label=self.language_manager.translate('menu.graph_context.class_discrete', 'Discrete'),
                            variable=nature_var,
                            value='discrete',
                            command=lambda i=layer_idx: _set_nature(i, 'discrete')
                        )
                        layer_menu.add_radiobutton(
                            label=self.language_manager.translate('menu.graph_context.class_continuous', 'Continuous'),
                            variable=nature_var,
                            value='continuous',
                            command=lambda i=layer_idx: _set_nature(i, 'continuous')
                        )

                        class_nature_menu.add_cascade(label=f"Layer {layer_idx}", menu=layer_menu)

                    menu.add_cascade(
                        label=self.language_manager.translate('menu.graph_context.class_nature', 'Class Nature'),
                        menu=class_nature_menu
                    )
                    item_count += 1

                colormaps_data = self._load_colormaps_catalog()
                continuous_data = colormaps_data.get("continuous", {})
                qualitative_data = colormaps_data.get("qualitative", [])

                def _build_aspect_colormap_menu(aspect: str, cont_key: str, qual_key: str) -> tk.Menu:
                    aspect_menu = tk.Menu(menu, tearoff=0)
                    mapped_layer = effective_map.get(aspect) if aspect in effective_order else None
                    if mapped_layer is None:
                        aspect_menu.add_command(
                            label=self.language_manager.translate(
                                'menu.graph_context.assign_layer_first',
                                'Assign a class layer first'
                            ),
                            state=tk.DISABLED
                        )
                        return aspect_menu

                    layer_nature = str(scatter_state.get('layer_nature', {}).get(int(mapped_layer), 'discrete')).strip().lower()
                    is_continuous = layer_nature == 'continuous'
                    target_key = cont_key if is_continuous else qual_key
                    default_token = '__default__'
                    current_value = str(config.get(target_key, default_token))
                    cmap_var = tk.StringVar(value=current_value)
                    _keep_var_ref(cmap_var)

                    aspect_menu.add_radiobutton(
                        label=self.language_manager.translate('menu.graph_context.use_default_colormap', 'Default (settings)'),
                        variable=cmap_var,
                        value=default_token,
                        command=lambda ck=cont_key, qk=qual_key: (
                            self._remove_graph_section_config_option(
                                instance_alias,
                                section_id,
                                ck,
                                popup_refresh_callback=popup_refresh_callback,
                                refresh_analysis=False
                            ),
                            self._remove_graph_section_config_option(
                                instance_alias,
                                section_id,
                                qk,
                                popup_refresh_callback=popup_refresh_callback,
                                refresh_analysis=True
                            )
                        )
                    )
                    aspect_menu.add_separator()

                    has_any_maps = False
                    if is_continuous:
                        if isinstance(continuous_data, dict):
                            for category, cmaps in continuous_data.items():
                                if not isinstance(cmaps, list) or not cmaps:
                                    continue
                                category_menu = tk.Menu(aspect_menu, tearoff=0)
                                for cmap_name in cmaps:
                                    cmap_name = str(cmap_name)
                                    category_menu.add_radiobutton(
                                        label=cmap_name,
                                        variable=cmap_var,
                                        value=cmap_name,
                                        command=lambda v=cmap_name, k=target_key: self._update_graph_section_config_option(
                                            instance_alias,
                                            section_id,
                                            k,
                                            v,
                                            popup_refresh_callback=popup_refresh_callback,
                                            refresh_analysis=True
                                        )
                                    )
                                aspect_menu.add_cascade(label=str(category), menu=category_menu)
                                has_any_maps = True
                        elif isinstance(continuous_data, list):
                            for cmap_name in continuous_data:
                                cmap_name = str(cmap_name)
                                aspect_menu.add_radiobutton(
                                    label=cmap_name,
                                    variable=cmap_var,
                                    value=cmap_name,
                                    command=lambda v=cmap_name, k=target_key: self._update_graph_section_config_option(
                                        instance_alias,
                                        section_id,
                                        k,
                                        v,
                                        popup_refresh_callback=popup_refresh_callback,
                                        refresh_analysis=True
                                    )
                                )
                                has_any_maps = True
                    else:
                        if isinstance(qualitative_data, list):
                            for cmap_name in qualitative_data:
                                cmap_name = str(cmap_name)
                                aspect_menu.add_radiobutton(
                                    label=cmap_name,
                                    variable=cmap_var,
                                    value=cmap_name,
                                    command=lambda v=cmap_name, k=target_key: self._update_graph_section_config_option(
                                        instance_alias,
                                        section_id,
                                        k,
                                        v,
                                        popup_refresh_callback=popup_refresh_callback,
                                        refresh_analysis=True
                                    )
                                )
                                has_any_maps = True

                    if not has_any_maps:
                        aspect_menu.add_command(label='-', state=tk.DISABLED)

                    return aspect_menu

                class_colormaps_menu = tk.Menu(menu, tearoff=0)
                class_colormaps_menu.add_cascade(
                    label=aspects_display.get('color', 'Color'),
                    menu=_build_aspect_colormap_menu('color', 'class_color_cmap_continuous', 'class_color_cmap_qualitative')
                )
                class_colormaps_menu.add_cascade(
                    label=aspects_display.get('edge', 'Edge Color'),
                    menu=_build_aspect_colormap_menu('edge', 'class_edge_cmap_continuous', 'class_edge_cmap_qualitative')
                )

                menu.add_cascade(
                    label=self.language_manager.translate('menu.graph_context.class_colormaps', 'Class Colormaps'),
                    menu=class_colormaps_menu
                )
                item_count += 1

        if str(graph_type).strip().lower() == 'line':
            legend_menu = tk.Menu(menu, tearoff=0)

            default_suffix = " " + self.language_manager.translate('menu.default_tag', '(default)')

            raw_show_mode = str(config.get('legend_show_mode', '')).strip().lower()
            if raw_show_mode not in {'auto', 'yes', 'no'}:
                legacy_show_legend = config.get('show_legend')
                if isinstance(legacy_show_legend, bool):
                    raw_show_mode = 'yes' if legacy_show_legend else 'no'
                else:
                    raw_show_mode = 'auto'

            show_mode_var = tk.StringVar(value=raw_show_mode)
            _keep_var_ref(show_mode_var)

            def _set_line_legend_show_mode(value: str) -> None:
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'legend_show_mode',
                    str(value),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            show_menu = tk.Menu(legend_menu, tearoff=0)
            show_menu.add_radiobutton(
                label=self.language_manager.translate('menu.graph_context.legend_show_auto', 'Auto') + default_suffix,
                variable=show_mode_var,
                value='auto',
                command=lambda: _set_line_legend_show_mode('auto')
            )
            show_menu.add_radiobutton(
                label=self.language_manager.translate('menu.graph_context.legend_show_yes', 'Yes'),
                variable=show_mode_var,
                value='yes',
                command=lambda: _set_line_legend_show_mode('yes')
            )
            show_menu.add_radiobutton(
                label=self.language_manager.translate('menu.graph_context.legend_show_no', 'No'),
                variable=show_mode_var,
                value='no',
                command=lambda: _set_line_legend_show_mode('no')
            )
            legend_menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend_show', 'Show Legend'),
                menu=show_menu
            )

            legend_elements_cfg = config.get('legend_elements', {}) if isinstance(config.get('legend_elements', {}), dict) else {}
            legend_elements_state = dict(legend_elements_cfg)

            def _line_legend_element_default(key: str) -> bool:
                return key in {'datasets', 'color'}

            def _set_line_legend_element(element_key: str, enabled: bool) -> None:
                legend_elements_state[element_key] = bool(enabled)
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'legend_elements',
                    dict(legend_elements_state),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            legend_elements_menu = tk.Menu(legend_menu, tearoff=0)
            legend_element_specs = [
                ('datasets', 'menu.graph_context.legend_element_datasets', 'Datasets'),
                ('color', 'menu.graph_context.legend_element_color', 'Color'),
                ('linestyle', 'menu.graph_context.legend_element_line_style', 'Line Style'),
                ('marker', 'menu.graph_context.legend_element_marker', 'Marker'),
            ]

            for element_key, label_key, fallback in legend_element_specs:
                current_enabled = bool(legend_elements_cfg.get(element_key, _line_legend_element_default(element_key)))
                element_var = tk.BooleanVar(value=current_enabled)
                _keep_var_ref(element_var)
                legend_elements_menu.add_checkbutton(
                    label=self.language_manager.translate(label_key, fallback),
                    variable=element_var,
                    onvalue=True,
                    offvalue=False,
                    command=lambda k=element_key, v=element_var: _set_line_legend_element(k, bool(v.get()))
                )

            legend_menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend_elements', 'Elements'),
                menu=legend_elements_menu
            )

            raw_position = str(config.get('legend_position', 'auto')).strip().lower()
            if raw_position not in {'auto', 'nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'}:
                raw_position = 'auto'

            raw_location = str(config.get('legend_location', 'inside')).strip().lower()
            if raw_location not in {'inside', 'outside'}:
                raw_location = 'inside'

            legend_position_root_menu = tk.Menu(legend_menu, tearoff=0)
            legend_position_menu = tk.Menu(legend_position_root_menu, tearoff=0)
            legend_location_menu = tk.Menu(legend_position_root_menu, tearoff=0)

            legend_position_var = tk.StringVar(value=raw_position)
            legend_location_var = tk.StringVar(value=raw_location)
            _keep_var_ref(legend_position_var)
            _keep_var_ref(legend_location_var)

            def _set_line_legend_position(value: str) -> None:
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'legend_position',
                    str(value),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            def _set_line_legend_location(value: str) -> None:
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'legend_location',
                    str(value),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            legend_position_specs = [
                ('auto', self.language_manager.translate('menu.graph_context.legend_show_auto', 'Auto') + default_suffix),
                ('nw', 'NW'), ('n', 'N'), ('ne', 'NE'), ('e', 'E'), ('se', 'SE'), ('s', 'S'), ('sw', 'SW'), ('w', 'W'),
            ]

            for position_value, position_label in legend_position_specs:
                legend_position_menu.add_radiobutton(
                    label=position_label,
                    variable=legend_position_var,
                    value=position_value,
                    command=lambda v=position_value: _set_line_legend_position(v)
                )

            legend_location_menu.add_radiobutton(
                label=self.language_manager.translate('menu.graph_context.legend_location_inside', 'Inside') + default_suffix,
                variable=legend_location_var,
                value='inside',
                command=lambda: _set_line_legend_location('inside')
            )
            legend_location_menu.add_radiobutton(
                label=self.language_manager.translate('menu.graph_context.legend_location_outside', 'Outside'),
                variable=legend_location_var,
                value='outside',
                command=lambda: _set_line_legend_location('outside')
            )

            legend_position_root_menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend_position_value', 'Position'),
                menu=legend_position_menu
            )
            legend_position_root_menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend_location', 'Location'),
                menu=legend_location_menu
            )

            legend_menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend_position_root', 'Position'),
                menu=legend_position_root_menu
            )

            menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.legend', 'Legend'),
                menu=legend_menu
            )
            item_count += 1

            execution_results = self.analysis_data.get(instance_alias, {}).get('execution_results', {})
            outputs = self._get_execution_data_sources(execution_results, instance_alias) if isinstance(execution_results, dict) else {}
            datasets_cfg = config.get('datasets') if isinstance(config.get('datasets'), list) else None
            line_state = line_state_for_menu if line_state_for_menu is not None else self._compute_line_class_layer_state(
                config,
                datasets_cfg,
                outputs,
                execution_results.get('inputs', {}) if isinstance(execution_results, dict) else {},
            )
            layer_count = int(line_state.get('layer_count', 0))

            if layer_count > 0:
                line_aspects_display = {
                    'marker': self.language_manager.translate('menu.graph_context.class_marker', 'Marker'),
                    'color': self.language_manager.translate('menu.graph_context.class_color', 'Color'),
                    'linestyle': self.language_manager.translate('menu.graph_context.class_line_style', 'Line Style')
                }

                available_aspects = list(line_state.get('available_aspects', []))
                max_active = int(line_state.get('max_active', len(available_aspects)))
                effective_order = list(line_state.get('effective_order', []))
                effective_map = dict(line_state.get('effective_map', {}))

                class_layers_menu = tk.Menu(menu, tearoff=0)

                def _persist_line_layer_settings(new_order: List[str], new_map: Dict[str, int]) -> None:
                    self._update_graph_section_config_option(
                        instance_alias,
                        section_id,
                        'class_layer_user_defined',
                        True,
                        popup_refresh_callback=popup_refresh_callback,
                        refresh_analysis=False
                    )
                    self._update_graph_section_config_option(
                        instance_alias,
                        section_id,
                        'class_layer_order',
                        list(new_order),
                        popup_refresh_callback=popup_refresh_callback,
                        refresh_analysis=False
                    )
                    self._update_graph_section_config_option(
                        instance_alias,
                        section_id,
                        'class_layer_map',
                        dict(new_map),
                        popup_refresh_callback=popup_refresh_callback,
                        refresh_analysis=True
                    )

                for aspect in available_aspects:
                    aspect_layer_menu = tk.Menu(class_layers_menu, tearoff=0)
                    current_layer = int(effective_map.get(aspect, 0) if aspect in effective_order else 0)
                    layer_var = tk.IntVar(value=current_layer)
                    _keep_var_ref(layer_var)

                    def _set_line_aspect_layer(aspect_name: str, selected_layer: int) -> None:
                        current_order = [str(a).strip().lower() for a in effective_order if str(a).strip().lower() in available_aspects]
                        normalized_map: Dict[str, int] = {}
                        for key, value in effective_map.items():
                            try:
                                norm_key = str(key).strip().lower()
                                if norm_key in available_aspects:
                                    normalized_map[norm_key] = int(value)
                            except Exception:
                                continue

                        if not current_order:
                            raw_order_cfg = config.get('class_layer_order', [])
                            if isinstance(raw_order_cfg, str):
                                tokens = [tok.strip().lower() for tok in re.split(r'[;,\s]+', raw_order_cfg) if tok and tok.strip()]
                                current_order = [tok for tok in tokens if tok in available_aspects]
                            elif isinstance(raw_order_cfg, list):
                                current_order = [str(a).strip().lower() for a in raw_order_cfg if str(a).strip().lower() in available_aspects]

                            raw_map_cfg = config.get('class_layer_map', {})
                            if isinstance(raw_map_cfg, dict):
                                for k, v in raw_map_cfg.items():
                                    try:
                                        norm_k = str(k).strip().lower()
                                        if norm_k in available_aspects:
                                            normalized_map[norm_k] = int(v)
                                    except Exception:
                                        continue
                            elif isinstance(raw_map_cfg, str):
                                parsed_order, parsed_map = self._parse_class_layer_mapping_description(raw_map_cfg)
                                for key in parsed_order:
                                    if key in available_aspects and key in parsed_map:
                                        try:
                                            normalized_map[key] = int(parsed_map[key])
                                        except Exception:
                                            continue

                        if int(selected_layer) <= 0:
                            current_order = [a for a in current_order if a != aspect_name]
                            normalized_map.pop(aspect_name, None)
                            _persist_line_layer_settings(current_order, normalized_map)
                            return

                        if aspect_name not in current_order:
                            if len(current_order) >= max_active:
                                return
                            current_order.append(aspect_name)
                        normalized_map[aspect_name] = max(1, min(int(selected_layer), layer_count))
                        _persist_line_layer_settings(current_order, normalized_map)

                    aspect_layer_menu.add_radiobutton(
                        label=self.language_manager.translate('menu.graph_context.none', 'None'),
                        variable=layer_var,
                        value=0,
                        command=lambda a=aspect: _set_line_aspect_layer(a, 0)
                    )
                    aspect_layer_menu.add_separator()

                    for layer_idx in range(1, layer_count + 1):
                        aspect_layer_menu.add_radiobutton(
                            label=f"Layer {layer_idx}",
                            variable=layer_var,
                            value=layer_idx,
                            command=lambda a=aspect, l=layer_idx: _set_line_aspect_layer(a, l)
                        )

                    class_layers_menu.add_cascade(
                        label=f"{line_aspects_display.get(aspect, aspect.title())} Layer",
                        menu=aspect_layer_menu
                    )

                class_layers_menu.add_separator()

                def _edit_line_mapping_text() -> None:
                    current_desc = ", ".join(f"{k}:{effective_map.get(k, i+1)}" for i, k in enumerate(effective_order))
                    user_text = simpledialog.askstring(
                        title=self.language_manager.translate('menu.graph_context.class_layer_mapping', 'Class Layer Mapping'),
                        prompt=self.language_manager.translate(
                            'menu.graph_context.class_layer_mapping_prompt',
                            "Describe mapping as 'color:1, linestyle:2, marker:3'"
                        ),
                        initialvalue=current_desc,
                        parent=self.root
                    )
                    if user_text is None:
                        return
                    parsed_order, parsed_map = self._parse_class_layer_mapping_description(user_text)
                    parsed_order = [a for a in parsed_order if a in available_aspects]
                    parsed_order = parsed_order[:max_active]
                    if not parsed_order and available_aspects:
                        parsed_order = [available_aspects[0]]
                    bounded_map = {a: max(1, min(int(parsed_map.get(a, 1)), layer_count)) for a in parsed_order}
                    _persist_line_layer_settings(parsed_order, bounded_map)

                class_layers_menu.add_command(
                    label=self.language_manager.translate('menu.graph_context.class_layer_mapping_entry', 'Layer Mapping Description...'),
                    command=_edit_line_mapping_text
                )

                menu.add_cascade(
                    label=self.language_manager.translate('menu.graph_context.class_layers', 'Class Layers'),
                    menu=class_layers_menu
                )
                item_count += 1

                numeric_layers = list(line_state.get('numeric_layers', []))
                if numeric_layers:
                    class_nature_menu = tk.Menu(menu, tearoff=0)
                    current_nature = line_state.get('layer_nature', {})

                    for layer_idx in numeric_layers:
                        layer_menu = tk.Menu(class_nature_menu, tearoff=0)
                        nature_var = tk.StringVar(value=str(current_nature.get(layer_idx, 'discrete')))
                        _keep_var_ref(nature_var)

                        def _set_line_nature(idx: int, value: str) -> None:
                            overrides = config.get('class_layer_nature', {}) if isinstance(config.get('class_layer_nature', {}), dict) else {}
                            updated = dict(overrides)
                            updated[str(idx)] = value
                            self._update_graph_section_config_option(
                                instance_alias,
                                section_id,
                                'class_layer_nature',
                                updated,
                                popup_refresh_callback=popup_refresh_callback,
                                refresh_analysis=True
                            )

                        layer_menu.add_radiobutton(
                            label=self.language_manager.translate('menu.graph_context.class_discrete', 'Discrete'),
                            variable=nature_var,
                            value='discrete',
                            command=lambda i=layer_idx: _set_line_nature(i, 'discrete')
                        )
                        layer_menu.add_radiobutton(
                            label=self.language_manager.translate('menu.graph_context.class_continuous', 'Continuous'),
                            variable=nature_var,
                            value='continuous',
                            command=lambda i=layer_idx: _set_line_nature(i, 'continuous')
                        )

                        class_nature_menu.add_cascade(label=f"Layer {layer_idx}", menu=layer_menu)

                    menu.add_cascade(
                        label=self.language_manager.translate('menu.graph_context.class_nature', 'Class Nature'),
                        menu=class_nature_menu
                    )
                    item_count += 1

                colormaps_data = self._load_colormaps_catalog()
                continuous_data = colormaps_data.get("continuous", {})
                qualitative_data = colormaps_data.get("qualitative", [])

                def _build_line_color_colormap_menu() -> tk.Menu:
                    aspect_menu = tk.Menu(menu, tearoff=0)
                    mapped_layer = effective_map.get('color') if 'color' in effective_order else None
                    if mapped_layer is None:
                        aspect_menu.add_command(
                            label=self.language_manager.translate(
                                'menu.graph_context.assign_layer_first',
                                'Assign a class layer first'
                            ),
                            state=tk.DISABLED
                        )
                        return aspect_menu

                    layer_nature = str(line_state.get('layer_nature', {}).get(int(mapped_layer), 'discrete')).strip().lower()
                    is_continuous = layer_nature == 'continuous'
                    target_key = 'class_color_cmap_continuous' if is_continuous else 'class_color_cmap_qualitative'
                    default_token = '__default__'
                    current_value = str(config.get(target_key, default_token))
                    cmap_var = tk.StringVar(value=current_value)
                    _keep_var_ref(cmap_var)

                    aspect_menu.add_radiobutton(
                        label=self.language_manager.translate('menu.graph_context.use_default_colormap', 'Default (settings)'),
                        variable=cmap_var,
                        value=default_token,
                        command=lambda: (
                            self._remove_graph_section_config_option(
                                instance_alias,
                                section_id,
                                'class_color_cmap_continuous',
                                popup_refresh_callback=popup_refresh_callback,
                                refresh_analysis=False
                            ),
                            self._remove_graph_section_config_option(
                                instance_alias,
                                section_id,
                                'class_color_cmap_qualitative',
                                popup_refresh_callback=popup_refresh_callback,
                                refresh_analysis=True
                            )
                        )
                    )
                    aspect_menu.add_separator()

                    has_any_maps = False
                    if is_continuous:
                        if isinstance(continuous_data, dict):
                            for category, cmaps in continuous_data.items():
                                if not isinstance(cmaps, list) or not cmaps:
                                    continue
                                category_menu = tk.Menu(aspect_menu, tearoff=0)
                                for cmap_name in cmaps:
                                    cmap_name = str(cmap_name)
                                    category_menu.add_radiobutton(
                                        label=cmap_name,
                                        variable=cmap_var,
                                        value=cmap_name,
                                        command=lambda v=cmap_name, k=target_key: self._update_graph_section_config_option(
                                            instance_alias,
                                            section_id,
                                            k,
                                            v,
                                            popup_refresh_callback=popup_refresh_callback,
                                            refresh_analysis=True
                                        )
                                    )
                                aspect_menu.add_cascade(label=str(category), menu=category_menu)
                                has_any_maps = True
                    else:
                        if isinstance(qualitative_data, list):
                            for cmap_name in qualitative_data:
                                cmap_name = str(cmap_name)
                                aspect_menu.add_radiobutton(
                                    label=cmap_name,
                                    variable=cmap_var,
                                    value=cmap_name,
                                    command=lambda v=cmap_name, k=target_key: self._update_graph_section_config_option(
                                        instance_alias,
                                        section_id,
                                        k,
                                        v,
                                        popup_refresh_callback=popup_refresh_callback,
                                        refresh_analysis=True
                                    )
                                )
                                has_any_maps = True

                    if not has_any_maps:
                        aspect_menu.add_command(label='-', state=tk.DISABLED)
                    return aspect_menu

                class_colormaps_menu = tk.Menu(menu, tearoff=0)
                class_colormaps_menu.add_cascade(
                    label=line_aspects_display.get('color', 'Color'),
                    menu=_build_line_color_colormap_menu()
                )

                menu.add_cascade(
                    label=self.language_manager.translate('menu.graph_context.class_colormaps', 'Class Colormaps'),
                    menu=class_colormaps_menu
                )
                item_count += 1

        rendered_dataset_entries = self._get_rendered_dataset_visibility_entries(instance_alias, section_id)
        if rendered_dataset_entries:
            visibility_menu = tk.Menu(menu, tearoff=0)
            visibility_cfg = config.get('dataset_visibility', {}) if isinstance(config.get('dataset_visibility', {}), dict) else {}
            visibility_state: Dict[str, bool] = {
                str(entry.get('key', '')): bool(visibility_cfg.get(str(entry.get('key', '')), True))
                for entry in rendered_dataset_entries
            }

            label_counts: Dict[str, int] = {}
            for entry in rendered_dataset_entries:
                label = str(entry.get('label', 'Dataset')).strip() or 'Dataset'
                label_counts[label] = label_counts.get(label, 0) + 1

            duplicate_seen: Dict[str, int] = {}

            def _persist_dataset_visibility_state() -> None:
                self._update_graph_section_config_option(
                    instance_alias,
                    section_id,
                    'dataset_visibility',
                    dict(visibility_state),
                    popup_refresh_callback=popup_refresh_callback,
                    refresh_analysis=True
                )

            def _set_dataset_visibility(vis_key: str, var_obj: tk.BooleanVar) -> None:
                visible_count = sum(1 for is_visible in visibility_state.values() if is_visible)
                requested_visible = bool(var_obj.get())
                currently_visible = bool(visibility_state.get(vis_key, True))

                if not requested_visible and currently_visible and visible_count <= 1:
                    var_obj.set(True)
                    self._show_fading_message(
                        self.language_manager.translate(
                            'ui.messages.dataset_visibility_keep_one',
                            'At least one dataset must remain visible.'
                        )
                    )
                    return

                visibility_state[vis_key] = requested_visible
                _persist_dataset_visibility_state()

            for entry in rendered_dataset_entries:
                vis_key = str(entry.get('key', ''))
                base_label = str(entry.get('label', 'Dataset')).strip() or 'Dataset'
                duplicate_seen[base_label] = duplicate_seen.get(base_label, 0) + 1
                display_label = (
                    f"{base_label} ({duplicate_seen[base_label]})"
                    if label_counts.get(base_label, 0) > 1
                    else base_label
                )

                vis_var = tk.BooleanVar(value=bool(visibility_state.get(vis_key, True)))
                _keep_var_ref(vis_var)
                visibility_menu.add_checkbutton(
                    label=display_label,
                    variable=vis_var,
                    onvalue=True,
                    offvalue=False,
                    command=lambda k=vis_key, v=vis_var: _set_dataset_visibility(k, v)
                )

            menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.dataset_visibility', 'Dataset Visibility'),
                menu=visibility_menu
            )
            item_count += 1

        axis_type_values = ('linear', 'log10', 'log2', 'ln')
        axis_scale_label = self.language_manager.translate('menu.graph_context.axis_scale', 'Axis Scale')
        axis_scale_labels = {
            'linear': self.language_manager.translate('menu.graph_context.axis_scale_linear', 'Linear'),
            'log10': self.language_manager.translate('menu.graph_context.axis_scale_log10', 'Log10'),
            'log2': self.language_manager.translate('menu.graph_context.axis_scale_log2', 'Log2'),
            'ln': self.language_manager.translate('menu.graph_context.axis_scale_ln', 'Ln')
        }

        axis_root_menu = tk.Menu(menu, tearoff=0)
        axis_root_count = 0

        flip_for_axis_menu = False
        if self._is_graph_option_supported(graph_type, 'flip_xy', config):
            flip_for_axis_menu = _normalize_bool_setting(config.get('flip_xy'), False)

        axis_specs = [
            ('x_axis', 'x_axis', 'x_axis_type', 'x_force_integer', 'x_reverse_axis', 'menu.graph_context.axis_x', 'X Axis'),
            ('y_axis', 'y_axis', 'y_axis_type', 'y_force_integer', 'y_reverse_axis', 'menu.graph_context.axis_y', 'Y Axis'),
            ('z_axis', 'z_axis', 'z_axis_type', 'z_force_integer', 'z_reverse_axis', 'menu.graph_context.axis_z', 'Z Axis')
        ]

        if flip_for_axis_menu:
            axis_specs = [
                ('x_axis', 'y_axis', 'x_axis_type', 'x_force_integer', 'x_reverse_axis', 'menu.graph_context.axis_x', 'X Axis'),
                ('y_axis', 'x_axis', 'y_axis_type', 'y_force_integer', 'y_reverse_axis', 'menu.graph_context.axis_y', 'Y Axis'),
                ('z_axis', 'z_axis', 'z_axis_type', 'z_force_integer', 'z_reverse_axis', 'menu.graph_context.axis_z', 'Z Axis')
            ]

        for _display_axis_key, actual_axis_key, axis_type_option, axis_force_option, axis_reverse_option, axis_label_key, axis_fallback in axis_specs:
            has_axis_type = self._is_graph_option_supported(graph_type, axis_type_option, config)
            has_force_integer = self._is_graph_option_supported(graph_type, axis_force_option, config)
            has_reverse_axis = self._is_graph_option_supported(graph_type, axis_reverse_option, config)
            if not has_axis_type and not has_force_integer and not has_reverse_axis:
                continue

            axis_menu = tk.Menu(axis_root_menu, tearoff=0)
            axis_cfg = config.get(actual_axis_key, {}) if isinstance(config.get(actual_axis_key), dict) else {}

            if has_axis_type:
                scale_menu = tk.Menu(axis_menu, tearoff=0)
                axis_type_var = tk.StringVar(value=str(axis_cfg.get('axis_type', 'linear')).strip().lower())
                if axis_type_var.get() not in axis_type_values:
                    axis_type_var.set('linear')
                _keep_var_ref(axis_type_var)

                def _set_axis_type(value: str, _axis_key=actual_axis_key) -> None:
                    self._update_graph_axis_config_option(
                        instance_alias,
                        section_id,
                        _axis_key,
                        'axis_type',
                        value,
                        popup_refresh_callback=popup_refresh_callback,
                        refresh_analysis=True
                    )

                for axis_type_name in axis_type_values:
                    scale_menu.add_radiobutton(
                        label=axis_scale_labels.get(axis_type_name, axis_type_name),
                        variable=axis_type_var,
                        value=axis_type_name,
                        command=lambda v=axis_type_name, setter=_set_axis_type: setter(v)
                    )

                axis_menu.add_cascade(label=axis_scale_label, menu=scale_menu)

            if has_force_integer:
                force_mode_var = tk.StringVar(
                    value=_normalize_force_integer_mode_setting(axis_cfg.get('force_integer', False))
                )
                _keep_var_ref(force_mode_var)

                def _set_force_integer_mode(mode: str, _axis_key=actual_axis_key) -> None:
                    normalized_mode = _normalize_force_integer_mode_setting(mode)
                    force_value: Any
                    if normalized_mode == 'true':
                        force_value = True
                    elif normalized_mode == 'conditional':
                        force_value = 'conditional'
                    else:
                        force_value = False

                    self._update_graph_axis_config_option(
                        instance_alias,
                        section_id,
                        _axis_key,
                        'force_integer',
                        force_value,
                        popup_refresh_callback=popup_refresh_callback,
                        refresh_analysis=True
                    )

                force_menu = tk.Menu(axis_menu, tearoff=0)
                force_mode_options = [
                    (
                        'false',
                        self.language_manager.translate('menu.graph_context.force_integer_mode_off', 'Off')
                    ),
                    (
                        'true',
                        self.language_manager.translate('menu.graph_context.force_integer_mode_on', 'Always')
                    ),
                    (
                        'conditional',
                        self.language_manager.translate('menu.graph_context.force_integer_mode_conditional', 'Conditional')
                    )
                ]
                for mode_value, mode_label in force_mode_options:
                    force_menu.add_radiobutton(
                        label=mode_label,
                        variable=force_mode_var,
                        value=mode_value,
                        command=lambda v=mode_value: _set_force_integer_mode(v)
                    )

                axis_menu.add_cascade(
                    label=self.language_manager.translate('menu.graph_context.force_integer_ticks', 'Force Integer Ticks'),
                    menu=force_menu
                )

            if has_reverse_axis:
                reverse_var = tk.BooleanVar(value=bool(axis_cfg.get('reverse_axis', False)))
                _keep_var_ref(reverse_var)

                def _set_reverse_axis(_axis_key=actual_axis_key, _reverse_var=reverse_var) -> None:
                    self._update_graph_axis_config_option(
                        instance_alias,
                        section_id,
                        _axis_key,
                        'reverse_axis',
                        bool(_reverse_var.get()),
                        popup_refresh_callback=popup_refresh_callback,
                        refresh_analysis=True
                    )

                axis_menu.add_checkbutton(
                    label=self.language_manager.translate('menu.graph_context.reverse_axis', 'Reverse Axis'),
                    variable=reverse_var,
                    onvalue=True,
                    offvalue=False,
                    command=_set_reverse_axis
                )

            axis_root_menu.add_cascade(
                label=self.language_manager.translate(axis_label_key, axis_fallback),
                menu=axis_menu
            )
            axis_root_count += 1

        if axis_root_count > 0:
            menu.add_cascade(
                label=self.language_manager.translate('menu.graph_context.axes', 'Axes'),
                menu=axis_root_menu
            )
            item_count += 1

        return menu if item_count > 0 else None

    def _attach_graph_context_menu(self, canvas, graph_type: str, config: dict,
                                   instance_alias: str, section_id: Tuple[int, int],
                                   popup_refresh_callback: Optional[Callable[[], None]] = None) -> None:
        """Attach right-click graph context menu when graph type has supported options."""
        try:
            widget = canvas.get_tk_widget()
        except Exception:
            return

        def _on_right_click(event) -> None:
            self._set_active_analysis_section(instance_alias, section_id)

            context_menu = self._build_graph_context_menu(
                graph_type,
                config,
                instance_alias,
                section_id,
                popup_refresh_callback=popup_refresh_callback
            )
            if context_menu is None:
                return

            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()

        try:
            widget.bind("<Button-3>", _on_right_click, add="+")
        except tk.TclError:
            pass

    def _preload_heavy_dependencies(self):
        """Preload heavy/optional modules when eager mode is selected."""
        self._get_graph_renderer()
        self._get_reporting_functions()
        self._get_add_graph_dialog_func()
        self._get_add_table_dialog_func()
        self._get_add_text_dialog_func()
        self._get_routing_map_window_class()
        self._get_matplotlib_pyplot()

    def _set_import_loading_mode(self, mode: str):
        """Persist and apply import loading mode preference."""
        normalized = self._normalize_import_loading_mode(mode)
        if self.import_loading_mode_var.get() != normalized:
            self.import_loading_mode_var.set(normalized)

        if normalized == self.import_loading_mode:
            return

        self.import_loading_mode = normalized
        self.settings_manager.set("import_loading_mode", normalized)

        if normalized == "eager":
            self._preload_heavy_dependencies()
            self._show_fading_message(
                self.language_manager.translate(
                    "ui.messages.import_mode_eager_enabled",
                    "Eager loading enabled. Heavy modules are preloaded now and on startup."
                )
            )
        else:
            self._show_fading_message(
                self.language_manager.translate(
                    "ui.messages.import_mode_lazy_enabled",
                    "Lazy loading enabled. Full startup impact applies on next launch."
                )
            )

    def _set_display_splashscreen(self, enabled: Any):
        """Persist startup splashscreen visibility preference."""
        normalized = _normalize_bool_setting(enabled, True)
        if bool(self.display_splashscreen_var.get()) != normalized:
            self.display_splashscreen_var.set(normalized)
        if normalized == self.display_splashscreen:
            return
        self.display_splashscreen = normalized
        self.settings_manager.set("display_splashscreen", normalized)

    def _set_categories_start_collapsed(self, enabled: Any):
        """Persist whether function categories should start collapsed."""
        normalized = _normalize_bool_setting(enabled, False)
        if bool(self.categories_start_collapsed_var.get()) != normalized:
            self.categories_start_collapsed_var.set(normalized)
        if normalized == self.categories_start_collapsed:
            return
        self.categories_start_collapsed = normalized
        self.settings_manager.set("categories_start_collapsed", normalized)

    def _set_graph_font_scale(self, scale: float, notify: bool = True):
        """Persist and apply graph relative font scale preference."""
        normalized = self._normalize_graph_font_scale(scale)
        normalized_str = self._format_graph_font_scale_value(normalized)

        if self.graph_font_scale_var.get() != normalized_str:
            self.graph_font_scale_var.set(normalized_str)

        if abs(normalized - self.graph_font_scale) < 1e-9:
            return

        self.graph_font_scale = normalized
        self.settings_manager.set("graph_font_scale", normalized)

        if notify:
            percent_text = f"{int(round(normalized * 100))}%"
            self._show_fading_message(
                self.language_manager.translate(
                    "ui.messages.graph_font_scale_changed_to",
                    "Graph font size changed to"
                ) + f" {percent_text}."
            )

        if getattr(self, "current_tab", None) == "analysis":
            self._show_analysis_tab()

    def _get_graph_renderer(self):
        """Lazy-load graph renderer to reduce startup import time."""
        if not hasattr(self, "_graph_renderer") or self._graph_renderer is None:
            import graph_renderer
            self._graph_renderer = graph_renderer
        return self._graph_renderer

    def _get_reporting_functions(self):
        """Lazy-load report builders to avoid report stack imports at startup."""
        if not hasattr(self, "_reporting_funcs") or self._reporting_funcs is None:
            from chemometrics.reporting import build_latex_document, compile_latex_to_pdf
            self._reporting_funcs = (build_latex_document, compile_latex_to_pdf)
        return self._reporting_funcs

    def _get_add_graph_dialog_func(self):
        """Lazy-load add-graph dialog callable."""
        if not hasattr(self, "_add_graph_dialog_fn") or self._add_graph_dialog_fn is None:
            from add_graph_dialog import show_add_graph_dialog
            self._add_graph_dialog_fn = show_add_graph_dialog
        return self._add_graph_dialog_fn

    def _get_add_table_dialog_func(self):
        """Lazy-load add-table dialog callable."""
        if not hasattr(self, "_add_table_dialog_fn") or self._add_table_dialog_fn is None:
            from add_table_dialog import show_add_table_dialog
            self._add_table_dialog_fn = show_add_table_dialog
        return self._add_table_dialog_fn

    def _get_add_text_dialog_func(self):
        """Lazy-load add-text dialog callable."""
        if not hasattr(self, "_add_text_dialog_fn") or self._add_text_dialog_fn is None:
            from add_text_dialog import show_add_text_dialog
            self._add_text_dialog_fn = show_add_text_dialog
        return self._add_text_dialog_fn

    def _get_routing_map_window_class(self):
        """Lazy-load routing map window class."""
        if not hasattr(self, "_routing_map_window_cls") or self._routing_map_window_cls is None:
            from routing_map_window import RoutingMapWindow
            self._routing_map_window_cls = RoutingMapWindow
        return self._routing_map_window_cls

    def _get_matplotlib_pyplot(self):
        """Lazy-load matplotlib pyplot for figure cleanup/export helpers."""
        if not hasattr(self, "_matplotlib_pyplot") or self._matplotlib_pyplot is None:
            from matplotlib import pyplot as plt
            self._matplotlib_pyplot = plt
        return self._matplotlib_pyplot

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

        if not self._notification_root_bind_set:
            self.root.bind("<Configure>", self._on_root_configure_notifications, add="+")
            self._notification_root_bind_set = True

        toast = tk.Toplevel(self.root)
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

        self.notification_toasts.append(toast)
        self._reposition_notification_toasts()

        fade_steps = 12
        fade_step_ms = 85

        def fade_out(step=fade_steps):
            if not toast.winfo_exists():
                return
            if step <= 0:
                self._remove_notification_toast(toast)
                return
            try:
                toast.attributes("-alpha", max(0.0, step / fade_steps))
            except tk.TclError:
                if step <= 1:
                    self._remove_notification_toast(toast)
                return
            toast.after(fade_step_ms, lambda: fade_out(step - 1))

        toast.after(max(400, duration_ms), fade_out)

    def _on_root_configure_notifications(self, event=None):
        """Keep toast stack anchored to the root window during moves/resizes."""
        self._reposition_notification_toasts()

    def _remove_notification_toast(self, toast: tk.Toplevel):
        """Remove toast from stack and reflow remaining notifications."""
        try:
            if toast in self.notification_toasts:
                self.notification_toasts.remove(toast)
            if toast.winfo_exists():
                toast.destroy()
        except tk.TclError:
            pass
        self._reposition_notification_toasts()

    def _reposition_notification_toasts(self):
        """Layout all active toast notifications in a vertical stack."""
        active_toasts: List[tk.Toplevel] = []
        for toast in self.notification_toasts:
            try:
                if toast.winfo_exists():
                    active_toasts.append(toast)
            except tk.TclError:
                continue
        self.notification_toasts = active_toasts

        if not self.notification_toasts:
            return

        self.root.update_idletasks()
        y = self.root.winfo_rooty() + self.notification_stack_margin_top
        for toast in self.notification_toasts:
            try:
                toast.update_idletasks()
                x = self.root.winfo_rootx() + self.root.winfo_width() - toast.winfo_reqwidth() - self.notification_stack_margin_right
                toast.geometry(f"+{max(0, x)}+{max(0, y)}")
                y += toast.winfo_reqheight() + self.notification_stack_spacing
            except tk.TclError:
                continue

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

    def _refresh_function_specs(self):
        """Reload merged function specs from core + available add-ons."""
        global FUNCTION_SPECS, ADDON_REGISTRY
        payload = load_combined_function_specs(
            Path(__file__).parent,
            language=self.language_manager.get_language()
        )
        FUNCTION_SPECS = payload.get("specs", {})
        ADDON_REGISTRY = payload.get("addon_registry", {})
        self.addon_registry = ADDON_REGISTRY

        for warning in self.addon_registry.get("warnings", []):
            print(f"Add-on warning: {warning}")
    
    def _load_gui_configs(self):
        """Load function-specific GUI configuration files with language support."""
        self._refresh_function_specs()
        gui_listing = FUNCTION_SPECS.get("gui_listing", {})
        current_language = get_language_manager().get_language()
        function_to_addon = self.addon_registry.get("function_to_addon", {})
        self.gui_configs = {}
        
        for func_alias, func_info in gui_listing.items():
            config_file = func_info.get("config_path")
            if config_file:
                # Parse the config file path
                config_path = Path(config_file)
                config_name = config_path.name

                if config_path.is_absolute() and config_path.exists():
                    try:
                        with open(config_path, encoding='utf-8-sig') as f:
                            self.gui_configs[func_alias] = json.load(f)
                        addon_id = function_to_addon.get(func_alias)
                        if addon_id:
                            original_category = str(self.gui_configs[func_alias].get("category", "")).strip()
                            addon_category = f"Add-ons\\{addon_id}"
                            self.gui_configs[func_alias]["category"] = (
                                f"{addon_category}\\{original_category}" if original_category else addon_category
                            )
                        continue
                    except json.JSONDecodeError as e:
                        print(f"ERROR: Invalid JSON in {config_path}: {e}")
                        raise
                
                # Try language-specific folder first (gui_configs/[language]/[config_name])
                lang_folder = Path(__file__).parent / "gui_configs" / current_language
                lang_config_path = lang_folder / config_name
                
                if lang_config_path.exists():
                    try:
                        with open(lang_config_path, encoding='utf-8-sig') as f:
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
                        with open(en_config_path, encoding='utf-8-sig') as f:
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

            addon_id = function_to_addon.get(func_alias)
            if addon_id and func_alias in self.gui_configs:
                original_category = str(self.gui_configs[func_alias].get("category", "")).strip()
                addon_category = f"Add-ons\\{addon_id}"
                self.gui_configs[func_alias]["category"] = (
                    f"{addon_category}\\{original_category}" if original_category else addon_category
                )
    
    def _load_theme(self):
        """Load and apply persisted UI theme preference."""
        self._apply_selected_theme(self.selected_theme, persist=False, notify=False)

    def _normalize_sun_valley_fonts(self):
        """Scale sv_ttk named fonts back to the app's original size and family."""
        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            base_size = int(default_font.actual("size"))
            if base_size == 0:
                base_size = 10
            base_family = str(default_font.actual("family") or _get_ui_font_family())

            font_specs = {
                "SunValleyCaptionFont": {"size": base_size, "weight": "normal"},
                "SunValleyBodyFont": {"size": base_size, "weight": "normal"},
                "SunValleyBodyStrongFont": {"size": base_size, "weight": "bold"},
                "SunValleyBodyLargeFont": {"size": base_size + 2, "weight": "normal"},
                "SunValleySubtitleFont": {"size": base_size + 4, "weight": "bold"},
                "SunValleyTitleFont": {"size": base_size + 7, "weight": "bold"},
                "SunValleyTitleLargeFont": {"size": base_size + 11, "weight": "bold"},
                "SunValleyDisplayFont": {"size": base_size + 18, "weight": "bold"},
            }

            for font_name, spec in font_specs.items():
                try:
                    named_font = tkfont.nametofont(font_name)
                    named_font.configure(
                        family=base_family,
                        size=spec["size"],
                        weight=spec["weight"],
                        slant="roman",
                    )
                except tk.TclError:
                    continue
        except Exception as e:
            print(f"Could not normalize sv_ttk fonts: {e}")
    
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
        self.tools_menu.add_command(
            label=self.language_manager.translate("menu.execution_report", "Execution Report"),
            command=self._show_execution_report_from_menu,
            state=tk.DISABLED
        )
        self.execution_report_menu_index = self.tools_menu.index("end")
        self._set_execution_report_menu_state(self.latest_execution_report is not None)
        self.tools_menu.add_command(
            label=self.language_manager.translate("menu.model_log", "Model Log"),
            command=self._show_model_log_popup,
            state=tk.DISABLED
        )
        self.model_log_menu_index = self.tools_menu.index("end")
        self._set_model_log_menu_state(self.latest_timing_report is not None)

        
        # Settings Menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.language_manager.translate("menu.settings", "Settings"), menu=settings_menu)
        
        # Language submenu
        lang_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label=self.language_manager.translate("menu.language", "Language"), menu=lang_menu)
        
        for lang_code, lang_name in self.language_manager.SUPPORTED_LANGUAGES.items():
            lang_menu.add_radiobutton(
                label=lang_name,
                variable=self.language_var,
                value=lang_code,
                command=lambda code=lang_code: self._change_language(code)
            )

        # Theme submenu
        theme_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label=self.language_manager.translate("menu.theme", "Theme"), menu=theme_menu)
        for theme_id, theme_label in self._get_available_theme_options():
            theme_menu.add_radiobutton(
                label=theme_label,
                variable=self.theme_var,
                value=theme_id,
                command=lambda tid=theme_id: self._set_theme(tid)
            )
        
        # Colormap submenu
        colormap_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label=self.language_manager.translate("menu.colormap", "Colormap"), menu=colormap_menu)

        startup_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(
            label=self.language_manager.translate("menu.startup", "Startup"),
            menu=startup_menu
        )

        # Import loading mode submenu
        loading_menu = tk.Menu(settings_menu, tearoff=0)
        startup_menu.add_cascade(
            label=self.language_manager.translate("menu.import_loading_mode", "Import Loading"),
            menu=loading_menu
        )
        loading_menu.add_radiobutton(
            label=self.language_manager.translate("menu.import_loading_lazy", "Lazy (faster startup)"),
            variable=self.import_loading_mode_var,
            value="lazy",
            command=lambda: self._set_import_loading_mode("lazy")
        )
        loading_menu.add_radiobutton(
            label=self.language_manager.translate("menu.import_loading_eager", "Eager (load all at startup)"),
            variable=self.import_loading_mode_var,
            value="eager",
            command=lambda: self._set_import_loading_mode("eager")
        )
        startup_menu.add_separator()
        startup_menu.add_checkbutton(
            label=self.language_manager.translate("menu.display_splashscreen", "Display Splashscreen"),
            variable=self.display_splashscreen_var,
            onvalue=True,
            offvalue=False,
            command=lambda: self._set_display_splashscreen(self.display_splashscreen_var.get())
        )
        startup_menu.add_checkbutton(
            label=self.language_manager.translate("menu.categories_start_collapsed", "Start Categories Collapsed"),
            variable=self.categories_start_collapsed_var,
            onvalue=True,
            offvalue=False,
            command=lambda: self._set_categories_start_collapsed(self.categories_start_collapsed_var.get())
        )

        graph_font_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(
            label=self.language_manager.translate("menu.graph_font_size", "Graph Font Size"),
            menu=graph_font_menu
        )
        self._populate_graph_font_scale_menu(
            graph_font_menu,
            self.graph_font_scale_var,
            lambda scale: self._set_graph_font_scale(scale)
        )
        
        # Load available colormaps
        colormaps_data = self._load_colormaps_catalog()
        
        # Continuous colormaps submenu
        continuous_menu = tk.Menu(colormap_menu, tearoff=0)
        colormap_menu.add_cascade(label=self.language_manager.translate("menu.colormap_continuous", "Continuous"), menu=continuous_menu)
        
        # Handle both old flat structure and new nested structure for continuous colormaps
        continuous_data = colormaps_data.get("continuous", {})
        if isinstance(continuous_data, list):
            # Old flat structure - treat as a single category
            for cmap in continuous_data:
                continuous_menu.add_radiobutton(
                    label=cmap,
                    variable=self.colormap_var,
                    value=cmap,
                    command=lambda cm=cmap: self._change_colormap(cm)
                )
        else:
            # New nested structure with subcategories
            for category, cmaps in continuous_data.items():
                category_menu = tk.Menu(continuous_menu, tearoff=0)
                continuous_menu.add_cascade(label=category, menu=category_menu)
                for cmap in cmaps:
                    category_menu.add_radiobutton(
                        label=cmap,
                        variable=self.colormap_var,
                        value=cmap,
                        command=lambda cm=cmap: self._change_colormap(cm)
                    )
        
        # Qualitative colormaps submenu
        qualitative_menu = tk.Menu(colormap_menu, tearoff=0)
        colormap_menu.add_cascade(label=self.language_manager.translate("menu.colormap_qualitative", "Qualitative"), menu=qualitative_menu)
        for cmap in colormaps_data.get("qualitative", []):
            qualitative_menu.add_radiobutton(
                label=cmap,
                variable=self.qualitative_colormap_var,
                value=cmap,
                command=lambda cm=cmap: self._change_qualitative_colormap(cm)
            )
        
        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.language_manager.translate("menu.help", "Help"), menu=help_menu)
        help_menu.add_command(
            label=self.language_manager.translate("menu.user_manual", "User Manual"),
            command=self._open_user_manual,
        )
        help_menu.add_separator()
        help_menu.add_command(
            label=self.language_manager.translate("menu.project_license", "Project License (Apache-2.0)"),
            command=lambda: self._show_license_file_popup(
                self.language_manager.translate("licenses.project_license", "Project License (Apache-2.0)"),
                PROJECT_LICENSE_PATH,
            ),
        )
        help_menu.add_command(
            label=self.language_manager.translate("menu.eula", "End-User License Agreement"),
            command=lambda: self._show_license_file_popup(
                self.language_manager.translate("licenses.eula", "End-User License Agreement"),
                EULA_PATH,
            ),
        )
        help_menu.add_command(label=self.language_manager.translate("menu.licenses", "Third-Party Licenses"), command=self._show_licenses_dialog)
        help_menu.add_command(label=self.language_manager.translate("menu.acknowledgements", "Acknowledgements"), command=self._show_acknowledgements_dialog)
        help_menu.add_separator()
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

    def _set_execution_report_menu_state(self, enabled: bool):
        """Enable or disable the Execution Report menu item."""
        if self.tools_menu is None or self.execution_report_menu_index is None:
            return
        state = tk.NORMAL if enabled else tk.DISABLED
        try:
            self.tools_menu.entryconfig(self.execution_report_menu_index, state=state)
        except tk.TclError:
            pass

    def _set_model_log_menu_state(self, enabled: bool):
        """Enable or disable the Model Log menu item."""
        if self.tools_menu is None or self.model_log_menu_index is None:
            return
        state = tk.NORMAL if enabled else tk.DISABLED
        try:
            self.tools_menu.entryconfig(self.model_log_menu_index, state=state)
        except tk.TclError:
            pass

    def _store_timing_report(self, run_type_label: str, timing_report: Optional[Dict[str, Any]], stop_at_function_alias: Optional[str] = None):
        """Persist latest timing report and enable menu access."""
        if not timing_report:
            self.latest_timing_report = None
            self._set_timing_report_menu_state(False)
            self._set_model_log_menu_state(False)
            return

        self.latest_timing_report = {
            'run_type': run_type_label,
            'stop_at_function_alias': stop_at_function_alias,
            'total_execution_time': timing_report.get('total_execution_time', 0.0),
            'lazy_loading_time': timing_report.get('lazy_loading_time', 0.0),
            'pipeline_execution_time': timing_report.get('pipeline_execution_time', 0.0),
            'function_timings': timing_report.get('function_timings', []),
            'executed_function_count': timing_report.get('executed_function_count', 0),
            'partial_run': timing_report.get('partial_run', False),
            'timestamp': datetime.now().isoformat()
        }
        self._set_timing_report_menu_state(True)
        self._set_model_log_menu_state(True)

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
        _set_window_icon(report_win, "Icon")
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
            f"{self.language_manager.translate('ui.timing_report.lazy_loading_time', 'Library loading stage')}: {self._format_execution_seconds(report.get('lazy_loading_time', 0.0))}",
            f"{self.language_manager.translate('ui.timing_report.pipeline_execution_time', 'Pipeline execution stage')}: {self._format_execution_seconds(report.get('pipeline_execution_time', 0.0))}",
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

    def _show_execution_report_from_menu(self) -> None:
        """Show execution report popup from the Tools menu."""
        report = self.latest_execution_report
        if report is None:
            self._show_fading_warning(
                self.language_manager.translate("ui.messages.no_execution_report", "No execution report available. Run the model first.")
            )
            return
        if not self._has_execution_report_entries(report):
            report_win = tk.Toplevel(self.root)
            _set_window_icon(report_win, "Icon")
            report_win.title(self.language_manager.translate("ui.dialogs.execution_report", "Execution Report"))
            report_win.geometry("480x200")
            report_win.transient(self.root)
            report_win.grab_set()
            container = ttk.Frame(report_win, padding=16)
            container.pack(fill=tk.BOTH, expand=True)
            msg = ttk.Label(
                container,
                text=self.language_manager.translate(
                    "ui.execution_report.no_entries",
                    "No messages, warnings, or errors were generated during the last run."
                ),
                wraplength=420,
                justify="left",
            )
            msg.pack(anchor="w", pady=(8, 16))
            close_btn = ttk.Button(container, text=self.language_manager.translate("ui.buttons.close", "Close"), command=report_win.destroy)
            close_btn.pack(anchor="e")
            return
        run_type_label = (self.latest_timing_report or {}).get('run_type', '')
        self._show_execution_report_popup(report, run_type_label=run_type_label)

    def _show_model_log_popup(self) -> None:
        """Show the latest model run log in a popup window."""
        log_path = _get_runtime_model_log_path()
        try:
            if not log_path.exists():
                self._show_fading_warning(
                    self.language_manager.translate("ui.messages.no_model_log", "No model log available. Run the model first.")
                )
                return
            log_text = log_path.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.model_log_read_error", f"Could not read model log: {e}")
            )
            return

        report_win = tk.Toplevel(self.root)
        _set_window_icon(report_win, "Icon")
        report_win.title(self.language_manager.translate("ui.dialogs.model_log", "Model Log"))
        report_win.geometry("700x520")
        report_win.transient(self.root)
        report_win.grab_set()

        container = ttk.Frame(report_win, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        text_frame = ttk.Frame(container)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        log_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 10),
            padx=8,
            pady=8,
        )
        log_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=log_widget.yview)

        log_widget.insert(tk.END, log_text)
        log_widget.config(state=tk.DISABLED)

        close_btn = ttk.Button(
            container,
            text=self.language_manager.translate("ui.buttons.close", "Close"),
            command=report_win.destroy,
        )
        close_btn.pack(anchor="e", pady=(10, 0))

    def _has_execution_report_entries(self, execution_report: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(execution_report, dict):
            return False
        entries = execution_report.get('entries', [])
        return isinstance(entries, list) and len(entries) > 0

    def _store_execution_report(self, execution_report: Optional[Dict[str, Any]]) -> None:
        if not isinstance(execution_report, dict):
            self.latest_execution_report = None
            self._set_execution_report_menu_state(False)
            return
        self.latest_execution_report = copy.deepcopy(execution_report)
        self._set_execution_report_menu_state(True)
        if self._has_execution_report_entries(execution_report):
            self._highlight_methodology_from_execution_report(self.latest_execution_report)

    def _normalize_methodology_highlight_level(self, level: Any) -> Optional[str]:
        normalized = str(level or "").strip().lower()
        return normalized if normalized in {"warning", "error"} else None

    def _methodology_highlight_rank(self, level: Any) -> int:
        normalized = self._normalize_methodology_highlight_level(level)
        return {
            "warning": 1,
            "error": 2,
        }.get(normalized, 0)

    def _prune_methodology_highlights(self) -> None:
        valid_aliases = set(self.methodology_list)
        stale_aliases = [alias for alias in self.methodology_row_highlights if alias not in valid_aliases]
        for alias in stale_aliases:
            self.methodology_row_highlights.pop(alias, None)

    def _get_effective_methodology_highlight_level(self, instance_alias: str) -> Optional[str]:
        source_map = self.methodology_row_highlights.get(instance_alias, {})
        best_level = None
        best_rank = 0
        for entry in source_map.values():
            level = self._normalize_methodology_highlight_level(entry.get("level") if isinstance(entry, dict) else None)
            rank = self._methodology_highlight_rank(level)
            if rank > best_rank:
                best_rank = rank
                best_level = level
        return best_level

    def _apply_methodology_row_highlights(self) -> None:
        if not hasattr(self, "methodology_listbox"):
            return

        self._prune_methodology_highlights()

        for idx, instance_alias in enumerate(self.methodology_list):
            level = self._get_effective_methodology_highlight_level(instance_alias)
            try:
                if level:
                    colors = self.methodology_highlight_palette.get(level, {})
                    self.methodology_listbox.itemconfig(
                        idx,
                        background=colors.get("bg", ""),
                        foreground=colors.get("fg", ""),
                    )
                else:
                    colors = self._get_methodology_theme_colors()
                    self.methodology_listbox.itemconfig(
                        idx,
                        background=colors["bg"],
                        foreground=colors["fg"],
                    )
            except tk.TclError:
                # Item styling can fail during transient list rebuilds.
                continue

    def _highlight_methodology_function(
        self,
        instance_alias: str,
        level: str = "warning",
        source: str = "generic",
        dismiss_on_click: bool = False,
    ) -> bool:
        normalized_level = self._normalize_methodology_highlight_level(level)
        if not normalized_level:
            return False
        if instance_alias not in self.methodology_list:
            return False

        source_key = str(source or "generic")
        source_map = self.methodology_row_highlights.setdefault(instance_alias, {})
        existing = source_map.get(source_key, {})
        existing_level = self._normalize_methodology_highlight_level(existing.get("level") if isinstance(existing, dict) else None)

        if self._methodology_highlight_rank(existing_level) > self._methodology_highlight_rank(normalized_level):
            normalized_level = existing_level  # Keep strongest level for each source.

        source_map[source_key] = {
            "level": normalized_level,
            "dismiss_on_click": bool(dismiss_on_click),
        }
        self._apply_methodology_row_highlights()
        return True

    def _highlight_methodology_functions(
        self,
        instance_aliases: Optional[List[str]] = None,
        function_names: Optional[List[str]] = None,
        level: str = "warning",
        source: str = "generic",
        dismiss_on_click: bool = False,
        clear_source_first: bool = False,
        function_name_occurrences: Optional[Dict[str, List[int]]] = None,
    ) -> int:
        """Highlight multiple methodology function rows.

        Returns the number of aliases that were successfully highlighted.
        This helper is intentionally generic so future features (for example,
        model-load annotations) can reuse the same source-tagged mechanism.

        Args:
            instance_aliases: Explicit instance aliases (e.g., ["pca_analysis#2"]).
            function_names: Base function aliases to target (e.g., ["pca_analysis"]).
            function_name_occurrences: Optional 1-based occurrence filters for
                base aliases, e.g. {"pca_analysis": [2, 3]}.
        """
        source_key = str(source or "generic")
        occurrence_filter_map: Dict[str, set] = {}

        if isinstance(function_name_occurrences, dict):
            for key, values in function_name_occurrences.items():
                base_key = str(key or "")
                if not base_key:
                    continue
                allowed = set()
                if isinstance(values, list):
                    for raw_value in values:
                        try:
                            parsed = int(raw_value)
                        except (TypeError, ValueError):
                            continue
                        if parsed > 0:
                            allowed.add(parsed)
                occurrence_filter_map[base_key] = allowed

        if clear_source_first:
            for alias in list(self.methodology_row_highlights.keys()):
                source_map = self.methodology_row_highlights.get(alias, {})
                if source_key in source_map:
                    source_map.pop(source_key, None)
                    if source_map:
                        self.methodology_row_highlights[alias] = source_map
                    else:
                        self.methodology_row_highlights.pop(alias, None)

        target_aliases: List[str] = []

        for alias in instance_aliases or []:
            instance_alias = str(alias or "")
            if not instance_alias:
                continue
            if instance_alias in self.methodology_list:
                target_aliases.append(instance_alias)

        for base_name in function_names or []:
            normalized_base_name = str(base_name or "")
            if not normalized_base_name:
                continue

            allowed_occurrences = occurrence_filter_map.get(normalized_base_name, set())
            occurrence_idx = 0
            for idx, base_alias in enumerate(self.function_base_aliases):
                if base_alias != normalized_base_name:
                    continue
                occurrence_idx += 1
                if allowed_occurrences and occurrence_idx not in allowed_occurrences:
                    continue
                if 0 <= idx < len(self.methodology_list):
                    target_aliases.append(self.methodology_list[idx])

        highlighted_count = 0
        seen_aliases = set()
        for alias in target_aliases:
            instance_alias = str(alias or "")
            if not instance_alias or instance_alias in seen_aliases:
                continue
            seen_aliases.add(instance_alias)
            if self._highlight_methodology_function(
                instance_alias=instance_alias,
                level=level,
                source=source_key,
                dismiss_on_click=dismiss_on_click,
            ):
                highlighted_count += 1

        return highlighted_count

    def _clear_methodology_function_highlight(self, instance_alias: str, source: Optional[str] = None) -> None:
        source_map = self.methodology_row_highlights.get(instance_alias)
        if not source_map:
            return

        if source is None:
            self.methodology_row_highlights.pop(instance_alias, None)
        else:
            source_map.pop(str(source), None)
            if not source_map:
                self.methodology_row_highlights.pop(instance_alias, None)

        self._apply_methodology_row_highlights()

    def _clear_methodology_click_dismissable_highlights(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.methodology_list):
            return

        instance_alias = self.methodology_list[idx]
        source_map = self.methodology_row_highlights.get(instance_alias, {})
        removed_any = False

        for source_key, entry in list(source_map.items()):
            if isinstance(entry, dict) and entry.get("dismiss_on_click"):
                source_map.pop(source_key, None)
                removed_any = True

        if source_map:
            self.methodology_row_highlights[instance_alias] = source_map
        else:
            self.methodology_row_highlights.pop(instance_alias, None)

        if removed_any:
            self._apply_methodology_row_highlights()

    def _clear_methodology_highlight_source_until(self, source: str, stop_idx: Optional[int] = None) -> None:
        """Clear a highlight source up to an optional methodology index (inclusive)."""
        source_key = str(source or "")
        if not source_key:
            return

        if not self.methodology_list:
            return

        if stop_idx is None:
            max_idx = len(self.methodology_list) - 1
        else:
            try:
                max_idx = int(stop_idx)
            except (TypeError, ValueError):
                return
            if max_idx < 0:
                return
            max_idx = min(max_idx, len(self.methodology_list) - 1)

        removed_any = False
        for idx in range(0, max_idx + 1):
            instance_alias = self.methodology_list[idx]
            source_map = self.methodology_row_highlights.get(instance_alias, {})
            if source_key in source_map:
                source_map.pop(source_key, None)
                removed_any = True
                if source_map:
                    self.methodology_row_highlights[instance_alias] = source_map
                else:
                    self.methodology_row_highlights.pop(instance_alias, None)

        if removed_any:
            self._apply_methodology_row_highlights()

    def _highlight_methodology_from_execution_report(self, execution_report: Optional[Dict[str, Any]]) -> None:
        if not isinstance(execution_report, dict):
            return

        entries = execution_report.get("entries", [])
        if not isinstance(entries, list):
            return

        # Consolidate entries: keep the strongest level per instance_alias so that
        # every unique instance is processed exactly once.  Processing each raw
        # entry individually causes _apply_methodology_row_highlights to be called
        # after every entry; when the first function has many entries its
        # intermediate redraws can leave later occurrences of the same function
        # type without a highlight.
        best_levels: Dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            level = self._normalize_methodology_highlight_level(entry.get("level"))
            if not level:
                continue
            instance_alias = str(entry.get("instance_alias", "") or "")
            if not instance_alias:
                continue
            existing = best_levels.get(instance_alias)
            if self._methodology_highlight_rank(level) > self._methodology_highlight_rank(existing):
                best_levels[instance_alias] = level

        for instance_alias, level in best_levels.items():
            self._highlight_methodology_function(
                instance_alias=instance_alias,
                level=level,
                source="execution_report",
                dismiss_on_click=True,
            )

    def _resolve_execution_report_entry_text(self, entry: Dict[str, Any]) -> str:
        level = str(entry.get('level', 'message') or 'message').lower()
        code = entry.get('code')
        fallback_text = str(entry.get('text', '') or '')
        base_alias = str(entry.get('base_alias', '') or '')

        if code and base_alias:
            config = self.gui_configs.get(base_alias, {})
            specs = config.get('execution_report_specs', {}) if isinstance(config, dict) else {}
            bucket_key = {
                'message': 'messages',
                'warning': 'warnings',
                'error': 'errors',
            }.get(level, 'messages')
            bucket = specs.get(bucket_key, {}) if isinstance(specs, dict) else {}
            mapped_text = bucket.get(code) if isinstance(bucket, dict) else None
            if mapped_text:
                return str(mapped_text)

        return fallback_text

    def _get_instance_display_name(self, instance_alias: str, base_alias: str) -> str:
        config = self.gui_configs.get(base_alias, {})
        config_display_name = config.get("display_name", base_alias)
        if instance_alias in self.methodology_list:
            idx = self.methodology_list.index(instance_alias)
            existing_count = self.function_base_aliases[:idx].count(base_alias)
            if existing_count > 0:
                return f"{config_display_name} #{existing_count + 1}"
        return config_display_name

    def _show_execution_report_popup(self, execution_report: Optional[Dict[str, Any]], run_type_label: str) -> None:
        if not self._has_execution_report_entries(execution_report):
            return

        report_win = tk.Toplevel(self.root)
        _set_window_icon(report_win, "Icon")
        report_win.title(self.language_manager.translate("ui.dialogs.execution_report", "Execution Report"))
        report_win.geometry("700x500")
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

        entries = execution_report.get('entries', [])
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for entry in entries:
            group_key = str(entry.get('instance_alias', '') or '')
            grouped.setdefault(group_key, []).append(entry)

        ordered_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
        for instance_alias in self.methodology_list:
            if instance_alias in grouped:
                ordered_groups.append((instance_alias, grouped.pop(instance_alias)))
        for remaining_alias, remaining_entries in grouped.items():
            ordered_groups.append((remaining_alias, remaining_entries))

        counts = execution_report.get('counts', {}) if isinstance(execution_report, dict) else {}
        message_count = int(counts.get('message', 0) or 0)
        warning_count = int(counts.get('warning', 0) or 0)
        error_count = int(counts.get('error', 0) or 0)

        lines = [
            self.language_manager.translate("ui.execution_report.title", "Execution Report"),
            "",
            f"{self.language_manager.translate('ui.execution_report.run_type', 'Run type')}: {run_type_label}",
            (
                f"{self.language_manager.translate('ui.execution_report.summary', 'Summary')}: "
                f"{self.language_manager.translate('ui.execution_report.messages', 'Messages')}={message_count}, "
                f"{self.language_manager.translate('ui.execution_report.warnings', 'Warnings')}={warning_count}, "
                f"{self.language_manager.translate('ui.execution_report.errors', 'Errors')}={error_count}"
            ),
            "",
        ]

        for instance_alias, func_entries in ordered_groups:
            first_entry = func_entries[0] if func_entries else {}
            base_alias = str(first_entry.get('base_alias', '') or '')
            display_name = self._get_instance_display_name(instance_alias, base_alias) if base_alias else instance_alias
            lines.append(f"{self.language_manager.translate('ui.execution_report.function', 'Function')}: {display_name}")

            for item in func_entries:
                level = str(item.get('level', 'message') or 'message').lower()
                level_label = {
                    'message': self.language_manager.translate('ui.execution_report.messages_singular', 'Message'),
                    'warning': self.language_manager.translate('ui.execution_report.warning_singular', 'Warning'),
                    'error': self.language_manager.translate('ui.execution_report.error_singular', 'Error'),
                }.get(level, self.language_manager.translate('ui.execution_report.messages_singular', 'Message'))

                entry_text = self._resolve_execution_report_entry_text(item)
                if not entry_text:
                    entry_text = str(item.get('text', '') or '')
                lines.append(f"  - [{level_label}] {entry_text}")
            lines.append("")

        report_text.insert(tk.END, "\n".join(lines).rstrip())
        report_text.config(state=tk.DISABLED)

        close_btn = ttk.Button(
            container,
            text=self.language_manager.translate("ui.buttons.close", "Close"),
            command=report_win.destroy
        )
        close_btn.pack(anchor="e", pady=(10, 0))
    
    def _change_language(self, language_code: str):
        """Change the application language and save setting."""
        self.language_var.set(language_code)
        self.language_manager.set_language(language_code)
        self.settings_manager.set("language", language_code)
        if self._pending_language_refresh_id is not None:
            try:
                self.root.after_cancel(self._pending_language_refresh_id)
            except Exception:
                pass

        self._pending_language_refresh_id = self.root.after_idle(self._apply_language_refresh)

    def _apply_language_refresh(self):
        """Apply deferred UI refresh after language change."""
        self._pending_language_refresh_id = None
        try:
            self._refresh_ui_text()
        except tk.TclError:
            # Avoid hard crash if widgets are in a transient state during refresh.
            self.root.after(25, self._refresh_ui_text)
    
    def _change_colormap(self, colormap_name: str):
        """Change the default continuous colormap and save setting."""
        self.colormap_var.set(colormap_name)
        self.settings_manager.set("colormap", colormap_name)
        self._show_fading_message(
            self.language_manager.translate("ui.messages.colormap_changed_to", "Colormap changed to") +
            f" '{colormap_name}'.\n" +
            self.language_manager.translate("ui.messages.colormap_used_new_plots", "This will be used for new plots.")
        )
    
    def _change_qualitative_colormap(self, colormap_name: str):
        """Change the qualitative colormap and save setting."""
        self.qualitative_colormap_var.set(colormap_name)
        self.settings_manager.set("qualitative_colormap", colormap_name)
        self._show_fading_message(
            self.language_manager.translate("ui.messages.qual_colormap_changed_to", "Qualitative colormap changed to") +
            f" '{colormap_name}'.\n" +
            self.language_manager.translate("ui.messages.colormap_used_new_plots", "This will be used for new plots.")
        )
    
    def _show_acknowledgements_dialog(self):
        """Show the Acknowledgements dialog."""
        ack_file = Path(__file__).parent / "acknowledgements.json"
        try:
            with open(ack_file, encoding='utf-8') as f:
                ack_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            ack_data = {}

        current_lang = self.language_manager.get_language()
        lang_info = ack_data.get(current_lang, ack_data.get("en", {}))

        ack_win = tk.Toplevel(self.root)
        _set_window_icon(ack_win, "Icon")
        ack_win.title(lang_info.get("title", "Acknowledgements"))
        ack_win.geometry("550x500")
        ack_win.resizable(False, False)
        ack_win.transient(self.root)
        ack_win.grab_set()

        ack_win.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (ack_win.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (ack_win.winfo_height() // 2)
        ack_win.geometry(f"+{x}+{y}")

        ui_font_family = _get_ui_font_family()

        # Funding agencies logos header (keeps each logo aspect ratio intact).
        logos_frame = ttk.Frame(ack_win)
        logos_frame.pack(fill=tk.X, padx=20, pady=20)

        ack_win._ack_logo_images = []
        logo_files = ["cnpq-logo.png", "cimol-logo.png"]
        max_logo_width = 220
        max_logo_height = 90

        try:
            from PIL import Image, ImageTk
            graphics_dir = Path(__file__).parent / "Graphics"
            logo_photos = []
            for filename in logo_files:
                logo_path = graphics_dir / filename
                if not logo_path.exists():
                    continue

                logo_image = Image.open(logo_path)
                # Trim transparent padding so visual centering matches what users perceive.
                if logo_image.mode in ("RGBA", "LA"):
                    alpha = logo_image.getchannel("A")
                    bbox = alpha.getbbox()
                    if bbox:
                        logo_image = logo_image.crop(bbox)
                src_w, src_h = logo_image.size
                if src_w <= 0 or src_h <= 0:
                    continue

                scale = min(max_logo_width / src_w, max_logo_height / src_h, 1.0)
                target_w = max(1, int(src_w * scale))
                target_h = max(1, int(src_h * scale))
                resized = logo_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
                logo_photo = ImageTk.PhotoImage(resized)
                logo_photos.append(logo_photo)
                ack_win._ack_logo_images.append(logo_photo)

            logo_row = ttk.Frame(logos_frame)
            logo_row.pack(anchor=tk.CENTER)

            if len(logo_photos) == 1:
                logo_label = ttk.Label(logo_row, image=logo_photos[0])
                logo_label.pack()
            else:
                for index, logo_photo in enumerate(logo_photos[:2]):
                    slot = ttk.Frame(logo_row, width=max_logo_width + 30, height=max_logo_height)
                    slot.pack_propagate(False)
                    slot.pack(side=tk.LEFT, padx=(0, 10) if index == 0 else (10, 0))

                    logo_label = ttk.Label(slot, image=logo_photo)
                    logo_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        except Exception as e:
            print(f"Warning: Could not load acknowledgement logos: {e}")

        desc_frame = ttk.LabelFrame(
            ack_win,
            text=self.language_manager.translate("menu.acknowledgements", "Acknowledgements"),
            padding=15,
        )
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        text_widget = tk.Text(
            desc_frame,
            wrap=tk.WORD,
            height=12,
            width=50,
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
        )
        text_widget.pack(fill=tk.BOTH, expand=True)

        description = lang_info.get("description", "")
        text_widget.insert(1.0, description)
        text_widget.config(state=tk.DISABLED)

        link_frame = ttk.Frame(ack_win)
        link_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        website_url = ack_data.get("website", "https://github.com")
        website_label = lang_info.get("website_label", "Visit our website")

        link_label = tk.Label(
            link_frame, text=website_label, fg="blue", cursor="hand2",
            font=(ui_font_family, 9, "underline"),
        )
        link_label.pack()

        def open_ack_link(event=None):
            import webbrowser
            webbrowser.open(website_url)

        link_label.bind("<Button-1>", open_ack_link)

        close_btn = ttk.Button(
            ack_win,
            text=self.language_manager.translate("ui.buttons.close", "Close"),
            command=ack_win.destroy,
        )
        close_btn.pack(pady=(0, 10))

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
        _set_window_icon(about_win, "Icon")
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
            base_dir = Path(__file__).parent
            icon_path = base_dir / "Graphics" / "Icon.png"
            if not icon_path.exists():
                icon_path = base_dir / "Graphics" / "Icon.ico"
            if not icon_path.exists():
                icon_path = base_dir / "Icon.ico"
            if icon_path.exists():
                # Load and resize the icon
                from PIL import Image, ImageTk
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
        
        program_name = about_data.get("program_name", "Chemometric Studio")
        version = _get_application_version("1.0.0")
        ui_font_family = _get_ui_font_family()

        name_label = ttk.Label(text_frame, text=program_name, font=(ui_font_family, 16, "bold"))
        name_label.pack(anchor=tk.W)

        version_label = ttk.Label(text_frame, text=f"Version {version}", font=(ui_font_family, 10))
        version_label.pack(anchor=tk.W)

        version_text = lang_info.get("version_text", "")
        if version_text:
            version_text_label = ttk.Label(text_frame, text=version_text, font=(ui_font_family, 9))
            version_text_label.pack(anchor=tk.W)
        
        # Description frame
        desc_frame = ttk.LabelFrame(about_win, text=self.language_manager.translate("menu.about", "About"), padding=15)
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        # Text widget for description
        text_widget = tk.Text(
            desc_frame,
            wrap=tk.WORD,
            height=12,
            width=50,
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        description = lang_info.get("description", "")
        text_widget.insert(1.0, description)
        text_widget.config(state=tk.DISABLED)  # Make it read-only
        
        # Hyperlink frame at the bottom
        link_frame = ttk.Frame(about_win)
        link_frame.pack(fill=tk.X, padx=20, pady=10)
        
        website_url = about_data.get("website", "https://github.com")
        website_label = lang_info.get("website_label", "Visit our website")
        
        link_label = tk.Label(link_frame, text=website_label, fg="blue", cursor="hand2", font=(ui_font_family, 9, "underline"))
        link_label.pack()
        
        def open_link(event=None):
            import webbrowser
            webbrowser.open(website_url)
        
        link_label.bind("<Button-1>", open_link)
        
        # Close button
        close_btn = ttk.Button(about_win, text=self.language_manager.translate("ui.buttons.close", "Close"), command=about_win.destroy)
        close_btn.pack(pady=(0, 10))

    def _open_user_manual(self):
        """Open the HTML5 user manual in the system default browser."""
        if not MANUAL_INDEX_PATH.exists():
            messagebox.showerror(
                self.language_manager.translate("ui.dialogs.error", "Error"),
                self.language_manager.translate(
                    "ui.messages.manual_missing",
                    "User manual not found. Expected file:"
                ) + f"\n{MANUAL_INDEX_PATH}"
            )
            return

        try:
            webbrowser.open(MANUAL_INDEX_PATH.resolve().as_uri(), new=2)
        except Exception as exc:
            messagebox.showerror(
                self.language_manager.translate("ui.dialogs.error", "Error"),
                self.language_manager.translate(
                    "ui.messages.manual_open_failed",
                    "Could not open the user manual in the default browser:"
                ) + f"\n{MANUAL_INDEX_PATH}\n\n{exc}"
            )

    def _open_path_with_system_default(self, target_path: Path):
        """Open file/folder with system default app."""
        if not target_path.exists():
            messagebox.showerror(
                self.language_manager.translate("ui.dialogs.error", "Error"),
                self.language_manager.translate("ui.messages.license_path_missing", "License file or folder not found:") + f"\n{target_path}"
            )
            return

        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target_path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target_path)])
            else:
                subprocess.Popen(["xdg-open", str(target_path)])
        except Exception as exc:
            messagebox.showerror(
                self.language_manager.translate("ui.dialogs.error", "Error"),
                self.language_manager.translate("ui.messages.license_open_failed", "Could not open license file or folder:") +
                f"\n{target_path}\n\n{exc}"
            )

    def _show_license_file_popup(self, title: str, file_path: Path):
        """Show a license/notice text file inside a scrollable popup."""
        if not file_path.exists():
            messagebox.showerror(
                self.language_manager.translate("ui.dialogs.error", "Error"),
                self.language_manager.translate("ui.messages.license_path_missing", "License file or folder not found:") + f"\n{file_path}"
            )
            return

        try:
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file_path.read_text(encoding="latin-1", errors="replace")
        except Exception as exc:
            messagebox.showerror(
                self.language_manager.translate("ui.dialogs.error", "Error"),
                self.language_manager.translate("ui.messages.license_open_failed", "Could not open license file or folder:") +
                f"\n{file_path}\n\n{exc}"
            )
            return

        viewer = tk.Toplevel(self.root)
        _set_window_icon(viewer, "Icon")
        viewer.title(f"{title}")
        viewer.geometry("840x620")
        viewer.resizable(True, True)
        viewer.transient(self.root)
        viewer.grab_set()

        viewer.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (viewer.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (viewer.winfo_height() // 2)
        viewer.geometry(f"+{x}+{y}")

        container = ttk.Frame(viewer, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        text_frame = ttk.Frame(container)
        text_frame.pack(fill=tk.BOTH, expand=True)

        text_widget = tk.Text(text_frame, wrap=tk.WORD, bg="#f8f8f8", fg="#000000")
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.insert("1.0", content)
        text_widget.configure(state=tk.DISABLED)

        close_btn = ttk.Button(
            container,
            text=self.language_manager.translate("ui.buttons.close", "Close"),
            command=viewer.destroy,
        )
        close_btn.pack(anchor="e", pady=(10, 0))

    def _show_licenses_dialog(self):
        """Show licenses dialog with quick links to third-party notices."""
        licenses_win = tk.Toplevel(self.root)
        _set_window_icon(licenses_win, "Icon")
        licenses_win.title(self.language_manager.translate("licenses.title", "Third-Party Licenses"))
        licenses_win.geometry("680x420")
        licenses_win.resizable(False, False)
        licenses_win.transient(self.root)
        licenses_win.grab_set()

        licenses_win.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (licenses_win.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (licenses_win.winfo_height() // 2)
        licenses_win.geometry(f"+{x}+{y}")

        container = ttk.Frame(licenses_win, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        description = ttk.Label(
            container,
            text=self.language_manager.translate(
                "licenses.description",
                "Open bundled license and third-party notice files."
            ),
            justify=tk.LEFT,
            wraplength=620,
        )
        description.pack(anchor=tk.W, pady=(0, 10))

        files_to_show = [
            (
                self.language_manager.translate("licenses.overview", "Licenses Overview"),
                LICENSES_DIR / "README.md",
            ),
            (
                self.language_manager.translate("licenses.font_ofl", "Selawik OFL-1.1"),
                LICENSES_DIR / "Fonts" / "Selawik" / "OFL-1.1.txt",
            ),
            (
                self.language_manager.translate("licenses.python_notices", "Python Third-Party Notices"),
                LICENSES_DIR / "Python" / "THIRD-PARTY-NOTICES.md",
            ),
            (
                self.language_manager.translate("licenses.other_notices", "Other Third-Party Notices"),
                LICENSES_DIR / "References" / "THIRD-PARTY-NOTICES.md",
            ),
        ]

        files_frame = ttk.Frame(container)
        files_frame.pack(fill=tk.BOTH, expand=True)

        missing_suffix = self.language_manager.translate("licenses.missing", "(missing)")
        for title, path in files_to_show:
            row = ttk.Frame(files_frame)
            row.pack(fill=tk.X, pady=4)

            exists = path.exists()
            display_title = title if exists else f"{title} {missing_suffix}"
            try:
                display_path = str(path.relative_to(BASE_DIR)).replace("\\", "/")
            except ValueError:
                display_path = path.name
            label = ttk.Label(row, text=f"{display_title}: {display_path}", anchor=tk.W)
            label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            open_btn = ttk.Button(
                row,
                text=self.language_manager.translate("licenses.open", "Open"),
                command=lambda t=title, p=path: self._show_license_file_popup(t, p),
            )
            if not exists:
                open_btn.configure(state=tk.DISABLED)
            open_btn.pack(side=tk.RIGHT, padx=(10, 0))

        footer = ttk.Frame(container)
        footer.pack(fill=tk.X, pady=(10, 0))

        open_folder_btn = ttk.Button(
            footer,
            text=self.language_manager.translate("licenses.open_folder", "Open Licenses Folder"),
            command=lambda: self._open_path_with_system_default(LICENSES_DIR),
        )
        open_folder_btn.pack(side=tk.LEFT)

        close_btn = ttk.Button(
            footer,
            text=self.language_manager.translate("ui.buttons.close", "Close"),
            command=licenses_win.destroy,
        )
        close_btn.pack(side=tk.RIGHT)
    

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
        workspace_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))
        self.workspace_frame = workspace_frame

        # Keep progress popup centered over workspace while window/layout changes
        if not self._execution_progress_root_bind_set:
            self.root.bind("<Configure>", self._on_execution_progress_anchor_configure, add="+")
            self._execution_progress_root_bind_set = True
        workspace_frame.bind("<Configure>", self._on_execution_progress_anchor_configure, add="+")
        
        self._build_control_bar(workspace_frame)
        
        # Tab content frame
        self.tab_content_frame = ttk.Frame(workspace_frame)
        self.tab_content_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(10, 3))
        
        # Initialize tabs as empty; will be created on demand
        self.current_tab = None
        self._show_setup_tab()
    
    def _build_functions_panel(self, parent: ttk.Frame):
        """Build collapsible list of available functions grouped by category."""
        panel_bg = self._get_theme_background_color()
        panel_fg = self._get_theme_foreground_color()
        functions_frame = tk.LabelFrame(
            parent,
            text=self.language_manager.translate("ui.panels.functions_by_category", "Functions by Category"),
            height=200,
            width=200,
            bg=panel_bg,
            fg=panel_fg,
            bd=1,
            relief=tk.GROOVE,
            labelanchor="n",
        )
        functions_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        functions_frame.pack_propagate(False)
        
        # Pack scrollbar first, then canvas so scrollbar is visible
        scrollbar = ttk.Scrollbar(functions_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas = tk.Canvas(functions_frame, highlightthickness=0, bg=panel_bg, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)
        
        scrollable_frame = ttk.Frame(canvas)
        
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
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mousewheel to canvas for scrolling
        def _on_mousewheel(event):
            step = int(-1 * (event.delta / 120))
            if step == 0:
                return

            first, last = canvas.yview()
            if (step < 0 and first <= 0.0) or (step > 0 and last >= 1.0):
                return

            canvas.yview_scroll(step, "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
        
        gui_listing = FUNCTION_SPECS.get("gui_listing", {})
        category_tree: Dict[str, Any] = {"children": {}, "functions": []}

        hidden_library_aliases = {
            "workflow_loop_end",
            "workflow_parallel_end",
            "workflow_ensemble_end",
        }

        for func_alias in gui_listing.keys():
            if func_alias in hidden_library_aliases:
                continue
            config = self.gui_configs.get(func_alias, {})
            parsed_path = self._parse_category_path(config.get("category", "Uncategorized"))

            node = category_tree
            for segment in parsed_path:
                node = node["children"].setdefault(segment, {"children": {}, "functions": []})
            node["functions"].append((func_alias, config))

        for category_key in sorted(category_tree["children"].keys(), key=self._category_sort_key):
            self._add_collapsible_category(
                scrollable_frame,
                category_key[1],
                category_tree["children"][category_key],
                canvas,
                depth=0
            )

    def _parse_category_segment(self, segment: Any) -> Tuple[Optional[int], str]:
        """Parse optional numeric ordering prefix from a category segment."""
        raw_segment = str(segment).strip() if segment is not None else ""
        if not raw_segment:
            return None, "Uncategorized"

        match = re.match(r"^(\d+)\.\s*(.+)$", raw_segment)
        if match:
            return int(match.group(1)), match.group(2).strip()
        return None, raw_segment

    def _parse_category_path(self, category_value: Any) -> List[Tuple[Optional[int], str]]:
        """Split category path by backslashes and parse ordering prefixes."""
        raw_category = str(category_value).strip() if category_value is not None else ""
        if not raw_category:
            raw_category = "Uncategorized"

        raw_segments = [part.strip() for part in raw_category.split("\\") if part.strip()]
        if not raw_segments:
            raw_segments = ["Uncategorized"]

        return [self._parse_category_segment(part) for part in raw_segments]

    def _category_sort_key(self, category_key: Tuple[Optional[int], str]) -> Tuple[int, int, str]:
        """Sort indexed categories by number, then regular categories alphabetically."""
        order_index, label = category_key
        normalized_label = label.lower()
        if order_index is None:
            return 1, 0, normalized_label
        return 0, order_index, normalized_label

    def _add_collapsible_category(self, parent: ttk.Frame, category: str, category_node: Dict[str, Any], canvas: tk.Canvas, depth: int = 0):
        """Create nested collapsible category with function buttons."""
        category_frame = ttk.Frame(parent)
        left_indent = 5 + depth * 6
        category_frame.pack(fill=tk.X, padx=(left_indent, 0), pady=3)
        
        # Mousewheel binding helper
        def bind_mousewheel(widget):
            def _on_mousewheel(event):
                step = int(-1 * (event.delta / 120))
                if step == 0:
                    return

                first, last = canvas.yview()
                if (step < 0 and first <= 0.0) or (step > 0 and last >= 1.0):
                    return

                canvas.yview_scroll(step, "units")
            widget.bind("<MouseWheel>", _on_mousewheel)
        
        # Bind to category frame so scrolling works on empty space
        bind_mousewheel(category_frame)
        
        button_frame = ttk.Frame(category_frame)
        button_frame.pack(fill=tk.X)
        bind_mousewheel(button_frame)
        
        collapsed = tk.BooleanVar(value=self.categories_start_collapsed)

        functions_container = ttk.Frame(category_frame)
        bind_mousewheel(functions_container)

        def apply_state():
            if collapsed.get():
                functions_container.pack_forget()
            else:
                functions_container.pack(fill=tk.X, padx=(6, 0), pady=5)
            toggle_btn.configure(text=f"{_ui_symbol('collapsed') if collapsed.get() else _ui_symbol('expanded')} {category}")
        
        def toggle():
            collapsed.set(not collapsed.get())
            apply_state()
        
        toggle_button_width = max(18, 30 - depth * 3)
        toggle_btn = ttk.Button(button_frame, command=toggle, width=toggle_button_width, style="FunctionsPanel.TButton")
        toggle_btn.pack(fill=tk.X)
        bind_mousewheel(toggle_btn)

        for child_key in sorted(category_node.get("children", {}).keys(), key=self._category_sort_key):
            self._add_collapsible_category(
                functions_container,
                child_key[1],
                category_node["children"][child_key],
                canvas,
                depth=depth + 1
            )

        for func_alias, func_info in category_node.get("functions", []):
            display_name = func_info.get("display_name", func_alias)
            function_button_width = max(14, 25 - depth * 3)
            func_btn = ttk.Button(
                functions_container,
                text=display_name,
                command=lambda alias=func_alias: self._add_to_methodology(alias),
                width=function_button_width,
                style="FunctionsPanel.TButton"
            )
            func_btn.pack(fill=tk.X, pady=2)
            bind_mousewheel(func_btn)

        apply_state()
    
    def _add_to_methodology(self, func_alias: str):
        """Add function to methodology list (with duplicate handling using function aliasing)."""
        wrapper_templates = {
            "workflow_loop_start": ["workflow_loop_start", "workflow_loop_end"],
            "workflow_loop_end": ["workflow_loop_start", "workflow_loop_end"],
            "workflow_parallel_start": ["workflow_parallel_start", "workflow_parallel_branch", "workflow_parallel_end"],
            "workflow_parallel_end": ["workflow_parallel_start", "workflow_parallel_branch", "workflow_parallel_end"],
            "workflow_ensemble_start": ["workflow_ensemble_start", "workflow_ensemble_member", "workflow_ensemble_end"],
            "workflow_ensemble_end": ["workflow_ensemble_start", "workflow_ensemble_member", "workflow_ensemble_end"],
        }

        aliases_to_add = wrapper_templates.get(func_alias, [func_alias])
        first_added_idx: Optional[int] = None

        for alias in aliases_to_add:
            new_func_idx = self._append_methodology_item(alias)
            if first_added_idx is None:
                first_added_idx = new_func_idx

            # Auto-create routing for inputs that match previous outputs
            self._auto_create_routing(new_func_idx, alias)

        if first_added_idx is not None:
            self._refresh_methodology_listbox(selected_idx=first_added_idx)

    def _append_methodology_item(self, func_alias: str) -> int:
        """Append one methodology item and return its new index."""
        existing_count = self.function_base_aliases.count(func_alias)

        if existing_count > 0:
            instance_alias = f"{func_alias}#{existing_count + 1}"
        else:
            instance_alias = func_alias

        self.methodology_list.append(instance_alias)
        self.function_base_aliases.append(func_alias)
        self.function_configs[instance_alias] = {}
        return len(self.methodology_list) - 1

    def _is_workflow_control(self, base_alias: str) -> bool:
        return base_alias in self.workflow_control_aliases

    def _get_workflow_scope_signature(self, target_idx: int) -> Tuple:
        """Return active workflow scope signature before target index.

        Signature includes loop/parallel/ensemble nesting and current branch/member so auto-routing
        can be constrained to the same branch context.
        """
        return _svc_get_workflow_scope_signature(self.function_base_aliases, target_idx)

    def _can_auto_route_between(self, src_idx: int, dst_idx: int) -> bool:
        return _svc_can_auto_route_between(
            self.function_base_aliases,
            self.workflow_control_aliases,
            src_idx,
            dst_idx,
        )

    def _get_base_output_spec_keys(self, func_alias: str) -> List[str]:
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

    def _get_passforward_config(self, func_alias: str) -> Dict[str, Any]:
        """Return normalized passforward config for a function, or empty dict when unavailable."""
        return _svc_get_passforward_config(
            self.gui_configs,
            func_alias,
            normalize_bool_setting=_normalize_bool_setting,
        )

    def _is_passforward_compatible(self, func_alias: str) -> bool:
        return _svc_is_passforward_compatible(
            self.gui_configs,
            func_alias,
            normalize_bool_setting=_normalize_bool_setting,
        )

    def _is_passforward_enabled(self, instance_alias: str, base_alias: Optional[str] = None) -> bool:
        return _svc_is_passforward_enabled(
            instance_alias=instance_alias,
            base_alias=base_alias,
            methodology_list=self.methodology_list,
            function_base_aliases=self.function_base_aliases,
            function_configs=self.function_configs,
            gui_configs=self.gui_configs,
            normalize_bool_setting=_normalize_bool_setting,
        )

    def _get_passforward_output_aliases(self, instance_alias: str, func_alias: str) -> Dict[str, str]:
        return _svc_get_passforward_output_aliases(
            instance_alias=instance_alias,
            func_alias=func_alias,
            methodology_list=self.methodology_list,
            function_base_aliases=self.function_base_aliases,
            function_configs=self.function_configs,
            gui_configs=self.gui_configs,
            normalize_bool_setting=_normalize_bool_setting,
        )

    def _get_output_spec_keys(self, func_alias: str, instance_alias: Optional[str] = None) -> List[str]:
        base_keys = self._get_base_output_spec_keys(func_alias)
        keys = list(base_keys)
        if instance_alias:
            passforward_aliases = self._get_passforward_output_aliases(instance_alias, func_alias)
            for key in passforward_aliases.keys():
                if key not in keys:
                    keys.append(key)
        return keys

    def _get_output_aliases_for_instance(self, func_idx: int) -> Dict[str, str]:
        if func_idx < 0 or func_idx >= len(self.methodology_list) or func_idx >= len(self.function_base_aliases):
            return {}
        instance_alias = self.methodology_list[func_idx]
        base_alias = self.function_base_aliases[func_idx]
        func_config = self.gui_configs.get(base_alias, {})
        outputs = dict(func_config.get("output_aliases", {}))
        outputs.update(self._get_passforward_output_aliases(instance_alias, base_alias))
        return outputs

    def _get_active_passforward_output_keys(self, instance_alias: str, base_alias: Optional[str] = None) -> set:
        """Return passforward output destination keys active for this instance."""
        return _svc_get_active_passforward_output_keys(
            instance_alias=instance_alias,
            base_alias=base_alias,
            methodology_list=self.methodology_list,
            function_base_aliases=self.function_base_aliases,
            function_configs=self.function_configs,
            gui_configs=self.gui_configs,
            normalize_bool_setting=_normalize_bool_setting,
        )

    def _prune_inactive_passforward_routes(self, instance_alias: str) -> int:
        """Remove routes sourced from disabled passforward outputs for a function instance."""
        if instance_alias not in self.methodology_list:
            return 0
        src_idx = self.methodology_list.index(instance_alias)
        if src_idx >= len(self.function_base_aliases):
            return 0

        base_alias = self.function_base_aliases[src_idx]
        active_outputs = set(self._get_output_spec_keys(base_alias, instance_alias))
        removed = 0

        for key, routing_info in list(self.routing_lines.items()):
            info = routing_info if isinstance(routing_info, dict) else {}
            existing_src_idx = info.get("src_idx", key[0] if isinstance(key, tuple) and len(key) > 0 else -1)
            src_param_key = info.get("src_param_key", key[1] if isinstance(key, tuple) and len(key) > 1 else "")

            try:
                existing_src_idx = int(existing_src_idx)
            except (TypeError, ValueError):
                continue

            if existing_src_idx != src_idx:
                continue
            if src_param_key in active_outputs:
                continue

            del self.routing_lines[key]
            removed += 1

        return removed

    def _set_passforward_enabled(self, instance_alias: str, enabled: bool):
        if instance_alias not in self.function_configs:
            self.function_configs[instance_alias] = {}
        self.function_configs[instance_alias]["__passforward_enabled__"] = bool(enabled)

        removed_count = self._prune_inactive_passforward_routes(instance_alias)
        self._recalculate_auto_routing()

        if removed_count > 0:
            self._set_routing_status(
                self.language_manager.translate(
                    "ui.messages.passforward_routes_removed",
                    "Disabled passforward removed routes that depended on passforward-only outputs"
                )
            )

        if self.current_tab == "routing":
            self._draw_canvas()

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
                # Synonym-style aliases are intentionally ignored for auto-routing.
                candidates.append((key, [key]))
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
        elif base_alias == "workflow_ensemble_start":
            text = f"┌ {display_name}"
        elif base_alias == "workflow_ensemble_member":
            text = f"├ {display_name}"
        elif base_alias == "workflow_ensemble_end":
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
            if base_alias in ("workflow_loop_end", "workflow_parallel_end", "workflow_ensemble_end"):
                depth = max(0, depth - 1)

            if base_alias in ("workflow_parallel_branch", "workflow_ensemble_member"):
                item_depth = max(0, depth - 1)
            else:
                item_depth = depth

            self.methodology_listbox.insert(tk.END, self._get_methodology_item_display(idx, item_depth))

            if base_alias in ("workflow_loop_start", "workflow_parallel_start", "workflow_ensemble_start"):
                depth += 1

        if selected_idx is not None and 0 <= selected_idx < len(self.methodology_list):
            self.methodology_listbox.selection_clear(0, tk.END)
            self.methodology_listbox.selection_set(selected_idx)
            self.methodology_listbox.activate(selected_idx)
            self.methodology_listbox.see(selected_idx)

        self._apply_methodology_row_highlights()
    
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
            if self._destination_has_manual_route(new_func_idx, dst_param_key):
                # Manual routing has priority for destination occupancy.
                continue
            # Find the immediately previous function that outputs this parameter
            for src_idx in range(new_func_idx - 1, -1, -1):  # Check backwards from newest to oldest
                src_base_alias = self.function_base_aliases[src_idx]  # Get the base alias
                if not self._can_auto_route_between(src_idx, new_func_idx):
                    continue
                src_instance_alias = self.methodology_list[src_idx] if src_idx < len(self.methodology_list) else ""
                src_outputs = self._get_output_spec_keys(src_base_alias, src_instance_alias)
                matched_src_param = None

                # Phase 1: direct output-name match
                for candidate in input_candidates:
                    if candidate in src_outputs:
                        matched_src_param = candidate
                        break

                if matched_src_param:
                    # Found the most recent function with this output
                    # Create automatic routing connection
                    key = (src_idx, matched_src_param, new_func_idx, dst_param_key)

                    if self._destination_has_manual_route(new_func_idx, dst_param_key):
                        # Manual routing might have been added while iterating candidates.
                        break
                    
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

    def _destination_has_manual_route(self, dst_idx: int, dst_param_key: str) -> bool:
        """Return True when destination already has any non-auto routing link."""
        for key, routing_info in self.routing_lines.items():
            if not isinstance(key, tuple) or len(key) < 4:
                continue

            info = routing_info if isinstance(routing_info, dict) else {}
            existing_dst_idx = info.get("dst_idx", key[2])
            existing_dst_param = info.get("dst_param_key", key[3])

            try:
                existing_dst_idx = int(existing_dst_idx)
            except (TypeError, ValueError):
                continue

            if existing_dst_idx != dst_idx or existing_dst_param != dst_param_key:
                continue

            if not info.get("auto_created", False):
                return True

        return False

    def _remove_routes_to_destination(self, dst_idx: int, dst_param_key: str, exclude_key: Optional[Tuple] = None):
        """Remove all routed links targeting the destination parameter, optionally excluding one key."""
        keys_to_remove = []
        for key, routing_info in self.routing_lines.items():
            if not isinstance(key, tuple) or len(key) < 4:
                continue
            if exclude_key is not None and key == exclude_key:
                continue

            info = routing_info if isinstance(routing_info, dict) else {}
            existing_dst_idx = info.get("dst_idx", key[2])
            existing_dst_param = info.get("dst_param_key", key[3])

            try:
                existing_dst_idx = int(existing_dst_idx)
            except (TypeError, ValueError):
                continue

            if existing_dst_idx == dst_idx and existing_dst_param == dst_param_key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.routing_lines[key]

        return len(keys_to_remove)

    def _set_routing_status(self, message: str):
        """Update routing tab status feedback when available."""
        if hasattr(self, "routing_status_var") and self.routing_status_var is not None:
            try:
                self.routing_status_var.set(str(message))
            except Exception:
                pass
    
    def _build_methodology_panel(self, parent: ttk.Frame):
        """Build methodology list with add/remove buttons."""
        panel_bg = self._get_theme_background_color()
        panel_fg = self._get_theme_foreground_color()
        list_frame = tk.LabelFrame(
            parent,
            text=self.language_manager.translate("ui.panels.selected_functions", "Selected Functions"),
            height=200,
            bg=panel_bg,
            fg=panel_fg,
            bd=1,
            relief=tk.GROOVE,
            labelanchor="n",
        )
        list_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        list_frame.pack_propagate(False)
        
        self.methodology_listbox = tk.Listbox(
            list_frame,
            height=10,
            selectmode=tk.SINGLE,
            activestyle="none",
            bg=panel_bg,
            fg=panel_fg,
            selectbackground=self._get_methodology_theme_colors()["select_bg"],
            selectforeground=self._get_methodology_theme_colors()["select_fg"],
            highlightthickness=0,
            borderwidth=0,
            highlightbackground=panel_bg,
            highlightcolor=panel_bg,
        )
        self.methodology_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.methodology_listbox.bind("<<ListboxSelect>>", self._on_methodology_select)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.methodology_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.methodology_listbox.config(yscrollcommand=scrollbar.set)
        self._configure_methodology_listbox_theme()
        
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=0, pady=10)
        
        up_btn = ttk.Button(button_frame, text="⬆", width=3, command=self._move_selected_methodology_up)
        up_btn.pack(side=tk.LEFT, padx=(0, 4))

        down_btn = ttk.Button(button_frame, text="⬇", width=3, command=self._move_selected_methodology_down)
        down_btn.pack(side=tk.LEFT, padx=4)

        delete_btn = ttk.Button(button_frame, text="✕", width=3, command=self._remove_from_methodology)
        delete_btn.pack(side=tk.LEFT, padx=4)
        
        clear_btn = ttk.Button(button_frame, text=self.language_manager.translate("ui.buttons.clear_methodology", "Clear All"), command=self._clear_methodology)
        clear_btn.pack(side=tk.LEFT, padx=(8, 0))

    def _get_active_methodology_index(self) -> Optional[int]:
        """Return selected methodology index from listbox selection or active row."""
        if not hasattr(self, "methodology_listbox"):
            return None

        selection = self.methodology_listbox.curselection()
        if selection:
            idx = selection[0]
            if 0 <= idx < len(self.methodology_list):
                return idx

        try:
            idx = int(self.methodology_listbox.index(tk.ACTIVE))
        except (tk.TclError, ValueError, TypeError):
            return None

        if 0 <= idx < len(self.methodology_list):
            return idx
        return None

    def _move_selected_methodology_up(self):
        """Move selected methodology item one position up."""
        idx = self._get_active_methodology_index()
        if idx is None or idx <= 0:
            return
        self._move_methodology_item(idx, idx - 1)

    def _move_selected_methodology_down(self):
        """Move selected methodology item one position down."""
        idx = self._get_active_methodology_index()
        if idx is None or idx >= len(self.methodology_list) - 1:
            return
        self._move_methodology_item(idx, idx + 1)

    def _remap_routing_indices(self, old_to_new_idx: Dict[int, int]):
        """Remap routing lines from old function indices to new indices."""
        self.routing_lines = _svc_remap_manual_routing_lines(self.routing_lines, old_to_new_idx)

    def _recalculate_auto_routing(self):
        """Rebuild automatic routing lines from current methodology order."""
        self.routing_lines = _svc_filter_manual_routing_lines(self.routing_lines)

        for idx, base_alias in enumerate(self.function_base_aliases):
            self._auto_create_routing(idx, base_alias)

    def _move_methodology_item(self, src_idx: int, dst_idx: int):
        """Move a methodology item and keep routing synchronized."""
        item_count = len(self.methodology_list)
        if item_count <= 1:
            return
        if src_idx < 0 or dst_idx < 0 or src_idx >= item_count or dst_idx >= item_count:
            return
        if src_idx == dst_idx:
            return

        original_indices = list(range(item_count))
        moved_index = original_indices.pop(src_idx)
        original_indices.insert(dst_idx, moved_index)
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(original_indices)}

        moved_instance_alias = self.methodology_list.pop(src_idx)
        moved_base_alias = self.function_base_aliases.pop(src_idx)
        self.methodology_list.insert(dst_idx, moved_instance_alias)
        self.function_base_aliases.insert(dst_idx, moved_base_alias)

        self._remap_routing_indices(old_to_new_idx)
        self._recalculate_auto_routing()

        self.selected_function_idx = dst_idx
        self._refresh_methodology_listbox(selected_idx=dst_idx)

        if self.current_tab == "analysis":
            self._show_analysis_tab()
        elif self.current_tab == "custom_analysis":
            self._show_custom_analysis_tab()
        elif self.current_tab == "setup":
            self._show_setup_tab()
        elif self.current_tab == "routing":
            self._show_routing_tab()
    
    def _on_methodology_select(self, event=None):
        """Handle methodology list selection."""
        selection = self.methodology_listbox.curselection()
        if selection:
            self.selected_function_idx = selection[0]
            self._clear_methodology_click_dismissable_highlights(self.selected_function_idx)
            # Only refresh function-specific tabs (Setup and Analysis)
            # Routing and Report are not function-specific, so don't refresh them
            if self.current_tab == "analysis":
                self._show_analysis_tab()
            elif self.current_tab == "setup":
                self._show_setup_tab()
            # Routing and Report tabs are not function-specific, don't refresh them

    def _remove_instance_persistent_state(self, instance_alias: str):
        """Remove all persisted state tied to a methodology instance alias."""
        if not isinstance(instance_alias, str) or not instance_alias:
            return

        # Setup/runtime configuration for the removed instance.
        self.function_configs.pop(instance_alias, None)

        # Analysis structures and caches for the removed instance.
        if hasattr(self, 'analysis_data') and isinstance(self.analysis_data, dict):
            self.analysis_data.pop(instance_alias, None)

            # Remove stale custom-analysis sections that referenced this source alias.
            custom_info = self.analysis_data.get(self.CUSTOM_ANALYSIS_ALIAS)
            if isinstance(custom_info, dict):
                pages = custom_info.get('pages', [])
                changed = False
                for page in pages if isinstance(pages, list) else []:
                    if not isinstance(page, dict):
                        continue
                    sections = page.get('sections', [])
                    if not isinstance(sections, list):
                        continue
                    for section_idx, section_data in enumerate(sections):
                        if not isinstance(section_data, dict):
                            continue
                        source = section_data.get('_custom_source')
                        source_alias = source.get('instance_alias') if isinstance(source, dict) else None
                        if source_alias == instance_alias:
                            sections[section_idx] = {'type': None}
                            changed = True

                if changed:
                    custom_info['execution_results'] = {}
                    active_sections = custom_info.get('active_sections', {})
                    if isinstance(active_sections, dict):
                        for page_idx, section_idx in list(active_sections.items()):
                            try:
                                page_i = int(page_idx)
                                section_i = int(section_idx)
                            except (TypeError, ValueError):
                                active_sections.pop(page_idx, None)
                                continue
                            if page_i < 0 or page_i >= len(pages):
                                active_sections.pop(page_idx, None)
                                continue
                            page_data = pages[page_i] if isinstance(pages[page_i], dict) else None
                            section_list = page_data.get('sections', []) if isinstance(page_data, dict) else []
                            section_data = section_list[section_i] if isinstance(section_list, list) and 0 <= section_i < len(section_list) else None
                            if not isinstance(section_data, dict) or section_data.get('type') is None:
                                active_sections.pop(page_idx, None)

        # Prevent stale highlight state from reappearing when alias is reused.
        if hasattr(self, 'methodology_row_highlights') and isinstance(self.methodology_row_highlights, dict):
            self.methodology_row_highlights.pop(instance_alias, None)

        if hasattr(self, 'selected_button'):
            self.selected_button = None
    
    def _remove_from_methodology(self):
        """Remove selected item from methodology."""
        idx = self._get_active_methodology_index()
        if idx is None:
            return

        start_idx, end_idx = self._get_wrapper_deletion_range(idx)
        self._remove_methodology_range(start_idx, end_idx)
    
    def _clear_methodology(self):
        """Clear all methodology items."""
        removed_aliases = list(self.methodology_list)
        self.methodology_list.clear()
        self.function_base_aliases.clear()
        self.routing_lines.clear()
        for instance_alias in removed_aliases:
            self._remove_instance_persistent_state(instance_alias)

        # Reset custom-analysis container when methodology is fully cleared.
        self.custom_analysis_data = {'pages': [], 'current_page': 0, 'active_sections': {}}
        if not hasattr(self, 'analysis_data') or not isinstance(self.analysis_data, dict):
            self.analysis_data = {}
        self.analysis_data[self.CUSTOM_ANALYSIS_ALIAS] = self.custom_analysis_data

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

        custom_analysis_btn = ttk.Button(
            control_frame,
            text=self.language_manager.translate("ui.tabs.custom_analysis", "C. Analysis"),
            command=self._show_custom_analysis_tab,
            width=12
        )
        custom_analysis_btn.pack(side=tk.LEFT, padx=5)
        
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
        active_stage = completed if completed >= total else min(total, completed + 1)
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

            if base_alias == "__lazy_loading__":
                display_name = self.language_manager.translate(
                    "ui.progress.lazy_loading_stage",
                    "Loading libraries"
                )

            if display_name:
                current_func_label = f" • {display_name}"

        if self.execution_progress_status_label:
            mode = self.execution_progress_mode or self.language_manager.translate("ui.buttons.run_model", "Run Model")
            self.execution_progress_status_label.configure(text=f"{mode}: {active_stage}/{total}{current_func_label}")
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
        self.root.title(self.language_manager.translate("ui.main_title", "Chemometric Studio"))
        
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

        passforward_cfg = self._get_passforward_config(base_alias)
        if passforward_cfg:
            passforward_frame = ttk.Frame(title_frame)
            passforward_frame.pack(side=tk.RIGHT, padx=(8, 12))

            pf_label = ttk.Label(passforward_frame, text="Passforward", font=("Arial", 9, "bold"))
            pf_label.pack(side=tk.LEFT, padx=(0, 4))

            passforward_var = tk.BooleanVar(value=self._is_passforward_enabled(instance_alias, base_alias))
            pf_style = ttk.Style(self.root)
            pf_bg = (
                pf_style.lookup("TFrame", "background")
                or pf_style.lookup(".", "background")
                or self.root.cget("bg")
                or "#f0f0f0"
            )
            pf_track_on = (
                pf_style.lookup("TCheckbutton", "indicatorcolor")
                or pf_style.lookup(".", "selectbackground")
                or pf_style.lookup(".", "focuscolor")
                or "#22c55e"
            )
            pf_track_off = (
                pf_style.lookup("TCheckbutton", "indicatorbackground")
                or pf_style.lookup(".", "bordercolor")
                or "#8c8c8c"
            )
            pf_knob_fill = (
                pf_style.lookup("TEntry", "fieldbackground")
                or pf_style.lookup(".", "fieldbackground")
                or "#ffffff"
            )
            pf_knob_outline = (
                pf_style.lookup("TCheckbutton", "indicatorforeground")
                or pf_style.lookup(".", "foreground")
                or "#d0d0d0"
            )

            passforward_toggle = ToggleSwitch(
                passforward_frame,
                variable=passforward_var,
                bg=pf_bg,
                track_on_color=pf_track_on,
                track_off_color=pf_track_off,
                knob_fill_color=pf_knob_fill,
                knob_outline_color=pf_knob_outline,
                command=lambda a=instance_alias, v=passforward_var: self._set_passforward_enabled(a, v.get())
            )
            passforward_toggle.pack(side=tk.LEFT)

            passforward_desc = passforward_cfg.get("description", "")
            tooltip_text = passforward_desc or "Expose mapped passforward outputs for downstream routing"
            Tooltip(passforward_toggle, tooltip_text)
            Tooltip(pf_label, tooltip_text)
        
        if not layout:
            label = ttk.Label(self.tab_content_frame, text=self.language_manager.translate("ui.messages.no_config", "No configuration available for this function"))
            label.pack(padx=20, pady=20)
            return
        
        # Create scrollable form frame
        scroll_container = ttk.Frame(self.tab_content_frame)
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create canvas and scrollbar
        setup_bg = self._get_theme_background_color()
        form_canvas = tk.Canvas(
            scroll_container,
            bg=setup_bg,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(scroll_container, orient=tk.VERTICAL, command=form_canvas.yview)
        scrollable_frame = ttk.Frame(form_canvas)
        
        # Configure scrolling
        def _update_form_scrollregion(_event=None):
            bbox = form_canvas.bbox("all")
            if bbox is None:
                return
            form_canvas.configure(scrollregion=bbox)

            content_height = bbox[3] - bbox[1]
            viewport_height = max(1, form_canvas.winfo_height())
            if content_height <= viewport_height:
                form_canvas.yview_moveto(0.0)

        scrollable_frame.bind("<Configure>", _update_form_scrollregion)
        form_window = form_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _on_form_canvas_configure(event):
            try:
                # Keep form content stretched to canvas width so no off-theme side strip remains.
                form_canvas.itemconfigure(form_window, width=event.width)
                form_canvas.configure(bg=setup_bg)
            except tk.TclError:
                pass

        form_canvas.bind("<Configure>", _on_form_canvas_configure)
        form_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        form_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel to canvas (only when over this specific canvas/frame)
        def _on_mousewheel(event):
            try:
                step = int(-1 * (event.delta / 120))
                if step == 0:
                    return

                first, last = form_canvas.yview()
                if (step < 0 and first <= 0.0) or (step > 0 and last >= 1.0):
                    return

                form_canvas.yview_scroll(step, "units")
            except tk.TclError:
                # Canvas was destroyed, ignore the scroll event
                pass
        form_canvas.bind("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
        
        form_frame = scrollable_frame
        
        # Initialize function config if needed (use instance_alias as key)
        if instance_alias not in self.function_configs:
            self.function_configs[instance_alias] = {}

        # Keep inherited setup inputs synchronized with nearest upstream values.
        self._sync_upstream_linked_inputs(instance_alias)
        
        func_config = self.function_configs[instance_alias]
        swept_locked_params = self._get_swept_param_locks_for_index(self.selected_function_idx)
        ensemble_locked_params = self._get_ensemble_param_locks_for_index(self.selected_function_idx)
        locked_params = set(swept_locked_params) | set(ensemble_locked_params)

        def _get_lock_reason_text(param_name: str) -> str:
            if param_name in swept_locked_params:
                return "Swept by loop"
            if param_name in ensemble_locked_params:
                return "Controlled by Ensemble Start"
            return "Locked"
        
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
                "instance_alias": instance_alias,
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
                
                # Persist while typing so switching methodology items does not drop unsaved text.
                def on_entry_key_release(event, n=name, e_widget=entry, a=instance_alias, vw=visible_widgets, ch=category_headers):
                    self._save_widget_value(a, n, e_widget.get())
                    self._update_field_visibility(a, vw, ch)
                entry.bind("<KeyRelease>", on_entry_key_release)

                if name in locked_params:
                    entry.configure(state="disabled")
                    lock_label = ttk.Label(input_container, text=_get_lock_reason_text(name), font=("Arial", 8, "italic"))
                    lock_label.pack(anchor=tk.W, padx=20, pady=(0, 2))
                
                widget_data["widget"] = entry
                
            elif widget_type == "combobox":
                values = widget_spec.get("values", [])
                value_aliases = widget_spec.get("value_aliases", values)  # Use values as fallback if no aliases
                editable_combo = bool(widget_spec.get("editable", False))

                values, value_aliases, hidden_actual_values = self._filter_hidden_setup_choices(values, value_aliases)
                
                # Create mapping from alias to actual value
                alias_to_value = {str(alias): value for alias, value in zip(value_aliases, values)}
                value_to_alias = {str(value): str(alias) for value, alias in zip(values, value_aliases)}
                
                # Display aliases in the combobox
                combo_state = "normal" if editable_combo else "readonly"
                combo = ttk.Combobox(input_container, values=value_aliases, width=37, state=combo_state)
                
                # Get the current value
                current_value = func_config.get(name, default if default else "")
                if current_value and str(current_value) not in hidden_actual_values:
                    # Normalize: if current_value is a display alias (legacy model), convert to actual value
                    actual_value = alias_to_value.get(str(current_value), current_value)
                    display_text = value_to_alias.get(str(actual_value), str(current_value))
                    combo.set(display_text)
                    self._save_widget_value(instance_alias, name, actual_value)
                elif default and str(default) not in hidden_actual_values:
                    # Set default - get the corresponding alias
                    display_text = value_to_alias.get(str(default), str(default))
                    combo.set(display_text)
                    self._save_widget_value(instance_alias, name, default)
                elif value_aliases:
                    combo.set(value_aliases[0])
                    self._save_widget_value(instance_alias, name, alias_to_value.get(value_aliases[0], value_aliases[0]))
                else:
                    combo.set("")
                    self._save_widget_value(instance_alias, name, "")
                
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
                    blocked_values = field_data.get("blocked_values", set())
                    if str(actual_value) in blocked_values:
                        c_widget.set("")
                        self._save_widget_value(a, n, "")
                        self._update_field_visibility(a, vw, ch)
                        return
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
                    blocked_values = field_data.get("blocked_values", set())
                    if str(actual_value) in blocked_values:
                        c_widget.set("")
                        self._save_widget_value(a, n, "")
                        self._update_field_visibility(a, vw, ch)
                        return
                    self._save_widget_value(a, n, actual_value)
                    self._update_field_visibility(a, vw, ch)
                combo.bind("<FocusOut>", on_combo_focus_out)

                if name in locked_params:
                    combo.configure(state="disabled")
                    lock_label = ttk.Label(input_container, text=_get_lock_reason_text(name), font=("Arial", 8, "italic"))
                    lock_label.pack(anchor=tk.W, padx=20, pady=(0, 2))
                
                widget_data["widget"] = combo
                widget_data["alias_to_value"] = alias_to_value
                widget_data["value_to_alias"] = value_to_alias  # Store for later reference if needed
                widget_data["blocked_values"] = hidden_actual_values
                
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
                    lock_label = ttk.Label(input_container, text=_get_lock_reason_text(name), font=("Arial", 8, "italic"))
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
                existing_cfg = self.function_configs.get(instance_alias, {})
                had_persisted_value = save_to in existing_cfg or name in existing_cfg

                check_vars = []
                for actual_value, display_value in zip(values, value_aliases):
                    var = tk.BooleanVar(value=actual_value in selected_set)
                    chk = ttk.Checkbutton(checklist_frame, text=display_value, variable=var)
                    chk.pack(anchor=tk.W, pady=(0, 2))
                    check_vars.append((var, actual_value, chk))

                def on_checklist_change(*_args, vars_with_values=check_vars, a=instance_alias, field=name, save_field=save_to):
                    selected = [actual for var, actual, _widget in vars_with_values if var.get()]
                    self._save_widget_value(a, field, selected)
                    if save_field != field:
                        self._save_widget_value(a, save_field, ",".join(selected))

                for var, _actual, _widget in check_vars:
                    var.trace_add("write", on_checklist_change)

                if name in locked_params:
                    for _var, _actual, chk_widget in check_vars:
                        chk_widget.configure(state="disabled")
                    lock_label = ttk.Label(input_container, text=_get_lock_reason_text(name), font=("Arial", 8, "italic"))
                    lock_label.pack(anchor=tk.W, padx=20, pady=(0, 2))

                # Avoid clearing persisted checklist values when dynamic options are not yet available.
                if check_vars or not had_persisted_value:
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
                count_offset = int(widget_spec.get("count_offset", 0) or 0)
                count = max(1, self._resolve_dynamic_count_source(func_config, str(count_source)) + count_offset)
                item_labels = self._resolve_dynamic_item_labels(func_config, widget_spec, count)
                values = widget_spec.get("values", [])
                value_aliases = widget_spec.get("value_aliases", values)
                default_value = widget_spec.get("default", values[0] if values else "")

                values, value_aliases, hidden_actual_values = self._filter_hidden_setup_choices(values, value_aliases)
                if str(default_value) in hidden_actual_values:
                    default_value = values[0] if values else ""
                
                # Create mapping from alias to actual value
                alias_to_value = {str(alias): value for alias, value in zip(value_aliases, values)}
                value_to_alias = {str(value): str(alias) for value, alias in zip(values, value_aliases)}
                
                # Container frame for all comboboxes
                list_frame = ttk.Frame(input_container)
                list_frame.pack(anchor=tk.W, padx=20, pady=(0, 5), fill=tk.X)
                
                combo_widgets = []
                combo_item_frames = []
                current_values = func_config.get(name, [])
                if isinstance(current_values, str):
                    current_values = [v.strip() for v in current_values.split(',') if v.strip()]
                
                for i in range(count):
                    item_frame = ttk.Frame(list_frame)
                    item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
                    combo_item_frames.append(item_frame)
                    
                    item_label = ttk.Label(item_frame, text=f"  {item_labels[i]}:", width=18)
                    item_label.pack(side=tk.LEFT)
                    
                    combo = ttk.Combobox(item_frame, values=value_aliases, width=30, state="readonly")
                    
                    # Set current or default value
                    if i < len(current_values) and current_values[i] and str(current_values[i]) not in hidden_actual_values:
                        actual_item = alias_to_value.get(str(current_values[i]), current_values[i])
                        display_text = value_to_alias.get(str(actual_item), str(current_values[i]))
                        combo.set(display_text)
                    else:
                        display_text = value_to_alias.get(str(default_value), str(default_value))
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
                widget_data["item_frames"] = combo_item_frames
                widget_data["list_frame"] = list_frame
                widget_data["count_source"] = count_source
                widget_data["is_dynamic_list"] = True
                widget_data["blocked_values"] = hidden_actual_values

                self._apply_dynamic_list_item_visibility(instance_alias, name, widget_data, visible_widgets)
            
            elif widget_type == "file_selector_list":
                # Dynamic list of file selectors based on count_source parameter
                count_source = widget_spec.get("count_source", "nway_flag")
                count_offset = int(widget_spec.get("count_offset", 0) or 0)
                count = max(1, self._resolve_dynamic_count_source(func_config, str(count_source)) + count_offset)
                
                # Container frame for all file selectors
                list_frame = ttk.Frame(input_container)
                list_frame.pack(anchor=tk.W, padx=20, pady=(0, 5), fill=tk.X)
                
                file_widgets = []
                file_item_frames = []
                current_values = func_config.get(name, [])
                if isinstance(current_values, str):
                    current_values = [v.strip() for v in current_values.split(';') if v.strip()]
                
                for i in range(count):
                    item_frame = ttk.Frame(list_frame)
                    item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
                    file_item_frames.append(item_frame)
                    
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

                    # Persist while typing so switching methodology items does not drop unsaved text.
                    def on_file_key_release(event, widgets=file_widgets, n=name, a=instance_alias):
                        values_list = [w.get() for w in widgets]
                        self._save_widget_value(a, n, values_list)
                    file_entry.bind("<KeyRelease>", on_file_key_release)
                    
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
                widget_data["item_frames"] = file_item_frames
                widget_data["list_frame"] = list_frame
                widget_data["count_source"] = count_source
                widget_data["is_dynamic_list"] = True

                self._apply_dynamic_list_item_visibility(instance_alias, name, widget_data, visible_widgets)
            
            elif widget_type == "entry_list":
                # Dynamic list of entry fields based on count_source parameter
                count_source = widget_spec.get("count_source", "nway_flag")
                count_offset = int(widget_spec.get("count_offset", 0) or 0)
                count = max(1, self._resolve_dynamic_count_source(func_config, str(count_source)) + count_offset)
                default_value = widget_spec.get("default", "")
                item_labels = self._resolve_dynamic_item_labels(func_config, widget_spec, count)
                
                # Container frame for all entries
                list_frame = ttk.Frame(input_container)
                list_frame.pack(anchor=tk.W, padx=20, pady=(0, 5), fill=tk.X)
                
                entry_widgets = []
                entry_item_frames = []
                current_values = func_config.get(name, [])
                if isinstance(current_values, str):
                    current_values = [v.strip() for v in current_values.split(';') if v.strip()]
                
                for i in range(count):
                    item_frame = ttk.Frame(list_frame)
                    item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
                    entry_item_frames.append(item_frame)
                    
                    item_label = ttk.Label(item_frame, text=f"  {item_labels[i]}:", width=18)
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

                    # Persist while typing so switching methodology items does not drop unsaved text.
                    def on_entry_key_release(event, widgets=entry_widgets, n=name, a=instance_alias):
                        values_list = [w.get() for w in widgets]
                        self._save_widget_value(a, n, values_list)
                    entry.bind("<KeyRelease>", on_entry_key_release)
                
                # Save initial values
                initial_values = [w.get() for w in entry_widgets]
                self._save_widget_value(instance_alias, name, initial_values)
                
                widget_data["widget"] = entry_widgets
                widget_data["item_frames"] = entry_item_frames
                widget_data["list_frame"] = list_frame
                widget_data["count_source"] = count_source
                widget_data["is_dynamic_list"] = True

                self._apply_dynamic_list_item_visibility(instance_alias, name, widget_data, visible_widgets)

            elif widget_type == "checkbutton_list":
                # Dynamic horizontal list of checkbuttons based on count_source parameter.
                count_source = widget_spec.get("count_source", "nway_flag")
                count_offset = int(widget_spec.get("count_offset", 0) or 0)
                count = max(1, self._resolve_dynamic_count_source(func_config, str(count_source)) + count_offset)
                default_value = bool(widget_spec.get("default", False))
                item_labels = widget_spec.get("item_labels", [])
                item_label_prefix = widget_spec.get("item_label_prefix", "")

                list_frame = ttk.Frame(input_container)
                list_frame.pack(anchor=tk.W, padx=20, pady=(0, 5), fill=tk.X)

                raw_values = func_config.get(name, [])
                if isinstance(raw_values, str):
                    raw_parts = [segment.strip() for segment in raw_values.replace(';', ',').split(',') if segment.strip()]
                    current_values = [part.lower() in ("1", "true", "yes", "on") for part in raw_parts]
                elif isinstance(raw_values, list):
                    current_values = [bool(v) for v in raw_values]
                else:
                    current_values = []

                check_vars = []
                check_widgets = []

                def on_checkbutton_list_change(*_args, vars_ref=check_vars, a=instance_alias, n=name, vw=visible_widgets, ch=category_headers):
                    self._save_widget_value(a, n, [var.get() for var in vars_ref])
                    self._update_field_visibility(a, vw, ch)

                for i in range(count):
                    initial_value = current_values[i] if i < len(current_values) else default_value
                    var = tk.BooleanVar(value=initial_value)

                    if i < len(item_labels):
                        item_text = str(item_labels[i])
                    elif item_label_prefix:
                        item_text = f"{item_label_prefix} {i + 1}"
                    else:
                        item_text = f"[{i + 1}]"

                    check = ttk.Checkbutton(list_frame, text=item_text, variable=var)
                    check.pack(side=tk.LEFT, padx=(0, 10), pady=(0, 2))
                    var.trace_add("write", on_checkbutton_list_change)

                    check_vars.append(var)
                    check_widgets.append(check)

                if name in locked_params:
                    for check in check_widgets:
                        check.configure(state="disabled")
                    lock_label = ttk.Label(input_container, text="Swept by loop", font=("Arial", 8, "italic"))
                    lock_label.pack(anchor=tk.W, padx=20, pady=(0, 2))

                # Save initial values
                self._save_widget_value(instance_alias, name, [var.get() for var in check_vars])

                widget_data["widget"] = check_vars
                widget_data["list_frame"] = list_frame
                widget_data["count_source"] = count_source
                widget_data["is_dynamic_list"] = True
            
            elif widget_type == "sample_paths_list":
                # Dynamic list of multi-file selectors - each sample can have multiple files
                count_source = widget_spec.get("count_source", "num_samples")
                count_offset = int(widget_spec.get("count_offset", 0) or 0)
                count = max(1, self._resolve_dynamic_count_source(func_config, str(count_source)) + count_offset)
                
                # Container frame for all sample entries
                list_frame = ttk.Frame(input_container)
                list_frame.pack(anchor=tk.W, padx=20, pady=(0, 5), fill=tk.X)
                
                sample_widgets = []  # List of (entry_widget, files_list) tuples
                sample_item_frames = []
                current_values = func_config.get(name, [])
                if isinstance(current_values, str):
                    # Parse semicolon-separated samples, where each sample has comma-separated files
                    current_values = []
                
                for i in range(count):
                    item_frame = ttk.Frame(list_frame)
                    item_frame.pack(anchor=tk.W, pady=(0, 4), fill=tk.X)
                    sample_item_frames.append(item_frame)
                    
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

                    # Persist while typing so switching methodology items does not drop unsaved text.
                    def on_sample_key_release(event, idx=i, f_widget=file_entry, widgets=sample_widgets, n=name, a=instance_alias):
                        text = f_widget.get()
                        if text.strip():
                            parsed_files = [f.strip() for f in text.split(';') if f.strip()]
                            widgets[idx]["files"] = parsed_files
                        else:
                            widgets[idx]["files"] = []
                        values_list = [w["files"] for w in widgets]
                        self._save_widget_value(a, n, values_list)
                    file_entry.bind("<KeyRelease>", on_sample_key_release)
                
                # Save initial values
                initial_values = [w["files"] for w in sample_widgets]
                self._save_widget_value(instance_alias, name, initial_values)
                
                widget_data["widget"] = sample_widgets
                widget_data["item_frames"] = sample_item_frames
                widget_data["list_frame"] = list_frame
                widget_data["count_source"] = count_source
                widget_data["is_dynamic_list"] = True

                self._apply_dynamic_list_item_visibility(instance_alias, name, widget_data, visible_widgets)
        
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

    def _find_matching_control_start(self, end_idx: int, start_alias: str, end_alias: str) -> int:
        depth = 0
        for idx in range(end_idx, -1, -1):
            base_alias = self.function_base_aliases[idx]
            if base_alias == end_alias:
                depth += 1
            elif base_alias == start_alias:
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    def _get_wrapper_deletion_range(self, idx: int) -> Tuple[int, int]:
        """Return [start_idx, end_idx] to remove for control wrappers or single item."""
        if idx < 0 or idx >= len(self.function_base_aliases):
            return idx, idx

        base_alias = self.function_base_aliases[idx]

        wrapper_pairs = {
            "workflow_loop_start": ("workflow_loop_start", "workflow_loop_end"),
            "workflow_loop_end": ("workflow_loop_start", "workflow_loop_end"),
            "workflow_parallel_start": ("workflow_parallel_start", "workflow_parallel_end"),
            "workflow_parallel_end": ("workflow_parallel_start", "workflow_parallel_end"),
            "workflow_ensemble_start": ("workflow_ensemble_start", "workflow_ensemble_end"),
            "workflow_ensemble_end": ("workflow_ensemble_start", "workflow_ensemble_end"),
        }

        pair = wrapper_pairs.get(base_alias)
        if pair is None:
            return idx, idx

        start_alias, end_alias = pair
        if base_alias == start_alias:
            start_idx = idx
            end_idx = self._find_matching_control_end(start_idx, start_alias, end_alias)
            if end_idx >= 0:
                return start_idx, end_idx
            return idx, idx

        end_idx = idx
        start_idx = self._find_matching_control_start(end_idx, start_alias, end_alias)
        if start_idx >= 0:
            return start_idx, end_idx
        return idx, idx

    def _remove_methodology_range(self, start_idx: int, end_idx: int):
        """Remove an inclusive methodology range while preserving routing consistency."""
        item_count = len(self.methodology_list)
        if item_count == 0:
            return

        start_idx = max(0, int(start_idx))
        end_idx = min(item_count - 1, int(end_idx))
        if start_idx > end_idx:
            return

        removed_aliases = self.methodology_list[start_idx:end_idx + 1]
        removed_indices = set(range(start_idx, end_idx + 1))

        keep_indices = [idx for idx in range(item_count) if idx not in removed_indices]
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}

        self.methodology_list = [self.methodology_list[idx] for idx in keep_indices]
        self.function_base_aliases = [self.function_base_aliases[idx] for idx in keep_indices]

        for instance_alias in removed_aliases:
            self._remove_instance_persistent_state(instance_alias)

        self._remap_routing_indices(old_to_new_idx)
        self._recalculate_auto_routing()

        if self.methodology_list:
            selected_idx = min(start_idx, len(self.methodology_list) - 1)
            self.selected_function_idx = selected_idx
            self._refresh_methodology_listbox(selected_idx=selected_idx)
        else:
            self.selected_function_idx = None
            self._refresh_methodology_listbox()

        self._clear_tab()

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

    def _get_ensemble_param_locks_for_index(self, target_idx: Optional[int]) -> set:
        """Return parameter names controlled by enclosing ensemble settings for given function index."""
        locked = set()
        if target_idx is None or target_idx < 0 or target_idx >= len(self.methodology_list):
            return locked

        target_base_alias = self.function_base_aliases[target_idx]
        if target_base_alias != "classification_one_class":
            return locked

        ensemble_stack: List[int] = []
        for idx in range(target_idx + 1):
            base_alias = self.function_base_aliases[idx]
            if base_alias == "workflow_ensemble_start":
                ensemble_stack.append(idx)
            elif base_alias == "workflow_ensemble_end" and ensemble_stack:
                ensemble_stack.pop()

        # Inner-most enclosing ensemble takes precedence.
        for ensemble_idx in reversed(ensemble_stack):
            ensemble_instance = self.methodology_list[ensemble_idx]
            ensemble_cfg = self.function_configs.get(ensemble_instance, {})
            ensemble_task_type = str(ensemble_cfg.get("ensemble_task_type", "regression") or "regression").strip().lower()
            if ensemble_task_type != "classification":
                continue

            ref_class = ensemble_cfg.get("one_class_reference_class", "")
            if ref_class is None or str(ref_class).strip() == "":
                ref_lock = False
            else:
                ref_lock = True

            unknown_label = ensemble_cfg.get("one_class_unknown_label", "")
            if unknown_label is None or str(unknown_label).strip() == "":
                unknown_lock = False
            else:
                unknown_lock = True

            if ref_lock:
                locked.add("one_class_reference_class")
            if unknown_lock:
                locked.add("one_class_unknown_label")
            if ref_lock or unknown_lock:
                break

        return locked

    def _get_widget_spec_by_name(self, base_alias: str, field_name: str) -> Optional[Dict[str, Any]]:
        config = self.gui_configs.get(base_alias, {})
        layout = config.get("setup", {}).get("layout", [])
        for field in layout:
            if field.get("name") == field_name:
                return field
        return None

    def _filter_hidden_setup_choices(self, values: List[Any], value_aliases: List[Any]) -> Tuple[List[Any], List[str], Set[str]]:
        """Filter setup-tab selectable options hidden with '$X.' alias marker."""
        filtered_values: List[Any] = []
        filtered_aliases: List[str] = []
        hidden_actual_values: Set[str] = set()

        for idx, actual_value in enumerate(values):
            alias_value = value_aliases[idx] if idx < len(value_aliases) else actual_value
            alias_text = str(alias_value)
            actual_text = str(actual_value)

            if alias_text.startswith("$X."):
                hidden_actual_values.add(actual_text)
                continue

            filtered_values.append(actual_value)
            filtered_aliases.append(alias_text)

        return filtered_values, filtered_aliases, hidden_actual_values

    def _set_setup_combobox_options(self, widget_data: Dict[str, Any], actual_values: List[str], display_values: List[str], selected_actual: Optional[str] = None):
        combo = widget_data.get("widget")
        if combo is None:
            return
        filtered_actual_values, filtered_display_values, hidden_actual_values = self._filter_hidden_setup_choices(actual_values, display_values)
        alias_to_value = {str(alias): value for alias, value in zip(filtered_display_values, filtered_actual_values)}
        value_to_alias = {str(value): str(alias) for value, alias in zip(filtered_actual_values, filtered_display_values)}
        combo.configure(values=filtered_display_values)
        widget_data["alias_to_value"] = alias_to_value
        widget_data["value_to_alias"] = value_to_alias
        widget_data["blocked_values"] = hidden_actual_values

        selected_text = str(selected_actual) if selected_actual is not None else ""
        if selected_text in hidden_actual_values:
            selected_actual = None

        resolved_actual: Optional[Any] = None

        if selected_actual:
            selected_alias = value_to_alias.get(str(selected_actual), str(selected_actual))
            combo.set(selected_alias)
            resolved_actual = alias_to_value.get(str(selected_alias), selected_actual)
        elif combo.get().strip() and combo.get().strip() in alias_to_value:
            resolved_actual = alias_to_value.get(combo.get().strip())
        elif filtered_display_values:
            combo.set(filtered_display_values[0])
            resolved_actual = alias_to_value.get(filtered_display_values[0], filtered_display_values[0])
        else:
            combo.set("")

        # Persist programmatic defaults so first visible option is truly selected even without manual click.
        if resolved_actual is not None:
            instance_alias = widget_data.get("instance_alias")
            field_name = widget_data.get("field_name")
            if isinstance(instance_alias, str) and isinstance(field_name, str) and field_name:
                self._save_widget_value(instance_alias, field_name, resolved_actual)

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
            if save_field != field:
                self._save_widget_value(a, save_field, ",".join(selected))

        for var, _actual, _widget in check_vars:
            var.trace_add("write", on_checklist_change)

        # If no options are currently available, keep persisted values untouched.
        if check_vars:
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
                is_boolean = type_name == "bool" or str(field.get("type", "")).lower() == "bool" or field_widget == "checkbutton"
                is_choice = field_widget == "combobox"
                is_sweep_choice = is_choice or is_boolean

                if current_mode == "sweep_numeric" and not is_numeric:
                    continue
                if current_mode == "sweep_choice" and not is_sweep_choice:
                    continue
                if current_mode not in ("sweep_numeric", "sweep_choice"):
                    continue

                target_value = f"{body_instance}.{field_name}"
                target_label = field.get("label", field_name)
                target_alias = f"{body_display} [{body_instance}] · {target_label}"

                sweep_targets_actual.append(target_value)
                sweep_targets_display.append(target_alias)

                if is_choice:
                    choice_values = field.get("values", [])
                    choice_aliases = field.get("value_aliases", choice_values)
                    choice_values, choice_aliases, _hidden_values = self._filter_hidden_setup_choices(choice_values, choice_aliases)
                elif is_boolean:
                    choice_values = ["false", "true"]
                    choice_aliases = ["Off", "On"]
                else:
                    choice_values = []
                    choice_aliases = []
                target_meta[target_value] = {
                    "is_choice": is_sweep_choice,
                    "is_boolean": is_boolean,
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
            # Prefer explicit checklist state when available, then fall back to serialized sweep_values.
            cfg = self.function_configs.get(instance_alias, {})
            current_choice_raw = cfg.get("sweep_choice_values", None)
            if isinstance(current_choice_raw, list) and current_choice_raw:
                current_sweep_raw = current_choice_raw
            elif isinstance(current_choice_raw, str) and current_choice_raw.strip():
                current_sweep_raw = current_choice_raw
            else:
                current_sweep_raw = cfg.get("sweep_values", "")
            if isinstance(current_sweep_raw, str):
                selected_choice_values = [segment.strip() for segment in current_sweep_raw.split(',') if segment.strip()]
            elif isinstance(current_sweep_raw, list):
                selected_choice_values = [str(v) for v in current_sweep_raw]
            else:
                selected_choice_values = []

            # Accept legacy alias-based payloads by canonicalizing to actual values.
            alias_to_actual = {
                str(alias): str(actual)
                for actual, alias in zip(choice_values, choice_aliases)
            }
            canonical_selected: List[str] = []
            valid_actual = {str(v) for v in choice_values}
            for raw in selected_choice_values:
                token = str(raw).strip()
                if not token:
                    continue
                if token in valid_actual:
                    canonical_selected.append(token)
                    continue
                mapped = alias_to_actual.get(token)
                if mapped is not None:
                    canonical_selected.append(mapped)

            selected_choice_values = canonical_selected

            if target_info.get("is_boolean"):
                valid_values = {str(v) for v in choice_values}
                selected_normalized = [str(v).strip().lower() for v in selected_choice_values if str(v).strip()]
                has_valid_selection = any(v in valid_values for v in selected_normalized)
                if not has_valid_selection:
                    selected_choice_values = ["false", "true"]

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

        # Immediately propagate values used by downstream inherited setup fields.
        self._propagate_linked_input_value(func_alias, param_name, value)

    def _sync_upstream_linked_inputs(self, target_instance_alias: str):
        """Sync inherited setup inputs from nearest upstream provider."""
        if target_instance_alias not in self.methodology_list:
            return

        target_idx = self.methodology_list.index(target_instance_alias)
        base_alias = self.function_base_aliases[target_idx]
        layout = self.gui_configs.get(base_alias, {}).get("setup", {}).get("layout", [])
        if not layout:
            return

        target_config = self.function_configs.setdefault(target_instance_alias, {})

        for field_info in layout:
            field_name = field_info.get("name")
            if not field_name:
                continue

            input_type = field_info.get("input_type", "user")
            if input_type != "inherited":
                continue

            source_idx = self._find_nearest_upstream_provider_index(target_idx, field_name)
            if source_idx is None:
                continue

            source_alias = self.methodology_list[source_idx]
            source_value = self.function_configs.get(source_alias, {}).get(field_name)
            if source_value is None:
                continue

            target_config[field_name] = source_value

    def _propagate_linked_input_value(self, source_instance_alias: str, param_name: str, value: Any):
        """Propagate a changed parameter to downstream inherited fields when applicable."""
        if source_instance_alias not in self.methodology_list:
            return

        source_idx = self.methodology_list.index(source_instance_alias)

        for dst_idx in range(source_idx + 1, len(self.methodology_list)):
            dst_alias = self.methodology_list[dst_idx]
            dst_base_alias = self.function_base_aliases[dst_idx]
            dst_layout = self.gui_configs.get(dst_base_alias, {}).get("setup", {}).get("layout", [])

            receives_param = False
            for field_info in dst_layout:
                if field_info.get("name") != param_name:
                    continue
                if field_info.get("input_type", "user") == "inherited":
                    receives_param = True
                    break

            if not receives_param:
                continue

            provider_idx = self._find_nearest_upstream_provider_index(dst_idx, param_name)
            if provider_idx != source_idx:
                continue

            self.function_configs.setdefault(dst_alias, {})[param_name] = value

    def _find_nearest_upstream_provider_index(self, target_idx: int, param_name: str) -> Optional[int]:
        """Return nearest upstream function index that explicitly provides param_name in setup."""
        for prev_idx in range(target_idx - 1, -1, -1):
            prev_base_alias = self.function_base_aliases[prev_idx]
            prev_layout = self.gui_configs.get(prev_base_alias, {}).get("setup", {}).get("layout", [])

            for prev_field_info in prev_layout:
                if prev_field_info.get("name") != param_name:
                    continue
                if prev_field_info.get("input_type", "user") != "inherited":
                    return prev_idx

        return None
    
    def _show_help_popup(self, title: str, short_desc: str, long_desc: str):
        """Show a popup window with function help information."""
        popup = tk.Toplevel(self.root)
        popup.title(f"{self.language_manager.translate('ui.dialogs.help_for', 'Help:')} {title}")
        popup.geometry("600x400")
        
        _set_window_icon(popup, "Info")
        
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

        def _evaluate_single_condition(condition_field, condition_value, widget_spec):
            """Evaluate one visible_if condition against current setup state."""
            if condition_field == "input_type":
                actual_input_type = widget_spec.get("input_type", "user")
                return actual_input_type == condition_value

            current_value = func_config.get(condition_field)

            widget_data_for_condition = visible_widgets.get(condition_field, {})
            if "variable" in widget_data_for_condition:
                current_value = widget_data_for_condition["variable"].get()
            else:
                cond_widget_spec = widget_data_for_condition.get("widget_spec", {})
                cond_widget_type = cond_widget_spec.get("widget")
                cond_widget = widget_data_for_condition.get("widget")
                if cond_widget_type == "checkbutton_list" and isinstance(cond_widget, list):
                    current_value = [bool(var.get()) for var in cond_widget]

            if isinstance(condition_value, dict) and "operator" in condition_value:
                operator = condition_value.get("operator", "==")
                expected_value = condition_value.get("value")

                # List-aware operators used by checkbutton_list fields.
                if operator in {"any_true", "all_true", "any_false", "all_false", "index_true", "index_false"}:
                    if isinstance(current_value, (list, tuple)):
                        bool_values = [bool(v) for v in current_value]
                    else:
                        bool_values = [bool(current_value)]

                    if operator == "any_true":
                        return any(bool_values)
                    if operator == "all_true":
                        return bool_values and all(bool_values)
                    if operator == "any_false":
                        return any((not val) for val in bool_values)
                    if operator == "all_false":
                        return (not bool_values) or all((not val) for val in bool_values)
                    if operator in {"index_true", "index_false"}:
                        idx = int(condition_value.get("index", -1))
                        if idx < 0 or idx >= len(bool_values):
                            return False
                        return bool_values[idx] if operator == "index_true" else (not bool_values[idx])
                    return True

                try:
                    current_val = int(current_value) if current_value is not None else 0
                    expected_val = int(expected_value) if expected_value is not None else 0
                except (ValueError, TypeError):
                    current_val = str(current_value) if current_value is not None else ""
                    expected_val = str(expected_value) if expected_value is not None else ""

                if operator == "==":
                    return current_val == expected_val
                if operator == "!=":
                    return current_val != expected_val
                if operator == ">":
                    return current_val > expected_val
                if operator == "<":
                    return current_val < expected_val
                if operator == ">=":
                    return current_val >= expected_val
                if operator == "<=":
                    return current_val <= expected_val
                return True

            if isinstance(condition_value, list):
                return current_value in condition_value
            return current_value == condition_value

        def _evaluate_visible_if(visible_if_expr, widget_spec):
            """Evaluate visible_if expressions, including nested any/all condition groups."""
            if not visible_if_expr:
                return True
            if visible_if_expr is False:
                return False
            if not isinstance(visible_if_expr, dict):
                return bool(visible_if_expr)

            if "any" in visible_if_expr:
                any_conditions = visible_if_expr.get("any", [])
                if not isinstance(any_conditions, list):
                    return False
                return any(_evaluate_visible_if(cond, widget_spec) for cond in any_conditions)

            if "all" in visible_if_expr:
                all_conditions = visible_if_expr.get("all", [])
                if not isinstance(all_conditions, list):
                    return False
                return all(_evaluate_visible_if(cond, widget_spec) for cond in all_conditions)

            for condition_field, condition_value in visible_if_expr.items():
                if not _evaluate_single_condition(condition_field, condition_value, widget_spec):
                    return False
            return True
        
        for field_name, widget_data in visible_widgets.items():
            visible_if = widget_data.get("visible_if")
            container = widget_data.get("container")
            widget_spec = widget_data.get("widget_spec", {})
            grid_params = widget_data.get("grid_params", {})
            input_type = str(widget_spec.get("input_type", "user")).lower()
            
            # Hide routed/inherited setup inputs by default; only user inputs are shown.
            should_show = input_type == "user"
            
            # Handle explicit false (for always-hidden fields)
            if visible_if is False:
                should_show = False
            elif visible_if:
                should_show = _evaluate_visible_if(visible_if, widget_spec)
            
            # Show or hide the container using grid with stored parameters
            if should_show:
                container.grid(**grid_params)  # Re-show with original grid parameters
                widget_spec = widget_data.get("widget_spec", {})
                if widget_spec.get("widget") in {"entry_list", "combobox_list", "file_selector_list", "sample_paths_list"}:
                    self._apply_dynamic_list_item_visibility(func_alias, field_name, widget_data, visible_widgets)
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
        """Rebuild dynamic list widgets when their count source changes."""
        for field_name, widget_data in visible_widgets.items():
            if not widget_data.get("is_dynamic_list"):
                continue
            
            count_source = widget_data.get("count_source")
            if not count_source:
                continue
            
            # Get current count from config
            widget_spec = widget_data.get("widget_spec", {})
            count_offset = int(widget_spec.get("count_offset", 0) or 0)
            new_count = max(1, self._resolve_dynamic_count_source(func_config, str(count_source)) + count_offset)
            
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
            elif widget_type == "checkbutton_list":
                self._rebuild_checkbutton_list(func_alias, field_name, widget_data, widget_spec,
                                              list_frame, new_count, current_values, visible_widgets)
            elif widget_type == "sample_paths_list":
                self._rebuild_sample_paths_list(func_alias, field_name, widget_data, widget_spec,
                                               list_frame, new_count, current_values, visible_widgets)

    def _apply_dynamic_list_item_visibility(self, func_alias: str, field_name: str, widget_data: Dict,
                                            visible_widgets: Dict) -> None:
        """Apply optional per-item visibility rules for dynamic list rows.

        Expected widget_spec format:
        "item_visible_if": {
            "field": "constraint_name",
            "operator": "index_true" | "index_false" | "any_true" | "all_true" | "any_false" | "all_false"
        }
        """
        widget_spec = widget_data.get("widget_spec", {})
        rule = widget_spec.get("item_visible_if")
        if not isinstance(rule, dict):
            return

        source_field = str(rule.get("field", "")).strip()
        if not source_field:
            return

        operator = str(rule.get("operator", "index_true")).strip().lower()
        source_widget_data = visible_widgets.get(source_field, {})

        source_values: List[bool] = []
        source_widget = source_widget_data.get("widget")
        source_widget_spec = source_widget_data.get("widget_spec", {})
        source_widget_type = source_widget_spec.get("widget")

        if source_widget_type == "checkbutton_list" and isinstance(source_widget, list):
            source_values = [bool(var.get()) for var in source_widget]
        elif "variable" in source_widget_data:
            source_values = [bool(source_widget_data["variable"].get())]
        else:
            func_config = self.function_configs.get(func_alias, {})
            raw = func_config.get(source_field)
            if isinstance(raw, list):
                source_values = [bool(v) for v in raw]
            elif raw is not None:
                source_values = [bool(raw)]

        item_frames = widget_data.get("item_frames", [])
        if not isinstance(item_frames, list) or not item_frames:
            return

        show_flags: List[bool] = []
        for idx, _frame in enumerate(item_frames):
            if operator == "index_false":
                show_item = idx < len(source_values) and (not bool(source_values[idx]))
            elif operator == "any_true":
                show_item = bool(source_values) and any(source_values)
            elif operator == "all_true":
                show_item = bool(source_values) and all(source_values)
            elif operator == "any_false":
                show_item = bool(source_values) and any((not v) for v in source_values)
            elif operator == "all_false":
                show_item = (not source_values) or all((not v) for v in source_values)
            else:
                # Default: index_true
                show_item = idx < len(source_values) and bool(source_values[idx])
            show_flags.append(bool(show_item))

        # Repack rows in canonical index order to avoid order drift from pack_forget/pack cycles.
        for frame in item_frames:
            frame.pack_forget()
        for idx, frame in enumerate(item_frames):
            if idx < len(show_flags) and show_flags[idx]:
                frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)

    def _apply_entry_list_item_visibility(self, func_alias: str, field_name: str, widget_data: Dict,
                                          visible_widgets: Dict) -> None:
        """Backward-compatible wrapper for dynamic list item visibility."""
        self._apply_dynamic_list_item_visibility(func_alias, field_name, widget_data, visible_widgets)

    def _resolve_dynamic_count_source(self, func_config: Dict[str, Any], count_source: str) -> int:
        """Resolve dynamic list count from a source key or simple +/- integer expression."""
        def _coerce_count(value: Any) -> Optional[int]:
            try:
                return int(float(value))
            except Exception:
                return None

        def _infer_nway_flag() -> Optional[int]:
            x_data = func_config.get("X_cal")
            x_ndim = getattr(x_data, "ndim", None)
            if isinstance(x_ndim, int) and x_ndim >= 1:
                return int(max(0, x_ndim - 1))

            axis_info = func_config.get("axis_n_info")
            if isinstance(axis_info, (list, tuple)) and len(axis_info) > 0:
                return int(len(axis_info))

            return None

        def _resolve_key_value(key: str) -> int:
            raw = func_config.get(key, None)
            parsed = _coerce_count(raw)
            if parsed is not None:
                return int(parsed)
            if key == "nway_flag":
                inferred = _infer_nway_flag()
                if inferred is not None:
                    return int(inferred)
            return 1

        expr = str(count_source or "").strip()
        if not expr:
            return 1

        direct_value = _coerce_count(func_config.get(expr, None))
        if direct_value is not None:
            return max(1, int(direct_value))

        import re

        # Supported patterns: "field+1", "field - 2", and numeric literals.
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*([+-])\s*(\d+)", expr)
        if match:
            key, op, delta_text = match.groups()
            base = _resolve_key_value(key)
            delta = int(delta_text)
            value = base + delta if op == "+" else base - delta
            return max(1, int(value))

        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr):
            return max(1, _resolve_key_value(expr))

        try:
            return max(1, int(float(expr)))
        except Exception:
            return 1

    def _resolve_dynamic_item_labels(self, func_config: Dict[str, Any], widget_spec: Dict, count: int) -> List[str]:
        """Resolve optional per-item labels for dynamic list widgets."""
        explicit_labels = widget_spec.get("item_labels", [])
        if isinstance(explicit_labels, list) and explicit_labels:
            labels = [str(v) for v in explicit_labels[:count]]
            if len(labels) < count:
                labels.extend([f"[{idx + 1}]" for idx in range(len(labels), count)])
            return labels

        source_key = str(widget_spec.get("item_labels_source", "")).strip()
        source_labels: List[str] = []
        if source_key:
            raw_source = func_config.get(source_key)
            if isinstance(raw_source, list):
                source_labels = [str(v).strip() for v in raw_source if str(v).strip()]
            elif isinstance(raw_source, str):
                import re
                source_labels = [
                    token.strip()
                    for token in re.split(r"[;,\t\n]+", raw_source)
                    if token and token.strip()
                ]

        item_label_prefix = str(widget_spec.get("item_label_prefix", "")).strip()
        labels: List[str] = []
        for idx in range(count):
            if idx < len(source_labels):
                labels.append(source_labels[idx])
            elif item_label_prefix:
                labels.append(f"{item_label_prefix} {idx + 1}")
            else:
                labels.append(f"[{idx + 1}]")
        return labels
    
    def _rebuild_combobox_list(self, func_alias: str, field_name: str, widget_data: Dict, 
                                widget_spec: Dict, list_frame, count: int, current_values: list,
                                visible_widgets: Dict):
        """Rebuild a combobox_list widget with new count."""
        values = widget_spec.get("values", [])
        value_aliases = widget_spec.get("value_aliases", values)
        default_value = widget_spec.get("default", values[0] if values else "")
        label_text = widget_spec.get("label", field_name)

        values, value_aliases, hidden_actual_values = self._filter_hidden_setup_choices(values, value_aliases)
        if str(default_value) in hidden_actual_values:
            default_value = values[0] if values else ""
        
        alias_to_value = {str(alias): value for alias, value in zip(value_aliases, values)}
        value_to_alias = {str(value): str(alias) for value, alias in zip(values, value_aliases)}
        item_labels = self._resolve_dynamic_item_labels(self.function_configs.get(func_alias, {}), widget_spec, count)
        
        combo_widgets = []
        combo_item_frames = []
        
        for i in range(count):
            item_frame = ttk.Frame(list_frame)
            item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
            combo_item_frames.append(item_frame)
            
            item_label = ttk.Label(item_frame, text=f"  {item_labels[i]}:", width=18)
            item_label.pack(side=tk.LEFT)
            
            combo = ttk.Combobox(item_frame, values=value_aliases, width=30, state="readonly")
            
            # Set current or default value
            if i < len(current_values) and current_values[i] and str(current_values[i]) not in hidden_actual_values:
                actual_item = alias_to_value.get(str(current_values[i]), current_values[i])
                display_text = value_to_alias.get(str(actual_item), str(current_values[i]))
                combo.set(display_text)
            else:
                display_text = value_to_alias.get(str(default_value), str(default_value))
                combo.set(display_text)
            
            combo.pack(side=tk.LEFT, padx=(5, 0))
            combo_widgets.append(combo)
            
            # Binding to save all combobox values as a list
            def on_combo_list_selected(event, widgets=combo_widgets, n=field_name, a=func_alias, a2v=alias_to_value):
                values_list = []
                for w in widgets:
                    selected_alias = w.get()
                    actual_value = a2v.get(selected_alias, selected_alias)
                    if str(actual_value) in hidden_actual_values:
                        continue
                    values_list.append(actual_value)
                self._save_widget_value(a, n, values_list)
            combo.bind("<<ComboboxSelected>>", on_combo_list_selected)
        
        # Save values
        new_values = []
        for w in combo_widgets:
            selected_alias = w.get()
            actual_value = alias_to_value.get(selected_alias, selected_alias)
            if str(actual_value) in hidden_actual_values:
                continue
            new_values.append(actual_value)
        self._save_widget_value(func_alias, field_name, new_values)
        
        # Update widget_data
        widget_data["widget"] = combo_widgets
        widget_data["item_frames"] = combo_item_frames
        widget_data["blocked_values"] = hidden_actual_values

        self._apply_dynamic_list_item_visibility(func_alias, field_name, widget_data, visible_widgets)
    
    def _rebuild_file_selector_list(self, func_alias: str, field_name: str, widget_data: Dict,
                                     widget_spec: Dict, list_frame, count: int, current_values: list,
                                     visible_widgets: Dict):
        """Rebuild a file_selector_list widget with new count."""
        label_text = widget_spec.get("label", field_name)
        
        file_widgets = []
        file_item_frames = []
        
        for i in range(count):
            item_frame = ttk.Frame(list_frame)
            item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
            file_item_frames.append(item_frame)
            
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

            # Persist while typing so switching methodology items does not drop unsaved text.
            def on_file_key_release(event, widgets=file_widgets, n=field_name, a=func_alias):
                values_list = [w.get() for w in widgets]
                self._save_widget_value(a, n, values_list)
            file_entry.bind("<KeyRelease>", on_file_key_release)
            
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
        widget_data["item_frames"] = file_item_frames

        self._apply_dynamic_list_item_visibility(func_alias, field_name, widget_data, visible_widgets)

    def _rebuild_entry_list(self, func_alias: str, field_name: str, widget_data: Dict,
                            widget_spec: Dict, list_frame, count: int, current_values: list,
                            visible_widgets: Dict):
        """Rebuild an entry_list widget with new count."""
        default_value = widget_spec.get("default", "")
        item_labels = self._resolve_dynamic_item_labels(self.function_configs.get(func_alias, {}), widget_spec, count)
        
        entry_widgets = []
        entry_item_frames = []
        
        for i in range(count):
            item_frame = ttk.Frame(list_frame)
            item_frame.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
            entry_item_frames.append(item_frame)
            
            item_label = ttk.Label(item_frame, text=f"  {item_labels[i]}:", width=18)
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

            # Persist while typing so switching methodology items does not drop unsaved text.
            def on_entry_key_release(event, widgets=entry_widgets, n=field_name, a=func_alias):
                values_list = [w.get() for w in widgets]
                self._save_widget_value(a, n, values_list)
            entry.bind("<KeyRelease>", on_entry_key_release)
        
        # Save initial values
        initial_values = [w.get() for w in entry_widgets]
        self._save_widget_value(func_alias, field_name, initial_values)
        
        # Update widget_data
        widget_data["widget"] = entry_widgets
        widget_data["item_frames"] = entry_item_frames

        self._apply_dynamic_list_item_visibility(func_alias, field_name, widget_data, visible_widgets)

    def _rebuild_checkbutton_list(self, func_alias: str, field_name: str, widget_data: Dict,
                                  widget_spec: Dict, list_frame, count: int, current_values: list,
                                  visible_widgets: Dict):
        """Rebuild a checkbutton_list widget with new count."""
        default_value = bool(widget_spec.get("default", False))
        item_labels = widget_spec.get("item_labels", [])
        item_label_prefix = widget_spec.get("item_label_prefix", "")

        bool_values = [bool(v) for v in current_values] if isinstance(current_values, list) else []
        check_vars = []

        def on_checkbutton_list_change(*_args, vars_ref=check_vars, a=func_alias, n=field_name, vw=visible_widgets):
            self._save_widget_value(a, n, [var.get() for var in vars_ref])
            self._update_field_visibility(a, vw)

        for i in range(count):
            initial_value = bool_values[i] if i < len(bool_values) else default_value
            var = tk.BooleanVar(value=initial_value)

            if i < len(item_labels):
                item_text = str(item_labels[i])
            elif item_label_prefix:
                item_text = f"{item_label_prefix} {i + 1}"
            else:
                item_text = f"[{i + 1}]"

            check = ttk.Checkbutton(list_frame, text=item_text, variable=var)
            check.pack(side=tk.LEFT, padx=(0, 10), pady=(0, 2))
            var.trace_add("write", on_checkbutton_list_change)
            check_vars.append(var)

        # Save initial values
        self._save_widget_value(func_alias, field_name, [var.get() for var in check_vars])

        # Update widget_data
        widget_data["widget"] = check_vars

    def _rebuild_sample_paths_list(self, func_alias: str, field_name: str, widget_data: Dict,
                                    widget_spec: Dict, list_frame, count: int, current_values: list,
                                    visible_widgets: Dict):
        """Rebuild a sample_paths_list widget with new count."""
        label_text = widget_spec.get("label", field_name)
        
        sample_widgets = []
        sample_item_frames = []
        
        for i in range(count):
            item_frame = ttk.Frame(list_frame)
            item_frame.pack(anchor=tk.W, pady=(0, 4), fill=tk.X)
            sample_item_frames.append(item_frame)
            
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

            # Persist while typing so switching methodology items does not drop unsaved text.
            def on_sample_key_release(event, idx=i, f_widget=file_entry, widgets=sample_widgets, n=field_name, a=func_alias):
                text = f_widget.get()
                if text.strip():
                    parsed_files = [f.strip() for f in text.split(';') if f.strip()]
                    widgets[idx]["files"] = parsed_files
                else:
                    widgets[idx]["files"] = []
                values_list = [w["files"] for w in widgets]
                self._save_widget_value(a, n, values_list)
            file_entry.bind("<KeyRelease>", on_sample_key_release)
        
        # Save initial values
        initial_values = [w["files"] for w in sample_widgets]
        self._save_widget_value(func_alias, field_name, initial_values)
        
        # Update widget_data
        widget_data["widget"] = sample_widgets
        widget_data["item_frames"] = sample_item_frames

        self._apply_dynamic_list_item_visibility(func_alias, field_name, widget_data, visible_widgets)

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
        left_frame = tk.LabelFrame(
            selection_frame,
            text=self.language_manager.translate("ui.messages.output", "Output"),
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
            bd=1,
            relief=tk.GROOVE,
            labelanchor="n",
        )
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.output_func_var = tk.StringVar(value="--")
        self.output_func_combo = ttk.Combobox(left_frame, textvariable=self.output_func_var, 
                                              state="readonly", width=30)
        self.output_func_combo.pack(fill=tk.X)
        self.output_func_combo.bind("<<ComboboxSelected>>", lambda e: self._on_output_func_selected())
        
        # Right side - Input function selection
        right_frame = tk.LabelFrame(
            selection_frame,
            text=self.language_manager.translate("ui.messages.input", "Input"),
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
            bd=1,
            relief=tk.GROOVE,
            labelanchor="n",
        )
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

        self.routing_status_var = tk.StringVar(value="")
        ttk.Label(
            button_frame,
            textvariable=self.routing_status_var,
            foreground="#1f6aa5"
        ).pack(side=tk.LEFT, padx=(12, 0))
        
        # Canvas area for visual connections
        canvas_frame = tk.LabelFrame(
            routing_frame,
            text=self.language_manager.translate("ui.messages.connections", "Connections"),
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
            bd=1,
            relief=tk.GROOVE,
            labelanchor="n",
        )
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Canvas with scrollbar
        self.routing_canvas = tk.Canvas(
            canvas_frame,
            bg=self._get_theme_background_color(),
            highlightthickness=0,
            height=400,
        )
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.routing_canvas.yview)
        self.routing_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.routing_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel to canvas for vertical scrolling
        def _on_mousewheel(event):
            step = int(-1 * (event.delta / 120))
            if step == 0:
                return

            first, last = self.routing_canvas.yview()
            if (step < 0 and first <= 0.0) or (step > 0 and last >= 1.0):
                return

            self.routing_canvas.yview_scroll(step, "units")
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
            outputs = self._get_output_aliases_for_instance(output_idx)
            passforward_output_keys = set(self._get_passforward_output_aliases(instance_alias, base_alias).keys())
            
            # Draw label centered above output buttons
            self.routing_canvas.create_text(left_x + 65, current_y, text=self.language_manager.translate("ui.messages.outputs", "Outputs"), font=("Arial", 11, "bold"), anchor="center")
            current_y += 25
            
            # Draw output buttons
            for param_key, param_name in outputs.items():
                # Check if this button is selected
                is_selected = self.selected_button is not None and \
                             self.selected_button == (output_idx, True, param_key, param_name)
                is_passforward_output = param_key in passforward_output_keys

                if is_passforward_output:
                    normal_bg = "#e3620c"
                    selected_bg = "#c15614"
                    active_bg = "#7c2d12"
                else:
                    normal_bg = "#dc2626"
                    selected_bg = "#a82434"
                    active_bg = "#7f1d1d"
                
                # Create button with smaller height for better layout
                button = tk.Button(
                    self.routing_canvas,
                    text=f"{param_name} ►",
                    font=("Arial", 9, "bold"),
                    width=16,
                    height=1,
                    bg=selected_bg if is_selected else normal_bg,
                    fg="white",
                    activebackground=active_bg if is_selected else selected_bg,
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
        bbox = self.routing_canvas.bbox("all")
        if bbox is not None:
            self.routing_canvas.configure(scrollregion=bbox)

            content_height = bbox[3] - bbox[1]
            viewport_height = max(1, self.routing_canvas.winfo_height())
            if content_height <= viewport_height:
                self.routing_canvas.yview_moveto(0.0)
        
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
            self._set_routing_status(
                self.language_manager.translate(
                    "ui.messages.routing_connection_removed",
                    "Connection removed"
                )
            )
        else:
            # Enforce one active incoming route per destination input.
            removed_count = self._remove_routes_to_destination(dst_idx, dst_param_key)

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
                "src_nested_key": self.routing_nested_key_var.get().strip() if hasattr(self, "routing_nested_key_var") else "",
                "auto_created": False
            }

            if removed_count > 0:
                self._set_routing_status(
                    self.language_manager.translate(
                        "ui.messages.routing_connection_replaced",
                        "Connection replaced: destination input now uses the selected source"
                    )
                )
            else:
                self._set_routing_status(
                    self.language_manager.translate(
                        "ui.messages.routing_connection_created",
                        "Connection created"
                    )
                )
        
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
        RoutingMapWindow = self._get_routing_map_window_class()
        RoutingMapWindow(
            self.root,
            self.methodology_list,
            self.function_base_aliases,
            self.routing_lines,
            self.gui_configs,
            FUNCTION_SPECS,
            self.function_configs,
        )

    def _position_paned_sash(self, paned, orient=tk.HORIZONTAL):
        """Position the first PanedWindow sash to the middle for its orientation."""
        try:
            # Force window to update first
            paned.update_idletasks()
            if orient == tk.VERTICAL:
                parent_size = paned.winfo_height()
            else:
                parent_size = paned.winfo_width()

            if parent_size > 1:  # Only set if window has been rendered
                sash_pos = parent_size // 2
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

    def _ensure_custom_analysis_state(self) -> dict:
        """Ensure custom analysis state is initialized and registered under analysis_data."""
        if not hasattr(self, 'custom_analysis_data') or not isinstance(self.custom_analysis_data, dict):
            self.custom_analysis_data = {
                'pages': [],
                'current_page': 0,
                'active_sections': {}
            }

        self.custom_analysis_data.setdefault('pages', [])
        self.custom_analysis_data.setdefault('current_page', 0)
        self.custom_analysis_data.setdefault('active_sections', {})

        if not hasattr(self, 'analysis_data') or not isinstance(self.analysis_data, dict):
            self.analysis_data = {}

        self.analysis_data[self.CUSTOM_ANALYSIS_ALIAS] = self.custom_analysis_data
        return self.custom_analysis_data

    def _show_custom_analysis_tab(self):
        """Show global Custom Analysis tab that reuses analysis page/navigation structure."""
        self._clear_tab()
        self.current_tab = "custom_analysis"

        analysis_info = self._ensure_custom_analysis_state()
        self._ensure_analysis_section_styles()

        control_frame = ttk.Frame(self.tab_content_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        title = ttk.Label(
            control_frame,
            text=self.language_manager.translate("ui.tabs.custom_analysis", "C. Analysis"),
            font=("Arial", 11, "bold")
        )
        title.pack(side=tk.LEFT, padx=5)

        add_text_btn = ttk.Button(
            control_frame,
            text=self.language_manager.translate("ui.buttons.add_text", "Add text"),
            command=self._show_custom_add_text_dialog
        )
        add_text_btn.pack(side=tk.LEFT, padx=5)

        remove_section_btn = ttk.Button(
            control_frame,
            text=self.language_manager.translate("ui.buttons.remove_section", "Remove section"),
            command=lambda: self._show_remove_section_dialog(self.CUSTOM_ANALYSIS_ALIAS)
        )
        remove_section_btn.pack(side=tk.LEFT, padx=5)

        spacer = ttk.Frame(control_frame)
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)

        add_page_btn = ttk.Button(
            control_frame,
            text=self.language_manager.translate("ui.buttons.add_page", "Add Page"),
            command=lambda: self._show_add_page_dialog(self.CUSTOM_ANALYSIS_ALIAS)
        )
        add_page_btn.pack(side=tk.RIGHT, padx=5)

        remove_page_btn = ttk.Button(
            control_frame,
            text=self.language_manager.translate("ui.buttons.remove_page", "Remove Page"),
            command=lambda: self._remove_current_page(self.CUSTOM_ANALYSIS_ALIAS)
        )
        remove_page_btn.pack(side=tk.RIGHT, padx=5)

        pages = analysis_info.get('pages', [])
        visible_pages = [(idx, page) for idx, page in enumerate(pages)]

        current_page = analysis_info.get('current_page', 0)
        visible_page_idx = 0
        for i, (idx, _page) in enumerate(visible_pages):
            if idx == current_page:
                visible_page_idx = i
                break

        if visible_pages:
            resolved_current_page_idx = visible_pages[visible_page_idx][0]
            analysis_info['current_page'] = resolved_current_page_idx

        nav_frame = ttk.Frame(self.tab_content_frame)
        nav_frame.pack(fill=tk.X, padx=10, pady=(0, 2), side=tk.BOTTOM)

        left_nav_col = ttk.Frame(nav_frame)
        left_nav_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        left_nav_col.pack_propagate(False)

        right_nav_col = ttk.Frame(nav_frame)
        right_nav_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(12, 0))
        right_nav_col.pack_propagate(False)

        self.analysis_data[self.CUSTOM_ANALYSIS_ALIAS]['page_nav_host'] = left_nav_col
        self.analysis_data[self.CUSTOM_ANALYSIS_ALIAS]['active_nav_host'] = right_nav_col
        self._update_analysis_bottom_split(self.CUSTOM_ANALYSIS_ALIAS)
        self.root.after_idle(lambda: self._update_analysis_bottom_split(self.CUSTOM_ANALYSIS_ALIAS))

        current_page_data_for_layout = visible_pages[visible_page_idx][1] if visible_pages else None
        if current_page_data_for_layout is not None:
            self._update_active_nav_host_geometry(self.CUSTOM_ANALYSIS_ALIAS, current_page_data_for_layout)

        left_nav_stack = ttk.Frame(left_nav_col)
        left_nav_stack.pack(side=tk.BOTTOM, anchor='sw', fill=tk.X)

        page_nav_row = ttk.Frame(left_nav_stack)
        page_nav_row.pack(side=tk.BOTTOM, fill=tk.X)

        if visible_pages:
            current_idx, current_page_data = visible_pages[visible_page_idx]
            page_title = current_page_data.get('title', f'Page {current_idx + 1}')
            page_info = f"Page {visible_page_idx + 1}/{len(visible_pages)}: {page_title}"
        else:
            page_info = self.language_manager.translate("ui.messages.no_pages_available", "No pages available")

        prev_btn = ttk.Button(
            page_nav_row,
            text=_ui_symbol("prev") + " " + self.language_manager.translate("ui.buttons.previous", "Previous"),
            width=10,
            command=lambda: self._switch_analysis_page_relative(self.CUSTOM_ANALYSIS_ALIAS, -1)
        )
        prev_btn.pack(side=tk.LEFT, padx=2)

        next_btn = ttk.Button(
            page_nav_row,
            text=self.language_manager.translate("ui.buttons.next", "Next") + " " + _ui_symbol("next"),
            width=10,
            command=lambda: self._switch_analysis_page_relative(self.CUSTOM_ANALYSIS_ALIAS, 1)
        )
        next_btn.pack(side=tk.LEFT, padx=2)

        if visible_pages:
            ttk.Label(
                page_nav_row,
                text=self.language_manager.translate("ui.messages.page_selector", "Page:"),
                font=("Arial", 9)
            ).pack(side=tk.LEFT, padx=(10, 6))

            page_options = []
            for display_idx, (page_idx, page_data) in enumerate(visible_pages):
                option_title = page_data.get('title', f'Page {page_idx + 1}')
                page_options.append(f"{display_idx + 1}/{len(visible_pages)}: {option_title}")

            page_combo = ttk.Combobox(page_nav_row, state="readonly", values=page_options, width=59)
            page_combo.current(visible_page_idx)
            page_combo.bind(
                "<<ComboboxSelected>>",
                lambda e, cb=page_combo, vp=visible_pages: self._on_analysis_page_selected(self.CUSTOM_ANALYSIS_ALIAS, cb, vp)
            )
            page_combo.pack(side=tk.LEFT, padx=4)
        else:
            page_label = ttk.Label(page_nav_row, text=page_info, font=("Arial", 9))
            page_label.pack(side=tk.LEFT, padx=10)

        content_frame = ttk.Frame(self.tab_content_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, side=tk.TOP)

        overlay_visible = self._show_analysis_loading_overlay(nav_frame)
        if overlay_visible:
            try:
                self.root.update_idletasks()
            except Exception:
                pass

        try:
            if visible_pages:
                _current_idx, page_data = visible_pages[visible_page_idx]
                self._render_analysis_page(content_frame, self.CUSTOM_ANALYSIS_ALIAS, page_data)
            else:
                empty_label = ttk.Label(
                    content_frame,
                    text=self.language_manager.translate("ui.messages.no_pages_available", "No pages available"),
                    font=("Arial", 10, "italic")
                )
                empty_label.pack(expand=True)
        finally:
            if overlay_visible:
                self._hide_analysis_loading_overlay()

    
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
                    'current_page': analysis_config.get('current_page', 0),
                    'active_sections': {}
                }
            else:
                # Fallback to default structure
                self.analysis_data[instance_alias] = {
                    'pages': [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}],
                    'current_page': 0,
                    'active_sections': {}
                }
        
        analysis_info = self.analysis_data[instance_alias]
        if 'active_sections' not in analysis_info:
            analysis_info['active_sections'] = {}
        self._ensure_analysis_result_selection(instance_alias)
        analysis_info = self.analysis_data[instance_alias]
        self._ensure_analysis_section_styles()
        
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

        # Add text button
        add_text_btn = ttk.Button(control_frame, text=self.language_manager.translate("ui.buttons.add_text", "Add text"),
                      command=lambda: self._show_add_text_dialog(instance_alias))
        add_text_btn.pack(side=tk.LEFT, padx=5)
        
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
        
        # Only skip condition filtering when there is no execution payload at all.
        # Runs where setup defaults were never persisted can still have empty inputs.
        execution_results = analysis_info.get('execution_results')
        has_execution_results = isinstance(execution_results, dict) and bool(execution_results)
        
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

        if visible_pages:
            resolved_current_page_idx = visible_pages[visible_page_idx][0]
            analysis_info['current_page'] = resolved_current_page_idx
        
        # Navigation frame (bottom): page navigation row + result navigation row
        nav_frame = ttk.Frame(self.tab_content_frame)
        nav_frame.pack(fill=tk.X, padx=10, pady=(0, 2), side=tk.BOTTOM)

        left_nav_col = ttk.Frame(nav_frame)
        left_nav_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        left_nav_col.pack_propagate(False)

        right_nav_col = ttk.Frame(nav_frame)
        right_nav_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(12, 0))
        right_nav_col.pack_propagate(False)

        self.analysis_data[instance_alias]['page_nav_host'] = left_nav_col
        self.analysis_data[instance_alias]['active_nav_host'] = right_nav_col
        self._update_analysis_bottom_split(instance_alias)
        self.root.after_idle(lambda ia=instance_alias: self._update_analysis_bottom_split(ia))

        current_page_data_for_layout = visible_pages[visible_page_idx][1] if visible_pages else None
        if current_page_data_for_layout is not None:
            self._update_active_nav_host_geometry(instance_alias, current_page_data_for_layout)

        left_nav_stack = ttk.Frame(left_nav_col)
        left_nav_stack.pack(side=tk.BOTTOM, anchor='sw', fill=tk.X)

        page_nav_row = ttk.Frame(left_nav_stack)
        page_nav_row.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Page display label (first)
        if visible_pages:
            current_idx, current_page_data = visible_pages[visible_page_idx]
            page_title = current_page_data.get('title', f'Page {current_idx + 1}')
            page_info = f"Page {visible_page_idx + 1}/{len(visible_pages)}: {page_title}"
        else:
            page_info = self.language_manager.translate("ui.messages.no_pages_available", "No pages available")
        
        # Previous page button
        prev_btn = ttk.Button(page_nav_row, text=_ui_symbol("prev") + " " + self.language_manager.translate("ui.buttons.previous", "Previous"), width=10,
                     command=lambda: self._switch_analysis_page_relative(instance_alias, -1))
        prev_btn.pack(side=tk.LEFT, padx=2)
        
        # Next page button
        next_btn = ttk.Button(page_nav_row, text=self.language_manager.translate("ui.buttons.next", "Next") + " " + _ui_symbol("next"), width=10,
                             command=lambda: self._switch_analysis_page_relative(instance_alias, 1))
        next_btn.pack(side=tk.LEFT, padx=2)

        # Page selector in the title position for quick direct navigation.
        if visible_pages:
            ttk.Label(
                page_nav_row,
                text=self.language_manager.translate("ui.messages.page_selector", "Page:"),
                font=("Arial", 9)
            ).pack(side=tk.LEFT, padx=(10, 6))

            page_options = []
            for display_idx, (page_idx, page_data) in enumerate(visible_pages):
                option_title = page_data.get('title', f'Page {page_idx + 1}')
                page_options.append(f"{display_idx + 1}/{len(visible_pages)}: {option_title}")

            page_combo = ttk.Combobox(
                page_nav_row,
                state="readonly",
                values=page_options,
                width=59
            )
            page_combo.current(visible_page_idx)
            page_combo.bind(
                "<<ComboboxSelected>>",
                lambda e, ia=instance_alias, cb=page_combo, vp=visible_pages: self._on_analysis_page_selected(ia, cb, vp)
            )
            page_combo.pack(side=tk.LEFT, padx=4)
        else:
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
            result_nav_row = ttk.Frame(left_nav_stack)
            result_nav_row.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))

            result_prev_btn = ttk.Button(
                result_nav_row,
                text=_ui_symbol("prev") + " " + self.language_manager.translate("ui.buttons.previous", "Previous"),
                width=10,
                command=lambda: self._switch_analysis_result_relative(instance_alias, -1)
            )
            result_prev_btn.pack(side=tk.LEFT, padx=2)

            result_next_btn = ttk.Button(
                result_nav_row,
                text=self.language_manager.translate("ui.buttons.next", "Next") + " " + _ui_symbol("next"),
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

        overlay_visible = self._show_analysis_loading_overlay(nav_frame)
        if overlay_visible:
            try:
                self.root.update_idletasks()
            except Exception:
                pass
        
        # Display current page
        try:
            if visible_pages:
                current_idx, page_data = visible_pages[visible_page_idx]
                self._render_analysis_page(content_frame, instance_alias, page_data)
        finally:
            if overlay_visible:
                self._hide_analysis_loading_overlay()

    def _show_analysis_loading_overlay(self, bottom_anchor: tk.Widget = None) -> bool:
        """Display muted loading overlay above bottom page/nav controls."""
        try:
            if self.tab_content_frame is None or not self.tab_content_frame.winfo_exists():
                return False
        except Exception:
            return False

        style = ttk.Style()
        overlay_bg = (
            style.lookup("TLabelframe", "bordercolor")
            or style.lookup("TFrame", "background")
            or self.tab_content_frame.cget("background")
            or "#d9d9d9"
        )
        overlay_fg = style.lookup("TLabel", "foreground") or "black"

        overlay = getattr(self, '_analysis_loading_overlay', None)
        overlay_exists = False
        if overlay is not None:
            try:
                overlay_exists = bool(overlay.winfo_exists())
            except Exception:
                overlay_exists = False

        if overlay_exists:
            try:
                overlay_exists = str(overlay.master) == str(self.tab_content_frame)
            except Exception:
                overlay_exists = False

        label = getattr(self, '_analysis_loading_label', None)
        label_exists = False
        if label is not None:
            try:
                label_exists = bool(label.winfo_exists())
            except Exception:
                label_exists = False

        if not overlay_exists:
            self._analysis_loading_overlay = tk.Frame(self.tab_content_frame, bg=overlay_bg, bd=0, highlightthickness=0)
            self._analysis_loading_label = tk.Label(
                self._analysis_loading_overlay,
                text=self.language_manager.translate("ui.messages.loading", "Loading..."),
                bg=overlay_bg,
                fg=overlay_fg,
                font=("Arial", 11, "bold")
            )
            self._analysis_loading_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self._analysis_loading_overlay.configure(bg=overlay_bg)
            if not label_exists:
                self._analysis_loading_label = tk.Label(
                    self._analysis_loading_overlay,
                    text=self.language_manager.translate("ui.messages.loading", "Loading..."),
                    bg=overlay_bg,
                    fg=overlay_fg,
                    font=("Arial", 11, "bold")
                )
                self._analysis_loading_label.place(relx=0.5, rely=0.5, anchor="center")
            else:
                self._analysis_loading_label.configure(
                text=self.language_manager.translate("ui.messages.loading", "Loading..."),
                bg=overlay_bg,
                fg=overlay_fg
                )

        self._position_analysis_loading_overlay(bottom_anchor)
        self._analysis_loading_overlay.lift()
        return True

    def _position_analysis_loading_overlay(self, bottom_anchor: tk.Widget = None):
        """Position loading overlay to only cover area above bottom controls."""
        try:
            self.tab_content_frame.update_idletasks()
        except Exception:
            pass

        total_height = 0
        try:
            total_height = int(self.tab_content_frame.winfo_height())
        except Exception:
            total_height = 0

        anchor_y = 0
        if bottom_anchor is not None:
            try:
                if bottom_anchor.winfo_exists():
                    anchor_y = int(bottom_anchor.winfo_y())
            except Exception:
                anchor_y = 0

        cover_height = anchor_y if anchor_y > 1 else total_height
        if cover_height <= 1:
            cover_height = 400

        try:
            self._analysis_loading_overlay.place(x=0, y=0, relwidth=1, height=cover_height)
        except Exception:
            pass

    def _hide_analysis_loading_overlay(self):
        """Hide analysis loading overlay if currently displayed."""
        overlay = getattr(self, '_analysis_loading_overlay', None)
        if overlay is None:
            return
        try:
            if overlay.winfo_exists():
                overlay.place_forget()
        except Exception:
            pass

    def _ensure_analysis_section_styles(self):
        """Ensure deterministic ttk styles for active/inactive section outlines."""
        if getattr(self, '_analysis_section_styles_initialized', False):
            return

        self._analysis_section_active_border = "#b5b5b5"

        style = ttk.Style()

        # Sample base Labelframe bevel colors from the active theme for consistency.
        base_border = style.lookup("TLabelframe", "bordercolor") or "#dedede"
        base_light = style.lookup("TLabelframe", "lightcolor") or "#eaeaea"
        base_dark = style.lookup("TLabelframe", "darkcolor") or "#eeeeee"

        self._analysis_section_inactive_border = base_border
        self._analysis_section_inactive_light = base_light
        self._analysis_section_inactive_dark = base_dark

        # Keep active border darker while preserving light/dark bevel relationship.
        self._analysis_section_active_light = base_light
        self._analysis_section_active_dark = base_dark

        # Build section-specific styles from clam border element to reduce theme dependency.
        try:
            style.element_create("AnalysisSectionInactive.border", "from", "clam", "Labelframe.border")
        except tk.TclError:
            pass
        try:
            style.element_create("AnalysisSectionActive.border", "from", "clam", "Labelframe.border")
        except tk.TclError:
            pass

        def _replace_layout_element(layout_spec, old_element: str, new_element: str):
            replaced = []
            for element, options in layout_spec:
                elem_name = new_element if element == old_element else element
                opts = dict(options)
                if 'children' in opts and isinstance(opts['children'], list):
                    opts['children'] = _replace_layout_element(opts['children'], old_element, new_element)
                replaced.append((elem_name, opts))
            return replaced

        try:
            base_layout = style.layout("TLabelframe")
            if base_layout:
                inactive_layout = _replace_layout_element(copy.deepcopy(base_layout), "Labelframe.border", "AnalysisSectionInactive.border")
                active_layout = _replace_layout_element(copy.deepcopy(base_layout), "Labelframe.border", "AnalysisSectionActive.border")
                style.layout("AnalysisSectionInactive.TLabelframe", inactive_layout)
                style.layout("AnalysisSectionActive.TLabelframe", active_layout)
        except Exception:
            pass

        style.configure(
            "AnalysisSectionInactive.TLabelframe",
            bordercolor=self._analysis_section_inactive_border,
            lightcolor=self._analysis_section_inactive_light,
            darkcolor=self._analysis_section_inactive_dark,
            borderwidth=1,
            relief=tk.GROOVE
        )
        style.configure(
            "AnalysisSectionActive.TLabelframe",
            bordercolor=self._analysis_section_active_border,
            lightcolor=self._analysis_section_active_light,
            darkcolor=self._analysis_section_active_dark,
            borderwidth=1,
            relief=tk.GROOVE
        )

        self._analysis_section_styles_initialized = True

    def _bind_section_activation(self, widget: tk.Widget, instance_alias: str, section_id: tuple):
        """Bind click events so any click inside the section activates it."""
        try:
            widget.bind(
                "<Button-1>",
                lambda e, ia=instance_alias, sid=section_id: self._on_analysis_section_clicked(ia, sid),
                add=True
            )
            for child in widget.winfo_children():
                self._bind_section_activation(child, instance_alias, section_id)
        except Exception:
            pass

    def _on_analysis_section_clicked(self, instance_alias: str, section_id: tuple):
        """Handle section click and update active section state."""
        self._set_active_analysis_section(instance_alias, section_id)

    def _set_active_analysis_section(self, instance_alias: str, section_id: tuple):
        """Persist and apply active section for current function/page in session."""
        if instance_alias not in self.analysis_data:
            return

        analysis_info = self.analysis_data[instance_alias]
        page_idx, section_idx = section_id
        if 'active_sections' not in analysis_info:
            analysis_info['active_sections'] = {}

        if analysis_info['active_sections'].get(page_idx) == section_idx:
            return

        analysis_info['active_sections'][page_idx] = section_idx
        if instance_alias == self.CUSTOM_ANALYSIS_ALIAS:
            self._sync_custom_active_section_execution_results()
        self._apply_analysis_section_styles(instance_alias)
        self._update_active_nav_host_geometry(instance_alias)
        self._render_active_section_navigation(instance_alias)

    def _update_analysis_bottom_split(self, instance_alias: str):
        """Apply fixed 60/40 width split for page controls and active navigation host."""
        analysis_info = self.analysis_data.get(instance_alias, {})
        left_col = analysis_info.get('page_nav_host')
        right_col = analysis_info.get('active_nav_host')
        if left_col is None or right_col is None:
            return

        try:
            if not left_col.winfo_exists() or not right_col.winfo_exists():
                return
        except Exception:
            return

        try:
            self.tab_content_frame.update_idletasks()
            total_width = self.tab_content_frame.winfo_width()
        except Exception:
            total_width = 0

        if total_width <= 1:
            total_width = 1200

        spacing = 12
        usable = max(320, total_width - 20 - spacing)
        right_width = int(usable * 0.40)
        left_width = usable - right_width

        try:
            left_col.configure(width=left_width)
            right_col.configure(width=right_width)
        except tk.TclError:
            return
        except Exception:
            return

    def _get_min_bottom_controls_height(self) -> int:
        """Minimum height needed for page/cycle controls block."""
        return 64

    def _set_bottom_hosts_height(self, instance_alias: str, height: int):
        """Apply the same reserved bottom height to left and right nav hosts."""
        analysis_info = self.analysis_data.get(instance_alias, {})
        nav_host = analysis_info.get('active_nav_host')
        page_nav_host = analysis_info.get('page_nav_host')

        if nav_host is not None:
            try:
                nav_host.configure(height=height)
            except Exception:
                pass

        if page_nav_host is not None:
            try:
                page_nav_host.configure(height=height)
            except Exception:
                pass

    def _get_cached_page_nav_height(self, instance_alias: str, page_idx: int) -> int:
        analysis_info = self.analysis_data.get(instance_alias, {})
        cache = analysis_info.get('page_nav_heights', {})
        return int(cache.get(page_idx, 0) or 0)

    def _set_cached_page_nav_height(self, instance_alias: str, page_idx: int, height: int):
        if instance_alias not in self.analysis_data:
            return
        analysis_info = self.analysis_data[instance_alias]
        if 'page_nav_heights' not in analysis_info:
            analysis_info['page_nav_heights'] = {}
        prev = int(analysis_info['page_nav_heights'].get(page_idx, 0) or 0)
        analysis_info['page_nav_heights'][page_idx] = max(prev, int(height))

    def _prime_page_nav_height_cache(self, instance_alias: str, page_idx: int, rendered_sections: list):
        """Precompute max Data Slicing panel height for all rendered sections on a page."""
        analysis_info = self.analysis_data.get(instance_alias, {})
        nav_host = analysis_info.get('active_nav_host')
        if nav_host is None:
            return

        try:
            if not nav_host.winfo_exists():
                return
        except Exception:
            return

        min_height = self._get_min_bottom_controls_height()
        max_required = min_height

        for entry in rendered_sections:
            section_data = entry.get('section_data', {}) or {}
            section_type = section_data.get('type')
            config = section_data.get('config', {}) or {}
            section_id = entry.get('section_id')
            if section_type not in ('graph', 'table') or not section_id:
                continue
            nav_axes = config.get('data_slicing', [])
            if not nav_axes:
                continue

            temp_panel = ttk.LabelFrame(nav_host, text=self.language_manager.translate("ui.labels.data_slicing", "Data Slicing"), padding=5)
            temp_panel.pack_forget()
            temp_panel.place(x=-10000, y=-10000)

            try:
                _render_alias, _execution_results, section_outputs = self._get_section_render_outputs(
                    instance_alias,
                    section_id,
                    section_data=section_data,
                )
                self._clear_navigation_labels_for_section(instance_alias, section_id)
                if section_type == 'graph':
                    slice_state = analysis_info.get('graph_slices', {}).get(section_id)
                    if slice_state:
                        self._create_navigation_controls(
                            temp_panel,
                            instance_alias,
                            section_id,
                            section_outputs,
                            config,
                            slice_state,
                            include_widget_refs=False,
                        )
                else:
                    slice_state = analysis_info.get('table_slices', {}).get(section_id)
                    if slice_state:
                        self._create_table_navigation_controls(
                            temp_panel,
                            instance_alias,
                            section_id,
                            section_outputs,
                            config,
                            slice_state,
                            include_widget_refs=False,
                        )

                temp_panel.update_idletasks()
                required = temp_panel.winfo_reqheight() + 8
                if required > max_required:
                    max_required = required
            except Exception:
                pass
            finally:
                try:
                    temp_panel.destroy()
                except Exception:
                    pass

        self._set_cached_page_nav_height(instance_alias, page_idx, max_required)
        stable_height = max(self._get_cached_page_nav_height(instance_alias, page_idx), min_height)
        self._set_bottom_hosts_height(instance_alias, stable_height)

    def _estimate_section_navigation_footprint(self, instance_alias: str, section_entry: dict, outputs: dict) -> tuple:
        """Estimate (rows, items_per_row) needed by a section navigation UI."""
        rows = 0
        items_per_row = 0

        section_data = section_entry.get('section_data', {}) if isinstance(section_entry, dict) else {}
        section_type = section_data.get('type')
        config = section_data.get('config', {}) if isinstance(section_data, dict) else {}
        nav_axes = config.get('data_slicing', []) if isinstance(config, dict) else []

        if section_type not in ('graph', 'table') or not nav_axes:
            return rows, items_per_row

        if section_type == 'table':
            visible_nav = 0
            for nav_item in nav_axes:
                if isinstance(nav_item, dict):
                    if nav_item.get('show_navigation_menu', True):
                        visible_nav += 1
                else:
                    visible_nav += 1
            if visible_nav > 0:
                rows += visible_nav
                items_per_row = max(items_per_row, 1)
            return rows, items_per_row

        # Graph footprint
        visible_nav = 0
        for nav_item in nav_axes:
            if isinstance(nav_item, dict):
                if nav_item.get('show_navigation_menu', False):
                    visible_nav += 1
            else:
                visible_nav += 1
        if visible_nav > 0:
            rows += visible_nav
            items_per_row = max(items_per_row, 1)

        # MD controls footprint (if applicable)
        if config.get('show_md_menu', False):
            data_source = None
            nested_key = None
            graph_type = config.get('graph_type', '')
            if 'aux_axis' in config:
                data_source = config.get('z_axis', {}).get('data_source')
                nested_key = config.get('z_axis', {}).get('nested_key')
            elif graph_type in ('heatmap', 'contour', '3d_surf'):
                data_source = config.get('z_axis', {}).get('data_source')
                nested_key = config.get('z_axis', {}).get('nested_key')
            elif 'datasets' in config:
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
                axis_config = config.get('y_axis', {}) or config.get('x_axis', {})
                data_source = axis_config.get('data_source')
                nested_key = axis_config.get('nested_key')

            data = self._get_data_from_source(outputs, data_source, nested_key) if data_source else None
            if data is not None and not isinstance(data, np.ndarray):
                try:
                    data = np.array(data)
                except (ValueError, TypeError):
                    data = None

            if isinstance(data, np.ndarray) and data.ndim >= 4:
                specified_dims = set()
                for nav_item in nav_axes:
                    if isinstance(nav_item, dict):
                        dim = nav_item.get('dimension')
                        if dim is not None:
                            specified_dims.add(dim)
                md_combinations = self._compute_dimension_combinations(data.shape, specified_dims, 2)
                if md_combinations:
                    rows += 1  # combobox row
                    # Reserve using worst-case navigable dimensions across combinations
                    all_dims = set(range(len(data.shape)))
                    max_navigable_dims = 0
                    for combo in md_combinations:
                        navigable_dims = all_dims - set(combo) - specified_dims
                        max_navigable_dims = max(max_navigable_dims, len(navigable_dims))
                    if max_navigable_dims > 0:
                        rows += max_navigable_dims
                        items_per_row = max(items_per_row, 1)

        return rows, items_per_row

    def _update_active_nav_host_geometry(self, instance_alias: str, page_data: Optional[dict] = None):
        """Reserve nav host space using the largest navigation needs on current page."""
        analysis_info = self.analysis_data.get(instance_alias)
        if not analysis_info:
            return

        nav_host = analysis_info.get('active_nav_host')
        page_nav_host = analysis_info.get('page_nav_host')
        if nav_host is None:
            return

        try:
            if not nav_host.winfo_exists():
                return
        except Exception:
            return

        if page_data is None:
            current_page = analysis_info.get('current_page', 0)
            pages = analysis_info.get('pages', [])
            if current_page < 0 or current_page >= len(pages):
                return
            page_data = pages[current_page]
        else:
            current_page = analysis_info.get('current_page', 0)

        sections = page_data.get('sections', []) if isinstance(page_data, dict) else []
        section_entries = []
        for section_idx, section_data in enumerate(sections):
            if not isinstance(section_data, dict):
                continue
            if section_data.get('type') is None:
                continue
            if section_data.get('condition'):
                if not self._evaluate_condition(instance_alias, section_data.get('condition')):
                    continue
            section_entries.append({
                'section_id': (current_page, section_idx),
                'section_data': section_data
            })

        has_md_menu = False

        max_rows = 0
        max_items = 0
        for entry in section_entries:
            cfg = (entry.get('section_data', {}) or {}).get('config', {}) or {}
            if cfg.get('show_md_menu', False):
                has_md_menu = True
            section_id = entry.get('section_id')
            section_data = entry.get('section_data')
            _render_alias, _execution_results, section_outputs = self._get_section_render_outputs(
                instance_alias,
                section_id,
                section_data=section_data,
            )
            rows, items = self._estimate_section_navigation_footprint(instance_alias, entry, section_outputs)
            max_rows = max(max_rows, rows)
            max_items = max(max_items, items)

        # Ensure left-side page/result controls always fit within reserved bottom height.
        history_entries = analysis_info.get('execution_history', [])
        min_left_height = self._get_min_bottom_controls_height()

        if max_rows <= 0:
            nav_host.configure(width=1, height=min_left_height)
            cached_height = self._get_cached_page_nav_height(instance_alias, current_page)
            stable_height = max(min_left_height, cached_height)
            self._set_cached_page_nav_height(instance_alias, current_page, stable_height)
            self._set_bottom_hosts_height(instance_alias, stable_height)
            analysis_info['active_nav_host_width'] = 1
            analysis_info['active_nav_wrap_columns'] = 1
            return

        self._update_analysis_bottom_split(instance_alias)

        try:
            nav_host.update_idletasks()
            width = nav_host.winfo_width()
        except Exception:
            width = 0
        if width <= 1:
            width = 420 if has_md_menu else 320

        wrap_columns = 1
        total_rows = max_rows
        if has_md_menu:
            base_height = 34
            row_height = 38
        else:
            base_height = 16
            row_height = 38
        height = base_height + total_rows * row_height
        height = max(height, min_left_height)

        nav_host.configure(width=width, height=height)
        cached_height = self._get_cached_page_nav_height(instance_alias, current_page)
        stable_height = max(height, min_left_height, cached_height)
        self._set_cached_page_nav_height(instance_alias, current_page, stable_height)
        self._set_bottom_hosts_height(instance_alias, stable_height)
        analysis_info['active_nav_host_width'] = width
        analysis_info['active_nav_wrap_columns'] = wrap_columns

    def _apply_analysis_section_styles(self, instance_alias: str):
        """Apply active/inactive border styles to rendered section containers."""
        if instance_alias not in self.analysis_data:
            return

        analysis_info = self.analysis_data[instance_alias]
        current_page = analysis_info.get('current_page', 0)
        active_sections = analysis_info.get('active_sections', {})
        active_section_idx = active_sections.get(current_page)

        rendered_sections = analysis_info.get('rendered_sections', [])
        for entry in rendered_sections:
            container = entry.get('container')
            section_id = entry.get('section_id')
            if not container or not section_id:
                continue
            _, section_idx = section_id
            style_name = "AnalysisSectionActive.TLabelframe" if section_idx == active_section_idx else "AnalysisSectionInactive.TLabelframe"
            try:
                container.configure(style=style_name)
            except Exception:
                pass

    def _clear_navigation_labels_for_section(self, instance_alias: str, section_id: tuple):
        """Remove stale navigation label references for a section."""
        if hasattr(self, '_nav_labels'):
            keys_to_remove = [k for k in self._nav_labels.keys() if k[0] == instance_alias and k[1] == section_id]
            for key in keys_to_remove:
                del self._nav_labels[key]

        if hasattr(self, '_md_nav_labels'):
            keys_to_remove = [k for k in self._md_nav_labels.keys() if k[0] == instance_alias and k[1] == section_id]
            for key in keys_to_remove:
                del self._md_nav_labels[key]

        if hasattr(self, '_table_nav_labels'):
            keys_to_remove = [k for k in self._table_nav_labels.keys() if k[0] == instance_alias and k[1] == section_id]
            for key in keys_to_remove:
                del self._table_nav_labels[key]

        if hasattr(self, '_var_labels'):
            keys_to_remove = [k for k in self._var_labels.keys() if k[0] == instance_alias and k[1] == section_id]
            for key in keys_to_remove:
                del self._var_labels[key]

    def _render_active_section_navigation(self, instance_alias: str):
        """Render data slicing navigation for the active section in bottom-right host."""
        if instance_alias not in self.analysis_data:
            return

        analysis_info = self.analysis_data[instance_alias]
        nav_host = analysis_info.get('active_nav_host')
        if nav_host is None:
            return

        try:
            if not nav_host.winfo_exists():
                return
        except Exception:
            return

        for widget in nav_host.winfo_children():
            widget.destroy()

        self._update_active_nav_host_geometry(instance_alias)

        current_page = analysis_info.get('current_page', 0)
        stable_page_height = max(
            self._get_min_bottom_controls_height(),
            self._get_cached_page_nav_height(instance_alias, current_page)
        )
        active_sections = analysis_info.get('active_sections', {})
        active_section_idx = active_sections.get(current_page)
        if active_section_idx is None:
            self._set_bottom_hosts_height(instance_alias, stable_page_height)
            return

        rendered_sections = analysis_info.get('rendered_sections', [])
        active_entry = None
        for entry in rendered_sections:
            section_id = entry.get('section_id')
            if section_id == (current_page, active_section_idx):
                active_entry = entry
                break

        if not active_entry:
            self._set_bottom_hosts_height(instance_alias, stable_page_height)
            return

        section_data = active_entry.get('section_data', {}) or {}
        section_type = section_data.get('type')
        config = section_data.get('config', {}) or {}
        section_id = active_entry.get('section_id')
        if not section_id:
            return

        if section_type not in ('graph', 'table'):
            self._set_bottom_hosts_height(instance_alias, stable_page_height)
            return

        nav_axes = config.get('data_slicing', [])
        if not nav_axes:
            self._set_bottom_hosts_height(instance_alias, stable_page_height)
            return

        nav_panel = ttk.LabelFrame(
            nav_host,
            text=self.language_manager.translate("ui.labels.data_slicing", "Data Slicing"),
            padding=5
        )
        nav_panel.pack(side=tk.BOTTOM, fill=tk.X, anchor='sw')

        _render_alias, _execution_results, outputs = self._get_section_render_outputs(
            instance_alias,
            section_id,
            section_data=section_data,
        )

        self._clear_navigation_labels_for_section(instance_alias, section_id)

        if section_type == 'graph':
            slice_state = analysis_info.get('graph_slices', {}).get(section_id)
            if slice_state:
                self._create_navigation_controls(nav_panel, instance_alias, section_id, outputs, config, slice_state)
        elif section_type == 'table':
            slice_state = analysis_info.get('table_slices', {}).get(section_id)
            if slice_state:
                self._create_table_navigation_controls(nav_panel, instance_alias, section_id, outputs, config, slice_state)

        # Final height sync based on actual rendered navigation content.
        try:
            nav_panel.update_idletasks()
            required_height = nav_panel.winfo_reqheight() + 8
        except Exception:
            required_height = stable_page_height
        min_height = stable_page_height
        cached_height = self._get_cached_page_nav_height(instance_alias, current_page)
        stable_height = max(required_height, min_height, cached_height)
        self._set_cached_page_nav_height(instance_alias, current_page, stable_height)
        self._set_bottom_hosts_height(instance_alias, stable_height)
    
    def _switch_analysis_page(self, instance_alias: str, page_idx: int):
        """Switch to a different analysis page."""
        if instance_alias in self.analysis_data:
            self.analysis_data[instance_alias]['current_page'] = page_idx
            if instance_alias == self.CUSTOM_ANALYSIS_ALIAS:
                self._show_custom_analysis_tab()
            else:
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
        if instance_alias == self.CUSTOM_ANALYSIS_ALIAS:
            self._show_custom_analysis_tab()
        else:
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

    def _on_analysis_page_selected(
        self,
        instance_alias: str,
        combo_widget: ttk.Combobox,
        visible_pages: List[Tuple[int, Dict[str, Any]]]
    ):
        try:
            selected_visible_idx = combo_widget.current()
        except Exception:
            selected_visible_idx = -1
        if selected_visible_idx is None or selected_visible_idx < 0:
            return
        if selected_visible_idx >= len(visible_pages):
            return

        selected_page_idx = visible_pages[selected_visible_idx][0]
        if instance_alias in self.analysis_data:
            self.analysis_data[instance_alias]['current_page'] = selected_page_idx
            if instance_alias == self.CUSTOM_ANALYSIS_ALIAS:
                self._show_custom_analysis_tab()
            else:
                self._show_analysis_tab()

    def _get_setup_default_for_condition_parameter(self, instance_alias: str, parameter: Any) -> Tuple[bool, Any]:
        """Resolve a condition parameter to its setup default when execution inputs omitted it."""
        if not isinstance(parameter, str):
            return False, None

        param_name = parameter.strip()
        if not param_name:
            return False, None

        # Defaults are setup inputs; output-prefixed parameters do not have setup defaults.
        if param_name.startswith('out.') or param_name.startswith('pf.'):
            return False, None
        if param_name.startswith('in.'):
            param_name = param_name[3:]
        if not param_name or '.' in param_name:
            return False, None

        if instance_alias not in self.methodology_list:
            return False, None

        idx = self.methodology_list.index(instance_alias)
        if idx < 0 or idx >= len(self.function_base_aliases):
            return False, None

        base_alias = self.function_base_aliases[idx]
        widget_spec = self._get_widget_spec_by_name(base_alias, param_name)
        if not isinstance(widget_spec, dict) or 'default' not in widget_spec:
            return False, None

        default_value = widget_spec.get('default')

        # Match setup combobox behavior when a default points to a hidden choice.
        if str(widget_spec.get('widget', '')).strip().lower() == 'combobox':
            raw_values = widget_spec.get('values', [])
            raw_aliases = widget_spec.get('value_aliases', raw_values)
            filtered_values, _filtered_aliases, hidden_actual_values = self._filter_hidden_setup_choices(raw_values, raw_aliases)
            if str(default_value) in hidden_actual_values:
                default_value = filtered_values[0] if filtered_values else ""

        return True, default_value
    
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

        if instance_alias not in self.analysis_data:
            return True

        execution_results = self.analysis_data[instance_alias].get('execution_results', {})
        inputs = execution_results.get('inputs', {})
        outputs = execution_results.get('outputs', {})
        pf_output_keys = self._get_active_passforward_output_keys(instance_alias)
        filtered_outputs = outputs
        if isinstance(outputs, dict) and pf_output_keys:
            filtered_outputs = {
                key: value
                for key, value in outputs.items()
                if not (isinstance(key, str) and key in pf_output_keys)
            }
        combined_sources = self._get_execution_data_sources(execution_results, instance_alias)

        return _svc_evaluate_condition(
            condition,
            combined_sources=combined_sources,
            inputs=inputs,
            outputs=outputs,
            filtered_outputs=filtered_outputs,
            get_setup_default_for_parameter=lambda parameter: self._get_setup_default_for_condition_parameter(instance_alias, parameter),
        )

    def _get_custom_section_source(self, section_data: dict) -> Optional[dict]:
        if not isinstance(section_data, dict):
            return None
        source = section_data.get('_custom_source')
        return source if isinstance(source, dict) else None

    def _resolve_section_render_instance_alias(
        self,
        instance_alias: str,
        *,
        section_data: Optional[dict] = None,
        section_id: Optional[tuple] = None,
    ) -> str:
        """Resolve source function alias used for section data rendering context."""
        if instance_alias != self.CUSTOM_ANALYSIS_ALIAS:
            return instance_alias

        source_meta = self._get_custom_section_source(section_data) if section_data else None
        if not source_meta and section_id is not None:
            page_idx, section_idx = section_id
            pages = self.analysis_data.get(self.CUSTOM_ANALYSIS_ALIAS, {}).get('pages', [])
            if 0 <= page_idx < len(pages):
                sections = pages[page_idx].get('sections', []) if isinstance(pages[page_idx], dict) else []
                if 0 <= section_idx < len(sections) and isinstance(sections[section_idx], dict):
                    source_meta = self._get_custom_section_source(sections[section_idx])

        source_alias = source_meta.get('instance_alias') if isinstance(source_meta, dict) else None
        if isinstance(source_alias, str) and source_alias:
            return source_alias
        return instance_alias

    def _get_section_data_by_id(self, instance_alias: str, section_id: Optional[tuple]) -> Optional[dict]:
        """Return section data by (page_idx, section_idx) for an analysis instance."""
        if not isinstance(section_id, tuple) or len(section_id) != 2:
            return None
        try:
            page_idx = int(section_id[0])
            section_idx = int(section_id[1])
        except (TypeError, ValueError):
            return None

        pages = self.analysis_data.get(instance_alias, {}).get('pages', [])
        if not (0 <= page_idx < len(pages)):
            return None
        page_data = pages[page_idx] if isinstance(pages[page_idx], dict) else None
        if not isinstance(page_data, dict):
            return None
        sections = page_data.get('sections', [])
        if not isinstance(sections, list) or not (0 <= section_idx < len(sections)):
            return None
        return sections[section_idx] if isinstance(sections[section_idx], dict) else None

    def _get_section_render_outputs(
        self,
        instance_alias: str,
        section_id: Optional[tuple],
        section_data: Optional[dict] = None,
    ) -> tuple:
        """Resolve render alias and outputs for a section-specific data context."""
        resolved_section = section_data if isinstance(section_data, dict) else self._get_section_data_by_id(instance_alias, section_id)
        render_instance_alias = self._resolve_section_render_instance_alias(
            instance_alias,
            section_data=resolved_section,
            section_id=section_id,
        )
        execution_results = self.analysis_data.get(render_instance_alias, {}).get('execution_results', {})
        outputs = self._get_execution_data_sources(execution_results, render_instance_alias)
        return render_instance_alias, execution_results, outputs

    def _set_custom_execution_results_from_section(self, section_data: dict):
        """Set custom-analysis execution_results from the referenced source section."""
        self._ensure_custom_analysis_state()
        source = self._get_custom_section_source(section_data)
        if not source:
            self.analysis_data[self.CUSTOM_ANALYSIS_ALIAS]['execution_results'] = {}
            return

        source_alias = source.get('instance_alias')
        if isinstance(source_alias, str) and source_alias not in self.analysis_data and source_alias in self.methodology_list:
            src_idx = self.methodology_list.index(source_alias)
            src_base_alias = self.function_base_aliases[src_idx] if src_idx < len(self.function_base_aliases) else source_alias
            analysis_config = self.gui_configs.get(src_base_alias, {}).get('analysis') if src_base_alias in self.gui_configs else None
            if isinstance(analysis_config, dict):
                self.analysis_data[source_alias] = {
                    'pages': copy.deepcopy(analysis_config.get('pages', [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}])),
                    'current_page': analysis_config.get('current_page', 0),
                    'active_sections': {}
                }
            else:
                self.analysis_data[source_alias] = {
                    'pages': [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}],
                    'current_page': 0,
                    'active_sections': {}
                }

        source_info = self.analysis_data.get(source_alias, {}) if isinstance(source_alias, str) else {}
        source_exec = source_info.get('execution_results', {}) if isinstance(source_info, dict) else {}
        self.analysis_data[self.CUSTOM_ANALYSIS_ALIAS]['execution_results'] = copy.deepcopy(source_exec) if isinstance(source_exec, dict) else {}

    def _sync_custom_active_section_execution_results(self):
        """Keep custom-analysis execution context aligned with the active referenced section."""
        self._ensure_custom_analysis_state()
        info = self.analysis_data.get(self.CUSTOM_ANALYSIS_ALIAS, {})
        pages = info.get('pages', [])
        current_page = info.get('current_page', 0)
        if current_page < 0 or current_page >= len(pages):
            info['execution_results'] = {}
            return

        active_sections = info.get('active_sections', {})
        section_idx = active_sections.get(current_page)
        if section_idx is None:
            info['execution_results'] = {}
            return

        sections = pages[current_page].get('sections', [])
        if section_idx < 0 or section_idx >= len(sections):
            info['execution_results'] = {}
            return

        self._set_custom_execution_results_from_section(sections[section_idx])

    def _render_custom_empty_section(self, container: ttk.Frame, section_idx: int):
        """Render empty custom section placeholder with centered Add Section action."""
        holder = ttk.Frame(container)
        holder.pack(fill=tk.BOTH, expand=True)

        add_btn = ttk.Button(
            holder,
            text=self.language_manager.translate("ui.buttons.add_section", "Add Section"),
            command=lambda idx=section_idx: self._show_custom_add_section_dialog(idx)
        )
        add_btn.place(relx=0.5, rely=0.5, anchor="center")

    def _custom_analysis_has_results(self) -> bool:
        """Return whether any methodology function has populated analysis execution data."""
        if not hasattr(self, 'analysis_data') or not isinstance(self.analysis_data, dict):
            return False

        for instance_alias in self.methodology_list:
            exec_result = self.analysis_data.get(instance_alias, {}).get('execution_results', {})
            if isinstance(exec_result, dict) and bool(exec_result):
                return True
        return False
    
    def _render_analysis_page(self, parent: ttk.Frame, instance_alias: str, page_data: dict):
        """Render the current analysis page with the specified layout."""
        layout_type = page_data.get('layout', 'fp')
        sections = page_data.get('sections', [])
        current_page = self.analysis_data.get(instance_alias, {}).get('current_page', 0)
        rendered_sections = []
        is_custom = instance_alias == self.CUSTOM_ANALYSIS_ALIAS
        
        # Create layout containers
        containers = self._create_layout_containers(parent, layout_type)
        
        if is_custom:
            for resolved_section_idx, container in enumerate(containers):
                section_data = sections[resolved_section_idx] if resolved_section_idx < len(sections) else {'type': None}

                if section_data.get('type') is None:
                    self._render_custom_empty_section(container, resolved_section_idx)
                    continue

                source_meta = self._get_custom_section_source(section_data)
                eval_alias = source_meta.get('instance_alias') if source_meta else instance_alias
                if section_data.get('condition') and not self._evaluate_condition(eval_alias, section_data.get('condition')):
                    placeholder = ttk.Label(container, text=self.language_manager.translate("ui.messages.empty_section", "[Empty Section]"), foreground="gray")
                    placeholder.pack(expand=True)
                    continue

                self._set_custom_execution_results_from_section(section_data)
                self._render_section(container, instance_alias, section_data, resolved_section_idx)

                section_id = (current_page, resolved_section_idx)
                rendered_sections.append({
                    'section_id': section_id,
                    'container': container,
                    'section_data': section_data
                })
                self._bind_section_activation(container, instance_alias, section_id)
        else:
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
                    resolved_section_idx = section_idx - 1
                    self._render_section(container, instance_alias, section_data, resolved_section_idx)

                    section_id = (current_page, resolved_section_idx)
                    rendered_sections.append({
                        'section_id': section_id,
                        'container': container,
                        'section_data': section_data
                    })
                    self._bind_section_activation(container, instance_alias, section_id)
                    break
                else:
                    # No more sections with passing conditions
                    placeholder = ttk.Label(container, text=self.language_manager.translate("ui.messages.empty_section", "[Empty Section]"), foreground="gray")
                    placeholder.pack(expand=True)

        analysis_info = self.analysis_data.get(instance_alias, {})
        analysis_info['rendered_sections'] = rendered_sections

        if rendered_sections:
            active_sections = analysis_info.get('active_sections', {})
            selected_section_idx = active_sections.get(current_page)
            rendered_indices = {entry['section_id'][1] for entry in rendered_sections}
            if selected_section_idx not in rendered_indices:
                first_section_idx = rendered_sections[0]['section_id'][1]
                active_sections[current_page] = first_section_idx
                analysis_info['active_sections'] = active_sections

        if is_custom:
            self._sync_custom_active_section_execution_results()

        # Lock page bottom reservation to the largest data-slicing panel on this page.
        self._prime_page_nav_height_cache(instance_alias, current_page, rendered_sections)

        self._apply_analysis_section_styles(instance_alias)
        self._render_active_section_navigation(instance_alias)
    
    def _create_layout_containers(self, parent: ttk.Frame, layout_type: str) -> list:
        """Create layout containers based on layout type."""
        containers = []
        # Use consistent padding for all section containers to prevent border clipping
        section_padding = 8

        def _create_section_container(parent_widget):
            section_frame = ttk.LabelFrame(
                parent_widget,
                text=self.language_manager.translate("ui.labels.section", "Section"),
                padding=section_padding,
                style="AnalysisSectionInactive.TLabelframe"
            )
            return section_frame
        
        if layout_type == 'fd':  # Four sections (2x2 grid)
            # Use nested paned windows to ensure equal space distribution
            main_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
            main_paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Top row with horizontal paned window for 2 side-by-side containers
            top_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
            main_paned.add(top_paned, weight=1)
            for j in range(2):
                container = _create_section_container(top_paned)
                top_paned.add(container, weight=1)
                containers.append(container)
            
            # Bottom row with horizontal paned window for 2 side-by-side containers
            bottom_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
            main_paned.add(bottom_paned, weight=1)
            for j in range(2):
                container = _create_section_container(bottom_paned)
                bottom_paned.add(container, weight=1)
                containers.append(container)
            
            # Position sashes after rendering
            parent.after_idle(lambda: self._position_fd_sashes(main_paned, top_paned, bottom_paned))
        
        elif layout_type == 'fp':  # Full page (1 section)
            container = _create_section_container(parent)
            container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            containers.append(container)
        
        elif layout_type == 'ns':  # North-South (2 sections: top, bottom)
            paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
            paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

            top_frame = _create_section_container(paned)
            paned.add(top_frame, weight=1)
            containers.append(top_frame)

            bottom_frame = _create_section_container(paned)
            paned.add(bottom_frame, weight=1)
            containers.append(bottom_frame)

            parent.after_idle(lambda: self._position_paned_sash(paned, orient=tk.VERTICAL))
        
        elif layout_type == 'ew':  # East-West (2 sections: left, right)
            paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
            paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            left_container = _create_section_container(paned)
            paned.add(left_container, weight=1)
            containers.append(left_container)
            
            right_container = _create_section_container(paned)
            paned.add(right_container, weight=1)
            containers.append(right_container)
            
            parent.after_idle(lambda: self._position_paned_sash(paned))
        
        elif layout_type == 'sd':  # South Divided (3 sections: 1 top, 2 bottom)
            # Use vertical paned window for top/bottom division
            main_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
            main_paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Top section
            top_frame = _create_section_container(main_paned)
            main_paned.add(top_frame, weight=1)
            containers.append(top_frame)
            
            # Bottom side with horizontal paned window for 2 side-by-side containers
            bottom_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
            main_paned.add(bottom_paned, weight=1)
            for j in range(2):
                container = _create_section_container(bottom_paned)
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
                container = _create_section_container(top_paned)
                top_paned.add(container, weight=1)
                containers.append(container)
            
            # Bottom section
            bottom_frame = _create_section_container(main_paned)
            main_paned.add(bottom_frame, weight=1)
            containers.append(bottom_frame)
        
        elif layout_type == 'ed':  # East Divided (3 sections: 1 left, 2 right stacked)
            paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
            paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Left side (single container)
            left_container = _create_section_container(paned)
            paned.add(left_container, weight=1)
            containers.append(left_container)
            
            # Right side with vertical paned window for 2 stacked containers
            right_paned = ttk.PanedWindow(paned, orient=tk.VERTICAL)
            paned.add(right_paned, weight=1)
            
            for i in range(2):
                container = _create_section_container(right_paned)
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
                container = _create_section_container(left_paned)
                left_paned.add(container, weight=1)
                containers.append(container)
            
            # Right side (single container)
            right_container = _create_section_container(paned)
            paned.add(right_container, weight=1)
            containers.append(right_container)
            
            parent.after_idle(lambda: self._position_paned_sash(paned))
        
        else:
            # Default to full page for unknown layouts
            container = _create_section_container(parent)
            container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            containers.append(container)
        
        return containers
    
    def _render_section(self, parent: ttk.Frame, instance_alias: str, section_data: dict, section_idx: int = 0,
                        is_popup: bool = False, graph_font_scale_override: Optional[float] = None,
                        popup_refresh_callback: Optional[Callable[[], None]] = None,
                        section_id_override: Optional[Tuple[int, int]] = None):
        """Render a section (graph, table, or text)."""
        section_type = section_data.get('type')
        
        # Resolve section frame title with section-level title preferred, then config fallbacks.
        config = section_data.get('config', {})
        section_title = (
            section_data.get('title')
            or config.get('title')
            or config.get('graph_title')
            or self.language_manager.translate("ui.labels.section", "Section")
        )
        if hasattr(parent, 'configure'):
            try:
                parent.configure(text=section_title)
            except tk.TclError:
                pass  # Parent may not be a LabelFrame
        
        if section_type == 'graph':
            self._render_graph_section(
                parent,
                instance_alias,
                section_data,
                section_idx,
                graph_font_scale_override=graph_font_scale_override,
                is_popup=is_popup,
                popup_refresh_callback=popup_refresh_callback,
                section_id_override=section_id_override,
            )
        elif section_type == 'table':
            self._render_table_section(
                parent,
                instance_alias,
                section_data,
                section_idx,
                section_id_override=section_id_override,
            )
        elif section_type == 'text':
            self._render_text_section(parent, instance_alias, section_data)
        else:
            # Empty section
            label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.empty", "[Empty]"), foreground="gray")
            label.pack(expand=True)
        
        # Add popup button AFTER content is rendered (only if not already in a popup)
        if not is_popup:
            self._create_section_popup_button(parent, instance_alias, section_idx, section_data)
    
    def _resolve_axis_label(self, axis_config: dict, outputs: dict, axis_index: Optional[int] = None,
                            slice_indices: Optional[dict] = None) -> str:
        """Resolve an axis label from axis configuration.
        
        The label configuration supports:
        - Direct string: "label": "My Axis"
        - Variable reference: "label": "variable_name"
        - Variable with index: "label": "variable_name", "l_index": 0
        - Dynamic labels from array: "axis_labels": "pc_labels" (uses axis_index if provided)
        - Dynamic labels with dimension: "axis_labels": "y_titles", "axis_labels_dimension": 1
          (looks up current index for that dimension from slice_indices when axis_index is None)
        - Label template: "label_template": "$y_{{Ref}} ({label})$" where {label} is replaced
          by the resolved axis_labels value
        
        If the variable is a list and l_index is provided, returns the value at that index.
        For axis_labels with dynamic indexing, uses the axis_index parameter if provided.
        Otherwise, returns the variable value as a string.
        
        Args:
            axis_config: Axis configuration dict with 'label' or 'axis_labels' field
            outputs: Dictionary of execution outputs
            axis_index: Optional current axis index for dynamic label selection
            slice_indices: Optional dict mapping dimension -> current index (base_indices)
            
        Returns:
            Resolved label as string, or empty string if not found
        """
        if not axis_config:
            return ""

        label_template = axis_config.get('label_template')

        # Check for axis_labels (dynamic labels based on current index)
        axis_labels_config = axis_config.get('axis_labels')
        axis_labels_nested = axis_config.get('axis_labels_nested')
        if axis_labels_config:
            # Resolve effective index: prefer explicit axis_index, then fall back to
            # slice_indices[axis_labels_dimension] for general (non-axis) slice navigation.
            effective_index = axis_index
            axis_labels_dimension = axis_config.get('axis_labels_dimension')
            if effective_index is None and slice_indices is not None:
                if axis_labels_dimension is not None:
                    effective_index = slice_indices.get(axis_labels_dimension)
            if effective_index is not None:
                # axis_labels points to a variable containing an array of labels
                # Use helper to support nested dictionary access
                labels_data = self._get_data_from_source(outputs, axis_labels_config, axis_labels_nested)
                if labels_data is not None:
                    resolved = None
                    index_dims = axis_config.get('axis_labels_index_dimensions')
                    if isinstance(index_dims, (list, tuple)) and len(index_dims) > 0 and isinstance(slice_indices, dict):
                        try:
                            cursor = labels_data
                            for dim_id_raw in index_dims:
                                dim_id = int(dim_id_raw)
                                if axis_labels_dimension is not None and dim_id == int(axis_labels_dimension):
                                    idx_value = int(effective_index)
                                else:
                                    idx_value = int(slice_indices.get(dim_id, 0))
                                if isinstance(cursor, np.ndarray):
                                    cursor = cursor[idx_value]
                                elif isinstance(cursor, (list, tuple)):
                                    cursor = cursor[idx_value]
                                else:
                                    cursor = None
                                    break
                            if cursor is not None and not isinstance(cursor, (list, tuple, np.ndarray)):
                                resolved = str(cursor)
                        except Exception:
                            resolved = None

                    if resolved is None and isinstance(labels_data, (list, np.ndarray)):
                        try:
                            resolved = str(labels_data[effective_index])
                        except (IndexError, TypeError):
                            resolved = None

                    if resolved is not None:
                        if label_template:
                            result = label_template.replace('{label}', resolved)
                            result = result.replace('{nav_idx}', str(effective_index + 1))
                            result = result.replace('{nav_idx0}', str(effective_index))
                            return result
                        return resolved

        label_config = axis_config.get('label')
        if not label_config:
            return ""
        
        # First try to resolve as variable reference
        if isinstance(label_config, str):
            label_nested = axis_config.get('label_nested')
            data = self._get_data_from_source(outputs, label_config, label_nested)
            if data is not None:
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

            # It's a literal string label
            return label_config
        else:
            return str(label_config)
    
    def _get_variable_label(self, outputs: dict, var_name: Any, dimension: int, index: int, fallback: bool = True) -> Optional[str]:
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

        try:
            if isinstance(var_name, dict):
                data_source = var_name.get('data_source')
                nested_key = var_name.get('nested_key')
                if not data_source:
                    return fallback_label
                data = self._get_data_from_source(outputs, data_source, nested_key)
            elif isinstance(var_name, str):
                if var_name not in outputs:
                    return fallback_label
                data = outputs[var_name]
            else:
                return fallback_label
        except Exception:
            return fallback_label

        if data is None:
            return fallback_label

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

    def _get_variable_label_with_indices(
        self,
        outputs: dict,
        var_name: Any,
        dimension: int,
        index: int,
        indices: Optional[dict] = None,
        index_dimensions: Optional[Any] = None,
        fallback: bool = True,
    ) -> Optional[str]:
        """Resolve variable labels with optional multidimensional index mapping.

        When ``index_dimensions`` is provided (e.g., [0, 2]), this method walks
        the labels tensor using current slice indices for all mapped dimensions,
        replacing the target ``dimension`` with ``index``.
        """
        fallback_label = f"V{index + 1}" if fallback else None

        if not isinstance(index_dimensions, (list, tuple)) or len(index_dimensions) == 0:
            return self._get_variable_label(outputs, var_name, dimension, index, fallback=fallback)

        try:
            if isinstance(var_name, dict):
                data_source = var_name.get('data_source')
                nested_key = var_name.get('nested_key')
                if not data_source:
                    return fallback_label
                data = self._get_data_from_source(outputs, data_source, nested_key)
            elif isinstance(var_name, str):
                if var_name not in outputs:
                    return fallback_label
                data = outputs[var_name]
            else:
                return fallback_label
        except Exception:
            return fallback_label

        if data is None:
            return fallback_label

        try:
            cursor = data
            indices = indices if isinstance(indices, dict) else {}
            for dim_id_raw in index_dimensions:
                dim_id = int(dim_id_raw)
                idx_value = int(index) if dim_id == int(dimension) else int(indices.get(dim_id, 0))
                if isinstance(cursor, np.ndarray):
                    cursor = cursor[idx_value]
                elif isinstance(cursor, (list, tuple)):
                    cursor = cursor[idx_value]
                else:
                    return fallback_label

            if isinstance(cursor, (list, tuple, np.ndarray)):
                return fallback_label
            return str(cursor) if cursor is not None else fallback_label
        except Exception:
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

    def _resolve_scatter_line_source_value(self, raw_value: Any, outputs: dict, *, implicit_lookup: bool = False, slice_indices: Optional[Dict[int, int]] = None) -> Any:
        """Resolve scatter-line field values from literals or output sources.

        Supported source forms:
        - {"var": "name"}
        - {"data_source": "name", "nested_key": "a.b", "slice_dimension": 1}
        - "name" (only when implicit_lookup=True and key exists in outputs)

        When slice_indices is provided and the resolved value is an ndarray, the method
        applies the slice for each dimension present in slice_indices so that per-pairing
        line data can be indexed correctly without exposing the full array to the renderer.
        The dict entry may also carry an explicit ``slice_dimension`` key to restrict
        slicing to that single dimension only.
        """
        resolved: Any

        if isinstance(raw_value, dict):
            source_name = raw_value.get('data_source', raw_value.get('var', raw_value.get('source')))
            nested_key = raw_value.get('nested_key')
            explicit_slice_dim = raw_value.get('slice_dimension')
            if isinstance(source_name, str) and source_name:
                resolved = self._get_data_from_source(outputs, source_name, nested_key)
            else:
                return raw_value
        elif implicit_lookup and isinstance(raw_value, str) and raw_value in outputs:
            explicit_slice_dim = None
            resolved = outputs[raw_value]
        else:
            return raw_value

        def _slice_ndarray(arr: np.ndarray, dim_override: Optional[Any] = None) -> np.ndarray:
            if arr.ndim < 2 or not slice_indices:
                return arr
            if dim_override is not None:
                try:
                    dims_to_slice = {int(dim_override)}
                except (TypeError, ValueError):
                    dims_to_slice = set(slice_indices.keys())
            else:
                dims_to_slice = set(slice_indices.keys())
            out_arr = arr
            for dim in sorted(dims_to_slice, reverse=True):
                idx = slice_indices.get(dim)
                if idx is None or dim >= out_arr.ndim:
                    continue
                max_idx = out_arr.shape[dim] - 1
                idx = max(0, min(int(idx), max_idx))
                out_arr = np.take(out_arr, idx, axis=dim)
            return out_arr

        def _slice_nested(value: Any) -> Any:
            if not slice_indices:
                return value
            if isinstance(value, np.ndarray):
                return _slice_ndarray(value)
            if isinstance(value, dict):
                return {k: _slice_nested(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_slice_nested(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_slice_nested(v) for v in value)
            return value

        # Apply slicing to ndarrays and packed nested structures.
        if slice_indices:
            if isinstance(resolved, np.ndarray):
                resolved = _slice_ndarray(resolved, explicit_slice_dim)
            elif isinstance(resolved, dict):
                resolved = _slice_nested(resolved)

        return resolved

    def _resolve_scatter_reference_lines(self, config: dict, outputs: dict, slice_indices: Optional[Dict[int, int]] = None) -> List[Dict[str, Any]]:
        """Resolve scatter reference-line config entries from JSON and outputs.

        Accepted config keys:
        - scatter_lines
        - reference_lines
        - guide_lines

        When slice_indices is provided it is forwarded to _resolve_scatter_line_source_value
        so that per-pairing array data sources are indexed to a scalar/1-D value matching
        the currently active navigation slice.
        """
        source_candidates = (
            config.get('scatter_lines'),
            config.get('reference_lines'),
            config.get('guide_lines'),
        )

        raw_entries: List[Any] = []
        for candidate in source_candidates:
            if candidate is None:
                continue
            if isinstance(candidate, list):
                raw_entries.extend(candidate)
            elif isinstance(candidate, dict):
                raw_entries.append(candidate)

        resolved_entries: List[Dict[str, Any]] = []
        implicit_lookup_fields = {
            'value', 'values', 'x', 'y', 'x1', 'y1', 'x2', 'y2',
            'x_start', 'y_start', 'x_end', 'y_end', 'point1', 'point2', 'points',
            'label', 'labels', 'linewidth', 'thickness', 'width', 'alpha'
        }

        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue

            resolved_entry: Dict[str, Any] = {}
            for key, value in raw_entry.items():
                should_lookup = key in implicit_lookup_fields

                if isinstance(value, list):
                    resolved_entry[key] = [
                        self._resolve_scatter_line_source_value(item, outputs, implicit_lookup=should_lookup, slice_indices=slice_indices)
                        for item in value
                    ]
                else:
                    resolved_entry[key] = self._resolve_scatter_line_source_value(
                        value,
                        outputs,
                        implicit_lookup=should_lookup,
                        slice_indices=slice_indices,
                    )

            resolved_entries.append(resolved_entry)

        return resolved_entries
    
    def _render_graph_section(self, parent: ttk.Frame, instance_alias: str, section_data: dict,
                              section_idx: int = 0, graph_font_scale_override: Optional[float] = None,
                              is_popup: bool = False,
                              popup_refresh_callback: Optional[Callable[[], None]] = None,
                              section_id_override: Optional[Tuple[int, int]] = None):
        """Render a graph using matplotlib with optional navigation controls."""
        try:
            config = section_data.get('config', {})
            graph_type = config.get('graph_type', 'scatter')

            # Keep custom tab state independent, but resolve data context from source function.
            if isinstance(section_id_override, tuple) and len(section_id_override) == 2:
                section_id = section_id_override
            else:
                current_page = self.analysis_data.get(instance_alias, {}).get('current_page', 0)
                section_id = (current_page, section_idx)
            render_instance_alias = self._resolve_section_render_instance_alias(
                instance_alias,
                section_data=section_data,
                section_id=section_id,
            )
            
            # Get execution results
            if render_instance_alias not in self.analysis_data:
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), foreground="gray")
                label.pack(expand=True)
                return
            
            execution_results = self.analysis_data[render_instance_alias].get('execution_results', {})
            if not execution_results:
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), foreground="gray")
                label.pack(expand=True)
                return
            
            if execution_results.get('status') != 'success':
                log_hint = f"Execution failed. Check log: {_get_runtime_model_log_path()}"
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.execution_failed_check_log", log_hint), foreground="red")
                label.pack(expand=True)
                return
            
            outputs = self._get_execution_data_sources(execution_results, render_instance_alias)
            
            # Initialize slice state if needed
            # Use (page_index, section_idx) tuple as key to ensure each graph has unique state per page
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
                                dimension = nav_item['dimension']
                                # Get default from nav_item or axis config
                                default_val = nav_item.get('default', None)
                                if default_val is None:
                                    default_val = config.get(f'{target_axis}_axis', {}).get('default_column', 0)
                                if target_axis not in current_slice['axis_indices']:
                                    current_slice['axis_indices'][target_axis] = {}
                                current_slice['axis_indices'][target_axis][dimension] = default_val
            
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

                existing_x_axis = config.get('x_axis', {}) if isinstance(config.get('x_axis'), dict) else {}
                existing_y_axis = config.get('y_axis', {}) if isinstance(config.get('y_axis'), dict) else {}
                
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
                
                # Heatmap matrices are row-major: first active dim -> rows (y),
                # second active dim -> columns (x).
                x_axis_config = None
                if len(md_active_dims) > 1:
                    dim_idx = md_active_dims[1]
                    x_label = labels[dim_idx] if dim_idx < len(labels) else f"Axis {dim_idx}"
                    x_axis_config = existing_x_axis.copy()
                    x_axis_config.update({'data_source': data_source, 'index': dim_idx, 'label': x_label})
                
                # First active dim maps to heatmap row axis (y).
                y_axis_config = None
                if len(md_active_dims) > 0:
                    dim_idx = md_active_dims[0]
                    y_label = labels[dim_idx] if dim_idx < len(labels) else f"Axis {dim_idx}"
                    y_axis_config = existing_y_axis.copy()
                    y_axis_config.update({'data_source': data_source, 'index': dim_idx, 'label': y_label})
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

            # Only apply axis-specific indices that are currently declared in data_slicing.
            active_axis_dims = {'x': set(), 'y': set(), 'z': set()}
            for nav_item in nav_axes:
                if not isinstance(nav_item, dict):
                    continue
                target_axis = nav_item.get('axis')
                if target_axis not in active_axis_dims:
                    continue
                dim_val = nav_item.get('dimension')
                try:
                    dim_idx = int(dim_val)
                except (TypeError, ValueError):
                    continue
                active_axis_dims[target_axis].add(dim_idx)

            # Extract the actual index values from axis_indices
            if 'x' in axis_indices and x_dimension is not None:
                x_axis_idx = axis_indices['x'].get(x_dimension)
            if 'y' in axis_indices and y_dimension is not None:
                y_axis_idx = axis_indices['y'].get(y_dimension)
            if 'z' in axis_indices and z_dimension is not None:
                z_axis_idx = axis_indices['z'].get(z_dimension)
            
            resolved_x_label = self._resolve_axis_label(x_axis_config, outputs, axis_index=x_axis_idx, slice_indices=base_indices)
            if resolved_x_label:
                x_axis_config['label'] = resolved_x_label
            
            resolved_y_label = self._resolve_axis_label(y_axis_config, outputs, axis_index=y_axis_idx, slice_indices=base_indices)
            if resolved_y_label:
                y_axis_config['label'] = resolved_y_label
            
            # Also resolve z-axis label if present
            z_axis_config = config.get('z_axis', {})
            if z_axis_config:
                z_axis_config = z_axis_config.copy()
                resolved_z_label = self._resolve_axis_label(z_axis_config, outputs, axis_index=z_axis_idx, slice_indices=base_indices)
                if resolved_z_label:
                    z_axis_config['label'] = resolved_z_label
            
            # Merge axis indices for y (base + md + axis-specific)
            # Extract y FIRST so __index__ on x-axis can use y_data as reference
            y_indices = base_indices.copy()
            y_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'y' in axis_indices and isinstance(axis_indices.get('y'), dict):
                for dim_key, dim_index in axis_indices['y'].items():
                    try:
                        dim_int = int(dim_key)
                    except (TypeError, ValueError):
                        continue
                    if dim_int in active_axis_dims['y']:
                        y_indices[dim_int] = dim_index
            y_data = self._extract_axis_data(outputs, y_axis_config, y_indices)
            
            # Merge axis indices for x (base + md + axis-specific)
            x_indices = base_indices.copy()
            x_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'x' in axis_indices and isinstance(axis_indices.get('x'), dict):
                for dim_key, dim_index in axis_indices['x'].items():
                    try:
                        dim_int = int(dim_key)
                    except (TypeError, ValueError):
                        continue
                    if dim_int in active_axis_dims['x']:
                        x_indices[dim_int] = dim_index
            # Pass y_data as reference for row index generation if needed
            x_data = self._extract_axis_data(outputs, x_axis_config, x_indices, ref_data=y_data)
            
            # Merge axis indices for z (base + md + axis-specific)
            z_indices = base_indices.copy()
            z_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'z' in axis_indices and isinstance(axis_indices.get('z'), dict):
                for dim_key, dim_index in axis_indices['z'].items():
                    try:
                        dim_int = int(dim_key)
                    except (TypeError, ValueError):
                        continue
                    if dim_int in active_axis_dims['z']:
                        z_indices[dim_int] = dim_index
            # For 4D+ multi-dimensional display, exclude dimensions that are being displayed
            # (md_active_dims) from z_indices AFTER all merges to avoid slicing them away
            if md_active_dims:
                for dim in md_active_dims:
                    z_indices.pop(dim, None)  # Remove if present
            # Pass x_data or y_data as reference for row index generation if needed
            z_data = self._extract_axis_data(outputs, config.get('z_axis', {}), z_indices, ref_data=x_data if x_data is not None else y_data)
            
            # Navigation controls are rendered globally for the active section
            
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

                    # Resolve dataset axis labels from variables (including nested lookups).
                    ds_x_axis = ds_x_axis.copy() if isinstance(ds_x_axis, dict) else {}
                    ds_y_axis = ds_y_axis.copy() if isinstance(ds_y_axis, dict) else {}
                    ds_z_axis = ds_z_axis.copy() if isinstance(ds_z_axis, dict) else {}
                    resolved_ds_x_label = self._resolve_axis_label(ds_x_axis, outputs, axis_index=x_axis_idx, slice_indices=base_indices)
                    if resolved_ds_x_label:
                        ds_x_axis['label'] = resolved_ds_x_label
                    resolved_ds_y_label = self._resolve_axis_label(ds_y_axis, outputs, axis_index=y_axis_idx, slice_indices=base_indices)
                    if resolved_ds_y_label:
                        ds_y_axis['label'] = resolved_ds_y_label
                    if ds_z_axis:
                        resolved_ds_z_label = self._resolve_axis_label(ds_z_axis, outputs, axis_index=z_axis_idx, slice_indices=base_indices)
                        if resolved_ds_z_label:
                            ds_z_axis['label'] = resolved_ds_z_label
                    
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
                    ds_class_layers = None
                    if 'class_labels' in dataset_cfg:
                        class_source = dataset_cfg['class_labels']
                        class_val = self._get_data_from_source(outputs, class_source)
                        if class_val is not None:
                            if isinstance(class_val, (list, np.ndarray)):
                                ds_class_layers = self._normalize_class_data_matrix(class_val)
                                ds_class_data = self._normalize_class_labels_for_plot(class_val)
                    
                    # Dataset is valid and will be rendered
                    dataset_entry = {
                        'x_data': ds_x_data,
                        'y_data': ds_y_data,
                        'label': dataset_label,
                        'visibility_key': f'cfg:{dataset_idx}',
                        'x_axis': ds_x_axis,  # Preserve axis config for label extraction
                        'y_axis': ds_y_axis   # Preserve axis config for label extraction
                    }
                    if graph_type == 'scatter':
                        dataset_entry['marker'] = dataset_cfg.get('marker', 'o')
                    elif 'marker' in dataset_cfg:
                        dataset_entry['marker'] = dataset_cfg.get('marker')
                    if ds_z_data is not None:
                        dataset_entry['z_data'] = ds_z_data
                    if ds_class_data is not None:
                        dataset_entry['class_data'] = ds_class_data
                    if ds_class_layers is not None:
                        dataset_entry['class_layers'] = ds_class_layers
                    # Include color if specified (used as fallback when no class_data)
                    if 'color' in dataset_cfg:
                        dataset_entry['color'] = dataset_cfg['color']
                    # Include point_labels_source if specified per-dataset
                    ds_point_labels_source = dataset_cfg.get('point_labels_source', dataset_cfg.get('sample_labels_source'))
                    if ds_point_labels_source:
                        dataset_entry['point_labels_source'] = ds_point_labels_source
                        ds_point_labels_index = dataset_cfg.get('point_labels_index')
                        if ds_point_labels_index is not None:
                            dataset_entry['point_labels_index'] = ds_point_labels_index
                    extracted_datasets.append(dataset_entry)
            
            # If main plot has class_labels config, treat it as a dataset for proper class coloring with qualitative colormap
            if 'class_labels' in config and graph_type in ('scatter', 'line') and x_data is not None and y_data is not None:
                class_source = config['class_labels']
                class_val = self._get_data_from_source(outputs, class_source)
                if class_val is not None:
                    if isinstance(class_val, (list, np.ndarray)):
                        main_class_layers = self._normalize_class_data_matrix(class_val)
                        main_class_data = self._normalize_class_labels_for_plot(class_val)
                        
                        # Create extracted_datasets if it doesn't exist
                        if extracted_datasets is None:
                            extracted_datasets = []
                        
                        # Build main dataset with class data
                        main_dataset = {
                            'x_data': x_data,
                            'y_data': y_data,
                            'label': 'Main Dataset',
                            'visibility_key': 'main_class_dataset',
                            'class_data': main_class_data,
                            'x_axis': x_axis_config,
                            'y_axis': y_axis_config
                        }
                        # Scatter uses marker for dataset identity; line uses linestyle
                        if graph_type == 'scatter':
                            main_dataset['marker'] = 'o'
                        elif graph_type == 'line' and 'marker' in config:
                            main_dataset['marker'] = config.get('marker')
                        if main_class_layers is not None:
                            main_dataset['class_layers'] = main_class_layers
                        if z_data is not None:
                            main_dataset['z_data'] = z_data
                        # Include point_labels_source if specified at config level
                        cfg_point_labels_source = config.get('point_labels_source', config.get('sample_labels_source'))
                        if cfg_point_labels_source:
                            main_dataset['point_labels_source'] = cfg_point_labels_source
                        
                        # Add main dataset at the beginning of the list
                        extracted_datasets.insert(0, main_dataset)
                        
                        # Clear x/y/z data so they won't conflict with multi-dataset rendering
                        x_data = None
                        y_data = None
                        z_data = None

            extracted_datasets = self._assign_dataset_style_slots(extracted_datasets)
            extracted_datasets = self._apply_dataset_visibility_filter(config, extracted_datasets)
            
            # Extract sample labels for tooltip display from individual datasets
            sample_labels = None
            sample_labels_by_dataset = None
            
            # If we have extracted datasets, collect their point_labels_source
            if extracted_datasets and len(extracted_datasets) > 0:
                sample_labels_by_dataset = {}
                for dataset_entry in extracted_datasets:
                    ds_label = dataset_entry.get('label')
                    ds_source = dataset_entry.get('point_labels_source')
                    if ds_label and ds_source and ds_source in outputs:
                        labels_data = outputs[ds_source]
                        ds_labels_index = dataset_entry.get('point_labels_index')
                        if ds_labels_index is not None and isinstance(labels_data, list) and labels_data and isinstance(labels_data[0], list):
                            labels_data = labels_data[ds_labels_index] if ds_labels_index < len(labels_data) else labels_data[0]
                        if isinstance(labels_data, (list, np.ndarray)):
                            sample_labels_by_dataset[ds_label] = [str(lbl) for lbl in labels_data]
            else:
                # For single dataset, check for point_labels_source in config
                point_labels_source = config.get('point_labels_source', config.get('sample_labels_source'))
                if isinstance(point_labels_source, str):
                    # Single sample labels source
                    if point_labels_source in outputs:
                        labels_data = outputs[point_labels_source]
                        point_labels_index = config.get('point_labels_index')
                        if point_labels_index is not None and isinstance(labels_data, list) and labels_data and isinstance(labels_data[0], list):
                            labels_data = labels_data[point_labels_index] if point_labels_index < len(labels_data) else labels_data[0]
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

            # Resolve reference-line slicing with the same effective indices used for axis extraction,
            # including axis-specific navigation dimensions (e.g., SBS pairing on x/y axis selectors).
            line_slice_indices = base_indices.copy()
            line_slice_indices.update(md_slice_indices)
            for axis_name in ('x', 'y', 'z'):
                axis_dim_set = active_axis_dims.get(axis_name, set())
                axis_index_map = axis_indices.get(axis_name, {}) if isinstance(axis_indices.get(axis_name), dict) else {}
                for dim_key, dim_index in axis_index_map.items():
                    try:
                        dim_int = int(dim_key)
                    except (TypeError, ValueError):
                        continue
                    if dim_int in axis_dim_set:
                        line_slice_indices[dim_int] = dim_index

            normalized_graph_type = str(graph_type).strip().lower()

            if normalized_graph_type in {'scatter', 'line'}:
                render_config['scatter_reference_lines'] = self._resolve_scatter_reference_lines(config, outputs, slice_indices=line_slice_indices)

            if normalized_graph_type in {'scatter', 'line'}:
                if normalized_graph_type == 'scatter':
                    class_state = self._compute_scatter_class_layer_state(
                        config,
                        datasets_config,
                        outputs,
                        execution_results.get('inputs', {}) if isinstance(execution_results, dict) else {},
                    )
                    configured_order_raw = config.get('class_layer_order', [])
                    if class_state.get('is_multi_dataset') and isinstance(configured_order_raw, list) and 'marker' in [str(v).strip().lower() for v in configured_order_raw]:
                        self._show_graph_warning_once(
                            instance_alias,
                            section_id,
                            'scatter_marker_reserved',
                            self.language_manager.translate(
                                'ui.messages.scatter_marker_reserved_multi_dataset',
                                'Scatter class-layer marker mapping is disabled for multiple datasets; marker is reserved for dataset identity.'
                            )
                        )
                else:
                    class_state = self._compute_line_class_layer_state(
                        config,
                        datasets_config,
                        outputs,
                        execution_results.get('inputs', {}) if isinstance(execution_results, dict) else {},
                    )
                    if class_state.get('marker_explicit'):
                        render_config['line_marker_reserved'] = True

                effective_order = list(class_state.get('effective_order', []))
                effective_map = dict(class_state.get('effective_map', {}))
                layer_nature = dict(class_state.get('layer_nature', {}))

                render_config['class_layer_order_effective'] = effective_order
                render_config['class_layer_map_effective'] = effective_map
                render_config['class_layer_nature_effective'] = {str(k): str(v) for k, v in layer_nature.items()}
                render_config['class_layer_count'] = int(class_state.get('layer_count', 0))
                render_config['class_color_palette_mode'] = str(config.get('class_color_palette_mode', 'auto'))
                render_config['class_edge_palette_mode'] = str(config.get('class_edge_palette_mode', 'auto'))
                render_config['class_color_cmap_continuous'] = str(config.get('class_color_cmap_continuous', self.settings_manager.get('colormap', 'viridis')))
                render_config['class_edge_cmap_continuous'] = str(config.get('class_edge_cmap_continuous', self.settings_manager.get('colormap', 'viridis')))
                render_config['class_color_cmap_qualitative'] = str(config.get('class_color_cmap_qualitative', self.settings_manager.get('qualitative_colormap', 'tab10')))
                render_config['class_edge_cmap_qualitative'] = str(config.get('class_edge_cmap_qualitative', self.settings_manager.get('qualitative_colormap', 'tab10')))

                model_payload = outputs.get('model') if isinstance(outputs, dict) else None
                if isinstance(model_payload, dict):
                    ordered_labels = self._build_class_value_order_effective(outputs, model_payload)
                    if ordered_labels:
                        render_config['class_value_order_effective'] = ordered_labels

                if normalized_graph_type == 'line':
                    series_source = config.get('line_series_labels_source')
                    if isinstance(series_source, str) and series_source:
                        series_data = self._get_data_from_source(outputs, series_source)
                        if isinstance(series_data, (list, np.ndarray)):
                            try:
                                labels_arr = np.asarray(series_data, dtype=object).reshape(-1)
                                render_config['line_series_labels'] = [str(item) for item in labels_arr.tolist()]
                            except Exception:
                                pass

            if normalized_graph_type in {'scatter', 'line'}:
                render_config['scatter_reference_lines'] = self._resolve_scatter_reference_lines(config, outputs, slice_indices=line_slice_indices)
            
            # Render graph using graph_renderer module
            graph_renderer = self._get_graph_renderer()
            active_graph_font_scale = self._normalize_graph_font_scale(
                graph_font_scale_override if graph_font_scale_override is not None else self.graph_font_scale
            )
            fig, ax = graph_renderer.render_graph_figure(
                graph_type, render_config, x_data, y_data, z_data, x_axis_config, y_axis_config,
                default_cmap=self.settings_manager.get('colormap', 'viridis'),
                datasets=extracted_datasets,
                qualitative_cmap=self.settings_manager.get('qualitative_colormap', 'tab10'),
                sample_labels=sample_labels,
                sample_labels_by_dataset=sample_labels_by_dataset,
                font_scale=active_graph_font_scale
            )
            
            # Embed figure in tkinter within a managed frame
            canvas, canvas_frame = graph_renderer.embed_figure_in_tkinter(
                fig,
                parent,
                pinned_labels=self._globally_pinned_point_labels,
                on_label_pin_toggled=self._on_global_point_label_pin_toggled,
            )
            
            # Popup rendering must not overwrite main-tab canvas/metadata entries.
            if not is_popup:
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

            self._attach_graph_context_menu(
                canvas,
                graph_type,
                config,
                instance_alias,
                section_id,
                popup_refresh_callback=popup_refresh_callback if is_popup else None
            )
            
        except Exception as e:
            label = ttk.Label(
                parent,
                text=self.language_manager.translate("ui.messages.error_rendering_graph", "Error rendering graph:") + f" {str(e)}",
                foreground="red"
            )
            label.pack(expand=True)
    
    def _get_data_from_source(self, outputs: dict, data_source: Any, nested_key: str = None) -> Any:
        """Extract data from a source, supporting nested dictionary access and special markers.
        
        Args:
            outputs: Dictionary of execution outputs
            data_source: Key to access in outputs (e.g., 'metrics', 'model_results').
                        Also accepts a list/tuple for fallback lookup order, where
                        each item can be a source name string or a dict containing
                        {'data_source': <name>, 'nested_key': <path>}.
                        Can also be special markers:
                        - "__index__": Auto-generated row indices (requires reference_source)
                        - "row_index": Alias for __index__
            nested_key: Optional key for nested dictionary access (e.g., 'pct_variance_explained')
                       Can be a single key or dot-separated path (e.g., 'stats.mean')
        
        Returns:
            The extracted data, or None if not found
        """
        return _svc_get_data_from_source(outputs, data_source, nested_key)

        # Support fallback source resolution (first non-None wins).
        if isinstance(data_source, (list, tuple)):
            for candidate in data_source:
                candidate_source = candidate
                candidate_nested = nested_key

                if isinstance(candidate, dict):
                    candidate_source = candidate.get('data_source', candidate.get('var', candidate.get('source')))
                    candidate_nested = candidate.get('nested_key', nested_key)

                if candidate_source is None:
                    continue

                value = self._get_data_from_source(outputs, candidate_source, candidate_nested)
                if value is not None:
                    return value

            return None

        if isinstance(data_source, dict):
            source_name = data_source.get('data_source', data_source.get('var', data_source.get('source')))
            if source_name is None:
                return None
            resolved_nested = data_source.get('nested_key', nested_key)
            return self._get_data_from_source(outputs, source_name, resolved_nested)

        # Handle special index markers
        if data_source in ("__index__", "row_index"):
            # This will be handled specially - return marker to caller
            return "__index__"
        
        # Get the base data from outputs with prefixed/unprefixed compatibility
        if data_source not in outputs:
            source_key = str(data_source) if data_source is not None else ""
            fallback_keys = []
            if source_key.startswith('in.') or source_key.startswith('out.') or source_key.startswith('pf.'):
                fallback_keys.append(source_key.split('.', 1)[1])
            elif source_key:
                fallback_keys.extend([f"out.{source_key}", f"in.{source_key}", f"pf.{source_key}"])

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

        nested_key_normalized = str(nested_key).strip().lower() if nested_key is not None else ''

        # Pseudo keys for common metadata on array/list-like values
        if nested_key_normalized in ('shape', 'ndim', 'size', 'len', 'length'):
            try:
                if nested_key_normalized == 'shape':
                    if isinstance(data, np.ndarray):
                        return tuple(data.shape)
                    if isinstance(data, (list, tuple)):
                        return (len(data),)
                    return None
                if nested_key_normalized == 'ndim':
                    if isinstance(data, np.ndarray):
                        return int(data.ndim)
                    if isinstance(data, (list, tuple)):
                        return 1
                    return 0
                if nested_key_normalized == 'size':
                    if isinstance(data, np.ndarray):
                        return int(data.size)
                    if isinstance(data, (list, tuple)):
                        return int(len(data))
                    if data is None:
                        return 0
                    return 1
                if isinstance(data, (list, tuple, dict, np.ndarray)):
                    return int(len(data))
                return None
            except Exception:
                return None
        
        # Handle nested access via dot-separated path supporting both dict keys and
        # integer list/array indices (e.g. "calibration.matrix.1.1").
        if isinstance(data, (dict, list, tuple, np.ndarray)):
            keys_path = nested_key.split('.') if '.' in nested_key else [nested_key]

            try:
                for key in keys_path:
                    if isinstance(data, dict):
                        if key in data:
                            data = data[key]
                        else:
                            return None
                    elif isinstance(data, (list, tuple, np.ndarray)):
                        try:
                            idx = int(key)
                        except (ValueError, TypeError):
                            return None
                        if 0 <= idx < len(data):
                            data = data[idx]
                        else:
                            return None
                    else:
                        return None
                return data
            except (KeyError, IndexError, TypeError, AttributeError):
                return None

        return None

    def _resolve_text_selector_index(self, token: Any, length: int, default: int = 0) -> int:
        """Resolve index token for text selectors (supports int, first, last)."""
        if length <= 0:
            return 0

        if token is None:
            idx = default
        elif isinstance(token, (int, np.integer)):
            idx = int(token)
        else:
            raw = str(token).strip().lower()
            if raw in ('', 'none'):
                idx = default
            elif raw == 'first':
                idx = 0
            elif raw == 'last':
                idx = length - 1
            else:
                try:
                    idx = int(raw)
                except (TypeError, ValueError):
                    idx = default

        if idx < 0:
            idx = length + idx
        return max(0, min(length - 1, idx))

    def _format_text_binding_atom(self, value: Any, value_format: str = '') -> str:
        """Format a single text-binding value with optional numeric format."""
        if value is None:
            return ''

        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)

        if isinstance(value, np.ndarray):
            try:
                return json.dumps(value.tolist(), ensure_ascii=False)
            except Exception:
                return str(value)

        fmt = str(value_format or '').strip()
        if fmt:
            try:
                return format(value, fmt)
            except Exception:
                return str(value)

        return str(value)

    def _coerce_text_binding_sequence(self, value: Any, dict_mode: str = 'values', value_format: str = '') -> List[str]:
        """Normalize value to a 1D list of formatted strings for text-table rendering."""
        if value is None:
            return []

        if isinstance(value, np.ndarray):
            try:
                flat_values = value.reshape(-1).tolist()
            except Exception:
                flat_values = [value]
        elif isinstance(value, (list, tuple)):
            flat_values = list(value)
        elif isinstance(value, dict):
            mode = str(dict_mode or 'values').strip().lower()
            if mode == 'keys':
                flat_values = list(value.keys())
            elif mode == 'items':
                flat_values = [f"{k}: {v}" for k, v in value.items()]
            else:
                flat_values = list(value.values())
        else:
            flat_values = [value]

        return [self._format_text_binding_atom(item, value_format=value_format) for item in flat_values]

    def _extract_text_binding_value(self, outputs: dict, binding: dict) -> Any:
        """Resolve one text binding from configured source + selector."""
        if not isinstance(binding, dict):
            return None

        data_source = binding.get('data_source')
        nested_key = binding.get('nested_key')
        value = self._get_data_from_source(outputs, data_source, nested_key)
        if value is None:
            return None

        selector = binding.get('selector', {}) if isinstance(binding.get('selector', {}), dict) else {}
        mode = str(selector.get('mode', 'value')).strip().lower() or 'value'

        if mode == 'value':
            return value

        if not isinstance(value, np.ndarray):
            if isinstance(value, (list, tuple)):
                array_value = np.array(value, dtype=object)
            elif isinstance(value, dict):
                array_value = np.array(list(value.values()), dtype=object)
            else:
                array_value = None
        else:
            array_value = value

        if array_value is None:
            return value if mode == 'value' else None

        flat_values = list(array_value.reshape(-1).tolist())
        if not flat_values:
            return None

        if mode == 'index':
            token = selector.get('index')
            idx = self._resolve_text_selector_index(token, len(flat_values), default=0)
            return flat_values[idx]

        if mode == 'range':
            start_token = selector.get('start')
            end_token = selector.get('end')
            start_idx = self._resolve_text_selector_index(start_token, len(flat_values), default=0)
            end_idx = self._resolve_text_selector_index(end_token, len(flat_values), default=len(flat_values) - 1)
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            return flat_values[start_idx:end_idx + 1]

        return value

    def _resolve_text_table_binding(self, outputs: dict, binding: dict) -> str:
        """Resolve advanced table binding into text rows.

        Expected binding shape:
        {
            "name": "table_name",
            "table": {
                "columns": [
                    {"header": "A", "data_source": "vec_a"},
                    {"header": "B", "data_source": "vec_b"},
                    {"header": "Info", "data_source": "my_dict", "dict_mode": "items"}
                ],
                "column_separator": "\t",
                "row_separator": "\n",
                "missing_value": "",
                "include_header": true,
                "row_count_mode": "max"
            }
        }
        """
        table_cfg = binding.get('table', {}) if isinstance(binding.get('table', {}), dict) else {}
        columns_cfg = table_cfg.get('columns', [])
        if not isinstance(columns_cfg, list) or not columns_cfg:
            return ''

        column_separator = str(table_cfg.get('column_separator', '\t'))
        row_separator = str(table_cfg.get('row_separator', '\n'))
        missing_value = str(table_cfg.get('missing_value', ''))
        include_header = bool(table_cfg.get('include_header', True))
        row_count_mode = str(table_cfg.get('row_count_mode', 'max')).strip().lower() or 'max'

        headers: List[str] = []
        columns_data: List[List[str]] = []

        for idx, col_cfg in enumerate(columns_cfg):
            if not isinstance(col_cfg, dict):
                continue

            data_source = col_cfg.get('data_source')
            nested_key = col_cfg.get('nested_key')
            selector = col_cfg.get('selector', {'mode': 'value'})
            dict_mode = str(col_cfg.get('dict_mode', 'values'))
            value_format = str(col_cfg.get('value_format', ''))

            header = col_cfg.get('header')
            if header is None or str(header).strip() == '':
                header = str(col_cfg.get('name', data_source if data_source is not None else f"C{idx + 1}"))
            headers.append(str(header))

            temp_binding = {
                'data_source': data_source,
                'nested_key': nested_key,
                'selector': selector if isinstance(selector, dict) else {'mode': 'value'}
            }
            extracted = self._extract_text_binding_value(outputs, temp_binding)
            sequence = self._coerce_text_binding_sequence(extracted, dict_mode=dict_mode, value_format=value_format)
            columns_data.append(sequence)

        if not columns_data:
            return ''

        lengths = [len(col) for col in columns_data]
        if row_count_mode == 'min':
            row_count = min(lengths) if lengths else 0
        else:
            row_count = max(lengths) if lengths else 0

        lines: List[str] = []
        if include_header and headers:
            lines.append(column_separator.join(headers))

        for row_idx in range(row_count):
            row_values: List[str] = []
            for col in columns_data:
                row_values.append(col[row_idx] if row_idx < len(col) else missing_value)
            lines.append(column_separator.join(row_values))

        return row_separator.join(lines)

    def _resolve_text_section_content(self, outputs: dict, config: dict) -> str:
        """Resolve text template using configured bindings and extracted data."""
        return _svc_resolve_text_section_content(outputs, config)

    def _render_text_section(self, parent: ttk.Frame, instance_alias: str, section_data: dict):
        """Render a resolved text section in the analysis tab."""
        try:
            config = section_data.get('config', {}) if isinstance(section_data, dict) else {}
            raw_bindings = config.get('bindings', []) if isinstance(config, dict) else []
            bindings = raw_bindings if isinstance(raw_bindings, list) else []
            has_bindings = len(bindings) > 0
            fallback_template = config.get('text_template', '') if isinstance(config, dict) else ''
            if not isinstance(fallback_template, str):
                fallback_template = str(fallback_template)

            render_instance_alias = self._resolve_section_render_instance_alias(
                instance_alias,
                section_data=section_data,
            )

            if render_instance_alias not in self.analysis_data:
                if not has_bindings:
                    resolved_text = fallback_template
                else:
                    label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), foreground="gray")
                    label.pack(expand=True)
                    return
            else:
                execution_results = self.analysis_data[render_instance_alias].get('execution_results', {})
                if not execution_results:
                    if not has_bindings:
                        resolved_text = fallback_template
                    else:
                        label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), foreground="gray")
                        label.pack(expand=True)
                        return
                elif execution_results.get('status') != 'success':
                    if not has_bindings:
                        resolved_text = fallback_template
                    else:
                        log_hint = f"Execution failed. Check log: {_get_runtime_model_log_path()}"
                        label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.execution_failed_check_log", log_hint), foreground="red")
                        label.pack(expand=True)
                        return
                else:
                    outputs = self._get_execution_data_sources(execution_results, render_instance_alias)
                    resolved_text = self._resolve_text_section_content(outputs, config)

            text_widget = tk.Text(parent, wrap=tk.WORD, relief=tk.FLAT, borderwidth=0)
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.configure(font="TkFixedFont")
            text_widget.insert('1.0', resolved_text)
            text_widget.configure(state=tk.DISABLED)
        except Exception as e:
            label = ttk.Label(
                parent,
                text=self.language_manager.translate("ui.messages.error_rendering_text", "Error rendering text:") + f" {str(e)}",
                foreground="red"
            )
            label.pack(expand=True)

    def _get_execution_data_sources(self, execution_results: dict, instance_alias: str = None) -> dict:
        """Get combined execution data sources (inputs + routed inputs + outputs).

        Precedence on key collisions: direct inputs < routed inputs < outputs.
        """
        try:
            routed_inputs = self._resolve_routed_inputs(instance_alias, execution_results) if instance_alias else {}
            inherited_inputs = self._resolve_inherited_upstream_outputs(instance_alias, execution_results) if instance_alias else {}
            pf_output_keys = self._get_active_passforward_output_keys(instance_alias) if instance_alias else set()

            return _svc_build_execution_data_sources(
                execution_results=execution_results,
                instance_alias=instance_alias,
                routed_inputs=routed_inputs,
                inherited_inputs=inherited_inputs,
                active_passforward_output_keys=pf_output_keys,
            )
        except Exception:
            return {}

    def build_runtime_state_snapshot(self) -> Dict[str, Any]:
        """Build an immutable-style snapshot of runtime state for service consumers."""
        return {
            "methodology_list": list(self.methodology_list),
            "function_base_aliases": list(self.function_base_aliases),
            "routing_lines": copy.deepcopy(self.routing_lines),
            "analysis_data": copy.deepcopy(self.analysis_data),
            "function_configs": copy.deepcopy(self.function_configs),
            "gui_configs": copy.deepcopy(self.gui_configs),
            "workflow_control_aliases": set(self.workflow_control_aliases),
        }

    def _resolve_routed_inputs(self, instance_alias: str, target_execution_results: Optional[dict] = None) -> dict:
        """Resolve routed input values for a function instance from upstream execution outputs."""
        return _svc_resolve_routed_inputs(
            instance_alias=instance_alias,
            methodology_list=self.methodology_list,
            routing_lines=self.routing_lines,
            analysis_data=self.analysis_data,
            target_execution_results=target_execution_results,
        )

    def _resolve_inherited_upstream_outputs(self, instance_alias: str, target_execution_results: Optional[dict] = None) -> dict:
        """Resolve inherited upstream outputs for contextual analysis rendering.

        Uses the best matching upstream execution snapshot for the target history context.
        """
        return _svc_resolve_inherited_upstream_outputs(
            instance_alias=instance_alias,
            methodology_list=self.methodology_list,
            analysis_data=self.analysis_data,
            target_execution_results=target_execution_results,
            can_auto_route_between_fn=self._can_auto_route_between,
        )
    
    def _extract_axis_data(self, outputs: dict, axis_config: dict, indices: dict = None, ref_data: np.ndarray = None) -> Optional[np.ndarray]:
        """Extract data for an axis from execution outputs."""
        return _svc_extract_axis_data(outputs, axis_config, indices=indices, ref_data=ref_data)
    
    def _extract_sliced_data(self, data: np.ndarray, indices: dict) -> np.ndarray:
        """Extract sliced data from multi-dimensional array using indices dictionary.
        
        Args:
            data: NumPy array to slice
            indices: Dictionary mapping dimension to index (e.g., {0: 5, 1: 2})
        
        Returns:
            Sliced data array
        """
        return _svc_extract_sliced_data(data, indices)
    
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
        return _svc_compute_dimension_combinations(data_shape, specified_dims, ndim)
    
    def _create_table_navigation_controls(self, parent_frame: ttk.Frame, instance_alias: str,
                                         section_id: tuple, outputs: dict, config: dict,
                                         slice_state: dict,
                                         on_change: Optional[Callable[[], None]] = None,
                                         include_widget_refs: bool = True) -> None:
        """Create navigation controls for table data slicing.
        
        Similar to graph navigation but for table dimensions.
        """
        try:
            nav_axes = config.get('data_slicing', [])
            if not nav_axes:
                return
            
            # Get a representative source array to determine shape/bounds.
            # Support both single-source tables (data_source) and multi-column tables (columns).
            data = None
            data_source = config.get('data_source')
            if data_source and data_source in outputs:
                data = outputs[data_source]
            else:
                columns_cfg = config.get('columns', [])
                if isinstance(columns_cfg, list):
                    for col_spec in columns_cfg:
                        if not isinstance(col_spec, dict):
                            continue
                        col_source = col_spec.get('data_source')
                        col_nested = col_spec.get('nested_key')
                        if not col_source:
                            continue
                        candidate = self._get_data_from_source(outputs, col_source, col_nested)
                        if candidate is None:
                            continue
                        data = candidate
                        break

            if data is None:
                return

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
            
            # Reserve fixed title width so prev/next buttons stay aligned across items.
            nav_label_width_raw = self.language_manager.translate("ui.layout.navigation_title_width", 18)
            try:
                nav_label_width = int(nav_label_width_raw)
            except (TypeError, ValueError):
                nav_label_width = 18
            nav_label_width = max(10, min(nav_label_width, 40))

            # Create navigation frame
            nav_frame = ttk.Frame(parent_frame)
            nav_frame.pack(anchor='w', padx=4, pady=1)
            
            # For each navigable axis, create controls
            for nav_item in nav_axes:
                # Parse navigation item
                if isinstance(nav_item, dict):
                    axis_name = nav_item.get('name', 'Dimension')
                    dimension = nav_item.get('dimension', 0)
                    show_nav = nav_item.get('show_navigation_menu', True)
                    nav_id = self._normalize_navigation_id(nav_item)
                    nav_labels_source = nav_item.get('labels_source')
                    nav_labels_dimension = nav_item.get('labels_dimension', None)
                    nav_labels_index_dimensions = nav_item.get('labels_index_dimensions')
                else:
                    # Old format: just a string
                    axis_name = nav_item
                    dimension = nav_axes.index(nav_item)
                    show_nav = True
                    nav_id = None
                    nav_labels_source = None
                    nav_labels_dimension = None
                    nav_labels_index_dimensions = None
                
                # Skip if navigation is disabled
                if not show_nav:
                    continue
                
                axis_frame = ttk.Frame(nav_frame)
                axis_frame.pack(anchor='w', pady=1)
                
                # Get max index from data shape
                max_index = data.shape[dimension] - 1 if dimension < len(data.shape) else 0
                
                # Get current index
                indices = slice_state.get('indices', {})
                current_index = indices.get(dimension, 0)
                
                # Axis label
                label_text = f"{axis_name}: {current_index + 1}/{max_index + 1}"
                show_inline_pairing_hint = not bool(config.get('variable_labels'))
                pairing_hint = self._get_table_pairing_hint(outputs, config, axis_name, dimension, current_index)
                if show_inline_pairing_hint and pairing_hint:
                    label_text = f"{label_text} ({pairing_hint})"
                label = ttk.Label(axis_frame, text=label_text, width=nav_label_width)
                label.pack(side=tk.LEFT, padx=3)
                
                # Previous button
                prev_btn = ttk.Button(
                    axis_frame,
                    text="<",
                    width=3,
                    command=lambda d=dimension, an=axis_name, m=max_index, nid=nav_id, cb=on_change: self._on_table_navigate_slice(
                        instance_alias, section_id, -1, d, an, m, nid, cb
                    )
                )
                prev_btn.pack(side=tk.LEFT, padx=1)
                
                # Index display
                index_label = ttk.Label(axis_frame, text=str(current_index + 1), width=3)
                index_label.pack(side=tk.LEFT, padx=1)
                
                # Next button
                next_btn = ttk.Button(
                    axis_frame,
                    text=">",
                    width=3,
                    command=lambda d=dimension, an=axis_name, m=max_index, nid=nav_id, cb=on_change: self._on_table_navigate_slice(
                        instance_alias, section_id, 1, d, an, m, nid, cb
                    )
                )
                next_btn.pack(side=tk.LEFT, padx=1)

                # Variable labels - show current value label after buttons (graph-style)
                var_labels_config = nav_labels_source if nav_labels_source else config.get('variable_labels')
                if var_labels_config:
                    label_dim = int(nav_labels_dimension) if nav_labels_dimension is not None else int(dimension)
                    var_label_text = self._get_variable_label_with_indices(
                        outputs,
                        var_labels_config,
                        label_dim,
                        current_index,
                        indices=indices,
                        index_dimensions=nav_labels_index_dimensions,
                    )
                    if var_label_text:
                        var_label = ttk.Label(axis_frame, text=f"[{var_label_text}]", foreground="gray")
                        var_label.pack(side=tk.LEFT, padx=5)
                        if include_widget_refs:
                            if not hasattr(self, '_var_labels'):
                                self._var_labels = {}
                            var_label_key = (instance_alias, section_id, dimension, axis_name, None)
                            self._var_labels[var_label_key] = (
                                var_label,
                                var_labels_config,
                                label_dim,
                                nav_labels_index_dimensions,
                            )
                
                # Store reference for updates
                if include_widget_refs:
                    if not hasattr(self, '_table_nav_labels'):
                        self._table_nav_labels = {}
                    label_key = (instance_alias, section_id, dimension, axis_name)
                    self._table_nav_labels[label_key] = (index_label, label)
        
        except Exception as e:
            print(f"Error creating table navigation controls: {str(e)}")

    def _get_table_pairing_hint(self, outputs: dict, config: dict, axis_name: str,
                                dimension: int, current_index: int) -> str:
        """Return a concise pairing hint (e.g., C2->Y1) for table slice navigation."""
        try:
            if dimension != 1:
                return ""

            # Only show pairing hints when explicitly configured;
            labels_source = config.get('pairing_labels_source')
            if not labels_source:
                return ""
            labels_value = self._get_data_from_source(outputs, labels_source)
            if isinstance(labels_value, np.ndarray):
                labels_value = np.asarray(labels_value).reshape(-1).tolist()
            if isinstance(labels_value, list) and int(current_index) < len(labels_value):
                label_text = str(labels_value[int(current_index)]).strip()
                if label_text:
                    return label_text

            axis_text = str(axis_name or "").lower()
            if "pair" not in axis_text and "y" not in axis_text:
                return ""

            mapping_source = config.get('pairing_mapping_source', 'component_y_mapping')
            mapping_raw = self._get_data_from_source(outputs, mapping_source)
            if not isinstance(mapping_raw, dict) or not mapping_raw:
                return ""

            target_y_col = int(current_index) + 1
            matched_components = []
            for comp_key, y_col_value in mapping_raw.items():
                try:
                    comp_idx = int(comp_key)
                    y_col = int(y_col_value)
                except (TypeError, ValueError):
                    continue
                if y_col == target_y_col:
                    matched_components.append(comp_idx)

            if not matched_components:
                return f"Y{target_y_col} unpaired"

            matched_components = sorted(set(int(c) for c in matched_components))
            if len(matched_components) == 1:
                return f"C{matched_components[0]}->Y{target_y_col}"
            comp_text = ",".join(str(c) for c in matched_components)
            return f"C[{comp_text}]->Y{target_y_col}"
        except Exception:
            return ""
    
    def _on_table_navigate_slice(self, instance_alias: str, section_id: tuple, direction: int,
                                 dimension: int, axis_name: str, max_index: int,
                                 nav_id: Optional[str] = None,
                                 on_change: Optional[Callable[[], None]] = None) -> None:
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

            section_data = self._get_section_data_by_id(instance_alias, section_id)
            _render_alias, _execution_results, outputs = self._get_section_render_outputs(
                instance_alias,
                section_id,
                section_data=section_data,
            )
            
            # Update label if it exists
            source_nav_item = {
                'name': axis_name,
                'dimension': dimension,
                'axis': None,
                'nav_id': nav_id,
            }
            self._update_table_nav_widgets(
                instance_alias,
                section_id,
                source_nav_item,
                new_idx,
                max_index,
                outputs
            )

            if hasattr(self, '_var_labels'):
                stale_var_keys = []
                for key, var_tuple in list(self._var_labels.items()):
                    if len(key) != 5:
                        continue
                    key_instance, key_section, key_dimension, _axis_name, key_target_axis = key
                    if key_instance != instance_alias or key_section != section_id:
                        continue
                    if key_dimension != dimension or key_target_axis is not None:
                        continue
                    try:
                        if not isinstance(var_tuple, tuple) or len(var_tuple) < 3:
                            stale_var_keys.append(key)
                            continue
                        var_label, var_labels_config, dim = var_tuple[0], var_tuple[1], var_tuple[2]
                        index_dims = var_tuple[3] if len(var_tuple) > 3 else None
                        if not var_label.winfo_exists():
                            stale_var_keys.append(key)
                            continue
                        var_label_text = self._get_variable_label_with_indices(
                            outputs,
                            var_labels_config,
                            dim,
                            new_idx,
                            indices=indices,
                            index_dimensions=index_dims,
                        )
                        if var_label_text:
                            var_label.config(text=f"[{var_label_text}]")
                        else:
                            var_label.config(text="")
                    except Exception:
                        stale_var_keys.append(key)

                for stale_key in stale_var_keys:
                    self._var_labels.pop(stale_key, None)

            synced_table_sections = self._sync_table_nav_group(
                instance_alias,
                section_id,
                source_nav_item,
                new_idx,
                max_index
            )
            synced_graph_sections = self._sync_graph_nav_group(
                instance_alias,
                section_id,
                source_nav_item,
                new_idx,
                max_index
            )

            # Refresh all linked sections that changed.
            sections_to_refresh = [section_id] + synced_table_sections + synced_graph_sections
            seen_sections = set()
            for sid in sections_to_refresh:
                if sid in seen_sections:
                    continue
                seen_sections.add(sid)
                if sid in self.analysis_data.get(instance_alias, {}).get('table_slices', {}):
                    self._refresh_table(instance_alias, sid)
                elif sid in self.analysis_data.get(instance_alias, {}).get('graph_slices', {}):
                    self._update_graph_with_slice(instance_alias, sid, dimension)

            # Rebuild active navigation panel to reflect synchronized state.
            self._render_active_section_navigation(instance_alias)

            if callable(on_change):
                on_change()
        
        except Exception as e:
            print(f"Error navigating table slice: {str(e)}")

    def _get_table_navigation_data_array(self, outputs: dict, config: dict) -> Optional[np.ndarray]:
        """Resolve representative ndarray used for table navigation bounds."""
        data_source = config.get('data_source')
        data = None
        if data_source:
            nested_key = config.get('nested_key')
            data = self._get_data_from_source(outputs, data_source, nested_key)

        if data is None:
            columns_cfg = config.get('columns', [])
            if isinstance(columns_cfg, list):
                for col_spec in columns_cfg:
                    if not isinstance(col_spec, dict):
                        continue
                    col_source = col_spec.get('data_source')
                    if not col_source:
                        continue
                    col_nested = col_spec.get('nested_key')
                    candidate = self._get_data_from_source(outputs, col_source, col_nested)
                    if candidate is not None:
                        data = candidate
                        break

        if isinstance(data, np.ndarray):
            return data

        try:
            return np.array(data)
        except (ValueError, TypeError):
            return None

    def _update_table_nav_widgets(self, instance_alias: str, section_id: tuple, nav_item: dict,
                                  new_index: int, max_index: int, outputs: dict) -> None:
        """Refresh table navigation widgets and labels for one nav item."""
        dimension = nav_item.get('dimension', 0)
        axis_name = nav_item.get('name', 'Dimension')

        if hasattr(self, '_table_nav_labels'):
            label_key = (instance_alias, section_id, dimension, axis_name)
            if label_key in self._table_nav_labels:
                index_label, full_label = self._table_nav_labels[label_key]
                try:
                    index_label.config(text=str(new_index + 1))
                except Exception:
                    pass

                analysis_info = self.analysis_data.get(instance_alias, {})
                pages = analysis_info.get('pages', []) if isinstance(analysis_info, dict) else []
                page_idx, section_idx = section_id
                section_cfg = {}
                if 0 <= page_idx < len(pages):
                    sections = pages[page_idx].get('sections', []) if isinstance(pages[page_idx], dict) else []
                    if 0 <= section_idx < len(sections) and isinstance(sections[section_idx], dict):
                        section_cfg = sections[section_idx].get('config', {}) or {}

                label_text = f"{axis_name}: {new_index + 1}/{max_index + 1}"
                show_inline_pairing_hint = not bool(section_cfg.get('variable_labels'))
                pairing_hint = self._get_table_pairing_hint(outputs, section_cfg, axis_name, dimension, new_index)
                if show_inline_pairing_hint and pairing_hint:
                    label_text = f"{label_text} ({pairing_hint})"
                try:
                    full_label.config(text=label_text)
                except Exception:
                    pass

    def _sync_table_nav_group(self, instance_alias: str, source_section_id: tuple,
                              source_nav_item: dict, new_index: int,
                              expected_max_index: int) -> List[tuple]:
        """Sync table nav items with the same nav_id when member ranges match."""
        nav_id = self._normalize_navigation_id(source_nav_item)
        if not nav_id:
            return []

        analysis_info = self.analysis_data.get(instance_alias, {})
        table_slices = analysis_info.get('table_slices', {})
        if not isinstance(table_slices, dict):
            table_slices = {}
            analysis_info['table_slices'] = table_slices

        source_section_data = self._get_section_data_by_id(instance_alias, source_section_id)
        source_render_alias = self._resolve_section_render_instance_alias(
            instance_alias,
            section_data=source_section_data,
            section_id=source_section_id,
        )

        # Build candidates from configured table sections so sync works even when
        # a table section has not been rendered yet (and thus has no slice state).
        pages = analysis_info.get('pages', []) if isinstance(analysis_info, dict) else []
        candidates = []
        for page_idx, page_data in enumerate(pages):
            if not isinstance(page_data, dict):
                continue
            sections = page_data.get('sections', [])
            if not isinstance(sections, list):
                continue

            for section_idx, section_data in enumerate(sections):
                if not isinstance(section_data, dict):
                    continue
                if section_data.get('type') != 'table':
                    continue

                section_id = (page_idx, section_idx)
                section_config = section_data.get('config', {}) if isinstance(section_data.get('config', {}), dict) else {}

                candidate_render_alias = self._resolve_section_render_instance_alias(
                    instance_alias,
                    section_data=section_data,
                    section_id=section_id,
                )
                if candidate_render_alias != source_render_alias:
                    continue

                _alias, _exec_results, candidate_outputs = self._get_section_render_outputs(
                    instance_alias,
                    section_id,
                    section_data=section_data,
                )

                section_state = table_slices.get(section_id)
                if not isinstance(section_state, dict):
                    section_state = {
                        'indices': {},
                        'data_slicing': section_config.get('data_slicing', []),
                        'outputs': candidate_outputs,
                        'config': section_config,
                    }
                    table_slices[section_id] = section_state

                section_state['outputs'] = candidate_outputs
                section_state['config'] = section_config
                if not isinstance(section_state.get('indices'), dict):
                    section_state['indices'] = {}

                config = section_state.get('config', {})
                nav_axes = config.get('data_slicing', []) if isinstance(config, dict) else []
                if not isinstance(nav_axes, list):
                    continue

                for nav_item in nav_axes:
                    if not isinstance(nav_item, dict):
                        continue
                    dim_value = nav_item.get('dimension')
                    if dim_value is None:
                        continue
                    try:
                        dim = int(dim_value)
                    except (TypeError, ValueError):
                        continue
                    if dim not in section_state['indices']:
                        section_state['indices'][dim] = nav_item.get('default', 0)

                data = self._get_table_navigation_data_array(candidate_outputs, config)
                if data is None or not isinstance(data, np.ndarray):
                    continue

                for nav_item in nav_axes:
                    if not isinstance(nav_item, dict):
                        continue
                    if self._normalize_navigation_id(nav_item) != nav_id:
                        continue

                    dim_value = nav_item.get('dimension')
                    if dim_value is None:
                        continue
                    try:
                        dimension = int(dim_value)
                    except (TypeError, ValueError):
                        continue
                    if dimension < 0 or dimension >= len(data.shape):
                        continue

                    max_index = data.shape[dimension] - 1
                    candidates.append((section_id, section_state, nav_item, max_index, candidate_outputs))

        if len(candidates) == 0:
            return []

        ranges = {entry[3] for entry in candidates}
        if len(ranges) != 1 or expected_max_index not in ranges:
            return []

        changed_sections = set()
        for section_id, section_state, nav_item, max_index, candidate_outputs in candidates:
            indices = section_state.get('indices', {})
            dimension = int(nav_item.get('dimension', 0))
            try:
                current_index = int(indices.get(dimension, nav_item.get('default', 0)))
            except (TypeError, ValueError):
                current_index = 0
            current_index = max(0, min(current_index, max_index))

            if current_index != new_index:
                indices[dimension] = new_index
                section_state['indices'] = indices
                changed_sections.add(section_id)

            self._update_table_nav_widgets(
                instance_alias,
                section_id,
                nav_item,
                new_index,
                max_index,
                candidate_outputs
            )

        changed_sections.discard(source_section_id)
        return list(changed_sections)

    def _resolve_table_slice_title(self, outputs: dict, config: dict, indices: dict) -> str:
        """Resolve a slice-aware subtitle/title token for tables (e.g., selected Y label)."""
        try:
            source = config.get('slice_title_source')
            if not source:
                return ""
            values = self._get_data_from_source(outputs, source)
            if values is None:
                return ""
            dim = int(config.get('slice_title_dimension', 1))
            idx = int(indices.get(dim, 0)) if isinstance(indices, dict) else 0
            token = ""

            index_dims = config.get('slice_title_index_dimensions')
            if isinstance(index_dims, (list, tuple)) and len(index_dims) > 0 and isinstance(indices, dict):
                try:
                    cursor = values
                    for dim_id_raw in index_dims:
                        dim_id = int(dim_id_raw)
                        idx_value = idx if dim_id == dim else int(indices.get(dim_id, 0))
                        if isinstance(cursor, np.ndarray):
                            cursor = cursor[idx_value]
                        elif isinstance(cursor, (list, tuple)):
                            cursor = cursor[idx_value]
                        else:
                            cursor = None
                            break
                    if cursor is not None and not isinstance(cursor, (list, tuple, np.ndarray)):
                        token = str(cursor).strip()
                except Exception:
                    token = ""

            if not token:
                arr = np.asarray(values).reshape(-1).tolist()
                if not arr:
                    return ""
                if idx < 0 or idx >= len(arr):
                    return ""
                token = str(arr[idx]).strip()

            if not token:
                return ""
            prefix = str(config.get('slice_title_prefix', '')).strip()
            if prefix:
                return f"{prefix}: {token}"
            return token
        except Exception:
            return ""
    
    def _render_table_section(self, parent: ttk.Frame, instance_alias: str, section_data: dict,
                              section_idx: int = 0,
                              section_id_override: Optional[Tuple[int, int]] = None):
        """Render a comprehensive data table with sorting, filtering, and formatting."""
        try:
            config = section_data.get('config', {})
            data_source = config.get('data_source')

            # Keep custom tab state independent, but resolve data context from source function.
            if isinstance(section_id_override, tuple) and len(section_id_override) == 2:
                section_id = section_id_override
            else:
                current_page = self.analysis_data.get(instance_alias, {}).get('current_page', 0)
                section_id = (current_page, section_idx)
            render_instance_alias = self._resolve_section_render_instance_alias(
                instance_alias,
                section_data=section_data,
                section_id=section_id,
            )
            
            # Get execution results
            if render_instance_alias not in self.analysis_data:
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), foreground="gray")
                label.pack(expand=True)
                return
            
            execution_results = self.analysis_data[render_instance_alias].get('execution_results', {})
            if not execution_results:
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), foreground="gray")
                label.pack(expand=True)
                return
            
            if execution_results.get('status') != 'success':
                log_hint = f"Execution failed. Check log: {_get_runtime_model_log_path()}"
                label = ttk.Label(parent, text=self.language_manager.translate("ui.messages.execution_failed_check_log", log_hint), foreground="red")
                label.pack(expand=True)
                return
            
            outputs = self._get_execution_data_sources(execution_results, render_instance_alias)

            # Initialize table slices state for data slicing support
            # Use (page, section_idx) tuple as section ID to ensure unique state per page
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
            nav_axes = config.get('data_slicing', [])
            current_indices = current_slice.get('indices', {})
            
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
                        continue
                    
                    # Convert to numpy array if needed
                    if not isinstance(col_data, np.ndarray):
                        col_data = np.array(col_data)

                    # Apply data slicing before flattening so navigation works on
                    # source dimensions (e.g., Y pairing/component), not table columns.
                    if nav_axes:
                        col_data = self._extract_sliced_data(col_data, current_indices)
                    
                    # Ensure 1D
                    col_data = col_data.flatten()
                    
                    data_columns.append(col_data)
                    col_headers.append(col_name)

                # Skip rendering only if no columns are available for current outputs
                if not data_columns:
                    label = ttk.Label(
                        parent,
                        text=self.language_manager.translate(
                            "ui.messages.no_columns_available_current_run",
                            "No configured table columns are available for the current run."
                        ),
                        foreground="gray"
                    )
                    label.pack(expand=True)
                    return
                
                # Stack columns into a 2D array
                try:
                    min_len = min(len(col) for col in data_columns)
                    trimmed_columns = [col[:min_len] for col in data_columns]
                    data = np.column_stack(trimmed_columns)
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
            
            # Extract sliced data if data_slicing is configured
            if nav_axes:
                # For multi-column tables, slicing was already applied per source column.
                if not columns_config:
                    data = self._extract_sliced_data(data, current_indices)
                # Update slice info in config for display
                current_index = list(current_indices.values())[0] if current_indices else 0
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
            if isinstance(row_headers, str):
                resolved_row_headers = self._get_data_from_source(outputs, row_headers)
                if resolved_row_headers is not None:
                    row_headers = np.asarray(resolved_row_headers).flatten().tolist()
            elif isinstance(row_headers, dict):
                rhs_source = row_headers.get('data_source')
                rhs_nested = row_headers.get('nested_key')
                if rhs_source:
                    resolved_row_headers = self._get_data_from_source(outputs, rhs_source, rhs_nested)
                    if resolved_row_headers is not None:
                        row_headers = np.asarray(resolved_row_headers).flatten().tolist()
            
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
                slice_title = self._resolve_table_slice_title(outputs, config, current_indices)
                title_text = table_title
                if slice_title:
                    title_text = f"{table_title} - {slice_title}"
                title_label = ttk.Label(main_frame, text=title_text, font=('Arial', 10, 'bold'))
                title_label.pack(anchor='w', pady=(0, 5))
            
            # Add info bar (shape, stats). Some tables (e.g., list-of-dicts)
            # are object arrays and do not support numeric reductions.
            numeric_data = None
            try:
                numeric_data = np.asarray(data, dtype=float)
            except Exception:
                numeric_data = None

            if numeric_data is not None and numeric_data.size > 0:
                finite_vals = numeric_data[np.isfinite(numeric_data)]
                if finite_vals.size > 0:
                    info_text = (
                        f"Shape: {data.shape} | Type: {data.dtype} | "
                        f"Min: {np.min(finite_vals):.4f} | Max: {np.max(finite_vals):.4f} | "
                        f"Mean: {np.mean(finite_vals):.4f}"
                    )
                else:
                    info_text = (
                        f"Shape: {data.shape} | Type: {data.dtype} | "
                        f"Elements: {data.size} | Numeric finite values: 0"
                    )
            else:
                info_text = (
                    f"Shape: {data.shape} | Type: {data.dtype} | "
                    f"Elements: {data.size} | Non-numeric/object data"
                )
            info_label = ttk.Label(main_frame, text=info_text, font=('Arial', 8), foreground='gray')
            info_label.pack(anchor='w', pady=(0, 5))
            
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
            value_format = str(config.get('value_format', '')).strip()
            for row_idx in range(num_rows):
                if row_headers is not None and row_idx < len(row_headers):
                    row_label = str(row_headers[row_idx])
                else:
                    row_label = str(row_idx + 1)  # Display 1-based index for user
                
                # Format values
                values = []
                for col_idx in range(num_cols):
                    val = display_data[row_idx, col_idx]
                    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
                        if value_format:
                            try:
                                formatted = format(val, value_format)
                            except Exception:
                                formatted = self._format_table_numeric_value(val, decimal_places)
                        else:
                            formatted = self._format_table_numeric_value(val, decimal_places)
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

    def _format_table_numeric_value(self, value: Any, decimal_places: int) -> str:
        """Format table numeric values with adaptive fixed/scientific notation."""
        try:
            numeric = float(value)
        except Exception:
            return str(value)

        if not np.isfinite(numeric):
            return str(value)

        if numeric == 0.0:
            return "0"

        # Adaptive table formatting defaults to up to 4 fractional digits.
        # This keeps tables readable while preserving tiny values via scientific notation.
        default_places = 4
        try:
            requested_places = int(decimal_places)
        except Exception:
            requested_places = default_places
        places = min(default_places, max(0, requested_places))

        abs_value = abs(numeric)
        # Engineering-oriented defaults: scientific notation outside [1e-2, 1e3).
        small_threshold = 1e-2
        large_threshold = 1e3

        # Keep very small/large numbers visible with scientific notation.
        if abs_value < small_threshold or abs_value >= large_threshold:
            return f"{numeric:.{max(1, places)}e}"

        fixed = f"{numeric:.{places}f}"
        if "." in fixed:
            fixed = fixed.rstrip("0").rstrip(".")
        return "0" if fixed in {"-0", "-0.0"} else fixed
    
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
            _set_window_icon(stats_window, "Icon")
            stats_window.title(f"{title} - {self.language_manager.translate('ui.labels.statistics', 'Statistics')}")
            stats_window.geometry("500x400")
            
            # Calculate statistics; fallback gracefully for non-numeric/object data.
            numeric_data = None
            try:
                numeric_data = np.asarray(data, dtype=float)
            except Exception:
                numeric_data = None

            if numeric_data is not None and numeric_data.size > 0:
                flat = numeric_data.reshape(-1)
                finite = flat[np.isfinite(flat)]
                nan_count = int(np.isnan(flat).sum())
                inf_count = int(np.isinf(flat).sum())

                if finite.size > 0:
                    stats_text = f"""
DATA STATISTICS
===============

Shape: {data.shape}
Data Type: {data.dtype}

Basic Statistics:
    Min: {np.min(finite):.6f}
    Max: {np.max(finite):.6f}
    Mean: {np.mean(finite):.6f}
    Median: {np.median(finite):.6f}
    Std Dev: {np.std(finite):.6f}
    Variance: {np.var(finite):.6f}

Quartiles:
    Q1 (25%): {np.percentile(finite, 25):.6f}
    Q2 (50%): {np.percentile(finite, 50):.6f}
    Q3 (75%): {np.percentile(finite, 75):.6f}

Count:
    Total Elements: {flat.size}
    Finite Elements: {finite.size}
    Non-zero Elements (finite): {np.count_nonzero(finite)}
    Zero Elements (finite): {np.sum(finite == 0)}
    NaN Elements: {nan_count}
    Inf Elements: {inf_count}
                    """
                else:
                    stats_text = f"""
DATA STATISTICS
===============

Shape: {data.shape}
Data Type: {data.dtype}

No finite numeric values were found.

Count:
    Total Elements: {flat.size}
    NaN Elements: {nan_count}
    Inf Elements: {inf_count}
                    """
            else:
                object_count = int(np.asarray(data, dtype=object).size)
                stats_text = f"""
DATA STATISTICS
===============

Shape: {data.shape}
Data Type: {data.dtype}

This table contains non-numeric/object values.
Descriptive numeric statistics are not available.

Count:
    Total Elements: {object_count}
                """
            
            text_widget = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.insert('1.0', stats_text)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"Error showing statistics: {str(e)}")
    
    def _refresh_table(self, instance_alias: str, section_id: tuple) -> None:
        """Refresh the table display with current slicing."""
        try:
            analysis_info = self.analysis_data.get(instance_alias)
            if not analysis_info:
                return

            rendered_sections = analysis_info.get('rendered_sections', [])
            entry = next((e for e in rendered_sections if e.get('section_id') == section_id), None)
            if entry is None:
                return

            container = entry.get('container')
            section_data = entry.get('section_data')
            if container is None or section_data is None:
                return

            if not container.winfo_exists():
                return

            # Clear existing table content
            for child in container.winfo_children():
                child.destroy()

            section_idx = section_id[1]
            self._render_section(container, instance_alias, section_data, section_idx)

        except Exception as e:
            print(f"Error refreshing table: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _show_section_popup(self, instance_alias: str, section_idx: int, section_data: dict):
        """Show a popup window with the same content as a section."""
        try:
            # Create a new popup window
            popup = tk.Toplevel(self.root)
            _set_window_icon(popup, "Icon")
            popup.title(
                self.language_manager.translate("ui.labels.section_prefix", "Section:") +
                f" {section_data.get('config', {}).get('title', self.language_manager.translate('ui.labels.section', 'Section'))}"
            )
            popup.geometry("900x700")

            popup_graph_font_scale = self.graph_font_scale
            popup_graph_font_scale_var = tk.StringVar(value=self._format_graph_font_scale_value(popup_graph_font_scale))
            popup_page = self.analysis_data.get(instance_alias, {}).get('current_page', 0)
            popup_section_id = (popup_page, section_idx)

            content_frame = ttk.Frame(popup)
            nav_host_frame = ttk.Frame(popup)

            def render_popup_content():
                for widget in content_frame.winfo_children():
                    widget.destroy()
                for widget in nav_host_frame.winfo_children():
                    widget.destroy()

                self._render_section(
                    content_frame,
                    instance_alias,
                    section_data,
                    section_idx,
                    is_popup=True,
                    graph_font_scale_override=popup_graph_font_scale if section_data.get('type') == 'graph' else None,
                    popup_refresh_callback=render_popup_content,
                    section_id_override=popup_section_id,
                )

                config = section_data.get('config', {}) if isinstance(section_data, dict) else {}
                nav_axes = config.get('data_slicing', []) if isinstance(config, dict) else []
                section_type = section_data.get('type') if isinstance(section_data, dict) else None
                if section_type not in ('graph', 'table') or not nav_axes:
                    return

                nav_panel = ttk.LabelFrame(
                    nav_host_frame,
                    text=self.language_manager.translate("ui.labels.data_slicing", "Data Slicing"),
                    padding=5,
                )
                nav_panel.pack(side=tk.BOTTOM, fill=tk.X, anchor='sw', padx=10, pady=(0, 10))

                _render_alias, _execution_results, outputs = self._get_section_render_outputs(
                    instance_alias,
                    popup_section_id,
                    section_data=section_data,
                )
                analysis_info = self.analysis_data.get(instance_alias, {})

                if section_type == 'graph':
                    slice_state = analysis_info.get('graph_slices', {}).get(popup_section_id)
                    if isinstance(slice_state, dict):
                        self._create_navigation_controls(
                            nav_panel,
                            instance_alias,
                            popup_section_id,
                            outputs,
                            config,
                            slice_state,
                            on_change=render_popup_content,
                            include_widget_refs=False,
                        )
                else:
                    slice_state = analysis_info.get('table_slices', {}).get(popup_section_id)
                    if isinstance(slice_state, dict):
                        self._create_table_navigation_controls(
                            nav_panel,
                            instance_alias,
                            popup_section_id,
                            outputs,
                            config,
                            slice_state,
                            on_change=render_popup_content,
                            include_widget_refs=False,
                        )

            if section_data.get('type') == 'graph':
                popup_menubar = tk.Menu(popup)
                popup.config(menu=popup_menubar)

                popup_settings_menu = tk.Menu(popup_menubar, tearoff=0)
                popup_menubar.add_cascade(
                    label=self.language_manager.translate("menu.settings", "Settings"),
                    menu=popup_settings_menu
                )

                popup_graph_font_menu = tk.Menu(popup_settings_menu, tearoff=0)
                popup_settings_menu.add_cascade(
                    label=self.language_manager.translate("menu.graph_font_size", "Graph Font Size"),
                    menu=popup_graph_font_menu
                )

                def _set_popup_graph_font_scale(scale: float):
                    nonlocal popup_graph_font_scale
                    popup_graph_font_scale = self._normalize_graph_font_scale(scale)
                    popup_graph_font_scale_var.set(self._format_graph_font_scale_value(popup_graph_font_scale))
                    render_popup_content()

                self._populate_graph_font_scale_menu(
                    popup_graph_font_menu,
                    popup_graph_font_scale_var,
                    _set_popup_graph_font_scale
                )
            
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
                    command=lambda sid=popup_section_id: self._save_section_graph_as_csv(
                        instance_alias,
                        section_idx,
                        section_id_override=sid,
                    )
                )
                save_data_btn.pack(side=tk.LEFT, padx=5)
            
            # Add close button on the right
            close_btn = ttk.Button(button_frame, text=self.language_manager.translate("ui.buttons.close", "Close"), command=popup.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)
            
            # Create a frame for the content
            content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            nav_host_frame.pack(fill=tk.X, side=tk.BOTTOM)

            # Render the section in the popup (with is_popup=True to skip popup button)
            render_popup_content()
        
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
    
    def _sanitize_export_token(self, value: Any, fallback: str = "data") -> str:
        """Sanitize a token for filesystem-safe export filenames."""
        token = str(value).strip() if value is not None else ""
        token = "".join(ch if (ch.isalnum() or ch in ("-", "_", " ")) else "_" for ch in token)
        token = re.sub(r"\s+", "_", token).strip("_")
        return token if token else fallback

    def _normalize_export_matrix(self, value: Any) -> Optional[np.ndarray]:
        """Normalize scalar/1D/2D values into a 2D ndarray for CSV export."""
        if isinstance(value, np.ndarray):
            arr = value
        else:
            try:
                arr = np.asarray(value, dtype=object)
            except Exception:
                return None

        if arr.ndim == 0:
            return np.asarray([[arr.item()]], dtype=object)
        if arr.ndim == 1:
            return np.asarray(arr.reshape(-1, 1), dtype=object)
        if arr.ndim == 2:
            return np.asarray(arr, dtype=object)
        return None

    def _iter_export_matrices(self, value: Any, base_name: str) -> List[Tuple[str, np.ndarray]]:
        """Expand any value into one or more 2D matrices ready for CSV export."""
        matrices: List[Tuple[str, np.ndarray]] = []

        def _append_matrix(name: str, matrix_value: Any) -> None:
            matrix = self._normalize_export_matrix(matrix_value)
            if matrix is not None:
                matrices.append((name, matrix))

        def _walk(current_value: Any, current_name: str) -> None:
            if current_value is None:
                return

            if isinstance(current_value, dict):
                for key, nested in current_value.items():
                    key_token = self._sanitize_export_token(key, "key")
                    _walk(nested, f"{current_name}_{key_token}")
                return

            if isinstance(current_value, np.ndarray):
                arr = current_value
            elif isinstance(current_value, (list, tuple)):
                try:
                    arr = np.asarray(current_value)
                except Exception:
                    arr = None

                # Ragged lists (dtype=object) are exported element by element.
                if arr is None or arr.dtype == object:
                    if not current_value:
                        return
                    scalar_only = all(not isinstance(item, (list, tuple, np.ndarray, dict)) for item in current_value)
                    if scalar_only:
                        _append_matrix(current_name, np.asarray(current_value, dtype=object).reshape(-1, 1))
                    else:
                        for idx, nested in enumerate(current_value):
                            _walk(nested, f"{current_name}_item{idx + 1}")
                    return
            else:
                _append_matrix(current_name, np.asarray([[current_value]], dtype=object))
                return

            if arr.ndim <= 2:
                _append_matrix(current_name, arr)
                return

            # For 3D+ arrays, export one matrix per leading-dimension slice.
            for leading_indices in np.ndindex(arr.shape[:-2]):
                slice_name = "_".join(f"d{dim + 1}i{idx + 1}" for dim, idx in enumerate(leading_indices))
                export_name = f"{current_name}_{slice_name}" if slice_name else current_name
                _append_matrix(export_name, arr[leading_indices])

        _walk(value, self._sanitize_export_token(base_name, "data"))
        return matrices

    def _write_matrix_csv(self, file_path: Path, matrix: np.ndarray) -> None:
        """Write a 2D matrix to CSV preserving mixed numeric/string/object values."""
        import csv

        with open(file_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            matrix_obj = np.asarray(matrix, dtype=object)
            for row in matrix_obj:
                serialized_row = []
                for value in row:
                    if isinstance(value, np.generic):
                        serialized_row.append(value.item())
                    else:
                        serialized_row.append(value)
                writer.writerow(serialized_row)

    def _humanize_export_item_name(self, item_name: str) -> str:
        """Convert internal export item names into user-facing checkbox labels."""
        raw = str(item_name or "").strip()
        if not raw:
            return self.language_manager.translate("ui.labels.export_item_data", "Data")

        canonical = self._canonical_transpose_group_key(raw)
        tokens = [tok for tok in canonical.split("_") if tok]
        if not tokens:
            return self.language_manager.translate("ui.labels.export_item_data", "Data")

        label_map = {
            "x": self.language_manager.translate("ui.labels.export_item_x", "X"),
            "y": self.language_manager.translate("ui.labels.export_item_y", "Y"),
            "z": self.language_manager.translate("ui.labels.export_item_z", "Z"),
            "class": self.language_manager.translate("ui.labels.export_item_class", "Class"),
            "labels": self.language_manager.translate("ui.labels.export_item_labels", "Labels"),
            "data": self.language_manager.translate("ui.labels.export_item_data", "Data"),
        }

        tail = tokens[-1]
        if tail in label_map:
            if len(tokens) == 1:
                return label_map[tail]

            prefix = " ".join(tokens[:-1]).strip()
            prefix_label = prefix[:1].upper() + prefix[1:] if prefix else ""
            if prefix_label:
                return f"{prefix_label} - {label_map[tail]}"
            return label_map[tail]

        text = " ".join(tokens).strip()
        return text[:1].upper() + text[1:] if text else self.language_manager.translate("ui.labels.export_item_data", "Data")

    def _canonical_transpose_group_key(self, item_name: str) -> str:
        """Return a canonical checkbox key so equivalent export sources share one option."""
        name = str(item_name or "").strip().lower()
        if not name:
            return "data"

        tokens = [tok for tok in name.split("_") if tok]
        if len(tokens) >= 2:
            tail2 = tokens[-2:]
            if tail2 in (["x", "axis"], ["x", "data"]):
                return "x"
            if tail2 in (["y", "axis"], ["y", "data"]):
                return "y"
            if tail2 in (["z", "axis"], ["z", "data"]):
                return "z"
            if tail2 in (["class", "labels"], ["class", "data"]):
                return "class"
            if tail2 in (["point", "labels"], ["sample", "labels"]):
                return "labels"

        if tokens[-1:] == ["labels"] and len(tokens) >= 2:
            # Merge all non-class label sources (including per-dataset labels)
            # into one shared transpose toggle.
            if len(tokens) >= 2 and tokens[-2] == "class":
                return "class"
            return "labels"
        return name

    def _build_transpose_checkbox_groups(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """Build ordered checkbox groups mapping one checkbox to one-or-many item names."""
        groups: List[Dict[str, Any]] = []
        group_index: Dict[str, int] = {}

        for item_name in item_names:
            canonical = self._canonical_transpose_group_key(item_name)
            if canonical not in group_index:
                group_index[canonical] = len(groups)
                groups.append({
                    "key": canonical,
                    "label": self._humanize_export_item_name(canonical),
                    "items": [item_name],
                })
            else:
                idx = group_index[canonical]
                if item_name not in groups[idx]["items"]:
                    groups[idx]["items"].append(item_name)

        return groups

    def _show_export_scope_and_transpose_dialog(
        self,
        has_slicing: bool,
        all_item_names: List[str],
        visible_item_names: List[str],
    ) -> Optional[Tuple[str, Set[str]]]:
        """Prompt export scope and transpose choices in one centered popup."""
        dialog = tk.Toplevel(self.root)
        _set_window_icon(dialog, "Icon")
        dialog.title(self.language_manager.translate("ui.dialogs.export_data_scope", "Export Data Scope"))
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        result: Dict[str, Any] = {"scope": None, "transpose": set()}

        body = ttk.Frame(dialog, padding=12)
        body.pack(fill=tk.BOTH, expand=True)

        prompt = self.language_manager.translate(
            "ui.messages.export_scope_prompt",
            "Choose what to export and which matrices should be transposed:"
        )
        ttk.Label(body, text=prompt, justify=tk.LEFT, wraplength=600).pack(anchor=tk.W)

        if has_slicing:
            detail = self.language_manager.translate(
                "ui.messages.export_scope_prompt_detail",
                "All exports every matrix used by the graph definition. Visible exports only the matrices used by the graph currently shown in the popup."
            )
            ttk.Label(body, text=detail, justify=tk.LEFT, wraplength=600, foreground="gray").pack(anchor=tk.W, pady=(6, 8))
        else:
            ttk.Label(
                body,
                text=self.language_manager.translate(
                    "ui.messages.export_scope_all_only",
                    "This section has no slicing; export scope is All."
                ),
                justify=tk.LEFT,
                wraplength=600,
                foreground="gray"
            ).pack(anchor=tk.W, pady=(6, 8))

        prompt_transpose = self.language_manager.translate(
            "ui.messages.export_transpose_prompt",
            "Select matrices to transpose before export (default is off):"
        )
        ttk.Label(body, text=prompt_transpose, justify=tk.LEFT, wraplength=600).pack(anchor=tk.W, pady=(0, 6))

        combined_item_names: List[str] = []
        seen_names: Set[str] = set()
        for item_name in list(all_item_names) + list(visible_item_names):
            if item_name in seen_names:
                continue
            seen_names.add(item_name)
            combined_item_names.append(item_name)

        transpose_groups = self._build_transpose_checkbox_groups(combined_item_names)

        transpose_vars: Dict[str, tk.BooleanVar] = {}

        transpose_group = ttk.LabelFrame(
            body,
            text=self.language_manager.translate("ui.labels.transpose", "Transpose"),
            padding=6,
        )
        transpose_group.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        if transpose_groups:
            for group in transpose_groups:
                group_key = group["key"]
                var = tk.BooleanVar(value=False)
                transpose_vars[group_key] = var
                ttk.Checkbutton(
                    transpose_group,
                    text=str(group["label"]),
                    variable=var,
                ).pack(anchor=tk.W, pady=1)
        else:
            ttk.Label(transpose_group, text=self.language_manager.translate("ui.messages.none", "None"), foreground="gray").pack(anchor=tk.W)

        button_row = ttk.Frame(body)
        button_row.pack(fill=tk.X)

        def _close_with(scope: Optional[str]) -> None:
            if scope is None:
                result["scope"] = None
                result["transpose"] = set()
                dialog.destroy()
                return

            result["scope"] = scope
            result["transpose"] = {name for name, var in transpose_vars.items() if bool(var.get())}
            dialog.destroy()

        ttk.Button(
            button_row,
            text=self.language_manager.translate("ui.buttons.all", "All"),
            command=lambda: _close_with("all"),
        ).pack(side=tk.LEFT, padx=(0, 6))

        if has_slicing:
            ttk.Button(
                button_row,
                text=self.language_manager.translate("ui.buttons.visible", "Visible"),
                command=lambda: _close_with("visible"),
            ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            button_row,
            text=self.language_manager.translate("ui.buttons.cancel", "Cancel"),
            command=lambda: _close_with(None),
        ).pack(side=tk.RIGHT)

        dialog.bind("<Escape>", lambda _event: _close_with(None))

        dialog.update_idletasks()
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        win_w = dialog.winfo_width()
        win_h = dialog.winfo_height()
        pos_x = root_x + max(0, (root_w - win_w) // 2)
        pos_y = root_y + max(0, (root_h - win_h) // 2)
        dialog.geometry(f"+{pos_x}+{pos_y}")

        self.root.wait_window(dialog)
        scope_value = result.get("scope")
        if scope_value is None:
            return None
        return scope_value, set(result.get("transpose", set()))

    def _collect_graph_export_items_all(self, outputs: dict, config: dict, metadata: dict) -> List[Tuple[str, Any]]:
        """Collect all raw data sources used by the graph definition."""
        items: List[Tuple[str, Any]] = []

        def _resolve_labels_value(source_name: Any, index_value: Any = None) -> Any:
            if not source_name:
                return None
            labels_value = self._get_data_from_source(outputs, source_name)
            if labels_value is None:
                return None
            if index_value is not None and isinstance(labels_value, list) and labels_value and isinstance(labels_value[0], list):
                try:
                    idx = int(index_value)
                    if idx < 0:
                        idx = 0
                    if idx >= len(labels_value):
                        idx = len(labels_value) - 1
                    labels_value = labels_value[idx]
                except Exception:
                    pass
            return labels_value

        def _append_axis(axis_cfg: Any, name: str) -> None:
            if not isinstance(axis_cfg, dict):
                return
            axis_value = self._extract_axis_data(outputs, axis_cfg, indices=None)
            if axis_value is not None:
                items.append((name, axis_value))

        _append_axis(config.get("x_axis"), "x_axis")
        _append_axis(config.get("y_axis"), "y_axis")
        _append_axis(config.get("z_axis"), "z_axis")

        class_source = config.get("class_labels")
        if class_source:
            class_value = self._get_data_from_source(outputs, class_source)
            if class_value is not None:
                items.append(("class_labels", class_value))

        point_labels_source = config.get("point_labels_source", config.get("sample_labels_source"))
        point_labels_index = config.get("point_labels_index")
        labels_value = _resolve_labels_value(point_labels_source, point_labels_index)
        if labels_value is not None:
            items.append(("labels", labels_value))

        aux_axis = config.get("aux_axis") if isinstance(config, dict) else None
        if isinstance(aux_axis, dict):
            aux_data_source = aux_axis.get("data_source")
            if aux_data_source:
                aux_data = self._get_data_from_source(outputs, aux_data_source)
                if aux_data is not None:
                    items.append(("aux_axis_data", aux_data))
            labels_cfg = aux_axis.get("labels")
            if isinstance(labels_cfg, str) and labels_cfg:
                labels_data = self._get_data_from_source(outputs, labels_cfg)
                if labels_data is not None:
                    items.append(("aux_axis_labels", labels_data))
            elif labels_cfg is not None:
                items.append(("aux_axis_labels", labels_cfg))

        datasets_cfg = config.get("datasets") if isinstance(config.get("datasets"), list) else []
        for idx, dataset_cfg in enumerate(datasets_cfg):
            if not isinstance(dataset_cfg, dict):
                continue
            ds_label = self._sanitize_export_token(dataset_cfg.get("label", f"dataset_{idx + 1}"), f"dataset_{idx + 1}")
            prefix = f"{ds_label}"
            _append_axis(dataset_cfg.get("x_axis"), f"{prefix}_x_axis")
            _append_axis(dataset_cfg.get("y_axis"), f"{prefix}_y_axis")
            _append_axis(dataset_cfg.get("z_axis"), f"{prefix}_z_axis")

            ds_class_source = dataset_cfg.get("class_labels")
            if ds_class_source:
                ds_class_value = self._get_data_from_source(outputs, ds_class_source)
                if ds_class_value is not None:
                    items.append((f"{prefix}_class_labels", ds_class_value))

            ds_labels_source = dataset_cfg.get("point_labels_source", dataset_cfg.get("sample_labels_source"))
            ds_labels_index = dataset_cfg.get("point_labels_index")
            ds_labels_value = _resolve_labels_value(ds_labels_source, ds_labels_index)
            if ds_labels_value is not None:
                items.append((f"{prefix}_labels", ds_labels_value))

        if not items:
            # Last fallback to runtime metadata when config-level sources are unavailable.
            for key in ("x_data", "y_data", "z_data"):
                value = metadata.get(key)
                if value is not None:
                    items.append((key, value))

        return items

    def _collect_graph_export_items_visible(self, metadata: dict) -> List[Tuple[str, Any]]:
        """Collect only currently visible graph payload data."""
        items: List[Tuple[str, Any]] = []
        outputs = metadata.get("outputs", {}) if isinstance(metadata.get("outputs"), dict) else {}
        config = metadata.get("config", {}) if isinstance(metadata.get("config"), dict) else {}

        def _resolve_labels_value(source_name: Any, index_value: Any = None) -> Any:
            if not source_name:
                return None
            labels_value = self._get_data_from_source(outputs, source_name)
            if labels_value is None:
                return None
            if index_value is not None and isinstance(labels_value, list) and labels_value and isinstance(labels_value[0], list):
                try:
                    idx = int(index_value)
                    if idx < 0:
                        idx = 0
                    if idx >= len(labels_value):
                        idx = len(labels_value) - 1
                    labels_value = labels_value[idx]
                except Exception:
                    pass
            return labels_value

        extracted_datasets = metadata.get("extracted_datasets")
        if isinstance(extracted_datasets, list) and extracted_datasets:
            for idx, dataset in enumerate(extracted_datasets):
                if not isinstance(dataset, dict):
                    continue
                ds_label = self._sanitize_export_token(dataset.get("label", f"dataset_{idx + 1}"), f"dataset_{idx + 1}")
                for key in ("x_data", "y_data", "z_data", "class_data"):
                    value = dataset.get(key)
                    if value is not None:
                        items.append((f"{ds_label}_{key}", value))

                ds_labels_source = dataset.get("point_labels_source")
                ds_labels_index = dataset.get("point_labels_index")
                ds_labels_value = _resolve_labels_value(ds_labels_source, ds_labels_index)
                if ds_labels_value is not None:
                    items.append((f"{ds_label}_labels", ds_labels_value))
            return items

        for key in ("x_data", "y_data", "z_data"):
            value = metadata.get(key)
            if value is not None:
                items.append((key, value))

        point_labels_source = config.get("point_labels_source", config.get("sample_labels_source"))
        point_labels_index = config.get("point_labels_index")
        labels_value = _resolve_labels_value(point_labels_source, point_labels_index)
        if labels_value is not None:
            items.append(("labels", labels_value))

        return items

    def _save_section_graph_as_csv(self, instance_alias: str, section_idx: int, section_id_override: Optional[tuple] = None):
        """Save graph data as matrix CSV files (one file per matrix)."""
        try:
            if instance_alias not in self.analysis_data:
                self._show_fading_error(
                    self.language_manager.translate("ui.messages.analysis_data_not_found", "Analysis data not found")
                )
                return

            if isinstance(section_id_override, tuple) and len(section_id_override) == 2:
                section_id = section_id_override
            else:
                current_page = self.analysis_data[instance_alias].get('current_page', 0)
                section_id = (current_page, section_idx)

            graph_data_metadata = self.analysis_data[instance_alias].get('graph_data_metadata', {})
            metadata = graph_data_metadata.get(section_id)
            if not isinstance(metadata, dict):
                self._show_fading_error(
                    self.language_manager.translate("ui.messages.graph_metadata_not_found", "Graph metadata not found for this section")
                )
                return

            config = metadata.get("config", {}) if isinstance(metadata.get("config"), dict) else {}
            outputs = metadata.get("outputs", {}) if isinstance(metadata.get("outputs"), dict) else {}
            has_slicing = bool(config.get("data_slicing"))

            all_export_items = self._collect_graph_export_items_all(outputs, config, metadata)
            visible_export_items = self._collect_graph_export_items_visible(metadata)

            all_matrix_cache: Dict[str, List[Tuple[str, np.ndarray]]] = {}
            for item_name, item_value in all_export_items:
                item_matrices = self._iter_export_matrices(item_value, item_name)
                if item_matrices:
                    all_matrix_cache[item_name] = item_matrices

            visible_matrix_cache: Dict[str, List[Tuple[str, np.ndarray]]] = {}
            for item_name, item_value in visible_export_items:
                item_matrices = self._iter_export_matrices(item_value, item_name)
                if item_matrices:
                    visible_matrix_cache[item_name] = item_matrices

            dialog_result = self._show_export_scope_and_transpose_dialog(
                has_slicing=has_slicing,
                all_item_names=list(all_matrix_cache.keys()),
                visible_item_names=list(visible_matrix_cache.keys()),
            )
            if dialog_result is None:
                return
            scope, transpose_choices = dialog_result

            dir_path = filedialog.askdirectory(
                title=self.language_manager.translate("ui.dialogs.select_export_directory", "Select directory to save graph data"),
                mustexist=True
            )
            if not dir_path:
                return

            graph_title = metadata.get('graph_title', 'graph_data')
            safe_title = self._sanitize_export_token(graph_title, "graph_data")

            matrix_cache = visible_matrix_cache if scope == "visible" else all_matrix_cache

            files_created: List[str] = []
            used_names: Dict[str, int] = {}

            for item_name, matrix_entries in matrix_cache.items():
                for matrix_name, matrix in matrix_entries:
                    safe_matrix_name = self._sanitize_export_token(matrix_name, "data")
                    filename_base = f"{safe_title}__{safe_matrix_name}"
                    duplicate_count = used_names.get(filename_base, 0)
                    used_names[filename_base] = duplicate_count + 1
                    if duplicate_count > 0:
                        filename = f"{filename_base}_{duplicate_count + 1}.csv"
                    else:
                        filename = f"{filename_base}.csv"

                    transpose_key = self._canonical_transpose_group_key(item_name)
                    matrix_to_write = matrix.T if transpose_key in transpose_choices else matrix
                    file_path = Path(dir_path) / filename
                    self._write_matrix_csv(file_path, matrix_to_write)
                    files_created.append(filename)

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

        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.graph_data_save_failed", "Failed to save graph data:") + f" {str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _create_section_popup_button(self, parent: ttk.Frame, instance_alias: str, section_idx: int, section_data: dict):
        """Create a small floating button in the upper right corner of a section (hovering over content)."""
        try:
            def get_section_help_content(data: dict) -> Tuple[str, str, str]:
                if not isinstance(data, dict):
                    return "", "", ""

                section_config = data.get('config', {}) if isinstance(data.get('config', {}), dict) else {}
                section_title = section_config.get(
                    'title',
                    self.language_manager.translate('ui.labels.section', 'Section')
                )

                short_desc = data.get('short_description', '') or section_config.get('short_description', '')
                long_desc = data.get('long_description', '') or section_config.get('long_description', '')

                # Fallback: if runtime analysis_data is stale, pull descriptions from function gui_config
                if not short_desc and not long_desc:
                    try:
                        if instance_alias in self.methodology_list:
                            selected_idx = self.methodology_list.index(instance_alias)
                            if 0 <= selected_idx < len(self.function_base_aliases):
                                base_alias = self.function_base_aliases[selected_idx]
                                gui_analysis = self.gui_configs.get(base_alias, {}).get('analysis', {})
                                gui_pages = gui_analysis.get('pages', [])
                                current_page_idx = self.analysis_data.get(instance_alias, {}).get('current_page', 0)
                                if 0 <= current_page_idx < len(gui_pages):
                                    gui_sections = gui_pages[current_page_idx].get('sections', [])
                                    if 0 <= section_idx < len(gui_sections):
                                        gui_section = gui_sections[section_idx] if isinstance(gui_sections[section_idx], dict) else {}
                                        gui_config = gui_section.get('config', {}) if isinstance(gui_section.get('config', {}), dict) else {}
                                        short_desc = gui_section.get('short_description', '') or gui_config.get('short_description', '')
                                        long_desc = gui_section.get('long_description', '') or gui_config.get('long_description', '')
                                        if section_title == self.language_manager.translate('ui.labels.section', 'Section'):
                                            section_title = gui_config.get('title', section_title)
                    except Exception:
                        pass

                return section_title, str(short_desc), str(long_desc)

            # Defer button creation to allow content to render first
            def create_button():
                section_title, short_desc, long_desc = get_section_help_content(section_data)

                # Create a small help button using same size/shape as popup button
                if short_desc or long_desc:
                    help_btn = ttk.Button(
                        parent,
                        text="ℹ",
                        width=2,
                        command=lambda: self._show_help_popup(section_title, short_desc, long_desc)
                    )
                    help_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-28, y=2)
                    help_btn.lift()
                    Tooltip(help_btn, short_desc if short_desc else self.language_manager.translate("ui.tooltips.click_for_info", "Click for more information"))

                # Create popup button
                popup_btn = ttk.Button(
                    parent, 
                    text=_ui_symbol("expand"),
                    width=2,
                    command=lambda: self._show_section_popup(instance_alias, section_idx, section_data)
                )
                # Position in upper right corner using place()
                popup_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-2, y=2)
                popup_btn.lift()
            
            # Schedule button creation for after all packing/rendering is done
            parent.after(100, create_button)
        
        except Exception as e:
            print(f"Error creating section popup button: {str(e)}")

    def _normalize_navigation_id(self, nav_item: dict) -> Optional[str]:
        """Return normalized nav_id for a navigation item, or None when not configured."""
        if not isinstance(nav_item, dict):
            return None
        nav_id = nav_item.get('nav_id')
        if nav_id is None:
            return None
        nav_id = str(nav_id).strip()
        return nav_id if nav_id else None

    def _get_graph_navigation_data_array(self, outputs: dict, config: dict) -> Optional[np.ndarray]:
        """Resolve the ndarray used to determine navigation bounds for graph slicing."""
        data_source = None
        nested_key = None
        graph_type = config.get('graph_type', '')

        if 'aux_axis' in config:
            data_source = config.get('z_axis', {}).get('data_source')
            nested_key = config.get('z_axis', {}).get('nested_key')
        elif graph_type in ('heatmap', 'contour', '3d_surf'):
            data_source = config.get('z_axis', {}).get('data_source')
            nested_key = config.get('z_axis', {}).get('nested_key')
        elif 'datasets' in config:
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
            axis_config = config.get('y_axis', {}) or config.get('x_axis', {})
            data_source = axis_config.get('data_source')
            nested_key = axis_config.get('nested_key')

        if not data_source:
            return None

        data = self._get_data_from_source(outputs, data_source, nested_key)
        if isinstance(data, np.ndarray):
            return data

        try:
            return np.array(data)
        except (ValueError, TypeError):
            return None

    def _get_graph_nav_item_index(self, slice_state: dict, config: dict, nav_item: dict, max_index: int) -> int:
        """Get current index for a graph navigation item, clamped to valid bounds."""
        dimension = nav_item.get('dimension', 0)
        target_axis = nav_item.get('axis')

        if target_axis:
            axis_indices_dict = slice_state.get('axis_indices', {}).get(target_axis, {})
            default_col = nav_item.get('default', None)
            if default_col is None:
                default_col = config.get(f'{target_axis}_axis', {}).get(
                    'default_column',
                    0 if target_axis == 'x' else 1 if target_axis == 'y' else 2
                )
            current_index = axis_indices_dict.get(dimension, default_col)
        else:
            indices = slice_state.get('indices', {})
            default_idx = nav_item.get('default', 0)
            current_index = indices.get(dimension, default_idx)

        try:
            current_index = int(current_index)
        except (TypeError, ValueError):
            current_index = 0

        if current_index < 0:
            return 0
        if current_index > max_index:
            return max_index
        return current_index

    def _set_graph_nav_item_index(self, slice_state: dict, nav_item: dict, new_index: int) -> None:
        """Set index for a graph navigation item in the appropriate state bucket."""
        dimension = nav_item.get('dimension', 0)
        target_axis = nav_item.get('axis')

        if target_axis:
            if 'axis_indices' not in slice_state:
                slice_state['axis_indices'] = {}
            if target_axis not in slice_state['axis_indices']:
                slice_state['axis_indices'][target_axis] = {}
            slice_state['axis_indices'][target_axis][dimension] = new_index
        else:
            indices = slice_state.get('indices', {})
            indices[dimension] = new_index
            slice_state['indices'] = indices

    def _update_graph_nav_widgets(self, instance_alias: str, section_id: tuple, nav_item: dict,
                                  new_index: int, max_index: int, outputs: dict) -> None:
        """Refresh navigation index and variable-label widgets for a graph nav item."""
        dimension = nav_item.get('dimension', 0)
        target_axis = nav_item.get('axis')

        if hasattr(self, '_nav_labels'):
            stale_nav_keys = []
            for key, label_pair in list(self._nav_labels.items()):
                if len(key) != 5:
                    continue
                key_instance, key_section, key_dimension, key_axis_name, key_target_axis = key
                if key_instance != instance_alias or key_section != section_id:
                    continue
                if key_dimension != dimension or key_target_axis != target_axis:
                    continue
                try:
                    index_label, full_label = label_pair
                    if not index_label.winfo_exists() or not full_label.winfo_exists():
                        stale_nav_keys.append(key)
                        continue
                    index_label.config(text=str(new_index + 1))
                    full_label.config(text=f"{key_axis_name}: {new_index + 1}/{max_index + 1}")
                except Exception:
                    stale_nav_keys.append(key)

            for stale_key in stale_nav_keys:
                self._nav_labels.pop(stale_key, None)

        if hasattr(self, '_var_labels'):
            stale_var_keys = []
            for key, var_tuple in list(self._var_labels.items()):
                if len(key) != 5:
                    continue
                key_instance, key_section, key_dimension, _axis_name, key_target_axis = key
                if key_instance != instance_alias or key_section != section_id:
                    continue
                if key_dimension != dimension or key_target_axis != target_axis:
                    continue
                try:
                    if not isinstance(var_tuple, tuple) or len(var_tuple) < 3:
                        stale_var_keys.append(key)
                        continue
                    var_label, var_labels_config, dim = var_tuple[0], var_tuple[1], var_tuple[2]
                    index_dims = var_tuple[3] if len(var_tuple) > 3 else None
                    if not var_label.winfo_exists():
                        stale_var_keys.append(key)
                        continue
                    state_indices = {}
                    try:
                        state_indices = (
                            self.analysis_data
                            .get(instance_alias, {})
                            .get('table_slices', {})
                            .get(section_id, {})
                            .get('indices', {})
                        )
                    except Exception:
                        state_indices = {}
                    var_label_text = self._get_variable_label_with_indices(
                        outputs,
                        var_labels_config,
                        dim,
                        new_index,
                        indices=state_indices,
                        index_dimensions=index_dims,
                    )
                    if var_label_text:
                        var_label.config(text=f"[{var_label_text}]")
                    else:
                        var_label.config(text="")
                except Exception:
                    stale_var_keys.append(key)

            for stale_key in stale_var_keys:
                self._var_labels.pop(stale_key, None)

    def _sync_graph_nav_group(self, instance_alias: str, source_section_id: tuple,
                              source_nav_item: dict, new_index: int,
                              expected_max_index: int) -> List[tuple]:
        """Sync nav items with the same nav_id when all members share identical range."""
        nav_id = self._normalize_navigation_id(source_nav_item)
        if not nav_id:
            return []

        analysis_info = self.analysis_data.get(instance_alias, {})
        graph_slices = analysis_info.get('graph_slices', {})
        if not isinstance(graph_slices, dict) or not graph_slices:
            return []

        source_section_data = self._get_section_data_by_id(instance_alias, source_section_id)
        source_render_alias = self._resolve_section_render_instance_alias(
            instance_alias,
            section_data=source_section_data,
            section_id=source_section_id,
        )

        candidates = []
        for section_id, section_state in graph_slices.items():
            if not isinstance(section_state, dict):
                continue

            section_data = self._get_section_data_by_id(instance_alias, section_id)
            candidate_render_alias = self._resolve_section_render_instance_alias(
                instance_alias,
                section_data=section_data,
                section_id=section_id,
            )
            if candidate_render_alias != source_render_alias:
                continue

            _alias, _exec_results, candidate_outputs = self._get_section_render_outputs(
                instance_alias,
                section_id,
                section_data=section_data,
            )

            config = section_state.get('config', {})
            nav_axes = config.get('data_slicing', []) if isinstance(config, dict) else []
            if not isinstance(nav_axes, list):
                continue

            data = self._get_graph_navigation_data_array(candidate_outputs, config)
            if data is None or not isinstance(data, np.ndarray):
                continue

            for nav_item in nav_axes:
                if not isinstance(nav_item, dict):
                    continue
                if self._normalize_navigation_id(nav_item) != nav_id:
                    continue

                dim_value = nav_item.get('dimension')
                if dim_value is None:
                    continue
                try:
                    dimension = int(dim_value)
                except (TypeError, ValueError):
                    continue
                if dimension < 0 or dimension >= len(data.shape):
                    continue

                max_index = data.shape[dimension] - 1
                candidates.append((section_id, section_state, config, nav_item, max_index, candidate_outputs))

        if len(candidates) == 0:
            return []

        ranges = {entry[4] for entry in candidates}
        if len(ranges) != 1 or expected_max_index not in ranges:
            return []

        changed_sections = set()
        for section_id, section_state, config, nav_item, max_index, candidate_outputs in candidates:
            current_index = self._get_graph_nav_item_index(section_state, config, nav_item, max_index)
            if current_index != new_index:
                self._set_graph_nav_item_index(section_state, nav_item, new_index)
                changed_sections.add(section_id)

            self._update_graph_nav_widgets(
                instance_alias,
                section_id,
                nav_item,
                new_index,
                max_index,
                candidate_outputs
            )

        changed_sections.discard(source_section_id)
        return list(changed_sections)
    
    def _create_navigation_controls(self, parent_frame: ttk.Frame, instance_alias: str,
                                   section_id: tuple, outputs: dict, config: dict,
                                   slice_state: dict,
                                   on_change: Optional[Callable[[], None]] = None,
                                   include_widget_refs: bool = True) -> None:
        """Create navigation controls (arrow buttons) for multi-dimensional data slicing and axis selection.
        
        Supports both slicing and axis selection:
        Slicing: "data_slicing": [{"name": "Samples", "dimension": 0}]
        Axis selection: "data_slicing": [{"name": "X-Axis", "dimension": 1, "axis": "x"}, {"name": "Y-Axis", "dimension": 1, "axis": "y"}]
        """
        try:
            nav_axes = config.get('data_slicing', [])
            if not nav_axes:
                return
            
            # Get the data array used to determine navigation bounds.
            data = self._get_graph_navigation_data_array(outputs, config)
            if data is None or not isinstance(data, np.ndarray):
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
                                dimension = nav_item['dimension']
                                # Get default from nav_item or axis config
                                default_val = nav_item.get('default', None)
                                if default_val is None:
                                    default_val = config.get(f'{target_axis}_axis', {}).get('default_column', 0)
                                # Validate default is in bounds
                                max_idx = data.shape[dimension] - 1 if dimension < len(data.shape) else 0
                                if default_val < 0 or default_val > max_idx:
                                    default_val = 0
                                if target_axis not in slice_state['axis_indices']:
                                    slice_state['axis_indices'][target_axis] = {}
                                slice_state['axis_indices'][target_axis][dimension] = default_val
            
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
                        md_frame.pack(anchor='w', padx=4, pady=2)
                        
                        # Combination selector
                        combo_select_frame = ttk.Frame(md_frame)
                        combo_select_frame.pack(anchor='w', padx=5, pady=2)
                        
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
                        
                        host_width = self.analysis_data.get(instance_alias, {}).get('active_nav_host_width', 420)
                        combo_width = max(22, min(42, int((host_width - 140) / 7)))
                        combo_dropdown = ttk.Combobox(combo_select_frame, 
                                                     values=combo_options, state='readonly', width=combo_width)
                        combo_dropdown.current(current_combo_idx if current_combo_idx < len(combo_options) else 0)
                        combo_dropdown.pack(side=tk.LEFT, padx=5)
                        combo_dropdown.bind('<<ComboboxSelected>>', 
                                           lambda e, cb=on_change: self._on_md_combo_changed(instance_alias, section_id,
                                                                               combo_dropdown.current(), md_combinations, cb))
                        
                        # Create slicing controls for dimensions NOT in current combination
                        # These are the dimensions we navigate through (not displayed on x/y axes)
                        if current_combo_idx < len(md_combinations):
                            current_combo = md_combinations[current_combo_idx]
                            # Find all dimensions that are NOT in the combination and NOT specified
                            all_dims = set(range(len(data.shape)))
                            navigable_dims = sorted(all_dims - set(current_combo) - specified_dims)
                            
                            for dim in navigable_dims:
                                dim_frame = ttk.Frame(md_frame)
                                dim_frame.pack(anchor='w', padx=4, pady=1)
                                
                                # Get max index for this dimension
                                max_index = data.shape[dim] - 1 if dim < len(data.shape) else 0
                                
                                # Get current index
                                current_index = slice_state.get('md_slice_indices', {}).get(dim, 0)
                                
                                # Dimension label - use dimension labels if available
                                dim_name = dim_labels.get(dim, f"Dimension {dim}")
                                label_text = f"{dim_name}: {current_index + 1}/{max_index + 1}"
                                label = ttk.Label(dim_frame, text=label_text, width=16)
                                label.pack(side=tk.LEFT, padx=3)
                                
                                # Previous button - capture max_index by value with m=max_index
                                prev_btn = ttk.Button(
                                    dim_frame,
                                    text="<",
                                    width=3,
                                    command=lambda d=dim, m=max_index, cb=on_change: self._on_md_navigate(
                                        instance_alias, section_id, -1, d, m, cb
                                    )
                                )
                                prev_btn.pack(side=tk.LEFT, padx=1)
                                
                                # Index display
                                index_label = ttk.Label(dim_frame, text=str(current_index + 1), width=3)
                                index_label.pack(side=tk.LEFT, padx=1)
                                
                                # Next button - capture max_index by value with m=max_index
                                next_btn = ttk.Button(
                                    dim_frame,
                                    text=">",
                                    width=3,
                                    command=lambda d=dim, m=max_index, cb=on_change: self._on_md_navigate(
                                        instance_alias, section_id, 1, d, m, cb
                                    )
                                )
                                next_btn.pack(side=tk.LEFT, padx=1)
                                
                                # Variable labels - show current value label between buttons if configured
                                var_labels_config = config.get('variable_labels')
                                if var_labels_config:
                                    var_label_text = self._get_variable_label(outputs, var_labels_config, dim, current_index)
                                    if var_label_text:
                                        var_label = ttk.Label(dim_frame, text=f"[{var_label_text}]", foreground="gray")
                                        var_label.pack(side=tk.LEFT, padx=5)
                                        # Store reference for updates
                                        if include_widget_refs:
                                            if not hasattr(self, '_var_labels'):
                                                self._var_labels = {}
                                            var_label_key = (instance_alias, section_id, dim, 'md')
                                            self._var_labels[var_label_key] = (var_label, var_labels_config, dim)
                                
                                # Store reference for updates
                                if include_widget_refs:
                                    if not hasattr(self, '_md_nav_labels'):
                                        self._md_nav_labels = {}
                                    label_key = (instance_alias, section_id, dim)
                                    self._md_nav_labels[label_key] = (index_label, label, dim_labels.get(dim))
            
            # Reserve fixed title width so prev/next buttons stay aligned across items.
            nav_label_width_raw = self.language_manager.translate("ui.layout.navigation_title_width", 18)
            try:
                nav_label_width = int(nav_label_width_raw)
            except (TypeError, ValueError):
                nav_label_width = 18
            nav_label_width = max(10, min(nav_label_width, 40))

            # For each navigable axis, create controls if enabled for that item
            axes_frame = ttk.Frame(parent_frame)
            axes_frame.pack(anchor='w', padx=4, pady=1)

            for nav_item in nav_axes:
                # Parse navigation item - support both old and new formats
                if isinstance(nav_item, dict):
                    # New format: {"name": "Samples", "dimension": 0} or {"name": "X-Axis", "dimension": 1, "axis": "x"}
                    axis_name = nav_item.get('name', 'Axis')
                    dimension = nav_item.get('dimension', 0)
                    target_axis = nav_item.get('axis')  # 'x', 'y', 'z', or None for slicing
                    nav_id = self._normalize_navigation_id(nav_item)
                    nav_labels_source = nav_item.get('labels_source')
                    nav_labels_dimension = nav_item.get('labels_dimension', None)
                    # Check if this item should show navigation (default to False - must be explicitly enabled)
                    show_nav = nav_item.get('show_navigation_menu', False)
                else:
                    # Old format: just a string "Samples"
                    axis_name = nav_item
                    dimension = nav_axes.index(nav_item)  # Position in the list
                    target_axis = None
                    nav_id = None
                    nav_labels_source = None
                    nav_labels_dimension = None
                    show_nav = True  # Show by default for old format
                
                # Skip this item if navigation menu is disabled
                if not show_nav:
                    continue
                
                axis_frame = ttk.Frame(axes_frame)
                axis_frame.pack(anchor='w', pady=1)
                
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
                label = ttk.Label(axis_frame, text=label_text, width=nav_label_width)
                label.pack(side=tk.LEFT, padx=3)
                
                # Previous button - capture max_index by value with m=max_index
                prev_btn = ttk.Button(
                    axis_frame,
                    text="<",
                    width=3,
                    command=lambda an=axis_name, d=dimension, ax=target_axis, m=max_index, nid=nav_id, cb=on_change: self._on_navigate_slice(
                        instance_alias, section_id, -1, d, an, m, ax, nid, cb
                    )
                )
                prev_btn.pack(side=tk.LEFT, padx=1)
                
                # Index display - show 1-based for user
                index_label = ttk.Label(axis_frame, text=str(current_index + 1), width=3)
                index_label.pack(side=tk.LEFT, padx=1)
                
                # Next button - capture max_index by value with m=max_index
                next_btn = ttk.Button(
                    axis_frame,
                    text=">",
                    width=3,
                    command=lambda an=axis_name, d=dimension, ax=target_axis, m=max_index, nid=nav_id, cb=on_change: self._on_navigate_slice(
                        instance_alias, section_id, 1, d, an, m, ax, nid, cb
                    )
                )
                next_btn.pack(side=tk.LEFT, padx=1)
                
                # Variable labels - show current value label after buttons if configured
                var_labels_config = nav_labels_source if nav_labels_source else config.get('variable_labels')
                if var_labels_config:
                    label_dim = int(nav_labels_dimension) if nav_labels_dimension is not None else int(dimension)
                    var_label_text = self._get_variable_label(outputs, var_labels_config, label_dim, current_index)
                    if var_label_text:
                        var_label = ttk.Label(axis_frame, text=f"[{var_label_text}]", foreground="gray")
                        var_label.pack(side=tk.LEFT, padx=5)
                        # Store reference for updates
                        if include_widget_refs:
                            if not hasattr(self, '_var_labels'):
                                self._var_labels = {}
                            var_label_key = (instance_alias, section_id, dimension, axis_name, target_axis)
                            self._var_labels[var_label_key] = (var_label, var_labels_config, label_dim)
                
                # Store reference to index label for updates
                if include_widget_refs:
                    if not hasattr(self, '_nav_labels'):
                        self._nav_labels = {}
                    # Use a tuple that includes axis name and target_axis to ensure uniqueness
                    label_key = (instance_alias, section_id, dimension, axis_name, target_axis)
                    self._nav_labels[label_key] = (index_label, label)
        
        except Exception as e:
            print(f"Error creating navigation controls: {str(e)}")
    
    def _on_navigate_slice(self, instance_alias: str, section_id: tuple, direction: int,
                          dimension: int, axis_name: str, max_index: int,
                          target_axis: str = None, nav_id: Optional[str] = None,
                          on_change: Optional[Callable[[], None]] = None) -> None:
        """Handle navigation button click to change slice index or axis selection.
        
        Args:
            target_axis: 'x', 'y', 'z' for axis selection, or None for dimension slicing
            nav_id: Optional navigation group id for synchronized controls
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
                source_nav_item = {
                    'name': axis_name,
                    'dimension': dimension,
                    'axis': target_axis,
                    'nav_id': nav_id
                }

                self._set_graph_nav_item_index(current_state, source_nav_item, new_index)

                section_data = self._get_section_data_by_id(instance_alias, section_id)
                _render_alias, _execution_results, outputs = self._get_section_render_outputs(
                    instance_alias,
                    section_id,
                    section_data=section_data,
                )

                self._update_graph_nav_widgets(
                    instance_alias,
                    section_id,
                    source_nav_item,
                    new_index,
                    max_index,
                    outputs
                )

                synced_graph_sections = self._sync_graph_nav_group(
                    instance_alias,
                    section_id,
                    source_nav_item,
                    new_index,
                    max_index
                )

                synced_table_sections = self._sync_table_nav_group(
                    instance_alias,
                    section_id,
                    source_nav_item,
                    new_index,
                    max_index
                )

                sections_to_refresh = [section_id] + synced_graph_sections + synced_table_sections
                seen_sections = set()
                for sid in sections_to_refresh:
                    if sid in seen_sections:
                        continue
                    seen_sections.add(sid)

                    if sid in self.analysis_data.get(instance_alias, {}).get('graph_slices', {}):
                        self._update_graph_with_slice(instance_alias, sid, dimension)
                    elif sid in self.analysis_data.get(instance_alias, {}).get('table_slices', {}):
                        self._refresh_table(instance_alias, sid)

                # Rebuild active navigation panel to reflect synchronized indices everywhere.
                self._render_active_section_navigation(instance_alias)

                if callable(on_change):
                    on_change()
        
        except Exception as e:
            print(f"Error navigating slice: {str(e)}")
    
    def _on_md_combo_changed(self, instance_alias: str, section_id: tuple,
                            combo_index: int, combinations: List[Tuple[int, ...]],
                            on_change: Optional[Callable[[], None]] = None) -> None:
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
            
            # Rebuild shared active-section navigation controls
            self._render_active_section_navigation(instance_alias)
            
            # Update the graph with the new slice
            self._update_graph_with_slice(instance_alias, section_id, 0)
            self._update_graph_with_slice(instance_alias, section_id, 0)

            if callable(on_change):
                on_change()
        
        except Exception as e:
            print(f"Error changing MD combination: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _on_md_navigate(self, instance_alias: str, section_id: tuple, direction: int,
                       dimension: int, max_index: int,
                       on_change: Optional[Callable[[], None]] = None) -> None:
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
                        var_tuple = self._var_labels[var_key]
                        if not isinstance(var_tuple, tuple) or len(var_tuple) < 3:
                            self._var_labels.pop(var_key, None)
                        else:
                            var_label, var_labels_config, dim = var_tuple[0], var_tuple[1], var_tuple[2]
                            # Get section-context outputs to resolve variable label.
                            section_data = self._get_section_data_by_id(instance_alias, section_id)
                            _render_alias, _execution_results, outputs = self._get_section_render_outputs(
                                instance_alias,
                                section_id,
                                section_data=section_data,
                            )
                            var_label_text = self._get_variable_label(outputs, var_labels_config, dim, new_index)
                            if var_label_text:
                                var_label.config(text=f"[{var_label_text}]")
                            else:
                                var_label.config(text="")
                
                # Refresh the graph with new multi-dimensional slice
                self._update_graph_with_slice(instance_alias, section_id, dimension)

                if callable(on_change):
                    on_change()
        
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
            # Resolve fresh outputs from this section's source context.
            section_data = self._get_section_data_by_id(instance_alias, section_id)
            _render_alias, execution_results, outputs = self._get_section_render_outputs(
                instance_alias,
                section_id,
                section_data=section_data,
            )
            execution_inputs = execution_results.get('inputs', {}) if isinstance(execution_results, dict) else {}
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

                existing_x_axis = config.get('x_axis', {}) if isinstance(config.get('x_axis'), dict) else {}
                existing_y_axis = config.get('y_axis', {}) if isinstance(config.get('y_axis'), dict) else {}
                
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
                
                # Heatmap matrices are row-major: first active dim -> rows (y),
                # second active dim -> columns (x).
                x_axis_config = None
                if len(md_active_dims) > 1:
                    dim_idx = md_active_dims[1]
                    x_label = labels[dim_idx] if dim_idx < len(labels) else f"Axis {dim_idx}"
                    x_axis_config = existing_x_axis.copy()
                    x_axis_config.update({'data_source': data_source, 'index': dim_idx, 'label': x_label})
                
                # First active dim maps to heatmap row axis (y).
                y_axis_config = None
                if len(md_active_dims) > 0:
                    dim_idx = md_active_dims[0]
                    y_label = labels[dim_idx] if dim_idx < len(labels) else f"Axis {dim_idx}"
                    y_axis_config = existing_y_axis.copy()
                    y_axis_config.update({'data_source': data_source, 'index': dim_idx, 'label': y_label})
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

            # Only apply axis-specific indices that are currently declared in data_slicing.
            active_axis_dims = {'x': set(), 'y': set(), 'z': set()}
            for nav_item in nav_axes:
                if not isinstance(nav_item, dict):
                    continue
                target_axis = nav_item.get('axis')
                if target_axis not in active_axis_dims:
                    continue
                dim_val = nav_item.get('dimension')
                try:
                    dim_idx = int(dim_val)
                except (TypeError, ValueError):
                    continue
                active_axis_dims[target_axis].add(dim_idx)
            
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
            if 'y' in axis_indices and isinstance(axis_indices.get('y'), dict):
                for dim_key, dim_index in axis_indices['y'].items():
                    try:
                        dim_int = int(dim_key)
                    except (TypeError, ValueError):
                        continue
                    if dim_int in active_axis_dims['y']:
                        y_indices[dim_int] = dim_index
            y_data = self._extract_axis_data(outputs, y_axis_config, y_indices)
            
            # Merge axis indices for x (base + md + axis-specific)
            x_indices = base_indices.copy()
            x_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'x' in axis_indices and isinstance(axis_indices.get('x'), dict):
                for dim_key, dim_index in axis_indices['x'].items():
                    try:
                        dim_int = int(dim_key)
                    except (TypeError, ValueError):
                        continue
                    if dim_int in active_axis_dims['x']:
                        x_indices[dim_int] = dim_index
            # Pass y_data as reference for row index generation if needed
            x_data = self._extract_axis_data(outputs, x_axis_config, x_indices, ref_data=y_data)
            
            # Merge axis indices for z (base + md + axis-specific)
            z_indices = base_indices.copy()
            z_indices.update(md_slice_indices)  # Add multi-dimensional slices
            if 'z' in axis_indices and isinstance(axis_indices.get('z'), dict):
                for dim_key, dim_index in axis_indices['z'].items():
                    try:
                        dim_int = int(dim_key)
                    except (TypeError, ValueError):
                        continue
                    if dim_int in active_axis_dims['z']:
                        z_indices[dim_int] = dim_index
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
                            resolved_x_label = self._resolve_axis_label(temp_x_axis, outputs, axis_index=x_axis_idx, slice_indices=base_indices)
                            if resolved_x_label:
                                temp_x_axis['label'] = resolved_x_label
                            render_config['x_axis'] = temp_x_axis
                            break
                    
                    for dataset_cfg in datasets_config:
                        if not config.get('y_axis') and 'y_axis' in dataset_cfg:
                            temp_y_axis = dataset_cfg['y_axis'].copy()
                            resolved_y_label = self._resolve_axis_label(temp_y_axis, outputs, axis_index=y_axis_idx, slice_indices=base_indices)
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
                        exec_inputs = execution_inputs
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

                    # Resolve dataset axis labels from variables (including nested lookups).
                    ds_x_axis = ds_x_axis.copy() if isinstance(ds_x_axis, dict) else {}
                    ds_y_axis = ds_y_axis.copy() if isinstance(ds_y_axis, dict) else {}
                    ds_z_axis = ds_z_axis.copy() if isinstance(ds_z_axis, dict) else {}
                    resolved_ds_x_label = self._resolve_axis_label(ds_x_axis, outputs, axis_index=x_axis_idx, slice_indices=base_indices)
                    if resolved_ds_x_label:
                        ds_x_axis['label'] = resolved_ds_x_label
                    resolved_ds_y_label = self._resolve_axis_label(ds_y_axis, outputs, axis_index=y_axis_idx, slice_indices=base_indices)
                    if resolved_ds_y_label:
                        ds_y_axis['label'] = resolved_ds_y_label
                    if ds_z_axis:
                        resolved_ds_z_label = self._resolve_axis_label(ds_z_axis, outputs, axis_index=z_axis_idx, slice_indices=base_indices)
                        if resolved_ds_z_label:
                            ds_z_axis['label'] = resolved_ds_z_label
                    
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
                    ds_class_layers = None
                    if 'class_labels' in dataset_cfg:
                        class_source = dataset_cfg['class_labels']
                        class_val = self._get_data_from_source(outputs, class_source)
                        if class_val is not None:
                            if isinstance(class_val, (list, np.ndarray)):
                                ds_class_layers = self._normalize_class_data_matrix(class_val)
                                ds_class_data = self._normalize_class_labels_for_plot(class_val)
                    
                    # Dataset is valid and will be rendered
                    dataset_entry = {
                        'x_data': ds_x_data,
                        'y_data': ds_y_data,
                        'label': dataset_label,
                        'visibility_key': f'cfg:{dataset_idx}',
                        'x_axis': ds_x_axis,  # Preserve axis config for label extraction
                        'y_axis': ds_y_axis   # Preserve axis config for label extraction
                    }
                    if graph_type == 'scatter':
                        dataset_entry['marker'] = dataset_cfg.get('marker', 'o')
                    elif 'marker' in dataset_cfg:
                        dataset_entry['marker'] = dataset_cfg.get('marker')
                    if ds_z_data is not None:
                        dataset_entry['z_data'] = ds_z_data
                    if ds_class_data is not None:
                        dataset_entry['class_data'] = ds_class_data
                    if ds_class_layers is not None:
                        dataset_entry['class_layers'] = ds_class_layers
                    # Include color if specified (used as fallback when no class_data)
                    if 'color' in dataset_cfg:
                        dataset_entry['color'] = dataset_cfg['color']
                    # Include point_labels_source if specified per-dataset
                    ds_point_labels_source = dataset_cfg.get('point_labels_source', dataset_cfg.get('sample_labels_source'))
                    if ds_point_labels_source:
                        dataset_entry['point_labels_source'] = ds_point_labels_source
                        ds_point_labels_index = dataset_cfg.get('point_labels_index')
                        if ds_point_labels_index is not None:
                            dataset_entry['point_labels_index'] = ds_point_labels_index
                    extracted_datasets.append(dataset_entry)
            
            # If main plot has class_labels config, treat it as a dataset for proper class coloring with qualitative colormap
            if 'class_labels' in config and graph_type in ('scatter', 'line') and x_data is not None and y_data is not None:
                class_source = config['class_labels']
                class_val = self._get_data_from_source(outputs, class_source)
                if class_val is not None:
                    if isinstance(class_val, (list, np.ndarray)):
                        main_class_layers = self._normalize_class_data_matrix(class_val)
                        main_class_data = self._normalize_class_labels_for_plot(class_val)
                        
                        # Create extracted_datasets if it doesn't exist
                        if extracted_datasets is None:
                            extracted_datasets = []
                        
                        # Build main dataset with class data
                        main_dataset = {
                            'x_data': x_data,
                            'y_data': y_data,
                            'label': 'Main Dataset',
                            'visibility_key': 'main_class_dataset',
                            'class_data': main_class_data
                        }
                        # Scatter uses marker for dataset identity; line uses linestyle
                        if graph_type == 'scatter':
                            main_dataset['marker'] = 'o'
                        elif graph_type == 'line' and 'marker' in config:
                            main_dataset['marker'] = config.get('marker')
                        if main_class_layers is not None:
                            main_dataset['class_layers'] = main_class_layers
                        if z_data is not None:
                            main_dataset['z_data'] = z_data
                        # Include point_labels_source if specified at config level
                        cfg_point_labels_source = config.get('point_labels_source', config.get('sample_labels_source'))
                        if cfg_point_labels_source:
                            main_dataset['point_labels_source'] = cfg_point_labels_source
                        
                        # Add main dataset at the beginning of the list
                        extracted_datasets.insert(0, main_dataset)
                        
                        # Clear x/y/z data so they won't conflict with multi-dataset rendering
                        x_data = None
                        y_data = None
                        z_data = None

            extracted_datasets = self._assign_dataset_style_slots(extracted_datasets)
            extracted_datasets = self._apply_dataset_visibility_filter(config, extracted_datasets)
            
            # Extract sample labels for tooltip display from individual datasets
            sample_labels = None
            sample_labels_by_dataset = None
            
            # If we have extracted datasets, collect their point_labels_source
            if extracted_datasets and len(extracted_datasets) > 0:
                sample_labels_by_dataset = {}
                for dataset_entry in extracted_datasets:
                    ds_label = dataset_entry.get('label')
                    ds_source = dataset_entry.get('point_labels_source')
                    if ds_label and ds_source and ds_source in outputs:
                        labels_data = outputs[ds_source]
                        ds_labels_index = dataset_entry.get('point_labels_index')
                        if ds_labels_index is not None and isinstance(labels_data, list) and labels_data and isinstance(labels_data[0], list):
                            labels_data = labels_data[ds_labels_index] if ds_labels_index < len(labels_data) else labels_data[0]
                        if isinstance(labels_data, (list, np.ndarray)):
                            sample_labels_by_dataset[ds_label] = [str(lbl) for lbl in labels_data]
            else:
                # For single dataset, check for point_labels_source in config
                point_labels_source = config.get('point_labels_source', config.get('sample_labels_source'))
                if isinstance(point_labels_source, str):
                    # Single sample labels source
                    if point_labels_source in outputs:
                        labels_data = outputs[point_labels_source]
                        point_labels_index = config.get('point_labels_index')
                        if point_labels_index is not None and isinstance(labels_data, list) and labels_data and isinstance(labels_data[0], list):
                            labels_data = labels_data[point_labels_index] if point_labels_index < len(labels_data) else labels_data[0]
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

            # Resolve reference-line slicing with the same effective indices used for axis extraction,
            # including axis-specific navigation dimensions (e.g., SBS pairing on x/y axis selectors).
            line_slice_indices = base_indices.copy()
            line_slice_indices.update(md_slice_indices)
            for axis_name in ('x', 'y', 'z'):
                axis_dim_set = active_axis_dims.get(axis_name, set())
                axis_index_map = axis_indices.get(axis_name, {}) if isinstance(axis_indices.get(axis_name), dict) else {}
                for dim_key, dim_index in axis_index_map.items():
                    try:
                        dim_int = int(dim_key)
                    except (TypeError, ValueError):
                        continue
                    if dim_int in axis_dim_set:
                        line_slice_indices[dim_int] = dim_index

            normalized_graph_type = str(graph_type).strip().lower()

            if normalized_graph_type in {'scatter', 'line'}:
                render_config['scatter_reference_lines'] = self._resolve_scatter_reference_lines(config, outputs, slice_indices=line_slice_indices)

            if normalized_graph_type in {'scatter', 'line'}:
                if normalized_graph_type == 'scatter':
                    class_state = self._compute_scatter_class_layer_state(
                        config,
                        datasets_config,
                        outputs,
                        execution_inputs,
                    )
                    configured_order_raw = config.get('class_layer_order', [])
                    if class_state.get('is_multi_dataset') and isinstance(configured_order_raw, list) and 'marker' in [str(v).strip().lower() for v in configured_order_raw]:
                        self._show_graph_warning_once(
                            instance_alias,
                            section_id,
                            'scatter_marker_reserved',
                            self.language_manager.translate(
                                'ui.messages.scatter_marker_reserved_multi_dataset',
                                'Scatter class-layer marker mapping is disabled for multiple datasets; marker is reserved for dataset identity.'
                            )
                        )
                else:
                    class_state = self._compute_line_class_layer_state(
                        config,
                        datasets_config,
                        outputs,
                        execution_inputs,
                    )
                    if class_state.get('marker_explicit'):
                        render_config['line_marker_reserved'] = True

                effective_order = list(class_state.get('effective_order', []))
                effective_map = dict(class_state.get('effective_map', {}))
                layer_nature = dict(class_state.get('layer_nature', {}))

                render_config['class_layer_order_effective'] = effective_order
                render_config['class_layer_map_effective'] = effective_map
                render_config['class_layer_nature_effective'] = {str(k): str(v) for k, v in layer_nature.items()}
                render_config['class_layer_count'] = int(class_state.get('layer_count', 0))
                render_config['class_color_palette_mode'] = str(config.get('class_color_palette_mode', 'auto'))
                render_config['class_edge_palette_mode'] = str(config.get('class_edge_palette_mode', 'auto'))
                render_config['class_color_cmap_continuous'] = str(config.get('class_color_cmap_continuous', self.settings_manager.get('colormap', 'viridis')))
                render_config['class_edge_cmap_continuous'] = str(config.get('class_edge_cmap_continuous', self.settings_manager.get('colormap', 'viridis')))
                render_config['class_color_cmap_qualitative'] = str(config.get('class_color_cmap_qualitative', self.settings_manager.get('qualitative_colormap', 'tab10')))
                render_config['class_edge_cmap_qualitative'] = str(config.get('class_edge_cmap_qualitative', self.settings_manager.get('qualitative_colormap', 'tab10')))

                model_payload = outputs.get('model') if isinstance(outputs, dict) else None
                if isinstance(model_payload, dict):
                    ordered_labels = self._build_class_value_order_effective(outputs, model_payload)
                    if ordered_labels:
                        render_config['class_value_order_effective'] = ordered_labels

                if normalized_graph_type == 'line':
                    series_source = config.get('line_series_labels_source')
                    if isinstance(series_source, str) and series_source:
                        series_data = self._get_data_from_source(outputs, series_source)
                        if isinstance(series_data, (list, np.ndarray)):
                            try:
                                labels_arr = np.asarray(series_data, dtype=object).reshape(-1)
                                render_config['line_series_labels'] = [str(item) for item in labels_arr.tolist()]
                            except Exception:
                                pass
            
            # Render graph using graph_renderer module
            graph_renderer = self._get_graph_renderer()
            fig, ax = graph_renderer.render_graph_figure(
                graph_type, render_config, x_data, y_data, z_data, x_axis_config, y_axis_config,
                default_cmap=self.settings_manager.get('colormap', 'viridis'),
                datasets=extracted_datasets,
                qualitative_cmap=self.settings_manager.get('qualitative_colormap', 'tab10'),
                sample_labels=sample_labels,
                sample_labels_by_dataset=sample_labels_by_dataset,
                font_scale=self.graph_font_scale
            )
            
            # Update existing canvas with new figure
            graph_renderer.update_embedded_figure(
                fig,
                instance_alias,
                section_id,
                self.analysis_data,
                None,
                pinned_labels=self._globally_pinned_point_labels,
                on_label_pin_toggled=self._on_global_point_label_pin_toggled,
            )

            updated_canvas_data = self.analysis_data.get(instance_alias, {}).get('graph_canvases', {}).get(section_id)
            if updated_canvas_data:
                updated_canvas, _updated_frame = updated_canvas_data
                try:
                    _tk_widget = updated_canvas.get_tk_widget()
                    if not _tk_widget.winfo_exists():
                        updated_canvas_data = None
                except Exception:
                    updated_canvas_data = None
            if updated_canvas_data:
                updated_canvas, _updated_frame = updated_canvas_data
                self._bind_section_activation(updated_canvas.get_tk_widget(), instance_alias, section_id)
                self._attach_graph_context_menu(
                    updated_canvas,
                    graph_type,
                    config,
                    instance_alias,
                    section_id,
                    popup_refresh_callback=None
                )

            # Keep export payload aligned with the currently visible graph slice.
            if 'graph_data_metadata' not in self.analysis_data[instance_alias]:
                self.analysis_data[instance_alias]['graph_data_metadata'] = {}

            z_data_source = None
            if 'aux_axis' in config:
                z_data_source = config.get('z_axis', {}).get('data_source')
            elif config.get('graph_type') in ('heatmap', '3d_surf', 'contour'):
                z_data_source = config.get('z_axis', {}).get('data_source')

            self.analysis_data[instance_alias]['graph_data_metadata'][section_id] = {
                'x_data': x_data.copy() if isinstance(x_data, np.ndarray) else x_data,
                'y_data': y_data.copy() if isinstance(y_data, np.ndarray) else y_data,
                'z_data': z_data.copy() if isinstance(z_data, np.ndarray) else z_data,
                'z_data_source': z_data_source,
                'x_axis_config': x_axis_config.copy() if x_axis_config else {},
                'y_axis_config': y_axis_config.copy() if y_axis_config else {},
                'z_axis_config': z_axis_config.copy() if isinstance(z_axis_config, dict) else {},
                'extracted_datasets': extracted_datasets,
                'graph_type': graph_type,
                'graph_title': config.get('graph_title', config.get('title', 'Graph')),
                'outputs': outputs,
                'config': config,
            }
            
        except Exception as e:
            print(f"Error updating graph with slice: {str(e)}")
            print(traceback.format_exc())
    
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
        _set_window_icon(dialog, "Icon")
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
                if instance_alias == self.CUSTOM_ANALYSIS_ALIAS:
                    self._show_custom_analysis_tab()
                else:
                    self._show_analysis_tab()
        
        ok_btn = ttk.Button(button_frame, text=self.language_manager.translate("ui.buttons.remove", "Remove"), command=remove_selected)
        ok_btn.pack(padx=5)
        
        cancel_btn = ttk.Button(button_frame, text=self.language_manager.translate("ui.buttons.cancel", "Cancel"), command=dialog.destroy)
        cancel_btn.pack(padx=5)
    
    def _show_add_graph_dialog(self, instance_alias: str):
        """Show the Add Graph dialog."""
        show_add_graph_dialog = self._get_add_graph_dialog_func()
        show_add_graph_dialog(self.root, self, instance_alias)
    
    def _show_add_table_dialog(self, instance_alias: str):
        """Show the Add Table dialog."""
        show_add_table_dialog = self._get_add_table_dialog_func()
        show_add_table_dialog(self.root, self, instance_alias)

    def _show_add_text_dialog(self, instance_alias: str):
        """Show the Add Text dialog."""
        show_add_text_dialog = self._get_add_text_dialog_func()
        show_add_text_dialog(self.root, self, instance_alias)

    def _show_simple_add_text_dialog(
        self,
        instance_alias: str,
        refresh_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Show a notepad-style popup to add plain formatted text to a page section."""
        if instance_alias == self.CUSTOM_ANALYSIS_ALIAS:
            analysis_info = self._ensure_custom_analysis_state()
        else:
            if instance_alias not in self.analysis_data:
                self._show_fading_warning(
                    self.language_manager.translate(
                        "ui.messages.no_data_run_first",
                        "No data available. Please run 'Run Model' or 'Run to here' first."
                    )
                )
                return
            analysis_info = self.analysis_data.get(instance_alias, {}) if isinstance(self.analysis_data, dict) else {}

        pages = analysis_info.get('pages', []) if isinstance(analysis_info, dict) else []
        current_page_idx = analysis_info.get('current_page', 0)

        if current_page_idx < 0 or current_page_idx >= len(pages):
            self._show_fading_warning(
                self.language_manager.translate("ui.messages.no_pages_available", "No pages available")
            )
            return

        current_page = pages[current_page_idx] if isinstance(pages[current_page_idx], dict) else {}
        sections = current_page.get('sections', []) if isinstance(current_page, dict) else []

        empty_sections: List[Tuple[int, int, str]] = []
        for section_idx, section in enumerate(sections):
            if isinstance(section, dict) and section.get('type') is None:
                empty_sections.append((current_page_idx, section_idx, f"Section {section_idx + 1}"))

        if not empty_sections:
            self._show_fading_warning(
                self.language_manager.translate(
                    "ui.messages.no_empty_sections",
                    "No empty sections available. Add a new page or remove existing sections first."
                )
            )
            return

        dialog = tk.Toplevel(self.root)
        _set_window_icon(dialog, "Icon")
        dialog.title(self.language_manager.translate("ui.dialogs.add_text", "Add Text"))
        dialog.geometry("860x640")
        dialog.transient(self.root)
        dialog.grab_set()

        container = ttk.Frame(dialog, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        target_frame = ttk.LabelFrame(
            container,
            text=self.language_manager.translate("ui.labels.target_section", "Target Section"),
            padding=8,
        )
        target_frame.grid(row=0, column=0, sticky='ew', pady=(0, 8))

        section_var = tk.StringVar(value=empty_sections[0][2])
        section_combo = ttk.Combobox(
            target_frame,
            textvariable=section_var,
            values=[desc for _, _, desc in empty_sections],
            state='readonly',
            width=52,
        )
        section_combo.pack(fill=tk.X, expand=True)

        title_frame = ttk.LabelFrame(
            container,
            text=self.language_manager.translate("ui.labels.section_title", "Section Title"),
            padding=8,
        )
        title_frame.grid(row=1, column=0, sticky='ew', pady=(0, 8))

        title_var = tk.StringVar(value=self.language_manager.translate("ui.labels.text", "Text"))
        ttk.Entry(title_frame, textvariable=title_var).pack(fill=tk.X)

        text_frame = ttk.LabelFrame(
            container,
            text=self.language_manager.translate("ui.labels.text_content", "Text Content"),
            padding=8,
        )
        text_frame.grid(row=2, column=0, sticky='nsew')
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        text_widget = tk.Text(text_frame, wrap=tk.WORD, undo=True)
        text_widget.grid(row=0, column=0, sticky='nsew')
        text_widget.configure(font='TkFixedFont')

        text_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_scrollbar.grid(row=0, column=1, sticky='ns')
        text_widget.configure(yscrollcommand=text_scrollbar.set)

        button_row = ttk.Frame(container)
        button_row.grid(row=3, column=0, sticky='e', pady=(10, 0))

        def _add_text_section() -> None:
            selected_desc = section_var.get().strip()
            if not selected_desc:
                self._show_fading_warning(
                    self.language_manager.translate("ui.messages.select_target_section", "Please select a target section")
                )
                return

            target_lookup = {desc: (page_idx, sec_idx) for page_idx, sec_idx, desc in empty_sections}
            target = target_lookup.get(selected_desc)
            if target is None:
                self._show_fading_error(
                    self.language_manager.translate(
                        "ui.messages.selected_target_section_invalid",
                        "Selected target section is invalid"
                    )
                )
                return

            text_content = text_widget.get('1.0', tk.END).rstrip('\n')
            if not text_content.strip():
                self._show_fading_warning("Please enter text content")
                return

            page_idx, section_idx = target
            section_title = title_var.get().strip() or self.language_manager.translate("ui.labels.text", "Text")
            pages[page_idx]['sections'][section_idx] = {
                'type': 'text',
                'config': {
                    'title': section_title,
                    'text_template': text_content,
                    'bindings': [],
                    'wrap': 'word',
                },
            }

            dialog.destroy()
            if callable(refresh_callback):
                refresh_callback()
            elif instance_alias == self.CUSTOM_ANALYSIS_ALIAS:
                self._show_custom_analysis_tab()
            else:
                self._show_analysis_tab()
            self._show_fading_success("Text section added")

        ttk.Button(
            button_row,
            text=self.language_manager.translate("ui.buttons.add_text", "Add Text"),
            command=_add_text_section,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            button_row,
            text=self.language_manager.translate("ui.buttons.cancel", "Cancel"),
            command=dialog.destroy,
        ).pack(side=tk.LEFT)

    def _show_custom_add_text_dialog(self) -> None:
        """Show a notepad-style popup to add plain formatted text in Custom Analysis."""
        self._show_simple_add_text_dialog(
            self.CUSTOM_ANALYSIS_ALIAS,
            refresh_callback=self._show_custom_analysis_tab,
        )

    def _build_custom_section_copy(self, source_alias: str, source_page_idx: int, source_section_idx: int, section_data: dict) -> dict:
        copied = copy.deepcopy(section_data) if isinstance(section_data, dict) else {'type': None}
        copied['_custom_source'] = {
            'instance_alias': source_alias,
            'page_idx': source_page_idx,
            'section_idx': source_section_idx,
        }
        return copied

    def _collect_custom_analysis_candidates(self) -> List[dict]:
        """Collect visible section/page candidates from all methodology analysis tabs."""
        candidates: List[dict] = []

        for idx, source_alias in enumerate(self.methodology_list):
            if source_alias not in self.analysis_data:
                base_alias = self.function_base_aliases[idx] if idx < len(self.function_base_aliases) else source_alias
                analysis_config = self.gui_configs.get(base_alias, {}).get('analysis') if base_alias in self.gui_configs else None
                if analysis_config:
                    self.analysis_data[source_alias] = {
                        'pages': copy.deepcopy(analysis_config.get('pages', [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}])),
                        'current_page': analysis_config.get('current_page', 0),
                        'active_sections': {}
                    }
                else:
                    self.analysis_data[source_alias] = {
                        'pages': [{'title': 'Default', 'layout': 'fp', 'sections': [{'type': None}]}],
                        'current_page': 0,
                        'active_sections': {}
                    }

            source_info = self.analysis_data.get(source_alias, {}) if isinstance(self.analysis_data, dict) else {}
            pages = source_info.get('pages', []) if isinstance(source_info, dict) else []
            exec_results = source_info.get('execution_results', {}) if isinstance(source_info, dict) else {}
            has_execution = isinstance(exec_results, dict) and bool(exec_results)
            base_alias = self.function_base_aliases[idx] if idx < len(self.function_base_aliases) else source_alias
            display_name = self.gui_configs.get(base_alias, {}).get('display_name', base_alias)

            visible_pages = []
            for page_idx, page_data in enumerate(pages):
                if page_data.get('condition') and has_execution:
                    if not self._evaluate_condition(source_alias, page_data.get('condition')):
                        continue
                visible_pages.append((page_idx, page_data))

            total_visible = len(visible_pages)
            for visible_idx, (page_idx, page_data) in enumerate(visible_pages):
                page_title = page_data.get('title', f'Page {page_idx + 1}')
                page_label = f"{visible_idx + 1}/{total_visible}: {page_title}"
                page_sections = page_data.get('sections', []) if isinstance(page_data, dict) else []

                candidates.append({
                    'kind': 'page',
                    'source_alias': source_alias,
                    'source_display_name': display_name,
                    'page_idx': page_idx,
                    'page_label': page_label,
                    'page_data': page_data,
                })

                for section_idx, section_data in enumerate(page_sections):
                    if section_data.get('type') is None:
                        continue
                    if section_data.get('condition') and has_execution:
                        if not self._evaluate_condition(source_alias, section_data.get('condition')):
                            continue
                    section_title = section_data.get('title') or section_data.get('config', {}).get('title') or f"Section {section_idx + 1}"
                    candidates.append({
                        'kind': 'section',
                        'source_alias': source_alias,
                        'source_display_name': display_name,
                        'page_idx': page_idx,
                        'page_label': page_label,
                        'section_idx': section_idx,
                        'section_title': section_title,
                        'section_data': section_data,
                    })

        return candidates

    def _show_custom_add_section_dialog(self, target_section_idx: int):
        """Choose a source section/page from existing analysis tabs for custom analysis."""
        self._ensure_custom_analysis_state()

        if not self._custom_analysis_has_results():
            self._show_fading_warning(
                self.language_manager.translate(
                    "ui.messages.no_data_run_first",
                    "No data available. Please run 'Run Model' or 'Run to here' first."
                )
            )
            return

        custom_info = self.analysis_data.get(self.CUSTOM_ANALYSIS_ALIAS, {})
        pages = custom_info.get('pages', [])
        current_page_idx = custom_info.get('current_page', 0)
        if current_page_idx < 0 or current_page_idx >= len(pages):
            self._show_fading_warning(
                self.language_manager.translate("ui.messages.no_pages_available", "No pages available")
            )
            return

        current_page = pages[current_page_idx]
        current_sections = current_page.get('sections', [])
        if target_section_idx < 0 or target_section_idx >= len(current_sections):
            return

        candidates = self._collect_custom_analysis_candidates()
        section_candidates = [c for c in candidates if c.get('kind') == 'section']
        if not section_candidates:
            self._show_fading_warning(
                self.language_manager.translate("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first.")
            )
            return

        dialog = tk.Toplevel(self.root)
        _set_window_icon(dialog, "Icon")
        dialog.title(self.language_manager.translate("ui.dialogs.add_section", "Add Section"))
        dialog.geometry("1100x680")
        dialog.transient(self.root)
        dialog.grab_set()

        body = ttk.Frame(dialog, padding=10)
        body.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(body)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.LabelFrame(
            body,
            text=self.language_manager.translate("ui.labels.preview", "Preview"),
            padding=8
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        tree = ttk.Treeview(left_frame, show="tree")
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tree_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=tree_scroll.set)

        node_to_candidate: Dict[str, dict] = {}
        grouped: Dict[str, Dict[str, Any]] = {}

        for candidate in candidates:
            source_alias = candidate.get('source_alias', '')
            if source_alias not in grouped:
                grouped[source_alias] = {
                    'display_name': candidate.get('source_display_name', source_alias),
                    'pages': {}
                }
            if candidate.get('kind') == 'page':
                grouped[source_alias]['pages'][candidate['page_idx']] = {
                    'page_candidate': candidate,
                    'sections': []
                }

        for candidate in section_candidates:
            source_alias = candidate.get('source_alias', '')
            page_idx = candidate.get('page_idx')
            grouped.setdefault(source_alias, {'display_name': source_alias, 'pages': {}})
            grouped[source_alias]['pages'].setdefault(page_idx, {'page_candidate': None, 'sections': []})
            grouped[source_alias]['pages'][page_idx]['sections'].append(candidate)

        for source_alias in self.methodology_list:
            if source_alias not in grouped:
                continue
            source_group = grouped[source_alias]
            func_node = tree.insert(
                "",
                tk.END,
                text=f"{source_group.get('display_name', source_alias)} ({source_alias})",
                open=False
            )
            for page_idx in sorted(source_group.get('pages', {}).keys()):
                page_bundle = source_group['pages'][page_idx]
                page_candidate = page_bundle.get('page_candidate')
                if page_candidate is None:
                    continue
                page_node = tree.insert(func_node, tk.END, text=page_candidate.get('page_label', f"Page {page_idx + 1}"), open=False)
                node_to_candidate[page_node] = page_candidate
                for section_candidate in page_bundle.get('sections', []):
                    section_text = section_candidate.get('section_title', f"Section {section_candidate.get('section_idx', 0) + 1}")
                    section_node = tree.insert(page_node, tk.END, text=section_text)
                    node_to_candidate[section_node] = section_candidate

        selected_candidate: Dict[str, Any] = {}

        def _render_preview(candidate: dict):
            for child in right_frame.winfo_children():
                child.destroy()

            if not candidate:
                ttk.Label(
                    right_frame,
                    text=self.language_manager.translate(
                        "ui.messages.select_section_or_page_preview",
                        "Select a section or page to preview."
                    ),
                    foreground="gray"
                ).pack(anchor=tk.W)
                return

            kind = candidate.get('kind')
            if kind == 'section':
                section_data = copy.deepcopy(candidate.get('section_data', {}))
                source_alias = candidate.get('source_alias')
                source_page_idx = candidate.get('page_idx', 0)
                source_section_idx = candidate.get('section_idx', 0)
                section_copy = self._build_custom_section_copy(source_alias, source_page_idx, source_section_idx, section_data)
                self._set_custom_execution_results_from_section(section_copy)
                preview_container = ttk.LabelFrame(
                    right_frame,
                    text=section_copy.get('title') or section_copy.get('config', {}).get('title') or self.language_manager.translate("ui.labels.section", "Section"),
                    padding=5
                )
                preview_container.pack(fill=tk.BOTH, expand=True)
                self._render_section(preview_container, self.CUSTOM_ANALYSIS_ALIAS, section_copy, source_section_idx)
            else:
                page_data = candidate.get('page_data', {})
                ttk.Label(
                    right_frame,
                    text=f"{candidate.get('source_alias', '')} • {candidate.get('page_label', '')}",
                    font=("Arial", 10, "bold")
                ).pack(anchor=tk.W, pady=(0, 8))
                ttk.Label(
                    right_frame,
                    text=self.language_manager.translate("ui.messages.layout_value", "Layout: {layout}").format(layout=page_data.get('layout', 'fp'))
                ).pack(anchor=tk.W)
                sections = page_data.get('sections', [])
                visible_names = []
                for sec_idx, sec in enumerate(sections):
                    if sec.get('type') is None:
                        continue
                    name = sec.get('title') or sec.get('config', {}).get('title') or f"Section {sec_idx + 1}"
                    visible_names.append(name)
                summary = "\n".join(visible_names) if visible_names else self.language_manager.translate(
                    "ui.messages.no_visible_sections",
                    "No visible sections"
                )
                text = tk.Text(right_frame, height=18, wrap=tk.WORD)
                text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
                text.insert('1.0', summary)
                text.configure(state=tk.DISABLED)

        def _on_tree_select(_event=None):
            selected_id = tree.focus()
            candidate = node_to_candidate.get(selected_id)
            selected_candidate.clear()
            if candidate:
                selected_candidate.update(candidate)
            _render_preview(candidate)

        tree.bind("<<TreeviewSelect>>", _on_tree_select)
        _render_preview({})

        button_row = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        button_row.pack(fill=tk.X)

        def _apply_selection():
            candidate = dict(selected_candidate)
            if not candidate:
                self._show_fading_warning(
                    self.language_manager.translate(
                        "ui.messages.select_section_or_page_first",
                        "Select a section or page first."
                    )
                )
                return

            if candidate.get('kind') == 'page':
                replace_page = messagebox.askyesno(
                    self.language_manager.translate("ui.dialogs.confirm", "Confirm"),
                    self.language_manager.translate(
                        "ui.messages.replace_custom_page_confirm",
                        "Replace the current custom page with the selected page?"
                    )
                )
                if not replace_page:
                    return

                source_alias = candidate.get('source_alias')
                source_page_idx = candidate.get('page_idx', 0)
                source_page = copy.deepcopy(candidate.get('page_data', {}))
                sections = source_page.get('sections', []) if isinstance(source_page, dict) else []
                mapped_sections = []
                for sec_idx, section_data in enumerate(sections):
                    if section_data.get('type') is None:
                        mapped_sections.append({'type': None})
                    else:
                        mapped_sections.append(self._build_custom_section_copy(source_alias, source_page_idx, sec_idx, section_data))
                source_page['sections'] = mapped_sections
                pages[current_page_idx] = source_page
            else:
                source_alias = candidate.get('source_alias')
                source_page_idx = candidate.get('page_idx', 0)
                source_section_idx = candidate.get('section_idx', 0)
                section_data = candidate.get('section_data', {})
                current_sections[target_section_idx] = self._build_custom_section_copy(
                    source_alias,
                    source_page_idx,
                    source_section_idx,
                    section_data,
                )

            dialog.destroy()
            self._show_custom_analysis_tab()

        add_btn = ttk.Button(button_row, text=self.language_manager.translate("ui.buttons.add", "Add"), command=_apply_selection)
        add_btn.pack(side=tk.LEFT)

        cancel_btn = ttk.Button(button_row, text=self.language_manager.translate("ui.buttons.cancel", "Cancel"), command=dialog.destroy)
        cancel_btn.pack(side=tk.RIGHT)
    
    def _show_add_page_dialog(self, instance_alias: str):
        """Show dialog to add a new page."""
        if instance_alias not in self.analysis_data:
            return
        
        dialog = tk.Toplevel(self.root)
        _set_window_icon(dialog, "Icon")
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
            ('fp', self.language_manager.translate("ui.labels.layout_fp", "Full\nPage")),
            ('ns', self.language_manager.translate("ui.labels.layout_ns", "North-\nSouth")),
            ('ew', self.language_manager.translate("ui.labels.layout_ew", "East-\nWest")),
            ('fd', self.language_manager.translate("ui.labels.layout_fd", "Four\nSections")),
            ('sd', self.language_manager.translate("ui.labels.layout_sd", "South\nDivision")),
            ('nd', self.language_manager.translate("ui.labels.layout_nd", "North\nDivision")),
            ('ed', self.language_manager.translate("ui.labels.layout_ed", "East\nDivision")),
            ('wd', self.language_manager.translate("ui.labels.layout_wd", "West\nDivision")),
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
            colors = self._get_methodology_theme_colors()
            layout_var.set(layout_code)
            for btn in button_map.values():
                btn.config(relief=tk.RAISED, bg=colors["bg"])
            btn_frame.config(relief=tk.SUNKEN, bg=colors["select_bg"])
        
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
                    btn_frame = tk.Frame(
                        row_frame,
                        relief=tk.RAISED,
                        borderwidth=2,
                        width=95,
                        height=125,
                        bg=self._get_theme_background_color(),
                    )
                    btn_frame.pack_propagate(False)
                    btn_frame.pack(side=tk.LEFT, padx=5, pady=0)
                    
                    # Canvas with layout visualization
                    canvas = tk.Canvas(
                        btn_frame,
                        width=80,
                        height=80,
                        bg=self._get_theme_background_color(),
                        highlightthickness=1,
                        highlightbackground='#999999'
                    )
                    canvas.pack(padx=3, pady=(3, 0))
                    draw_layout_visualization(canvas, layout_code)
                    
                    # Label
                    label = tk.Label(
                        btn_frame,
                        text=layout_desc,
                        font=("Arial", 7),
                        bg=self._get_theme_background_color(),
                        fg=self._get_theme_foreground_color(),
                        justify=tk.CENTER,
                    )
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
            if instance_alias == self.CUSTOM_ANALYSIS_ALIAS:
                self._show_custom_analysis_tab()
            else:
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

        if not pages:
            self._show_fading_warning(
                self.language_manager.translate("ui.messages.no_pages_available", "No pages available")
            )
            return
        
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
            if instance_alias == self.CUSTOM_ANALYSIS_ALIAS:
                self._show_custom_analysis_tab()
            else:
                self._show_analysis_tab()

    def _build_progress_callback(self):
        """Create execution progress callback used by analyst runs."""
        return self._get_run_controller().build_progress_callback(self._update_execution_progress)

    def _get_run_controller(self) -> RunController:
        """Return run controller, creating lazily for __new__-constructed tests."""
        controller = getattr(self, 'run_controller', None)
        if controller is None:
            controller = RunController()
            self.run_controller = controller
        return controller

    def _show_latest_execution_report_popup(self, run_type_label: str) -> None:
        """Show popup for the latest stored execution report."""
        self._get_run_controller().show_latest_execution_report_popup(
            show_popup_fn=lambda report, run_label: self._show_execution_report_popup(report, run_type_label=run_label),
            latest_execution_report=self.latest_execution_report,
            run_type_label=run_type_label,
        )

    def _restore_execution_report_popup(self, run_type_label: str) -> None:
        """Best-effort restore and display of the most recent execution report."""
        self._get_run_controller().restore_execution_report_popup(
            consume_report_fn=consume_last_execution_report_safe,
            store_report_fn=self._store_execution_report,
            show_popup_for_label_fn=self._show_latest_execution_report_popup,
            run_type_label=run_type_label,
        )

    def _store_timing_report_for_run(self, run_type_label: str, timing_report: Any, stop_at_function_alias: Optional[str]) -> None:
        """Store timing report through shared coordinator payload shaping."""
        self._get_run_controller().store_timing_report_for_run(
            store_timing_report_fn=self._store_timing_report,
            run_type_label=run_type_label,
            timing_report=timing_report,
            stop_at_function_alias=stop_at_function_alias,
        )

    def _apply_run_feedback(self, run_feedback: Dict[str, Any]) -> None:
        """Apply feedback payload by showing message and finishing progress."""
        self._get_run_controller().apply_run_feedback(
            run_feedback=run_feedback,
            show_success_fn=self._show_fading_success,
            show_error_fn=self._show_fading_error,
            finish_progress_fn=lambda success: self._finish_execution_progress(success=success),
        )
    
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

            # Reset stale execution-report highlights only for the run scope.
            self._clear_methodology_highlight_source_until("execution_report", stop_idx=stop_at_idx)

            self._begin_execution_progress(
                total_steps=stop_at_idx + 2,
                mode_label=self.language_manager.translate("ui.buttons.run_to_here", "Run to here")
            )
            
            # Generate model.json first
            if not self._generate_model_json():
                self._show_fading_error(
                    self.language_manager.translate("ui.messages.generate_model_json_failed", "Failed to generate model.json")
                )
                return

            if not hasattr(self, 'analysis_data'):
                self.analysis_data = {}

            run_type_label = self.language_manager.translate("ui.buttons.run_to_here", "Run to here")
            runtime_log_path = _get_runtime_model_log_path()
            messages = {
                "execution_failed_prefix": self.language_manager.translate("ui.messages.execution_failed", "Model execution failed:"),
                "partial_results_text": self.language_manager.translate("ui.messages.partial_results_loaded", "Partial results were loaded for completed functions."),
                "executed_up_to_prefix": self.language_manager.translate("ui.messages.model_executed_up_to", "Model executed up to"),
                "results_loaded_text": self.language_manager.translate("ui.messages.results_loaded_analysis", "Results loaded for analysis."),
            }

            orchestrated = orchestrate_run_execution(
                run_mode="partial",
                run_type_label=run_type_label,
                methodology_list=self.methodology_list,
                function_base_aliases=self.function_base_aliases,
                function_configs=self.function_configs,
                gui_configs=self.gui_configs,
                analysis_data=self.analysis_data,
                stop_at_function_idx=stop_at_idx,
                stop_at_function_alias=instance_alias,
                progress_callback=self._build_progress_callback(),
                messages=messages,
            )

            log_text = str(orchestrated.get('log_text', ''))
            runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(runtime_log_path, "w", encoding='utf-8') as f:
                f.write(log_text)

            if not orchestrated.get('ok'):
                self._restore_execution_report_popup(run_type_label)
                self._finish_execution_progress(success=False)
                error_msg = build_run_to_here_exception_message(
                    self.language_manager.translate("ui.messages.run_model_failed", "Failed to run model:"),
                    orchestrated.get('exception', ''),
                )
                self._show_fading_error(error_msg)
                print(f"ERROR: {error_msg}")
                return
            
            self._store_execution_report(orchestrated.get('execution_report'))
            timing_store_args = orchestrated.get('timing_store_args', {})
            if isinstance(timing_store_args, dict):
                self._store_timing_report(**timing_store_args)
            
            # Note: graph_canvases, graph_slices, and table_slices are already cleared
            # by _clear_execution_cache() at the start of this method
            
            self._show_latest_execution_report_popup(run_type_label)
            self._apply_run_feedback(orchestrated.get('run_feedback', {}))
            
            # Refresh the analysis tab to show results
            self._show_analysis_tab()
            
        except Exception as e:
            self._restore_execution_report_popup(self.language_manager.translate("ui.buttons.run_to_here", "Run to here"))
            self._finish_execution_progress(success=False)
            runtime_log_path = _get_runtime_model_log_path()
            runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(runtime_log_path, "w", encoding='utf-8') as f:
                f.write(build_runtime_error_log_contents(str(e), traceback.format_exc()))
            error_msg = build_run_to_here_exception_message(
                self.language_manager.translate("ui.messages.run_model_failed", "Failed to run model:"),
                str(e),
            )
            self._show_fading_error(error_msg)
            print(f"ERROR: {error_msg}")
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

        preview_save_frame = tk.LabelFrame(
            sidebar_frame,
            text=self.language_manager.translate("ui.labels.preview_save", "Preview / Save"),
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
            bd=1,
            relief=tk.GROOVE,
            labelanchor="n",
        )
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

        elements_frame = tk.LabelFrame(
            vertical_split,
            text=self.language_manager.translate("ui.labels.elements", "Elements"),
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
            bd=1,
            relief=tk.GROOVE,
            labelanchor="n",
        )
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
            ("analysis_text", self._report_element_label("analysis_text")),
            ("graph", self._report_element_label("graph")),
            ("table", self._report_element_label("table")),
            ("page_break", self._report_element_label("page_break")),
        ]

        elements_scroll_container = ttk.Frame(elements_frame)
        elements_scroll_container.pack(fill=tk.BOTH, expand=True)

        elements_scrollbar = tk.Scrollbar(elements_scroll_container, orient=tk.VERTICAL)
        elements_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        elements_canvas = tk.Canvas(
            elements_scroll_container,
            highlightthickness=0,
            bg=self._get_theme_background_color(),
            yscrollcommand=elements_scrollbar.set,
        )
        elements_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        elements_scrollbar.configure(command=elements_canvas.yview)

        elements_inner = ttk.Frame(elements_canvas)
        elements_canvas.create_window((0, 0), window=elements_inner, anchor="nw")

        def _update_elements_scrollregion(_event=None):
            bbox = elements_canvas.bbox("all")
            if bbox is None:
                return
            elements_canvas.configure(scrollregion=bbox)

            content_height = bbox[3] - bbox[1]
            viewport_height = max(1, elements_canvas.winfo_height())
            if content_height <= viewport_height:
                elements_canvas.yview_moveto(0.0)

        elements_inner.bind("<Configure>", _update_elements_scrollregion)

        def _elements_mousewheel(event):
            try:
                step = int(-1 * (event.delta / 120))
                if step == 0:
                    return

                first, last = elements_canvas.yview()
                if (step < 0 and first <= 0.0) or (step > 0 and last >= 1.0):
                    return

                elements_canvas.yview_scroll(step, "units")
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

        structure_frame = tk.LabelFrame(
            vertical_split,
            text=self.language_manager.translate("ui.labels.structure", "Structure"),
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
            bd=1,
            relief=tk.GROOVE,
            labelanchor="n",
        )
        vertical_split.add(structure_frame, weight=1)

        structure_list_frame = ttk.Frame(structure_frame)
        structure_list_frame.pack(fill=tk.BOTH, expand=True)

        self.report_structure_listbox = tk.Listbox(
            structure_list_frame,
            height=12,
            selectmode=tk.SINGLE,
            exportselection=False,
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
            selectbackground=self._get_methodology_theme_colors()["select_bg"],
            selectforeground=self._get_methodology_theme_colors()["select_fg"],
            highlightbackground=self._get_theme_background_color(),
            highlightcolor=self._get_theme_background_color(),
        )
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
            text=_ui_symbol("up"),
            width=4,
            command=lambda: self._move_report_element(-1)
        )
        up_btn.pack(side=tk.LEFT, padx=2)

        down_btn = ttk.Button(
            reorder_btn_frame,
            text=_ui_symbol("down"),
            width=4,
            command=lambda: self._move_report_element(1)
        )
        down_btn.pack(side=tk.LEFT, padx=2)

        editor_frame = tk.LabelFrame(
            right_frame,
            text=self.language_manager.translate("ui.labels.element_configuration", "Element Configuration"),
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
            bd=1,
            relief=tk.GROOVE,
            labelanchor="n",
        )
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 4))
        self.report_editor_frame = editor_frame

        preview_frame = tk.LabelFrame(
            right_frame,
            text=self.language_manager.translate("ui.labels.report_preview", "Report Preview (LaTeX)"),
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
            bd=1,
            relief=tk.GROOVE,
            labelanchor="n",
        )
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        preview_text = tk.Text(
            preview_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            height=14,
            bg=self._get_theme_background_color(),
            fg=self._get_theme_foreground_color(),
            insertbackground=self._get_theme_foreground_color(),
        )
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
            'analysis_text': {'text_ref': None, 'title': self.language_manager.translate("ui.messages.default_analysis_text", "Analysis Text")},
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
            'analysis_text': 'Analysis Text',
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

    def _normalize_class_labels_for_plot(self, value: Any) -> Optional[np.ndarray]:
        """Normalize class labels to one value per sample (uses first column for multi-layer inputs)."""
        return _svc_normalize_class_labels_for_plot(value)

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

        elif element_type == 'analysis_text':
            text_options = self._collect_report_text_options()
            ttk.Label(self.report_editor_frame, text=self.language_manager.translate("ui.labels.text_source", "Text source")).pack(anchor='w', pady=(0, 2))

            option_labels = [entry['label'] for entry in text_options]
            selected_ref = settings.get('text_ref')
            initial_label = ''
            for entry in text_options:
                if entry['ref'] == selected_ref:
                    initial_label = entry['label']
                    break
            if not initial_label and option_labels:
                initial_label = option_labels[0]
                self._update_report_element_setting(index, 'text_ref', text_options[0]['ref'])

            source_var = tk.StringVar(value=initial_label)
            source_combo = ttk.Combobox(self.report_editor_frame, textvariable=source_var, values=option_labels, state='readonly')
            source_combo.pack(fill=tk.X, pady=(0, 8))
            if not option_labels:
                source_combo.configure(state='disabled')

            def on_text_source_change(event=None, idx=index):
                label = source_var.get()
                for entry in text_options:
                    if entry['label'] == label:
                        self._update_report_element_setting(idx, 'text_ref', entry['ref'])
                        break

            source_combo.bind('<<ComboboxSelected>>', on_text_source_change)

            ttk.Label(self.report_editor_frame, text=self.language_manager.translate("ui.labels.title", "Title")).pack(anchor='w', pady=(0, 2))
            title_var = tk.StringVar(value=settings.get('title', self.language_manager.translate("ui.messages.default_analysis_text", "Analysis Text")))
            title_entry = ttk.Entry(self.report_editor_frame, textvariable=title_var)
            title_entry.pack(fill=tk.X, pady=(0, 8))
            title_entry.bind('<KeyRelease>', lambda e, idx=index: self._update_report_element_setting(idx, 'title', title_var.get()))

            if not option_labels:
                ttk.Label(
                    self.report_editor_frame,
                    text=self.language_manager.translate("ui.messages.no_text_reports_available", "No analysis text reports available. Run analysis and configure text sections first."),
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

    def _collect_report_text_options(self) -> List[Dict[str, Any]]:
        """Collect text sections from all analysis tabs for report selection."""
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
                    if section.get('type') != 'text':
                        continue
                    if not self._is_report_section_visible(instance_alias, section):
                        continue
                    config = section.get('config', {})
                    section_title = config.get('title') or f"{self.language_manager.translate('ui.report_elements.text', 'Text')} {section_idx + 1}"
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

    def _resolve_report_text_data(self, text_ref: Dict[str, Any]) -> Tuple[str, str]:
        """Resolve report text section reference to (title, rendered_text)."""
        instance_alias, section_data, analysis_info = self._get_report_section(text_ref)
        if not instance_alias or not isinstance(section_data, dict) or not isinstance(analysis_info, dict):
            return '', ''

        if section_data.get('type') != 'text':
            return '', ''

        execution_results = analysis_info.get('execution_results', {})
        if execution_results.get('status') != 'success':
            return '', ''

        config = section_data.get('config', {}) if isinstance(section_data.get('config', {}), dict) else {}
        outputs = self._get_execution_data_sources(execution_results, instance_alias)
        rendered_text = self._resolve_text_section_content(outputs, config)
        title = str(config.get('title', '') or '')
        return title, str(rendered_text or '')

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
        version = _get_application_version("0.0")

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

            graph_renderer = self._get_graph_renderer()
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
                qualitative_cmap=self.settings_manager.get('qualitative_colormap', 'tab10'),
                font_scale=self.graph_font_scale
            )
            fig.savefig(str(image_path), dpi=220, bbox_inches='tight')
            plt = self._get_matplotlib_pyplot()
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
        value_format = str(config.get('value_format', '')).strip()
        rows: List[List[str]] = []
        for row in data:
            row_values: List[str] = []
            for value in row:
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                    if value_format:
                        try:
                            row_values.append(format(value, value_format))
                        except Exception:
                            row_values.append(self._format_table_numeric_value(value, decimal_places))
                    else:
                        row_values.append(self._format_table_numeric_value(value, decimal_places))
                else:
                    row_values.append(str(value))
            rows.append(row_values)

        row_labels: List[str] = []
        if isinstance(row_headers_config, str):
            resolved_row_headers = self._get_data_from_source(outputs, row_headers_config)
            if resolved_row_headers is not None:
                row_labels = [str(v) for v in list(np.asarray(resolved_row_headers).flatten())[:len(rows)]]
        elif isinstance(row_headers_config, dict):
            rhs_source = row_headers_config.get('data_source')
            rhs_nested = row_headers_config.get('nested_key')
            if rhs_source:
                resolved_row_headers = self._get_data_from_source(outputs, rhs_source, rhs_nested)
                if resolved_row_headers is not None:
                    row_labels = [str(v) for v in list(np.asarray(resolved_row_headers).flatten())[:len(rows)]]
        elif isinstance(row_headers_config, (list, np.ndarray)):
            row_labels = [str(v) for v in list(row_headers_config)[:len(rows)]]

        if not row_labels:
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
            elif element_type == 'analysis_text':
                text_ref = settings.get('text_ref')
                source_title, resolved_text = self._resolve_report_text_data(text_ref) if text_ref else ('', '')
                display_title = str(settings.get('title', '') or source_title or '').strip()

                text_parts: List[str] = []
                if display_title:
                    text_parts.append(display_title)
                if resolved_text:
                    if text_parts:
                        text_parts.append('')
                    text_parts.append(resolved_text)

                settings['text'] = "\n".join(text_parts).strip()
                settings.setdefault('font_size', 11)
                settings.setdefault('align', 'left')
                settings.setdefault('bold', False)
                settings.setdefault('italic', False)
                settings.setdefault('underline', False)
                resolved.append({'type': 'text', 'settings': settings})
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

        with self._temporary_workspace_dir(prefix='cm_report_preview_') as temp_dir:
            assets_dir = Path(temp_dir) / 'assets'
            elements = self._build_resolved_report_elements(assets_dir)
            build_latex_document, _ = self._get_reporting_functions()
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
        build_latex_document, compile_latex_to_pdf = self._get_reporting_functions()
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
            # Refresh merged specs before persisting model data.
            self._refresh_function_specs()
            specs_data = FUNCTION_SPECS
            parameter_types = specs_data.get("parameter_types", {})
            model_data = build_model_payload(
                methodology_list=self.methodology_list,
                function_base_aliases=self.function_base_aliases,
                function_configs=self.function_configs,
                gui_configs=self.gui_configs,
                routing_lines=self.routing_lines,
                parameter_types=parameter_types,
                addon_registry=self.addon_registry,
                app_version=_get_application_version("1.0"),
                is_passforward_enabled=self._is_passforward_enabled,
                analysis_data=getattr(self, 'analysis_data', None),
                serialize_analysis_data=self._serialize_analysis_data,
                report_data=getattr(self, 'report_data', None),
            )
            model_data["custom_analysis"] = self._serialize_custom_analysis_data()

            write_model_payload(model_data, _get_runtime_model_json_path())
            
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
                if instance_alias == self.CUSTOM_ANALYSIS_ALIAS:
                    continue
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

    def _serialize_custom_analysis_data(self) -> dict:
        """Serialize global custom analysis tab data for model.json custom_analysis."""
        try:
            data = self._ensure_custom_analysis_state()
            return {
                'pages': copy.deepcopy(data.get('pages', [])),
                'current_page': int(data.get('current_page', 0) or 0)
            }
        except Exception as e:
            print(f"Warning: Failed to serialize custom analysis data: {e}")
            return {'pages': [], 'current_page': 0}
    
    def _deserialize_analysis_data(self, analysis_config: dict):
        """Deserialize analysis_data from model.json."""
        try:
            if not analysis_config:
                return
            
            # Initialize analysis_data if needed
            if not hasattr(self, 'analysis_data'):
                self.analysis_data = {}
            
            for instance_alias, config_data in analysis_config.items():
                if instance_alias == self.CUSTOM_ANALYSIS_ALIAS:
                    continue
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

    def _deserialize_custom_analysis_data(self, custom_analysis_config: dict):
        """Deserialize global custom analysis data from model.json custom_analysis."""
        try:
            pages = []
            current_page = 0
            if isinstance(custom_analysis_config, dict):
                pages = copy.deepcopy(custom_analysis_config.get('pages', []))
                current_page = int(custom_analysis_config.get('current_page', 0) or 0)

            self.custom_analysis_data = {
                'pages': pages,
                'current_page': current_page,
                'active_sections': {},
            }

            if not hasattr(self, 'analysis_data') or not isinstance(self.analysis_data, dict):
                self.analysis_data = {}
            self.analysis_data[self.CUSTOM_ANALYSIS_ALIAS] = self.custom_analysis_data
        except Exception as e:
            print(f"Warning: Failed to deserialize custom analysis data: {e}")
            self.custom_analysis_data = {'pages': [], 'current_page': 0, 'active_sections': {}}
    
    def _clean_tempfiles(self):
        """Clean the tempfiles directory."""
        if self.tempfiles_dir.exists():
            try:
                shutil.rmtree(self.tempfiles_dir)
            except Exception as e:
                print(f"Warning: Could not clean tempfiles: {e}")
        self.tempfiles_dir.mkdir(parents=True, exist_ok=True)

    def _temporary_workspace_dir(self, prefix: str = "cm_tmp_"):
        """Create temporary directories inside the app tempfiles root."""
        self.tempfiles_dir.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix=prefix, dir=str(self.tempfiles_dir))
    
    def _on_close(self):
        """Clean up tempfiles and close the application."""
        self._clean_tempfiles()
        self.root.destroy()
    
    def _show_save_model_dialog(self):
        """Show dialog to choose between saving with or without data."""
        dialog = tk.Toplevel(self.root)
        _set_window_icon(dialog, "Icon")
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
                with open(config_path, encoding='utf-8-sig') as f:
                    config = json.load(f)
            except Exception:
                return path_params
        
        # Check setup.layout for ispath parameters
        for param in config.get("setup", {}).get("layout", []):
            if param.get("ispath", False):
                path_params.append(param.get("name"))
        
        return path_params

    def _extract_source_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract portable metadata for a source file."""
        abs_path = file_path.resolve()
        metadata: Dict[str, Any] = {
            "file_path": str(abs_path),
            "file_name": abs_path.name,
            "file_stem": abs_path.stem,
            "file_extension": abs_path.suffix
        }
        try:
            stat_info = abs_path.stat()
            metadata["file_size_bytes"] = stat_info.st_size
            metadata["created_time"] = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            metadata["modified_time"] = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
        except OSError:
            metadata["file_size_bytes"] = None
            metadata["created_time"] = None
            metadata["modified_time"] = None

        return metadata
    
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
            with self._temporary_workspace_dir(prefix='cm_save_with_data_') as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Read and modify model.json
                model_path = _get_runtime_model_json_path()
                with open(model_path, encoding='utf-8-sig') as f:
                    model_data = json.load(f)
                
                # Process model.json: collect all file paths and copy them
                files_to_copy = []
                packaged_source_metadata: Dict[str, Dict[str, Any]] = {}
                
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
                                                    temp_ref = f"tempfiles/{src_file.name}"
                                                    nested_paths.append(temp_ref)
                                                    packaged_source_metadata[temp_ref] = self._extract_source_file_metadata(src_file)
                                                else:
                                                    nested_paths.append(file_path)
                                            else:
                                                nested_paths.append(file_path)
                                        new_paths.append(nested_paths)
                                    elif isinstance(item, str) and item:
                                        src_file = Path(item)
                                        if src_file.exists():
                                            files_to_copy.append(src_file)
                                            temp_ref = f"tempfiles/{src_file.name}"
                                            new_paths.append(temp_ref)
                                            packaged_source_metadata[temp_ref] = self._extract_source_file_metadata(src_file)
                                        else:
                                            new_paths.append(item)
                                    else:
                                        new_paths.append(item)
                                params[param_name] = new_paths
                            elif isinstance(param_value, str) and param_value:
                                src_file = Path(param_value)
                                if src_file.exists():
                                    files_to_copy.append(src_file)
                                    temp_ref = f"tempfiles/{src_file.name}"
                                    params[param_name] = temp_ref
                                    packaged_source_metadata[temp_ref] = self._extract_source_file_metadata(src_file)

                if packaged_source_metadata:
                    model_data["packaged_source_metadata"] = packaged_source_metadata
                
                # Write modified model.json
                model_tmpfile = tmpdir_path / "model.json"
                with open(model_tmpfile, "w", encoding='utf-8') as f:
                    json.dump(model_data, f, indent=2, ensure_ascii=False)
                
                # Create files subdirectory and copy data files
                files_dir = tmpdir_path / "files"
                files_dir.mkdir(exist_ok=True)
                for src_file in files_to_copy:
                    shutil.copy2(src_file, files_dir / src_file.name)
                
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
            with self._temporary_workspace_dir(prefix='cm_save_method_only_') as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Read and modify model.json - remove only file paths
                model_path = _get_runtime_model_json_path()
                with open(model_path, encoding='utf-8-sig') as f:
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
            with self._temporary_workspace_dir(prefix='cm_save_full_') as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Read model.json
                model_path = _get_runtime_model_json_path()
                with open(model_path, encoding='utf-8-sig') as f:
                    model_data = json.load(f)
                
                files_to_copy = []
                packaged_source_metadata: Dict[str, Dict[str, Any]] = {}
                
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
                                                    temp_ref = f"tempfiles/{src_file.name}"
                                                    nested_paths.append(temp_ref)
                                                    packaged_source_metadata[temp_ref] = self._extract_source_file_metadata(src_file)
                                                else:
                                                    nested_paths.append(file_path)
                                            else:
                                                nested_paths.append(file_path)
                                        new_paths.append(nested_paths)
                                    elif isinstance(item, str) and item:
                                        src_file = Path(item)
                                        if src_file.exists():
                                            files_to_copy.append(src_file)
                                            temp_ref = f"tempfiles/{src_file.name}"
                                            new_paths.append(temp_ref)
                                            packaged_source_metadata[temp_ref] = self._extract_source_file_metadata(src_file)
                                        else:
                                            new_paths.append(item)
                                    else:
                                        new_paths.append(item)
                                params[param_name] = new_paths
                            elif isinstance(param_value, str) and param_value:
                                src_file = Path(param_value)
                                if src_file.exists():
                                    files_to_copy.append(src_file)
                                    temp_ref = f"tempfiles/{src_file.name}"
                                    params[param_name] = temp_ref
                                    packaged_source_metadata[temp_ref] = self._extract_source_file_metadata(src_file)

                if packaged_source_metadata:
                    model_data["packaged_source_metadata"] = packaged_source_metadata
                
                # Write modified model.json
                model_tmpfile = tmpdir_path / "model.json"
                with open(model_tmpfile, "w", encoding='utf-8') as f:
                    json.dump(model_data, f, indent=2, ensure_ascii=False)
                
                # Create files subdirectory and copy data files
                files_dir = tmpdir_path / "files"
                files_dir.mkdir(exist_ok=True)
                for src_file in files_to_copy:
                    shutil.copy2(src_file, files_dir / src_file.name)
                
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
        model_suffix = str(file_path.suffix).strip().lower()
        is_with_data = file_path.suffix in (".mdcd", ".mdfd") or str(file_path).endswith((".mdcd", ".mdfd"))
        
        # Clear any cached execution results and graphs to ensure fresh data
        self._clear_execution_cache()
        
        # Reload GUI configs to ensure fresh state (prevents stale modifications)
        self._load_gui_configs()
        
        # Clean tempfiles before loading
        self._clean_tempfiles()
        
        # Create a temporary extraction directory
        with self._temporary_workspace_dir(prefix='cm_load_model_') as tmpdir:
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
                            shutil.copy2(src_file, self.tempfiles_dir / src_file.name)
            
            # Load model.json
            model_file = tmpdir_path / "model.json"
            if model_file.exists():
                model_path = _get_runtime_model_json_path()
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_file, encoding='utf-8-sig') as f:
                    content = f.read()
                with open(model_path, "w", encoding='utf-8') as f:
                    f.write(content)
        
        # Parse and load configuration from model.json
        self._parse_and_load_model_json()
        
        # Refresh GUI
        self._refresh_gui_from_config()

        # Clear previous model-load requirement highlights, then apply extension-specific guidance.
        self._highlight_methodology_functions(
            function_names=[],
            source="model_load_required_data",
            clear_source_first=True,
        )

        if model_suffix == ".mdcd":
            self._show_fading_message(
                self.language_manager.translate(
                    "ui.messages.model_load_requires_validation_data",
                    "This .mdcd model requires validation data. Please provide it in the validation data function(s)."
                )
            )
            self._highlight_methodology_functions(
                function_names=["validation_data_main"],
                level="warning",
                source="model_load_required_data",
                clear_source_first=False,
            )
        elif model_suffix == ".mdon":
            self._show_fading_message(
                self.language_manager.translate(
                    "ui.messages.model_load_requires_calibration_and_validation_data",
                    "This .mdon model requires calibration and validation data. Please provide them in the load data and validation data function(s)."
                )
            )
            self._highlight_methodology_functions(
                function_names=["validation_data_main", "load_data"],
                level="warning",
                source="model_load_required_data",
                clear_source_first=False,
            )
        
        self._show_fading_success(
            self.language_manager.translate("ui.messages.model_loaded", "Model loaded from:") + f"\n{file_path}"
        )
    
    def _parse_and_load_model_json(self):
        """Parse model.json and load configuration."""
        model_path = _get_runtime_model_json_path()
        if not model_path.exists():
            return
        
        self.methodology_list = []
        self.function_base_aliases = []
        self.function_configs = {}
        self.routing_lines = {}
        # Reset analysis_data completely when loading a new model
        # This ensures previous model's analysis changes don't persist
        self.analysis_data = {}
        self.custom_analysis_data = {'pages': [], 'current_page': 0, 'active_sections': {}}
        self.report_data = {'elements': [], 'selected_index': None}
        
        try:
            with open(model_path, encoding='utf-8-sig') as f:
                model_data = json.load(f)
        except Exception as e:
            self._show_fading_error(
                self.language_manager.translate("ui.messages.load_model_json_failed", "Failed to load model.json:") + f" {e}"
            )
            return

        metadata = model_data.get("metadata", {}) if isinstance(model_data, dict) else {}
        model_version = str(metadata.get("version", "")).strip()
        current_version = _get_application_version("0.0")
        if model_version and current_version:
            version_cmp = _compare_versions(model_version, current_version)
            if version_cmp > 0:
                self._show_fading_warning(
                    self.language_manager.translate(
                        "ui.messages.model_version_newer_warning",
                        "This model was created with a newer program version ({model_version}) than your current version ({current_version}). Some features may not load correctly."
                    ).format(model_version=model_version, current_version=current_version)
                )
            elif version_cmp < 0:
                self._show_fading_warning(
                    self.language_manager.translate(
                        "ui.messages.model_version_older_warning",
                        "This model was created with an older program version ({model_version}) than your current version ({current_version}). It will be upgraded when you save it again."
                    ).format(model_version=model_version, current_version=current_version)
                )

        required_addons = normalize_required_addons(metadata.get("required_addons", []))
        available_addons = set(self.addon_registry.get("available_addons", {}).keys())
        missing_addons = [addon_id for addon_id in required_addons if addon_id not in available_addons]
        if missing_addons:
            self._show_fading_error(
                "Cannot load model. Missing required add-ons: " + ", ".join(missing_addons)
            )
            return

        available_functions = set(FUNCTION_SPECS.get("gui_listing", {}).keys())
        model_base_aliases = [
            str(entry.get("base_alias", ""))
            for entry in model_data.get("functions", [])
            if isinstance(entry, dict)
        ]
        missing_functions = sorted({alias for alias in model_base_aliases if alias and alias not in available_functions})
        if missing_functions:
            self._show_fading_error(
                "Cannot load model. Unknown or unavailable functions: " + ", ".join(missing_functions)
            )
            return
        
        # Load functions
        packaged_source_metadata = model_data.get('packaged_source_metadata', {})

        for func_entry in model_data.get('functions', []):
            instance_alias = func_entry.get('instance_alias', '')
            base_alias = func_entry.get('base_alias', '')
            params = func_entry.get('parameters', {}).copy()
            passforward_entry = func_entry.get('passforward', {})
            source_metadata_overrides: Dict[str, Dict[str, Any]] = {}
            
            self.methodology_list.append(instance_alias)
            self.function_base_aliases.append(base_alias)
            
            # Convert tempfiles references to absolute paths and collect source metadata overrides
            def _convert_tempfile_paths(value: Any) -> Any:
                if isinstance(value, str) and value.startswith('tempfiles/'):
                    temp_ref = value.replace('\\', '/')
                    abs_path = str(self.tempfiles_dir / temp_ref.replace('tempfiles/', ''))
                    source_meta = packaged_source_metadata.get(temp_ref)
                    if isinstance(source_meta, dict):
                        source_metadata_overrides[abs_path] = copy.deepcopy(source_meta)
                    return abs_path
                if isinstance(value, list):
                    return [_convert_tempfile_paths(item) for item in value]
                return value

            for key, value in list(params.items()):
                params[key] = _convert_tempfile_paths(value)

            if base_alias in ('load_data', 'validation_data_main') and source_metadata_overrides:
                params['source_metadata_overrides'] = source_metadata_overrides

            legacy_passforward = _normalize_bool_setting(params.pop("__passforward_enabled__", False), default=False)
            loaded_passforward = False
            if isinstance(passforward_entry, dict):
                loaded_passforward = _normalize_bool_setting(passforward_entry.get("enabled", False), default=False)
            passforward_enabled = bool(loaded_passforward or legacy_passforward)
            if self._is_passforward_compatible(base_alias):
                params["__passforward_enabled__"] = passforward_enabled
            
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

        custom_analysis_config = model_data.get('custom_analysis', {})
        self._deserialize_custom_analysis_data(custom_analysis_config)

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

        # Timing/report popups are valid only for the latest successful run
        self.latest_timing_report = None
        self.latest_execution_report = None
        self._set_timing_report_menu_state(False)
        self._set_execution_report_menu_state(False)
        self._set_model_log_menu_state(False)
    
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

        # Reset stale execution-report highlights for the full methodology run.
        self._clear_methodology_highlight_source_until("execution_report")
        
        if not self._generate_model_json():
            return

        self._begin_execution_progress(
            total_steps=len(self.methodology_list) + 1,
            mode_label=self.language_manager.translate("ui.buttons.run_model", "Run Model")
        )

        run_type_label = self.language_manager.translate("ui.buttons.run_model", "Run Model")
        runtime_log_path = _get_runtime_model_log_path()
        messages = {
            "execution_failed_prefix": self.language_manager.translate("ui.messages.execution_failed", "Model execution failed:"),
            "partial_results_text": self.language_manager.translate("ui.messages.partial_results_loaded", "Partial results were loaded for completed functions."),
            "model_executed_template": self.language_manager.translate(
                "ui.messages.model_executed",
                "Model executed successfully. Output saved to {log_path}"
            ),
            "check_log_template": self.language_manager.translate(
                "ui.messages.check_log",
                "Check the log file at {log_path} for details"
            ),
        }
        
        try:
            if not hasattr(self, 'analysis_data'):
                self.analysis_data = {}

            orchestrated = orchestrate_run_execution(
                run_mode="full",
                run_type_label=run_type_label,
                methodology_list=self.methodology_list,
                function_base_aliases=self.function_base_aliases,
                function_configs=self.function_configs,
                gui_configs=self.gui_configs,
                analysis_data=self.analysis_data,
                stop_at_function_idx=None,
                stop_at_function_alias=None,
                progress_callback=self._build_progress_callback(),
                log_path=str(runtime_log_path),
                messages=messages,
            )

            log_text = str(orchestrated.get('log_text', ''))
            runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(runtime_log_path, "w") as f:
                f.write(log_text)

            if not orchestrated.get('ok'):
                self._restore_execution_report_popup(run_type_label)
                self._finish_execution_progress(success=False)
                self._show_fading_error(str(orchestrated.get('exception_feedback', '')))
                return

            self._store_execution_report(orchestrated.get('execution_report'))
            timing_store_args = orchestrated.get('timing_store_args', {})
            if isinstance(timing_store_args, dict):
                self._store_timing_report(**timing_store_args)
            
            self._show_latest_execution_report_popup(run_type_label)
            self._apply_run_feedback(orchestrated.get('run_feedback', {}))
            
            # Switch to analysis tab to show results
            if self.selected_function_idx is not None:
                self._show_analysis_tab()
            
        except Exception as e:
            self._restore_execution_report_popup(run_type_label)
            self._finish_execution_progress(success=False)
            runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(runtime_log_path, "w", encoding='utf-8') as f:
                f.write(build_runtime_error_log_contents(str(e), traceback.format_exc()))
            
            self._show_fading_error(
                build_full_run_exception_feedback(
                    exception_text=str(e),
                    log_path=str(runtime_log_path),
                    execution_failed_prefix=self.language_manager.translate("ui.messages.execution_failed", "Model execution failed:"),
                    check_log_template=self.language_manager.translate(
                        "ui.messages.check_log",
                        "Check the log file at {log_path} for details"
                    ),
                )
            )


def main():
    """Main entry point for the GUI."""
    settings_manager = get_settings_manager()
    show_splash = _normalize_bool_setting(settings_manager.get("display_splashscreen", True), True)
    splash_language = settings_manager.get("language", "en")
    splash_language_manager = get_language_manager()
    splash_language_manager.set_language(splash_language)
    splash_starting_text = splash_language_manager.translate("ui.messages.starting", "Starting...")

    root = tk.Tk()
    root_transparent_start = False
    if show_splash:
        try:
            root.attributes("-alpha", 0.0)
            root_transparent_start = True
        except tk.TclError:
            root_transparent_start = False
        root.withdraw()

    _set_window_icon(root, "Icon")

    splash = None
    if show_splash:
        splash = tk.Toplevel(root)
        splash.overrideredirect(True)
        splash.resizable(False, False)
        splash.attributes("-topmost", True)
        splash.lift()

        splash_image_path = Path(__file__).parent / "Graphics" / "splash.png"
        loaded_image = False
        if splash_image_path.exists():
            try:
                from PIL import Image, ImageTk, ImageDraw, ImageFont
                splash_image = Image.open(splash_image_path)

                if splash_image.mode != "RGBA":
                    splash_image = splash_image.convert("RGBA")

                try:
                    app_version = _get_application_version("0.0")
                    version_text = f"v{app_version}"
                    if SELAWIK_TTF_PATH.exists():
                        version_font = ImageFont.truetype(str(SELAWIK_TTF_PATH), SPLASH_VERSION_FONT_SIZE)
                        subtitle_font = ImageFont.truetype(str(SELAWIK_TTF_PATH), SPLASH_SUBTITLE_FONT_SIZE)
                    else:
                        version_font = ImageFont.load_default()
                        subtitle_font = ImageFont.load_default()

                    draw = ImageDraw.Draw(splash_image)
                    version_bbox = draw.textbbox((0, 0), version_text, font=version_font)
                    version_width = max(1, version_bbox[2] - version_bbox[0])
                    version_height = max(1, version_bbox[3] - version_bbox[1])
                    subtitle_bbox = draw.textbbox((0, 0), splash_starting_text, font=subtitle_font)
                    subtitle_width = max(1, subtitle_bbox[2] - subtitle_bbox[0])
                    subtitle_height = max(1, subtitle_bbox[3] - subtitle_bbox[1])
                    text_block_width = max(version_width, subtitle_width)
                    text_block_height = version_height + SPLASH_TEXT_LINE_SPACING + subtitle_height

                    margin = max(4, int(min(splash_image.width, splash_image.height) * SPLASH_VERSION_MARGIN_RATIO))
                    anchor_x = int(splash_image.width * SPLASH_VERSION_RELATIVE_POS[0])
                    anchor_y = int(splash_image.height * SPLASH_VERSION_RELATIVE_POS[1])

                    text_x = min(max(margin, anchor_x), splash_image.width - text_block_width - margin)
                    text_y = min(max(margin, anchor_y), splash_image.height - text_block_height - margin)

                    draw.text((text_x, text_y), version_text, font=version_font, fill=(255, 255, 255, 255)) # Starting version text colour
                    subtitle_y = text_y + version_height + SPLASH_TEXT_LINE_SPACING
                    draw.text((text_x, subtitle_y), splash_starting_text, font=subtitle_font, fill=(255, 255, 255, 255)) # Starting text colour
                except Exception:
                    pass

                if platform.system().lower() == "windows":
                    key_rgb = (255, 0, 255)
                    splash_bg = "#ff00ff"

                    alpha_channel = splash_image.getchannel("A")
                    opaque_mask = alpha_channel.point(lambda alpha: 255 if alpha > 0 else 0)
                    rgb_image = splash_image.convert("RGB")
                    key_background = Image.new("RGB", splash_image.size, key_rgb)
                    composited = Image.composite(rgb_image, key_background, opaque_mask)

                    splash_photo = ImageTk.PhotoImage(composited)
                    splash.configure(bg=splash_bg)
                    splash.wm_attributes("-transparentcolor", splash_bg)
                    image_label = tk.Label(
                        splash,
                        image=splash_photo,
                        borderwidth=0,
                        highlightthickness=0,
                        bg=splash_bg,
                    )
                else:
                    splash_photo = ImageTk.PhotoImage(splash_image)
                    image_label = tk.Label(
                        splash,
                        image=splash_photo,
                        borderwidth=0,
                        highlightthickness=0,
                    )

                splash._splash_photo = splash_photo
                image_label.pack(fill=tk.BOTH, expand=True)
                loaded_image = True
            except ImportError:
                try:
                    splash_photo = tk.PhotoImage(file=str(splash_image_path))
                    splash._splash_photo = splash_photo
                    image_label = tk.Label(splash, image=splash_photo, borderwidth=0, highlightthickness=0)
                    image_label.pack(fill=tk.BOTH, expand=True)
                    loaded_image = True
                except tk.TclError:
                    loaded_image = False
            except (tk.TclError, OSError):
                loaded_image = False

        if not loaded_image:
            frame = tk.Frame(splash, padx=26, pady=20, relief=tk.SOLID, borderwidth=1)
            frame.pack(fill=tk.BOTH, expand=True)

            title_label = tk.Label(frame, text="Chemometric Studio", font=("Arial", 14, "bold"))
            title_label.pack(anchor="center", pady=(0, 6))

            status_label = tk.Label(frame, text="Loading...", font=("Arial", 10))
            status_label.pack(anchor="center")

        splash.update_idletasks()
        width = splash.winfo_reqwidth()
        height = splash.winfo_reqheight()
        screen_w = splash.winfo_screenwidth()
        screen_h = splash.winfo_screenheight()
        pos_x = (screen_w // 2) - (width // 2)
        pos_y = (screen_h // 2) - (height // 2)
        splash.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        splash.update()

    try:
        app = ChemometricsGUI(root)
    finally:
        if splash is not None and splash.winfo_exists():
            splash.destroy()

    if show_splash:
        if root_transparent_start:
            try:
                root.attributes("-alpha", 1.0)
            except tk.TclError:
                pass
        root.deiconify()

    root.mainloop()


if __name__ == "__main__":
    main()
