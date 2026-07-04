from __future__ import annotations

import shutil
import threading
import time
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk

from main_gui import (
    ChemometricsGUI,
    SELAWIK_TTF_PATH,
    SPLASH_SUBTITLE_FONT_SIZE,
    SPLASH_TEXT_LINE_SPACING,
    SPLASH_VERSION_FONT_SIZE,
    SPLASH_VERSION_MARGIN_RATIO,
    SPLASH_VERSION_RELATIVE_POS,
    _get_application_version,
    _normalize_bool_setting,
    _set_window_icon,
    _ui_symbol,
    get_language_manager,
    get_settings_manager,
)
class UserChemometricsGUI(ChemometricsGUI):
    """Simplified GUI flow for non-specialized users."""

    USER_STAGE_START = "start"
    USER_STAGE_READY_TO_RUN = "ready_to_run"
    USER_STAGE_GUIDED_SETUP = "guided_setup"
    USER_STAGE_POST_RUN = "post_run"

    def __init__(self, root: tk.Tk):
        self.user_stage: str = self.USER_STAGE_START
        self.loaded_model_suffix: str = ""
        self.guided_setup_indices: List[int] = []
        self.guided_setup_pointer: int = 0
        self._guided_prev_btn: Optional[ttk.Button] = None
        self._guided_next_btn: Optional[ttk.Button] = None
        self._guided_progress_var: Optional[tk.StringVar] = None
        super().__init__(root)

    def _refresh_ui_text(self):
        """Refresh all visible text while preserving state in user mode."""
        self._load_gui_configs()
        self.root.title(self.language_manager.translate("ui.main_title", "Chemometric Studio"))

        current_methodology_list = self.methodology_list.copy()
        current_base_aliases = self.function_base_aliases.copy()
        current_configs = self.function_configs.copy()
        current_routing = self.routing_lines.copy()
        current_selected_idx = self.selected_function_idx

        for widget in self.root.winfo_children():
            widget.destroy()

        self._build_menu_bar()
        self._build_ui()

        self.methodology_list = current_methodology_list
        self.function_base_aliases = current_base_aliases
        self.function_configs = current_configs
        self.routing_lines = current_routing
        self.selected_function_idx = current_selected_idx

        self._refresh_methodology_listbox(selected_idx=current_selected_idx)

        if self.user_stage == self.USER_STAGE_GUIDED_SETUP:
            self._show_current_guided_setup_function()
        elif self.user_stage == self.USER_STAGE_POST_RUN:
            if self._custom_analysis_has_content():
                self._show_custom_analysis_tab()
            else:
                self._show_analysis_tab()

    def _build_ui(self):
        """Build UI according to current user flow stage."""
        if self.user_stage in {self.USER_STAGE_START, self.USER_STAGE_READY_TO_RUN}:
            self._build_user_start_screen()
            return

        if self.user_stage == self.USER_STAGE_GUIDED_SETUP:
            self._build_user_guided_setup_layout()
            self._show_current_guided_setup_function()
            return

        self._build_user_post_run_layout()
        if self.selected_function_idx is None and self.methodology_list:
            self.selected_function_idx = 0
        self._refresh_methodology_listbox(selected_idx=self.selected_function_idx)

    def _build_user_start_screen(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.workspace_frame = container

        center = ttk.Frame(container)
        center.place(relx=0.5, rely=0.5, anchor="center")

        load_btn = ttk.Button(
            center,
            text="📂 " + self.language_manager.translate("ui.buttons.load_model", "Load Model"),
            command=self._show_load_model_dialog,
            width=26,
        )
        load_btn.pack(pady=4)

        if self.user_stage == self.USER_STAGE_READY_TO_RUN:
            run_btn = ttk.Button(
                center,
                text="🠊 " + self.language_manager.translate("ui.buttons.run_model", "Run Model"),
                command=self._run_model,
                width=26,
            )
            run_btn.pack(pady=4)

    def _build_user_guided_setup_layout(self) -> None:
        left_frame = ttk.Frame(self.root, width=260)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=10, pady=10)
        left_frame.pack_propagate(False)

        methodology_label = ttk.Label(
            left_frame,
            text=self.language_manager.translate("ui.panels.methodology", "Methodology"),
            font=("Arial", 10, "bold"),
        )
        methodology_label.pack(pady=(0, 5))

        self._build_user_methodology_panel(left_frame, selectable=False)

        workspace_frame = ttk.Frame(self.root)
        workspace_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))
        self.workspace_frame = workspace_frame

        if not self._execution_progress_root_bind_set:
            self.root.bind("<Configure>", self._on_execution_progress_anchor_configure, add="+")
            self._execution_progress_root_bind_set = True
        workspace_frame.bind("<Configure>", self._on_execution_progress_anchor_configure, add="+")

        self._build_guided_setup_controls(workspace_frame)

        self.tab_content_frame = ttk.Frame(workspace_frame)
        self.tab_content_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(10, 3))

    def _build_user_post_run_layout(self) -> None:
        left_frame = ttk.Frame(self.root, width=260)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=10, pady=10)
        left_frame.pack_propagate(False)

        methodology_label = ttk.Label(
            left_frame,
            text=self.language_manager.translate("ui.panels.methodology", "Methodology"),
            font=("Arial", 10, "bold"),
        )
        methodology_label.pack(pady=(0, 5))

        self._build_user_methodology_panel(left_frame, selectable=True)

        workspace_frame = ttk.Frame(self.root)
        workspace_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))
        self.workspace_frame = workspace_frame

        if not self._execution_progress_root_bind_set:
            self.root.bind("<Configure>", self._on_execution_progress_anchor_configure, add="+")
            self._execution_progress_root_bind_set = True
        workspace_frame.bind("<Configure>", self._on_execution_progress_anchor_configure, add="+")

        self._build_user_post_run_control_bar(workspace_frame)

        self.tab_content_frame = ttk.Frame(workspace_frame)
        self.tab_content_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(10, 3))

    def _build_user_methodology_panel(self, parent: ttk.Frame, selectable: bool) -> None:
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
        if selectable:
            self.methodology_listbox.bind("<<ListboxSelect>>", self._on_methodology_select)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.methodology_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.methodology_listbox.config(yscrollcommand=scrollbar.set)
        self._configure_methodology_listbox_theme()

    def _build_guided_setup_controls(self, parent: ttk.Frame) -> None:
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=0, pady=(0, 8))

        title = ttk.Label(
            control_frame,
            text=self.language_manager.translate("ui.tabs.setup", "Setup"),
            font=("Arial", 11, "bold"),
        )
        title.pack(side=tk.LEFT, padx=(0, 8))

        self._guided_progress_var = tk.StringVar(value="")
        progress_label = ttk.Label(control_frame, textvariable=self._guided_progress_var, font=("Arial", 9))
        progress_label.pack(side=tk.LEFT, padx=(0, 8))

        self._guided_prev_btn = ttk.Button(
            control_frame,
            text=_ui_symbol("prev") + " " + self.language_manager.translate("ui.buttons.previous", "Previous"),
            command=self._guided_setup_previous,
            width=12,
        )
        self._guided_prev_btn.pack(side=tk.LEFT, padx=4)

        self._guided_next_btn = ttk.Button(control_frame, width=12)
        self._guided_next_btn.pack(side=tk.LEFT, padx=4)

        spacer = ttk.Frame(control_frame)
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)

        load_btn = ttk.Button(
            control_frame,
            text="📂 " + self.language_manager.translate("ui.buttons.load_model", "Load Model"),
            command=self._show_load_model_dialog,
            width=14,
        )
        load_btn.pack(side=tk.RIGHT, padx=4)

        self._refresh_guided_navigation_widgets()

    def _build_user_post_run_control_bar(self, parent: ttk.Frame) -> None:
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        analysis_btn = ttk.Button(
            control_frame,
            text=self.language_manager.translate("ui.tabs.analysis", "Analysis"),
            command=self._show_analysis_tab,
            width=12,
        )
        analysis_btn.pack(side=tk.LEFT, padx=5)

        custom_analysis_btn = ttk.Button(
            control_frame,
            text=self.language_manager.translate("ui.tabs.custom_analysis", "C. Analysis"),
            command=self._show_custom_analysis_tab,
            width=12,
        )
        custom_analysis_btn.pack(side=tk.LEFT, padx=5)

        report_btn = ttk.Button(
            control_frame,
            text=self.language_manager.translate("ui.tabs.report", "Report"),
            command=self._generate_and_open_pdf_report,
            width=12,
        )
        report_btn.pack(side=tk.LEFT, padx=5)

        spacer = ttk.Frame(control_frame)
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)

        save_btn = ttk.Button(
            control_frame,
            text="💾 " + self.language_manager.translate("ui.buttons.save_model", "Save Model"),
            command=self._show_save_model_dialog,
            width=14,
        )
        save_btn.pack(side=tk.RIGHT, padx=5)

        load_btn = ttk.Button(
            control_frame,
            text="📂 " + self.language_manager.translate("ui.buttons.load_model", "Load Model"),
            command=self._show_load_model_dialog,
            width=14,
        )
        load_btn.pack(side=tk.RIGHT, padx=5)

        run_btn = ttk.Button(
            control_frame,
            text="🠊 " + self.language_manager.translate("ui.buttons.run_model", "Run Model"),
            command=self._run_model,
            width=12,
        )
        run_btn.pack(side=tk.RIGHT, padx=5)

    def _show_analysis_tab(self):
        """Show analysis tab with user-mode control restrictions."""
        super()._show_analysis_tab()

        if getattr(self, "current_tab", None) != "analysis":
            return
        if self.selected_function_idx is None or self.selected_function_idx >= len(self.methodology_list):
            return

        instance_alias = self.methodology_list[self.selected_function_idx]

        children = self.tab_content_frame.winfo_children()
        if not children:
            return

        control_frame = children[0]
        if not isinstance(control_frame, ttk.Frame):
            return

        add_graph_label = self.language_manager.translate("ui.buttons.add_graph", "Add graph")
        add_table_label = self.language_manager.translate("ui.buttons.add_table", "Add table")
        add_text_label = self.language_manager.translate("ui.buttons.add_text", "Add text")

        for widget in list(control_frame.winfo_children()):
            if not isinstance(widget, ttk.Button):
                continue
            text = str(widget.cget("text") or "")
            if text == add_graph_label or text == add_table_label:
                widget.destroy()
            elif text == add_text_label:
                widget.configure(
                    command=lambda alias=instance_alias: self._show_simple_add_text_dialog(
                        alias,
                        refresh_callback=self._show_analysis_tab,
                    )
                )

    def _show_custom_add_text_dialog(self) -> None:
        """Keep user-mode custom text insertion aligned with base GUI behavior."""
        super()._show_custom_add_text_dialog()

    def _render_text_section(self, parent: ttk.Frame, instance_alias: str, section_data: dict):
        """Use base text rendering so static custom text works without bindings."""
        super()._render_text_section(parent, instance_alias, section_data)

    def _refresh_gui_from_config(self):
        """Refresh internal data after model load in user mode."""
        self._refresh_methodology_listbox()
        self.selected_function_idx = None

    def _load_model(self, file_path: str):
        """Load model and transition user flow based on model extension."""
        model_path = Path(file_path)
        self.loaded_model_suffix = model_path.suffix.lower()
        super()._load_model(file_path)
        self._handle_post_model_load()

    def _handle_post_model_load(self) -> None:
        suffix = self.loaded_model_suffix

        if suffix in {".mdcd", ".mdon"}:
            self.guided_setup_indices = self._collect_guided_setup_indices(suffix)
            self.guided_setup_pointer = 0
            if self.guided_setup_indices:
                self.user_stage = self.USER_STAGE_GUIDED_SETUP
                self._rebuild_user_layout()
                return
            self.user_stage = self.USER_STAGE_READY_TO_RUN
            self._rebuild_user_layout()
            return

        if suffix == ".mdfd":
            self.user_stage = self.USER_STAGE_READY_TO_RUN
            self._rebuild_user_layout()
            return

        self.user_stage = self.USER_STAGE_START
        self._rebuild_user_layout()

    def _collect_guided_setup_indices(self, suffix: str) -> List[int]:
        required_aliases = {"validation_data_main"}
        if suffix == ".mdon":
            required_aliases.add("load_data")

        indices: List[int] = []
        for idx, base_alias in enumerate(self.function_base_aliases):
            if base_alias in required_aliases:
                indices.append(idx)
        return indices

    def _guided_setup_previous(self) -> None:
        if self.guided_setup_pointer <= 0:
            return
        self.guided_setup_pointer -= 1
        self._show_current_guided_setup_function()

    def _guided_setup_next_or_run(self) -> None:
        if self.guided_setup_pointer < len(self.guided_setup_indices) - 1:
            self.guided_setup_pointer += 1
            self._show_current_guided_setup_function()
            return
        self._run_model()

    def _refresh_guided_navigation_widgets(self) -> None:
        total = len(self.guided_setup_indices)
        current = self.guided_setup_pointer + 1 if total else 0

        if self._guided_progress_var is not None:
            self._guided_progress_var.set(f"{current}/{total}" if total else "")

        if self._guided_prev_btn is not None:
            prev_state = tk.NORMAL if self.guided_setup_pointer > 0 else tk.DISABLED
            self._guided_prev_btn.configure(state=prev_state)

        if self._guided_next_btn is not None:
            if self.guided_setup_pointer < total - 1:
                self._guided_next_btn.configure(
                    text=self.language_manager.translate("ui.buttons.next", "Next") + " " + _ui_symbol("next"),
                    command=self._guided_setup_next_or_run,
                )
            else:
                self._guided_next_btn.configure(
                    text="🠊 " + self.language_manager.translate("ui.buttons.run_model", "Run Model"),
                    command=self._guided_setup_next_or_run,
                )

    def _show_current_guided_setup_function(self) -> None:
        if not self.guided_setup_indices:
            return

        if self.guided_setup_pointer < 0:
            self.guided_setup_pointer = 0
        if self.guided_setup_pointer >= len(self.guided_setup_indices):
            self.guided_setup_pointer = len(self.guided_setup_indices) - 1

        target_idx = self.guided_setup_indices[self.guided_setup_pointer]
        if not (0 <= target_idx < len(self.methodology_list)):
            return

        self.selected_function_idx = target_idx
        self._refresh_methodology_listbox(selected_idx=target_idx)
        self._refresh_guided_navigation_widgets()
        super()._show_setup_tab()

    def _run_model(self):
        """Run model and switch to simplified post-run view on success."""
        super()._run_model()

        if not self._report_has_successful_execution():
            return

        self.user_stage = self.USER_STAGE_POST_RUN
        self._rebuild_user_layout()

        if self.selected_function_idx is None and self.methodology_list:
            self.selected_function_idx = 0
        self._refresh_methodology_listbox(selected_idx=self.selected_function_idx)

        if self._custom_analysis_has_content():
            self._show_custom_analysis_tab()
        else:
            self._show_analysis_tab()

    def _custom_analysis_has_content(self) -> bool:
        analysis_state = self._ensure_custom_analysis_state()
        pages = analysis_state.get("pages", []) if isinstance(analysis_state, dict) else []

        for page in pages:
            if not isinstance(page, dict):
                continue
            sections = page.get("sections", [])
            if not isinstance(sections, list):
                continue
            for section in sections:
                if not isinstance(section, dict):
                    continue
                if section.get("type") is not None:
                    return True
        return False

    def _generate_and_open_pdf_report(self) -> None:
        """Generate report PDF directly and open it, instead of opening the report tab."""
        if not self._report_has_successful_execution():
            self._show_fading_warning(
                self.language_manager.translate(
                    "ui.messages.run_model_before_report",
                    "Run the model before previewing or saving the report.",
                )
            )
            return

        if not self.report_data.get("elements"):
            self._show_fading_warning(
                self.language_manager.translate(
                    "ui.messages.no_report_elements",
                    "Add at least one report element before saving.",
                )
            )
            return

        reports_dir = self.tempfiles_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = reports_dir / f"CMStudio_report_{timestamp}.pdf"

        assets_dir = reports_dir / f"{output_pdf.stem}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        elements = self._build_resolved_report_elements(assets_dir)
        build_latex_document, compile_latex_to_pdf = self._get_reporting_functions()
        latex_source = build_latex_document(
            elements,
            self.language_manager.get_language(),
            self._build_report_footer_text(),
        )

        self._show_execution_progress_message(
            self.language_manager.translate(
                "ui.messages.generating_pdf_wait",
                "PDF is being generated. Please wait...",
            )
        )

        compile_result_holder: Dict[str, Any] = {}
        compile_exception_holder: Dict[str, str] = {}
        compile_done = threading.Event()

        def _compile_worker() -> None:
            try:
                compile_result_holder["result"] = compile_latex_to_pdf(latex_source, str(output_pdf))
            except Exception as exc:
                compile_exception_holder["error"] = str(exc)
            finally:
                compile_done.set()

        compile_thread = threading.Thread(target=_compile_worker, daemon=True)
        compile_thread.start()

        try:
            while not compile_done.is_set():
                self.root.update()
                time.sleep(0.03)

            if compile_exception_holder.get("error"):
                compile_result = {
                    "success": False,
                    "pdf_path": None,
                    "tex_path": str(output_pdf.with_suffix(".tex")),
                    "error": compile_exception_holder["error"],
                }
            else:
                compile_result = compile_result_holder.get("result") or {
                    "success": False,
                    "pdf_path": None,
                    "tex_path": str(output_pdf.with_suffix(".tex")),
                    "error": self.language_manager.translate(
                        "ui.messages.report_compile_failed",
                        "Report compilation failed.",
                    ),
                }
        finally:
            self._stop_execution_progress_message()

        if compile_result.get("success"):
            for suffix in (".tex", ".aux", ".log", ".out", ".toc"):
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
                self.language_manager.translate("ui.messages.report_saved_pdf", "Report saved successfully:")
                + f"\n{compile_result.get('pdf_path')}"
            )
            self._open_path_with_system_default(Path(compile_result.get("pdf_path", str(output_pdf))))
            return

        tex_path = compile_result.get("tex_path")
        error_message = compile_result.get("error") or self.language_manager.translate(
            "ui.messages.report_compile_failed",
            "Report compilation failed.",
        )
        self._show_fading_warning(
            self.language_manager.translate(
                "ui.messages.report_tex_saved",
                "LaTeX source saved, but PDF compilation failed.",
            )
            + f"\n{tex_path}\n\n{error_message}"
        )

    def _rebuild_user_layout(self) -> None:
        current_methodology_list = self.methodology_list.copy()
        current_base_aliases = self.function_base_aliases.copy()
        current_configs = self.function_configs.copy()
        current_routing = self.routing_lines.copy()
        current_selected_idx = self.selected_function_idx

        for widget in self.root.winfo_children():
            widget.destroy()

        self._build_menu_bar()
        self._build_ui()
        self._load_theme()

        self.methodology_list = current_methodology_list
        self.function_base_aliases = current_base_aliases
        self.function_configs = current_configs
        self.routing_lines = current_routing
        self.selected_function_idx = current_selected_idx

        self._refresh_methodology_listbox(selected_idx=current_selected_idx)


def main() -> None:
    """Entry point for user-mode GUI."""
    settings_manager = get_settings_manager()
    show_splash = _normalize_bool_setting(settings_manager.get("display_splashscreen", True), True)
    splash_language = settings_manager.get("language", "en")
    splash_language_manager = get_language_manager()
    splash_language_manager.set_language(splash_language)
    splash_starting_text = splash_language_manager.translate("ui.messages.starting", "Starting...")
    user_mode_prefix = splash_language_manager.translate("ui.messages.user_mode_prefix", "User mode - ")

    root = tk.Tk()
    root_transparent_start = False
    if show_splash:
        try:
            root.attributes("-alpha", 0.0)
            root_transparent_start = True
        except tk.TclError:
            root_transparent_start = False
        root.withdraw()

    _set_window_icon(root, "icon-user")

    splash = None
    if show_splash:
        splash = tk.Toplevel(root)
        splash.overrideredirect(True)
        splash.resizable(False, False)
        splash.attributes("-topmost", True)
        splash.lift()

        splash_image_path = Path(__file__).parent / "Graphics" / "splash-user.png"
        loaded_image = False
        if splash_image_path.exists():
            try:
                from PIL import Image, ImageTk, ImageDraw, ImageFont
                splash_image = Image.open(splash_image_path)

                if splash_image.mode != "RGBA":
                    splash_image = splash_image.convert("RGBA")

                try:
                    app_version = _get_application_version("0.0")
                    version_text = f"{user_mode_prefix}v{app_version}"
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

                    draw.text((text_x, text_y), version_text, font=version_font, fill=(255, 255, 255, 255))
                    subtitle_y = text_y + version_height + SPLASH_TEXT_LINE_SPACING
                    draw.text((text_x, subtitle_y), splash_starting_text, font=subtitle_font, fill=(255, 255, 255, 255))
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
        UserChemometricsGUI(root)
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
