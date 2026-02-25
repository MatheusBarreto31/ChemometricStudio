"""
Add Text Dialog Module

Provides a dialog for adding text sections to the analysis tab with:
- Formatted template text
- Placeholder bindings to input/output variables
- Nested dictionary key access
- Value, index, and range selection (supports first/last)
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
import platform
from typing import Optional, Dict, List, Tuple, Any

from dialog_data_source_utils import (
    append_prefixed_data_sources,
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


class AddTextDialog:
    """Dialog for adding a new text section in the analysis tab."""

    def __init__(self, parent, main_gui, instance_alias: str):
        self.parent = parent
        self.main_gui = main_gui
        self.instance_alias = instance_alias

        self.outputs = self._get_execution_outputs()
        if not self.outputs:
            self._notify(self._t("ui.messages.no_data_run_first", "No data available. Please run 'Run Model' or 'Run to here' first."), level="error")
            return

        self.empty_sections = self._find_empty_sections()
        if not self.empty_sections:
            self._notify(self._t("ui.messages.no_empty_sections", "No empty sections available. Add a new page or remove existing sections first."), level="warning")
            return

        self.bindings: List[Dict[str, Any]] = []

        self.dialog = tk.Toplevel(parent)
        _set_window_icon(self.dialog, "Icon")
        self.dialog.title(self._t("ui.dialogs.add_text", "Add Text"))
        self.dialog.geometry("920x620")
        self._center_window(self.dialog, 920, 620)

        self._build_ui()

    def _notify(self, message: str, level: str = "message"):
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
        if hasattr(self.main_gui, 'language_manager'):
            return self.main_gui.language_manager.translate(key, default)
        return default

    def _center_window(self, window, width: int, height: int):
        window.update_idletasks()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        window.geometry(f"{width}x{height}+{x}+{y}")

    def _get_execution_outputs(self) -> Optional[Dict[str, Any]]:
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

    def _append_prefixed_data_sources(self, combined_sources: Dict[str, Any], execution_results: Dict[str, Any]) -> None:
        append_prefixed_data_sources(
            combined_sources,
            execution_results,
            main_gui=self.main_gui,
            instance_alias=self.instance_alias,
        )

    def _find_empty_sections(self) -> List[Tuple[int, int, str]]:
        empty: List[Tuple[int, int, str]] = []
        pages = self.main_gui.analysis_data.get(self.instance_alias, {}).get('pages', [])
        for page_idx, page in enumerate(pages):
            page_title = page.get('title', f'Page {page_idx + 1}')
            sections = page.get('sections', [])
            for section_idx, section in enumerate(sections):
                if section.get('type') is None:
                    empty.append((page_idx, section_idx, f"Page '{page_title}' - Section {section_idx + 1}"))
        return empty

    def _build_ui(self):
        root = ttk.Frame(self.dialog)
        root.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        top_frame = ttk.LabelFrame(root, text=self._t("ui.labels.target_section", "Target Section"), padding=8)
        top_frame.pack(fill=tk.X, pady=(0, 8))

        self.section_var = tk.StringVar()
        section_combo = ttk.Combobox(
            top_frame,
            textvariable=self.section_var,
            values=[desc for _, _, desc in self.empty_sections],
            state='readonly',
            width=64,
        )
        section_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if self.empty_sections:
            section_combo.current(0)

        title_frame = ttk.LabelFrame(root, text=self._t("ui.labels.section_title", "Section Title"), padding=8)
        title_frame.pack(fill=tk.X, pady=(0, 8))
        self.title_var = tk.StringVar(value=self._t("ui.labels.text", "Text"))
        ttk.Entry(title_frame, textvariable=self.title_var).pack(fill=tk.X)

        main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main_paned)
        right = ttk.Frame(main_paned)
        main_paned.add(left, weight=3)
        main_paned.add(right, weight=2)

        template_frame = ttk.LabelFrame(left, text=self._t("ui.labels.text_template", "Text Template"), padding=8)
        template_frame.pack(fill=tk.BOTH, expand=True)

        hint = ttk.Label(
            template_frame,
            text=self._t(
                "ui.messages.text_template_hint",
                "Use placeholders like {metric}, {vector_item}, {range_values}. Numeric format in bindings uses Python format syntax (e.g., .4f)."
            ),
            foreground='gray'
        )
        hint.pack(anchor='w', pady=(0, 6))

        self.template_text = scrolledtext.ScrolledText(template_frame, wrap=tk.WORD, height=18)
        self.template_text.pack(fill=tk.BOTH, expand=True)
        self.template_text.insert(
            '1.0',
            "Model Summary\n"
            "============\n"
            "RMSE: {rmse}\n"
            "R²: {r2}\n"
            "Top values: {top_values}\n"
        )

        binding_frame = ttk.LabelFrame(right, text=self._t("ui.labels.bindings", "Bindings"), padding=8)
        binding_frame.pack(fill=tk.BOTH, expand=True)

        self._build_binding_editor(binding_frame)

        preview_frame = ttk.LabelFrame(root, text=self._t("ui.labels.preview", "Preview"), padding=8)
        preview_frame.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
        self.preview_text = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD, height=8)
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        self.preview_text.configure(state=tk.DISABLED)

        button_bar = ttk.Frame(root)
        button_bar.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(button_bar, text="⟳ " + self._t("ui.buttons.preview", "Preview"), command=self._preview).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_bar, text="✓ " + self._t("ui.buttons.add_text", "Add Text"), command=self._add_text).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_bar, text="✗ " + self._t("ui.buttons.cancel", "Cancel"), command=self.dialog.destroy).pack(side=tk.LEFT, padx=4)

    def _build_binding_editor(self, parent):
        form = ttk.Frame(parent)
        form.pack(fill=tk.X)

        self.bind_name_var = tk.StringVar()
        self.bind_source_var = tk.StringVar()
        self.bind_nested_var = tk.StringVar()
        self.bind_mode_var = tk.StringVar(value='value')
        self.bind_index_var = tk.StringVar(value='first')
        self.bind_start_var = tk.StringVar(value='first')
        self.bind_end_var = tk.StringVar(value='last')
        self.bind_sep_var = tk.StringVar(value=', ')
        self.bind_fmt_var = tk.StringVar(value='')

        def _row(label: str, widget, row_idx: int):
            ttk.Label(form, text=label).grid(row=row_idx, column=0, sticky='w', padx=(0, 6), pady=2)
            widget.grid(row=row_idx, column=1, sticky='ew', pady=2)

        form.columnconfigure(1, weight=1)

        _row(self._t("ui.labels.placeholder_name", "Placeholder Name"), ttk.Entry(form, textvariable=self.bind_name_var), 0)

        source_combo = ttk.Combobox(
            form,
            textvariable=self.bind_source_var,
            values=get_available_data_sources(self.outputs),
            state='readonly'
        )
        _row(self._t("ui.labels.input_output_source", "Input/Output Source"), source_combo, 1)

        nested_combo = ttk.Combobox(form, textvariable=self.bind_nested_var, values=[], state='normal')
        _row(self._t("ui.labels.nested_key_optional", "Nested key (optional)"), nested_combo, 2)

        mode_combo = ttk.Combobox(form, textvariable=self.bind_mode_var, values=['value', 'index', 'range'], state='readonly')
        _row(self._t("ui.labels.selection_mode", "Selection mode"), mode_combo, 3)

        _row(self._t("ui.labels.index_first_last", "Index (int|first|last)"), ttk.Entry(form, textvariable=self.bind_index_var), 4)
        _row(self._t("ui.labels.range_start", "Range start (int|first|last)"), ttk.Entry(form, textvariable=self.bind_start_var), 5)
        _row(self._t("ui.labels.range_end", "Range end (int|first|last)"), ttk.Entry(form, textvariable=self.bind_end_var), 6)
        _row(self._t("ui.labels.separator", "Separator"), ttk.Entry(form, textvariable=self.bind_sep_var), 7)
        _row(self._t("ui.labels.value_format", "Value format (optional)"), ttk.Entry(form, textvariable=self.bind_fmt_var), 8)

        def _on_source_change(_event=None):
            source = self.bind_source_var.get().strip()
            nested_keys = get_nested_keys(self.outputs, source) if source else []
            nested_combo.configure(values=nested_keys)
            if nested_keys and (self.bind_nested_var.get().strip() not in nested_keys):
                self.bind_nested_var.set(nested_keys[0])

        source_combo.bind('<<ComboboxSelected>>', _on_source_change)

        def _on_mode_change(_event=None):
            mode = self.bind_mode_var.get().strip()
            index_state = 'normal' if mode == 'index' else 'disabled'
            range_state = 'normal' if mode == 'range' else 'disabled'
            for child in form.winfo_children():
                pass
            # Keep enabled/disabled behavior simple by targeting entries via variable ownership.
            for widget in form.grid_slaves(column=1, row=4):
                widget.configure(state=index_state)
            for widget in form.grid_slaves(column=1, row=5):
                widget.configure(state=range_state)
            for widget in form.grid_slaves(column=1, row=6):
                widget.configure(state=range_state)
            for widget in form.grid_slaves(column=1, row=7):
                widget.configure(state='normal' if mode in ('value', 'range') else 'disabled')

        mode_combo.bind('<<ComboboxSelected>>', _on_mode_change)
        _on_mode_change()

        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, pady=(8, 4))
        ttk.Button(controls, text=self._t("ui.buttons.add_binding", "Add Binding"), command=self._add_binding).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(controls, text=self._t("ui.buttons.remove_binding", "Remove Binding"), command=self._remove_binding).pack(side=tk.LEFT)

        self.bindings_list = tk.Listbox(parent, height=10)
        self.bindings_list.pack(fill=tk.BOTH, expand=True)

    def _add_binding(self):
        name = self.bind_name_var.get().strip()
        source = self.bind_source_var.get().strip()
        nested_key = self.bind_nested_var.get().strip()
        mode = self.bind_mode_var.get().strip() or 'value'

        if not name:
            self._notify(self._t("ui.messages.binding_name_required", "Placeholder name is required"), level="warning")
            return
        if not source:
            self._notify(self._t("ui.messages.binding_source_required", "Input/Output source is required"), level="warning")
            return
        if any(item.get('name') == name for item in self.bindings):
            self._notify(self._t("ui.messages.binding_name_unique", "Placeholder name must be unique"), level="warning")
            return

        binding = {
            'name': name,
            'data_source': source,
            'nested_key': nested_key,
            'selector': {
                'mode': mode,
                'index': self.bind_index_var.get().strip(),
                'start': self.bind_start_var.get().strip(),
                'end': self.bind_end_var.get().strip(),
            },
            'separator': self.bind_sep_var.get(),
            'value_format': self.bind_fmt_var.get().strip(),
        }

        self.bindings.append(binding)
        desc = f"{name} <- {source}"
        if nested_key:
            desc += f".{nested_key}"
        desc += f" [{mode}]"
        self.bindings_list.insert(tk.END, desc)

    def _remove_binding(self):
        selection = self.bindings_list.curselection()
        if not selection:
            return
        idx = selection[0]
        del self.bindings[idx]
        self.bindings_list.delete(idx)

    def _build_text_config(self) -> Dict[str, Any]:
        template = self.template_text.get('1.0', tk.END).rstrip('\n')
        title = self.title_var.get().strip() or self._t("ui.labels.text", "Text")
        return {
            'title': title,
            'text_template': template,
            'bindings': self.bindings,
            'wrap': 'word',
        }

    def _preview(self):
        try:
            config = self._build_text_config()
            if hasattr(self.main_gui, '_resolve_text_section_content'):
                content = self.main_gui._resolve_text_section_content(self.outputs, config)
            else:
                content = config.get('text_template', '')

            self.preview_text.configure(state=tk.NORMAL)
            self.preview_text.delete('1.0', tk.END)
            self.preview_text.insert('1.0', content)
            self.preview_text.configure(state=tk.DISABLED)
        except Exception as exc:
            self._notify(self._t("ui.messages.preview_failed", "Failed to generate preview:") + f" {exc}", level="error")

    def _add_text(self):
        selected_section = self.section_var.get().strip()
        if not selected_section:
            self._notify(self._t("ui.messages.select_target_section", "Please select a target section"), level="warning")
            return

        section_idx = next((idx for idx, (_, _, desc) in enumerate(self.empty_sections) if desc == selected_section), -1)
        if section_idx < 0:
            self._notify(self._t("ui.messages.selected_target_section_invalid", "Selected target section is invalid"), level="error")
            return

        page_idx, sec_idx, _ = self.empty_sections[section_idx]
        config = self._build_text_config()

        pages = self.main_gui.analysis_data[self.instance_alias]['pages']
        pages[page_idx]['sections'][sec_idx] = {
            'type': 'text',
            'config': config,
        }

        self.main_gui._show_analysis_tab()
        self.dialog.destroy()
        self._notify(self._t("ui.messages.text_added_to", "Text added to") + f" Page {page_idx + 1}, Section {sec_idx + 1}", level="success")


def show_add_text_dialog(parent, main_gui, instance_alias: str):
    """Show the Add Text dialog."""
    AddTextDialog(parent, main_gui, instance_alias)
