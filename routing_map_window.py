"""
Routing Map Window - Displays a full view of all functions and their routing connections.

This module provides a standalone window showing the complete routing map with all functions,
their inputs/outputs, and the connections between them. Inherited variables are highlighted
in a different color than routed variables.

Connections are drawn between specific outputs and inputs, with intelligent routing to 
avoid crossing when possible. Arrows that skip functions curve around them.

Can be easily removed if no longer needed - just delete this file and remove the 
"Routing Map" button from main_gui.py.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont, ImageTk
import platform


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
            icon_image = Image.open(ico_path)
            icon_photo = ImageTk.PhotoImage(icon_image)
            window.iconphoto(True, icon_photo)
            window._icon_photo = icon_photo
        except Exception:
            pass


class RoutingMapWindow:
    """Display a full routing map in a new window."""
    HIDDEN_NODE_ALIASES = {
        "workflow_loop_end",
        "workflow_parallel_branch",
        "workflow_parallel_end",
        "workflow_ensemble_member",
        "workflow_ensemble_end",
    }

    CONTROL_NODE_STYLES = {
        "workflow_loop_start": {
            "title_bg": "#dbeafe",
            "body_bg": "#f0f7ff",
            "outline": "#3b82f6",
            "label": "Loop Start",
        },
        "workflow_loop_end": {
            "title_bg": "#dbeafe",
            "body_bg": "#f0f7ff",
            "outline": "#3b82f6",
            "label": "Loop End",
        },
        "workflow_parallel_start": {
            "title_bg": "#e9d5ff",
            "body_bg": "#f8f1ff",
            "outline": "#8b5cf6",
            "label": "Parallel Start",
        },
        "workflow_parallel_branch": {
            "title_bg": "#f3e8ff",
            "body_bg": "#faf5ff",
            "outline": "#a855f7",
            "label": "Parallel Branch",
        },
        "workflow_parallel_end": {
            "title_bg": "#e9d5ff",
            "body_bg": "#f8f1ff",
            "outline": "#8b5cf6",
            "label": "Parallel End",
        },
        "workflow_ensemble_start": {
            "title_bg": "#d1fae5",
            "body_bg": "#ecfdf5",
            "outline": "#059669",
            "label": "Ensemble Start",
        },
        "workflow_ensemble_member": {
            "title_bg": "#ccfbf1",
            "body_bg": "#f0fdfa",
            "outline": "#0d9488",
            "label": "Ensemble Member",
        },
        "workflow_ensemble_end": {
            "title_bg": "#d1fae5",
            "body_bg": "#ecfdf5",
            "outline": "#059669",
            "label": "Ensemble End",
        },
    }
    
    def __init__(self, parent, methodology_list: List[str], function_base_aliases: List[str],
                 routing_lines: Dict[Tuple, Dict], gui_configs: Dict[str, Dict],
                 function_specs: Dict[str, Dict], function_instance_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the routing map window.
        
        Args:
            parent: Parent window
            methodology_list: List of instance aliases in the methodology
            function_base_aliases: List of base aliases for each function
            routing_lines: Dictionary of routing connections
            gui_configs: Dictionary of GUI configurations for functions
            function_specs: Dictionary of function specifications with field definitions
        """
        self.methodology_list = methodology_list
        self.function_base_aliases = function_base_aliases
        self.routing_lines = routing_lines
        self.gui_configs = gui_configs
        self.function_specs = function_specs
        self.function_instance_configs = function_instance_configs or {}
        
        # Load individual function config files to get input_type info
        self.function_configs_loaded = {}
        for func_name in set(function_base_aliases):
            config_info = function_specs.get("gui_listing", {}).get(func_name, {})
            config_path = config_info.get("config_path", "")
            if config_path:
                # Try to load with language folder - default to 'en'
                # Try multiple language folders
                for lang in ['en', 'pt-br']:
                    try:
                        # Insert language folder into the path
                        # e.g., "gui_configs/load_data_config.json" -> "gui_configs/en/load_data_config.json"
                        path_parts = config_path.split('/')
                        if len(path_parts) >= 2:
                            path_parts.insert(1, lang)
                            lang_config_path = '/'.join(path_parts)
                        else:
                            lang_config_path = config_path
                        
                        full_path = Path(__file__).parent / lang_config_path
                        
                        if full_path.exists():
                            with open(full_path, encoding='utf-8') as f:
                                self.function_configs_loaded[func_name] = json.load(f)
                            break
                    except Exception as e:
                        continue
        
        # Create window - responsive to parent screen size
        self.window = tk.Toplevel(parent)
        _set_window_icon(self.window, "Icon")
        self.window.title("Routing Map - Full View")
        
        # Get parent window size and set routing map to use most of screen while leaving room for taskbar/title
        parent_width = parent.winfo_width() if parent.winfo_width() > 1 else 1280
        parent_height = parent.winfo_height() if parent.winfo_height() > 1 else 720
        
        # Use 90% of parent's dimensions or 720p default, whichever is available
        window_width = max(int(parent_width * 0.9), 1000)
        window_height = max(int(parent_height * 0.85), 650)
        self.window.geometry(f"{window_width}x{window_height}")
        
        # Configure style - white background to match canvas
        self.window.configure(bg="white")
        
        # Zoom level tracking
        self.zoom_level = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 3.0
        self.zoom_step = 1.15  # 15% zoom increment
        self.show_connected_only = tk.BooleanVar(value=True)
        self.visible_indices: List[int] = []
        self.top_overlay_bottom = 0
        
        # Create canvas with scrollbars
        self._create_canvas()
        
        # Draw the routing map
        self._draw_routing_map()
    
    def _get_input_type(self, func_name: str, param_name: str) -> str:
        """
        Get the input_type for a parameter in a function.
        
        Args:
            func_name: Base function name (e.g., 'load_data')
            param_name: Internal name of the parameter (key from input_aliases)
        
        Returns:
            The input_type: 'user', 'inherited', 'routed', etc.
        """
        config = self.function_configs_loaded.get(func_name, {})
        
        # Look in setup.layout array for the field matching the name
        setup = config.get("setup", {})
        layout = setup.get("layout", [])
        
        # First, try to match by internal name directly
        for field_info in layout:
            if field_info.get("name") == param_name:
                return field_info.get("input_type", "routed")
        
        # If not found by name, try matching by label (display name)
        # This handles cases where param_name might be a display name
        for field_info in layout:
            if field_info.get("label") == param_name:
                return field_info.get("input_type", "routed")
        
        return "routed"  # Default to routed if not found

    def _is_hidden_visual_node(self, base_alias: str) -> bool:
        """Return True when a control alias should be collapsed to a separator in the map."""
        return base_alias in self.HIDDEN_NODE_ALIASES

    def _get_passforward_config(self, base_alias: str) -> Dict[str, Any]:
        config = self.gui_configs.get(base_alias, {})
        raw_cfg = config.get("passforward", {})
        if not isinstance(raw_cfg, dict) or not bool(raw_cfg.get("compatible", False)):
            return {}
        mappings = raw_cfg.get("mappings", {})
        if not isinstance(mappings, dict) or not mappings:
            return {}
        return raw_cfg

    def _is_passforward_enabled_for_index(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.methodology_list) or idx >= len(self.function_base_aliases):
            return False
        base_alias = self.function_base_aliases[idx]
        if not self._get_passforward_config(base_alias):
            return False
        instance_alias = self.methodology_list[idx]
        cfg = self.function_instance_configs.get(instance_alias, {}) if isinstance(self.function_instance_configs, dict) else {}
        return bool(cfg.get("__passforward_enabled__", False))

    def _get_output_aliases_for_index(self, idx: int) -> Dict[str, str]:
        if idx < 0 or idx >= len(self.function_base_aliases):
            return {}
        base_alias = self.function_base_aliases[idx]
        config = self.gui_configs.get(base_alias, {})
        outputs = dict(config.get("output_aliases", {}))

        if not self._is_passforward_enabled_for_index(idx):
            return outputs

        passforward_cfg = self._get_passforward_config(base_alias)
        mappings = passforward_cfg.get("mappings", {}) if isinstance(passforward_cfg, dict) else {}
        for dst_key, mapping in mappings.items():
            if not isinstance(dst_key, str) or not dst_key.strip():
                continue
            label = ""
            if isinstance(mapping, dict):
                label = str(mapping.get("label", "") or mapping.get("display_name", "")).strip()
            outputs[dst_key] = label or f"Passforward {dst_key}"
        return outputs

    def _build_connection_lists(self, visible_idx_set: set) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build routed and inherited connections restricted to visually rendered nodes."""
        routed_connections: List[Dict[str, Any]] = []
        inherited_connections: List[Dict[str, Any]] = []

        # Keep only one effective explicit route per destination key.
        # Manual links always win over auto-created links.
        explicit_by_destination: Dict[Tuple[int, str], Dict[str, Any]] = {}

        for key, conn_info in self.routing_lines.items():
            info = conn_info if isinstance(conn_info, dict) else {}
            src_idx = info.get("src_idx", key[0])
            src_param_key = info.get("src_param_key", key[1])
            dst_idx = info.get("dst_idx", key[2])
            dst_param_key = info.get("dst_param_key", key[3])

            try:
                src_idx = int(src_idx)
                dst_idx = int(dst_idx)
            except (TypeError, ValueError):
                continue

            if src_idx not in visible_idx_set or dst_idx not in visible_idx_set:
                continue

            conn_record = {
                "src_idx": src_idx,
                "src_param_key": src_param_key,
                "dst_idx": dst_idx,
                "dst_param_key": dst_param_key,
                "auto_created": bool(info.get("auto_created", False)),
            }

            dst_key = (dst_idx, str(dst_param_key))
            current = explicit_by_destination.get(dst_key)
            if current is None:
                explicit_by_destination[dst_key] = conn_record
                continue

            current_auto = bool(current.get("auto_created", False))
            new_auto = conn_record["auto_created"]

            if current_auto and not new_auto:
                explicit_by_destination[dst_key] = conn_record
                continue

            if current_auto == new_auto and conn_record["src_idx"] > int(current.get("src_idx", -1)):
                # Deterministic tie-breaker for legacy duplicated links.
                explicit_by_destination[dst_key] = conn_record

        for conn in explicit_by_destination.values():
            src_idx = conn["src_idx"]
            src_param_key = conn["src_param_key"]
            dst_idx = conn["dst_idx"]
            dst_param_key = conn["dst_param_key"]

            dst_base = self.function_base_aliases[dst_idx]
            dst_config = self.gui_configs.get(dst_base, {})
            dst_inputs = dst_config.get("input_aliases", {})
            dst_param_name = dst_inputs.get(dst_param_key, "")
            is_inherited_input = self._get_input_type(dst_base, dst_param_key) == "inherited"

            conn_data = {
                "src_idx": src_idx,
                "src_param_key": src_param_key,
                "dst_idx": dst_idx,
                "dst_param_key": dst_param_key,
                "dst_param_name": dst_param_name,
                "is_inherited": is_inherited_input,
            }
            if is_inherited_input:
                inherited_connections.append(conn_data)
            else:
                routed_connections.append(conn_data)

        inherited_seen = {
            (conn["src_idx"], conn["src_param_key"], conn["dst_idx"], conn["dst_param_key"])
            for conn in inherited_connections
        }

        for dst_idx in self.visible_indices:
            dst_base = self.function_base_aliases[dst_idx]
            dst_config = self.gui_configs.get(dst_base, {})
            dst_inputs = dst_config.get("input_aliases", {})

            for dst_param_key, dst_param_name in dst_inputs.items():
                if self._get_input_type(dst_base, dst_param_key) != "inherited":
                    continue

                already_routed = False
                for key, conn_info in self.routing_lines.items():
                    if (conn_info.get("dst_idx", key[2]) == dst_idx and
                        conn_info.get("dst_param_key", key[3]) == dst_param_key):
                        already_routed = True
                        break
                if already_routed:
                    continue

                src_idx = None
                src_param_key = None

                # Prefer non-inherited input sources so inherited arrows originate
                # from the providing input (not from unrelated previous outputs).
                for prev_idx in range(dst_idx - 1, -1, -1):
                    if prev_idx not in visible_idx_set:
                        continue
                    prev_base = self.function_base_aliases[prev_idx]
                    prev_config = self.gui_configs.get(prev_base, {})
                    prev_inputs = prev_config.get("input_aliases", {})
                    if dst_param_key in prev_inputs and self._get_input_type(prev_base, dst_param_key) != "inherited":
                        src_idx = prev_idx
                        src_param_key = dst_param_key
                        break

                if src_idx is None:
                    for prev_idx in range(dst_idx - 1, -1, -1):
                        if prev_idx not in visible_idx_set:
                            continue
                        prev_base = self.function_base_aliases[prev_idx]
                        prev_config = self.gui_configs.get(prev_base, {})
                        prev_outputs = self._get_output_aliases_for_index(prev_idx)
                        if dst_param_key in prev_outputs:
                            src_idx = prev_idx
                            src_param_key = dst_param_key
                            break

                if src_idx is None:
                    continue

                conn_key = (src_idx, src_param_key, dst_idx, dst_param_key)
                if conn_key in inherited_seen:
                    continue

                inherited_seen.add(conn_key)
                inherited_connections.append({
                    "src_idx": src_idx,
                    "src_param_key": src_param_key,
                    "dst_idx": dst_idx,
                    "dst_param_key": dst_param_key,
                    "dst_param_name": dst_param_name,
                    "is_inherited": True,
                })

        routed_connections.sort(key=lambda c: (c["dst_idx"] - c["src_idx"], c["src_idx"], c["dst_idx"]))
        inherited_connections.sort(key=lambda c: (c["dst_idx"] - c["src_idx"], c["src_idx"], c["dst_idx"]))
        return routed_connections, inherited_connections

    def _collect_connected_ports(self, routed_connections: List[Dict[str, Any]],
                                 inherited_connections: List[Dict[str, Any]]) -> Tuple[Dict[int, set], Dict[int, set]]:
        """Build lookup sets for connected input/output keys per node index."""
        connected_inputs: Dict[int, set] = {}
        connected_outputs: Dict[int, set] = {}

        for conn in routed_connections + inherited_connections:
            connected_outputs.setdefault(conn["src_idx"], set()).add(conn["src_param_key"])
            connected_inputs.setdefault(conn["dst_idx"], set()).add(conn["dst_param_key"])

            # Inherited links can originate from a source input; keep that source
            # input visible in connected-only mode so the orange arrow endpoint exists.
            src_idx = conn["src_idx"]
            src_param_key = conn["src_param_key"]
            src_base = self.function_base_aliases[src_idx]
            src_inputs = self.gui_configs.get(src_base, {}).get("input_aliases", {})
            if src_param_key in src_inputs:
                connected_inputs.setdefault(src_idx, set()).add(src_param_key)

        visible_idx_set = set(self.visible_indices)

        # Include explicit connections even when one endpoint is hidden from the visual map.
        for key, conn_info in self.routing_lines.items():
            src_idx = conn_info.get("src_idx", key[0])
            src_param_key = conn_info.get("src_param_key", key[1])
            dst_idx = conn_info.get("dst_idx", key[2])
            dst_param_key = conn_info.get("dst_param_key", key[3])

            if src_idx in visible_idx_set:
                connected_outputs.setdefault(src_idx, set()).add(src_param_key)
            if dst_idx in visible_idx_set:
                connected_inputs.setdefault(dst_idx, set()).add(dst_param_key)

        # Include inherited inputs that can be satisfied by prior visible context,
        # even when they are not explicitly drawn in routing_lines.
        for dst_idx in self.visible_indices:
            dst_base = self.function_base_aliases[dst_idx]
            dst_inputs = self.gui_configs.get(dst_base, {}).get("input_aliases", {})
            for dst_param_key in dst_inputs.keys():
                if self._get_input_type(dst_base, dst_param_key) != "inherited":
                    continue

                if dst_param_key in connected_inputs.get(dst_idx, set()):
                    continue

                found_source = False
                for prev_idx in range(dst_idx - 1, -1, -1):
                    if prev_idx not in visible_idx_set:
                        continue
                    prev_base = self.function_base_aliases[prev_idx]
                    prev_config = self.gui_configs.get(prev_base, {})
                    prev_outputs = self._get_output_aliases_for_index(prev_idx)
                    prev_inputs = prev_config.get("input_aliases", {})

                    if dst_param_key in prev_outputs:
                        found_source = True
                    elif dst_param_key in prev_inputs and self._get_input_type(prev_base, dst_param_key) != "inherited":
                        found_source = True

                    if found_source:
                        connected_inputs.setdefault(dst_idx, set()).add(dst_param_key)
                        connected_outputs.setdefault(prev_idx, set()).add(dst_param_key)
                        break

        return connected_inputs, connected_outputs

    def _create_canvas(self):
        """Create the canvas with horizontal and vertical scrollbars and zoom/drag controls."""
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and controls frame
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(5, 5), padx=5)
        
        title = ttk.Label(title_frame, text="Complete Routing Map - All Functions and Connections")
        title.pack(side=tk.LEFT, padx=(0, 20))
        
        # Zoom controls
        ttk.Label(title_frame, text="Zoom:").pack(side=tk.LEFT, padx=(0, 5))
        self.zoom_label = ttk.Label(title_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(title_frame, text="−", command=self._zoom_out, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(title_frame, text="+", command=self._zoom_in, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(title_frame, text="Reset", command=self._zoom_reset).pack(side=tk.LEFT, padx=2)
        
        # Save button
        ttk.Button(title_frame, text="Save as Image", command=self._save_image).pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Label(title_frame, text="(Scroll to zoom, click & drag to pan)").pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)
        
        # Canvas frame with scrollbars
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Bind mousewheel for zooming
        def _on_mousewheel(event):
            # Zoom in/out based on wheel direction
            if event.delta > 0:
                self._zoom_in()
            else:
                self._zoom_out()
        
        # Bind for drag scrolling - use canvas scan methods for proper image-like panning
        def _on_canvas_press(event):
            self.canvas.scan_mark(event.x, event.y)
        
        def _on_canvas_drag(event):
            self.canvas.scan_dragto(event.x, event.y, gain=1)
        
        self.canvas.bind("<MouseWheel>", _on_mousewheel)
        self.canvas.bind("<Button-1>", _on_canvas_press)
        self.canvas.bind("<B1-Motion>", _on_canvas_drag)
        
        # Legend frame
        legend_frame = ttk.LabelFrame(main_frame, text="Legend", padding=5)
        legend_frame.pack(fill=tk.X, pady=(5, 5), padx=5)
        
        # Legend items
        legend_items = [
            ("Routed Connections", "#6F8370"),
            ("Inherited Variables", "#FF9800"),
            ("User-Inputted Variables", "#2196F3"),
        ]
        
        for label, color in legend_items:
            item_frame = ttk.Frame(legend_frame)
            item_frame.pack(side=tk.LEFT, padx=15)
            
            # Color box
            color_canvas = tk.Canvas(item_frame, width=20, height=20, bg=color, highlightthickness=1)
            color_canvas.pack(side=tk.LEFT, padx=(0, 5))
            
            # Label
            ttk.Label(item_frame, text=label).pack(side=tk.LEFT)

        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Checkbutton(
            footer_frame,
            text="Show connected inputs/outputs only",
            variable=self.show_connected_only,
            command=self._on_toggle_show_connected_only
        ).pack(side=tk.RIGHT)

    def _on_toggle_show_connected_only(self):
        """Redraw the map after toggling connected-only ports."""
        self._draw_routing_map()
    
    def _draw_routing_map(self):
        """Draw the complete routing map on a PIL image, then display on canvas."""
        self.canvas.delete("all")

        # Calculate layout
        spacing_x = 500
        node_width = 360
        base_start_x = 150
        start_y = 150  # Reserve more top space for block overlays and line lanes

        self.visible_indices = [
            idx for idx, base_alias in enumerate(self.function_base_aliases)
            if not self._is_hidden_visual_node(base_alias)
        ]
        if not self.visible_indices:
            self.visible_indices = list(range(len(self.methodology_list)))

        visible_idx_set = set(self.visible_indices)
        routed_connections, inherited_connections = self._build_connection_lists(visible_idx_set)
        connected_inputs, connected_outputs = self._collect_connected_ports(routed_connections, inherited_connections)

        inherited_source_fanout: Dict[int, int] = {}
        for conn in inherited_connections:
            src_idx = conn["src_idx"]
            inherited_source_fanout[src_idx] = inherited_source_fanout.get(src_idx, 0) + 1
        max_inherited_source_fanout = max(inherited_source_fanout.values()) if inherited_source_fanout else 0

        # Expand the left margin dynamically so inherited channels for dense nodes do not
        # clamp into a single trunk close to the map boundary.
        left_channel_margin = max(120, 50 + (max_inherited_source_fanout * 20))
        start_x = base_start_x + left_channel_margin

        skip_connection_count = sum(
            1
            for conn in routed_connections + inherited_connections
            if (conn["dst_idx"] - conn["src_idx"]) >= 1
        )
        destination_counts: Dict[int, int] = {}
        for conn in routed_connections + inherited_connections:
            destination_counts[conn["dst_idx"]] = destination_counts.get(conn["dst_idx"], 0) + 1
        max_destination_fanin = max(destination_counts.values()) if destination_counts else 0
        
        # Dictionary to store function boxes and input/output positions
        func_info = {}
        
        # First pass: calculate required heights for each function
        func_heights = {}
        for idx in self.visible_indices:
            base_alias = self.function_base_aliases[idx]
            func_config = self.gui_configs.get(base_alias, {})
            
            inputs = func_config.get("input_aliases", {})
            outputs = self._get_output_aliases_for_index(idx)

            if self.show_connected_only.get():
                visible_input_count = sum(
                    1
                    for key in inputs
                    if (
                        key in connected_inputs.get(idx, set())
                        or self._get_input_type(base_alias, key) == "inherited"
                    )
                )
                visible_output_count = sum(1 for key in outputs if key in connected_outputs.get(idx, set()))
            else:
                visible_input_count = len(inputs)
                visible_output_count = len(outputs)
            
            # Height needed: header (38) + items (22px each) + separators + padding
            max_items = max(visible_input_count, visible_output_count)
            required_height = 58 + (max_items * 22) + 16
            func_heights[idx] = required_height
        
        # Calculate canvas size needed
        visible_count = max(1, len(self.visible_indices))
        max_source_fanout = 0
        source_counts: Dict[int, int] = {}
        for conn in routed_connections + inherited_connections:
            source_counts[conn["src_idx"]] = source_counts.get(conn["src_idx"], 0) + 1
        if source_counts:
            max_source_fanout = max(source_counts.values())

        # Extra right margin scales with fan-out so heavily staggered channels do not clip.
        extra_right_margin = max(420, 220 + (max_source_fanout * 24))
        canvas_width = start_x + ((visible_count - 1) * spacing_x) + node_width + extra_right_margin
        max_func_height = max(func_heights.values()) if func_heights else 120
        # Reserve vertical space primarily from destination fan-in so dense targets
        # get enough compact lanes without global downward drift.
        extra_lane_space = max(120, max_destination_fanin * 22)
        canvas_height = start_y + max_func_height + 220 + extra_lane_space
        
        # Create PIL image
        self.pil_image = Image.new("RGB", (canvas_width, canvas_height), color="white")
        self.draw = ImageDraw.Draw(self.pil_image)
        
        # Store draw object for connection drawing
        self.routing_draw = self.draw
        
        # Second pass: draw all functions and track their positions
        for display_idx, idx in enumerate(self.visible_indices):
            base_alias = self.function_base_aliases[idx]
            func_config = self.gui_configs.get(base_alias, {})
            display_name = func_config.get("display_name", base_alias)
            
            func_width = node_width
            func_height = func_heights[idx]
            
            # Position in horizontal layout
            x = start_x + (display_idx * spacing_x)
            y = start_y
            
            # Draw function box and get input/output positions
            input_positions, output_positions = self._draw_function_box_pil(
                idx,
                x,
                y,
                func_width,
                func_height,
                display_name,
                func_config,
                connected_inputs,
                connected_outputs,
            )
            
            # Store all position information
            func_info[idx] = {
                "x": x,
                "y": y,
                "width": func_width,
                "height": func_height,
                "input_positions": input_positions,
                "output_positions": output_positions
            }

        # Draw control overlays and execution order before data-routing lines.
        self.top_overlay_bottom = 0
        self._draw_control_blocks(func_info)
        
        # Draw connections (output to input)
        self._draw_all_connections(func_info, canvas_height, routed_connections, inherited_connections)
        
        # Draw user input indicators
        self._draw_user_inputs(func_info)
        
        # Convert PIL image to PhotoImage and display on canvas
        self.photo_image = ImageTk.PhotoImage(self.pil_image)
        self.canvas.create_image(0, 0, image=self.photo_image, anchor="nw")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _get_separator_x(self, idx: int, func_info: Dict[int, Dict]) -> Optional[int]:
        """Return an x coordinate for a hidden separator alias based on neighboring visible nodes."""
        if idx in func_info:
            info = func_info[idx]
            return int(info["x"] + (info["width"] // 2))

        prev_idx = None
        next_idx = None
        for candidate in reversed(self.visible_indices):
            if candidate < idx:
                prev_idx = candidate
                break
        for candidate in self.visible_indices:
            if candidate > idx:
                next_idx = candidate
                break

        if prev_idx is not None and next_idx is not None:
            prev_info = func_info.get(prev_idx)
            next_info = func_info.get(next_idx)
            if prev_info and next_info:
                left_edge = prev_info["x"] + prev_info["width"]
                right_edge = next_info["x"]
                return int((left_edge + right_edge) / 2)
        elif prev_idx is not None:
            prev_info = func_info.get(prev_idx)
            if prev_info:
                return int(prev_info["x"] + prev_info["width"] + 20)
        elif next_idx is not None:
            next_info = func_info.get(next_idx)
            if next_info:
                return int(next_info["x"] - 20)

        return None

    def _find_matching_control_end(self, start_idx: int, start_alias: str, end_alias: str) -> int:
        """Find the matching control end node index for a given start index."""
        depth = 1
        for idx in range(start_idx + 1, len(self.methodology_list)):
            alias = self.function_base_aliases[idx]
            if alias == start_alias:
                depth += 1
            elif alias == end_alias:
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    def _collect_top_level_separators(self, start_idx: int, end_idx: int, separator_alias: str,
                                      start_alias: str, end_alias: str) -> List[int]:
        """Collect top-level separator markers inside a control block."""
        separators = []
        depth = 0
        for idx in range(start_idx + 1, end_idx):
            alias = self.function_base_aliases[idx]
            if alias == start_alias:
                depth += 1
            elif alias == end_alias and depth > 0:
                depth -= 1
            elif alias == separator_alias and depth == 0:
                separators.append(idx)
        return separators

    def _draw_execution_flow(self, func_info: Dict[int, Dict]):
        """Draw light control-flow arrows to show execution order left to right."""
        for pos in range(len(self.visible_indices) - 1):
            src_idx = self.visible_indices[pos]
            dst_idx = self.visible_indices[pos + 1]
            src = func_info.get(src_idx)
            dst = func_info.get(dst_idx)
            if not src or not dst:
                continue

            src_x = src["x"] + src["width"]
            src_y = src["y"] + (src["height"] // 2)
            dst_x = dst["x"]
            dst_y = dst["y"] + (dst["height"] // 2)

            mid_x = src_x + max(22, (dst_x - src_x) // 2)
            points = [
                (int(src_x), int(src_y)),
                (int(mid_x), int(src_y)),
                (int(mid_x), int(dst_y)),
                (int(dst_x), int(dst_y)),
            ]
            self._draw_curve_pil(points, "#9CA3AF")
            self._draw_arrow_pil(dst_x, dst_y, "right", "#9CA3AF")

    def _draw_control_blocks(self, func_info: Dict[int, Dict]):
        """Draw block-level annotations for loop, parallel, and ensemble controls."""
        block_defs = [
            ("workflow_loop_start", "workflow_loop_end", "#3b82f6", "Loop Block", None),
            ("workflow_parallel_start", "workflow_parallel_end", "#8b5cf6", "Parallel Block", "workflow_parallel_branch"),
            ("workflow_ensemble_start", "workflow_ensemble_end", "#059669", "Ensemble Block", "workflow_ensemble_member"),
        ]

        blocks = []
        for idx, alias in enumerate(self.function_base_aliases):
            for start_alias, end_alias, color, label, separator_alias in block_defs:
                if alias != start_alias:
                    continue
                end_idx = self._find_matching_control_end(idx, start_alias, end_alias)
                if end_idx < 0:
                    continue
                separators = []
                if separator_alias:
                    separators = self._collect_top_level_separators(
                        idx, end_idx, separator_alias, start_alias, end_alias
                    )
                blocks.append({
                    "start": idx,
                    "end": end_idx,
                    "color": color,
                    "label": label,
                    "separators": separators,
                })

        if not blocks:
            return

        blocks.sort(key=lambda b: ((b["end"] - b["start"]), b["start"]))
        lanes: List[List[Tuple[int, int]]] = []
        for block in blocks:
            lane_idx = 0
            while lane_idx < len(lanes):
                overlap = False
                for lane_start, lane_end in lanes[lane_idx]:
                    if not (block["end"] < lane_start or block["start"] > lane_end):
                        overlap = True
                        break
                if not overlap:
                    break
                lane_idx += 1
            if lane_idx == len(lanes):
                lanes.append([])
            lanes[lane_idx].append((block["start"], block["end"]))
            block["lane"] = lane_idx

        base_top = 32
        lane_step = 22
        max_overlay_bottom = 0
        for block in blocks:
            start_info = func_info.get(block["start"])
            block_visible_end = None
            for candidate in reversed(self.visible_indices):
                if block["start"] <= candidate <= block["end"]:
                    block_visible_end = candidate
                    break
            if block_visible_end is None:
                continue
            end_info = func_info.get(block_visible_end)
            if not start_info or not end_info:
                continue

            x1 = start_info["x"] - 16
            x2 = end_info["x"] + end_info["width"] + 16
            y = base_top + block.get("lane", 0) * lane_step
            color = block["color"]

            self.draw.line([(x1, y), (x2, y)], fill=color, width=3)
            self.draw.line([(x1, y), (x1, y + 16)], fill=color, width=3)
            self.draw.line([(x2, y), (x2, y + 16)], fill=color, width=3)
            max_overlay_bottom = max(max_overlay_bottom, y + 16)

            for sep_idx in block["separators"]:
                sep_x = self._get_separator_x(sep_idx, func_info)
                if sep_x is None:
                    continue
                self.draw.line([(sep_x, y), (sep_x, y + 12)], fill=color, width=2)

            label_text = block["label"]
            if block["separators"]:
                if "Parallel" in label_text:
                    label_text = f"{label_text} ({len(block['separators']) + 1} branches)"
                elif "Ensemble" in label_text:
                    label_text = f"{label_text} ({len(block['separators']) + 1} members)"

            try:
                label_font = ImageFont.truetype("arial.ttf", 13)
            except Exception:
                label_font = ImageFont.load_default()
            self.draw.text((x1 + 4, y - 6), label_text, fill=color, font=label_font, anchor="lb")

        self.top_overlay_bottom = max_overlay_bottom
    
    def _draw_user_inputs(self, func_info: Dict[int, Dict]):
        """Draw blue arrows for user-inputted variables on PIL image."""
        for idx, info in func_info.items():
            base_alias = self.function_base_aliases[idx]
            func_config = self.gui_configs.get(base_alias, {})
            inputs = func_config.get("input_aliases", {})
            
            for param_key, param_name in inputs.items():
                # Pass internal name (param_key), not display name
                input_type = self._get_input_type(base_alias, param_key)
                
                if input_type == "user":
                    # Draw arrow to this user input
                    if param_key in info["input_positions"]:
                        x, y = info["input_positions"][param_key]
                        
                        # Draw longer arrow pointing to the input
                        arrow_start_x = x - 35
                        self.draw.line(
                            [(arrow_start_x, y), (x - 5, y)],
                            fill="#2196F3",
                            width=3
                        )
                        
                        # Arrow head
                        self.draw.polygon(
                            [(x, y), (x - 5, y - 3), (x - 5, y + 3)],
                            fill="#2196F3",
                            outline="#2196F3"
                        )
    
    
    
    def _draw_function_box_pil(self, idx: int, x: int, y: int, width: int, height: int,
                               display_name: str, func_config: Dict[str, Any],
                               connected_inputs: Dict[int, set],
                               connected_outputs: Dict[int, set]) -> Tuple[Dict, Dict]:
        """
        Draw a function box with inputs and outputs on PIL image.
        
        Returns:
            Tuple of (input_positions, output_positions) dicts mapping param_key to (x, y)
        """
        # Resolve the function alias first so control-style lookup is always safe.
        base_alias = self.function_base_aliases[idx]
        control_style = self.CONTROL_NODE_STYLES.get(base_alias)
        body_fill = control_style["body_bg"] if control_style else "#f5f5f5"
        title_fill = control_style["title_bg"] if control_style else "#d8d8d8"
        outline_color = control_style["outline"] if control_style else "#555555"

        # Draw outer box with rounded appearance
        self.draw.rectangle(
            [(x, y), (x + width, y + height)],
            fill=body_fill,
            outline=outline_color,
            width=2
        )
        
        # Draw title header background
        self.draw.rectangle(
            [(x + 2, y + 2), (x + width - 2, y + 36)],
            fill=title_fill,
            outline=outline_color,
            width=1
        )
        
        # Draw title (centered)
        try:
            title_font = ImageFont.truetype("arial.ttf", 16)
        except:
            title_font = ImageFont.load_default()

        def _fit_label(text: str, max_width: int) -> str:
            if not text:
                return ""
            fitted = str(text)
            bbox = self.draw.textbbox((0, 0), fitted, font=item_font)
            if (bbox[2] - bbox[0]) <= max_width:
                return fitted

            suffix = "..."
            while len(fitted) > 1:
                fitted = fitted[:-1]
                candidate = f"{fitted}{suffix}"
                bbox = self.draw.textbbox((0, 0), candidate, font=item_font)
                if (bbox[2] - bbox[0]) <= max_width:
                    return candidate
            return suffix
        
        control_label = control_style.get("label") if control_style else None
        title_text = f"[{idx}] {control_label or display_name}"
        self.draw.text(
            (x + width // 2, y + 19),
            title_text,
            fill="#000000",
            font=title_font,
            anchor="mm"
        )
        
        # Inputs section (left side)
        input_positions = {}
        inputs = func_config.get("input_aliases", {})
        if self.show_connected_only.get():
            visible_inputs = [
                (k, v)
                for k, v in inputs.items()
                if (
                    k in connected_inputs.get(idx, set())
                    or self._get_input_type(base_alias, k) == "inherited"
                )
            ]
        else:
            visible_inputs = list(inputs.items())
        input_y = y + 45
        
        try:
            label_font = ImageFont.truetype("arial.ttf", 14)
            item_font = ImageFont.truetype("arial.ttf", 12)
        except:
            label_font = ImageFont.load_default()
            item_font = ImageFont.load_default()
        
        if visible_inputs:
            self.draw.text(
                (x + 10, input_y),
                "Inputs:",
                fill="#1f6aa5",
                font=label_font,
                anchor="lm"
            )
            input_y += 18
            
            for param_key, param_name in visible_inputs:
                # Connection point is at the left edge
                connection_x = x
                connection_y = input_y + 3
                
                # Store position for connection drawing
                input_positions[param_key] = (connection_x, connection_y)
                
                # Determine input type
                input_type = self._get_input_type(base_alias, param_key)
                
                # Draw colored dot based on input type
                if input_type == "user":
                    dot_color = "#2196F3"  # Blue for user input
                elif input_type == "inherited":
                    dot_color = "#FF9800"  # Orange for inherited
                else:
                    dot_color = "#1f6aa5"  # Blue for routed
                
                self.draw.ellipse(
                    [(connection_x - 4, connection_y - 4),
                     (connection_x + 4, connection_y + 4)],
                    fill=dot_color,
                    outline=dot_color
                )
                
                self.draw.text(
                    (x + 15, input_y),
                    f"• {_fit_label(param_name, 120)}",
                    fill="#222222",
                    font=item_font,
                    anchor="lm"
                )
                input_y += 22
        
        # Outputs section (right side)
        output_positions = {}
        outputs = self._get_output_aliases_for_index(idx)
        if self.show_connected_only.get():
            visible_outputs = [(k, v) for k, v in outputs.items() if k in connected_outputs.get(idx, set())]
        else:
            visible_outputs = list(outputs.items())
        output_y = y + 45
        
        if visible_outputs:
            self.draw.text(
                (x + width - 10, output_y),
                "Outputs:",
                fill="#dc2626",
                font=label_font,
                anchor="rm"
            )
            output_y += 18
            
            for param_key, param_name in visible_outputs:
                # Connection point is at the right edge
                connection_x = x + width
                connection_y = output_y + 3
                
                # Store position for connection drawing
                output_positions[param_key] = (connection_x, connection_y)
                
                # Draw indicator dot
                self.draw.ellipse(
                    [(connection_x - 4, connection_y - 4),
                     (connection_x + 4, connection_y + 4)],
                    fill="#dc2626",
                    outline="#dc2626"
                )
                
                self.draw.text(
                    (x + width - 15, output_y),
                    f"{_fit_label(param_name, 120)} •",
                    fill="#222222",
                    font=item_font,
                    anchor="rm"
                )
                output_y += 22
        
        return input_positions, output_positions
    
    def _draw_all_connections(self, func_info: Dict[int, Dict], canvas_height: int,
                              routed_connections: List[Dict[str, Any]],
                              inherited_connections: List[Dict[str, Any]]):
        """Draw all routing connections with smart routing to avoid crossing."""
        lane_usage: Dict[Tuple[int, int, bool], int] = {}

        # Draw routed first, inherited second.
        self._draw_connection_group(routed_connections, func_info, "#6F8370", False, lane_usage, canvas_height)
        self._draw_connection_group(inherited_connections, func_info, "#FF9800", True, lane_usage, canvas_height)

    def _allocate_lane(self, lane_usage: Dict[Tuple[int, int, bool], int],
                       src_idx: int, dst_idx: int, is_inherited: bool) -> int:
        """Allocate local lane index per span/connection type to avoid global drift."""
        lane_key = (src_idx, dst_idx, is_inherited)
        lane = lane_usage.get(lane_key, 0)
        lane_usage[lane_key] = lane + 1
        return lane
    
    def _draw_connection_group(self, connections: List[Dict[str, Any]], func_info: Dict[int, Dict],
                               color: str, is_inherited: bool,
                               lane_usage: Dict[Tuple[int, int, bool], int], canvas_height: int):
        """Draw a group of connections with intelligent routing."""
        # Use per-source-node channel allocation so different variables never collapse
        # into the same trunk immediately after leaving a function box.
        source_departure_counts: Dict[Tuple[int, str], int] = {}
        destination_approach_counts: Dict[Tuple[int, str], int] = {}

        ordered_connections = sorted(
            connections,
            key=lambda c: (
                c["src_idx"],
                c["dst_idx"],
                str(c["src_param_key"]),
                str(c["dst_param_key"]),
            ),
        )

        for conn in ordered_connections:
            src_idx = conn["src_idx"]
            dst_idx = conn["dst_idx"]
            src_param_key = conn["src_param_key"]
            dst_param_key = conn["dst_param_key"]

            if src_idx not in func_info or dst_idx not in func_info:
                continue

            src_info = func_info[src_idx]
            dst_info = func_info[dst_idx]

            src_x, src_y = None, None
            dst_x, dst_y = None, None

            if is_inherited:
                if src_param_key in src_info.get("input_positions", {}):
                    src_x, src_y = src_info["input_positions"][src_param_key]
                else:
                    continue
            else:
                if src_param_key in src_info.get("output_positions", {}):
                    src_x, src_y = src_info["output_positions"][src_param_key]
                else:
                    continue

            if dst_param_key in dst_info.get("input_positions", {}):
                dst_x, dst_y = dst_info["input_positions"][dst_param_key]
            else:
                continue

            skip_count = dst_idx - src_idx - 1
            # Allocate departure slot by source node and side, not by parameter.
            # This guarantees distinct "lanes" even when multiple different outputs
            # from the same function travel through the same region.
            source_key = (src_idx, "input" if is_inherited else "output")
            # Allocate approach channels per destination node side so different
            # variables headed to the same function do not collapse together.
            destination_key = (dst_idx, "input")
            departure_slot = source_departure_counts.get(source_key, 0)
            approach_slot = destination_approach_counts.get(destination_key, 0)
            source_departure_counts[source_key] = departure_slot + 1
            destination_approach_counts[destination_key] = approach_slot + 1

            if not is_inherited and skip_count == 0:
                self._draw_direct_connection(
                    src_x,
                    src_y,
                    dst_x,
                    dst_y,
                    color,
                )
                continue

            lane = self._allocate_lane(lane_usage, src_idx, dst_idx, is_inherited)
            # Keep skip lanes below in compact, monotonic bands for readability.
            route_above = False
            lane_tier = lane

            traversed = [
                idx for idx in self.visible_indices
                if src_idx <= idx <= dst_idx and idx in func_info
            ]
            if not traversed:
                continue

            min_y = min(func_info[idx]["y"] for idx in traversed)
            max_y = max(func_info[idx]["y"] + func_info[idx]["height"] for idx in traversed)
            lane_gap = 12 if not is_inherited else 14
            top_route_floor = max(12, self.top_overlay_bottom + 18)
            if route_above:
                route_y = max(top_route_floor, min_y - 24 - (lane_tier * lane_gap))
            else:
                # Keep inherited and routed paths in different below-bands so they do not share lanes.
                base_below_offset = 18 if not is_inherited else 40
                route_y = min(canvas_height - 16, max_y + base_below_offset + (lane_tier * lane_gap))

            if is_inherited:
                left_x = int(src_x - (38 + departure_slot * 18))
                approach_x = int(dst_x - (24 + approach_slot * 18))
                max_x = self.pil_image.width - 20
                left_x = max(6, min(left_x, min(max_x, int(src_x) - 8)))
                approach_x = max(6, min(approach_x, min(max_x, int(dst_x) - 8)))
                points = [
                    (int(src_x), int(src_y)),
                    (left_x, int(src_y)),
                    (left_x, int(route_y)),
                    (approach_x, int(route_y)),
                    (approach_x, int(dst_y)),
                    (int(dst_x), int(dst_y)),
                ]
            else:
                launch_x = int(src_x + 20 + departure_slot * 18)
                approach_x = int(dst_x - (22 + approach_slot * 18))
                max_x = self.pil_image.width - 20
                launch_x = max(20, min(launch_x, max_x))
                approach_x = max(20, min(approach_x, max_x))
                points = [
                    (int(src_x), int(src_y)),
                    (launch_x, int(src_y)),
                    (launch_x, int(route_y)),
                    (approach_x, int(route_y)),
                    (approach_x, int(dst_y)),
                    (int(dst_x), int(dst_y)),
                ]

            self._draw_curve_pil(points, color)
            self._draw_arrow_pil(dst_x, dst_y, "right", color)
    
    def _draw_direct_connection(self, src_x: int, src_y: int, dst_x: int, dst_y: int, color: str):
        """Draw a direct connection between adjacent functions on PIL image."""
        sx, sy = int(src_x), int(src_y)
        dx, dy = int(dst_x), int(dst_y)

        # If ports are aligned, keep the route perfectly straight.
        if sy == dy:
            self.draw.line([(sx, sy), (dx, dy)], fill=color, width=2)
            self._draw_arrow_pil(dst_x, dst_y, "right", color)
            return

        # Otherwise use a single, clean right-angle elbow between adjacent nodes.
        mid_x = int((sx + dx) / 2)
        points = [
            (sx, sy),
            (mid_x, sy),
            (mid_x, dy),
            (dx, dy),
        ]
        self.draw.line(points, fill=color, width=2)
        self._draw_arrow_pil(dst_x, dst_y, "right", color)
    
    def _draw_arrow_pil(self, x: int, y: int, direction: str, color: str):
        """Draw an arrow head at the given position on PIL image."""
        x = int(x)
        y = int(y)
        arrow_size = 6
        if direction == "right":
            self.draw.polygon(
                [(x, y),
                 (x - arrow_size, y - arrow_size // 2),
                 (x - arrow_size, y + arrow_size // 2)],
                fill=color,
                outline=color
            )
        else:  # left
            self.draw.polygon(
                [(x, y),
                 (x + arrow_size, y - arrow_size // 2),
                 (x + arrow_size, y + arrow_size // 2)],
                fill=color,
                outline=color
            )
    
    def _draw_skipping_connection(self, src_x: int, src_y: int, dst_x: int, dst_y: int,
                                  route_y: int, color: str):
        """Draw a connection that routes around intermediate functions on PIL image."""
        # Curve around from left side
        points = [(src_x, src_y), (src_x + 30, src_y), (src_x + 30, route_y),
                 (dst_x - 30, route_y), (dst_x - 30, dst_y), (dst_x, dst_y)]
        self._draw_curve_pil(points, color)
        self._draw_arrow_pil(dst_x, dst_y, "right", color)
    
    def _draw_curve_pil(self, points: List[Tuple[int, int]], color: str):
        """Draw orthogonal-only routing through the provided control points."""
        if len(points) < 2:
            return

        clean_points: List[Tuple[int, int]] = []
        for px, py in points:
            point = (int(px), int(py))
            if not clean_points or point != clean_points[-1]:
                clean_points.append(point)

        orth_points: List[Tuple[int, int]] = []
        for idx, current in enumerate(clean_points):
            if idx == 0:
                orth_points.append(current)
                continue

            prev_x, prev_y = orth_points[-1]
            cur_x, cur_y = current
            if prev_x != cur_x and prev_y != cur_y:
                # Always insert a right-angle corner, horizontal then vertical.
                orth_points.append((cur_x, prev_y))
            orth_points.append((cur_x, cur_y))

        dedup_points: List[Tuple[int, int]] = []
        for point in orth_points:
            if not dedup_points or point != dedup_points[-1]:
                dedup_points.append(point)

        if len(dedup_points) > 1:
            self.draw.line(dedup_points, fill=color, width=2)
    
    def _zoom_in(self):
        """Zoom in on the canvas."""
        new_zoom = min(self.max_zoom, self.zoom_level * self.zoom_step)
        if abs(new_zoom - self.zoom_level) > 1e-9:
            self.zoom_level = new_zoom
            self._apply_zoom()
    
    def _zoom_out(self):
        """Zoom out on the canvas."""
        new_zoom = max(self.min_zoom, self.zoom_level / self.zoom_step)
        if abs(new_zoom - self.zoom_level) > 1e-9:
            self.zoom_level = new_zoom
            self._apply_zoom()
    
    def _zoom_reset(self):
        """Reset zoom to 100%."""
        self.zoom_level = 1.0
        self._apply_zoom()
    
    def _apply_zoom(self):
        """Apply the current zoom level by scaling the PIL image."""
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        
        # Scale the PIL image
        new_width = int(self.pil_image.width * self.zoom_level)
        new_height = int(self.pil_image.height * self.zoom_level)
        
        # Use high-quality resampling
        zoomed_image = self.pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(zoomed_image)
        
        # Clear canvas and redraw
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo_image, anchor="nw")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _save_image(self):
        """Open a file dialog and save the routing map as an image."""
        # Open save file dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("BMP Image", "*.bmp"),
                ("All Files", "*.*")
            ],
            title="Save Routing Map",
            initialfile="routing_map.png"
        )
        
        # If user cancelled, return
        if not file_path:
            return
        
        try:
            # Save the original PIL image (not the zoomed version)
            self.pil_image.save(file_path)
            messagebox.showinfo(
                "Save Successful",
                f"Routing map saved successfully to:\n{file_path}",
                parent=self.window
            )
        except Exception as e:
            messagebox.showerror(
                "Save Failed",
                f"Failed to save image:\n{str(e)}",
                parent=self.window
            )
