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
    
    def __init__(self, parent, methodology_list: List[str], function_base_aliases: List[str],
                 routing_lines: Dict[Tuple, Dict], gui_configs: Dict[str, Dict], 
                 function_specs: Dict[str, Dict]):
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
        self.min_zoom = 0.5
        self.max_zoom = 3.0
        self.zoom_step = 1.15  # 15% zoom increment
        
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
    
    def _draw_routing_map(self):
        """Draw the complete routing map on a PIL image, then display on canvas."""
        # Calculate layout
        func_height_base = 50
        spacing_x = 380
        spacing_y = 250
        start_x = 150  # Extra space on left for inherited arrows to route through
        start_y = 100  # Space on top for above-routing of arrows
        
        # Dictionary to store function boxes and input/output positions
        func_info = {}
        
        # First pass: calculate required heights for each function
        func_heights = {}
        for idx, instance_alias in enumerate(self.methodology_list):
            base_alias = self.function_base_aliases[idx]
            func_config = self.gui_configs.get(base_alias, {})
            
            inputs = func_config.get("input_aliases", {})
            outputs = func_config.get("output_aliases", {})
            
            # Height needed: header (38) + items (18px each) + separators + padding
            max_items = max(len(inputs), len(outputs))
            required_height = 55 + (max_items * 18) + 15
            func_heights[idx] = required_height
        
        # Calculate canvas size needed
        max_idx = len(self.methodology_list) - 1
        canvas_width = start_x + (max_idx * spacing_x) + 350
        canvas_height = start_y + max(func_heights.values()) + 250
        
        # Create PIL image
        self.pil_image = Image.new("RGB", (canvas_width, canvas_height), color="white")
        self.draw = ImageDraw.Draw(self.pil_image)
        
        # Store draw object for connection drawing
        self.routing_draw = self.draw
        
        # Second pass: draw all functions and track their positions
        for idx, instance_alias in enumerate(self.methodology_list):
            base_alias = self.function_base_aliases[idx]
            func_config = self.gui_configs.get(base_alias, {})
            display_name = func_config.get("display_name", base_alias)
            
            func_width = 260
            func_height = func_heights[idx]
            
            # Position in horizontal layout
            x = start_x + (idx * spacing_x)
            y = start_y
            
            # Draw function box and get input/output positions
            input_positions, output_positions = self._draw_function_box_pil(
                idx, x, y, func_width, func_height, display_name, func_config
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
        
        # Draw connections (output to input)
        self._draw_all_connections(func_info, canvas_height)
        
        # Draw user input indicators
        self._draw_user_inputs(func_info)
        
        # Convert PIL image to PhotoImage and display on canvas
        self.photo_image = ImageTk.PhotoImage(self.pil_image)
        self.canvas.create_image(0, 0, image=self.photo_image, anchor="nw")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
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
                          display_name: str, func_config: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """
        Draw a function box with inputs and outputs on PIL image.
        
        Returns:
            Tuple of (input_positions, output_positions) dicts mapping param_key to (x, y)
        """
        # Draw outer box with rounded appearance
        self.draw.rectangle(
            [(x, y), (x + width, y + height)],
            fill="#f5f5f5",
            outline="#555555",
            width=2
        )
        
        # Draw title header background
        self.draw.rectangle(
            [(x + 2, y + 2), (x + width - 2, y + 36)],
            fill="#d8d8d8",
            outline="#999999",
            width=1
        )
        
        # Draw title (centered)
        try:
            title_font = ImageFont.truetype("arial.ttf", 16)
        except:
            title_font = ImageFont.load_default()
        
        title_text = f"[{idx}] {display_name}"
        self.draw.text(
            (x + width // 2, y + 19),
            title_text,
            fill="#000000",
            font=title_font,
            anchor="mm"
        )
        
        # Get base_alias from the index
        base_alias = self.function_base_aliases[idx]
        
        # Inputs section (left side)
        input_positions = {}
        inputs = func_config.get("input_aliases", {})
        input_y = y + 45
        
        try:
            label_font = ImageFont.truetype("arial.ttf", 14)
            item_font = ImageFont.truetype("arial.ttf", 13)
        except:
            label_font = ImageFont.load_default()
            item_font = ImageFont.load_default()
        
        if inputs:
            self.draw.text(
                (x + 10, input_y),
                "Inputs:",
                fill="#1f6aa5",
                font=label_font,
                anchor="lm"
            )
            input_y += 16
            
            for param_key, param_name in inputs.items():
                # Connection point is at the left edge
                connection_x = x
                connection_y = input_y + 3
                
                # Store position for connection drawing
                input_positions[param_key] = (connection_x, connection_y)
                
                # Determine input type
                input_type = self._get_input_type(base_alias, param_name)
                
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
                    f"• {param_name}",
                    fill="#222222",
                    font=item_font,
                    anchor="lm"
                )
                input_y += 18
        
        # Outputs section (right side)
        output_positions = {}
        outputs = func_config.get("output_aliases", {})
        output_y = y + 45
        
        if outputs:
            self.draw.text(
                (x + width - 10, output_y),
                "Outputs:",
                fill="#dc2626",
                font=label_font,
                anchor="rm"
            )
            output_y += 16
            
            for param_key, param_name in outputs.items():
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
                    f"{param_name} •",
                    fill="#222222",
                    font=item_font,
                    anchor="rm"
                )
                output_y += 18
        
        return input_positions, output_positions
    
    def _draw_all_connections(self, func_info: Dict[int, Dict], canvas_height: int):
        """Draw all routing connections with smart routing to avoid crossing."""
        # Separate connections by type and build full routing info
        routed_connections = []
        inherited_connections = []
        
        # First pass: build connection info from routing_lines (explicit routings)
        for key, conn_info in self.routing_lines.items():
            src_idx = conn_info.get("src_idx", key[0])
            src_param_key = conn_info.get("src_param_key", key[1])
            dst_idx = conn_info.get("dst_idx", key[2])
            dst_param_key = conn_info.get("dst_param_key", key[3])
            
            # Check if destination input is inherited
            dst_base = self.function_base_aliases[dst_idx]
            dst_config = self.gui_configs.get(dst_base, {})
            dst_inputs = dst_config.get("input_aliases", {})
            dst_param_name = dst_inputs.get(dst_param_key, "")
            
            is_inherited_input = False
            for field_key, field_info in dst_config.get("fields", {}).items():
                if field_info.get("name") == dst_param_name:
                    if field_info.get("input_type") == "inherited":
                        is_inherited_input = True
                        break
            
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
        
        # Second pass: find inherited variables that aren't in routing_lines
        # These come from previous functions or defaults
        for dst_idx in range(len(self.methodology_list)):
            dst_base = self.function_base_aliases[dst_idx]
            dst_config = self.gui_configs.get(dst_base, {})
            dst_inputs = dst_config.get("input_aliases", {})
            
            for dst_param_key, dst_param_name in dst_inputs.items():
                # Check if this input is inherited
                # IMPORTANT: Pass the internal name (dst_param_key), not the display name
                input_type = self._get_input_type(dst_base, dst_param_key)
                
                if input_type != "inherited":
                    continue
                
                # Check if this is already in routing_lines
                already_routed = False
                for key, conn_info in self.routing_lines.items():
                    if (conn_info.get("dst_idx", key[2]) == dst_idx and 
                        conn_info.get("dst_param_key", key[3]) == dst_param_key):
                        already_routed = True
                        break
                
                if already_routed:
                    continue
                
                # Find the source - for inherited inputs, find the closest previous function
                # that HAS this variable. Prioritize outputs (where it comes from), 
                # then non-inherited inputs as fallback
                src_idx = None
                src_param_key = None
                
                # First pass: Look for the CLOSEST previous function with this as an OUTPUT
                for prev_idx in range(dst_idx - 1, -1, -1):
                    prev_base = self.function_base_aliases[prev_idx]
                    prev_config = self.gui_configs.get(prev_base, {})
                    prev_outputs = prev_config.get("output_aliases", {})
                    
                    # Check if dst_param_key exists in the previous function's outputs
                    if dst_param_key in prev_outputs:
                        src_idx = prev_idx
                        src_param_key = dst_param_key
                        break
                
                # If not found in outputs, look for non-inherited inputs
                if src_idx is None:
                    for prev_idx in range(dst_idx - 1, -1, -1):
                        prev_base = self.function_base_aliases[prev_idx]
                        prev_config = self.gui_configs.get(prev_base, {})
                        prev_inputs = prev_config.get("input_aliases", {})
                        
                        # Check if this input exists and is NOT inherited
                        if dst_param_key in prev_inputs:
                            # Check if this input is inherited
                            prev_input_type = self._get_input_type(prev_base, dst_param_key)
                            if prev_input_type != "inherited":
                                # This is a user-inputted or other non-inherited input
                                src_idx = prev_idx
                                src_param_key = dst_param_key
                                break
                
                # Add as inherited connection if source found
                if src_idx is not None:
                    # Check if this src->dst pair already exists in inherited connections
                    already_exists = False
                    for existing_conn in inherited_connections:
                        if existing_conn["src_idx"] == src_idx and existing_conn["dst_idx"] == dst_idx and existing_conn["dst_param_key"] == dst_param_key:
                            already_exists = True
                            break
                    
                    if not already_exists:
                        inherited_connections.append({
                            "src_idx": src_idx,
                            "src_param_key": src_param_key,
                            "dst_idx": dst_idx,
                            "dst_param_key": dst_param_key,
                            "dst_param_name": dst_param_name,
                            "is_inherited": True,
                        })
        
        # Create shared routing tracking that groups by destination vertical region
        # All arrows going around boxes near the same destination coordinate their spacing
        route_vertical_tracks = {}  # {(vertical_region,): [(above_positions), (below_positions)]}
        
        # Draw routed connections first (green, may be below)
        self._draw_connection_group(routed_connections, func_info, "#6F8370", False, route_vertical_tracks, canvas_height)
        
        # Draw inherited connections on top (orange, routing from left)
        self._draw_connection_group(inherited_connections, func_info, "#FF9800", True, route_vertical_tracks, canvas_height)
    
    def _draw_connection_group(self, connections: List[Dict], func_info: Dict[int, Dict], 
                               color: str, is_inherited: bool, route_vertical_tracks: Dict, canvas_height: int):
        """Draw a group of connections with intelligent routing."""
        # Note: route_vertical_tracks tracks all arrows going around same destination region
        # This ensures inherited and routed arrows don't overlap
        
        # Track source departure points to fan out multiple arrows from same input
        source_departure_counts = {}  # {(src_idx, src_param_key): count}
        
        for conn in connections:
            src_idx = conn["src_idx"]
            dst_idx = conn["dst_idx"]
            src_param_key = conn["src_param_key"]
            dst_param_key = conn["dst_param_key"]
            
            # Check indices are valid
            if src_idx not in func_info or dst_idx not in func_info:
                continue
            
            src_info = func_info[src_idx]
            dst_info = func_info[dst_idx]
            
            # Get connection points
            src_x, src_y = None, None
            dst_x, dst_y = None, None
            
            # For inherited connections, source is from input position
            if is_inherited:
                if src_param_key in src_info.get("input_positions", {}):
                    src_x, src_y = src_info["input_positions"][src_param_key]
                else:
                    continue
            else:
                # For routed connections, source is from output position
                if src_param_key in src_info.get("output_positions", {}):
                    src_x, src_y = src_info["output_positions"][src_param_key]
                else:
                    continue
            
            # Destination is always an input
            if dst_param_key in dst_info.get("input_positions", {}):
                dst_x, dst_y = dst_info["input_positions"][dst_param_key]
            else:
                continue
            
            # Check if connection skips functions
            skip_count = dst_idx - src_idx - 1
            
            # Track how many times this source point is being used for departure
            source_key = (src_idx, src_param_key)
            if source_key not in source_departure_counts:
                source_departure_counts[source_key] = 0
            departure_offset_idx = source_departure_counts[source_key]
            source_departure_counts[source_key] += 1
            
            # For inherited connections, always route around (above/below) boxes
            # For routed connections, only route around if skipping functions
            if not is_inherited and skip_count == 0:
                # Direct connection to adjacent function (only for routed)
                self._draw_direct_connection(src_x, src_y, dst_x, dst_y, color)
            else:
                # Route around boxes - for all inherited, or routed that skip functions
                all_indices = list(range(src_idx, dst_idx + 1))
                min_y = min(func_info[i]["y"] for i in all_indices)
                max_y = max(func_info[i]["y"] + func_info[i]["height"] for i in all_indices)
                
                # Use the range of boxes being skipped as the key for routing tracking
                # This ensures all arrows routing around the same boxes coordinate vertically
                route_key = (src_idx, dst_idx)  # The specific route pair
                
                if route_key not in route_vertical_tracks:
                    route_vertical_tracks[route_key] = {"above": [], "below": []}
                
                # Count ALL arrows that route through overlapping box ranges
                # not just exact matches, to prevent collisions between different routes
                overlapping_above = 0
                overlapping_below = 0
                for other_key, other_tracks in route_vertical_tracks.items():
                    other_src, other_dst = other_key
                    # Check if box ranges overlap
                    if not (other_dst < src_idx or other_src > dst_idx):  # Ranges overlap
                        overlapping_above += len(other_tracks["above"])
                        overlapping_below += len(other_tracks["below"])
                
                above_count = overlapping_above
                below_count = overlapping_below
                
                # Check if routing above would go off-screen
                potential_above_y = min_y - 60 - (above_count * 10)
                would_go_offscreen_above = potential_above_y < 30
                
                if above_count <= below_count and not would_go_offscreen_above:
                    # Route above - use consistent 15px spacing
                    route_y = potential_above_y
                    route_vertical_tracks[route_key]["above"].append((src_idx, dst_idx, is_inherited))
                else:
                    # Route below instead - use consistent 15px spacing
                    route_y = min(canvas_height - 50, max_y + 40 + (below_count * 10))
                    route_vertical_tracks[route_key]["below"].append((src_idx, dst_idx, is_inherited))
                
                # For inherited (input-to-input), go left first, then curve around
                # For routed (output-to-input), curve right then around
                if is_inherited:
                    # Go LEFT from input, curve around, come back RIGHT to input
                    # Space them out more to avoid mixing at the destination
                    # But don't go too far left - minimum of 20 pixels from edge
                    left_spacing = 20 + (above_count + below_count) * 10
                    left_x = max(20, int(src_x - left_spacing))
                    right_clearance = 10 + (above_count + below_count) * 10  # Use same 10px increment
                    
                    # Stagger the approach to destination - each arrow approaches from different horizontal position
                    approach_x_offset = departure_offset_idx * 10  # Use same 10px increment
                    approach_x = int(dst_x - right_clearance - approach_x_offset)
                    
                    points = [
                        (int(src_x), int(src_y)),  # All leave at same point
                        (left_x, int(src_y)),      # Go left together
                        (left_x, int(route_y)),    # Down/up to route height
                        (approach_x, int(route_y)),     # Staggered horizontal approach
                        (approach_x, int(dst_y)),       # Curve down to input
                        (int(dst_x), int(dst_y))
                    ]
                    self._draw_curve_pil(points, color)
                else:
                    # Routed: curve right then around boxes
                    # All arrows leave at the same point, stagger only at destination
                    control_x1 = int(src_x + 10)
                    control_x2 = int(dst_x - 10 - (above_count + below_count) * 10)
                    
                    # Add vertical offset at destination for arrows reaching same input
                    approach_y_offset = departure_offset_idx * 10
                    approach_y = int(dst_y) + approach_y_offset
                    
                    points = [
                        (int(src_x), int(src_y)),  # All leave at same point
                        (control_x1, int(src_y)),
                        (control_x1, int(route_y)),
                        (control_x2, int(route_y)),
                        (control_x2, approach_y),       # Staggered vertical approach
                        (int(dst_x), int(dst_y))
                    ]
                    self._draw_curve_pil(points, color)
                
                self._draw_arrow_pil(dst_x, dst_y, "right", color)
    
    def _draw_direct_connection(self, src_x: int, src_y: int, dst_x: int, dst_y: int, color: str):
        """Draw a direct connection between adjacent functions on PIL image."""
        # S-curve with proper offset
        horizontal_distance = abs(dst_x - src_x)
        curve_offset = min(100, horizontal_distance * 0.4)
        
        # Create control points for the curve
        points = [
            (int(src_x), int(src_y)),
            (int(src_x + curve_offset * 0.6), int(src_y)),
            (int(dst_x - curve_offset * 0.6), int(dst_y)),
            (int(dst_x), int(dst_y))
        ]
        
        # Draw smooth curve
        self._draw_curve_pil(points, color)
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
        """Draw orthogonal lines through control points (only horizontal and vertical)."""
        if len(points) < 2:
            return
        
        # Draw orthogonal (right-angle) lines between control points
        # Each segment is drawn as either purely horizontal or purely vertical
        orthogonal_points = []
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            
            # Add the starting point
            if i == 0:
                orthogonal_points.append((int(p1[0]), int(p1[1])))
            
            # Determine if this segment should be horizontal first or vertical first
            # If x changes more than y, go horizontal first
            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])
            
            if dx > 0 and dy > 0:
                # Need a corner - create intermediate point for orthogonal routing
                # Alternate between horizontal-first and vertical-first based on segment index
                # This creates cleaner routing patterns
                if i % 2 == 0:
                    # Horizontal first, then vertical
                    orthogonal_points.append((int(p2[0]), int(p1[1])))
                else:
                    # Vertical first, then horizontal
                    orthogonal_points.append((int(p1[0]), int(p2[1])))
            
            # Add the end point
            orthogonal_points.append((int(p2[0]), int(p2[1])))
        
        # Remove consecutive duplicates
        unique_points = []
        for pt in orthogonal_points:
            if not unique_points or pt != unique_points[-1]:
                unique_points.append(pt)
        
        # Draw as a polyline with orthogonal segments
        if len(unique_points) > 1:
            self.draw.line(unique_points, fill=color, width=3)
    
    def _zoom_in(self):
        """Zoom in on the canvas."""
        new_zoom = self.zoom_level * self.zoom_step
        if new_zoom <= self.max_zoom:
            self.zoom_level = new_zoom
            self._apply_zoom()
    
    def _zoom_out(self):
        """Zoom out on the canvas."""
        new_zoom = self.zoom_level / self.zoom_step
        if new_zoom >= self.min_zoom:
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
