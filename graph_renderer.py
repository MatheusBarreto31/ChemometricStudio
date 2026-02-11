"""
Graph Rendering Module - Extracts and renders matplotlib figures for the Analysis Tab.

This module contains the core graph rendering logic used across the analysis tab,
including support for multiple graph types (scatter, line, bar, histogram, heatmap,
3D surface, contour) and data slicing/navigation.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm
import tkinter as tk
from tkinter import ttk


def render_graph_figure(graph_type: str, config: dict, x_data: Optional[np.ndarray],
                       y_data: Optional[np.ndarray], z_data: Optional[np.ndarray],
                       x_axis_config: dict, y_axis_config: dict, default_cmap: str = 'viridis',
                       datasets: Optional[List[Dict[str, Any]]] = None, 
                       qualitative_cmap: str = 'tab10',
                       sample_labels: Optional[List[str]] = None,
                       sample_labels_by_dataset: Optional[Dict[str, List[str]]] = None) -> Tuple[Figure, any]:
    """Create and render a matplotlib figure for the specified graph type.
    
    Args:
        graph_type: Type of graph ('scatter', 'line', 'bar', 'histogram', 'heatmap', '3d_surf', 'contour')
        config: Graph configuration dictionary
        x_data: X-axis data (numpy array)
        y_data: Y-axis data (numpy array)
        z_data: Z-axis data for 3D graphs (numpy array)
        x_axis_config: Configuration for x-axis
        y_axis_config: Configuration for y-axis
        default_cmap: Default colormap to use (can be overridden by config['cmap'])
        datasets: Optional list of dataset dicts for multi-dataset scatter plots
        qualitative_cmap: Qualitative colormap for class-based coloring
        sample_labels: Optional list of sample labels for tooltip display on scatter plots
        sample_labels_by_dataset: Optional dict mapping dataset labels to their sample labels
    
    Returns:
        Tuple of (Figure, axes) - the matplotlib figure and axes
    """
    # Use constrained_layout for automatic tight bounds that adapt to container size
    # This prevents clipping when section geometry changes while keeping tight margins
    # Add slight padding (w_pad/h_pad) for visual comfort around the plot area
    fig = Figure(figsize=(6, 4), dpi=100, constrained_layout={'w_pad': 0.1, 'h_pad': 0.1})
    
    # Make a copy of config to avoid modifying the original (which persists between runs)
    config = config.copy()
    
    # Set default colormap in config if not already set (allows JSON configs to override)
    if 'cmap' not in config:
        config['cmap'] = default_cmap
    
    # Check if z_axis is defined for 3D scatter or 3d_surf
    z_axis_config = config.get('z_axis', {})
    use_3d = (graph_type == 'scatter' or graph_type == '3d_surf') and z_axis_config and z_data is not None
    
    if use_3d:
        # Create 3D plot
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    # Render based on graph type
    if graph_type == 'scatter':
        _render_scatter(ax, x_data, y_data, z_data, config, use_3d, datasets, qualitative_cmap, 
                       sample_labels, sample_labels_by_dataset)
    elif graph_type == 'line':
        _render_line(ax, x_data, y_data, config)
    elif graph_type == 'bar':
        _render_bar(ax, x_data, y_data, config)
    elif graph_type == 'histogram':
        _render_histogram(ax, y_data, config)
    elif graph_type == 'heatmap':
        _render_heatmap(fig, ax, x_data, y_data, z_data, config)
    elif graph_type == '3d_surf':
        _render_3d_surface(ax, x_data, y_data, z_data, config)
    elif graph_type == 'contour':
        _render_contour(ax, x_data, y_data, z_data, config)
    
    # Add graph title only if graph_title is provided
    graph_title = config.get('graph_title')
    if graph_title:
        ax.set_title(graph_title)
    
    # constrained_layout handles margins automatically - no manual adjustment needed
    # This ensures tight bounds that adapt to any section geometry without clipping
    
    return fig, ax


def _render_scatter(ax, x_data: Optional[np.ndarray], y_data: Optional[np.ndarray],
                   z_data: Optional[np.ndarray], config: dict, use_3d: bool,
                   datasets: Optional[List[Dict[str, Any]]] = None,
                   qualitative_cmap: str = 'tab10',
                   sample_labels: Optional[List[str]] = None,
                   sample_labels_by_dataset: Optional[Dict[str, List[str]]] = None) -> None:
    """Render a scatter plot (2D or 3D), supporting single or multiple datasets with class coloring.
    
    For multiple datasets with class information:
    - Each dataset uses a unique marker type
    - Each class within a dataset uses a unique color from the qualitative colormap
    - Legend shows dataset-class combinations
    - Sample labels enable tooltip display on hover
    
    Args:
        ax: Matplotlib axes
        x_data: X-axis data
        y_data: Y-axis data
        z_data: Z-axis data (for 3D)
        config: Graph configuration
        use_3d: Whether to render as 3D scatter
        datasets: Optional list of dataset dicts for multi-dataset rendering
        qualitative_cmap: Name of qualitative colormap
        sample_labels: Optional list of sample labels for single dataset
        sample_labels_by_dataset: Optional dict mapping dataset labels to their sample labels
    """
    # If datasets provided, use multi-dataset rendering with class support
    if datasets and len(datasets) > 0:
        _render_scatter_multi_dataset(ax, datasets, config, use_3d, qualitative_cmap, 
                                     sample_labels_by_dataset)
    # Otherwise use traditional single dataset rendering
    elif x_data is not None and y_data is not None:
        if use_3d:
            # 3D scatter
            scatter = ax.scatter(x_data, y_data, z_data, alpha=0.6, picker=5)
            ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
            ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))
            ax.set_zlabel(config.get('z_axis', {}).get('label', 'Z'))
        else:
            # 2D scatter with picker enabled for tooltips
            scatter = ax.scatter(x_data, y_data, alpha=0.6, picker=5)
            ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
            ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))
        
        # Store sample labels on the scatter plot object for later retrieval
        if sample_labels is not None:
            scatter.sample_labels = sample_labels
            scatter.x_data = x_data
            scatter.y_data = y_data


def _render_scatter_multi_dataset(ax, datasets: List[Dict[str, Any]], config: dict, 
                                 use_3d: bool, qualitative_cmap: str,
                                 sample_labels_by_dataset: Optional[Dict[str, List[str]]] = None) -> None:
    """Render multiple datasets on the same scatter plot with class-based coloring.
    
    Each dataset uses a unique marker. Classes within datasets use colors from the qualitative colormap.
    Dataset color (if provided) is used as fallback when no class_data is available.
    Sample labels enable tooltip display on hover.
    
    Args:
        ax: Matplotlib axes
        datasets: List of dataset dicts, each containing:
            - 'x_data': numpy array
            - 'y_data': numpy array
            - 'z_data': numpy array (optional, for 3D)
            - 'label': str (dataset name)
            - 'marker': str (marker type, e.g., 'o', 's', '^')
            - 'class_data': numpy array of class labels (optional)
            - 'color': str (fallback color if no class_data, e.g., '#808080')
        config: Graph configuration
        use_3d: Whether to render as 3D scatter
        qualitative_cmap: Name of qualitative colormap
        sample_labels_by_dataset: Optional dict mapping dataset labels to their sample labels
    """
    # Get colormap
    try:
        cmap = cm.get_cmap(qualitative_cmap)
    except ValueError:
        cmap = cm.get_cmap('tab10')  # Fallback to tab10
    
    # Extract discrete color list from qualitative colormap
    # Qualitative colormaps have a fixed set of colors (e.g., tab10 has 10)
    if hasattr(cmap, 'colors'):
        cmap_colors = list(cmap.colors)
    else:
        # Fallback: sample N colors evenly from the colormap
        n_sample = getattr(cmap, 'N', 10)
        cmap_colors = [cmap(i / n_sample) for i in range(n_sample)]
    
    # Collect all unique classes across all datasets to ensure consistent coloring
    all_classes = set()
    for dataset in datasets:
        if 'class_data' in dataset and dataset['class_data'] is not None:
            unique_classes = np.unique(dataset['class_data'])
            all_classes.update(unique_classes)
    
    # Sort classes for consistent color assignment
    all_classes = sorted(list(all_classes))
    
    # Create class-to-color mapping using discrete colormap colors
    class_to_color = {}
    if all_classes:
        for idx, cls in enumerate(all_classes):
            class_to_color[cls] = cmap_colors[idx % len(cmap_colors)]
    
    # Track plotted items for legend
    legend_entries = []
    
    # Check if we have only one dataset (for cleaner legend)
    is_single_dataset = len(datasets) == 1
    
    # Plot each dataset
    for dataset in datasets:
        x_data = dataset.get('x_data')
        y_data = dataset.get('y_data')
        z_data = dataset.get('z_data')
        dataset_label = dataset.get('label', 'Dataset')
        marker = dataset.get('marker', 'o')
        class_data = dataset.get('class_data')
        fallback_color = dataset.get('color')  # Optional explicit color for non-class mode
        
        if x_data is None or y_data is None:
            continue
        
        # If no class data, plot all points with same color (explicit or default)
        if class_data is None:
            scatter_kwargs = {
                'marker': marker,
                'label': dataset_label,
                'alpha': 0.6,
                's': 30,
                'picker': 5  # Enable picking for tooltips
            }
            if fallback_color:
                scatter_kwargs['color'] = fallback_color
            
            if use_3d and z_data is not None:
                scatter = ax.scatter(x_data, y_data, z_data, **scatter_kwargs)
            else:
                scatter = ax.scatter(x_data, y_data, **scatter_kwargs)
            
            # Store sample labels on scatter object if provided
            if sample_labels_by_dataset and dataset_label in sample_labels_by_dataset:
                scatter.sample_labels = sample_labels_by_dataset[dataset_label]
                scatter.x_data = x_data
                scatter.y_data = y_data
            
            legend_entries.append(dataset_label)
        else:
            # Plot by class with different colors (class coloring takes precedence over fallback_color)
            unique_classes = np.unique(class_data)
            for cls in unique_classes:
                mask = class_data == cls
                x_subset = x_data[mask]
                y_subset = y_data[mask]
                
                # Get color for this class
                color = class_to_color.get(cls, 'C0')
                
                # Create label for legend - exclude dataset name if single dataset
                if is_single_dataset:
                    label = str(cls)
                else:
                    label = f"{dataset_label}-{cls}"
                
                if use_3d and z_data is not None:
                    z_subset = z_data[mask]
                    scatter = ax.scatter(x_subset, y_subset, z_subset, marker=marker, label=label, 
                             color=color, alpha=0.6, s=30, picker=5)
                else:
                    scatter = ax.scatter(x_subset, y_subset, marker=marker, label=label, 
                             color=color, alpha=0.6, s=30, picker=5)
                
                # Store sample labels on scatter object if provided
                # For class-colored datasets, subset the labels to match the masked data
                if sample_labels_by_dataset and dataset_label in sample_labels_by_dataset:
                    all_labels = sample_labels_by_dataset[dataset_label]
                    # Apply same mask to subset the labels
                    if mask is not None and len(all_labels) == len(mask):
                        labels_subset = [all_labels[i] for i in range(len(all_labels)) if mask[i]]
                        scatter.sample_labels = labels_subset
                        scatter.x_data = x_subset
                        scatter.y_data = y_subset
                
                legend_entries.append(label)
    
    # Set axis labels
    ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
    ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))
    if use_3d:
        ax.set_zlabel(config.get('z_axis', {}).get('label', 'Z'))
    
    # Add legend if we have multiple entries
    if len(legend_entries) > 1 or (len(legend_entries) == 1 and config.get('show_legend', False)):
        ax.legend(loc='best', fontsize='small')


def _render_line(ax, x_data: Optional[np.ndarray], y_data: Optional[np.ndarray],
                config: dict) -> None:
    """Render a line plot."""
    if x_data is not None and y_data is not None:
        marker = config.get('marker')  # None if absent, defaults to line only
        # Handle 2D arrays (matrices) by plotting each row as a separate line
        if isinstance(y_data, np.ndarray) and y_data.ndim == 2:
            for i, row in enumerate(y_data):
                ax.plot(row, marker=marker, label=f'Row {i+1}')
            # Show legend if enabled (default: False)
            if config.get('show_legend', False):
                ax.legend()
        else:
            ax.plot(x_data, y_data, marker=marker)
        ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
        ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))


def _render_bar(ax, x_data: Optional[np.ndarray], y_data: Optional[np.ndarray],
               config: dict) -> None:
    """Render a bar plot."""
    if x_data is not None and y_data is not None:
        if isinstance(x_data, np.ndarray) and x_data.ndim == 1:
            ax.bar(range(len(y_data)), y_data)
            ax.set_ylabel(config.get('y_axis', {}).get('label', 'Value'))
        else:
            ax.bar(x_data, y_data)
            ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
            ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))


def _render_histogram(ax, y_data: Optional[np.ndarray], config: dict) -> None:
    """Render a histogram."""
    if y_data is not None:
        ax.hist(y_data, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel(config.get('x_axis', {}).get('label', 'Value'))
        ax.set_ylabel('Frequency')


def _render_heatmap(fig, ax, x_data: Optional[np.ndarray], y_data: Optional[np.ndarray],
                   z_data: Optional[np.ndarray], config: dict) -> None:
    """Render a heatmap."""
    if x_data is not None and y_data is not None and z_data is not None:
        if isinstance(x_data, np.ndarray) and isinstance(y_data, np.ndarray) and isinstance(z_data, np.ndarray):
            # Create mesh grids from x and y data
            X, Y = np.meshgrid(x_data, y_data)
            # Use pcolormesh for proper axis mapping
            cmap = config.get('cmap', 'viridis')
            im = ax.pcolormesh(X, Y, z_data.T, cmap=cmap, shading='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
            ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))


def _render_3d_surface(ax, x_data: Optional[np.ndarray], y_data: Optional[np.ndarray],
                      z_data: Optional[np.ndarray], config: dict) -> None:
    """Render a 3D surface plot."""
    if x_data is not None and y_data is not None and z_data is not None:
        if isinstance(x_data, np.ndarray) and isinstance(y_data, np.ndarray) and isinstance(z_data, np.ndarray):
            # Create mesh grids from x and y data
            X, Y = np.meshgrid(x_data, y_data)
            # Use plot_surface or plot_wireframe for 3D surface plot
            use_wireframe = config.get('use_wireframe', False)
            cmap = config.get('cmap', 'viridis')
            if use_wireframe:
                ax.plot_wireframe(X, Y, z_data, cmap=cmap)
            else:
                ax.plot_surface(X, Y, z_data, cmap=cmap)
            ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
            ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))
            ax.set_zlabel(config.get('z_axis', {}).get('label', 'Z'))
            
            # Apply section aspect ratio to match X and Y data dimensions
            x_range = np.ptp(x_data)  # Peak-to-peak (max - min)
            y_range = np.ptp(y_data)
            
            # Calculate aspect ratio for X vs Y (the section/horizontal plane)
            # Keep Z at a fixed reasonable scale (1/3 of the max X-Y range)
            if x_range > 0 and y_range > 0:
                max_xy_range = max(x_range, y_range)
                z_scale = max_xy_range / 3  # Z gets 1/3 the scale of the larger X-Y dimension
                aspect_ratio = (x_range, y_range, z_scale)
                ax.set_box_aspect(aspect_ratio)


def _render_contour(ax, x_data: Optional[np.ndarray], y_data: Optional[np.ndarray],
                   z_data: Optional[np.ndarray], config: dict) -> None:
    """Render a contour plot."""
    if x_data is not None and y_data is not None and z_data is not None:
        if isinstance(x_data, np.ndarray) and isinstance(y_data, np.ndarray):
            X, Y = np.meshgrid(x_data, y_data)
            contour_type = config.get('contour_type', 'contourf')
            cmap = config.get('cmap', 'viridis')
            if contour_type == 'contourf':
                ax.contourf(X, Y, z_data, levels=10, cmap=cmap)
            else:
                ax.contour(X, Y, z_data, levels=10, cmap=cmap)
            ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
            ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))


def embed_figure_in_tkinter(fig: Figure, parent_frame: ttk.Frame) -> Tuple[FigureCanvasTkAgg, ttk.Frame]:
    """Embed a matplotlib figure in a tkinter widget with tooltip support.
    
    Args:
        fig: Matplotlib figure to embed
        parent_frame: Parent tkinter frame
    
    Returns:
        Tuple of (canvas, canvas_frame) for later updates
    """
    # Create a frame to hold the canvas for better geometry management
    # Minimal padding - constrained_layout handles figure margins automatically
    canvas_frame = ttk.Frame(parent_frame, padding=1)
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
    
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas_widget = canvas.get_tk_widget()
    # Use grid with sticky to maintain size during updates and prevent resize flash
    canvas_widget.grid(row=0, column=0, sticky='nsew')
    canvas_frame.grid_rowconfigure(0, weight=1)
    canvas_frame.grid_columnconfigure(0, weight=1)
    canvas.draw()
    
    # Set up tooltip display for scatter plots with sample labels
    _setup_scatter_tooltips(canvas, fig)
    
    return canvas, canvas_frame


def _setup_scatter_tooltips(canvas: FigureCanvasTkAgg, fig: Figure) -> None:
    """Set up hover tooltips for scatter plot data points.
    
    Args:
        canvas: The matplotlib canvas
        fig: The matplotlib figure
    """
    # Get axes from figure
    ax = fig.axes[0] if fig.axes else None
    if not ax:
        return
    
    # Find scatter collections with sample labels
    scatter_collections = [child for child in ax.collections 
                          if hasattr(child, 'sample_labels')]
    
    if not scatter_collections:
        return
    
    # Store state for hover tracking
    tooltip_state = {'annotation': None, 'last_ind': None}
    
    def on_motion(event):
        """Handle mouse motion to show/hide tooltips."""
        if event.inaxes != ax:
            # Mouse left the axes, hide tooltip
            if tooltip_state['annotation']:
                tooltip_state['annotation'].remove()
                tooltip_state['annotation'] = None
            canvas.draw_idle()
            return
        
        # Check each scatter collection for nearby points
        for scatter in scatter_collections:
            if not hasattr(scatter, 'sample_labels'):
                continue
            
            contains, inds = scatter.contains(event)
            
            if contains:
                # Found a point near the cursor
                ind = inds.get('ind', [None])[0] if isinstance(inds, dict) else (inds[0] if len(inds) > 0 else None)
                
                if ind is not None and ind != tooltip_state['last_ind']:
                    # Remove old annotation if it exists
                    if tooltip_state['annotation']:
                        tooltip_state['annotation'].remove()
                        tooltip_state['annotation'] = None
                    
                    # Get sample label
                    sample_labels = scatter.sample_labels
                    if ind < len(sample_labels):
                        label = str(sample_labels[ind])
                        
                        # Get point coordinates
                        if hasattr(scatter, 'x_data') and hasattr(scatter, 'y_data'):
                            x_data = scatter.x_data
                            y_data = scatter.y_data
                            if ind < len(x_data) and ind < len(y_data):
                                x = x_data[ind]
                                y = y_data[ind]
                                
                                # Create annotation (tooltip)
                                tooltip_state['annotation'] = ax.annotate(
                                    label,
                                    xy=(x, y),
                                    xytext=(8, 8),
                                    textcoords='offset points',
                                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1),
                                    fontsize=9,
                                    zorder=10
                                )
                                tooltip_state['last_ind'] = ind
                                canvas.draw_idle()
                                return
            else:
                # Mouse not over this scatter plot
                if tooltip_state['last_ind'] is not None:
                    # Remove tooltip if we were showing one from this scatter
                    if tooltip_state['annotation']:
                        tooltip_state['annotation'].remove()
                        tooltip_state['annotation'] = None
                    tooltip_state['last_ind'] = None
    
    # Connect the motion event
    canvas.mpl_connect('motion_notify_event', on_motion)
from typing import Union, Tuple

def update_embedded_figure(fig: Figure, instance_alias: str, section_id: Union[int, Tuple[int, int]],
                          analysis_data: dict, canvas_frame: ttk.Frame) -> None:
    """Update an already-embedded matplotlib figure with new data.
    
    Uses in-place update when possible to prevent resize flash during navigation.
    
    Args:
        fig: New matplotlib figure
        instance_alias: Identifier for the current function instance
        section_id: ID of the graph section being updated
        analysis_data: Dictionary storing analysis state
        canvas_frame: The frame containing the old canvas
    """
    # Get the stored canvas reference and update it
    if 'graph_canvases' in analysis_data.get(instance_alias, {}):
        canvas_data = analysis_data[instance_alias]['graph_canvases'].get(section_id)
        if canvas_data:
            old_canvas, stored_canvas_frame = canvas_data
            
            # Get current widget size to maintain dimensions
            old_widget = old_canvas.get_tk_widget()
            
            # Force geometry update to get accurate dimensions
            old_widget.update_idletasks()
            
            current_width = old_widget.winfo_width()
            current_height = old_widget.winfo_height()
            
            # If dimensions are too small (not yet rendered), use requested dimensions
            if current_width <= 1:
                current_width = old_widget.winfo_reqwidth()
            if current_height <= 1:
                current_height = old_widget.winfo_reqheight()
            
            # Destroy old canvas widget
            old_widget.destroy()
            
            # Resize figure to match container dimensions to prevent flash
            if current_width > 1 and current_height > 1:
                dpi = fig.get_dpi()
                fig.set_size_inches(current_width / dpi, current_height / dpi)
            
            # Create and embed new canvas with updated figure
            new_canvas = FigureCanvasTkAgg(fig, master=stored_canvas_frame)
            canvas_widget = new_canvas.get_tk_widget()
            # Use grid to maintain consistent sizing
            canvas_widget.grid(row=0, column=0, sticky='nsew')
            new_canvas.draw_idle()  # Use draw_idle for smoother updates
            
            # Set up tooltips for the updated figure
            _setup_scatter_tooltips(new_canvas, fig)
            
            # Update stored reference
            analysis_data[instance_alias]['graph_canvases'][section_id] = (new_canvas, stored_canvas_frame)
