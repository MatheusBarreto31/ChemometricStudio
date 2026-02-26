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
from matplotlib import colors as mcolors
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator, FuncFormatter
import tkinter as tk
from tkinter import ttk


def _normalize_axis_type(axis_config: dict) -> str:
    """Return normalized axis type from config, defaulting to linear."""
    axis_type = str(axis_config.get('axis_type', 'linear')).strip().lower()
    if axis_type in ('linear', 'log10', 'log2', 'ln'):
        return axis_type
    return 'linear'


def _apply_axis_scale_options(ax, config: dict, use_3d: bool = False) -> None:
    """Apply optional per-axis scaling from config.

    Supported per-axis options:
        - axis_type: str in {'linear', 'log10', 'log2', 'ln'}
    """

    def _format_ln_tick(value, _pos):
        if value <= 0:
            return ""
        exponent = np.log(value)
        rounded = int(np.round(exponent))
        if np.isclose(exponent, rounded, atol=1e-9):
            return rf"$e^{{{rounded}}}$"
        return rf"$e^{{{exponent:.2g}}}$"

    def _apply_scale(axis_obj, set_scale_func, axis_cfg: dict) -> None:
        axis_type = _normalize_axis_type(axis_cfg)
        if axis_type == 'linear':
            set_scale_func('linear')
            return

        if axis_type == 'log10':
            base = 10
        elif axis_type == 'log2':
            base = 2
        else:  # ln
            base = np.e

        set_scale_func('log', base=base)
        if axis_type == 'ln' and axis_obj is not None:
            axis_obj.set_major_formatter(FuncFormatter(_format_ln_tick))

    x_axis_cfg = config.get('x_axis', {})
    y_axis_cfg = config.get('y_axis', {})
    z_axis_cfg = config.get('z_axis', {})

    _apply_scale(ax.xaxis, ax.set_xscale, x_axis_cfg)
    _apply_scale(ax.yaxis, ax.set_yscale, y_axis_cfg)
    if use_3d and hasattr(ax, 'set_zscale'):
        _apply_scale(getattr(ax, 'zaxis', None), ax.set_zscale, z_axis_cfg)


def render_graph_figure(graph_type: str, config: dict, x_data: Optional[np.ndarray],
                       y_data: Optional[np.ndarray], z_data: Optional[np.ndarray],
                       x_axis_config: dict, y_axis_config: dict, default_cmap: str = 'viridis',
                       datasets: Optional[List[Dict[str, Any]]] = None, 
                       qualitative_cmap: str = 'tab10',
                       sample_labels: Optional[List[str]] = None,
                       sample_labels_by_dataset: Optional[Dict[str, List[str]]] = None,
                       font_scale: float = 1.0) -> Tuple[Figure, any]:
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
        _render_line(ax, x_data, y_data, config, datasets)
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

    _apply_axis_scale_options(ax, config, use_3d=use_3d)

    # Optional display and tick options for line/scatter axes
    if graph_type in ('line', 'scatter'):
        _apply_graph_display_options(ax, config, use_3d=use_3d)
        _apply_axis_tick_options(ax, config, use_3d=use_3d)
    
    # constrained_layout handles margins automatically - no manual adjustment needed
    # This ensures tight bounds that adapt to any section geometry without clipping
    _apply_relative_font_scale(fig, font_scale)
    fig._graph_font_scale = float(font_scale) if font_scale else 1.0
    
    return fig, ax


def _apply_relative_font_scale(fig: Figure, font_scale: float) -> None:
    """Apply relative font scaling to all figure text artists."""
    try:
        scale = float(font_scale)
    except (TypeError, ValueError):
        return

    if scale <= 0 or abs(scale - 1.0) < 1e-9:
        return

    def _scale_text(text_obj):
        if text_obj is None:
            return
        try:
            size = text_obj.get_fontsize()
            if size is not None:
                text_obj.set_fontsize(float(size) * scale)
        except Exception:
            pass

    for axes_obj in fig.axes:
        _scale_text(getattr(axes_obj, 'title', None))

        for axis_name in ('xaxis', 'yaxis', 'zaxis'):
            axis_obj = getattr(axes_obj, axis_name, None)
            if axis_obj is None:
                continue
            _scale_text(getattr(axis_obj, 'label', None))
            try:
                _scale_text(axis_obj.get_offset_text())
            except Exception:
                pass

        for getter_name in ('get_xticklabels', 'get_yticklabels', 'get_zticklabels'):
            getter = getattr(axes_obj, getter_name, None)
            if getter is None:
                continue
            try:
                for tick_text in getter():
                    _scale_text(tick_text)
            except Exception:
                continue

        for text_obj in getattr(axes_obj, 'texts', []):
            _scale_text(text_obj)

        legend = axes_obj.get_legend() if hasattr(axes_obj, 'get_legend') else None
        if legend is not None:
            _scale_text(legend.get_title())
            for text_obj in legend.get_texts():
                _scale_text(text_obj)


def _apply_axis_tick_options(ax, config: dict, use_3d: bool = False) -> None:
    """Apply optional axis tick rendering options from config.

    Supported per-axis options:
        - force_integer: bool (if true, force integer-only major ticks)
    """
    x_axis_cfg = config.get('x_axis', {})
    y_axis_cfg = config.get('y_axis', {})
    z_axis_cfg = config.get('z_axis', {})

    if x_axis_cfg.get('force_integer', False):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if y_axis_cfg.get('force_integer', False):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if use_3d and z_axis_cfg.get('force_integer', False) and hasattr(ax, 'zaxis'):
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))


def _apply_graph_display_options(ax, config: dict, use_3d: bool = False) -> None:
    """Apply optional graph-level display options.

    Supported options:
        - show_grid: bool (show axis grid)
        - show_origin: bool (draw dashed x=0 and y=0 reference lines for 2D)
    """
    if config.get('show_grid', False):
        ax.grid(True)

    if config.get('show_origin', False) and not use_3d:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)


def _is_confidence_ellipses_enabled(config: dict) -> bool:
    """Return whether confidence ellipse rendering is enabled in config."""
    value = config.get('confidence_ellipses', False)
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _parse_confidence_level(config: dict) -> float:
    """Parse confidence level from config and return a probability in (0, 1)."""
    raw_value = config.get('confidence_level', '95')
    if raw_value is None:
        raw_value = '95'

    try:
        value = str(raw_value).strip().replace('%', '')
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 95.0

    if confidence > 1.0:
        confidence /= 100.0

    if confidence <= 0.0 or confidence >= 1.0:
        confidence = 0.95

    return confidence


def _draw_confidence_ellipse(ax, x_data: np.ndarray, y_data: np.ndarray,
                             color: Any, confidence: float) -> None:
    """Draw a covariance-based confidence ellipse for 2D points."""
    if x_data is None or y_data is None:
        return

    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        return

    n = min(len(x), len(y))
    if n < 3:
        return

    x = x[:n]
    y = y[:n]
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite_mask) < 3:
        return

    x = x[finite_mask]
    y = y[finite_mask]

    cov = np.cov(x, y)
    if cov.shape != (2, 2):
        return

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    if np.allclose(eigvals, 0.0):
        return

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    chi2_quantile = -2.0 * np.log(1.0 - confidence)
    width = 2.0 * np.sqrt(eigvals[0] * chi2_quantile)
    height = 2.0 * np.sqrt(eigvals[1] * chi2_quantile)

    if width <= 0.0 or height <= 0.0:
        return

    major_axis = eigvecs[:, 0]
    angle_deg = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

    ellipse = Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=float(width),
        height=float(height),
        angle=float(angle_deg),
        facecolor='none',
        edgecolor=color,
        linewidth=1.8,
        alpha=0.9,
        zorder=4
    )
    ax.add_patch(ellipse)


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

        if config.get('show_labels', False) and sample_labels is not None:
            _render_point_labels(ax, x_data, y_data, sample_labels, use_3d=use_3d, z_data=z_data)

        if not use_3d and _is_confidence_ellipses_enabled(config):
            default_color = 'C0'
            facecolors = scatter.get_facecolors() if hasattr(scatter, 'get_facecolors') else None
            if facecolors is not None and len(facecolors) > 0:
                default_color = facecolors[0]
            confidence = _parse_confidence_level(config)
            _draw_confidence_ellipse(ax, x_data, y_data, default_color, confidence)


def _render_point_labels(ax, x_data, y_data, labels, use_3d: bool = False, z_data=None) -> None:
    """Render point labels for scatter plots when enabled."""
    if x_data is None or y_data is None or labels is None:
        return

    n_points = min(len(x_data), len(y_data), len(labels))
    if n_points <= 0:
        return

    for idx in range(n_points):
        label_text = str(labels[idx])
        if not label_text:
            continue

        if use_3d and z_data is not None and idx < len(z_data):
            ax.text(x_data[idx], y_data[idx], z_data[idx], label_text, fontsize=8, alpha=0.85)
        else:
            ax.annotate(
                label_text,
                xy=(x_data[idx], y_data[idx]),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=8,
                alpha=0.85
            )


def _normalize_scatter_legend_show_mode(config: dict) -> str:
    """Normalize scatter legend display mode: auto, yes, no."""
    mode = str(config.get('legend_show_mode', '')).strip().lower()
    if mode in {'auto', 'yes', 'no'}:
        return mode

    legacy_value = config.get('show_legend')
    if isinstance(legacy_value, bool):
        return 'yes' if legacy_value else 'no'

    return 'auto'


def _normalize_scatter_legend_elements(config: dict) -> Dict[str, bool]:
    """Normalize per-element legend toggle config with defaults."""
    defaults = {
        'datasets': True,
        'color': True,
        'marker': False,
        'fill': False,
        'edge': False,
    }
    raw = config.get('legend_elements', {})
    if isinstance(raw, dict):
        for key in list(defaults.keys()):
            if key in raw:
                defaults[key] = bool(raw.get(key))
    return defaults


def _get_scatter_legend_placement(config: dict) -> Dict[str, Any]:
    """Return Matplotlib legend placement kwargs for scatter legend config."""
    position = str(config.get('legend_position', 'auto')).strip().lower()
    if position not in {'auto', 'nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'}:
        position = 'auto'

    location = str(config.get('legend_location', 'inside')).strip().lower()
    if location not in {'inside', 'outside'}:
        location = 'inside'

    inside_loc_map = {
        'auto': 'best',
        'nw': 'upper left',
        'n': 'upper center',
        'ne': 'upper right',
        'e': 'center right',
        'se': 'lower right',
        's': 'lower center',
        'sw': 'lower left',
        'w': 'center left',
    }

    if location == 'inside':
        return {'loc': inside_loc_map.get(position, 'best')}

    outside_map = {
        'auto': ('upper left', (1.02, 1.0)),
        'nw': ('upper right', (-0.02, 1.0)),
        'n': ('lower center', (0.5, 1.02)),
        'ne': ('upper left', (1.02, 1.0)),
        'e': ('center left', (1.02, 0.5)),
        'se': ('lower left', (1.02, 0.0)),
        's': ('upper center', (0.5, -0.02)),
        'sw': ('lower right', (-0.02, 0.0)),
        'w': ('center right', (-0.02, 0.5)),
    }
    loc, anchor = outside_map.get(position, outside_map['auto'])
    return {'loc': loc, 'bbox_to_anchor': anchor, 'borderaxespad': 0.0}


def _render_scatter_multi_dataset(ax, datasets: List[Dict[str, Any]], config: dict, 
                                 use_3d: bool, qualitative_cmap: str,
                                 sample_labels_by_dataset: Optional[Dict[str, List[str]]] = None) -> None:
    """Render scatter with class-layer styling (marker, color, fill, edge-color)."""
    from matplotlib.lines import Line2D

    def _safe_cmap(name: str, fallback: str = 'viridis'):
        try:
            return cm.get_cmap(str(name))
        except Exception:
            return cm.get_cmap(fallback)

    def _flatten_axis(values: Any) -> np.ndarray:
        arr = np.asarray(values)
        if arr.ndim <= 1:
            return arr.reshape(-1)
        return arr.reshape(arr.shape[0], -1)[:, 0]

    def _normalize_class_layers(dataset: Dict[str, Any], n_points: int) -> Optional[np.ndarray]:
        layers = dataset.get('class_layers')
        if layers is None and dataset.get('class_data') is not None:
            layers = dataset.get('class_data')
        if layers is None:
            return None
        arr = np.asarray(layers, dtype=object)
        if arr.ndim == 0:
            arr = np.asarray([[arr.item()]], dtype=object)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        else:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[0] > n_points:
            arr = arr[:n_points, :]
        return arr

    def _layer_is_continuous(values: np.ndarray) -> Optional[bool]:
        cleaned = []
        has_decimal = False
        for raw in np.asarray(values, dtype=object).tolist():
            text = str(raw).strip()
            if text == '' or text.lower() in {'nan', 'none'}:
                continue
            cleaned.append(text)
            if any(ch in text for ch in ('.', 'e', 'E')):
                has_decimal = True
        if not cleaned:
            return None
        numeric = []
        for text in cleaned:
            try:
                numeric.append(float(text))
            except Exception:
                return None
        if has_decimal:
            return True
        unique = np.unique(np.asarray(numeric, dtype=float))
        if unique.size <= 1:
            return False
        span = float(unique.max() - unique.min())
        diffs = np.diff(np.sort(unique))
        median_diff = float(np.median(diffs)) if diffs.size > 0 else 0.0
        if span <= max(3.0, float(unique.size) * 3.0) and median_diff <= 2.0:
            return False
        return True

    def _as_float(values: np.ndarray) -> Optional[np.ndarray]:
        out = []
        for raw in np.asarray(values, dtype=object).tolist():
            try:
                out.append(float(str(raw).strip()))
            except Exception:
                return None
        return np.asarray(out, dtype=float)

    def _discrete_color_map(values: np.ndarray, cmap_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        cmap_obj = _safe_cmap(cmap_name, 'tab10')
        values_arr = np.asarray(values, dtype=object)
        unique_vals = [v for v in np.unique(values_arr)]
        if hasattr(cmap_obj, 'colors'):
            palette = list(cmap_obj.colors)
        else:
            n_sample = max(10, int(getattr(cmap_obj, 'N', 10) or 10))
            palette = [cmap_obj(i / max(1, n_sample - 1)) for i in range(n_sample)]
        mapping = {str(v): mcolors.to_rgba(palette[i % len(palette)]) for i, v in enumerate(unique_vals)}
        colors = np.array([mapping[str(v)] for v in values_arr], dtype=float)
        return colors, mapping

    def _continuous_color_map(values: np.ndarray, cmap_name: str) -> np.ndarray:
        numeric = _as_float(values)
        if numeric is None or numeric.size == 0:
            return np.tile(np.array([[0.4, 0.4, 0.4, 1.0]]), (len(values), 1))
        cmap_obj = _safe_cmap(cmap_name, 'viridis')
        vmin = float(np.nanmin(numeric))
        vmax = float(np.nanmax(numeric))
        if np.isclose(vmin, vmax):
            normed = np.zeros_like(numeric)
        else:
            normed = (numeric - vmin) / (vmax - vmin)
        return cmap_obj(normed)

    is_multi_dataset = len(datasets) > 1
    all_aspects = ['color', 'marker', 'fill', 'edge']
    available_aspects = ['color', 'marker', 'edge'] if is_multi_dataset else all_aspects

    configured_order = config.get('class_layer_order_effective', config.get('class_layer_order', []))
    if not isinstance(configured_order, list):
        configured_order = []
    configured_order = [str(a).strip().lower() for a in configured_order if str(a).strip().lower() in available_aspects]

    configured_map = config.get('class_layer_map_effective', config.get('class_layer_map', {}))
    if not isinstance(configured_map, dict):
        configured_map = {}

    nature_cfg = config.get('class_layer_nature_effective', config.get('class_layer_nature', {}))
    if not isinstance(nature_cfg, dict):
        nature_cfg = {}

    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    fill_style_cycle = ['full', 'none', 'left', 'right', 'bottom', 'top']
    dataset_legend_items = []
    color_legend_mapping: Dict[str, Any] = {}
    marker_legend_mapping: Dict[str, str] = {}
    fill_legend_mapping: Dict[str, str] = {}
    edge_legend_mapping: Dict[str, Any] = {}

    color_cmap_cont = str(config.get('class_color_cmap_continuous', config.get('cmap', 'viridis')))
    edge_cmap_cont = str(config.get('class_edge_cmap_continuous', config.get('cmap', 'viridis')))
    color_cmap_qual = str(config.get('class_color_cmap_qualitative', qualitative_cmap))
    edge_cmap_qual = str(config.get('class_edge_cmap_qualitative', qualitative_cmap))

    show_labels = config.get('show_labels', False)
    for dataset_idx, dataset in enumerate(datasets):
        x_data = _flatten_axis(dataset.get('x_data'))
        y_data = _flatten_axis(dataset.get('y_data'))
        z_data = dataset.get('z_data')
        dataset_label = dataset.get('label', 'Dataset')
        marker = dataset.get('marker', 'o')
        dataset_fillstyle = fill_style_cycle[dataset_idx % len(fill_style_cycle)]
        fallback_color = dataset.get('color')  # Optional explicit color for non-class mode

        if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
            continue

        n_points = min(len(x_data), len(y_data))
        x_data = np.asarray(x_data[:n_points])
        y_data = np.asarray(y_data[:n_points])
        z_data = np.asarray(z_data[:n_points]) if (use_3d and z_data is not None) else None

        class_layers = _normalize_class_layers(dataset, n_points)
        if class_layers is None or class_layers.size == 0:
            base_marker = 'o' if is_multi_dataset else marker
            marker_obj = MarkerStyle(base_marker, fillstyle=dataset_fillstyle if is_multi_dataset else 'full')
            scatter_kwargs = {
                'marker': marker_obj,
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

                if show_labels:
                    _render_point_labels(
                        ax,
                        x_data,
                        y_data,
                        sample_labels_by_dataset[dataset_label],
                        use_3d=use_3d,
                        z_data=z_data
                    )
            
            # Track for legend
            dataset_legend_items.append((dataset_label, base_marker, fallback_color or 'gray', dataset_fillstyle))
        else:
            n_layers = class_layers.shape[1]
            active_order = [a for a in configured_order if a in available_aspects]
            if not active_order:
                auto_count = min(len(available_aspects), n_layers)
                active_order = available_aspects[:auto_count]

            aspect_layer_values: Dict[str, np.ndarray] = {}
            aspect_nature: Dict[str, str] = {}
            for idx, aspect in enumerate(active_order):
                raw_layer_idx = configured_map.get(aspect, idx + 1)
                try:
                    layer_idx = int(raw_layer_idx)
                except Exception:
                    layer_idx = idx + 1

                if layer_idx <= 0:
                    continue

                layer_idx = max(1, min(layer_idx, n_layers))
                if layer_idx < 1 or layer_idx > n_layers:
                    continue

                values = class_layers[:, layer_idx - 1]
                aspect_layer_values[aspect] = values

                nature_raw = str(nature_cfg.get(str(layer_idx), nature_cfg.get(layer_idx, ''))).strip().lower()
                if nature_raw in {'discrete', 'continuous'}:
                    aspect_nature[aspect] = nature_raw
                else:
                    detected = _layer_is_continuous(values)
                    aspect_nature[aspect] = 'continuous' if detected is True else 'discrete'

            marker_values = np.asarray(aspect_layer_values['marker'], dtype=object) if 'marker' in aspect_layer_values else None
            base_marker = 'o' if is_multi_dataset else marker
            marker_vector = np.asarray([base_marker] * n_points, dtype=object)
            if marker_values is not None:
                unique_markers = np.unique(marker_values)
                marker_map = {str(v): marker_cycle[i % len(marker_cycle)] for i, v in enumerate(unique_markers)}
                marker_vector = np.asarray([marker_map[str(v)] for v in marker_values], dtype=object)
                for key, marker_symbol in marker_map.items():
                    marker_legend_mapping[key] = str(marker_symbol)

            fillstyle_vector = np.asarray([
                dataset_fillstyle if is_multi_dataset else 'full'
            ] * n_points, dtype=object)
            if (not is_multi_dataset) and 'fill' in aspect_layer_values:
                fill_values = np.asarray(aspect_layer_values['fill'], dtype=object)
                if aspect_nature.get('fill') == 'continuous':
                    numeric_fill = _as_float(fill_values)
                    if numeric_fill is not None and numeric_fill.size > 0:
                        vmin = float(np.nanmin(numeric_fill))
                        vmax = float(np.nanmax(numeric_fill))
                        if np.isclose(vmin, vmax):
                            fillstyle_vector = np.asarray(['full'] * n_points, dtype=object)
                        else:
                            normalized = (numeric_fill - vmin) / (vmax - vmin)
                            indices = np.clip(np.round(normalized * (len(fill_style_cycle) - 1)).astype(int), 0, len(fill_style_cycle) - 1)
                            fillstyle_vector = np.asarray([fill_style_cycle[idx] for idx in indices], dtype=object)
                else:
                    unique_fill = np.unique(fill_values)
                    fill_map = {str(v): fill_style_cycle[i % len(fill_style_cycle)] for i, v in enumerate(unique_fill)}
                    fillstyle_vector = np.asarray([fill_map[str(v)] for v in fill_values], dtype=object)
                    for key, fillstyle_name in fill_map.items():
                        fill_legend_mapping[key] = str(fillstyle_name)

            if 'color' in aspect_layer_values:
                color_values = np.asarray(aspect_layer_values['color'], dtype=object)
                use_cont = aspect_nature.get('color') == 'continuous'
                if use_cont:
                    face_rgba = _continuous_color_map(color_values, color_cmap_cont)
                else:
                    face_rgba, discrete_map = _discrete_color_map(color_values, color_cmap_qual)
                    for key, rgba in discrete_map.items():
                        color_legend_mapping[key] = rgba
            else:
                if fallback_color:
                    fallback_rgba = np.asarray(mcolors.to_rgba(fallback_color), dtype=float)
                else:
                    fallback_rgba = np.asarray(mcolors.to_rgba('C0'), dtype=float)
                face_rgba = np.tile(fallback_rgba, (n_points, 1))

            if 'edge' in aspect_layer_values:
                edge_values = np.asarray(aspect_layer_values['edge'], dtype=object)
                use_cont_edge = aspect_nature.get('edge') == 'continuous'
                if use_cont_edge:
                    edge_rgba = _continuous_color_map(edge_values, edge_cmap_cont)
                else:
                    edge_rgba, edge_discrete_map = _discrete_color_map(edge_values, edge_cmap_qual)
                    for key, rgba in edge_discrete_map.items():
                        edge_legend_mapping[key] = rgba
            else:
                edge_rgba = np.copy(face_rgba)
                edge_rgba[:, 3] = np.maximum(edge_rgba[:, 3], 0.85)

            unique_markers = [m for m in np.unique(marker_vector)]
            for marker_symbol in unique_markers:
                marker_mask = marker_vector == marker_symbol
                if not np.any(marker_mask):
                    continue
                marker_fillstyles = np.unique(fillstyle_vector[marker_mask])
                for fillstyle_name in marker_fillstyles:
                    point_mask = marker_mask & (fillstyle_vector == fillstyle_name)
                    if not np.any(point_mask):
                        continue

                    x_subset = x_data[point_mask]
                    y_subset = y_data[point_mask]
                    z_subset = z_data[point_mask] if (use_3d and z_data is not None) else None
                    face_subset = np.copy(face_rgba[point_mask])
                    edge_subset = np.copy(edge_rgba[point_mask])
                    if face_subset.ndim == 1:
                        face_subset = face_subset.reshape(1, -1)
                    if edge_subset.ndim == 1:
                        edge_subset = edge_subset.reshape(1, -1)

                    fillstyle_str = str(fillstyle_name)
                    marker_base = MarkerStyle(str(marker_symbol))
                    marker_is_filled = bool(marker_base.is_filled())
                    linewidths_subset = np.full(face_subset.shape[0], 1.8 if fillstyle_str == 'none' else 1.0, dtype=float)

                    scatter = None
                    if not marker_is_filled:
                        point_color = edge_subset if edge_subset is not None else face_subset
                        if use_3d and z_subset is not None:
                            scatter = ax.scatter(
                                x_subset,
                                y_subset,
                                z_subset,
                                marker=str(marker_symbol),
                                c=point_color,
                                alpha=0.7,
                                s=30,
                                linewidths=linewidths_subset,
                                picker=5
                            )
                        else:
                            scatter = ax.scatter(
                                x_subset,
                                y_subset,
                                marker=str(marker_symbol),
                                c=point_color,
                                alpha=0.7,
                                s=30,
                                linewidths=linewidths_subset,
                                picker=5
                            )
                    elif fillstyle_str in {'left', 'right', 'bottom', 'top'}:
                        outline_marker = MarkerStyle(str(marker_symbol), fillstyle='full')
                        fill_marker = MarkerStyle(str(marker_symbol), fillstyle=fillstyle_str)

                        if use_3d and z_subset is not None:
                            ax.scatter(
                                x_subset,
                                y_subset,
                                z_subset,
                                marker=outline_marker,
                                facecolors='none',
                                edgecolors=edge_subset,
                                alpha=0.7,
                                s=30,
                                linewidths=np.full(face_subset.shape[0], 1.2, dtype=float),
                                picker=0
                            )
                            scatter = ax.scatter(
                                x_subset,
                                y_subset,
                                z_subset,
                                marker=fill_marker,
                                c=face_subset,
                                edgecolors='none',
                                alpha=0.7,
                                s=30,
                                linewidths=0.0,
                                picker=5
                            )
                        else:
                            ax.scatter(
                                x_subset,
                                y_subset,
                                marker=outline_marker,
                                facecolors='none',
                                edgecolors=edge_subset,
                                alpha=0.7,
                                s=30,
                                linewidths=np.full(face_subset.shape[0], 1.2, dtype=float),
                                picker=0
                            )
                            scatter = ax.scatter(
                                x_subset,
                                y_subset,
                                marker=fill_marker,
                                c=face_subset,
                                edgecolors='none',
                                alpha=0.7,
                                s=30,
                                linewidths=0.0,
                                picker=5
                            )
                    elif fillstyle_str == 'none':
                        outline_marker = MarkerStyle(str(marker_symbol), fillstyle='full')
                        if use_3d and z_subset is not None:
                            scatter = ax.scatter(
                                x_subset,
                                y_subset,
                                z_subset,
                                marker=outline_marker,
                                facecolors='none',
                                edgecolors=edge_subset,
                                alpha=0.7,
                                s=30,
                                linewidths=np.full(face_subset.shape[0], 1.8, dtype=float),
                                picker=5
                            )
                        else:
                            scatter = ax.scatter(
                                x_subset,
                                y_subset,
                                marker=outline_marker,
                                facecolors='none',
                                edgecolors=edge_subset,
                                alpha=0.7,
                                s=30,
                                linewidths=np.full(face_subset.shape[0], 1.8, dtype=float),
                                picker=5
                            )
                    else:
                        marker_obj = MarkerStyle(str(marker_symbol), fillstyle=fillstyle_str)
                        if use_3d and z_subset is not None:
                            scatter = ax.scatter(
                                x_subset,
                                y_subset,
                                z_subset,
                                marker=marker_obj,
                                c=face_subset,
                                edgecolors=edge_subset,
                                alpha=0.7,
                                s=30,
                                linewidths=linewidths_subset,
                                picker=5
                            )
                        else:
                            scatter = ax.scatter(
                                x_subset,
                                y_subset,
                                marker=marker_obj,
                                c=face_subset,
                                edgecolors=edge_subset,
                                alpha=0.7,
                                s=30,
                                linewidths=linewidths_subset,
                                picker=5
                            )

                    if sample_labels_by_dataset and dataset_label in sample_labels_by_dataset:
                        all_labels = sample_labels_by_dataset[dataset_label]
                        if len(all_labels) >= n_points:
                            labels_subset = [all_labels[i] for i in range(n_points) if point_mask[i]]
                            scatter.sample_labels = labels_subset
                            scatter.x_data = x_subset
                            scatter.y_data = y_subset

            if show_labels and sample_labels_by_dataset and dataset_label in sample_labels_by_dataset:
                _render_point_labels(
                    ax,
                    x_data,
                    y_data,
                    sample_labels_by_dataset[dataset_label],
                    use_3d=use_3d,
                    z_data=z_data
                )

            dataset_legend_items.append((dataset_label, base_marker, 'gray', dataset_fillstyle))
    
    # Set axis labels
    ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
    ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))
    if use_3d:
        ax.set_zlabel(config.get('z_axis', {}).get('label', 'Z'))
    
    # Build custom legend with configurable groups: datasets and class-layer aspects
    legend_show_mode = _normalize_scatter_legend_show_mode(config)
    legend_elements = _normalize_scatter_legend_elements(config)

    legend_handles = []
    legend_labels = []

    legend_groups: List[Tuple[str, List[Tuple[Any, str]]]] = []

    if legend_elements.get('datasets', True) and len(dataset_legend_items) > 1:
        dataset_entries: List[Tuple[Any, str]] = []
        for dataset_label, marker, color, fillstyle in dataset_legend_items:
            handle = Line2D(
                [0],
                [0],
                marker=marker,
                color='none',
                markerfacecolor=color,
                markeredgecolor='gray',
                markersize=6,
                alpha=0.6,
                linewidth=0,
                fillstyle=fillstyle,
            )
            dataset_entries.append((handle, str(dataset_label)))
        if dataset_entries:
            legend_groups.append(('Datasets:', dataset_entries))

    if legend_elements.get('color', True) and color_legend_mapping:
        color_entries: List[Tuple[Any, str]] = []
        for cls, color in color_legend_mapping.items():
            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                           markeredgecolor='darkgray', markersize=6, alpha=0.6, linewidth=0)
            color_entries.append((handle, str(cls)))
        if color_entries:
            legend_groups.append(('Class Color:', color_entries))

    if legend_elements.get('marker', False) and marker_legend_mapping:
        marker_entries: List[Tuple[Any, str]] = []
        for cls, marker_symbol in marker_legend_mapping.items():
            handle = Line2D([0], [0], marker=str(marker_symbol), color='none', markerfacecolor='gray',
                           markeredgecolor='gray', markersize=6, alpha=0.7, linewidth=0)
            marker_entries.append((handle, str(cls)))
        if marker_entries:
            legend_groups.append(('Class Marker:', marker_entries))

    if legend_elements.get('fill', False) and fill_legend_mapping:
        fill_entries: List[Tuple[Any, str]] = []
        for cls, fillstyle_name in fill_legend_mapping.items():
            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
                           markeredgecolor='gray', markersize=6, alpha=0.7, linewidth=0,
                           fillstyle=str(fillstyle_name))
            fill_entries.append((handle, str(cls)))
        if fill_entries:
            legend_groups.append(('Class Fill:', fill_entries))

    if legend_elements.get('edge', False) and edge_legend_mapping:
        edge_entries: List[Tuple[Any, str]] = []
        for cls, edge_color in edge_legend_mapping.items():
            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
                           markeredgecolor=edge_color, markersize=6, alpha=0.8, linewidth=0,
                           markeredgewidth=1.6)
            edge_entries.append((handle, str(cls)))
        if edge_entries:
            legend_groups.append(('Class Edge:', edge_entries))

    show_group_titles = len(legend_groups) > 1
    for title, entries in legend_groups:
        if show_group_titles:
            legend_handles.append(Line2D([0], [0], marker='', color='none', linewidth=0))
            legend_labels.append(title)
        for handle, label in entries:
            legend_handles.append(handle)
            legend_labels.append(label)

    if legend_show_mode != 'no' and legend_handles:
        legend_place_kwargs = _get_scatter_legend_placement(config)
        ax.legend(legend_handles, legend_labels, fontsize='small', **legend_place_kwargs)

    if not use_3d and _is_confidence_ellipses_enabled(config) and len(datasets) > 0:
        confidence = _parse_confidence_level(config)
        first_dataset = datasets[0]
        first_x = _flatten_axis(first_dataset.get('x_data'))
        first_y = _flatten_axis(first_dataset.get('y_data'))
        first_class_data = first_dataset.get('class_data')
        first_color = first_dataset.get('color', 'C0')

        if first_x is not None and first_y is not None and len(first_x) > 0 and len(first_y) > 0:
            if first_class_data is not None:
                first_class_data = np.asarray(first_class_data)
                classes = np.unique(first_class_data)
                for cls in classes:
                    class_mask = first_class_data == cls
                    if not np.any(class_mask):
                        continue
                    class_color = color_legend_mapping.get(str(cls), first_color)
                    _draw_confidence_ellipse(ax, first_x[class_mask], first_y[class_mask], class_color, confidence)
            else:
                _draw_confidence_ellipse(ax, first_x, first_y, first_color, confidence)


def _render_line(ax, x_data: Optional[np.ndarray], y_data: Optional[np.ndarray],
                config: dict, datasets: Optional[List[Dict[str, Any]]] = None) -> None:
    """Render a line plot, supporting single or multiple datasets."""
    # If datasets provided, use multi-dataset rendering
    if datasets and len(datasets) > 0:
        _render_line_multi_dataset(ax, datasets, config)
    # Otherwise use traditional single dataset rendering
    elif x_data is not None and y_data is not None:
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


def _render_line_multi_dataset(ax, datasets: List[Dict[str, Any]], config: dict) -> None:
    """Render multiple line datasets on the same plot.
    
    Args:
        ax: Matplotlib axes
        datasets: List of dataset dicts, each containing:
            - 'x_data': numpy array
            - 'y_data': numpy array
            - 'x_axis': dict with axis config (for label extraction)
            - 'y_axis': dict with axis config (for label extraction)
            - 'label': str (dataset/line name)
            - 'marker': str (optional marker type, e.g., 'o', 's', '^')
            - 'color': str (optional, e.g., '#1f77b4')
        config: Graph configuration
    """
    x_axis_label = None
    y_axis_label = None
    
    # Plot each dataset
    for dataset in datasets:
        x_data = dataset.get('x_data')
        y_data = dataset.get('y_data')
        label = dataset.get('label', 'Dataset')
        marker = dataset.get('marker')
        color = dataset.get('color')
        
        # Extract axis labels from the first dataset
        if x_axis_label is None and 'x_axis' in dataset:
            x_axis_label = dataset['x_axis'].get('label')
        if y_axis_label is None and 'y_axis' in dataset:
            y_axis_label = dataset['y_axis'].get('label')
        
        if x_data is None or y_data is None:
            continue
        
        # Plot the line with optional marker and color
        plot_kwargs = {'label': label}
        if marker is not None:
            plot_kwargs['marker'] = marker
        if color is not None:
            plot_kwargs['color'] = color
        
        ax.plot(x_data, y_data, **plot_kwargs)
    
    # Set axis labels if found
    if x_axis_label:
        ax.set_xlabel(x_axis_label)
    if y_axis_label:
        ax.set_ylabel(y_axis_label)
    
    # Show legend automatically for multi-dataset
    if len(datasets) > 1:
        ax.legend()


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
    
    # Store state for hover tracking and pinned annotations
    tooltip_state = {
        'hover_annotation': None,
        'hover_key': None,
        'pinned_annotations': {}
    }
    graph_font_scale = getattr(fig, '_graph_font_scale', 1.0)

    def _make_annotation(scatter, ind: int, is_pinned: bool = False):
        """Create an annotation for a scatter point index."""
        sample_labels = scatter.sample_labels
        if ind >= len(sample_labels):
            return None

        if not (hasattr(scatter, 'x_data') and hasattr(scatter, 'y_data')):
            return None

        x_data = scatter.x_data
        y_data = scatter.y_data
        if ind >= len(x_data) or ind >= len(y_data):
            return None

        label = str(sample_labels[ind])
        x = x_data[ind]
        y = y_data[ind]

        return ax.annotate(
            label,
            xy=(x, y),
            xytext=(8, 8),
            textcoords='offset points',
            bbox=dict(
                boxstyle='round,pad=0.5',
                fc='lightyellow' if is_pinned else 'yellow',
                alpha=0.85 if is_pinned else 0.7
            ),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1),
            fontsize=max(1.0, 9 * float(graph_font_scale)),
            zorder=11 if is_pinned else 10
        )

    def _find_point_at_event(event):
        """Return (scatter, ind) for the first point under the cursor, else (None, None)."""
        for scatter in scatter_collections:
            if not hasattr(scatter, 'sample_labels'):
                continue

            contains, inds = scatter.contains(event)
            if not contains:
                continue

            ind = inds.get('ind', [None])[0] if isinstance(inds, dict) else (inds[0] if len(inds) > 0 else None)
            if ind is not None:
                return scatter, ind

        return None, None
    
    def on_motion(event):
        """Handle mouse motion to show/hide tooltips."""
        if event.inaxes != ax:
            # Mouse left the axes, hide only hover tooltip (keep pinned labels)
            if tooltip_state['hover_annotation']:
                tooltip_state['hover_annotation'].remove()
                tooltip_state['hover_annotation'] = None
            tooltip_state['hover_key'] = None
            canvas.draw_idle()
            return

        scatter, ind = _find_point_at_event(event)
        if scatter is None or ind is None:
            if tooltip_state['hover_annotation']:
                tooltip_state['hover_annotation'].remove()
                tooltip_state['hover_annotation'] = None
            tooltip_state['hover_key'] = None
            canvas.draw_idle()
            return

        point_key = (id(scatter), ind)
        if point_key == tooltip_state['hover_key']:
            return

        if tooltip_state['hover_annotation']:
            tooltip_state['hover_annotation'].remove()
            tooltip_state['hover_annotation'] = None

        # If the point is pinned, do not draw a duplicate hover annotation
        if point_key in tooltip_state['pinned_annotations']:
            tooltip_state['hover_key'] = point_key
            canvas.draw_idle()
            return

        hover_annotation = _make_annotation(scatter, ind, is_pinned=False)
        tooltip_state['hover_annotation'] = hover_annotation
        tooltip_state['hover_key'] = point_key
        canvas.draw_idle()

    def on_click(event):
        """Toggle pin/unpin for the scatter label under the click position."""
        if getattr(event, 'button', None) not in (1,):
            return

        if event.inaxes != ax:
            return

        scatter, ind = _find_point_at_event(event)
        if scatter is None or ind is None:
            return

        point_key = (id(scatter), ind)
        pinned_annotations = tooltip_state['pinned_annotations']

        # Toggle off if this point is already pinned
        if point_key in pinned_annotations:
            pinned_annotations[point_key].remove()
            del pinned_annotations[point_key]
            canvas.draw_idle()
            return

        # Pin the currently hovered label for this point only when it is displayed
        if tooltip_state['hover_key'] != point_key:
            return

        if tooltip_state['hover_annotation']:
            tooltip_state['hover_annotation'].remove()
            tooltip_state['hover_annotation'] = None

        pinned_annotation = _make_annotation(scatter, ind, is_pinned=True)
        if pinned_annotation is not None:
            pinned_annotations[point_key] = pinned_annotation
        canvas.draw_idle()
    
    # Connect the motion event
    canvas.mpl_connect('motion_notify_event', on_motion)
    canvas.mpl_connect('button_press_event', on_click)
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
