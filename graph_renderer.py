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


def _as_bool(value: Any, default: bool = False) -> bool:
    """Normalize mixed-type truthy config values to bool."""
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    return bool(value)


def _effective_flip_xy(graph_type: str, config: dict) -> bool:
    """Resolve effective flip_xy value with graph-specific defaults."""
    if not isinstance(config, dict):
        return False

    if 'flip_xy' not in config:
        return False

    return _as_bool(config.get('flip_xy'), default=False)


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
            # Keep matplotlib defaults for linear axes so existing custom
            # locators/formatters (e.g., categorical heatmap tick labels)
            # are not reset.
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

    # flip_xy defaults to False when not configured.
    flip_xy_effective = _effective_flip_xy(graph_type, config)
    config['_flip_xy_effective'] = flip_xy_effective
    if flip_xy_effective:
        x_data, y_data = y_data, x_data

        if graph_type == 'heatmap' and isinstance(z_data, np.ndarray) and z_data.ndim == 2:
            z_data = z_data.T

        x_axis_cfg = config.get('x_axis', {})
        y_axis_cfg = config.get('y_axis', {})
        config['x_axis'] = y_axis_cfg.copy() if isinstance(y_axis_cfg, dict) else y_axis_cfg
        config['y_axis'] = x_axis_cfg.copy() if isinstance(x_axis_cfg, dict) else x_axis_cfg

        if isinstance(datasets, list):
            flipped_datasets: List[Dict[str, Any]] = []
            for dataset in datasets:
                if not isinstance(dataset, dict):
                    flipped_datasets.append(dataset)
                    continue
                flipped_dataset = dataset.copy()
                flipped_dataset['x_data'] = dataset.get('y_data')
                flipped_dataset['y_data'] = dataset.get('x_data')
                ds_x_axis = dataset.get('x_axis')
                ds_y_axis = dataset.get('y_axis')
                if ds_x_axis is not None or ds_y_axis is not None:
                    flipped_dataset['x_axis'] = ds_y_axis
                    flipped_dataset['y_axis'] = ds_x_axis
                flipped_datasets.append(flipped_dataset)
            datasets = flipped_datasets
    
    # Set default colormap in config if not already set (allows JSON configs to override)
    if 'cmap' not in config:
        config['cmap'] = qualitative_cmap if str(graph_type).strip().lower() in {'line', 'scatter'} else default_cmap
    
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
        _render_line(ax, x_data, y_data, config, datasets, qualitative_cmap)
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

    # Optional display and tick options for supported graph axes
    if graph_type in ('line', 'scatter'):
        _apply_graph_display_options(ax, config, use_3d=use_3d)
        _apply_axis_tick_options(ax, config, use_3d=use_3d)
    elif graph_type == 'heatmap':
        _apply_axis_tick_options(ax, config, use_3d=False)

    _apply_axis_direction_options(ax, config, use_3d=use_3d)

    if str(graph_type).strip().lower() in {'scatter', 'line'} and not use_3d:
        _render_scatter_reference_lines(ax, config)
    
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


def _apply_axis_direction_options(ax, config: dict, use_3d: bool = False) -> None:
    """Apply optional axis direction options from config.

    Supported per-axis options:
        - reverse_axis: bool (if true, invert axis direction)
    """
    x_axis_cfg = config.get('x_axis', {})
    y_axis_cfg = config.get('y_axis', {})
    z_axis_cfg = config.get('z_axis', {})

    if x_axis_cfg.get('reverse_axis', False) and hasattr(ax, 'invert_xaxis'):
        ax.invert_xaxis()

    if y_axis_cfg.get('reverse_axis', False) and hasattr(ax, 'invert_yaxis'):
        ax.invert_yaxis()

    if use_3d and z_axis_cfg.get('reverse_axis', False) and hasattr(ax, 'invert_zaxis'):
        ax.invert_zaxis()


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


def _as_line_list(value: Any) -> List[Any]:
    """Convert scalar or array-like values to a flat Python list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return [arr.item()]
        return arr.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _pick_indexed(value: Any, index: int, default: Any = None) -> Any:
    """Pick value[index] for list-like values, otherwise return scalar value."""
    if isinstance(value, np.ndarray):
        value = np.asarray(value).reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        if index < len(value):
            return value[index]
        return value[-1]
    if value is None:
        return default
    return value


def _coerce_float(value: Any) -> Optional[float]:
    """Convert value to float when possible."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_line_orientation(line_cfg: Dict[str, Any]) -> str:
    """Normalize scatter reference-line orientation token."""
    token = str(
        line_cfg.get('orientation', line_cfg.get('type', line_cfg.get('mode', '')))
    ).strip().lower()

    if token in {'v', 'vertical', 'vline', 'axvline'}:
        return 'vertical'
    if token in {'h', 'horizontal', 'hline', 'axhline'}:
        return 'horizontal'
    if token in {'segment', 'line', 'between', 'between_points', 'points'}:
        return 'segment'
    return ''


def _render_scatter_reference_lines(ax, config: dict) -> None:
    """Render configurable scatter reference lines from resolved config entries."""
    raw_lines = config.get('scatter_reference_lines', [])
    if isinstance(raw_lines, dict):
        raw_lines = [raw_lines]
    if not isinstance(raw_lines, list) or not raw_lines:
        return

    for line_cfg in raw_lines:
        if not isinstance(line_cfg, dict):
            continue

        orientation = _normalize_line_orientation(line_cfg)
        if not orientation:
            continue

        style_color = line_cfg.get('color', line_cfg.get('colour', 'gray'))
        style_ls = line_cfg.get('linestyle', line_cfg.get('line_style', line_cfg.get('line_type', '--')))
        style_lw = line_cfg.get('linewidth', line_cfg.get('thickness', line_cfg.get('width', 1.0)))
        style_alpha = line_cfg.get('alpha', 0.9)
        labels_cfg = line_cfg.get('labels', line_cfg.get('label'))

        if orientation == 'vertical':
            values_raw = line_cfg.get('values', line_cfg.get('value', line_cfg.get('x')))
            values = _as_line_list(values_raw)
            if not values:
                continue

            for i, raw_value in enumerate(values):
                x_pos = _coerce_float(raw_value)
                if x_pos is None or not np.isfinite(x_pos):
                    continue

                color = _pick_indexed(style_color, i, 'gray')
                linestyle = _pick_indexed(style_ls, i, '--')
                linewidth = _coerce_float(_pick_indexed(style_lw, i, 1.0))
                alpha = _coerce_float(_pick_indexed(style_alpha, i, 0.9))
                label = _pick_indexed(labels_cfg, i)

                ax.axvline(
                    x=x_pos,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth if linewidth is not None else 1.0,
                    alpha=alpha if alpha is not None else 0.9,
                    zorder=5,
                )

                if label is not None and str(label).strip() != '':
                    y_min, y_max = ax.get_ylim()
                    y_anchor = y_max if y_max >= y_min else y_min
                    ax.text(
                        x_pos,
                        y_anchor,
                        str(label),
                        fontsize=8,
                        color=color,
                        alpha=alpha if alpha is not None else 0.9,
                        va='bottom',
                        ha='left',
                    )

        elif orientation == 'horizontal':
            values_raw = line_cfg.get('values', line_cfg.get('value', line_cfg.get('y')))
            values = _as_line_list(values_raw)
            if not values:
                continue

            for i, raw_value in enumerate(values):
                y_pos = _coerce_float(raw_value)
                if y_pos is None or not np.isfinite(y_pos):
                    continue

                color = _pick_indexed(style_color, i, 'gray')
                linestyle = _pick_indexed(style_ls, i, '--')
                linewidth = _coerce_float(_pick_indexed(style_lw, i, 1.0))
                alpha = _coerce_float(_pick_indexed(style_alpha, i, 0.9))
                label = _pick_indexed(labels_cfg, i)

                ax.axhline(
                    y=y_pos,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth if linewidth is not None else 1.0,
                    alpha=alpha if alpha is not None else 0.9,
                    zorder=5,
                )

                if label is not None and str(label).strip() != '':
                    x_min, x_max = ax.get_xlim()
                    x_anchor = x_max if x_max >= x_min else x_min
                    ax.text(
                        x_anchor,
                        y_pos,
                        str(label),
                        fontsize=8,
                        color=color,
                        alpha=alpha if alpha is not None else 0.9,
                        va='bottom',
                        ha='right',
                    )

        else:
            x1_raw = line_cfg.get('x1', line_cfg.get('x_start'))
            y1_raw = line_cfg.get('y1', line_cfg.get('y_start'))
            x2_raw = line_cfg.get('x2', line_cfg.get('x_end'))
            y2_raw = line_cfg.get('y2', line_cfg.get('y_end'))

            points = line_cfg.get('points')
            if points is not None and all(v is None for v in (x1_raw, y1_raw, x2_raw, y2_raw)):
                if isinstance(points, (list, tuple)) and len(points) == 2:
                    p1, p2 = points[0], points[1]
                    if isinstance(p1, (list, tuple)) and len(p1) >= 2 and isinstance(p2, (list, tuple)) and len(p2) >= 2:
                        x1_raw, y1_raw = p1[0], p1[1]
                        x2_raw, y2_raw = p2[0], p2[1]

            x1_vals = _as_line_list(x1_raw)
            y1_vals = _as_line_list(y1_raw)
            x2_vals = _as_line_list(x2_raw)
            y2_vals = _as_line_list(y2_raw)
            n_lines = max(len(x1_vals), len(y1_vals), len(x2_vals), len(y2_vals))
            if n_lines <= 0:
                continue

            for i in range(n_lines):
                x1 = _coerce_float(_pick_indexed(x1_vals, i))
                y1 = _coerce_float(_pick_indexed(y1_vals, i))
                x2 = _coerce_float(_pick_indexed(x2_vals, i))
                y2 = _coerce_float(_pick_indexed(y2_vals, i))
                if any(v is None or not np.isfinite(v) for v in (x1, y1, x2, y2)):
                    continue

                color = _pick_indexed(style_color, i, 'gray')
                linestyle = _pick_indexed(style_ls, i, '--')
                linewidth = _coerce_float(_pick_indexed(style_lw, i, 1.0))
                alpha = _coerce_float(_pick_indexed(style_alpha, i, 0.9))
                label = _pick_indexed(labels_cfg, i)

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth if linewidth is not None else 1.0,
                    alpha=alpha if alpha is not None else 0.9,
                    zorder=5,
                )

                if label is not None and str(label).strip() != '':
                    x_mid = (x1 + x2) / 2.0
                    y_mid = (y1 + y2) / 2.0
                    ax.text(
                        x_mid,
                        y_mid,
                        str(label),
                        fontsize=8,
                        color=color,
                        alpha=alpha if alpha is not None else 0.9,
                        va='bottom',
                        ha='left',
                    )


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
        cmap_name = str(config.get('cmap', qualitative_cmap))
        try:
            cmap_obj = cm.get_cmap(cmap_name)
        except Exception:
            cmap_obj = cm.get_cmap(qualitative_cmap)
        base_color = cmap_obj(0.0)
        if use_3d:
            # 3D scatter
            scatter = ax.scatter(x_data, y_data, z_data, color=base_color, alpha=0.6, picker=5)
            ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
            ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))
            ax.set_zlabel(config.get('z_axis', {}).get('label', 'Z'))
        else:
            # 2D scatter with picker enabled for tooltips
            scatter = ax.scatter(x_data, y_data, color=base_color, alpha=0.6, picker=5)
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
    scatter_cmap_name = str(config.get('cmap', qualitative_cmap))
    try:
        scatter_cmap_obj = _safe_cmap(scatter_cmap_name, qualitative_cmap)
        nonclass_base_color = mcolors.to_rgba(scatter_cmap_obj(0.0))
    except Exception:
        nonclass_base_color = mcolors.to_rgba('C0')

    show_labels = config.get('show_labels', False)
    for dataset_idx, dataset in enumerate(datasets):
        try:
            style_slot = int(dataset.get('style_slot', dataset_idx))
        except Exception:
            style_slot = dataset_idx
        x_data = _flatten_axis(dataset.get('x_data'))
        y_data = _flatten_axis(dataset.get('y_data'))
        z_data = dataset.get('z_data')
        dataset_label = dataset.get('label', 'Dataset')
        marker = dataset.get('marker', 'o')
        dataset_fillstyle = fill_style_cycle[style_slot % len(fill_style_cycle)]
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
            fillstyle_str = dataset_fillstyle if is_multi_dataset else 'full'
            marker_obj = MarkerStyle(base_marker, fillstyle=fillstyle_str)
            dataset_color = fallback_color if fallback_color else nonclass_base_color
            marker_is_filled = bool(MarkerStyle(base_marker).is_filled())

            if marker_is_filled and fillstyle_str in {'left', 'right', 'bottom', 'top'}:
                outline_marker = MarkerStyle(base_marker, fillstyle='full')
                fill_marker = MarkerStyle(base_marker, fillstyle=fillstyle_str)

                if use_3d and z_data is not None:
                    ax.scatter(
                        x_data,
                        y_data,
                        z_data,
                        marker=outline_marker,
                        facecolors='none',
                        edgecolors=dataset_color,
                        alpha=0.6,
                        s=30,
                        linewidths=1.2,
                        picker=0
                    )
                    scatter = ax.scatter(
                        x_data,
                        y_data,
                        z_data,
                        marker=fill_marker,
                        color=dataset_color,
                        edgecolors='none',
                        alpha=0.6,
                        s=30,
                        linewidths=0.0,
                        picker=5
                    )
                else:
                    ax.scatter(
                        x_data,
                        y_data,
                        marker=outline_marker,
                        facecolors='none',
                        edgecolors=dataset_color,
                        alpha=0.6,
                        s=30,
                        linewidths=1.2,
                        picker=0
                    )
                    scatter = ax.scatter(
                        x_data,
                        y_data,
                        marker=fill_marker,
                        color=dataset_color,
                        edgecolors='none',
                        alpha=0.6,
                        s=30,
                        linewidths=0.0,
                        picker=5
                    )
            else:
                scatter_kwargs = {
                    'marker': marker_obj,
                    'alpha': 0.6,
                    's': 30,
                    'picker': 5,  # Enable picking for tooltips
                    'color': dataset_color,
                }

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
            dataset_legend_items.append((dataset_label, base_marker, dataset_color, dataset_fillstyle))
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
        first_color = first_dataset.get('color')
        if not first_color:
            first_color = nonclass_base_color

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
                config: dict, datasets: Optional[List[Dict[str, Any]]] = None,
                qualitative_cmap: str = 'tab10') -> None:
    """Render a line plot, supporting single or multiple datasets with class coloring."""
    # If datasets provided, use multi-dataset rendering with class support
    if datasets and len(datasets) > 0:
        _render_line_multi_dataset(ax, datasets, config, qualitative_cmap)
    # Otherwise use traditional single dataset rendering
    elif x_data is not None and y_data is not None:
        cmap_name = config.get('cmap', qualitative_cmap)
        marker = config.get('marker')  # None if absent, defaults to line only
        color_cfg = config.get('color')
        # Handle 2D arrays (matrices) by plotting each row as a separate line
        if isinstance(y_data, np.ndarray) and y_data.ndim == 2:
            n_rows = y_data.shape[0]
            n_cols = y_data.shape[1]
            legend_show_mode = _normalize_scatter_legend_show_mode(config)

            series_labels_cfg = config.get('line_series_labels')
            series_labels: List[str] = []
            if isinstance(series_labels_cfg, (list, np.ndarray)):
                try:
                    flat_labels = np.asarray(series_labels_cfg, dtype=object).reshape(-1)
                    series_labels = [str(lbl) for lbl in flat_labels.tolist()]
                except Exception:
                    series_labels = []

            x_vector = None
            x_matrix = None
            if isinstance(x_data, np.ndarray):
                if x_data.ndim == 1 and int(x_data.shape[0]) == int(n_cols):
                    x_vector = np.asarray(x_data, dtype=float)
                elif x_data.ndim == 2 and int(x_data.shape[0]) == int(n_rows) and int(x_data.shape[1]) == int(n_cols):
                    x_matrix = np.asarray(x_data, dtype=float)

            try:
                cmap_obj = cm.get_cmap(cmap_name)
            except Exception:
                cmap_obj = cm.get_cmap('tab10')
            use_sequential_palette = bool(hasattr(cmap_obj, 'colors'))
            if use_sequential_palette:
                row_palette = [mcolors.to_rgba(c) for c in list(cmap_obj.colors)]
            for i, row in enumerate(y_data):
                if use_sequential_palette:
                    color = row_palette[i % len(row_palette)]
                else:
                    color = cmap_obj(i / max(1, n_rows - 1))
                if i < len(series_labels) and str(series_labels[i]).strip():
                    label_text = str(series_labels[i]).strip()
                else:
                    label_text = f'Row {i+1}'
                plot_kwargs: Dict[str, Any] = {'color': color, 'label': label_text}
                if marker is not None:
                    plot_kwargs['marker'] = marker
                if x_matrix is not None:
                    ax.plot(x_matrix[i], row, **plot_kwargs)
                elif x_vector is not None:
                    ax.plot(x_vector, row, **plot_kwargs)
                else:
                    ax.plot(row, **plot_kwargs)

            has_named_legend_content = any(str(lbl).strip() for lbl in series_labels)
            show_legend = False
            if legend_show_mode == 'yes':
                show_legend = True
            elif legend_show_mode == 'auto':
                show_legend = has_named_legend_content

            if show_legend:
                ax.legend()
        else:
            plot_kwargs = {}
            if marker is not None:
                plot_kwargs['marker'] = marker
            if color_cfg is not None:
                plot_kwargs['color'] = color_cfg
            else:
                try:
                    cmap_obj = cm.get_cmap(cmap_name)
                except Exception:
                    cmap_obj = cm.get_cmap('tab10')
                plot_kwargs['color'] = cmap_obj(0.0)
            ax.plot(x_data, y_data, **plot_kwargs)
        ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
        ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))


def _render_line_multi_dataset(ax, datasets: List[Dict[str, Any]], config: dict,
                               qualitative_cmap: str = 'tab10') -> None:
    """Render multiple line datasets with optional class-based coloring.

    Class aspects for line plots:
      - colour : one colour per class value (qualitative or continuous colormap)
      - linestyle : one linestyle per dataset (when multiple datasets are present)

    When class data is absent the lines are coloured using the configured colormap
    so that the persistent-settings colormap is always respected.

    Args:
        ax: Matplotlib axes
        datasets: List of dataset dicts, each containing:
            - 'x_data': numpy array
            - 'y_data': numpy array
            - 'x_axis': dict with axis config (for label extraction)
            - 'y_axis': dict with axis config (for label extraction)
            - 'label': str (dataset/line name)
            - 'marker': str (optional marker, e.g. 'o', 's', '^')
            - 'color': str (optional fallback colour, e.g. '#1f77b4')
            - 'class_data': array-like of class labels (optional)
            - 'class_layers': 2-D array of class layers (optional, first column used)
        config: Graph configuration dict
        qualitative_cmap: Qualitative colormap name for discrete class colours
    """
    from matplotlib.lines import Line2D

    def _safe_cmap_line(name: str, fallback: str = 'tab10'):
        try:
            return cm.get_cmap(str(name))
        except Exception:
            return cm.get_cmap(fallback)

    def _flatten_1d(values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=object)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim == 1:
            return arr
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

    def _as_float(values: np.ndarray) -> Optional[np.ndarray]:
        out: List[float] = []
        for raw in np.asarray(values, dtype=object).tolist():
            try:
                out.append(float(str(raw).strip()))
            except Exception:
                return None
        return np.asarray(out, dtype=float)

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

    def _discrete_color_map(values: np.ndarray, cmap_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        cmap_obj = _safe_cmap_line(cmap_name, 'tab10')
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
        cmap_obj = _safe_cmap_line(cmap_name, 'viridis')
        vmin = float(np.nanmin(numeric))
        vmax = float(np.nanmax(numeric))
        if np.isclose(vmin, vmax):
            normed = np.zeros_like(numeric)
        else:
            normed = (numeric - vmin) / (vmax - vmin)
        return cmap_obj(normed)

    def _discrete_symbol_map(values: np.ndarray, symbols: List[str]) -> Tuple[np.ndarray, Dict[str, str]]:
        values_arr = np.asarray(values, dtype=object)
        unique_vals = [v for v in np.unique(values_arr)]
        mapping = {str(v): symbols[i % len(symbols)] for i, v in enumerate(unique_vals)}
        mapped = np.asarray([mapping[str(v)] for v in values_arr], dtype=object)
        return mapped, mapping

    def _prepare_line_rows(x_values: Any, y_values: Any) -> Tuple[List[Optional[np.ndarray]], List[np.ndarray]]:
        y_arr = np.asarray(y_values)
        if y_arr.ndim <= 1:
            y_line = y_arr.reshape(-1)
            x_line = None
            if x_values is not None:
                x_arr = np.asarray(x_values).reshape(-1)
                if x_arr.shape[0] == y_line.shape[0]:
                    x_line = x_arr
            return [x_line], [y_line]

        y_matrix = y_arr.reshape(y_arr.shape[0], -1)
        # Keep legacy matrix-line behavior compatible with previous no-class rendering:
        # plot each row directly as a line (index on x-axis) instead of forcing x_values.
        x_rows = [None] * y_matrix.shape[0]
        y_rows = [y_matrix[i, :] for i in range(y_matrix.shape[0])]
        return x_rows, y_rows

    def _continuous_symbol_map(values: np.ndarray, symbols: List[str]) -> Tuple[np.ndarray, Dict[str, str]]:
        numeric = _as_float(values)
        if numeric is None or numeric.size == 0:
            mapped = np.asarray([symbols[0]] * len(values), dtype=object)
            return mapped, {'bin 1': symbols[0]}
        vmin = float(np.nanmin(numeric))
        vmax = float(np.nanmax(numeric))
        if np.isclose(vmin, vmax):
            mapped = np.asarray([symbols[0]] * len(numeric), dtype=object)
            return mapped, {'bin 1': symbols[0]}
        normed = (numeric - vmin) / (vmax - vmin)
        idxs = np.clip((normed * len(symbols)).astype(int), 0, len(symbols) - 1)
        mapped = np.asarray([symbols[i] for i in idxs], dtype=object)
        legend_map: Dict[str, str] = {}
        for i, symbol in enumerate(symbols):
            legend_map[f'bin {i + 1}'] = symbol
        return mapped, legend_map

    cmap_name = config.get('cmap', qualitative_cmap)
    dataset_linestyle_cycle = ['-', '--', '-.', ':']
    class_linestyle_cycle = ['-', '--', '-.', ':']
    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    is_multi_dataset = len(datasets) > 1
    line_marker_reserved = bool(config.get('line_marker_reserved', False))

    configured_order = config.get('class_layer_order_effective', config.get('class_layer_order', []))
    if not isinstance(configured_order, list):
        configured_order = []
    configured_order = [str(a).strip().lower() for a in configured_order]

    configured_map = config.get('class_layer_map_effective', config.get('class_layer_map', {}))
    if not isinstance(configured_map, dict):
        configured_map = {}

    nature_cfg = config.get('class_layer_nature_effective', config.get('class_layer_nature', {}))
    if not isinstance(nature_cfg, dict):
        nature_cfg = {}

    available_aspects = ['color']
    if not is_multi_dataset:
        available_aspects.append('linestyle')
    if not line_marker_reserved:
        available_aspects.append('marker')

    has_class = any(
        d.get('class_data') is not None or d.get('class_layers') is not None
        for d in datasets
    )

    x_axis_label = None
    y_axis_label = None

    if has_class:
        color_cmap_cont = str(config.get('class_color_cmap_continuous', config.get('cmap', 'viridis')))
        color_cmap_qual = str(config.get('class_color_cmap_qualitative', qualitative_cmap))

        dataset_legend_items: List[Tuple[str, str]] = []
        color_legend_mapping: Dict[str, Any] = {}
        linestyle_legend_mapping: Dict[str, str] = {}
        marker_legend_mapping: Dict[str, str] = {}

        for dataset_idx, dataset in enumerate(datasets):
            try:
                style_slot = int(dataset.get('style_slot', dataset_idx))
            except Exception:
                style_slot = dataset_idx
            x_data_d = dataset.get('x_data')
            y_data_d = dataset.get('y_data')
            dataset_label = dataset.get('label', f'Dataset {dataset_idx + 1}')
            dataset_marker = dataset.get('marker')

            if x_axis_label is None and 'x_axis' in dataset:
                x_axis_label = dataset['x_axis'].get('label')
            if y_axis_label is None and 'y_axis' in dataset:
                y_axis_label = dataset['y_axis'].get('label')

            if x_data_d is None or y_data_d is None:
                continue

            x_rows, y_rows = _prepare_line_rows(x_data_d, y_data_d)
            n_lines = len(y_rows)
            if n_lines <= 0:
                continue
            dataset_linestyle = dataset_linestyle_cycle[style_slot % len(dataset_linestyle_cycle)] if is_multi_dataset else '-'

            class_layers = _normalize_class_layers(dataset, n_lines)
            if class_layers is None or class_layers.size == 0:
                for line_idx, y_line in enumerate(y_rows):
                    plot_kwargs: Dict[str, Any] = {
                        'linestyle': dataset_linestyle,
                        'color': dataset.get('color', 'gray'),
                    }
                    if line_marker_reserved and dataset_marker is not None:
                        marker_text = str(dataset_marker).strip().lower()
                        if marker_text not in {'', 'none'}:
                            plot_kwargs['marker'] = dataset_marker

                    x_line = x_rows[line_idx]
                    if x_line is not None and len(x_line) == len(y_line):
                        ax.plot(x_line, y_line, **plot_kwargs)
                    else:
                        ax.plot(y_line, **plot_kwargs)
                if is_multi_dataset:
                    dataset_legend_items.append((str(dataset_label), dataset_linestyle))
                continue

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

                values = class_layers[:, layer_idx - 1]
                aspect_layer_values[aspect] = values

                nature_raw = str(nature_cfg.get(str(layer_idx), nature_cfg.get(layer_idx, ''))).strip().lower()
                if nature_raw in {'discrete', 'continuous'}:
                    aspect_nature[aspect] = nature_raw
                else:
                    detected = _layer_is_continuous(values)
                    aspect_nature[aspect] = 'continuous' if detected is True else 'discrete'

            color_vector = np.tile(np.asarray(mcolors.to_rgba(dataset.get('color', 'C0'))), (n_lines, 1))
            if 'color' in aspect_layer_values:
                color_values = np.asarray(aspect_layer_values['color'], dtype=object)
                if aspect_nature.get('color') == 'continuous':
                    color_vector = _continuous_color_map(color_values, color_cmap_cont)
                else:
                    color_vector, discrete_map = _discrete_color_map(color_values, color_cmap_qual)
                    for key, rgba in discrete_map.items():
                        color_legend_mapping[key] = rgba

            linestyle_vector = np.asarray([dataset_linestyle] * n_lines, dtype=object)
            if (not is_multi_dataset) and 'linestyle' in aspect_layer_values:
                style_values = np.asarray(aspect_layer_values['linestyle'], dtype=object)
                if aspect_nature.get('linestyle') == 'continuous':
                    linestyle_vector, style_map = _continuous_symbol_map(style_values, class_linestyle_cycle)
                else:
                    linestyle_vector, style_map = _discrete_symbol_map(style_values, class_linestyle_cycle)
                for key, value in style_map.items():
                    linestyle_legend_mapping[str(key)] = str(value)

            marker_vector = np.asarray([None] * n_lines, dtype=object)
            if line_marker_reserved and dataset_marker is not None:
                marker_text = str(dataset_marker).strip().lower()
                if marker_text not in {'', 'none'}:
                    marker_vector = np.asarray([dataset_marker] * n_lines, dtype=object)
            elif 'marker' in aspect_layer_values:
                marker_values = np.asarray(aspect_layer_values['marker'], dtype=object)
                if aspect_nature.get('marker') == 'continuous':
                    marker_vector, marker_map = _continuous_symbol_map(marker_values, marker_cycle)
                else:
                    marker_vector, marker_map = _discrete_symbol_map(marker_values, marker_cycle)
                for key, value in marker_map.items():
                    marker_legend_mapping[str(key)] = str(value)

            for line_idx in range(n_lines):
                plot_kwargs: Dict[str, Any] = {
                    'color': color_vector[line_idx],
                    'linestyle': str(linestyle_vector[line_idx]),
                }
                marker_value = marker_vector[line_idx]
                if marker_value is not None and str(marker_value).strip().lower() not in {'', 'none'}:
                    plot_kwargs['marker'] = marker_value

                x_line = x_rows[line_idx]
                y_line = y_rows[line_idx]
                if x_line is not None and len(x_line) == len(y_line):
                    ax.plot(x_line, y_line, **plot_kwargs)
                else:
                    ax.plot(y_line, **plot_kwargs)

            if is_multi_dataset:
                dataset_legend_items.append((str(dataset_label), dataset_linestyle))

        line_legend_show_mode = _normalize_scatter_legend_show_mode(config)
        line_legend_elements = {
            'datasets': True,
            'color': True,
            'linestyle': False,
            'marker': False,
        }
        raw_line_legend_elements = config.get('legend_elements', {})
        if isinstance(raw_line_legend_elements, dict):
            for key in list(line_legend_elements.keys()):
                if key in raw_line_legend_elements:
                    line_legend_elements[key] = bool(raw_line_legend_elements.get(key))

        legend_groups: List[Tuple[str, List[Tuple[Any, str]]]] = []
        if line_legend_elements.get('datasets', True) and is_multi_dataset and dataset_legend_items:
            ds_entries = [
                (Line2D([0], [0], color='gray', linestyle=ls, linewidth=1.6), lbl)
                for lbl, ls in dataset_legend_items
            ]
            legend_groups.append(('Datasets:', ds_entries))

        if line_legend_elements.get('color', True) and color_legend_mapping:
            color_entries: List[Tuple[Any, str]] = []
            for cls, rgba in color_legend_mapping.items():
                handle = Line2D([0], [0], color=rgba, linestyle='-', linewidth=2.0)
                color_entries.append((handle, str(cls)))
            legend_groups.append(('Class Color:', color_entries))

        if line_legend_elements.get('linestyle', False) and (not is_multi_dataset) and linestyle_legend_mapping:
            style_entries: List[Tuple[Any, str]] = []
            for cls, ls in linestyle_legend_mapping.items():
                handle = Line2D([0], [0], color='gray', linestyle=str(ls), linewidth=1.8)
                style_entries.append((handle, str(cls)))
            legend_groups.append(('Class Line Style:', style_entries))

        if line_legend_elements.get('marker', False) and marker_legend_mapping:
            marker_entries: List[Tuple[Any, str]] = []
            for cls, mk in marker_legend_mapping.items():
                handle = Line2D([0], [0], color='gray', linestyle='-', marker=str(mk), linewidth=1.2)
                marker_entries.append((handle, str(cls)))
            legend_groups.append(('Class Marker:', marker_entries))

        if line_legend_show_mode != 'no' and legend_groups:
            handles: List[Any] = []
            labels: List[str] = []
            show_titles = len(legend_groups) > 1
            for group_title, entries in legend_groups:
                if show_titles:
                    handles.append(Line2D([], [], visible=False))
                    labels.append(group_title)
                for handle, lbl in entries:
                    handles.append(handle)
                    labels.append(str(lbl))
            legend_placement = _get_scatter_legend_placement(config)
            ax.legend(handles=handles, labels=labels, fontsize='small', framealpha=0.8, **legend_placement)

    else:
        n_datasets = len(datasets)
        cmap_obj = _safe_cmap_line(cmap_name, 'tab10')
        use_sequential_palette = bool(hasattr(cmap_obj, 'colors'))
        if use_sequential_palette:
            color_palette = [mcolors.to_rgba(c) for c in list(cmap_obj.colors)]

        total_lines = 0
        for dataset in datasets:
            _x_rows_tmp, _y_rows_tmp = _prepare_line_rows(dataset.get('x_data'), dataset.get('y_data'))
            total_lines += len(_y_rows_tmp)

        style_total_lines = max(1, total_lines)
        for dataset in datasets:
            try:
                style_total_lines = max(style_total_lines, int(dataset.get('style_total_lines', style_total_lines)))
            except Exception:
                continue

        global_line_idx = 0

        for dataset_idx, dataset in enumerate(datasets):
            try:
                style_slot = int(dataset.get('style_slot', dataset_idx))
            except Exception:
                style_slot = dataset_idx
            try:
                style_line_start = int(dataset.get('style_line_start', global_line_idx))
            except Exception:
                style_line_start = global_line_idx

            x_data_d = dataset.get('x_data')
            y_data_d = dataset.get('y_data')
            label = dataset.get('label', f'Dataset {dataset_idx + 1}')
            marker = dataset.get('marker')
            color = dataset.get('color')

            if x_axis_label is None and 'x_axis' in dataset:
                x_axis_label = dataset['x_axis'].get('label')
            if y_axis_label is None and 'y_axis' in dataset:
                y_axis_label = dataset['y_axis'].get('label')

            if x_data_d is None or y_data_d is None:
                continue

            linestyle = dataset_linestyle_cycle[style_slot % len(dataset_linestyle_cycle)] if is_multi_dataset else '-'
            x_rows, y_rows = _prepare_line_rows(x_data_d, y_data_d)
            for line_idx, y_line in enumerate(y_rows):
                plot_kwargs = {'label': label if line_idx == 0 else None, 'linestyle': linestyle}
                if marker is not None:
                    marker_text = str(marker).strip().lower()
                    if marker_text not in {'', 'none'}:
                        plot_kwargs['marker'] = marker
                if color is not None:
                    plot_kwargs['color'] = color
                else:
                    color_slot = style_line_start + line_idx
                    if use_sequential_palette:
                        plot_kwargs['color'] = color_palette[color_slot % len(color_palette)]
                    else:
                        plot_kwargs['color'] = cmap_obj(color_slot / max(1, style_total_lines - 1))

                x_line = x_rows[line_idx]
                if x_line is not None and len(x_line) == len(y_line):
                    ax.plot(x_line, y_line, **plot_kwargs)
                else:
                    ax.plot(y_line, **plot_kwargs)
                global_line_idx += 1

        line_legend_show_mode = _normalize_scatter_legend_show_mode(config)
        line_legend_elements = {
            'datasets': True,
            'color': True,
            'linestyle': False,
            'marker': False,
        }
        raw_line_legend_elements = config.get('legend_elements', {})
        if isinstance(raw_line_legend_elements, dict):
            for key in list(line_legend_elements.keys()):
                if key in raw_line_legend_elements:
                    line_legend_elements[key] = bool(raw_line_legend_elements.get(key))

        if line_legend_show_mode != 'no' and line_legend_elements.get('datasets', True) and n_datasets > 1:
            legend_placement = _get_scatter_legend_placement(config)
            ax.legend(fontsize='small', framealpha=0.8, **legend_placement)

    # ---------------------------------------------------------------- axis labels
    if x_axis_label:
        ax.set_xlabel(x_axis_label)
    if y_axis_label:
        ax.set_ylabel(y_axis_label)


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
            def _prepare_axis(values: np.ndarray, expected_len: int, axis_name: str) -> Tuple[np.ndarray, Optional[List[str]], bool]:
                """Return numeric axis coordinates, optional labels, and optional fixed-tick hint."""
                arr = np.asarray(values, dtype=object).reshape(-1)

                if expected_len > 0:
                    if arr.size == 0:
                        raise ValueError(
                            f"Heatmap axis length mismatch: {axis_name} has 0 values, expected {expected_len}."
                        )
                    elif arr.size != expected_len:
                        raise ValueError(
                            f"Heatmap axis length mismatch: {axis_name} has {arr.size} values, expected {expected_len}."
                        )

                # Numeric axis (including numeric strings)
                try:
                    numeric_axis = np.asarray(arr, dtype=float)
                    if expected_len > 0 and numeric_axis.size != expected_len:
                        numeric_axis = np.arange(expected_len, dtype=float)
                    # Guard against non-increasing coordinates (e.g., padded duplicates),
                    # which can collapse edge cells in pcolormesh.
                    if numeric_axis.size > 1:
                        diffs = np.diff(numeric_axis)
                        if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0):
                            fallback_len = expected_len if expected_len > 0 else numeric_axis.size
                            numeric_axis = np.arange(fallback_len, dtype=float)
                    return numeric_axis, None, False
                except Exception:
                    pass

                # Treat fallback auto-generated labels (V1, V2, ...) as numeric-like.
                # This avoids classifying them as free-text categorical labels.
                normalized_labels = [str(v).strip() for v in arr.tolist()]
                auto_label_indices: List[float] = []
                for label in normalized_labels:
                    if len(label) >= 2 and label[0] in ('V', 'v') and label[1:].isdigit():
                        auto_label_indices.append(float(int(label[1:])))
                    else:
                        auto_label_indices = []
                        break
                if auto_label_indices and len(auto_label_indices) == len(normalized_labels):
                    # Numeric-like auto labels should behave like numeric axes.
                    # Do not force one tick per variable; allow Matplotlib to pick
                    # readable tick spacing for large matrices.
                    return np.asarray(auto_label_indices, dtype=float), None, False

                # Categorical/text axis -> map to integer coordinates and keep labels
                labels = [str(v) for v in arr.tolist()]
                coords = np.arange(len(labels), dtype=float)
                return coords, labels, False

            n_rows, n_cols = z_data.shape[0], z_data.shape[1]
            x_coords, x_labels, x_force_ticks = _prepare_axis(x_data, n_cols, 'x-axis')
            y_coords, y_labels, y_force_ticks = _prepare_axis(y_data, n_rows, 'y-axis')

            # Use 1D-axis pcolormesh for robust handling of center-style coordinates.
            cmap = config.get('cmap', 'viridis')
            im = ax.pcolormesh(x_coords, y_coords, z_data, cmap=cmap, shading='auto')
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(config.get('x_axis', {}).get('label', 'X'))
            ax.set_ylabel(config.get('y_axis', {}).get('label', 'Y'))

            if x_labels is not None:
                ax.set_xticks(x_coords)
                ax.set_xticklabels(x_labels)
                for tick_label in ax.get_xticklabels():
                    tick_label.set_rotation(45)
                    tick_label.set_horizontalalignment('right')
                    tick_label.set_rotation_mode('anchor')
            elif x_force_ticks:
                ax.set_xticks(x_coords)

            if y_labels is not None:
                ax.set_yticks(y_coords)
                ax.set_yticklabels(y_labels)
            elif y_force_ticks:
                ax.set_yticks(y_coords)

            if bool(config.get('annotate_heatmap', False)):
                clim = im.get_clim()
                try:
                    contrast_norm = mcolors.Normalize(vmin=float(clim[0]), vmax=float(clim[1]), clip=True)
                except Exception:
                    contrast_norm = None

                def _annotation_text_color(value: float) -> str:
                    """Choose annotation color from colormap luminance for contrast."""
                    if contrast_norm is None:
                        return 'black'

                    try:
                        normalized = float(contrast_norm(value))
                    except Exception:
                        return 'black'

                    if not np.isfinite(normalized):
                        return 'black'

                    try:
                        r, g, b, _ = im.cmap(normalized)
                    except Exception:
                        return 'black'

                    luminance = (0.2126 * float(r)) + (0.7152 * float(g)) + (0.0722 * float(b))
                    return 'white' if luminance < 0.5 else 'black'

                for row_idx in range(n_rows):
                    for col_idx in range(n_cols):
                        value = z_data[row_idx, col_idx]
                        if not np.isfinite(value):
                            continue
                        ax.text(
                            x_coords[col_idx],
                            y_coords[row_idx],
                            f"{value:.3g}",
                            ha='center',
                            va='center',
                            color=_annotation_text_color(float(value))
                        )


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

        try:
            point_index = int(ind)
        except Exception:
            # Fallback for rare non-scalar indices returned by backend events
            point_index = str(np.asarray(ind).reshape(-1).tolist())
        point_key = (id(scatter), point_index)
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

        hover_annotation = _make_annotation(scatter, point_index, is_pinned=False)
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

        try:
            point_index = int(ind)
        except Exception:
            point_index = str(np.asarray(ind).reshape(-1).tolist())
        point_key = (id(scatter), point_index)
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

        pinned_annotation = _make_annotation(scatter, point_index, is_pinned=True)
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
