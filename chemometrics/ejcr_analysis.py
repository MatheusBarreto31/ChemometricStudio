"""
EJCR (Ellipse Joint Confidence Region) for calibration diagnostics.

Based on the MVC2 MATLAB toolbox by Borin et al. (ejcrcalc.m).

The EJCR tests whether a regression model is unbiased: if the ideal point (slope=1,
intercept=0) lies inside the joint confidence ellipse drawn in slope-intercept parameter
space, the calibration is considered unbiased at the chosen confidence level.

This module is model-agnostic â€” it only requires arrays of reference and predicted values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import f as f_dist


__all__ = ["compute_ejcr", "ejcr_to_reference_line"]

# Default confidence levels used when none are specified
_DEFAULT_CONFIDENCE_LEVELS: List[float] = [0.90, 0.95, 0.99]

# Per-level default line styles applied by ejcr_to_reference_line
_DEFAULT_LEVEL_STYLES: Dict[str, Dict[str, Any]] = {
    "90": {"linestyle": ":", "linewidth": 1.2, "alpha": 0.75},
    "95": {"linestyle": "-", "linewidth": 1.5, "alpha": 0.90},
    "99": {"linestyle": "--", "linewidth": 1.2, "alpha": 0.75},
}


def _normalize_confidence_list(confidence: Any) -> List[float]:
    """Convert a confidence argument to a validated list of probabilities in (0, 1).

    Accepts None (â†’ default set), a single float, or a list.  Values > 1 are
    treated as percentages (e.g. 95 â†’ 0.95).
    """
    if confidence is None:
        return list(_DEFAULT_CONFIDENCE_LEVELS)
    if isinstance(confidence, (int, float)):
        c = float(confidence)
        if c > 1.0:
            c /= 100.0
        return [c] if 0.0 < c < 1.0 else [0.95]
    if isinstance(confidence, (list, tuple)):
        out: List[float] = []
        for v in confidence:
            try:
                c = float(v)
            except (TypeError, ValueError):
                continue
            if c > 1.0:
                c /= 100.0
            if 0.0 < c < 1.0:
                out.append(c)
        return out if out else list(_DEFAULT_CONFIDENCE_LEVELS)
    return list(_DEFAULT_CONFIDENCE_LEVELS)


def _conf_pct_str(confidence: float) -> str:
    """Format a confidence probability as an integer percent string: 0.95 â†’ '95'."""
    return str(int(round(confidence * 100.0)))


def _ellipse_for_level(
    a: float,
    b: float,
    c: float,
    slope: float,
    intercept: float,
    s2: float,
    n: int,
    confidence: float,
    n_points: int,
) -> Dict[str, Any]:
    """Compute one EJCR ellipse boundary and unbiasedness test for a single confidence level.

    Parameters are the pre-computed OLS fit quantities shared by all levels:
    ``a = Î£y_refÂ²``, ``b = 2Î£y_ref``, ``c = n`` (from the design matrix X^T X),
    and the residual variance ``s2``.
    """
    f_crit = float(f_dist.ppf(confidence, dfn=2, dfd=n - 2))
    d = 2.0 * s2 * f_crit

    denom = 4.0 * a * c - b * b
    if denom <= 0.0:
        raise ValueError("Degenerate EJCR: denom â‰¤ 0 â€” y_reference may be constant.")

    limx = float(np.sqrt(4.0 * c * d / denom))

    ds_upper = np.linspace(limx, -limx, n_points)
    ds_lower = np.linspace(-limx, limx, n_points)

    def _di_branch(ds: np.ndarray, sign: float) -> np.ndarray:
        disc = b * b * ds * ds - 4.0 * c * (a * ds * ds - d)
        return (-b * ds + sign * np.sqrt(np.maximum(disc, 0.0))) / (2.0 * c)

    di_upper = _di_branch(ds_upper, +1.0)
    di_lower = _di_branch(ds_lower, -1.0)

    ellipse_slope = np.concatenate(
        [ds_upper + slope, ds_lower + slope, [ds_upper[0] + slope]]
    )
    ellipse_intercept = np.concatenate(
        [di_upper + intercept, di_lower + intercept, [di_upper[0] + intercept]]
    )

    ds_ideal = 1.0 - slope
    di_ideal = 0.0 - intercept
    ideal_metric = a * ds_ideal ** 2 + b * ds_ideal * di_ideal + c * di_ideal ** 2
    is_unbiased = bool(ideal_metric <= d)

    pct = _conf_pct_str(confidence)
    bias_tag = "unbiased" if is_unbiased else "biased"

    return {
        "confidence": confidence,
        "confidence_pct": pct,
        "f_critical": f_crit,
        "f_scale": d,
        "is_unbiased": is_unbiased,
        "ellipse_slope": ellipse_slope.tolist(),
        "ellipse_intercept": ellipse_intercept.tolist(),
        # Ready-to-use entry for the graph renderer (style-free; use ejcr_to_reference_line
        # or ejcr_to_reference_line for styled versions)
        "reference_line_entry": {
            "orientation": "custom_ellipsis",
            "x": ellipse_slope.tolist(),
            "y": ellipse_intercept.tolist(),
            "confidence_level": pct,
            "label": f"{pct}% EJCR ({bias_tag})",
        },
        "_limx": float(limx),
        "_ideal_metric": float(ideal_metric),
    }


def compute_ejcr(
    y_reference,
    y_predicted,
    confidence=None,
    n_points: int = 100,
) -> Dict[str, Any]:
    """Compute EJCR ellipses for one or more confidence levels.

    Fits OLS on *y_predicted vs. y_reference*, builds the joint confidence
    region on slope and intercept in parameter space, and tests whether the
    ideal calibration point (slope=1, intercept=0) falls inside each ellipse.

    Parameters
    ----------
    y_reference:
        Reference (true) values, shape (n_samples,).
    y_predicted:
        Model-predicted values, shape (n_samples,).
    confidence:
        Confidence level(s).  Accepts:

        * ``None`` â€” uses the default set ``[0.90, 0.95, 0.99]``.
        * A single float in (0, 1) or as a percentage > 1 (e.g. ``95`` â†’ 0.95).
        * A list of floats (mixed probability / percentage notation accepted).
    n_points:
        Number of boundary points generated per ellipse path.

    Returns
    -------
    dict with keys:

    ``slope``, ``intercept``
        OLS fit parameters (predicted = intercept + slope Ã— reference).
    ``residual_variance``, ``residual_std``
        Variance and std-dev of residuals around the fitted line (2-DOF OLS).
    ``n_samples``
        Number of finite sample pairs used.
    ``ideal_slope``, ``ideal_intercept``
        Always 1.0 and 0.0 â€” the ideal calibration point to mark on the plot.
    ``ellipses``
        List of per-level dicts.  Each contains ``confidence``,
        ``confidence_pct``, ``f_critical``, ``f_scale``, ``is_unbiased``,
        ``ellipse_slope``, ``ellipse_intercept``, and ``reference_line_entry``
        â€” a ready-to-use (style-free) ``custom_ellipsis`` dict.
    ``reference_line_entries``
        Flat list of all ``reference_line_entry`` dicts â€” one per requested
        confidence level.  Assign directly to ``scatter_lines`` /
        ``reference_lines`` in a graph config for immediate use in the
        renderer.  Entries have no style keys; use :func:`ejcr_to_reference_line`
        for styled versions.
    """
    y_ref = np.asarray(y_reference, dtype=float).ravel()
    y_pred = np.asarray(y_predicted, dtype=float).ravel()

    n = min(len(y_ref), len(y_pred))
    y_ref, y_pred = y_ref[:n], y_pred[:n]

    valid = np.isfinite(y_ref) & np.isfinite(y_pred)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 3:
        raise ValueError(
            f"EJCR requires at least 3 finite sample pairs; got {n_valid}."
        )
    y_ref = y_ref[valid]
    y_pred = y_pred[valid]
    n = n_valid

    levels = _normalize_confidence_list(confidence)

    # --- OLS fit (computed once; shared by all confidence levels) ---
    x_centered = y_ref - np.mean(y_ref)
    y_centered = y_pred - np.mean(y_pred)
    Qxx = float(np.dot(x_centered, x_centered))
    if Qxx <= 0.0:
        raise ValueError("Zero variance in y_reference; cannot compute EJCR.")

    slope = float(np.dot(y_centered, x_centered) / Qxx)
    intercept = float(np.mean(y_pred) - slope * np.mean(y_ref))

    residuals = y_pred - (intercept + slope * y_ref)
    s2 = float(np.dot(residuals, residuals) / (n - 2))

    # Quadratic-form coefficients derived from the OLS design-matrix X^T X
    a = float(np.dot(y_ref, y_ref))
    b = 2.0 * float(np.sum(y_ref))
    c = float(n)

    # --- Ellipses for each requested level ---
    ellipses: List[Dict[str, Any]] = []
    for level in levels:
        ellipses.append(
            _ellipse_for_level(a, b, c, slope, intercept, s2, n, level, n_points)
        )

    reference_line_entries = [ell["reference_line_entry"] for ell in ellipses]

    return {
        "slope": slope,
        "intercept": intercept,
        "residual_variance": s2,
        "residual_std": float(np.sqrt(max(s2, 0.0))),
        "n_samples": n,
        "ideal_slope": 1.0,
        "ideal_intercept": 0.0,
        "ellipses": ellipses,
        "reference_line_entries": reference_line_entries,
        # Internal parameters
        "_Qxx": Qxx,
        "_a": a,
        "_b": b,
        "_c": c,
    }


def ejcr_to_reference_line(
    y_reference,
    y_predicted,
    confidence=None,
    color: str = "steelblue",
    use_level_styles: bool = True,
    include_ideal_point: bool = True,
    n_points: int = 100,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute EJCR and return *styled* scatter_reference_lines entries plus the raw result.

    Convenience wrapper around :func:`compute_ejcr` that adds visual style
    information so entries can be assigned directly to ``scatter_lines`` /
    ``reference_lines`` in a graph config without further editing.

    Parameters
    ----------
    y_reference, y_predicted, confidence, n_points:
        Forwarded to :func:`compute_ejcr`.
    color:
        Edge colour applied to all ellipses.  Default ``"steelblue"``.
    use_level_styles:
        When ``True`` (default), applies per-level defaults for linestyle,
        linewidth, and alpha:

        * 90% â€” dotted, 1.2 pt, Î± 0.75
        * 95% â€” solid, 1.5 pt, Î± 0.90
        * 99% â€” dashed, 1.2 pt, Î± 0.75

        Levels not in this table use solid / 1.5 pt / Î± 0.9.
    include_ideal_point:
        When ``True``, appends a marker entry for the ideal calibration point
        (slope=1, intercept=0).
    n_points:
        Points per ellipse boundary.

    Returns
    -------
    entries : list[dict]
        Ready-to-use ``custom_ellipsis`` entries for ``scatter_lines`` /
        ``reference_lines``.
    result : dict
        Full dict from :func:`compute_ejcr`.
    """
    result = compute_ejcr(
        y_reference, y_predicted, confidence=confidence, n_points=n_points
    )

    entries: List[Dict[str, Any]] = []
    _fallback_style: Dict[str, Any] = {"linestyle": "-", "linewidth": 1.5, "alpha": 0.9}
    for ell in result["ellipses"]:
        pct = ell["confidence_pct"]
        entry = dict(ell["reference_line_entry"])
        entry["color"] = color
        entry.update(
            _DEFAULT_LEVEL_STYLES.get(pct, _fallback_style)
            if use_level_styles
            else _fallback_style
        )
        entries.append(entry)

    if include_ideal_point:
        entries.append(
            {
                "orientation": "custom_ellipsis",
                "x": [result["ideal_slope"]],
                "y": [result["ideal_intercept"]],
                "color": "black",
                "linestyle": "none",
                "linewidth": 0,
                "alpha": 0.9,
                "marker": "o",
                "markersize": 7,
                "label": "Ideal (1, 0)",
            }
        )

    return entries, result

