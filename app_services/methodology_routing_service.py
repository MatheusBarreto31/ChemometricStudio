"""Helpers for methodology routing-line mutation and remapping."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def remap_manual_routing_lines(
    routing_lines: Mapping[Any, Any],
    old_to_new_idx: Mapping[int, int],
) -> Dict[Any, Any]:
    """Remap manual routing entries from old function indices to new ones."""
    remapped_routing: Dict[Any, Any] = {}

    for key, routing_info in routing_lines.items():
        if isinstance(routing_info, dict) and routing_info.get("auto_created"):
            continue

        src_idx = None
        src_param_key = None
        dst_idx = None
        dst_param_key = None

        if isinstance(key, tuple) and len(key) >= 4:
            src_idx, src_param_key, dst_idx, dst_param_key = key[:4]

        if isinstance(routing_info, dict):
            src_idx = routing_info.get("src_idx", src_idx)
            src_param_key = routing_info.get("src_param_key", src_param_key)
            dst_idx = routing_info.get("dst_idx", dst_idx)
            dst_param_key = routing_info.get("dst_param_key", dst_param_key)

        try:
            src_idx = int(src_idx)
            dst_idx = int(dst_idx)
        except (TypeError, ValueError):
            continue

        if src_idx not in old_to_new_idx or dst_idx not in old_to_new_idx:
            continue
        if src_param_key is None or dst_param_key is None:
            continue

        new_src_idx = old_to_new_idx[src_idx]
        new_dst_idx = old_to_new_idx[dst_idx]
        if new_src_idx >= new_dst_idx:
            continue

        new_key = (new_src_idx, src_param_key, new_dst_idx, dst_param_key)
        new_info = routing_info.copy() if isinstance(routing_info, dict) else {}
        new_info["src_idx"] = new_src_idx
        new_info["src_param_key"] = src_param_key
        new_info["dst_idx"] = new_dst_idx
        new_info["dst_param_key"] = dst_param_key
        remapped_routing[new_key] = new_info

    return remapped_routing


def filter_manual_routing_lines(routing_lines: Mapping[Any, Any]) -> Dict[Any, Any]:
    """Return only manual routing entries, excluding auto-created ones."""
    manual_routing: Dict[Any, Any] = {}
    for key, routing_info in routing_lines.items():
        if isinstance(routing_info, dict) and routing_info.get("auto_created"):
            continue
        manual_routing[key] = routing_info
    return manual_routing
