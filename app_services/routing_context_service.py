"""Workflow routing-context helpers.

Pure helpers for loop/parallel/ensemble scope compatibility checks.
"""

from __future__ import annotations

from typing import Sequence, Set, Tuple


ScopeSignature = Tuple[tuple, ...]


def get_workflow_scope_signature(
    function_base_aliases: Sequence[str],
    target_idx: int,
) -> ScopeSignature:
    """Return active workflow scope signature before target index."""
    active_stack = []
    loop_counter = 0
    parallel_counter = 0
    ensemble_counter = 0

    for idx in range(max(0, target_idx)):
        base_alias = function_base_aliases[idx] if idx < len(function_base_aliases) else ""
        if base_alias == "workflow_loop_start":
            loop_counter += 1
            active_stack.append(("loop", loop_counter))
        elif base_alias == "workflow_loop_end":
            for stack_idx in range(len(active_stack) - 1, -1, -1):
                if active_stack[stack_idx][0] == "loop":
                    active_stack.pop(stack_idx)
                    break
        elif base_alias == "workflow_parallel_start":
            parallel_counter += 1
            active_stack.append(("parallel", parallel_counter, 1))
        elif base_alias == "workflow_parallel_branch":
            for stack_idx in range(len(active_stack) - 1, -1, -1):
                if active_stack[stack_idx][0] == "parallel":
                    p_type, p_id, p_branch = active_stack[stack_idx]
                    active_stack[stack_idx] = (p_type, p_id, p_branch + 1)
                    break
        elif base_alias == "workflow_parallel_end":
            for stack_idx in range(len(active_stack) - 1, -1, -1):
                if active_stack[stack_idx][0] == "parallel":
                    active_stack.pop(stack_idx)
                    break
        elif base_alias == "workflow_ensemble_start":
            ensemble_counter += 1
            active_stack.append(("ensemble", ensemble_counter, 1))
        elif base_alias == "workflow_ensemble_member":
            for stack_idx in range(len(active_stack) - 1, -1, -1):
                if active_stack[stack_idx][0] == "ensemble":
                    e_type, e_id, e_member = active_stack[stack_idx]
                    active_stack[stack_idx] = (e_type, e_id, e_member + 1)
                    break
        elif base_alias == "workflow_ensemble_end":
            for stack_idx in range(len(active_stack) - 1, -1, -1):
                if active_stack[stack_idx][0] == "ensemble":
                    active_stack.pop(stack_idx)
                    break

    return tuple(active_stack)


def can_auto_route_between(
    function_base_aliases: Sequence[str],
    workflow_control_aliases: Set[str],
    src_idx: int,
    dst_idx: int,
) -> bool:
    """Return whether auto-routing is structurally valid from src to dst."""
    if src_idx < 0 or dst_idx < 0 or src_idx >= len(function_base_aliases) or dst_idx >= len(function_base_aliases):
        return False

    src_base = function_base_aliases[src_idx]
    dst_base = function_base_aliases[dst_idx]
    if src_base in workflow_control_aliases or dst_base in workflow_control_aliases:
        return False

    src_scope = get_workflow_scope_signature(function_base_aliases, src_idx)
    dst_scope = get_workflow_scope_signature(function_base_aliases, dst_idx)

    if src_scope == dst_scope:
        return True

    # Allow routing from outer scope into nested scope.
    if len(src_scope) <= len(dst_scope) and dst_scope[:len(src_scope)] == src_scope:
        return True

    # Disallow sibling/cross-branch routing.
    return False
