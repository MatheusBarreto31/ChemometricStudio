"""Condition evaluation helpers for analysis section/page visibility."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np


def _resolve_nested_path(source: Any, path: str) -> Tuple[bool, Any]:
    """Resolve dotted path lookups from dict/list/tuple/ndarray containers."""
    if not isinstance(path, str) or not path:
        return False, None

    if not isinstance(source, (dict, list, tuple, np.ndarray)):
        return False, None

    current = source
    for token in path.split('.'):
        if token == "":
            return False, None

        if isinstance(current, dict):
            if token not in current:
                return False, None
            current = current[token]
            continue

        if isinstance(current, (list, tuple, np.ndarray)):
            try:
                index = int(token)
            except (TypeError, ValueError):
                return False, None
            if index < 0 or index >= len(current):
                return False, None
            current = current[index]
            continue

        return False, None

    return True, current


def evaluate_condition(
    condition: Mapping[str, Any],
    *,
    combined_sources: Any,
    inputs: Any,
    outputs: Any,
    filtered_outputs: Any,
    get_setup_default_for_parameter: Optional[Callable[[Any], Tuple[bool, Any]]] = None,
) -> bool:
    """Evaluate a condition against resolved execution data sources."""
    if not condition or not isinstance(condition, dict):
        return True

    if 'all' in condition:
        sub_conditions = condition['all']
        if not isinstance(sub_conditions, list):
            return True
        return all(
            evaluate_condition(
                sub,
                combined_sources=combined_sources,
                inputs=inputs,
                outputs=outputs,
                filtered_outputs=filtered_outputs,
                get_setup_default_for_parameter=get_setup_default_for_parameter,
            )
            for sub in sub_conditions
        )

    if 'any' in condition:
        sub_conditions = condition['any']
        if not isinstance(sub_conditions, list):
            return True
        return any(
            evaluate_condition(
                sub,
                combined_sources=combined_sources,
                inputs=inputs,
                outputs=outputs,
                filtered_outputs=filtered_outputs,
                get_setup_default_for_parameter=get_setup_default_for_parameter,
            )
            for sub in sub_conditions
        )

    parameter = condition.get('parameter')
    operator = condition.get('operator', '==')
    expected_value = condition.get('value')

    if not parameter:
        return True

    actual_value = None
    resolved_value = False
    if isinstance(combined_sources, dict):
        if parameter in combined_sources:
            actual_value = combined_sources.get(parameter)
            resolved_value = True
        elif isinstance(parameter, str) and not parameter.startswith('in.') and not parameter.startswith('out.') and not parameter.startswith('pf.'):
            if f"in.{parameter}" in combined_sources:
                actual_value = combined_sources.get(f"in.{parameter}")
                resolved_value = True
            elif f"out.{parameter}" in combined_sources:
                actual_value = combined_sources.get(f"out.{parameter}")
                resolved_value = True
            elif f"pf.{parameter}" in combined_sources:
                actual_value = combined_sources.get(f"pf.{parameter}")
                resolved_value = True

    if not resolved_value and isinstance(parameter, str):
        candidate_paths = [parameter]
        if not parameter.startswith('in.') and not parameter.startswith('out.') and not parameter.startswith('pf.'):
            candidate_paths.extend([f"in.{parameter}", f"out.{parameter}", f"pf.{parameter}"])

        for candidate in candidate_paths:
            found_nested, nested_value = _resolve_nested_path(combined_sources, candidate)
            if found_nested:
                actual_value = nested_value
                resolved_value = True
                break

            if candidate.startswith('in.'):
                found_nested, nested_value = _resolve_nested_path(inputs, candidate[3:])
                if found_nested:
                    actual_value = nested_value
                    resolved_value = True
                    break
            elif candidate.startswith('out.'):
                found_nested, nested_value = _resolve_nested_path(filtered_outputs, candidate[4:])
                if found_nested:
                    actual_value = nested_value
                    resolved_value = True
                    break
            elif candidate.startswith('pf.'):
                found_nested, nested_value = _resolve_nested_path(outputs, candidate[3:])
                if found_nested:
                    actual_value = nested_value
                    resolved_value = True
                    break
            else:
                found_nested, nested_value = _resolve_nested_path(inputs, candidate)
                if found_nested:
                    actual_value = nested_value
                    resolved_value = True
                    break
                found_nested, nested_value = _resolve_nested_path(filtered_outputs, candidate)
                if found_nested:
                    actual_value = nested_value
                    resolved_value = True
                    break

    if actual_value is None and not resolved_value:
        if isinstance(inputs, dict) and parameter in inputs:
            actual_value = inputs.get(parameter)
            resolved_value = True
        elif isinstance(filtered_outputs, dict) and parameter in filtered_outputs:
            actual_value = filtered_outputs.get(parameter)
            resolved_value = True

    if operator == 'exists':
        if resolved_value:
            return actual_value is not None

        if isinstance(combined_sources, dict):
            if parameter in combined_sources:
                return combined_sources.get(parameter) is not None
            if isinstance(parameter, str) and not parameter.startswith('in.') and not parameter.startswith('out.') and not parameter.startswith('pf.'):
                in_key = f"in.{parameter}"
                out_key = f"out.{parameter}"
                pf_key = f"pf.{parameter}"
                if in_key in combined_sources:
                    return combined_sources.get(in_key) is not None
                if out_key in combined_sources:
                    return combined_sources.get(out_key) is not None
                if pf_key in combined_sources:
                    return combined_sources.get(pf_key) is not None

        if isinstance(parameter, str):
            nested_candidates = [parameter]
            if not parameter.startswith('in.') and not parameter.startswith('out.') and not parameter.startswith('pf.'):
                nested_candidates.extend([f"in.{parameter}", f"out.{parameter}", f"pf.{parameter}"])

            for candidate in nested_candidates:
                found_nested, nested_value = _resolve_nested_path(combined_sources, candidate)
                if found_nested:
                    return nested_value is not None

                if candidate.startswith('in.'):
                    found_nested, nested_value = _resolve_nested_path(inputs, candidate[3:])
                    if found_nested:
                        return nested_value is not None
                elif candidate.startswith('out.'):
                    found_nested, nested_value = _resolve_nested_path(filtered_outputs, candidate[4:])
                    if found_nested:
                        return nested_value is not None
                elif candidate.startswith('pf.'):
                    found_nested, nested_value = _resolve_nested_path(outputs, candidate[3:])
                    if found_nested:
                        return nested_value is not None
                else:
                    found_nested, nested_value = _resolve_nested_path(inputs, candidate)
                    if found_nested:
                        return nested_value is not None
                    found_nested, nested_value = _resolve_nested_path(filtered_outputs, candidate)
                    if found_nested:
                        return nested_value is not None

        in_exists = isinstance(inputs, dict) and parameter in inputs and inputs.get(parameter) is not None
        out_exists = isinstance(filtered_outputs, dict) and parameter in filtered_outputs and filtered_outputs.get(parameter) is not None
        return in_exists or out_exists

    if operator == 'not_exists':
        return not evaluate_condition(
            {'parameter': parameter, 'operator': 'exists', 'value': expected_value},
            combined_sources=combined_sources,
            inputs=inputs,
            outputs=outputs,
            filtered_outputs=filtered_outputs,
            get_setup_default_for_parameter=get_setup_default_for_parameter,
        )

    if not resolved_value and callable(get_setup_default_for_parameter):
        has_default, default_value = get_setup_default_for_parameter(parameter)
        if has_default:
            actual_value = default_value
            resolved_value = True

    if not resolved_value:
        return True

    try:
        if operator in ['>', '<', '>=', '<=']:
            try:
                actual_value = int(actual_value) if isinstance(actual_value, str) else actual_value
                expected_value = int(expected_value) if isinstance(expected_value, str) else expected_value
            except (ValueError, TypeError):
                actual_value = str(actual_value)
                expected_value = str(expected_value)
        else:
            actual_value = str(actual_value)
            expected_value = str(expected_value)
    except Exception:
        pass

    try:
        if operator == '==':
            return actual_value == expected_value
        if operator == '!=':
            return actual_value != expected_value
        if operator == '>':
            return actual_value > expected_value
        if operator == '<':
            return actual_value < expected_value
        if operator == '>=':
            return actual_value >= expected_value
        if operator == '<=':
            return actual_value <= expected_value
        if operator == 'in':
            return actual_value in expected_value
        if operator == 'contains':
            return expected_value in actual_value
        return True
    except (TypeError, ValueError):
        return False
