from typing import Optional, Callable, Dict, Any, List, Tuple
from time import perf_counter

def analyst_main(
    stop_at_function_idx: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    return_timing: bool = False
):
    """
    Execute the model configuration from model.json.
    
    Args:
        stop_at_function_idx: If provided, stop execution after this function index (0-based).
                            None means execute all functions.
    """
    print("Analyst mode selected.")
    
    global nway_flag

    import ast
    import copy
    import importlib
    import json
    import re
    from datetime import datetime
    
    # Load model configuration from model.json
    with open('model.json', 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    # Load function specs from external JSON file
    with open('function_specs.json', 'r', encoding='utf-8') as f:
        specs_data = json.load(f)
    
    return_specs = specs_data['return_specs']
    input_specs = specs_data['input_specs']
    
    # Convert import_map tuples back from list format
    import_map = {}
    for func_name, import_info in specs_data['import_map'].items():
        import_map[func_name] = tuple(import_info)
    
    workflow_control_aliases = {
        "workflow_loop_start",
        "workflow_loop_end",
        "workflow_parallel_start",
        "workflow_parallel_branch",
        "workflow_parallel_end"
    }

    # Extract function information from model
    functions_info = {}  # {instance_alias: {base_alias, parameters, parameter_types}}
    functions_list = []
    for func_entry in model_data.get('functions', []):
        instance_alias = func_entry.get('instance_alias', '')
        base_alias = func_entry.get('base_alias', '')
        params = func_entry.get('parameters', {}).copy()
        param_types = func_entry.get('parameter_types', {})
        
        functions_info[instance_alias] = {
            'base_alias': base_alias,
            'parameters': params,
            'parameter_types': param_types
        }
        functions_list.append({
            'model_idx': len(functions_list),
            'instance_alias': instance_alias,
            'base_alias': base_alias
        })
    
    # Extract routing information from model
    routing_map = {}  # {dst_instance: {dst_param: [source_mappings...]}}
    for route_entry in model_data.get('routing', []):
        src_info = route_entry.get('source', {})
        dst_info = route_entry.get('destination', {})
        
        src_alias = src_info.get('instance_alias', '')
        dst_alias = dst_info.get('instance_alias', '')
        src_param = src_info.get('param_key', '')
        src_nested_key = src_info.get('nested_key', '')
        dst_param = dst_info.get('param_key', '')
        
        if dst_alias not in routing_map:
            routing_map[dst_alias] = {}
        if dst_param not in routing_map[dst_alias]:
            routing_map[dst_alias][dst_param] = []
        routing_map[dst_alias][dst_param].append({
            'src_alias': src_alias,
            'src_param': src_param,
            'src_nested_key': src_nested_key
        })
    
    # Collect unique functions and import them
    unique_funcs = set()
    for instance_alias, info in functions_info.items():
        base_alias = info['base_alias']
        unique_funcs.add(base_alias)
    
    for func in unique_funcs:
        if func in workflow_control_aliases:
            continue
        if func in import_map:
            module_name, attr_name = import_map[func]
            module = importlib.import_module(module_name)
            globals()[func] = getattr(module, attr_name)
    
    def _convert_param_types(base_alias, param_name, value, type_info):
        """Convert a parameter value to its specified type."""
        if value is None:
            return None
        
        param_type = type_info.get(param_name, "str")
        
        if param_type == "int":
            if isinstance(value, str):
                return int(value)
            elif isinstance(value, (int, float)):
                return int(value)
            else:
                return value
        elif param_type == "float":
            if isinstance(value, str):
                return float(value)
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return value
        elif param_type == "bool":
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                return value.lower() in ('true', '1', 'yes')
            else:
                return bool(value)
        elif param_type == "list":
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                try:
                    parsed = ast.literal_eval(value)
                    return parsed if isinstance(parsed, list) else [parsed]
                except (ValueError, SyntaxError):
                    return [value]
            else:
                return [value] if value is not None else None
        else:
            # Default to string or passthrough
            return value
    
    outputs = {}  # {instance_alias: {param_key: value}}

    def _extract_nested(value: Any, nested_key: str):
        if not nested_key:
            return value, True
        current = value
        for part in str(nested_key).split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None, False
        return current, True

    def _resolve_routed_value(src_alias: str, src_param: str, nested_key: str, current_outputs: Dict[str, Dict[str, Any]]):
        if src_alias not in current_outputs:
            return None, False
        if src_param not in current_outputs[src_alias]:
            return None, False
        extracted, ok = _extract_nested(current_outputs[src_alias][src_param], nested_key)
        return extracted, ok

    def _find_matching_end(start_idx: int, start_alias: str, end_alias: str) -> int:
        depth = 0
        for idx in range(start_idx, len(functions_list)):
            base_alias = functions_list[idx]['base_alias']
            if base_alias == start_alias:
                depth += 1
            elif base_alias == end_alias:
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    def _parse_sweep_values(raw_value: Any) -> List[Any]:
        def _to_number(text: str):
            text = text.strip()
            if text == "":
                raise ValueError("empty")
            num = float(text)
            if num.is_integer():
                return int(num)
            return num

        def _expand_colon_interval(token: str):
            parts = [part.strip() for part in token.split(':')]
            if len(parts) not in (2, 3):
                return None
            try:
                start = float(parts[0])
                if len(parts) == 2:
                    end = float(parts[1])
                    step = 1.0 if end >= start else -1.0
                else:
                    step = float(parts[1])
                    end = float(parts[2])
                if step == 0:
                    return None

                values = []
                epsilon = abs(step) * 1e-9 + 1e-12
                current = start
                if step > 0:
                    while current <= end + epsilon:
                        values.append(int(current) if float(current).is_integer() else current)
                        current += step
                else:
                    while current >= end - epsilon:
                        values.append(int(current) if float(current).is_integer() else current)
                        current += step
                return values
            except Exception:
                return None

        def _expand_dash_interval(token: str):
            match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*", token)
            if not match:
                return None
            try:
                start = float(match.group(1))
                end = float(match.group(2))
                step = 1.0 if end >= start else -1.0
                values = []
                epsilon = 1e-9
                current = start
                if step > 0:
                    while current <= end + epsilon:
                        values.append(int(current) if float(current).is_integer() else current)
                        current += step
                else:
                    while current >= end - epsilon:
                        values.append(int(current) if float(current).is_integer() else current)
                        current += step
                return values
            except Exception:
                return None

        if raw_value is None:
            return []
        if isinstance(raw_value, list):
            parsed = []
            for value in raw_value:
                text_value = str(value).strip()
                if not text_value:
                    continue
                colon_values = _expand_colon_interval(text_value)
                if colon_values is not None:
                    parsed.extend(colon_values)
                    continue
                dash_values = _expand_dash_interval(text_value)
                if dash_values is not None:
                    parsed.extend(dash_values)
                    continue
                try:
                    parsed.append(_to_number(text_value))
                except Exception:
                    parsed.append(text_value)
            return parsed
        text = str(raw_value).strip()
        if not text:
            return []

        parsed_values: List[Any] = []
        for part in [segment.strip() for segment in text.split(',') if segment.strip()]:
            colon_values = _expand_colon_interval(part)
            if colon_values is not None:
                parsed_values.extend(colon_values)
                continue

            dash_values = _expand_dash_interval(part)
            if dash_values is not None:
                parsed_values.extend(dash_values)
                continue

            try:
                parsed_values.append(_to_number(part))
            except Exception:
                parsed_values.append(part)

        return parsed_values

    executed_steps = 0
    
    total_functions = len(functions_list)
    if stop_at_function_idx is not None:
        progress_total = min(total_functions, max(0, stop_at_function_idx + 1))
    else:
        progress_total = total_functions

    if progress_callback:
        try:
            progress_callback(0, progress_total, "", "")
        except Exception:
            pass

    function_timings = []
    execution_history_by_instance: Dict[str, List[Dict[str, Any]]] = {}
    model_start_time = perf_counter()
    loop_stack_context: List[Dict[str, Any]] = []
    parallel_stack_context: List[Dict[str, Any]] = []
    sweep_override_stack: List[Dict[str, set]] = []
    loop_counter = 0
    parallel_counter = 0

    def _snapshot_context() -> Dict[str, Any]:
        return {
            'loop_path': [
                {
                    'loop_id': entry.get('loop_id'),
                    'iteration': entry.get('iteration'),
                    'mode': entry.get('mode'),
                    'sweep_target': entry.get('sweep_target'),
                    'sweep_value': entry.get('sweep_value')
                }
                for entry in loop_stack_context
            ],
            'parallel_path': [
                {
                    'parallel_id': entry.get('parallel_id'),
                    'branch': entry.get('branch')
                }
                for entry in parallel_stack_context
            ]
        }

    def _execute_regular_function(entry: Dict[str, Any], current_outputs: Dict[str, Dict[str, Any]]):
        nonlocal executed_steps

        instance_alias = entry['instance_alias']
        model_idx = entry['model_idx']
        if instance_alias not in functions_info:
            return

        info = functions_info[instance_alias]
        base_alias = info['base_alias']
        params = info['parameters'].copy()
        param_types = info.get('parameter_types', {})

        if stop_at_function_idx is not None and model_idx > stop_at_function_idx:
            return

        if progress_callback:
            try:
                progress_callback(min(executed_steps, progress_total), progress_total, instance_alias, base_alias)
            except Exception:
                pass

        print(f"\nProcessing: {instance_alias} ({base_alias})")
        print(f"  Parameters: {params}")

        if instance_alias in routing_map:
            locked_params = set()
            for override_entry in sweep_override_stack:
                locked_params.update(override_entry.get(instance_alias, set()))
            for dst_param, source_mappings in routing_map[instance_alias].items():
                if dst_param in locked_params:
                    continue
                for source_mapping in source_mappings:
                    src_alias = source_mapping.get('src_alias', '')
                    src_param = source_mapping.get('src_param', '')
                    src_nested_key = source_mapping.get('src_nested_key', '')
                    routed_value, found = _resolve_routed_value(src_alias, src_param, src_nested_key, current_outputs)
                    if found:
                        params[dst_param] = routed_value
                        nested_suffix = f".{src_nested_key}" if src_nested_key else ""
                        print(f"  Routed {src_alias}.{src_param}{nested_suffix} -> {dst_param}")
                        break

        converted_params = {}
        for param_name, value in params.items():
            converted_params[param_name] = _convert_param_types(base_alias, param_name, value, param_types)
        params = converted_params

        print(f"Executing: {base_alias}")
        print(f"  Final arguments: {params}")

        if base_alias in globals():
            function_start_time = perf_counter()
            result = globals()[base_alias](**params)
            function_elapsed_seconds = perf_counter() - function_start_time
            function_outputs = {}
            if base_alias in return_specs:
                return_keys = return_specs[base_alias]
                if isinstance(result, (list, tuple)):
                    function_outputs = dict(zip(return_keys, result))
                else:
                    function_outputs = {return_keys[0]: result} if return_keys else {}
                current_outputs[instance_alias] = function_outputs

            history_entry = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'execution_time': function_elapsed_seconds,
                'outputs': copy.deepcopy(function_outputs if isinstance(function_outputs, dict) else {}),
                'inputs': copy.deepcopy(params),
                'history_context': _snapshot_context()
            }
            execution_history_by_instance.setdefault(instance_alias, []).append(history_entry)

            function_timings.append({
                "instance_alias": instance_alias,
                "base_alias": base_alias,
                "execution_time": function_elapsed_seconds
            })
            print(f"{base_alias} executed successfully.")
        else:
            print(f"Function {base_alias} not found.")

        executed_steps += 1
        if progress_callback:
            try:
                progress_callback(min(executed_steps, progress_total), progress_total, instance_alias, base_alias)
            except Exception:
                pass

    def _execute_range(start_idx: int, end_idx: int, current_outputs: Dict[str, Dict[str, Any]]):
        nonlocal loop_counter, parallel_counter
        idx = start_idx
        while idx <= end_idx and idx < len(functions_list):
            entry = functions_list[idx]
            base_alias = entry['base_alias']
            model_idx = entry['model_idx']

            if stop_at_function_idx is not None and model_idx > stop_at_function_idx:
                return len(functions_list)

            if base_alias == "workflow_loop_start":
                loop_end_idx = _find_matching_end(idx, "workflow_loop_start", "workflow_loop_end")
                if loop_end_idx < 0:
                    print("Warning: Loop Start without matching Loop End. Skipping control node.")
                    idx += 1
                    continue

                loop_instance_alias = entry['instance_alias']
                loop_info = functions_info.get(loop_instance_alias, {})
                loop_params = loop_info.get('parameters', {})

                try:
                    iterations = int(loop_params.get('iterations', 1))
                except Exception:
                    iterations = 1
                iterations = max(1, iterations)

                loop_mode = str(loop_params.get('mode', 'repeat') or 'repeat')
                sweep_target = str(loop_params.get('sweep_target', '') or '').strip()
                sweep_values = _parse_sweep_values(loop_params.get('sweep_values', ''))
                if loop_mode == "sweep_choice":
                    sweep_choice_values = loop_params.get('sweep_choice_values', [])
                    if isinstance(sweep_choice_values, str):
                        sweep_choice_values = [part.strip() for part in sweep_choice_values.split(',') if part.strip()]
                    if isinstance(sweep_choice_values, list) and sweep_choice_values:
                        sweep_values = [str(v) for v in sweep_choice_values]

                if loop_mode in ("sweep_numeric", "sweep_choice") and sweep_values:
                    iterations = len(sweep_values)
                benchmark_source = str(loop_params.get('benchmark_source', '') or '').strip()
                benchmark_nested_key = str(loop_params.get('benchmark_nested_key', '') or '').strip()
                benchmark_mode = str(loop_params.get('benchmark_mode', 'min') or 'min').lower()
                use_best_iteration = bool(loop_params.get('use_best_iteration', False))

                body_start = idx + 1
                body_end = loop_end_idx - 1
                if body_start > body_end:
                    idx = loop_end_idx + 1
                    continue

                target_instance = None
                target_param = None
                original_target_value = None
                if sweep_target and "." in sweep_target:
                    target_instance, target_param = sweep_target.split('.', 1)
                    if target_instance in functions_info and target_param:
                        original_target_value = functions_info[target_instance]['parameters'].get(target_param)
                    else:
                        target_instance = None
                        target_param = None

                best_score = None
                best_outputs_snapshot = None
                loop_counter += 1
                current_loop_id = loop_counter
                loop_stack_context.append({
                    'loop_id': current_loop_id,
                    'mode': loop_mode,
                    'iteration': 0,
                    'sweep_target': f"{target_instance}.{target_param}" if target_instance and target_param else "",
                    'sweep_value': None
                })

                print(f"\nExecuting loop block ({iterations} iteration(s), mode={loop_mode})")
                for iteration in range(iterations):
                    loop_stack_context[-1]['iteration'] = iteration + 1
                    loop_stack_context[-1]['sweep_value'] = None
                    sweep_override = {}
                    if loop_mode in ("sweep_numeric", "sweep_choice") and target_instance and target_param and sweep_values:
                        sweep_value = sweep_values[min(iteration, len(sweep_values) - 1)]
                        functions_info[target_instance]['parameters'][target_param] = sweep_value
                        loop_stack_context[-1]['sweep_value'] = sweep_value
                        sweep_override = {target_instance: {target_param}}
                        print(f"  Iteration {iteration + 1}: set {target_instance}.{target_param} = {sweep_value}")

                    sweep_override_stack.append(sweep_override)
                    _execute_range(body_start, body_end, current_outputs)
                    sweep_override_stack.pop()

                    if benchmark_source and "." in benchmark_source:
                        benchmark_parts = benchmark_source.split('.')
                        if len(benchmark_parts) >= 2:
                            b_instance = benchmark_parts[0]
                            b_output = benchmark_parts[1]
                            b_nested_from_source = '.'.join(benchmark_parts[2:]) if len(benchmark_parts) > 2 else ''
                            if benchmark_nested_key:
                                b_nested = benchmark_nested_key
                            else:
                                b_nested = b_nested_from_source
                            score, found = _resolve_routed_value(b_instance, b_output, b_nested, current_outputs)
                            if found and isinstance(score, (int, float)):
                                should_update = False
                                if best_score is None:
                                    should_update = True
                                elif benchmark_mode == 'max' and score > best_score:
                                    should_update = True
                                elif benchmark_mode != 'max' and score < best_score:
                                    should_update = True
                                if should_update:
                                    best_score = score
                                    best_outputs_snapshot = copy.deepcopy(current_outputs)

                if target_instance and target_param:
                    functions_info[target_instance]['parameters'][target_param] = original_target_value

                if use_best_iteration and best_outputs_snapshot is not None:
                    current_outputs.clear()
                    current_outputs.update(best_outputs_snapshot)
                    print(f"Loop block selected best iteration with score={best_score}")

                if loop_stack_context:
                    loop_stack_context.pop()

                idx = loop_end_idx + 1
                continue

            if base_alias == "workflow_parallel_start":
                parallel_end_idx = _find_matching_end(idx, "workflow_parallel_start", "workflow_parallel_end")
                if parallel_end_idx < 0:
                    print("Warning: Parallel Start without matching Parallel End. Skipping control node.")
                    idx += 1
                    continue

                parallel_instance_alias = entry['instance_alias']
                parallel_params = functions_info.get(parallel_instance_alias, {}).get('parameters', {})
                merge_strategy = str(parallel_params.get('merge_strategy', 'merge') or 'merge')

                block_start = idx + 1
                block_end = parallel_end_idx - 1
                if block_start > block_end:
                    idx = parallel_end_idx + 1
                    continue

                branch_ranges: List[Tuple[int, int]] = []
                branch_start = block_start
                nested_parallel_depth = 0
                for branch_idx in range(block_start, block_end + 1):
                    branch_alias = functions_list[branch_idx]['base_alias']
                    if branch_alias == "workflow_parallel_start":
                        nested_parallel_depth += 1
                    elif branch_alias == "workflow_parallel_end" and nested_parallel_depth > 0:
                        nested_parallel_depth -= 1
                    elif branch_alias == "workflow_parallel_branch" and nested_parallel_depth == 0:
                        if branch_start <= branch_idx - 1:
                            branch_ranges.append((branch_start, branch_idx - 1))
                        branch_start = branch_idx + 1
                if branch_start <= block_end:
                    branch_ranges.append((branch_start, block_end))

                baseline_outputs = copy.deepcopy(current_outputs)
                branch_output_snapshots: List[Dict[str, Dict[str, Any]]] = []
                parallel_counter += 1
                current_parallel_id = parallel_counter
                parallel_stack_context.append({
                    'parallel_id': current_parallel_id,
                    'branch': 0
                })

                print(f"\nExecuting parallel block ({len(branch_ranges)} branch(es))")
                for branch_idx, (range_start, range_end) in enumerate(branch_ranges, start=1):
                    parallel_stack_context[-1]['branch'] = branch_idx
                    branch_outputs = copy.deepcopy(baseline_outputs)
                    _execute_range(range_start, range_end, branch_outputs)
                    branch_output_snapshots.append(branch_outputs)

                if merge_strategy == "keep_last":
                    final_snapshot = branch_output_snapshots[-1] if branch_output_snapshots else baseline_outputs
                    current_outputs.clear()
                    current_outputs.update(final_snapshot)
                else:
                    current_outputs.clear()
                    current_outputs.update(baseline_outputs)
                    for snapshot in branch_output_snapshots:
                        current_outputs.update(snapshot)

                if parallel_stack_context:
                    parallel_stack_context.pop()

                idx = parallel_end_idx + 1
                continue

            if base_alias in ("workflow_loop_end", "workflow_parallel_branch", "workflow_parallel_end"):
                idx += 1
                continue

            _execute_regular_function(entry, current_outputs)
            idx += 1

        return idx

    _execute_range(0, len(functions_list) - 1, outputs)

    if stop_at_function_idx is not None:
        print(f"\nStopped after function index {stop_at_function_idx}")
    
    print("\n\nFunctions executed.")
    print("Final outputs:")
    for instance_alias, out_dict in outputs.items():
        print(f"\nOutputs from {instance_alias}:")
        for name, value in out_dict.items():
            print(f"  {name}: {value}")
    
    model_elapsed_seconds = perf_counter() - model_start_time
    timing_report: Dict[str, Any] = {
        "total_execution_time": model_elapsed_seconds,
        "function_timings": function_timings,
        "execution_history_by_instance": execution_history_by_instance,
        "executed_function_count": len(function_timings),
        "partial_run": stop_at_function_idx is not None,
        "stop_at_function_idx": stop_at_function_idx
    }

    if return_timing:
        return outputs, timing_report

    return outputs  # Return outputs for analysis tab


if __name__ == "__main__":
    analyst_main()
