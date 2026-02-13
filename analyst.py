from typing import Optional, Callable, Dict, Any
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
    import shlex
    import importlib
    import json
    
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
    
    # Extract function information from model
    functions_info = {}  # {instance_alias: {base_alias, parameters, parameter_types}}
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
    
    # Extract routing information from model
    routing_map = {}  # {dst_instance: {dst_param: src_instance.src_param}}
    for route_entry in model_data.get('routing', []):
        src_info = route_entry.get('source', {})
        dst_info = route_entry.get('destination', {})
        
        src_alias = src_info.get('instance_alias', '')
        dst_alias = dst_info.get('instance_alias', '')
        src_param = src_info.get('param_key', '')
        dst_param = dst_info.get('param_key', '')
        
        if dst_alias not in routing_map:
            routing_map[dst_alias] = {}
        routing_map[dst_alias][dst_param] = (src_alias, src_param)
    
    # Collect unique functions and import them
    unique_funcs = set()
    for instance_alias, info in functions_info.items():
        base_alias = info['base_alias']
        unique_funcs.add(base_alias)
    
    for func in unique_funcs:
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
    
    # Build ordered list of functions to execute
    functions_list = []  # Maintain order from model
    for func_entry in model_data.get('functions', []):
        instance_alias = func_entry.get('instance_alias', '')
        if instance_alias in functions_info:
            functions_list.append((instance_alias, functions_info[instance_alias]))
    
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
    model_start_time = perf_counter()

    # Execute each function in order
    for exec_idx, (instance_alias, info) in enumerate(functions_list):
        # Check if we should stop before this function
        if stop_at_function_idx is not None and exec_idx > stop_at_function_idx:
            print(f"\nStopping at function index {stop_at_function_idx}")
            break
        
        base_alias = info['base_alias']
        params = info['parameters'].copy()
        param_types = info.get('parameter_types', {})

        if progress_callback:
            try:
                # Function is starting now; completed count is functions before this one
                progress_callback(exec_idx, progress_total, instance_alias, base_alias)
            except Exception:
                pass
        
        print(f"\nProcessing: {instance_alias} ({base_alias})")
        print(f"  Parameters: {params}")
        
        # Apply routing: override params with values from previous outputs
        if instance_alias in routing_map:
            for dst_param, (src_alias, src_param) in routing_map[instance_alias].items():
                if src_alias in outputs and src_param in outputs[src_alias]:
                    params[dst_param] = outputs[src_alias][src_param]
                    print(f"  Routed {src_alias}.{src_param} -> {dst_param}")
        
        # Convert parameter types based on type information from model.json
        converted_params = {}
        for param_name, value in params.items():
            converted_params[param_name] = _convert_param_types(base_alias, param_name, value, param_types)
        params = converted_params
        
        print(f"Executing: {base_alias}")
        print(f"  Final arguments: {params}")
        
        # Call the function
        if base_alias in globals():
            function_start_time = perf_counter()
            result = globals()[base_alias](**params)
            function_elapsed_seconds = perf_counter() - function_start_time
            if base_alias in return_specs:
                # Store outputs under the instance alias
                return_keys = return_specs[base_alias]
                if isinstance(result, (list, tuple)):
                    outputs[instance_alias] = dict(zip(return_keys, result))
                else:
                    # Single return value
                    outputs[instance_alias] = {return_keys[0]: result} if return_keys else {}
            function_timings.append({
                "instance_alias": instance_alias,
                "base_alias": base_alias,
                "execution_time": function_elapsed_seconds
            })
            print(f"{base_alias} executed successfully.")
        else:
            print(f"Function {base_alias} not found.")

        if progress_callback:
            try:
                # Function has completed successfully
                progress_callback(exec_idx + 1, progress_total, instance_alias, base_alias)
            except Exception:
                pass
        
        # Stop after requested function
        if stop_at_function_idx is not None and exec_idx == stop_at_function_idx:
            print(f"\nStopped after function index {stop_at_function_idx}")
            break
    
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
        "executed_function_count": len(function_timings),
        "partial_run": stop_at_function_idx is not None,
        "stop_at_function_idx": stop_at_function_idx
    }

    if return_timing:
        return outputs, timing_report

    return outputs  # Return outputs for analysis tab


if __name__ == "__main__":
    analyst_main()
