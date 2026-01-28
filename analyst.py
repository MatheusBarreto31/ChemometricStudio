
def analyst_main():
    print("Analyst mode selected.")
    
    global nway_flag

    ### Inputs (add from GUI later)
    #nway_flag = 2  # Set this based on your data (1 for 1-way, 2+ for multi-way)
    #d_specs = ["spaces", "0", "x_matrix", ""]
    #data_path = [".venv/DataforTesting/x1.txt"]
    #y_path=None
    #var_path=None
    #smp_path=None
    #transpose=False

    import ast
    import shlex
    import importlib
    import json
    
    # Load function sequence from functions.txt
    # Format: each line "function_name arg1=value arg2=value ..."
    # e.g. load_data separator=comma num_headlines=0 data_type=x_matrix nway_flag=2 data_path=file1.txt,file2.txt
    with open('functions.txt', 'r', encoding='utf-8-sig') as f:
        function_lines = [line.strip() for line in f if line.strip()]
    
    # Load routing from routing.txt
    # Format: each line "source_func.source_output -> target_func.target_input"
    with open('routing.txt', 'r', encoding='utf-8-sig') as f:
        routings = [line.strip() for line in f if line.strip()]
    
    # Load function specs from external JSON file
    with open('function_specs.json', 'r', encoding='utf-8') as f:
        specs_data = json.load(f)
    
    return_specs = specs_data['return_specs']
    input_specs = specs_data['input_specs']
    
    # Convert import_map tuples back from list format
    import_map = {}
    for func_name, import_info in specs_data['import_map'].items():
        import_map[func_name] = tuple(import_info)
    
    # Collect unique functions and import them
    unique_funcs = set()
    for line in function_lines:
        parts = shlex.split(line)
        if parts:
            unique_funcs.add(parts[0])
    
    for func in unique_funcs:
        if func in import_map:
            module_name, attr_name = import_map[func]
            module = importlib.import_module(module_name)
            globals()[func] = getattr(module, attr_name)
    
    outputs = {}
    
    for line in function_lines:
        print(f"\nParsing line: {repr(line)}")
        # Parse function name and arguments
        # Support aliasing: "func_name as alias_name arg1=value ..."
        parts = line.split(None, 1)  # Split on first whitespace
        if not parts:
            continue
        
        func = parts[0]
        alias = func  # Default: alias is the function name itself
        
        # Check if this line uses aliasing
        if len(parts) > 1 and ' as ' in parts[1]:
            # Extract alias from "as alias_name"
            args_str = parts[1]
            as_match = re.match(r'^as\s+(\w+)\s+(.*)$', args_str)
            if as_match:
                alias = as_match.group(1)
                parts = [func, as_match.group(2)]  # Remaining args after alias
            else:
                # If format is wrong, proceed without aliasing
                alias = func
        
        kwargs = {}
        
        if len(parts) > 1:
            args_str = parts[1]
            # Parse key:value pairs, handling complex values like lists
            import re
            
            # Find all key:value pairs using a pattern that handles:
            # - Lists: d_specs:[...]
            # - Paths: data_path:/path/to/file
            # - Simple values: key:value
            # Key pattern: starts with letter/underscore, followed by word chars
            # Value pattern: everything until the next space-separated key: or end of string
            
            i = 0
            while i < len(args_str):
                # Skip whitespace
                while i < len(args_str) and args_str[i].isspace():
                    i += 1
                
                if i >= len(args_str):
                    break
                
                # Find next key: pattern
                match = re.match(r'([a-zA-Z_]\w*):', args_str[i:])
                if not match:
                    i += 1
                    continue
                
                key = match.group(1)
                value_start = i + len(key) + 1  # Position after "key:"
                i = value_start
                
                # Extract the value - continue until we hit the next key: or end
                value_chars = []
                bracket_depth = 0
                in_quotes = False
                quote_char = None
                
                while i < len(args_str):
                    ch = args_str[i]
                    
                    # Track quotes
                    if ch in ('"', "'") and (not in_quotes or quote_char == ch):
                        if in_quotes:
                            in_quotes = False
                        else:
                            in_quotes = True
                            quote_char = ch
                        value_chars.append(ch)
                        i += 1
                        continue
                    
                    # Track brackets
                    if not in_quotes:
                        if ch == '[':
                            bracket_depth += 1
                        elif ch == ']':
                            bracket_depth -= 1
                    
                    # Check if we've hit the next key (at start of new token)
                    if not in_quotes and bracket_depth == 0 and ch.isspace():
                        # Look ahead for next key:
                        remaining = args_str[i:].lstrip()
                        if re.match(r'[a-zA-Z_]\w*:', remaining):
                            break
                    
                    value_chars.append(ch)
                    i += 1
                
                value_str = ''.join(value_chars).strip()
                
                print(f"  Raw value for {key}: {repr(value_str)}")
                
                # Try to parse the value using literal_eval
                try:
                    parsed_value = ast.literal_eval(value_str)
                except (ValueError, SyntaxError) as e:
                    # If it fails, use the string as-is
                    parsed_value = value_str
                
                print(f"  Parsed value for {key}: {repr(parsed_value)}")
                kwargs[key] = parsed_value
        
        print(f"Executing: {func}")
        print(f"  Final arguments: {kwargs}")
        
        # Apply routing: override kwargs with routed values
        # Routes reference aliases or function names
        for route in routings:
            if f' -> {func}.' in route:
                src, tgt = route.split(' -> ')
                src_alias, src_out = src.split('.')
                tgt_in = tgt.split('.')[1]
                # Look up by alias (which may be the same as function name if not aliased)
                if src_alias in outputs and src_out in outputs[src_alias]:
                    kwargs[tgt_in] = outputs[src_alias][src_out]
        
        # Call the function
        if func in globals():
            result = globals()[func](**kwargs)
            if func in return_specs:
                # Store outputs under the alias (allows multiple calls to same function)
                outputs[alias] = dict(zip(return_specs[func], result))
            print(f"{func} executed successfully.")
        else:
            print(f"Function {func} not found.")
    
    print("All functions executed.")
    print("Final outputs:")
    for alias, out_dict in outputs.items():
        print(f"\nOutputs from {alias}:")
        for name, value in out_dict.items():
            print(f"{name}: {value}")


if __name__ == "__main__":
    analyst_main()
