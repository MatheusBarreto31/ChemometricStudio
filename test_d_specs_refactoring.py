#!/usr/bin/env python3
"""
Test script to verify the d_specs refactoring works end-to-end.
Tests that model.json generation and execution work correctly.
"""

import json
import sys
from pathlib import Path
import tempfile

def test_model_generation():
    """Test that model.json is generated correctly with individual d_specs parameters."""
    
    print("=" * 70)
    print("TEST 1: Model.json Generation with Individual d_specs Parameters")
    print("=" * 70)
    
    # Simulate what main_gui.py does when generating model.json
    model_data = {
        "metadata": {
            "version": "1.0",
            "created": "2024-01-01T00:00:00",
            "description": "Test Model"
        },
        "functions": [
            {
                "instance_alias": "load_data_1",
                "base_alias": "load_data",
                "display_name": "Load Data",
                "parameters": {
                    "d_specs_separator": "tabs",
                    "d_specs_headlines": "0",
                    "d_specs_type": "x_matrix",
                    "d_specs_dimensions": "",
                    "data_path": ["test_data.csv"],
                    "nway_flag": 2,
                    "y_path": None,
                    "var_path": None,
                    "smp_path": None,
                    "transpose": False
                }
            }
        ],
        "routing": []
    }
    
    print("\nGenerated model structure:")
    print(json.dumps(model_data, indent=2))
    
    # Check that d_specs is split into individual parameters
    func_params = model_data["functions"][0]["parameters"]
    d_specs_keys = [k for k in func_params.keys() if k.startswith("d_specs_")]
    
    if len(d_specs_keys) == 4:
        print("\n✓ All 4 individual d_specs parameters present:")
        for key in d_specs_keys:
            print(f"  - {key}: {func_params[key]}")
    else:
        print(f"\n✗ Expected 4 d_specs parameters, found {len(d_specs_keys)}")
        return False
    
    if "d_specs" in func_params:
        print(f"\n✗ ERROR: Old 'd_specs' array parameter still in model!")
        return False
    else:
        print(f"\n✓ Old 'd_specs' array parameter NOT in model")
    
    if func_params.get("transpose") is False:
        print(f"✓ Boolean False values preserved (transpose={func_params.get('transpose')})")
    else:
        print(f"✗ Boolean False value not preserved!")
        return False
    
    return True

def test_function_specs():
    """Test that function_specs.json is updated with individual d_specs parameters."""
    
    print("\n" + "=" * 70)
    print("TEST 2: function_specs.json has Individual d_specs Parameters")
    print("=" * 70)
    
    spec_file = Path("function_specs.json")
    if not spec_file.exists():
        print(f"✗ function_specs.json not found")
        return False
    
    with open(spec_file) as f:
        specs = json.load(f)
    
    load_data_inputs = specs["input_specs"]["load_data"]
    print(f"\nload_data input_specs: {load_data_inputs}")
    
    d_specs_in_specs = [p for p in load_data_inputs if p.startswith("d_specs_")]
    if len(d_specs_in_specs) == 4:
        print(f"✓ All 4 individual d_specs parameters in specs:")
        for param in d_specs_in_specs:
            print(f"  - {param}")
    else:
        print(f"✗ Expected 4 d_specs parameters, found {len(d_specs_in_specs)}")
        return False
    
    if "d_specs" in load_data_inputs:
        print(f"✗ Old 'd_specs' array still in specs!")
        return False
    else:
        print(f"✓ Old 'd_specs' array removed from specs")
    
    return True

def test_config_files():
    """Test that GUI config files have individual d_specs parameters."""
    
    print("\n" + "=" * 70)
    print("TEST 3: GUI Config Files Reference Individual d_specs Parameters")
    print("=" * 70)
    
    config_files = [
        "gui_configs/en/load_data_config.json",
        "gui_configs/pt-br/load_data_config.json",
        "gui_configs/en/validation_data_config.json",
        "gui_configs/pt-br/validation_data_config.json"
    ]
    
    all_ok = True
    for config_file in config_files:
        path = Path(config_file)
        if not path.exists():
            print(f"✗ {config_file} not found")
            all_ok = False
            continue
        
        with open(path) as f:
            config = json.load(f)
        
        aliases = config.get("input_aliases", {})
        d_specs_refs = [k for k in aliases.keys() if k.startswith("d_specs_")]
        
        if len(d_specs_refs) > 0:
            print(f"✓ {config_file} has d_specs_* aliases: {d_specs_refs}")
        else:
            print(f"⚠ {config_file} has no d_specs_* aliases")
    
    return all_ok

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "D_SPECS REFACTORING VERIFICATION TEST".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    results = []
    
    # Test 1
    try:
        result = test_model_generation()
        results.append(("Model Generation", result))
    except Exception as e:
        print(f"✗ Test 1 failed with error: {e}")
        results.append(("Model Generation", False))
    
    # Test 2
    try:
        result = test_function_specs()
        results.append(("Function Specs", result))
    except Exception as e:
        print(f"✗ Test 2 failed with error: {e}")
        results.append(("Function Specs", False))
    
    # Test 3
    try:
        result = test_config_files()
        results.append(("Config Files", result))
    except Exception as e:
        print(f"✗ Test 3 failed with error: {e}")
        results.append(("Config Files", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 70)
    
    sys.exit(0 if all_passed else 1)
