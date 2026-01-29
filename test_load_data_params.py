#!/usr/bin/env python3
"""Test script to verify load_data() works with individual d_specs parameters."""

import sys
from pathlib import Path
import json

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from chemometrics.data_input import load_data

def test_load_data_signature():
    """Test that load_data() accepts individual d_specs_* parameters."""
    
    print("=" * 60)
    print("TESTING load_data() SIGNATURE WITH INDIVIDUAL d_specs PARAMS")
    print("=" * 60)
    
    # Check function signature
    import inspect
    sig = inspect.signature(load_data)
    print(f"\nFunction signature:\n  load_data{sig}")
    
    params = list(sig.parameters.keys())
    print(f"\nParameters: {params}")
    
    # Check for d_specs_* parameters
    d_specs_params = [p for p in params if p.startswith('d_specs_')]
    print(f"\nFound d_specs_* parameters: {d_specs_params}")
    
    expected = ['d_specs_separator', 'd_specs_headlines', 'd_specs_type', 'd_specs_dimensions']
    if set(d_specs_params) == set(expected):
        print(f"✓ All expected d_specs_* parameters present")
    else:
        print(f"✗ Missing parameters: {set(expected) - set(d_specs_params)}")
        print(f"  Extra parameters: {set(d_specs_params) - set(expected)}")
    
    # Check that old d_specs parameter is NOT there
    if 'd_specs' in params:
        print(f"✗ ERROR: Old 'd_specs' parameter still in signature!")
        return False
    else:
        print(f"✓ Old 'd_specs' array parameter removed")
    
    print("\n" + "=" * 60)
    print("SIGNATURE TEST PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_load_data_signature()
    sys.exit(0 if success else 1)
