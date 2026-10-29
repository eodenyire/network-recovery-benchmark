#!/usr/bin/env python3
"""Test just the propagation cascade on W_true (no data generation)."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import PlantedNetworkGenerator

print("="*70)
print("TEST: Propagation Cascade on W_true")
print("="*70)
print()

# Load W_true
gen = PlantedNetworkGenerator(num_prts=20, edge_file='data/srtg20_edges.csv')
W_true = gen.generate()

print(f"W_true loaded: {(W_true > 0).sum()} edges")
print()

# Try to import propagation module
try:
    from propagation import PropagationSimulator
    print("PropagationSimulator imported successfully")
except ImportError as e:
    print(f"ERROR importing PropagationSimulator: {e}")
    print("Checking if src/propagation.py exists...")
    if os.path.exists('src/propagation.py'):
        print("  File exists. Import error is in the module itself.")
    else:
        print("  File does NOT exist. Need to create it.")
    sys.exit(1)

print()

# Initialize propagation simulator
try:
    print("Initializing PropagationSimulator...")
    sim = PropagationSimulator(W_true)
    print(f"  Simulator initialized. PRTs: {len(sim.prt_to_idx)}")
except Exception as e:
    print(f"ERROR initializing simulator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Run propagation
try:
    print("Running propagation: Technology_Risk +0.30 shock...")
    tech_idx = sim.prt_to_idx.get('Technology_Risk', 0)
    print(f"  Technology_Risk index: {tech_idx}")
    
    distress_path, _ = sim.propagate_shock(
        initial_shock_node_idx=tech_idx,
        shock_magnitude=0.30,
        max_periods=13
    )
    
    print(f"  Propagation complete: {len(distress_path)} periods")
    print()
    print("Cascade (first 5 periods):")
    for t in range(min(5, len(distress_path))):
        max_distress = distress_path[t].max()
        print(f"  Period {t}: max distress = {max_distress:.6f}")
    
    if len(distress_path) > 5:
        print(f"  ...")
        print(f"  Period 12: max distress = {distress_path[12].max():.6f}")
    
except Exception as e:
    print(f"ERROR in propagation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*70)
print("[OK] Propagation test complete")
print("="*70)
