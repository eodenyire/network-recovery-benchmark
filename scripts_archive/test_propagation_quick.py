#!/usr/bin/env python3
"""Quick propagation test to verify performance."""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import PlantedNetworkGenerator
from propagation import PropagationSimulator

print("Loading network...")
start = time.time()
gen = PlantedNetworkGenerator(num_prts=20, edge_file='data/srtg20_edges.csv')
W_true = gen.generate()
print(f"  Network loaded in {time.time() - start:.2f}s, shape={W_true.shape}")

print("\nInitializing propagation simulator...")
start = time.time()
sim = PropagationSimulator(W_true)
print(f"  Simulator ready in {time.time() - start:.2f}s")

print("\nPropagating shock...")
start = time.time()
tech_idx = sim.prt_to_idx.get('Technology_Risk', 19)
print(f"  Technology_Risk index: {tech_idx}")

distress_path, periods = sim.propagate_shock(
    initial_shock_node_idx=tech_idx,
    shock_magnitude=0.30,
    max_periods=13
)
elapsed = time.time() - start
print(f"  Propagation complete in {elapsed:.2f}s")
print(f"  Distress path shape: {distress_path.shape}")
print(f"  Periods to convergence: {periods}")

print("\nFirst 3 periods of Technology_Risk:")
for t in range(3):
    val = distress_path[t, tech_idx]
    print(f"  Period {t}: {val:.8f}")

print("\n✓ Test complete")
