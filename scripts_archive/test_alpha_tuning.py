#!/usr/bin/env python3
"""Tune alpha parameter to match expected cascade."""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import PlantedNetworkGenerator
from propagation import PropagationSimulator

print("="*70)
print("TUNING: Alpha parameter for propagation")
print("="*70)
print()

# Load expected cascade
csv_path = 'data/Contagion_Propagation.csv'
df_expected = pd.read_csv(csv_path, index_col='Period')

# Generate network
gen = PlantedNetworkGenerator(num_prts=20, edge_file='data/srtg20_edges.csv')
W_true = gen.generate()
sim = PropagationSimulator(W_true)
tech_idx = sim.prt_to_idx.get('Technology_Risk', 19)

# Test different alpha values
alphas_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for alpha in alphas_to_test:
    print(f"Testing alpha = {alpha}...")
    distress_path, _ = sim.propagate_shock(
        initial_shock_node_idx=tech_idx,
        shock_magnitude=0.30,
        max_periods=13,
        alpha=alpha
    )
    
    # Compute mean absolute error
    total_error = 0.0
    prt_names = list(sim.prt_to_idx.keys())
    for t in range(min(len(distress_path), len(df_expected))):
        for prt_name in prt_names:
            if prt_name in df_expected.columns:
                generated_val = distress_path[t, sim.prt_to_idx[prt_name]]
                csv_val = df_expected.loc[t, prt_name]
                total_error += abs(generated_val - csv_val)
    
    mean_error = total_error / (len(distress_path) * len(prt_names))
    status = "✓ BEST" if mean_error < 0.01 else "✓ GOOD" if mean_error < 0.03 else "✗"
    print(f"  Mean absolute error: {mean_error:.8f} {status}")
    
    # Show Period 2 comparison for debugging
    liquidity_idx = sim.prt_to_idx.get('Liquidity_Risk')
    if liquidity_idx is not None:
        gen_val = distress_path[2, liquidity_idx]
        exp_val = df_expected.loc[2, 'Liquidity_Risk']
        print(f"    Period 2 Liquidity_Risk: gen={gen_val:.8f}, exp={exp_val:.8f}, err={abs(gen_val - exp_val):.8f}")
    print()

print("="*70)
