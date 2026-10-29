#!/usr/bin/env python3
"""Validate the propagation cascade against Contagion_Propagation.csv."""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import PlantedNetworkGenerator
from propagation import PropagationSimulator

print("="*70)
print("VALIDATION: Propagation Cascade vs. Contagion_Propagation.csv")
print("="*70)
print()

# Load expected cascade
csv_path = 'data/Contagion_Propagation.csv'
print(f"Loading expected cascade from {csv_path}...")
df_expected = pd.read_csv(csv_path, index_col='Period')
print(f"  Shape: {df_expected.shape} (14 periods, 20 PRTs)")
print()

# Generate cascade from fixed DGP
print("Generating cascade from DGP...")
gen = PlantedNetworkGenerator(num_prts=20, edge_file='data/srtg20_edges.csv')
W_true = gen.generate()

sim = PropagationSimulator(W_true)
tech_idx = sim.prt_to_idx.get('Technology_Risk', 19)

distress_path, _ = sim.propagate_shock(
    initial_shock_node_idx=tech_idx,
    shock_magnitude=0.30,
    max_periods=13
)

print(f"  Cascade generated: shape={distress_path.shape}")
print()

# Extract PRT names from the propagation simulator
prt_names = list(sim.prt_to_idx.keys())

# Compare
print("Detailed comparison of first 6 periods:")
print()

for t in range(min(6, len(distress_path), len(df_expected))):
    print(f"Period {t}:")
    period_error = 0.0
    num_nonzero = 0
    for prt_name in prt_names:
        if prt_name in df_expected.columns:
            generated_val = distress_path[t, sim.prt_to_idx[prt_name]]
            csv_val = df_expected.loc[t, prt_name]
            if generated_val > 1e-6 or csv_val > 1e-6:
                abs_err = abs(generated_val - csv_val)
                num_nonzero += 1
                match = "[OK]" if abs_err < 1e-4 else "[!]"
                print(f"  {match} {prt_name:30s}: gen={generated_val:.8f}, exp={csv_val:.8f}, err={abs_err:.8f}")
                period_error += abs_err
    print(f"  Period total error: {period_error:.8f} ({num_nonzero} non-zero values)")
    print()

print()

# Overall check: sum of absolute errors
total_error = 0.0
for t in range(min(len(distress_path), len(df_expected))):
    for prt_name in prt_names:
        if prt_name in df_expected.columns:
            generated_val = distress_path[t, sim.prt_to_idx[prt_name]]
            csv_val = df_expected.loc[t, prt_name]
            total_error += abs(generated_val - csv_val)

mean_error = total_error / (len(distress_path) * len(prt_names))
print(f"Mean absolute error across all periods and PRTs: {mean_error:.8f}")

if mean_error < 1e-4:
    print("[OK] PASS: Cascade matches expected values")
else:
    print("[!] FAIL: Cascade does NOT match expected values")
    print("  The propagation rule may need adjustment")

print()
print("="*70)
