#!/usr/bin/env python3
"""Debug FEVD computation to see what VAR is producing."""

import sys
import os
import numpy as np
from statsmodels.tsa.api import VAR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import DataGenerator

def debug_var_fevd():
    """Check what FEVD produces for VAR model."""
    print("="*70)
    print("Debug: VAR FEVD Computation")
    print("="*70)
    print()
    
    # Generate data
    gen = DataGenerator('config.yaml')
    exp = gen.generate_experiment(
        time_steps=250,
        noise_level=0.1,
        network_density=2,
        use_documented_network=True
    )
    
    X = exp['kri_panel']
    print(f"X.shape: {X.shape}")
    print(f"X.mean: {X.mean():.3f}, X.std: {X.std():.3f}")
    print(f"X[0,0:5]: {X[0, :5]}")
    print()
    
    # Fit VAR
    print("Fitting VAR(1)...")
    model = VAR(X)
    result = model.fit(maxlags=1, ic=None)
    
    print(f"VAR params shape: {result.params.shape}")
    print(f"Residuals shape: {result.resid.shape}")
    print(f"Residuals std: {result.resid.std(axis=0).mean():.6f}")
    print()
    
    # FEVD at horizon 10
    print("Computing FEVD at horizon=10...")
    fevd_result = result.fevd(10)
    
    # Decomp is (horizon, n_vars, n_vars)
    decomp = fevd_result.decomp  # Shape: (10, 100, 100)
    print(f"FEVD decomp shape: {decomp.shape}")
    
    # FEVD should sum to 1 across sources for each variable/horizon
    print()
    print("FEVD sanity check (should sum to 1 across axis 2):")
    decomp_sum = decomp.sum(axis=2)  # (10, 100)
    print(f"  Min sum: {decomp_sum.min():.6f}")
    print(f"  Max sum: {decomp_sum.max():.6f}")
    print(f"  Mean sum: {decomp_sum.mean():.6f}")
    print()
    
    # Directional spillover: aggregate across horizon
    print("Computing directional spillover (aggregated across horizon)...")
    spillover = decomp.mean(axis=0)  # Average across 10 periods → (100, 100)
    print(f"Spillover shape: {spillover.shape}")
    print(f"Spillover std: {spillover.std():.6f}")
    print(f"Spillover min: {spillover.min():.6f}, max: {spillover.max():.6f}")
    print()
    
    # Check for constant rows
    row_stds = spillover.std(axis=1)
    print(f"Row stds (std of spillover to each var):")
    print(f"  Min: {row_stds.min():.6f}, Max: {row_stds.max():.6f}, Mean: {row_stds.mean():.6f}")
    print(f"  Num constant rows (std < 1e-6): {(row_stds < 1e-6).sum()}")
    print()
    
    # Check what the binarized result looks like
    print("Binarized (threshold=0.1):")
    binary = (spillover >= 0.1).astype(int)
    print(f"  Num edges: {binary.sum()}")
    print(f"  Expected edges (at 20×20, 2-4 per node): 40-80")
    print()
    
    # Sample spillover values
    print("Sample spillover values (first 5×5 block):")
    print(spillover[:5, :5])

if __name__ == "__main__":
    debug_var_fevd()
