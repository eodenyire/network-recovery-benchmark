#!/usr/bin/env python3
"""Ultra-quick DGP check: just load the network and check stationarity (5 seconds)."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import PlantedNetworkGenerator, TCopulaVARProcess

print("="*70)
print("QUICK DGP CHECK: Network Load and Stationarity")
print("="*70)
print()

# Load network
print("Loading W_true from data/srtg20_edges.csv...")
gen = PlantedNetworkGenerator(num_prts=20, edge_file='data/srtg20_edges.csv')
W_true = gen.generate()

print(f"W_true loaded: shape={W_true.shape}, edges={(W_true > 0).sum()}")
print(f"W_true min={W_true.min():.6f}, max={W_true.max():.6f}, std={W_true.std():.6f}")
print()

# Construct lag matrix (simplified version of what TCopulaVARProcess does)
print("Constructing lag matrix Φ(W_true)...")

num_prts = 20
kris_per_prt = 5
num_kris = 100

# Expand to KRI level
W_kri = np.kron(W_true, np.ones((kris_per_prt, kris_per_prt)))
print(f"W_kri expanded: shape={W_kri.shape}")

# Diagonal persistence
phi = np.ones(num_kris) * 0.75  # Fixed for quick test
print(f"Persistence φ: mean={phi.mean():.3f}, shape={phi.shape}")

# VAR(1) lag matrix
lambda_lag = 0.15  # transmission strength
A1 = np.diag(phi) + lambda_lag * W_kri

print(f"A1 = diag(φ) + λ*W_kri constructed: shape={A1.shape}")
print()

# Check stationarity
print("Checking stationarity...")
eigvals = np.linalg.eigvals(A1)
max_eig = np.abs(eigvals).max()
print(f"Max eigenvalue: {max_eig:.6f}")
print(f"Stationary (max_eig < 1)? {max_eig < 1}")

if max_eig >= 1.0:
    print("NOT stationary. Rescaling...")
    A1_rescaled = A1 / (max_eig * 1.05)
    eigvals_rescaled = np.linalg.eigvals(A1_rescaled)
    max_eig_rescaled = np.abs(eigvals_rescaled).max()
    print(f"After rescale: max_eig = {max_eig_rescaled:.6f}")

print()
print("="*70)
print("[OK] DGP check complete. No hang, stationarity OK.")
print("="*70)
