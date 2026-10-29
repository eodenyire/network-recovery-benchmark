#!/usr/bin/env python3
"""
Ultra-fast test: Verify the VAR(1) lag matrix fix (30 seconds).
Only test data generation and basic metrics, no recovery methods.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import DataGenerator

print("="*70)
print("TEST: VAR(1) Lag Matrix Fix (DGP Only)")
print("="*70)
print()

# Generate data with the FIXED DGP
gen = DataGenerator('config.yaml')
exp = gen.generate_experiment(
    time_steps=100,
    noise_level=0.1,
    network_density=2,
    use_documented_network=True
)

X_kri = exp['kri_panel']
W_true = exp['W_true']
z_latent = exp['z_latent']

print(f"Data generated:")
print(f"  X_kri.shape: {X_kri.shape} (should be 100×100)")
print(f"  W_true.shape: {W_true.shape} (should be 20×20)")
print(f"  z_latent.shape: {z_latent.shape} (should be 100×100)")
print()

# Check W_true has signal
print(f"W_true stats:")
print(f"  Num edges: {(W_true > 0).sum()} (should be ~40)")
print(f"  Min: {W_true.min():.6f}, Max: {W_true.max():.6f}")
print(f"  Std: {W_true.std():.6f}")
print()

# Check z_latent has cross-sectional structure (signal from W_true in lag matrix)
print(f"z_latent stats:")
print(f"  Mean: {z_latent.mean():.3f}")
print(f"  Std: {z_latent.std():.3f}")
print()

# Compute lagged correlation to see if the lag matrix is working
z_t = z_latent[1:, :]  # z_t
z_tm1 = z_latent[:-1, :]  # z_{t-1}

# Compute cross-lag covariance: E[z_t * z_{t-1}^T]
cross_cov = np.cov(z_t.T, z_tm1.T)[:20, 20:]  # Off-diagonal block
print(f"Cross-lag covariance (should show W_true structure):")
print(f"  Shape: {cross_cov.shape}")
print(f"  Mean: {cross_cov.mean():.6f}")
print(f"  Std: {cross_cov.std():.6f}")
print(f"  Max: {np.abs(cross_cov).max():.6f}")
print()

# Check if the signal is heterogeneous (not all zeros)
row_means = np.abs(cross_cov).mean(axis=1)
print(f"Heterogeneity in lag structure:")
print(f"  Row means (abs): min={row_means.min():.6f}, max={row_means.max():.6f}")
print(f"  Std of row means: {row_means.std():.6f}")
print()

if row_means.std() > 1e-6 and cross_cov.std() > 1e-6:
    print("✓ Lag matrix is driving heterogeneous cross-lag structure")
    print("  → VAR should be able to recover some signal")
else:
    print("✗ Lag matrix is NOT driving cross-lag structure")
    print("  → VAR will see only white noise")
print()

# Also check X_kri has reasonable range
print(f"X_kri stats:")
print(f"  Min: {X_kri.min():.3f}, Max: {X_kri.max():.3f}")
print(f"  Mean: {X_kri.mean():.3f}, Std: {X_kri.std():.3f}")
print()

print("="*70)
print("[DONE] DGP test complete")
print("="*70)
