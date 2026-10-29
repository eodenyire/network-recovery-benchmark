#!/usr/bin/env python3
"""Diagnostic test for VAR+DY recovery on single cell."""

import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.synthetic_generator import DataGenerator
from src.recovery_methods import VARDieboldYilmaz
from src.evaluation import RecoveryEvaluator

print("="*80)
print("DIAGNOSTIC: VAR+DY on Easy Cell (T=250, noise=0.1)")
print("="*80)
print()

# Generate one replicate
gen = DataGenerator("config.yaml")
experiment = gen.generate_experiment(
    time_steps=250,
    noise_level=0.1,
    network_density=2,
    t_df=5.0
)

W_true = experiment['W_true']
kri_panel = experiment['kri_panel']

print(f"Data generated:")
print(f"  W_true shape: {W_true.shape}")
print(f"  W_true density: {(W_true > 0).sum() / W_true.size:.4f} ({(W_true > 0).sum()} edges)")
print(f"  kri_panel shape: {kri_panel.shape}")
print()

# Check A1 matrix properties
print("Checking VAR lag matrix construction in synthetic_generator...")

from src.synthetic_generator import TCopulaVARProcess

kri_gen = TCopulaVARProcess(
    num_prts=20,
    kris_per_prt=5,
    time_steps=250,
    t_df=5.0,
    contagion_strength=0.4
)

W_kri = np.kron(W_true, np.ones((5, 5)))
phi = np.random.uniform(0.5, 0.7, 100)

lambda_lag = kri_gen.contagion_strength * 0.375
A1 = np.diag(phi) + lambda_lag * W_kri

print(f"  A1 shape: {A1.shape}")
print(f"  lambda_lag: {lambda_lag:.4f}")
print(f"  Diagonal persistence (phi): min={phi.min():.4f}, max={phi.max():.4f}")
print(f"  Off-diagonal (lag-matrix) contribution: max={np.abs(A1 - np.diag(np.diag(A1))).max():.4f}")

eigvals = np.linalg.eigvals(A1)
max_eig = np.abs(eigvals).max()
print(f"  Spectral radius: {max_eig:.4f} (stationary if < 1.0)")
print()

# Fit VAR+DY
print("Fitting VAR+DY on KRI panel...")
var_dy = VARDieboldYilmaz(lag_order=1)

# Fit and get the internal A_1 matrix
A_1, Sigma = var_dy._fit_var(kri_panel)
print(f"  Fitted A_1 shape: {A_1.shape}")
print(f"  Fitted A_1 diagonal mean: {np.diag(A_1).mean():.4f}")
print(f"  Fitted A_1 off-diagonal max: {np.abs(A_1 - np.diag(np.diag(A_1))).max():.4f}")

# Check if variance decomposition can extract signal
print()
print("Computing variance decomposition (Theta)...")
Theta = var_dy._variance_decomposition(A_1, Sigma, horizon=10)
print(f"  Theta (spillover matrix) shape: {Theta.shape}")
print(f"  Theta mean: {Theta.mean():.6f}")
print(f"  Theta std: {Theta.std():.6f}")
print(f"  Theta min/max: {Theta.min():.6f} / {Theta.max():.6f}")
print(f"  Theta sparsity: {(Theta > 0.1).sum() / Theta.size:.4f}")
print()

# Check correlation between Theta and A_1 off-diagonal
off_diag_A1 = A_1 - np.diag(np.diag(A_1))
off_diag_Theta = Theta - np.diag(np.diag(Theta))

print("Checking signal recovery:")
print(f"  A_1 (fitted) off-diagonal: mean={off_diag_A1.mean():.6f}, std={off_diag_A1.std():.6f}")
print(f"  Theta off-diagonal: mean={off_diag_Theta.mean():.6f}, std={off_diag_Theta.std():.6f}")

# Correlation
valid_mask = ~(np.isnan(off_diag_A1) | np.isnan(off_diag_Theta))
if valid_mask.sum() > 0:
    corr = np.corrcoef(off_diag_A1[valid_mask].flat, off_diag_Theta[valid_mask].flat)[0, 1]
    print(f"  Correlation (A_1 off-diag vs Theta off-diag): {corr:.6f}")
else:
    print(f"  Correlation: NaN (no valid data)")
print()

# Now aggregate to PRT level and evaluate
print("Aggregating to PRT level and evaluating...")
W_est_kri = var_dy.fit_and_recover(kri_panel, threshold=0.1)
print(f"  W_est (KRI-level) shape: {W_est_kri.shape}")
print(f"  W_est (KRI-level) nonzero: {(W_est_kri > 0).sum()}")

# Aggregate
evaluator = RecoveryEvaluator(W_true)
W_est_prt = W_est_kri.copy()

if W_est_prt.shape[0] == 100:
    print(f"  Aggregating 100x100 -> 20x20...")
    W_est_aggregated = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            W_est_aggregated[i, j] = W_est_prt[i*5:(i+1)*5, j*5:(j+1)*5].sum()
    
    row_sums = W_est_aggregated.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_est_prt = W_est_aggregated / row_sums
    print(f"  W_est (PRT-level after aggregation) shape: {W_est_prt.shape}")
    print(f"  W_est (PRT-level) nonzero: {(W_est_prt > 0).sum()}")
    print(f"  W_est (PRT-level) range: [{W_est_prt.min():.6f}, {W_est_prt.max():.6f}]")

# Evaluate
results = evaluator.evaluate(W_est_prt)
print()
print("Evaluation results:")
for metric, value in results.items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.6f}")

print()
print("="*80)
