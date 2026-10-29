#!/usr/bin/env python3
"""
Quick test: VAR+DY recovery on fixed DGP (60 seconds).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import DataGenerator
from recovery_methods import RecoveryMethodsFactory
from evaluation import RecoveryEvaluator

print("="*70)
print("TEST: VAR+DY Recovery on Fixed DGP")
print("="*70)
print()

# Generate data
gen = DataGenerator('config.yaml')
exp = gen.generate_experiment(
    time_steps=100,
    noise_level=0.1,
    network_density=2,
    use_documented_network=True
)

X_kri = exp['kri_panel']
W_true = exp['W_true']

print(f"Data: X_kri.shape={X_kri.shape}, W_true.shape={W_true.shape}")
print()

# Aggregate KRI to PRT
def aggregate_kri_to_prt(X_kri):
    T, n_kri = X_kri.shape
    assert n_kri == 100
    n_prt = 20
    X_prt = np.zeros((T, n_prt))
    for i in range(n_prt):
        X_prt[:, i] = X_kri[:, i*5:(i+1)*5].mean(axis=1)
    return X_prt

X_prt = aggregate_kri_to_prt(X_kri)
print(f"Aggregated: X_prt.shape={X_prt.shape}")
print()

# Recover with VAR+DY
print("Running VAR+DY...")
method = RecoveryMethodsFactory.create('var_dy')
W_est = method.fit_and_recover(X_prt, threshold=0.1)

print(f"W_est.shape={W_est.shape}, std={W_est.std():.6f}")
print(f"W_est min={W_est.min():.6f}, max={W_est.max():.6f}")
print()

# Evaluate
print("Evaluating...")
evaluator = RecoveryEvaluator(W_true)
metrics = evaluator.evaluate(W_est)

print("VAR+DY Metrics:")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall: {metrics['recall']:.4f}")
print(f"  F1-score: {metrics['f1_score']:.4f}")
print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"  Edge weight correlation: {metrics['edge_weight_correlation']:.4f}")
print(f"  IR-MSE: {metrics['impulse_response_mse']:.6f}")
print()

# Benchmark against trivial baseline
print("Trivial baseline (predict-all):")
predict_all_precision = (W_true > 0).sum() / (20 * 20)
predict_all_f1 = 2 * predict_all_precision / (predict_all_precision + 1.0)
print(f"  Precision: {predict_all_precision:.4f}")
print(f"  Recall: 1.0000")
print(f"  F1-score: {predict_all_f1:.4f}")
print(f"  AUC-ROC: 0.5000")
print()

# Check if VAR+DY beats baseline
print("Verdict:")
print(f"  VAR+DY AUC > 0.60? {metrics['auc_roc'] > 0.60} (AUC={metrics['auc_roc']:.4f})")
print(f"  VAR+DY F1 > baseline? {metrics['f1_score'] > predict_all_f1} (VAR+DY={metrics['f1_score']:.4f}, baseline={predict_all_f1:.4f})")

if metrics['auc_roc'] > 0.60:
    print()
    print("✓ SUCCESS: VAR+DY produces meaningful recovery signal!")
    print("  The DGP lag-matrix fix is working.")
else:
    print()
    print("✗ ISSUE: VAR+DY still near-random")
    print("  The signal may still be too weak or the VAR fitting has other issues.")

print()
print("="*70)
