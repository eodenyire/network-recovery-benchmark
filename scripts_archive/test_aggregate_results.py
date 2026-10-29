#!/usr/bin/env python3
"""Direct test of aggregate_results fix."""

import sys
import numpy as np
sys.path.insert(0, '.')

from src.evaluation import RecoveryEvaluator

print("\n" + "="*80)
print("TEST: aggregate_results() with dynamic metric detection")
print("="*80 + "\n")

# Test 1: Recovery estimator (6 metrics)
print("[Test 1] Recovery Estimator (VAR+DY, CoVaR, GNN)")
print("-" * 80)

recovery_results = [
    {
        'precision': 0.3,
        'recall': 0.5,
        'f1_score': 0.38,
        'auc_roc': 0.55,
        'edge_weight_correlation': 0.2,
        'impulse_response_mse': 0.05,
    },
    {
        'precision': 0.35,
        'recall': 0.48,
        'f1_score': 0.40,
        'auc_roc': 0.57,
        'edge_weight_correlation': 0.22,
        'impulse_response_mse': 0.04,
    }
]

agg_recovery = RecoveryEvaluator.aggregate_results(recovery_results)
print("Aggregated metrics:")
for metric_name in sorted(agg_recovery.keys()):
    mean_val = agg_recovery[metric_name]['mean']
    print(f"  {metric_name}: mean={mean_val:.4f}")

# Test 2: Propagation operator (MSE only, per Fix #1)
print("\n[Test 2] Propagation Operator (DebtRank - MSE only)")
print("-" * 80)

debtrank_results = [
    {
        'impulse_response_mse': 0.001,
    },
    {
        'impulse_response_mse': 0.002,
    }
]

agg_debtrank = RecoveryEvaluator.aggregate_results(debtrank_results)
print("Aggregated metrics:")
for metric_name in sorted(agg_debtrank.keys()):
    mean_val = agg_debtrank[metric_name]['mean']
    print(f"  {metric_name}: mean={mean_val:.6f}")

# Validation
print("\n" + "="*80)
print("VALIDATION")
print("="*80)

recovery_has_6 = len(agg_recovery) == 6
debtrank_has_1 = len(agg_debtrank) == 1
debtrank_is_mse = 'impulse_response_mse' in agg_debtrank

print(f"\n✓ Recovery estimator aggregates 6 metrics: {recovery_has_6}")
print(f"✓ DebtRank aggregates 1 metric only: {debtrank_has_1}")
print(f"✓ DebtRank metric is impulse_response_mse: {debtrank_is_mse}")

if recovery_has_6 and debtrank_has_1 and debtrank_is_mse:
    print("\n✓✓✓ FIX #1 VALIDATED: aggregate_results correctly handles both types")
else:
    print("\n✗✗✗ FIX #1 FAILED")
    sys.exit(1)

print("\n" + "="*80 + "\n")
