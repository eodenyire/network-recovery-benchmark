#!/usr/bin/env python3
"""
Task #3: Trivial Baselines (PREDICT-ALL, RANDOM-K)

These are closed-form baseline metrics that do NOT depend on the DGP.
They should be deterministic across all runs.

Expected:
- Predict-all precision: 0.10 (40 true edges / 400 possible edges at density 2)
- Predict-all F1: 0.18
- Predict-all AUC-ROC: 0.500

If predict-all precision = 0.070, that's the OLD BUG SIGNATURE.
"""

import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.recovery_methods import RecoveryMethodsFactory
from src.evaluation import RecoveryEvaluator

print("="*80)
print("TASK #3: TRIVIAL BASELINES (PREDICT-ALL, RANDOM-K)")
print("="*80)
print()

# Load the network
from src.synthetic_generator import PlantedNetworkGenerator

gen = PlantedNetworkGenerator(num_prts=20, edge_file='data/srtg20_edges.csv')
W_true = gen.generate()

print(f"Ground truth network:")
print(f"  Shape: {W_true.shape}")
print(f"  True edges: {(W_true > 0).sum()}")
print(f"  Possible edges: {W_true.size} (20x20 adjacency)")
print(f"  Base rate (density): {(W_true > 0).sum() / W_true.size:.4f}")
print()

# Create evaluator
evaluator = RecoveryEvaluator(W_true)

# Test 1: PREDICT-ALL (declare every edge present)
print("Test 1: PREDICT-ALL baseline")
print("-" * 80)

W_predict_all = np.ones((20, 20))  # Every possible edge
np.fill_diagonal(W_predict_all, 0)  # Except diagonal (no self-loops)

results_pa = evaluator.evaluate(W_predict_all)

print(f"  Precision:  {results_pa['precision']:.6f}  (expected: 0.10)")
print(f"  Recall:     {results_pa['recall']:.6f}  (expected: 1.00)")
print(f"  F1:         {results_pa['f1_score']:.6f}  (expected: 0.18)")
print(f"  AUC-ROC:    {results_pa['auc_roc']:.6f}  (expected: 0.500)")
print()

# Validation
if abs(results_pa['precision'] - 0.10) < 0.01:
    print("  [OK] Precision ~= 0.10 (CORRECT base rate)")
elif abs(results_pa['precision'] - 0.070) < 0.01:
    print("  [X] Precision ~= 0.070 (OLD BUG SIGNATURE - STOP HERE)")
    sys.exit(1)
else:
    print(f"  [!] Precision = {results_pa['precision']:.4f} (unexpected)")

print()

# Test 2: RANDOM-K (randomly declare K edges)
print("Test 2: RANDOM-K baseline (K = num true edges)")
print("-" * 80)

num_true_edges = (W_true > 0).sum()
num_trials = 10

random_f1_scores = []
random_auc_scores = []

for trial in range(num_trials):
    # Randomly place num_true_edges edges
    W_random = np.zeros((20, 20))
    possible_edges = np.argwhere((np.ones((20, 20)) - np.eye(20)) > 0)  # Exclude diagonal
    random_indices = np.random.choice(len(possible_edges), size=num_true_edges, replace=False)
    for idx in random_indices:
        i, j = possible_edges[idx]
        W_random[i, j] = 1.0
    
    results_rk = evaluator.evaluate(W_random)
    random_f1_scores.append(results_rk['f1_score'])
    random_auc_scores.append(results_rk['auc_roc'])

print(f"  F1 score:   mean={np.mean(random_f1_scores):.6f}  std={np.std(random_f1_scores):.6f}")
print(f"             (expected: ~0.10 ± 0.05)")
print(f"  AUC-ROC:    mean={np.mean(random_auc_scores):.6f}  std={np.std(random_auc_scores):.6f}")
print(f"             (expected: ~0.50 ± 0.03)")
print()

print("="*80)
print("SUMMARY")
print("="*80)

checks = []

# Check 1: Predict-all precision
if abs(results_pa['precision'] - 0.10) < 0.01:
    checks.append(("[OK]", "Predict-all precision = 0.10 (correct)"))
elif abs(results_pa['precision'] - 0.070) < 0.01:
    checks.append(("[X]", "Predict-all precision = 0.070 (OLD BUG)"))
else:
    checks.append(("[!]", f"Predict-all precision = {results_pa['precision']:.4f}"))

# Check 2: Predict-all F1
if abs(results_pa['f1_score'] - 0.18) < 0.02:
    checks.append(("[OK]", f"Predict-all F1 = {results_pa['f1_score']:.4f} (correct, ~0.18)"))
else:
    checks.append(("[!]", f"Predict-all F1 = {results_pa['f1_score']:.4f} (expected ~0.18)"))

# Check 3: Predict-all AUC
if abs(results_pa['auc_roc'] - 0.50) < 0.01:
    checks.append(("[OK]", f"Predict-all AUC-ROC = {results_pa['auc_roc']:.4f} (correct, 0.500)"))
else:
    checks.append(("[!]", f"Predict-all AUC-ROC = {results_pa['auc_roc']:.4f} (expected 0.500)"))

# Check 4: Random-K F1
if 0.05 <= np.mean(random_f1_scores) <= 0.15:
    checks.append(("[OK]", f"Random-K F1 = {np.mean(random_f1_scores):.4f} (correct, ~0.10)"))
else:
    checks.append(("[!]", f"Random-K F1 = {np.mean(random_f1_scores):.4f} (expected ~0.10)"))

# Check 5: Random-K AUC
if 0.47 <= np.mean(random_auc_scores) <= 0.53:
    checks.append(("[OK]", f"Random-K AUC = {np.mean(random_auc_scores):.4f} (correct, ~0.50)"))
else:
    checks.append(("[!]", f"Random-K AUC = {np.mean(random_auc_scores):.4f} (expected ~0.50)"))

for status, message in checks:
    print(f"  {status} {message}")

print()

ok_count = sum(1 for s, _ in checks if s == "[OK]")
total_count = len(checks)

if ok_count >= 4:
    print(f"[OK] PASS: Task #3 complete ({ok_count}/{total_count} checks passed)")
    print("  Trivial baselines are working correctly.")
else:
    print(f"[!] PARTIAL: Task #3 ({ok_count}/{total_count} checks passed)")
    print("  Review checks marked [!] for potential issues.")

print("="*80)
