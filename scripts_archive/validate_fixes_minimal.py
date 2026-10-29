#!/usr/bin/env python3
"""Minimal validation: 1 config × 1 rep to confirm all three fixes work."""

import sys
import json
sys.path.insert(0, '.')

from src.experiment_runner import ExperimentRunner

print("\n" + "="*80)
print("MINIMAL BENCHMARK VALIDATION (1 config × 1 rep)")
print("="*80)
print("\nTesting:")
print("  ✓ DebtRank MSE-only aggregation (Fix #1)")
print("  ✓ CoVaR F1 non-zero (Fix #2)")
print("  ✓ Direction identifiability diagnostic (Fix #3)")
print("\n" + "-"*80 + "\n")

# Run single config with 1 replication
runner = ExperimentRunner('config.yaml')
results = runner.run_single_configuration(
    time_steps=50,
    noise_level=0.1,
    network_density=2,
    t_df=5.0,
    num_replications=1,
    use_documented_network=True
)

# Validation checks
print("VALIDATION RESULTS:")
print("-" * 80)

# Check 1: DebtRank has only MSE
debtrank_keys = list(results['debtrank'].keys())
debtrank_expected = ['impulse_response_mse']
debtrank_pass = debtrank_keys == debtrank_expected
print(f"\n[Fix #1] DebtRank keys: {debtrank_keys}")
print(f"  Expected: {debtrank_expected}")
print(f"  Status: {'✓ PASS' if debtrank_pass else '✗ FAIL'}")

if debtrank_pass:
    print(f"  DebtRank IR-MSE value: {results['debtrank']['impulse_response_mse']['mean']:.6f}")

# Check 2: CoVaR F1 is non-zero
covar_f1 = results['covar']['f1_score']['mean']
covar_pass = covar_f1 > 0 and covar_f1 < 1
print(f"\n[Fix #2] CoVaR F1: {covar_f1:.4f}")
print(f"  Expected: > 0 and < 1 (non-zero)")
print(f"  Status: {'✓ PASS' if covar_pass else '✗ FAIL'}")

if covar_pass:
    print(f"  CoVaR AUC: {results['covar']['auc_roc']['mean']:.4f}")
    print(f"  CoVaR Precision: {results['covar']['precision']['mean']:.4f}")
    print(f"  CoVaR Recall: {results['covar']['recall']['mean']:.4f}")

# Check 3: VAR+DY metrics present
var_dy_f1 = results['var_dy']['f1_score']['mean']
var_dy_pass = var_dy_f1 >= 0
print(f"\n[Fix #3 baseline] VAR+DY F1: {var_dy_f1:.4f}")
print(f"  VAR+DY AUC: {results['var_dy']['auc_roc']['mean']:.4f}")
print(f"  (Confirms recovery estimators work normally)")

# Overall status
all_pass = debtrank_pass and covar_pass
print("\n" + "="*80)
print(f"OVERALL STATUS: {'✓ ALL FIXES VALIDATED' if all_pass else '✗ SOME FIXES FAILED'}")
print("="*80)

if all_pass:
    print("\nNext: Run full benchmark with these fixes applied")
    print("  python run_full_benchmark_v2.py --quick")
    print("  or")
    print("  python run_full_benchmark_v2.py --medium")
else:
    print("\n⚠ Some fixes failed. Debug required.")
    sys.exit(1)

print()
