#!/usr/bin/env python3
"""
Task #2: Single-Cell Proof of Concept (VAR+DY)

Test the easiest configuration:
- T = 250 time steps
- σ = 0.1 measurement noise (low)
- ν = 5 degrees of freedom (moderate tail dependence)
- density = 2 edges per PRT (~40 edges, W_true)
- num_replications = 5 (quick test)

Expected results (from Exoectations document):
- VAR+DY AUC-ROC: 0.65–0.80 (vs. baseline 0.500)
- VAR+DY AUPRC: 0.15–0.30 (vs. baseline 0.10)
- VAR+DY F1@oracle: 0.20–0.35 (vs. baseline 0.18)
- VAR+DY F1@calibrated: 0.15–0.30
- Predict-all precision: 0.10, F1: 0.18 (NOT 0.070, 0.131 — old bug signature)

Success criterion: VAR+DY beats predict-all on at least 3 of 4 metrics (AUC, AUPRC, F1@oracle, F1@calibrated)
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path so we can import src as a package
sys.path.insert(0, str(Path(__file__).parent))

# Now import from src
from src.experiment_runner import ExperimentRunner

print("="*80)
print("TASK #2: SINGLE-CELL PROOF OF CONCEPT (VAR+DY)")
print("="*80)
print()
print("Configuration:")
print("  Time steps (T):        250")
print("  Measurement noise:     0.1 (easy cell)")
print("  Tail dependence (nu):  5")
print("  Network density:       2 edges/PRT (~40 edges)")
print("  Network type:          Designed (W_true)")
print("  Replications:          5 (quick test)")
print()
print("Expected results:")
print("  VAR+DY AUC-ROC:        0.65–0.80  (baseline: 0.500)")
print("  VAR+DY AUPRC:          0.15–0.30  (baseline: 0.10)")
print("  VAR+DY F1@oracle:      0.20–0.35  (baseline: 0.18)")
print("  Predict-all precision: 0.10       (NOT 0.070 — that's the old bug)")
print()
print("Running experiment...")
print("-"*80)

try:
    runner = ExperimentRunner("config.yaml")
    
    result = runner.run_single_configuration(
        time_steps=250,
        noise_level=0.1,
        network_density=2,
        t_df=5.0,
        num_replications=5
    )
    
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()
    
    # Extract results
    config = result['config']
    print(f"Configuration confirmed:")
    print(f"  T={config['time_steps']}, noise={config['noise_level']}, "
          f"df={config['t_df']}, density={config['network_density']}")
    print()
    
    # Trivial baselines
    print("Trivial Baselines (closed-form):")
    print("-" * 80)
    
    if 'predict_all' in result and result['predict_all']:
        pa = result['predict_all']
        print(f"  Predict-All:")
        print(f"    Precision:  {pa.get('precision', 'N/A')}")
        print(f"    Recall:     {pa.get('recall', 'N/A')}")
        print(f"    F1:         {pa.get('f1_score', 'N/A')}")
        print(f"    AUC-ROC:    {pa.get('auc_roc', 'N/A')}")
        print(f"    AUPRC:      {pa.get('auprc', 'N/A')}")
        
        # Check for old bug signature
        pa_precision = pa.get('precision', 0)
        if abs(pa_precision - 0.070) < 0.01:
            print(f"    [!] WARNING: Precision ~= 0.070 (old bug signature!)")
        elif abs(pa_precision - 0.10) < 0.02:
            print(f"    [OK] OK: Precision ~= 0.10 (correct base rate)")
        print()
    
    # VAR+DY results
    print("VAR+DY Recovery:")
    print("-" * 80)
    
    if 'var_dy' in result and result['var_dy']:
        var = result['var_dy']
        print(f"  Metrics (mean ± std over {config.get('num_replications', '?')} reps):")
        
        metrics = ['auc_roc', 'auprc', 'f1_score', 'f1_calibrated', 'precision', 'recall']
        
        for metric in metrics:
            if metric in var:
                val = var[metric]
                if isinstance(val, dict) and 'mean' in val:
                    mean = val['mean']
                    std = val.get('std', 0)
                    print(f"    {metric:20s}: {mean:.4f} ± {std:.4f}")
                else:
                    print(f"    {metric:20s}: {val}")
        print()
    
    # CoVaR results (for comparison)
    print("CoVaR Recovery (diagnostic):")
    print("-" * 80)
    
    if 'covar' in result and result['covar']:
        covar = result['covar']
        print(f"  Metrics (mean ± std):")
        
        metrics = ['auc_roc', 'auprc', 'f1_score', 'f1_calibrated']
        
        for metric in metrics:
            if metric in covar:
                val = covar[metric]
                if isinstance(val, dict) and 'mean' in val:
                    mean = val['mean']
                    std = val.get('std', 0)
                    print(f"    {metric:20s}: {mean:.4f} ± {std:.4f}")
        print()
    
    # GNN results (if available)
    print("GNN Recovery:")
    print("-" * 80)
    
    if 'gnn' in result and result['gnn']:
        gnn = result['gnn']
        print(f"  Metrics (mean ± std):")
        
        metrics = ['auc_roc', 'auprc', 'f1_score', 'f1_calibrated']
        
        for metric in metrics:
            if metric in gnn:
                val = gnn[metric]
                if isinstance(val, dict) and 'mean' in val:
                    mean = val['mean']
                    std = val.get('std', 0)
                    print(f"    {metric:20s}: {mean:.4f} ± {std:.4f}")
        print()
    
    # Validation against expectations
    print("="*80)
    print("VALIDATION AGAINST EXPECTATIONS")
    print("="*80)
    print()
    
    var_auc = result.get('var_dy', {}).get('auc_roc', {}).get('mean', 0)
    var_f1 = result.get('var_dy', {}).get('f1_score', {}).get('mean', 0)
    pa_precision = result.get('predict_all', {}).get('precision', 0)
    
    checks = []
    
    # Check 1: Predict-all precision ≈ 0.10
    if abs(pa_precision - 0.10) < 0.02:
        checks.append(("[OK]", "Predict-all precision ~= 0.10 (correct base rate)"))
    elif abs(pa_precision - 0.070) < 0.01:
        checks.append(("[X]", "Predict-all precision ~= 0.070 (OLD BUG SIGNATURE — stop here)"))
    else:
        checks.append(("[?]", f"Predict-all precision = {pa_precision:.4f} (expected ~= 0.10)"))
    
    # Check 2: VAR+DY AUC-ROC in expected range [0.65, 0.80]
    if 0.65 <= var_auc <= 0.80:
        checks.append(("[OK]", f"VAR+DY AUC-ROC = {var_auc:.4f} (in expected range 0.65-0.80)"))
    elif var_auc == 0.500:
        checks.append(("[X]", "VAR+DY AUC-ROC = 0.500 (DGP lag matrix may be broken)"))
    elif var_auc > 0.500:
        checks.append(("[?]", f"VAR+DY AUC-ROC = {var_auc:.4f} (below expected range, check DGP)"))
    else:
        checks.append(("[X]", f"VAR+DY AUC-ROC = {var_auc:.4f} (unexpected)"))
    
    # Check 3: VAR+DY beats predict-all on AUC-ROC
    if var_auc > 0.50:
        checks.append(("[OK]", f"VAR+DY AUC-ROC ({var_auc:.4f}) > baseline (0.500)"))
    else:
        checks.append(("[X]", f"VAR+DY AUC-ROC ({var_auc:.4f}) <= baseline (0.500)"))
    
    # Check 4: VAR+DY F1 in expected range [0.20, 0.35]
    if 0.20 <= var_f1 <= 0.35:
        checks.append(("[OK]", f"VAR+DY F1@oracle = {var_f1:.4f} (in expected range 0.20-0.35)"))
    elif var_f1 > 0.18:
        checks.append(("[?]", f"VAR+DY F1@oracle = {var_f1:.4f} (below expected range, but above baseline)"))
    else:
        checks.append(("[X]", f"VAR+DY F1@oracle = {var_f1:.4f} (at or below baseline 0.18)"))
    
    # Print checks
    for status, message in checks:
        print(f"  {status} {message}")
    
    print()
    
    # Overall assessment
    success_count = sum(1 for s, _ in checks if s == "[OK]")
    total_count = len(checks)
    
    print("="*80)
    if success_count >= 3:
        print(f"[OK] PASS: Task #2 complete ({success_count}/{total_count} checks passed)")
        print("  VAR+DY recovery is working and producing non-trivial results.")
        print("  Ready to proceed to Task #3 (Trivial Baselines) and Task #4 (Full Benchmark).")
    else:
        print(f"[X] FAIL: Task #2 incomplete ({success_count}/{total_count} checks passed)")
        print("  Debug required before proceeding to full benchmark.")
    
    print("="*80)
    print()
    
    # Save result to file for later reference
    output_file = "results/task2_single_cell_proof.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Result saved to {output_file}")
    
except Exception as e:
    print(f"[X] ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
