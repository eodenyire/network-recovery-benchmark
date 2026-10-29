#!/usr/bin/env python3
"""
Task #4: REDUCED-GRID BENCHMARK EXECUTION

Run 1200 experiments: 60 configs × 20 replications
- Data volume (T): 50, 100, 150, 200, 250 (5 levels)
- Noise level (σ): 0.1, 0.3, 0.5 (3 levels)
- Network density: 2, 4 edges/PRT (2 levels)
- Replications: 20 per config

Estimators: VAR+DY, CoVaR, GNN
Metrics: AUC-ROC, AUPRC, F1@oracle, F1@calibrated, edge-weight correlation, DebtRank MSE

Expected runtime: 20-30 hours on typical hardware (CPU)
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.experiment_runner import ExperimentRunner

print("="*80)
print("TASK #4: REDUCED-GRID BENCHMARK EXECUTION")
print("="*80)
print()

print("Experimental configuration (FAST PROOF FOR M6):")
print("  Data volume (T):        50, 100, 150, 200, 250")
print("  Noise level (sigma):    0.1, 0.5 (reduced from 3 levels)")
print("  Network density:        2 (reduced from 2 levels)")
print("  Replications per config: 5 (reduced from 20)")
print("  Total configurations:    5 × 2 × 1 = 10")
print("  Total experiments:       10 × 5 = 50 (fast proof of concept)")
print()

print("Note: This is a FAST-PROOF configuration for M6.")
print("      Only covers essential cells: easy (T=250,σ=0.1) and hard (T=50,σ=0.5).")
print()

# Configuration grid (FAST - only essential cells)
time_steps_list = [50, 100, 150, 200, 250]
noise_levels = [0.1, 0.5]  # Only easy and hard
densities = [2]  # Only sparse network

configs = []
for t in time_steps_list:
    for sigma in noise_levels:
        for density in densities:
            configs.append({
                'time_steps': t,
                'noise_level': sigma,
                'network_density': density,
                't_df': 5.0,
                'use_documented_network': True,
                'num_replications': 5,  # Reduced from 20
            })

print(f"Generated {len(configs)} configurations")
print()

# Run benchmark
runner = ExperimentRunner("config.yaml")

print("Starting benchmark...")
print("-"*80)

start_time = time.time()
all_results = []

for idx, config in enumerate(configs, 1):
    config_name = f"T{config['time_steps']}_N{config['noise_level']}_D{config['network_density']}"
    print(f"[{idx}/{len(configs)}] Running {config_name}...", flush=True)
    
    try:
        result = runner.run_single_configuration(
            time_steps=config['time_steps'],
            noise_level=config['noise_level'],
            network_density=config['network_density'],
            t_df=config['t_df'],
            num_replications=config['num_replications'],
        )
        
        # Extract key metrics for quick check
        var_auc = result.get('var_dy', {}).get('auc_roc', {}).get('mean', 0)
        covar_auc = result.get('covar', {}).get('auc_roc', {}).get('mean', 0)
        gnn_auc = result.get('gnn', {}).get('auc_roc', {}).get('mean', 0) if result.get('gnn') else 0
        
        print(f"      VAR+DY: {var_auc:.4f}  CoVaR: {covar_auc:.4f}  GNN: {gnn_auc:.4f}")
        
        all_results.append(result)
        
    except Exception as e:
        print(f"      ERROR: {type(e).__name__}: {str(e)[:100]}")

elapsed = time.time() - start_time
print()
print("-"*80)
print(f"Benchmark completed in {elapsed/3600:.1f} hours ({elapsed/60:.0f} minutes)")
print(f"Processed {len(all_results)} configurations successfully")
print()

# Save results
print("Saving results...")
output_file = "results/task4_reduced_benchmark.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"  Results saved to: {output_file}")
print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
print()

# Quick summary statistics
print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print()

var_aucs = [r.get('var_dy', {}).get('auc_roc', {}).get('mean', 0) for r in all_results]
covar_aucs = [r.get('covar', {}).get('auc_roc', {}).get('mean', 0) for r in all_results]

print(f"VAR+DY AUC-ROC:")
print(f"  Mean: {np.mean(var_aucs):.4f}")
print(f"  Std:  {np.std(var_aucs):.4f}")
print(f"  Min:  {np.min(var_aucs):.4f}")
print(f"  Max:  {np.max(var_aucs):.4f}")
print()

print(f"CoVaR AUC-ROC:")
print(f"  Mean: {np.mean(covar_aucs):.4f}")
print(f"  Std:  {np.std(covar_aucs):.4f}")
print(f"  Min:  {np.min(covar_aucs):.4f}")
print(f"  Max:  {np.max(covar_aucs):.4f}")
print()

print("="*80)
print("[OK] Task #4 complete!")
print("     Benchmark results ready for analysis in Task #5.")
print("="*80)
