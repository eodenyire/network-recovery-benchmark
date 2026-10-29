#!/usr/bin/env python3
"""Inspect all results files for key metrics."""

import json
from pathlib import Path
import numpy as np

files_to_inspect = [
    'results/task4_reduced_benchmark.json',
    'results/benchmark_results_quick.json',
    'results/task4_enhanced_benchmark_with_random.json',
    'results/task4_random_robustness_check.json',
]

for file_path in files_to_inspect:
    path = Path(file_path)
    if not path.exists():
        print(f"\n{file_path}: FILE NOT FOUND")
        continue
    
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"\n{file_path}: JSON DECODE ERROR - {e}")
            continue
    
    # Handle list vs dict format
    if isinstance(data, list):
        configs = data
    else:
        configs = data.get('experiments', [])
    
    print(f"\n{'='*80}")
    print(f"FILE: {file_path}")
    print(f"{'='*80}")
    print(f"Number of config results: {len(configs)}")
    
    if not configs:
        print("  WARNING: No configurations found")
        continue
    
    # Find easiest cell: T=250, σ=0.1, D=2
    easiest = None
    for c in configs:
        cfg = c.get('config', {})
        if (cfg.get('time_steps') == 250 and 
            cfg.get('noise_level') == 0.1 and
            cfg.get('density', cfg.get('network_density')) == 2):
            easiest = c
            break
    
    if easiest:
        print(f"\nEasiest cell found: T=250, σ=0.1, D=2")
        print(f"  Config: {easiest.get('config')}")
        
        # VAR+DY
        var_dy = easiest.get('var_dy')
        if var_dy:
            if isinstance(var_dy, dict) and 'mean' in var_dy:
                auc = var_dy['mean'].get('auc_roc', 'N/A')
            else:
                auc = 'UNKNOWN FORMAT'
            print(f"  VAR+DY AUC-ROC: {auc}")
        else:
            print(f"  VAR+DY: NULL/MISSING")
        
        # CoVaR
        covar = easiest.get('covar')
        if covar:
            if isinstance(covar, dict) and 'mean' in covar:
                auc = covar['mean'].get('auc_roc', 'N/A')
            else:
                auc = 'UNKNOWN FORMAT'
            print(f"  CoVaR AUC-ROC: {auc}")
        else:
            print(f"  CoVaR: NULL/MISSING")
        
        # GNN
        gnn = easiest.get('gnn')
        print(f"  GNN: {type(gnn).__name__} ({'NULL' if gnn is None else 'POPULATED'})")
        
        # DebtRank
        debtrank = easiest.get('debtrank')
        if debtrank:
            if isinstance(debtrank, dict) and 'mean' in debtrank:
                ir_mse = debtrank['mean'].get('impulse_response_mse', 'N/A')
            else:
                ir_mse = 'UNKNOWN FORMAT'
            print(f"  DebtRank IR-MSE: {ir_mse}")
        else:
            print(f"  DebtRank: NULL/MISSING")
    else:
        print(f"  WARNING: Easiest cell (T=250, σ=0.1, D=2) NOT FOUND")
        # Show what cells ARE present
        sample_cfgs = [c.get('config') for c in configs[:3]]
        print(f"  Sample configs present: {sample_cfgs}")
