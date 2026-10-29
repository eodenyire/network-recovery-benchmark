#!/usr/bin/env python3
"""Check remaining results files for B3 violation."""

import json
import os

files = [
    "results/task4_enhanced_benchmark_with_random.json",
    "results/task4_random_robustness_check.json",
]

for filepath in files:
    if not os.path.exists(filepath):
        print(f"\n❌ {filepath}: FILE NOT FOUND")
        continue
    
    print(f"\n{'='*80}")
    print(f"FILE: {filepath}")
    print(f"{'='*80}")
    
    with open(filepath) as f:
        data = json.load(f)
    
    print(f"Total configs: {len(data)}")
    
    # Find T=250, σ=0.1, D=2
    found = False
    for idx, item in enumerate(data):
        cfg = item.get('config', {})
        if cfg.get('time_steps') == 250 and cfg.get('noise_level') == 0.1 and cfg.get('network_density') == 2:
            print(f"\n✓ TARGET CONFIG FOUND at index {idx}")
            print(f"  Config name: {cfg.get('config_name')}")
            
            # Check DebtRank metrics
            if 'debtrank' in item:
                dr = item['debtrank']
                has_auc = 'auc_roc' in dr
                has_f1 = 'f1_score' in dr
                has_mse = 'impulse_response_mse' in dr
                
                print(f"\n  DebtRank metrics:")
                if has_auc:
                    print(f"    ❌ AUC-ROC PRESENT: {dr['auc_roc'].get('mean', 'N/A')}")
                if has_f1:
                    print(f"    ❌ F1-Score PRESENT: {dr['f1_score'].get('mean', 'N/A')}")
                if has_mse:
                    print(f"    ✓ IR-MSE PRESENT: {dr['impulse_response_mse'].get('mean', 'N/A')}")
                
                if has_auc or has_f1:
                    print(f"\n  ⚠️  VIOLATION OF B3: DebtRank has classification metrics")
                else:
                    print(f"\n  ✓ COMPLIANT: DebtRank only has IR-MSE")
            
            # Check other estimators
            for est in ['var_dy', 'covar', 'gnn']:
                if est in item and item[est] is not None:
                    auc_val = item[est].get('auc_roc', {}).get('mean', 'N/A') if isinstance(item[est], dict) else 'N/A'
                    print(f"  {est.upper()} AUC-ROC: {auc_val}")
            
            found = True
            break
    
    if not found:
        print(f"\n✗ TARGET CONFIG NOT FOUND (T=250, σ=0.1, D=2)")
        print(f"Sample configs:")
        for idx in range(min(3, len(data))):
            cfg = data[idx].get('config', {})
            print(f"  [{idx}] {cfg.get('config_name', 'N/A')}")

print(f"\n{'='*80}")
print("END OF INSPECTION")
print(f"{'='*80}")
