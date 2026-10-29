#!/usr/bin/env python3
"""Comprehensive inspection of all results files for audit."""

import json
import os
from datetime import datetime

results_dir = "results"
files_to_check = [
    "benchmark_results_quick.json",
    "task4_reduced_benchmark.json",
    "task4_enhanced_benchmark_with_random.json",
    "task4_random_robustness_check.json",
]

for filename in files_to_check:
    filepath = os.path.join(results_dir, filename)
    if not os.path.exists(filepath):
        print(f"\n❌ {filename}: FILE NOT FOUND")
        continue
    
    file_size = os.path.getsize(filepath)
    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*80}")
    print(f"FILE: {filename}")
    print(f"Size: {file_size:,} bytes | Modified: {mod_time}")
    print(f"{'='*80}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            print(f"Structure: LIST with {len(data)} items")
            if len(data) > 0:
                first_item = data[0]
                print(f"First item keys: {list(first_item.keys())}")
                
                # Find target config (T=250, sigma=0.1, D=2)
                target_idx = None
                for idx, item in enumerate(data):
                    if item.get('config', {}).get('T') == 250 and \
                       item.get('config', {}).get('sigma') == 0.1 and \
                       item.get('config', {}).get('D') == 2:
                        target_idx = idx
                        break
                
                if target_idx is not None:
                    print(f"\n✓ TARGET CONFIG FOUND at index {target_idx}")
                    target = data[target_idx]
                    config = target.get('config', {})
                    print(f"  Config: T={config.get('T')}, σ={config.get('sigma')}, D={config.get('D')}")
                    
                    # Check metrics
                    for estimator in ['var_dy', 'covar', 'gnn', 'debtrank']:
                        if estimator in target:
                            metrics = target[estimator]
                            has_auc = 'auc_roc' in metrics
                            has_f1 = 'f1_score' in metrics
                            has_mse = 'impulse_response_mse' in metrics
                            
                            if estimator == 'debtrank':
                                if has_auc or has_f1:
                                    print(f"  ⚠️  VIOLATION: DebtRank has AUC={has_auc}, F1={has_f1}")
                                    if has_auc:
                                        print(f"      AUC value: {metrics['auc_roc'].get('mean', 'N/A')}")
                                    if has_f1:
                                        print(f"      F1 value: {metrics['f1_score'].get('mean', 'N/A')}")
                                if has_mse:
                                    print(f"  ✓ DebtRank has IR-MSE: {metrics.get('impulse_response_mse', {}).get('mean', 'N/A')}")
                            else:
                                if has_auc:
                                    print(f"  ✓ {estimator.upper()} AUC-ROC: {metrics['auc_roc'].get('mean', 'N/A'):.6f}")
                                else:
                                    print(f"  ✗ {estimator.upper()} NO AUC found")
                else:
                    print(f"\n✗ TARGET CONFIG NOT FOUND")
                    # Show what configs are in the file
                    configs_present = []
                    for item in data:
                        cfg = item.get('config', {})
                        configs_present.append(f"T={cfg.get('T')}, σ={cfg.get('sigma')}, D={cfg.get('D')}")
                    print(f"Configs present: {configs_present[:5]}{'...' if len(configs_present) > 5 else ''}")
        
        elif isinstance(data, dict):
            print(f"Structure: DICT with keys: {list(data.keys())}")
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON DECODE ERROR: {e}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

print(f"\n{'='*80}")
print("END OF INSPECTION")
print(f"{'='*80}")
