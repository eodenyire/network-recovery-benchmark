#!/usr/bin/env python3
"""
Task #4: ENHANCED BENCHMARK WITH RANDOM NETWORK ROBUSTNESS CHECK

Executes the SRTG-20 benchmark on BOTH:
1. Designed network (W_true from srtg20_edges.csv) — 50 experiments
2. Random sparse networks (Erdos-Renyi, same density) — 50 experiments

Total: 100 experiments (10 configs × 2 network types × 5 replications)

This confirms whether results hinge on the specific designed network
or whether they're robust to network structure (same density, different edges).

Expected runtime: ~8-10 minutes on typical hardware (CPU)
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.experiment_runner import ExperimentRunner

print("="*80)
print("TASK #4: ENHANCED BENCHMARK WITH RANDOM NETWORK ROBUSTNESS CHECK")
print("="*80)
print()

print("Experimental configuration:")
print("  Data volume (T):        50, 100, 150, 200, 250")
print("  Noise level (sigma):    0.1, 0.5")
print("  Network types:          DESIGNED (W_true) + RANDOM (same density)")
print("  Replications per config: 5")
print("  Total configurations:    5 × 2 × 2 = 20")
print("  Total experiments:       20 × 5 = 100")
print()

# Configuration grid
time_steps_list = [50, 100, 150, 200, 250]
noise_levels = [0.1, 0.5]
densities = [2]

configs_designed = []
configs_random = []

for t in time_steps_list:
    for sigma in noise_levels:
        for density in densities:
            # Designed network config
            configs_designed.append({
                'time_steps': t,
                'noise_level': sigma,
                'network_density': density,
                't_df': 5.0,
                'use_documented_network': True,  # ← Use designed W_true
                'num_replications': 5,
                'network_type': 'designed',
            })
            
            # Random network config
            configs_random.append({
                'time_steps': t,
                'noise_level': sigma,
                'network_density': density,
                't_df': 5.0,
                'use_documented_network': False,  # ← Use random network
                'num_replications': 5,
                'network_type': 'random',
            })

all_configs = configs_designed + configs_random

print(f"Generated {len(configs_designed)} designed-network configs")
print(f"Generated {len(configs_random)} random-network configs")
print(f"Total: {len(all_configs)} configs")
print()

# Run benchmark
runner = ExperimentRunner("config.yaml")

print("Starting enhanced benchmark (designed + random networks)...")
print("-"*80)

start_time = time.time()
all_results = []
designed_results = []
random_results = []

for idx, config in enumerate(all_configs, 1):
    net_type = config['network_type']
    config_name = f"{net_type.upper()} | T{config['time_steps']}_N{config['noise_level']}_D{config['network_density']}"
    print(f"[{idx:3d}/{len(all_configs)}] {config_name:40s} ...", end=" ", flush=True)
    
    try:
        result = runner.run_single_configuration(
            time_steps=config['time_steps'],
            noise_level=config['noise_level'],
            network_density=config['network_density'],
            t_df=config['t_df'],
            num_replications=config['num_replications'],
            use_documented_network=config['use_documented_network'],
        )
        
        # Extract key metrics for display
        var_auc = result.get('var_dy', {}).get('auc_roc', {}).get('mean', 0)
        covar_auc = result.get('covar', {}).get('auc_roc', {}).get('mean', 0)
        
        print(f"VAR+DY AUC: {var_auc:.4f}  CoVaR AUC: {covar_auc:.4f}")
        
        # Add network type to result for later analysis
        result['network_type'] = net_type
        all_results.append(result)
        
        if net_type == 'designed':
            designed_results.append(result)
        else:
            random_results.append(result)
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)[:80]}")

elapsed = time.time() - start_time

print()
print("-"*80)
print(f"Benchmark completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print(f"Completed: {len(all_results)}/{len(all_configs)} configurations")
print()

# Save results
output_file = "results/task4_enhanced_benchmark_with_random.json"
Path("results").mkdir(exist_ok=True)

# Convert numpy types to native Python types for JSON serialization
def convert_to_serializable(obj):
    """Recursively convert numpy and other types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

serializable_results = convert_to_serializable(all_results)

with open(output_file, 'w') as f:
    json.dump(serializable_results, f, indent=2)

print(f"Results saved to: {output_file}")

# Summary statistics
print()
print("="*80)
print("SUMMARY: DESIGNED vs. RANDOM NETWORKS")
print("="*80)

def summarize_results(results, name):
    if not results:
        print(f"\n{name}: No results")
        return
    
    var_auc_values = [r.get('var_dy', {}).get('auc_roc', {}).get('mean', 0) for r in results]
    covar_auc_values = [r.get('covar', {}).get('auc_roc', {}).get('mean', 0) for r in results]
    
    import numpy as np
    print(f"\n{name}:")
    print(f"  VAR+DY AUC:   mean={np.mean(var_auc_values):.4f}, std={np.std(var_auc_values):.4f}, range=[{np.min(var_auc_values):.4f}, {np.max(var_auc_values):.4f}]")
    print(f"  CoVaR AUC:    mean={np.mean(covar_auc_values):.4f}, std={np.std(covar_auc_values):.4f}, range=[{np.min(covar_auc_values):.4f}, {np.max(covar_auc_values):.4f}]")

summarize_results(designed_results, "DESIGNED NETWORKS")
summarize_results(random_results, "RANDOM NETWORKS")

print()
print("-"*80)

# Comparison
if designed_results and random_results:
    import numpy as np
    des_var = [r.get('var_dy', {}).get('auc_roc', {}).get('mean', 0) for r in designed_results]
    ran_var = [r.get('var_dy', {}).get('auc_roc', {}).get('mean', 0) for r in random_results]
    
    des_cov = [r.get('covar', {}).get('auc_roc', {}).get('mean', 0) for r in designed_results]
    ran_cov = [r.get('covar', {}).get('auc_roc', {}).get('mean', 0) for r in random_results]
    
    print("COMPARISON (Designed - Random):")
    print(f"  VAR+DY AUC difference:  {np.mean(des_var) - np.mean(ran_var):+.4f}  (designed {'>' if np.mean(des_var) > np.mean(ran_var) else '<'} random)")
    print(f"  CoVaR AUC difference:   {np.mean(des_cov) - np.mean(ran_cov):+.4f}  (designed {'>' if np.mean(des_cov) > np.mean(ran_cov) else '<'} random)")

print()
print("="*80)
print("INTERPRETATION:")
print("="*80)
print()
print("Outcome A (Designed ≈ Random):")
print("  → Recovery is driven by density and edge-count structure, not the specific design")
print("  → The documented network topology does NOT materially affect results")
print("  → Confirms robustness to design choices")
print()
print("Outcome B (Designed > Random):")
print("  → The documented network carries information random networks do not")
print("  → Documented edges are systematically stronger or better-placed")
print("  → Design choices in srtg20_edges.csv DO contribute to recovery signal")
print()
print("="*80)
