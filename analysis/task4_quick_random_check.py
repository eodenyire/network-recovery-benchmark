#!/usr/bin/env python3
"""
Task #4: QUICK RANDOM NETWORK CHECK (2 key cells only)

Tests ONLY:
- Easiest cell: T=250, σ=0.1
- Hardest cell: T=50, σ=0.5

On BOTH designed and random networks to confirm robustness.
Runtime: ~2-3 minutes for 4 configs × 5 reps = 20 experiments

This is Ivan's requirement: "add a robustness check with random sparse networks
of the same density so the results don't hinge on your design."
"""

import sys
import json
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.experiment_runner import ExperimentRunner

print("="*80)
print("TASK #4: QUICK RANDOM NETWORK ROBUSTNESS CHECK")
print("="*80)
print()

print("Testing 2 key configurations on both DESIGNED and RANDOM networks:")
print("  Config 1 (EASIEST):  T=250, σ=0.1 (lots of data, low noise)")
print("  Config 2 (HARDEST):  T=50, σ=0.5   (little data, high noise)")
print()
print("This tests whether results hinge on the specific designed network")
print("or whether they're robust across different network structures at same density.")
print()

# Key configs
configs = [
    {'t': 250, 'sigma': 0.1, 'name': 'EASIEST', 'network': 'designed'},
    {'t': 250, 'sigma': 0.1, 'name': 'EASIEST', 'network': 'random'},
    {'t': 50, 'sigma': 0.5, 'name': 'HARDEST', 'network': 'designed'},
    {'t': 50, 'sigma': 0.5, 'name': 'HARDEST', 'network': 'random'},
]

runner = ExperimentRunner("config.yaml")

print("Running experiments...")
print("-"*80)

start_time = time.time()
all_results = []

for idx, config in enumerate(configs, 1):
    net_type = config['network']
    use_designed = (net_type == 'designed')
    t = config['t']
    sigma = config['sigma']
    name = config['name']
    
    label = f"[{idx}/4] {name:8s} | T={t:3d}, σ={sigma:.1f} | {net_type.upper():8s} network"
    print(f"{label}...", end=" ", flush=True)
    
    try:
        result = runner.run_single_configuration(
            time_steps=t,
            noise_level=sigma,
            network_density=2,
            t_df=5.0,
            num_replications=5,
            use_documented_network=use_designed,
        )
        
        var_auc = result.get('var_dy', {}).get('auc_roc', {}).get('mean', 0)
        covar_auc = result.get('covar', {}).get('auc_roc', {}).get('mean', 0)
        
        print(f"VAR+DY: {var_auc:.4f}  CoVaR: {covar_auc:.4f}")
        
        result['network_type'] = net_type
        result['config_type'] = name
        all_results.append(result)
        
    except Exception as e:
        print(f"ERROR: {str(e)[:60]}")

elapsed = time.time() - start_time

print()
print("="*80)
print(f"Completed in {elapsed:.1f} seconds")
print()

# Save results
output_file = "results/task4_random_robustness_check.json"
Path("results").mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Results saved to: {output_file}")
print()

# Analysis
print("="*80)
print("ROBUSTNESS CHECK RESULTS")
print("="*80)
print()

import numpy as np

# Organize by config and network type
easiest_designed = [r for r in all_results if r.get('config_type') == 'EASIEST' and r.get('network_type') == 'designed']
easiest_random = [r for r in all_results if r.get('config_type') == 'EASIEST' and r.get('network_type') == 'random']
hardest_designed = [r for r in all_results if r.get('config_type') == 'HARDEST' and r.get('network_type') == 'designed']
hardest_random = [r for r in all_results if r.get('config_type') == 'HARDEST' and r.get('network_type') == 'random']

def extract_metrics(results, method='var_dy'):
    """Extract AUC and F1 from results"""
    aucs = []
    f1s = []
    for r in results:
        data = r.get(method, {})
        aucs.append(data.get('auc_roc', {}).get('mean', np.nan))
        f1s.append(data.get('f1_score', {}).get('mean', np.nan))
    return np.array(aucs), np.array(f1s)

print("EASIEST CELL (T=250, σ=0.1):")
print("-"*40)

var_des, f1_des = extract_metrics(easiest_designed, 'var_dy')
var_ran, f1_ran = extract_metrics(easiest_random, 'var_dy')
cov_des, _ = extract_metrics(easiest_designed, 'covar')
cov_ran, _ = extract_metrics(easiest_random, 'covar')

print(f"VAR+DY:")
print(f"  Designed network:  AUC={np.mean(var_des):.4f} (±{np.std(var_des):.4f}), F1={np.mean(f1_des):.4f}")
print(f"  Random network:    AUC={np.mean(var_ran):.4f} (±{np.std(var_ran):.4f}), F1={np.mean(f1_ran):.4f}")
print(f"  Difference:        {np.mean(var_des) - np.mean(var_ran):+.4f} (Designed {'>' if np.mean(var_des) > np.mean(var_ran) else '<'} Random)")

print(f"\nCoVaR:")
print(f"  Designed network:  AUC={np.mean(cov_des):.4f}")
print(f"  Random network:    AUC={np.mean(cov_ran):.4f}")
print(f"  Difference:        {np.mean(cov_des) - np.mean(cov_ran):+.4f}")

print()
print("HARDEST CELL (T=50, σ=0.5):")
print("-"*40)

var_des, f1_des = extract_metrics(hardest_designed, 'var_dy')
var_ran, f1_ran = extract_metrics(hardest_random, 'var_dy')
cov_des, _ = extract_metrics(hardest_designed, 'covar')
cov_ran, _ = extract_metrics(hardest_random, 'covar')

print(f"VAR+DY:")
print(f"  Designed network:  AUC={np.mean(var_des):.4f} (±{np.std(var_des):.4f}), F1={np.mean(f1_des):.4f}")
print(f"  Random network:    AUC={np.mean(var_ran):.4f} (±{np.std(var_ran):.4f}), F1={np.mean(f1_ran):.4f}")
print(f"  Difference:        {np.mean(var_des) - np.mean(var_ran):+.4f} (Designed {'>' if np.mean(var_des) > np.mean(var_ran) else '<'} Random)")

print(f"\nCoVaR:")
print(f"  Designed network:  AUC={np.mean(cov_des):.4f}")
print(f"  Random network:    AUC={np.mean(cov_ran):.4f}")
print(f"  Difference:        {np.mean(cov_des) - np.mean(cov_ran):+.4f}")

print()
print("="*80)
print("INTERPRETATION")
print("="*80)
print()

all_var_des = np.concatenate([extract_metrics([r for r in all_results if r.get('network_type') == 'designed'], 'var_dy')[0] for _ in [1]])
all_var_ran = np.concatenate([extract_metrics([r for r in all_results if r.get('network_type') == 'random'], 'var_dy')[0] for _ in [1]])

var_dif = np.mean(all_var_des) - np.mean(all_var_ran)

if abs(var_dif) < 0.02:
    print("✓ OUTCOME A: Designed ≈ Random")
    print("  → Recovery is robust to network design")
    print("  → Results driven by density structure, not specific edges")
    print("  → Documented network topology does NOT materially affect recovery")
elif var_dif > 0.05:
    print("✓ OUTCOME B: Designed > Random (significantly)")
    print("  → Designed network carries information random networks do not")
    print("  → Documented edges ARE systematically stronger/better-placed")
    print("  → Design choices in srtg20_edges.csv DO contribute")
else:
    print("⚠ OUTCOME C: Unclear (small difference)")
    print(f"  → Difference of {var_dif:+.4f} is at the edge of significance")
    print("  → May require more replications to resolve")

print()
print("="*80)
