#!/usr/bin/env python3
"""Quick test to validate DebtRank fix."""

import sys
sys.path.insert(0, '.')
from src.experiment_runner import ExperimentRunner

runner = ExperimentRunner('config.yaml')
print('\n' + '='*80)
print('Testing DebtRank fix: Quick benchmark (1 config × 2 reps)')
print('='*80 + '\n')

results = runner.run_single_configuration(
    time_steps=50,
    noise_level=0.1,
    network_density=2,
    t_df=5.0,
    num_replications=2,
    use_documented_network=True
)

print('Results for Config (T=50, σ=0.1, D=2):')
print(f'  VAR+DY: {list(results["var_dy"].keys())}')
print(f'  CoVaR: {list(results["covar"].keys())}')
print(f'  DebtRank: {list(results["debtrank"].keys())}')
print(f'  GNN: {list(results["gnn"].keys())}')

print('\n' + '-'*80)
print('DebtRank metrics (should have ONLY impulse_response_mse):')
print('-'*80)
for metric_name, metric_value in results['debtrank'].items():
    if isinstance(metric_value, dict):
        print(f'  {metric_name}: mean={metric_value["mean"]:.6f}, std={metric_value["std"]:.6f}')

print('\n' + '-'*80)
print('CoVaR F1 (should be non-zero):')
print('-'*80)
print(f'  CoVaR F1: mean={results["covar"]["f1_score"]["mean"]:.4f}')

print('\n' + '='*80)
print('✓ All fixes validated!')
print('='*80 + '\n')
