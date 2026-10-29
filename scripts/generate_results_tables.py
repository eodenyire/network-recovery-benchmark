#!/usr/bin/env python3
"""
Generate comprehensive results tables and summaries from the benchmark JSON
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# Load the final benchmark results
with open('results/FINAL/task4_enhanced_benchmark_with_random_FINAL.json', 'r') as f:
    benchmark_data = json.load(f)

print(f"Loaded {len(benchmark_data)} configuration results")
print("\nExtracting comprehensive statistics...")

# 1. Create detailed results table
results_list = []
for config in benchmark_data:
    row = {
        'Network Type': config.get('network_type', 'unknown'),
        'Time Steps': config.get('config', {}).get('time_steps'),
        'Noise Level': config.get('config', {}).get('noise_level'),
        'VAR+DY AUC': config.get('var_dy', {}).get('auc_roc', {}).get('mean', np.nan),
        'VAR+DY Precision': config.get('var_dy', {}).get('precision', {}).get('mean', np.nan),
        'VAR+DY Recall': config.get('var_dy', {}).get('recall', {}).get('mean', np.nan),
        'VAR+DY F1': config.get('var_dy', {}).get('f1_score', {}).get('mean', np.nan),
        'CoVaR AUC': config.get('covar', {}).get('auc_roc', {}).get('mean', np.nan),
        'CoVaR Precision': config.get('covar', {}).get('precision', {}).get('mean', np.nan),
        'CoVaR Recall': config.get('covar', {}).get('recall', {}).get('mean', np.nan),
        'CoVaR F1': config.get('covar', {}).get('f1_score', {}).get('mean', np.nan),
        'GNN AUC': config.get('gnn', {}).get('auc_roc', {}).get('mean', np.nan) if config.get('gnn') else np.nan,
        'GNN Precision': config.get('gnn', {}).get('precision', {}).get('mean', np.nan) if config.get('gnn') else np.nan,
        'GNN Recall': config.get('gnn', {}).get('recall', {}).get('mean', np.nan) if config.get('gnn') else np.nan,
        'GNN F1': config.get('gnn', {}).get('f1_score', {}).get('mean', np.nan) if config.get('gnn') else np.nan,
        'DebtRank MSE': config.get('debtrank', {}).get('impulse_response_mse', {}).get('mean', np.nan),
    }
    results_list.append(row)

df_results = pd.DataFrame(results_list)

# 2. Summary statistics by method and network type
designed = df_results[df_results['Network Type'] == 'designed']
random = df_results[df_results['Network Type'] == 'random']

print("\n=== DESIGNED NETWORK SUMMARY ===")
print(f"VAR+DY AUC: {designed['VAR+DY AUC'].mean():.4f} ± {designed['VAR+DY AUC'].std():.4f}")
print(f"CoVaR AUC: {designed['CoVaR AUC'].mean():.4f} ± {designed['CoVaR AUC'].std():.4f}")
print(f"GNN AUC: {designed['GNN AUC'].mean():.4f} ± {designed['GNN AUC'].std():.4f}")
print(f"DebtRank MSE: {designed['DebtRank MSE'].mean():.6f} ± {designed['DebtRank MSE'].std():.6f}")

print("\n=== RANDOM NETWORK SUMMARY ===")
print(f"VAR+DY AUC: {random['VAR+DY AUC'].mean():.4f} ± {random['VAR+DY AUC'].std():.4f}")
print(f"CoVaR AUC: {random['CoVaR AUC'].mean():.4f} ± {random['CoVaR AUC'].std():.4f}")
print(f"GNN AUC: {random['GNN AUC'].mean():.4f} ± {random['GNN AUC'].std():.4f}")
print(f"DebtRank MSE: {random['DebtRank MSE'].mean():.6f} ± {random['DebtRank MSE'].std():.6f}")

# 3. Save detailed results table
df_results.to_csv('results/detailed_benchmark_results.csv', index=False)
print("\n✓ Saved: results/detailed_benchmark_results.csv")

# 4. Create summary by data volume
summary_by_volume = df_results.groupby('Time Steps')[['VAR+DY AUC', 'CoVaR AUC', 'GNN AUC', 'DebtRank MSE']].agg(['mean', 'std'])
summary_by_volume.to_csv('results/summary_by_time_steps.csv')
print("✓ Saved: results/summary_by_time_steps.csv")

# 5. Create summary by noise level
summary_by_noise = df_results.groupby('Noise Level')[['VAR+DY AUC', 'CoVaR AUC', 'GNN AUC', 'DebtRank MSE']].agg(['mean', 'std'])
summary_by_noise.to_csv('results/summary_by_noise_level.csv')
print("✓ Saved: results/summary_by_noise_level.csv")

# 6. Create comparison table: Designed vs Random
comparison_data = {
    'Method': ['VAR+DY', 'CoVaR', 'GNN', 'DebtRank'],
    'Designed AUC/MSE': [
        f"{designed['VAR+DY AUC'].mean():.4f}",
        f"{designed['CoVaR AUC'].mean():.4f}",
        f"{designed['GNN AUC'].mean():.4f}",
        f"{designed['DebtRank MSE'].mean():.6f}"
    ],
    'Random AUC/MSE': [
        f"{random['VAR+DY AUC'].mean():.4f}",
        f"{random['CoVaR AUC'].mean():.4f}",
        f"{random['GNN AUC'].mean():.4f}",
        f"{random['DebtRank MSE'].mean():.6f}"
    ],
    'Difference': [
        f"{designed['VAR+DY AUC'].mean() - random['VAR+DY AUC'].mean():.4f}",
        f"{designed['CoVaR AUC'].mean() - random['CoVaR AUC'].mean():.4f}",
        f"{designed['GNN AUC'].mean() - random['GNN AUC'].mean():.4f}",
        f"{designed['DebtRank MSE'].mean() - random['DebtRank MSE'].mean():.6f}"
    ]
}
df_comparison = pd.DataFrame(comparison_data)
df_comparison.to_csv('results/designed_vs_random_comparison.csv', index=False)
print("✓ Saved: results/designed_vs_random_comparison.csv")

print("\nAll CSV files saved successfully!")
