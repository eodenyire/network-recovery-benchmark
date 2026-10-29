#!/usr/bin/env python3
"""
Generate publication-ready visualizations from benchmark results
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

# Load results
with open('results/FINAL/task4_enhanced_benchmark_with_random_FINAL.json', 'r') as f:
    benchmark_data = json.load(f)

# Create DataFrame
results_list = []
for config in benchmark_data:
    row = {
        'Network Type': config.get('network_type', 'unknown'),
        'Time Steps': config.get('config', {}).get('time_steps'),
        'Noise Level': config.get('config', {}).get('noise_level'),
        'VAR+DY AUC': config.get('var_dy', {}).get('auc_roc', {}).get('mean', np.nan),
        'CoVaR AUC': config.get('covar', {}).get('auc_roc', {}).get('mean', np.nan),
        'GNN AUC': config.get('gnn', {}).get('auc_roc', {}).get('mean', np.nan) if config.get('gnn') else np.nan,
        'DebtRank MSE': config.get('debtrank', {}).get('impulse_response_mse', {}).get('mean', np.nan),
    }
    results_list.append(row)

df = pd.DataFrame(results_list)

# Create results directory for figures
Path('results/figures').mkdir(exist_ok=True)

# Figure 1: Network Recovery Methods Comparison (AUC by Data Volume)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Designed networks
designed = df[df['Network Type'] == 'designed']
random = df[df['Network Type'] == 'random']

time_steps_designed = sorted(designed['Time Steps'].unique())
var_designed = [designed[designed['Time Steps'] == t]['VAR+DY AUC'].mean() for t in time_steps_designed]
covar_designed = [designed[designed['Time Steps'] == t]['CoVaR AUC'].mean() for t in time_steps_designed]
gnn_designed = [designed[designed['Time Steps'] == t]['GNN AUC'].mean() for t in time_steps_designed]

axes[0].plot(time_steps_designed, var_designed, 'o-', linewidth=2, markersize=8, label='VAR+DY')
axes[0].plot(time_steps_designed, covar_designed, 's-', linewidth=2, markersize=8, label='CoVaR')
axes[0].plot(time_steps_designed, gnn_designed, '^-', linewidth=2, markersize=8, label='GNN')
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Classifier')
axes[0].set_xlabel('Time Steps (T)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
axes[0].set_title('Designed Network Recovery Performance', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.15, 0.75])

# Random networks
time_steps_random = sorted(random['Time Steps'].unique())
var_random = [random[random['Time Steps'] == t]['VAR+DY AUC'].mean() for t in time_steps_random]
covar_random = [random[random['Time Steps'] == t]['CoVaR AUC'].mean() for t in time_steps_random]
gnn_random = [random[random['Time Steps'] == t]['GNN AUC'].mean() for t in time_steps_random]

axes[1].plot(time_steps_random, var_random, 'o-', linewidth=2, markersize=8, label='VAR+DY')
axes[1].plot(time_steps_random, covar_random, 's-', linewidth=2, markersize=8, label='CoVaR')
axes[1].plot(time_steps_random, gnn_random, '^-', linewidth=2, markersize=8, label='GNN')
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Classifier')
axes[1].set_xlabel('Time Steps (T)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
axes[1].set_title('Random Network Recovery Performance', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0.15, 0.85])

plt.tight_layout()
plt.savefig('results/figures/01_recovery_methods_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/01_recovery_methods_comparison.png")
plt.close()

# Figure 2: Designed vs Random Network Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

methods = ['VAR+DY', 'CoVaR', 'GNN', 'DebtRank']
metrics = ['VAR+DY AUC', 'CoVaR AUC', 'GNN AUC', 'DebtRank MSE']

for idx, (method, metric) in enumerate(zip(methods, metrics)):
    ax = axes[idx // 2, idx % 2]
    
    if metric == 'DebtRank MSE':
        data_designed = designed['DebtRank MSE'].values
        data_random = random['DebtRank MSE'].values
        y_label = 'Impulse Response MSE'
        is_mse = True
    else:
        data_designed = designed[metric].values
        data_random = random[metric].values
        y_label = 'AUC-ROC'
        is_mse = False
    
    positions = [1, 2]
    bp = ax.boxplot([data_designed, data_random], positions=positions, widths=0.6, patch_artist=True)
    ax.set_xticklabels(['Designed', 'Random'])
    
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.set_title(f'{method} Performance Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean values
    mean_designed = np.mean(data_designed)
    mean_random = np.mean(data_random)
    ax.plot([1], [mean_designed], 'D', color='darkblue', markersize=8, label='Mean (Designed)')
    ax.plot([2], [mean_random], 'D', color='darkred', markersize=8, label='Mean (Random)')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/02_designed_vs_random_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/02_designed_vs_random_boxplot.png")
plt.close()

# Figure 3: Effect of Noise Level
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

noise_levels = sorted(df['Noise Level'].unique())

# Designed
var_by_noise_des = [designed[designed['Noise Level'] == n]['VAR+DY AUC'].mean() for n in noise_levels]
covar_by_noise_des = [designed[designed['Noise Level'] == n]['CoVaR AUC'].mean() for n in noise_levels]

axes[0].plot(noise_levels, var_by_noise_des, 'o-', linewidth=2, markersize=10, label='VAR+DY')
axes[0].plot(noise_levels, covar_by_noise_des, 's-', linewidth=2, markersize=10, label='CoVaR')
axes[0].set_xlabel('Noise Level (σ)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
axes[0].set_title('Effect of Noise: Designed Networks', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Random
var_by_noise_ran = [random[random['Noise Level'] == n]['VAR+DY AUC'].mean() for n in noise_levels]
covar_by_noise_ran = [random[random['Noise Level'] == n]['CoVaR AUC'].mean() for n in noise_levels]

axes[1].plot(noise_levels, var_by_noise_ran, 'o-', linewidth=2, markersize=10, label='VAR+DY')
axes[1].plot(noise_levels, covar_by_noise_ran, 's-', linewidth=2, markersize=10, label='CoVaR')
axes[1].set_xlabel('Noise Level (σ)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
axes[1].set_title('Effect of Noise: Random Networks', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/03_effect_of_noise.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/03_effect_of_noise.png")
plt.close()

# Figure 4: Summary Heatmap - Method Performance
summary_data = {
    'VAR+DY\n(Designed)': [designed['VAR+DY AUC'].mean()],
    'CoVaR\n(Designed)': [designed['CoVaR AUC'].mean()],
    'GNN\n(Designed)': [designed['GNN AUC'].mean()],
    'VAR+DY\n(Random)': [random['VAR+DY AUC'].mean()],
    'CoVaR\n(Random)': [random['CoVaR AUC'].mean()],
    'GNN\n(Random)': [random['GNN AUC'].mean()],
}

fig, ax = plt.subplots(figsize=(12, 2))
heatmap_data = np.array([[summary_data[col][0] for col in summary_data.keys()]])
im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

ax.set_xticks(range(len(summary_data)))
ax.set_xticklabels(summary_data.keys(), fontsize=11, fontweight='bold')
ax.set_yticks([0])
ax.set_yticklabels(['AUC-ROC'], fontsize=11, fontweight='bold')

# Add values to heatmap
for i, col in enumerate(summary_data.keys()):
    ax.text(i, 0, f'{summary_data[col][0]:.3f}', ha='center', va='center', 
            color='white', fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax, label='AUC-ROC Score', shrink=0.8)
plt.title('Network Recovery Methods: Summary Performance', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/figures/04_performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/04_performance_heatmap.png")
plt.close()

# Figure 5: DebtRank Propagation Performance
fig, ax = plt.subplots(figsize=(10, 6))

debtrank_designed = designed['DebtRank MSE'].values
debtrank_random = random['DebtRank MSE'].values

positions = [1, 2]
bp = ax.boxplot([debtrank_designed, debtrank_random], positions=positions, widths=0.5, patch_artist=True)
ax.set_xticklabels(['Designed Network', 'Random Network'])

for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)

# Add individual points
ax.scatter([1] * len(debtrank_designed), debtrank_designed, alpha=0.5, s=50, color='blue')
ax.scatter([2] * len(debtrank_random), debtrank_random, alpha=0.5, s=50, color='red')

ax.set_ylabel('Impulse Response MSE', fontsize=12, fontweight='bold')
ax.set_title('DebtRank Propagation Operator Performance', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add mean markers
mean_des = np.mean(debtrank_designed)
mean_ran = np.mean(debtrank_random)
ax.plot([1], [mean_des], 'D', color='darkblue', markersize=10, label=f'Mean (Designed): {mean_des:.6f}')
ax.plot([2], [mean_ran], 'D', color='darkred', markersize=10, label=f'Mean (Random): {mean_ran:.6f}')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/05_debtrank_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/05_debtrank_performance.png")
plt.close()

print("\n✓ All visualizations generated successfully!")
print(f"\nTotal figures created: 5")
print("Location: results/figures/")
