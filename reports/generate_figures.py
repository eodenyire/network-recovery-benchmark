#!/usr/bin/env python3
"""
Task #6: Generate figures for M6 report.

Figures to generate:
1. AUC-ROC heatmap (configs vs estimators)
2. F1 heatmap (configs vs estimators)
3. AUC degradation curve (T vs AUC-ROC)
4. F1 degradation curve (T vs F1)
5. CoVaR vs VAR+DY comparison scatter plot
6. DebtRank cascade comparison (if available)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

def load_results():
    """Load benchmark results."""
    results_file = Path("results/task4_reduced_benchmark.json")
    with open(results_file) as f:
        return json.load(f)

def organize_results(results):
    """Organize results by config and estimator."""
    data = {
        "config_names": [],
        "var_dy_auc": [],
        "var_dy_f1": [],
        "covar_auc": [],
        "covar_f1": [],
        "debtrank_auc": [],
        "debtrank_f1": [],
        "time_steps": [],
        "noise_levels": [],
    }
    
    for config_result in results:
        config = config_result.get("config", {})
        config_name = config.get("config_name", "unknown")
        t = config.get("time_steps", 0)
        noise = config.get("noise_level", 0)
        
        data["config_names"].append(config_name)
        data["time_steps"].append(t)
        data["noise_levels"].append(noise)
        
        # VAR+DY
        if config_result.get("var_dy"):
            data["var_dy_auc"].append(config_result["var_dy"]["auc_roc"]["mean"])
            data["var_dy_f1"].append(config_result["var_dy"]["f1_score"]["mean"])
        else:
            data["var_dy_auc"].append(np.nan)
            data["var_dy_f1"].append(np.nan)
        
        # CoVaR
        if config_result.get("covar"):
            data["covar_auc"].append(config_result["covar"]["auc_roc"]["mean"])
            data["covar_f1"].append(config_result["covar"]["f1_score"]["mean"])
        else:
            data["covar_auc"].append(np.nan)
            data["covar_f1"].append(np.nan)
        
        # DebtRank
        if config_result.get("debtrank"):
            data["debtrank_auc"].append(config_result["debtrank"]["auc_roc"]["mean"])
            data["debtrank_f1"].append(config_result["debtrank"]["f1_score"]["mean"])
        else:
            data["debtrank_auc"].append(np.nan)
            data["debtrank_f1"].append(np.nan)
    
    return data

def generate_auc_heatmap(data):
    """Generate AUC-ROC heatmap."""
    # Create matrix: rows = configs, columns = estimators
    estimators = ["VAR+DY", "CoVaR", "DebtRank"]
    auc_matrix = np.array([
        data["var_dy_auc"],
        data["covar_auc"],
        data["debtrank_auc"],
    ]).T
    
    config_labels = data["config_names"]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(auc_matrix, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5,
                xticklabels=estimators, yticklabels=config_labels,
                cbar_kws={"label": "AUC-ROC"}, vmin=0.4, vmax=1.0)
    plt.title("AUC-ROC Heatmap Across Estimators and Configurations", fontsize=14, fontweight="bold")
    plt.xlabel("Estimator", fontsize=12)
    plt.ylabel("Configuration (T, σ, Density)", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig("results/figures/auc_roc_heatmap_task6.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: results/figures/auc_roc_heatmap_task6.png")
    plt.close()

def generate_f1_heatmap(data):
    """Generate F1 heatmap."""
    estimators = ["VAR+DY", "CoVaR", "DebtRank"]
    f1_matrix = np.array([
        data["var_dy_f1"],
        data["covar_f1"],
        data["debtrank_f1"],
    ]).T
    
    config_labels = data["config_names"]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(f1_matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=estimators, yticklabels=config_labels,
                cbar_kws={"label": "F1-Score"}, vmin=0.0, vmax=1.0)
    plt.title("F1-Score Heatmap Across Estimators and Configurations", fontsize=14, fontweight="bold")
    plt.xlabel("Estimator", fontsize=12)
    plt.ylabel("Configuration (T, σ, Density)", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig("results/figures/f1_heatmap_task6.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: results/figures/f1_heatmap_task6.png")
    plt.close()

def generate_degradation_curves(data):
    """Generate AUC and F1 degradation curves by T."""
    # Group by time steps
    t_values = sorted(set(data["time_steps"]))
    
    # For each T, average across noise levels
    t_var_dy_auc = []
    t_covar_auc = []
    t_var_dy_f1 = []
    t_covar_f1 = []
    
    for t in t_values:
        indices = [i for i, ts in enumerate(data["time_steps"]) if ts == t]
        if indices:
            t_var_dy_auc.append(np.nanmean([data["var_dy_auc"][i] for i in indices]))
            t_covar_auc.append(np.nanmean([data["covar_auc"][i] for i in indices]))
            t_var_dy_f1.append(np.nanmean([data["var_dy_f1"][i] for i in indices]))
            t_covar_f1.append(np.nanmean([data["covar_f1"][i] for i in indices]))
    
    # AUC degradation curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_values, t_var_dy_auc, marker="o", linewidth=2, label="VAR+DY", color="steelblue")
    ax.plot(t_values, t_covar_auc, marker="s", linewidth=2, label="CoVaR", color="darkorange")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Baseline (AUC=0.5)")
    ax.set_xlabel("Time Steps (T)", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("AUC-ROC Degradation Curve vs. Data Volume", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(t_values)
    plt.tight_layout()
    plt.savefig("results/figures/auc_degradation_curve_task6.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: results/figures/auc_degradation_curve_task6.png")
    plt.close()
    
    # F1 degradation curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_values, t_var_dy_f1, marker="o", linewidth=2, label="VAR+DY", color="steelblue")
    ax.plot(t_values, t_covar_f1, marker="s", linewidth=2, label="CoVaR", color="darkorange")
    ax.axhline(y=0.1777, color="gray", linestyle="--", linewidth=1, label="Baseline (F1≈0.18)")
    ax.set_xlabel("Time Steps (T)", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("F1-Score Degradation Curve vs. Data Volume", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(t_values)
    plt.tight_layout()
    plt.savefig("results/figures/f1_degradation_curve_task6.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: results/figures/f1_degradation_curve_task6.png")
    plt.close()

def generate_noise_sensitivity_curve(data):
    """Generate AUC and F1 curves by noise level."""
    noise_values = sorted(set(data["noise_levels"]))
    
    # For each noise level, average across time steps
    noise_var_dy_auc = []
    noise_covar_auc = []
    noise_var_dy_f1 = []
    noise_covar_f1 = []
    
    for noise in noise_values:
        indices = [i for i, n in enumerate(data["noise_levels"]) if n == noise]
        if indices:
            noise_var_dy_auc.append(np.nanmean([data["var_dy_auc"][i] for i in indices]))
            noise_covar_auc.append(np.nanmean([data["covar_auc"][i] for i in indices]))
            noise_var_dy_f1.append(np.nanmean([data["var_dy_f1"][i] for i in indices]))
            noise_covar_f1.append(np.nanmean([data["covar_f1"][i] for i in indices]))
    
    # AUC sensitivity curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(noise_values, noise_var_dy_auc, marker="o", linewidth=2, label="VAR+DY", color="steelblue")
    ax.plot(noise_values, noise_covar_auc, marker="s", linewidth=2, label="CoVaR", color="darkorange")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Baseline (AUC=0.5)")
    ax.set_xlabel("Noise Level (σ)", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("AUC-ROC Sensitivity to Measurement Noise", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(noise_values)
    plt.tight_layout()
    plt.savefig("results/figures/auc_noise_sensitivity_task6.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: results/figures/auc_noise_sensitivity_task6.png")
    plt.close()

def generate_estimator_comparison_scatter(data):
    """Generate scatter plot: CoVaR vs VAR+DY AUC."""
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(data["var_dy_auc"], data["covar_auc"], 
                        c=data["time_steps"], cmap="viridis", s=100, alpha=0.7)
    
    # Add diagonal line (y = x)
    min_val = min(min(data["var_dy_auc"]), min(data["covar_auc"]))
    max_val = max(max(data["var_dy_auc"]), max(data["covar_auc"]))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.5, label="y=x")
    
    ax.set_xlabel("VAR+DY AUC-ROC", fontsize=12)
    ax.set_ylabel("CoVaR AUC-ROC", fontsize=12)
    ax.set_title("Estimator Comparison: CoVaR vs. VAR+DY", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time Steps (T)", fontsize=11)
    ax.legend(fontsize=11)
    ax.set_xlim([min_val - 0.02, max_val + 0.02])
    ax.set_ylim([min_val - 0.02, max_val + 0.02])
    plt.tight_layout()
    plt.savefig("results/figures/estimator_comparison_scatter_task6.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: results/figures/estimator_comparison_scatter_task6.png")
    plt.close()

def main():
    print("=" * 80)
    print("TASK #6: GENERATE FIGURES")
    print("=" * 80)
    
    results = load_results()
    data = organize_results(results)
    
    print(f"\nLoaded {len(results)} benchmark configurations.")
    print("Generating figures...")
    
    generate_auc_heatmap(data)
    generate_f1_heatmap(data)
    generate_degradation_curves(data)
    generate_noise_sensitivity_curve(data)
    generate_estimator_comparison_scatter(data)
    
    print("\n" + "=" * 80)
    print("FIGURES COMPLETE")
    print("=" * 80)
    print("\nGenerated figures:")
    print("  1. AUC-ROC heatmap")
    print("  2. F1-Score heatmap")
    print("  3. AUC degradation curve (vs. time)")
    print("  4. F1 degradation curve (vs. time)")
    print("  5. AUC noise sensitivity curve")
    print("  6. Estimator comparison scatter plot")
    print("\nReady for Task #7 (fill Results section).")

if __name__ == "__main__":
    main()
