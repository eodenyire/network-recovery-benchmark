#!/usr/bin/env python3
"""
Lag Identifiability Diagnostic (FAST VERSION)
==============================================

Purpose: Quick check whether VAR(1) lag channel is actually present in synthetic data.

Method: Compute average autocorrelation structure at PRT level to detect lag effects.

This addresses Ivan's requirement and helps explain VAR+DY performance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
import yaml

from src.synthetic_generator import DataGenerator


def diagnose_lag_channel(time_steps=250, noise_level=0.1, network_density=2, t_df=5.0, num_reps=3):
    """
    Fast lag identifiability diagnostic.
    
    Args:
        time_steps: Length of KRI series
        noise_level: Measurement noise std dev
        network_density: Edges per PRT node
        t_df: t-copula degrees of freedom
        num_reps: Number of replications (fast: 3 only)
        
    Returns:
        Dictionary with lag channel diagnostics
    """
    print("\n" + "="*80)
    print("LAG IDENTIFIABILITY DIAGNOSTIC")
    print("="*80)
    print(f"\nConfig: T={time_steps}, σ={noise_level}, D={network_density}, ν={t_df}")
    print(f"Replications: {num_reps} (FAST MODE)\n")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate data
    data_gen = DataGenerator('config.yaml')
    experiments = data_gen.generate_monte_carlo(
        time_steps=time_steps,
        noise_level=noise_level,
        network_density=network_density,
        t_df=t_df,
        num_replications=num_reps,
        use_documented_network=True
    )
    
    # Track lag correlations
    all_lag_strengths = []
    
    for rep_idx, exp in enumerate(experiments):
        X_kri = exp['kri_panel']  # (T, 100) KRI panel
        W_true_prt = exp['W_true']  # (20, 20)
        
        print(f"[Rep {rep_idx+1}/{num_reps}] Computing lag structure...")
        
        # Aggregate KRI→PRT
        X_prt = np.zeros((time_steps, 20))
        for prt in range(20):
            kri_start, kri_end = prt * 5, (prt + 1) * 5
            X_prt[:, prt] = X_kri[:, kri_start:kri_end].mean(axis=1)
        
        # Standardize
        X_prt = (X_prt - X_prt.mean(axis=0)) / (X_prt.std(axis=0) + 1e-8)
        
        # Compute lag-1 autocorrelation for each PRT
        lag_1_corrs = []
        for i in range(20):
            x_t = X_prt[1:, i]
            x_t_minus_1 = X_prt[:-1, i]
            if np.std(x_t) > 1e-8 and np.std(x_t_minus_1) > 1e-8:
                corr = np.corrcoef(x_t, x_t_minus_1)[0, 1]
                lag_1_corrs.append(np.abs(corr))
        
        mean_lag_1 = np.mean(lag_1_corrs) if lag_1_corrs else 0
        
        # Compute lag-5 autocorrelation
        lag_5_corrs = []
        for i in range(20):
            x_t = X_prt[5:, i]
            x_t_minus_5 = X_prt[:-5, i]
            if np.std(x_t) > 1e-8 and np.std(x_t_minus_5) > 1e-8:
                corr = np.corrcoef(x_t, x_t_minus_5)[0, 1]
                lag_5_corrs.append(np.abs(corr))
        
        mean_lag_5 = np.mean(lag_5_corrs) if lag_5_corrs else 0
        
        all_lag_strengths.append({'lag_1': mean_lag_1, 'lag_5': mean_lag_5})
        print(f"  Lag-1 autocorr: {mean_lag_1:.4f} | Lag-5: {mean_lag_5:.4f}")
    
    # Aggregate
    avg_lag_1 = np.mean([s['lag_1'] for s in all_lag_strengths])
    avg_lag_5 = np.mean([s['lag_5'] for s in all_lag_strengths])
    
    results = {
        'config': {
            'time_steps': time_steps,
            'noise_level': noise_level,
            'network_density': network_density,
            't_df': t_df,
            'num_replications': num_reps,
        },
        'lag_1_autocorr': float(avg_lag_1),
        'lag_5_autocorr': float(avg_lag_5),
        'decay_rate': float((avg_lag_1 - avg_lag_5) / (avg_lag_1 + 1e-10)),
    }
    
    # Print results
    print("\n" + "-"*80)
    print("LAG CHANNEL STRENGTH (Mean Absolute Autocorrelation)")
    print("-"*80)
    print(f"Lag 1: {avg_lag_1:.4f}")
    print(f"Lag 5: {avg_lag_5:.4f}")
    decay = (avg_lag_1 - avg_lag_5) / (avg_lag_1 + 1e-10) * 100
    print(f"Decay (Lag 1→5): {decay:.1f}%")
    
    print("\n" + "-"*80)
    print("DIAGNOSIS")
    print("-"*80)
    
    if avg_lag_1 > 0.12:
        print(f"✓ LAG CHANNEL STRONG: Mean lag-1 autocorr is {avg_lag_1:.4f}")
        print("  → VAR(1) lag matrix IS being transmitted through DGP")
        print("  → VAR+DY's poor recovery (AUC≈0.51) suggests:")
        print("    • Aggregation loss in KRI→PRT conversion")
        print("    • Or stationarity rescaling weakened the signal")
    elif avg_lag_1 > 0.06:
        print(f"⚠ LAG CHANNEL MODERATE: Mean lag-1 autocorr is {avg_lag_1:.4f}")
        print("  → Lag effects exist but weak")
        print("  → Stationarity rescaling may be limiting signal")
    else:
        print(f"✗ LAG CHANNEL WEAK: Mean lag-1 autocorr is only {avg_lag_1:.4f}")
        print("  → Lag matrix is not strongly transmitted")
        print("  → This explains VAR+DY's difficulty")
    
    print("\n" + "="*80 + "\n")
    
    return results


if __name__ == "__main__":
    # Run diagnostic on easiest cell
    results = diagnose_lag_channel(
        time_steps=250,
        noise_level=0.1,
        network_density=2,
        t_df=5.0,
        num_reps=3
    )
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/lag_identifiability_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to results/lag_identifiability_diagnostic.json")
