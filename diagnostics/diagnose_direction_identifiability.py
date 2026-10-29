#!/usr/bin/env python3
"""
Direction Identifiability Diagnostic
=====================================

Purpose: Determine whether the VAR(1) lag matrix structure (A1 with W_true in off-diagonal)
is actually recoverable from observational data, not just theoretically present.

Method: 
1. Compute partial correlation controlling for contemporaneous effects
2. Compare partial correlation structure (off-diagonals) to W_true direction
3. Quantify what fraction of W_true's direction is revealed by lag channel

This addresses Iván's "make sure direction isn't identifiable" requirement.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path
import json
import yaml

from src.synthetic_generator import DataGenerator


def partial_correlation(X, i, j, controls):
    """
    Compute partial correlation of X[:,i] and X[:,j] controlling for variables in controls.
    
    Args:
        X: (T, n) data matrix
        i, j: Variable indices
        controls: List of control variable indices
        
    Returns:
        Partial correlation coefficient
    """
    if len(controls) == 0:
        # Simple correlation if no controls
        corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    # Residualize i against controls
    X_design = np.column_stack([np.ones(len(X))] + [X[:, c] for c in controls])
    try:
        coef_i = np.linalg.lstsq(X_design, X[:, i], rcond=None)[0]
        residual_i = X[:, i] - X_design @ coef_i
    except:
        return 0.0
    
    # Residualize j against controls
    try:
        coef_j = np.linalg.lstsq(X_design, X[:, j], rcond=None)[0]
        residual_j = X[:, j] - X_design @ coef_j
    except:
        return 0.0
    
    # Correlation of residuals
    if np.std(residual_i) > 1e-8 and np.std(residual_j) > 1e-8:
        corr = np.corrcoef(residual_i, residual_j)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    else:
        return 0.0


def diagnose_direction_identifiability(time_steps=250, noise_level=0.1, network_density=2, t_df=5.0, num_reps=3):
    """
    Assess whether W_true's direction is identifiable in the lag channel.
    
    Args:
        time_steps: Length of KRI series
        noise_level: Measurement noise
        network_density: Edges per PRT
        t_df: t-copula degrees of freedom
        num_reps: Number of replications
        
    Returns:
        Dictionary with direction recovery diagnostics
    """
    print("\n" + "="*80)
    print("DIRECTION IDENTIFIABILITY DIAGNOSTIC")
    print("="*80)
    print(f"\nConfig: T={time_steps}, σ={noise_level}, D={network_density}, ν={t_df}")
    print(f"Replications: {num_reps}\n")
    
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
    
    all_recovery_ratios = []
    
    for rep_idx, exp in enumerate(experiments):
        X_kri = exp['kri_panel']  # (T, 100) KRI panel
        W_true_prt = exp['W_true']  # (20, 20)
        
        print(f"[Rep {rep_idx+1}/{num_reps}] Computing lag-channel partial correlations...")
        
        # Aggregate KRI→PRT
        X_prt = np.zeros((time_steps, 20))
        for prt in range(20):
            kri_start, kri_end = prt * 5, (prt + 1) * 5
            X_prt[:, prt] = X_kri[:, kri_start:kri_end].mean(axis=1)
        
        # Standardize
        X_prt = (X_prt - X_prt.mean(axis=0)) / (X_prt.std(axis=0) + 1e-8)
        
        # For each edge in W_true, compute:
        # - Lag-1 partial correlation: partial_corr(X_t[j], X_{t-1}[i] | X_t)
        # This measures whether i→j is revealed by the lag channel after removing contemporaneous effects
        
        directed_edges = []
        recoverable_count = 0
        total_edges = 0
        
        for i in range(20):
            for j in range(20):
                if i == j:
                    continue
                
                if W_true_prt[i, j] > 0:
                    # Edge i→j exists in W_true
                    total_edges += 1
                    
                    # Compute lag-1 partial correlation
                    # i→j should appear as correlation between X_t[j] and X_{t-1}[i]
                    # when controlling for X_t (contemporaneous effects)
                    
                    X_lag = np.column_stack([X_prt[:-1], X_prt[1:]])  # [X_{t-1}, X_t]
                    
                    # Partial corr: X_t[j] ~ X_{t-1}[i] | [all other X_t variables]
                    controls_t = [k + 20 for k in range(20) if k != j]  # All X_t except j
                    
                    try:
                        # Residualize X_t[j] against X_t variables
                        X_design = np.column_stack([np.ones(len(X_lag))] + [X_lag[:, c] for c in controls_t])
                        coef = np.linalg.lstsq(X_design, X_lag[:, 20 + j], rcond=None)[0]
                        residual_j = X_lag[:, 20 + j] - X_design @ coef
                        
                        # Residualize X_{t-1}[i] against X_t variables
                        coef_i = np.linalg.lstsq(X_design, X_lag[:, i], rcond=None)[0]
                        residual_i = X_lag[:, i] - X_design @ coef_i
                        
                        # Correlation of residuals
                        if np.std(residual_i) > 1e-8 and np.std(residual_j) > 1e-8:
                            partial_corr = np.corrcoef(residual_i, residual_j)[0, 1]
                            if not np.isnan(partial_corr) and partial_corr > 0.05:
                                recoverable_count += 1
                                directed_edges.append({
                                    'edge': f'{i}->{j}',
                                    'partial_corr': float(partial_corr),
                                })
                    except Exception as e:
                        pass  # Skip edges with computation errors
        
        recovery_ratio = recoverable_count / total_edges if total_edges > 0 else 0
        all_recovery_ratios.append(recovery_ratio)
        
        print(f"  Recoverable edges: {recoverable_count}/{total_edges} ({100*recovery_ratio:.1f}%)")
    
    # Aggregate
    mean_recovery_ratio = np.mean(all_recovery_ratios) if all_recovery_ratios else 0
    
    results = {
        'config': {
            'time_steps': time_steps,
            'noise_level': noise_level,
            'network_density': network_density,
            't_df': t_df,
            'num_replications': num_reps,
        },
        'direction_recovery_ratio': float(mean_recovery_ratio),
        'direction_identifiable': bool(mean_recovery_ratio > 0.5),
    }
    
    # Print results
    print("\n" + "-"*80)
    print("DIRECTION IDENTIFIABILITY RESULTS")
    print("-"*80)
    print(f"Mean recovery ratio: {mean_recovery_ratio:.1%}")
    print("(Fraction of W_true's directed edges revealed by lag-channel partial correlation)")
    
    print("\n" + "-"*80)
    print("DIAGNOSIS")
    print("-"*80)
    
    if mean_recovery_ratio > 0.5:
        print(f"✓ DIRECTION IDENTIFIABLE: {mean_recovery_ratio:.1%} of edges visible in lag channel")
        print("  → VAR+DY should theoretically recover direction")
        print("  → Poor VAR+DY AUC suggests algorithmic or estimation issue")
    elif mean_recovery_ratio > 0.2:
        print(f"⚠ PARTIAL IDENTIFIABILITY: {mean_recovery_ratio:.1%} of edges visible")
        print("  → Only partial direction recovery expected")
    else:
        print(f"✗ NOT IDENTIFIABLE: Only {mean_recovery_ratio:.1%} of edges visible in lag channel")
        print("  → Direction fundamentally unrecoverable from this data")
        print("  → This fully explains VAR+DY's performance")
    
    print("\n" + "="*80 + "\n")
    
    return results


if __name__ == "__main__":
    # Run diagnostic
    results = diagnose_direction_identifiability(
        time_steps=250,
        noise_level=0.1,
        network_density=2,
        t_df=5.0,
        num_reps=3
    )
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/direction_identifiability_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to results/direction_identifiability_diagnostic.json")
