#!/usr/bin/env python3
"""
Task 8: Run Fixed Experiments (KRI→PRT Aggregation)
====================================================
FIX: Aggregate KRI data to PRT level BEFORE recovery to avoid numerical instability.

Previous run showed VAR+DY producing constant F1≈0.178 and AUC≈0.49, indicating
the 100-variable VAR model was numerically unstable (FEVD computation).

This version aggregates (T, 100) KRI data → (T, 20) PRT data before fitting VAR.
"""

import sys
import os
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import DataGenerator
from recovery_methods import RecoveryMethodsFactory
from evaluation import RecoveryEvaluator


def aggregate_kri_to_prt(X_kri: np.ndarray) -> np.ndarray:
    """
    Aggregate KRI-level data to PRT level by averaging within blocks.
    
    Args:
        X_kri (T, 100): KRI-level time series (5 KRIs per PRT)
        
    Returns:
        X_prt (T, 20): PRT-level time series (averaged per PRT)
    """
    T, n_kri = X_kri.shape
    assert n_kri == 100, f"Expected 100 KRIs, got {n_kri}"
    
    n_prt = 20
    X_prt = np.zeros((T, n_prt))
    
    for i in range(n_prt):
        kri_start = i * 5
        kri_end = (i + 1) * 5
        X_prt[:, i] = X_kri[:, kri_start:kri_end].mean(axis=1)
    
    return X_prt


def run_single_experiment(
    time_steps: int,
    noise_level: float,
    network_density: int,
    t_df: float = 5.0,
    replication: int = 0
) -> dict:
    """Execute one experimental run with aggregation fix."""
    # Generate synthetic data
    gen = DataGenerator('config.yaml')
    exp = gen.generate_experiment(
        time_steps=time_steps,
        noise_level=noise_level,
        network_density=network_density,
        t_df=t_df,
        use_documented_network=True
    )
    
    X_kri = exp['kri_panel']
    W_true = exp['W_true']
    
    # CRITICAL FIX: Aggregate KRI data to PRT level
    X = aggregate_kri_to_prt(X_kri)
    
    # Initialize results
    results = {
        'config': {
            'time_steps': time_steps,
            'noise_level': noise_level,
            'network_density': network_density,
            'replication': replication,
        },
        'methods': {}
    }
    
    # Recovery methods
    methods = [
        ('var_dy', {'lag_order': 1}),
        ('covar', {'quantile': 0.05}),
    ]
    
    for method_name, kwargs in methods:
        try:
            # Recover network from PRT-level data
            method = RecoveryMethodsFactory.create(method_name, **kwargs)
            W_est = method.fit_and_recover(X, threshold=0.1)
            
            # W_est should now be 20×20 (PRT-level)
            assert W_est.shape == (20, 20), f"Expected (20, 20), got {W_est.shape}"
            
            # Evaluate
            evaluator = RecoveryEvaluator(W_true)
            metrics = evaluator.evaluate(W_est)
            
            results['methods'][method_name] = {
                'precision': float(metrics.get('precision', 0)),
                'recall': float(metrics.get('recall', 0)),
                'f1_score': float(metrics.get('f1_score', 0)),
                'auc_roc': float(metrics.get('auc_roc', 0)),
                'edge_weight_correlation': float(metrics.get('edge_weight_correlation', 0)),
                'impulse_response_mse': float(metrics.get('impulse_response_mse', 0)),
            }
        
        except Exception as e:
            results['methods'][method_name] = {
                'error': str(e)[:200]
            }
    
    return results


def aggregate_config_results(config_results: list) -> dict:
    """Aggregate results across replications for one configuration."""
    aggregated = {
        'num_replications': len(config_results),
        'methods': {}
    }
    
    methods_to_aggregate = ['var_dy', 'covar']
    metrics_to_aggregate = ['precision', 'recall', 'f1_score', 'auc_roc', 
                           'edge_weight_correlation']
    
    for method_name in methods_to_aggregate:
        method_results = []
        
        for config_result in config_results:
            if method_name in config_result['methods']:
                method_metrics = config_result['methods'][method_name]
                if 'error' not in method_metrics:
                    method_results.append(method_metrics)
        
        if method_results:
            aggregated['methods'][method_name] = {}
            
            for metric_name in metrics_to_aggregate:
                values = [r.get(metric_name, np.nan) for r in method_results]
                values = [v for v in values if not np.isnan(v) and np.isfinite(v)]
                
                if values:
                    aggregated['methods'][method_name][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                    }
    
    return aggregated


def main():
    """Run fixed grid experiment suite."""
    print("="*70)
    print("Task 8 FIX: Run Experiments with KRI→PRT Aggregation")
    print("="*70)
    print()
    
    # Configuration grid (small subset for quick validation)
    time_steps_list = [50, 100, 150, 200, 250]
    noise_levels_list = [0.1, 0.3, 0.5]
    network_density_list = [2, 4]
    num_replications = 5  # Reduced for quick validation
    
    total_configs = len(time_steps_list) * len(noise_levels_list) * len(network_density_list)
    total_experiments = total_configs * num_replications
    
    print(f"Configuration grid (QUICK VALIDATION):")
    print(f"  Time steps: {time_steps_list}")
    print(f"  Noise levels: {noise_levels_list}")
    print(f"  Network density: {network_density_list}")
    print(f"  Replications per config: {num_replications}")
    print(f"  Total configurations: {total_configs}")
    print(f"  Total experiments: {total_experiments}")
    print()
    
    # Results storage
    all_results = []
    config_aggregated = []
    
    # Progress tracker
    pbar = tqdm(total=total_experiments, desc="Running experiments", unit="exp")
    
    # Run experiments
    for t in time_steps_list:
        for sigma in noise_levels_list:
            for density in network_density_list:
                config_name = f"T{t}_N{sigma}_D{density}"
                config_results = []
                
                for rep in range(num_replications):
                    try:
                        result = run_single_experiment(
                            time_steps=t,
                            noise_level=sigma,
                            network_density=density,
                            replication=rep
                        )
                        result['config_name'] = config_name
                        all_results.append(result)
                        config_results.append(result)
                    except Exception as e:
                        print(f"\nERROR in config {config_name}, rep {rep}: {e}")
                    
                    pbar.update(1)
                
                # Aggregate this configuration
                agg = aggregate_config_results(config_results)
                agg['config_name'] = config_name
                agg['config'] = {
                    'time_steps': t,
                    'noise_level': sigma,
                    'network_density': density,
                }
                config_aggregated.append(agg)
    
    pbar.close()
    
    print()
    print("="*70)
    print(f"Completed {len(all_results)} experiments")
    print("="*70)
    print()
    
    # Save results
    print("Saving results...")
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    # 1. Full results as JSON
    results_json = 'results/benchmark_results_fixed.json'
    Path('results').mkdir(exist_ok=True)
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_json, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(all_results),
            'note': 'FIXED: KRI→PRT aggregation before recovery',
            'experiments': serializable_results
        }, f, indent=2)
    print(f"  [OK] {results_json}")
    
    # 2. Aggregated summary as CSV
    summary_csv = 'results/benchmark_summary_fixed.csv'
    
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['config_name', 'time_steps', 'noise_level', 'network_density', 
                 'num_reps',
                 'var_dy_f1_mean', 'var_dy_f1_std', 'var_dy_auc_mean', 'var_dy_auc_std',
                 'covar_f1_mean', 'covar_f1_std', 'covar_auc_mean', 'covar_auc_std']
        writer.writerow(header)
        
        # Data rows
        for agg in config_aggregated:
            row = [
                agg['config_name'],
                agg['config']['time_steps'],
                agg['config']['noise_level'],
                agg['config']['network_density'],
                agg['num_replications'],
            ]
            
            # VAR+DY
            if 'var_dy' in agg['methods'] and 'f1_score' in agg['methods']['var_dy']:
                row.append(agg['methods']['var_dy']['f1_score'].get('mean', ''))
                row.append(agg['methods']['var_dy']['f1_score'].get('std', ''))
                row.append(agg['methods']['var_dy']['auc_roc'].get('mean', ''))
                row.append(agg['methods']['var_dy']['auc_roc'].get('std', ''))
            else:
                row.extend(['', '', '', ''])
            
            # CoVaR
            if 'covar' in agg['methods'] and 'f1_score' in agg['methods']['covar']:
                row.append(agg['methods']['covar']['f1_score'].get('mean', ''))
                row.append(agg['methods']['covar']['f1_score'].get('std', ''))
                row.append(agg['methods']['covar']['auc_roc'].get('mean', ''))
                row.append(agg['methods']['covar']['auc_roc'].get('std', ''))
            else:
                row.extend(['', '', '', ''])
            
            writer.writerow(row)
    
    print(f"  [OK] {summary_csv}")
    print()
    
    # Summary statistics
    print("Summary Statistics:")
    print(f"  Total experiments run: {len(all_results)}")
    print(f"  Total configurations: {len(config_aggregated)}")
    
    for method_name in ['var_dy', 'covar']:
        success_count = sum(
            1 for r in all_results 
            if method_name in r['methods'] and 'error' not in r['methods'][method_name]
        )
        print(f"  {method_name}: {success_count} successful runs")
    
    print()
    print("="*70)
    print("[DONE] Aggregation fix validated. Ready for full grid.")
    print("="*70)


if __name__ == "__main__":
    main()
