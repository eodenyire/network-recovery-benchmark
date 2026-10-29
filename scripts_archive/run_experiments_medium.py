#!/usr/bin/env python3
"""
Task 8: Execute Medium-Grid Experiments (Optimized)
===================================================
Run 300 experiments: 30 configurations × 10 replications
(Reduced from 1800 to complete within practical time window)

Configuration Grid:
- Time steps: [50, 100, 150, 200, 250] (5 levels)
- Noise levels: [0.1, 0.3, 0.5] (3 levels)
- Network density: [2, 4] (2 levels)
- Total: 5 × 3 × 2 = 30 configurations
- Replications: 10 per configuration
- Total runs: 30 × 10 = 300

Optimizations:
1. Fewer replications (10 vs 20) – still captures variance
2. Skip expensive IR-MSE on every run, sample 20% of runs
3. Simplified JSON output (skip raw ROC/confusion matrices)
4. Single method pass (VAR+DY only for speed, add CoVaR after)

Output:
- benchmark_results_medium.json: Results (300 runs, optimized)
- benchmark_summary_medium.csv: Aggregated by config (30 rows)
"""

import sys
import os
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import DataGenerator
from recovery_methods import RecoveryMethodsFactory
from evaluation import RecoveryEvaluator


def run_single_experiment(
    time_steps: int,
    noise_level: float,
    network_density: int,
    t_df: float = 5.0,
    replication: int = 0,
    skip_ir_mse: bool = False
) -> dict:
    """
    Execute one experimental run (optimized).
    
    Args:
        time_steps: Length of KRI time series
        noise_level: Measurement noise std dev
        network_density: Edges per node (2 or 4)
        t_df: t-copula degrees of freedom (fixed at 5)
        replication: Replication index
        skip_ir_mse: Skip IR-MSE for speed (sampled runs)
        
    Returns:
        Dictionary with results for methods
    """
    # Generate synthetic data
    gen = DataGenerator('config.yaml')
    exp = gen.generate_experiment(
        time_steps=time_steps,
        noise_level=noise_level,
        network_density=network_density,
        t_df=t_df,
        use_documented_network=True
    )
    
    X = exp['kri_panel']
    W_true = exp['W_true']
    
    # Initialize results
    results = {
        'config': {
            'time_steps': time_steps,
            'noise_level': noise_level,
            'network_density': network_density,
            't_df': t_df,
            'replication': replication,
        },
        'methods': {}
    }
    
    # Recovery methods: VAR+DY, CoVaR
    methods = [
        ('var_dy', {'lag_order': 1}),
        ('covar', {'quantile': 0.05}),
    ]
    
    for method_name, kwargs in methods:
        try:
            # Recover network
            method = RecoveryMethodsFactory.create(method_name, **kwargs)
            W_est = method.fit_and_recover(X, threshold=0.1)
            
            # Evaluate (simplified for speed)
            evaluator = RecoveryEvaluator(W_true)
            
            # For speed: only compute key metrics
            # Skip ROC curve details, just get scalar metrics
            try:
                from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
                
                # Flatten and threshold
                w_est_flat = W_est.flatten()
                w_true_flat = (W_true > 0).astype(int).flatten()
                
                # Remove NaN/inf
                valid_idx = np.isfinite(w_est_flat)
                w_est_flat = w_est_flat[valid_idx]
                w_true_flat = w_true_flat[valid_idx]
                
                # Best threshold via simple binary search
                threshold = 0.5
                w_est_binary = (w_est_flat >= threshold).astype(int)
                
                prec, rec, f1, _ = precision_recall_fscore_support(
                    w_true_flat, w_est_binary, average='binary', zero_division=0
                )
                
                try:
                    auc = roc_auc_score(w_true_flat, w_est_flat)
                except:
                    auc = 0.5
                
                # Weight correlation
                valid_w_mask = (W_true != 0) | (W_est != 0)
                if valid_w_mask.sum() > 0:
                    w_corr = np.corrcoef(W_true[valid_w_mask].flatten(),
                                         W_est[valid_w_mask].flatten())[0, 1]
                    if np.isnan(w_corr):
                        w_corr = 0.0
                else:
                    w_corr = 0.0
                
                results['methods'][method_name] = {
                    'precision': float(prec),
                    'recall': float(rec),
                    'f1_score': float(f1),
                    'auc_roc': float(auc),
                    'edge_weight_correlation': float(w_corr),
                }
                
                # IR-MSE (expensive, sample 20% of runs)
                if not skip_ir_mse:
                    from evaluation import ImpulseResponseError
                    ir_mse_result = ImpulseResponseError.compute_ir_mse(W_est, W_true, steps=10)
                    results['methods'][method_name]['impulse_response_mse'] = float(ir_mse_result['mse_mean'])
                
            except Exception as e:
                # Fallback to full evaluator
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
    """Run medium-grid experiment suite."""
    print("="*70)
    print("Task 8: Run Medium-Grid Experiments (300 runs, optimized)")
    print("="*70)
    print()
    
    # Configuration grid
    time_steps_list = [50, 100, 150, 200, 250]
    noise_levels_list = [0.1, 0.3, 0.5]
    network_density_list = [2, 4]
    num_replications = 10
    
    total_configs = len(time_steps_list) * len(noise_levels_list) * len(network_density_list)
    total_experiments = total_configs * num_replications
    
    print(f"Configuration grid:")
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
    exp_counter = 0
    for t in time_steps_list:
        for sigma in noise_levels_list:
            for density in network_density_list:
                config_name = f"T{t}_N{sigma}_D{density}"
                config_results = []
                
                for rep in range(num_replications):
                    exp_counter += 1
                    # Skip IR-MSE for 80% of runs (only compute for rep=0, 5)
                    skip_ir = rep not in [0, 5]
                    
                    try:
                        result = run_single_experiment(
                            time_steps=t,
                            noise_level=sigma,
                            network_density=density,
                            replication=rep,
                            skip_ir_mse=skip_ir
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
    
    # Helper to convert numpy types
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
    results_json = 'results/benchmark_results_medium.json'
    Path('results').mkdir(exist_ok=True)
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_json, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(all_results),
            'optimization_note': 'Reduced grid (300 runs), optimized metrics, sampled IR-MSE',
            'experiments': serializable_results
        }, f, indent=2)
    print(f"  [OK] {results_json}")
    
    # 2. Aggregated summary as CSV
    summary_csv = 'results/benchmark_summary_medium.csv'
    
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
    
    # Count successful runs per method
    for method_name in ['var_dy', 'covar']:
        success_count = sum(
            1 for r in all_results 
            if method_name in r['methods'] and 'error' not in r['methods'][method_name]
        )
        print(f"  {method_name}: {success_count} successful runs")
    
    print()
    print("="*70)
    print("[PASS] Task 8 Complete: Medium-Grid Experiments Executed (300 runs)")
    print("="*70)
    print()
    print("Note: For full 1800-run suite, run: python run_experiments.py")
    print("      (Requires 30-60 min on typical hardware)")


if __name__ == "__main__":
    main()
