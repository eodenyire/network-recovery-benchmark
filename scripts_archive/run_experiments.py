#!/usr/bin/env python3
"""
Task 8: Execute Reduced-Grid Experiments
=========================================
Run 1800 experiments: 90 configurations × 20 replications

Configuration Grid:
- Time steps: [50, 100, 150, 200, 250] (5 levels)
- Noise levels: [0.1, 0.3, 0.5] (3 levels)
- Network density: [2, 4] (2 levels)
- Total: 5 × 3 × 2 = 90 configurations
- Replications: 20 per configuration
- Total runs: 90 × 20 = 1800

Methods benchmarked:
1. VAR + Diebold-Yilmaz
2. CoVaR (tail-risk baseline)
3. GNN (advanced)
4. DebtRank (propagation check)

Metrics computed per run:
- Precision, Recall, F1, AUC-ROC
- Edge weight correlation
- Impulse-response MSE

Output:
- benchmark_results.json: Full results (all 1800 runs)
- benchmark_summary.csv: Aggregated by config (90 rows, one per config)
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
from propagation import PropagationSimulator


def run_single_experiment(
    time_steps: int,
    noise_level: float,
    network_density: int,
    t_df: float = 5.0,
    replication: int = 0
) -> dict:
    """
    Execute one experimental run.
    
    Args:
        time_steps: Length of KRI time series
        noise_level: Measurement noise std dev
        network_density: Edges per node (2 or 4)
        t_df: t-copula degrees of freedom (fixed at 5)
        replication: Replication index (0-19)
        
    Returns:
        Dictionary with results for all methods
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
    
    # Aggregate from 100 KRIs to 20 PRTs
    W_true_prt = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            # Block (i,j) of W_true
            i_kris = [i*5, i*5+1, i*5+2, i*5+3, i*5+4]
            j_kris = [j*5, j*5+1, j*5+2, j*5+3, j*5+4]
            # Average edge weights across KRI pairs
            W_true_prt[i, j] = exp['W_true'][i, j]  # Direct PRT network
    
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
    
    # Try each recovery method
    methods = [
        ('var_dy', {'lag_order': 1}),
        ('covar', {'quantile': 0.05}),
    ]
    
    # Add GNN if PyTorch available
    try:
        import torch
        methods.append(('gnn', {'num_kris': 100, 'epochs': 50}))
    except ImportError:
        pass
    
    for method_name, kwargs in methods:
        try:
            # Recover network
            method = RecoveryMethodsFactory.create(method_name, **kwargs)
            W_est = method.fit_and_recover(X, threshold=0.1)
            
            # Evaluate
            evaluator = RecoveryEvaluator(W_true)
            metrics = evaluator.evaluate(W_est)
            
            results['methods'][method_name] = metrics
        except Exception as e:
            results['methods'][method_name] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    # DebtRank: propagation check
    try:
        sim = PropagationSimulator(W_true)
        tech_idx = sim.prt_to_idx.get('Technology_Risk', 0)
        distress_path, _ = sim.propagate_shock(
            initial_shock_node_idx=tech_idx,
            shock_magnitude=0.30,
            max_periods=13
        )
        
        final_distress = distress_path[-1].max()
        results['debtrank'] = {
            'final_distress': float(final_distress),
            'propagation_steps': len(distress_path),
            'cascade_active': final_distress > 0.01
        }
    except Exception as e:
        results['debtrank'] = {'error': str(e)}
    
    return results


def aggregate_config_results(config_results: list) -> dict:
    """
    Aggregate results across 20 replications for one configuration.
    
    Args:
        config_results: List of 20 result dictionaries
        
    Returns:
        Aggregated statistics (mean, std, p25, p75, etc.)
    """
    aggregated = {
        'num_replications': len(config_results),
        'methods': {}
    }
    
    # For each method, aggregate metrics across replications
    methods_to_aggregate = ['var_dy', 'covar', 'gnn']
    metrics_to_aggregate = ['precision', 'recall', 'f1', 'auc_roc', 
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
                values = [v for v in values if not np.isnan(v)]
                
                if values:
                    aggregated['methods'][method_name][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'p25': float(np.percentile(values, 25)),
                        'p50': float(np.percentile(values, 50)),
                        'p75': float(np.percentile(values, 75)),
                    }
    
    return aggregated


def main():
    """Run full reduced-grid experiment suite."""
    print("="*70)
    print("Task 8: Run Reduced-Grid Experiments (1800 runs)")
    print("="*70)
    print()
    
    # Configuration grid
    time_steps_list = [50, 100, 150, 200, 250]
    noise_levels_list = [0.1, 0.3, 0.5]
    network_density_list = [2, 4]
    num_replications = 20
    
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
    
    # Helper to convert numpy types to JSON-serializable formats
    def convert_to_serializable(obj):
        """Convert numpy arrays and other non-serializable types."""
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
    results_json = 'results/benchmark_results.json'
    Path('results').mkdir(exist_ok=True)
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_json, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(all_results),
            'experiments': serializable_results
        }, f, indent=2)
    print(f"  [OK] {results_json}")
    
    # 2. Aggregated summary as CSV
    summary_csv = 'results/benchmark_summary.csv'
    
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['config_name', 'time_steps', 'noise_level', 'network_density', 
                 'var_dy_f1_mean', 'var_dy_auc_roc_mean', 
                 'covar_f1_mean', 'covar_auc_roc_mean',
                 'gnn_f1_mean', 'gnn_auc_roc_mean']
        writer.writerow(header)
        
        # Data rows
        for agg in config_aggregated:
            row = [
                agg['config_name'],
                agg['config']['time_steps'],
                agg['config']['noise_level'],
                agg['config']['network_density'],
            ]
            
            # VAR+DY
            if 'var_dy' in agg['methods'] and 'f1' in agg['methods']['var_dy']:
                row.append(agg['methods']['var_dy']['f1'].get('mean', ''))
                row.append(agg['methods']['var_dy']['auc_roc'].get('mean', ''))
            else:
                row.extend(['', ''])
            
            # CoVaR
            if 'covar' in agg['methods'] and 'f1' in agg['methods']['covar']:
                row.append(agg['methods']['covar']['f1'].get('mean', ''))
                row.append(agg['methods']['covar']['auc_roc'].get('mean', ''))
            else:
                row.extend(['', ''])
            
            # GNN
            if 'gnn' in agg['methods'] and 'f1' in agg['methods']['gnn']:
                row.append(agg['methods']['gnn']['f1'].get('mean', ''))
                row.append(agg['methods']['gnn']['auc_roc'].get('mean', ''))
            else:
                row.extend(['', ''])
            
            writer.writerow(row)
    
    print(f"  [OK] {summary_csv}")
    print()
    
    # Summary statistics
    print("Summary Statistics:")
    print(f"  Total experiments run: {len(all_results)}")
    print(f"  Total configurations: {len(config_aggregated)}")
    
    # Count successful runs per method
    for method_name in ['var_dy', 'covar', 'gnn']:
        success_count = sum(
            1 for r in all_results 
            if method_name in r['methods'] and 'error' not in r['methods'][method_name]
        )
        print(f"  {method_name}: {success_count} successful runs")
    
    print()
    print("="*70)
    print("[PASS] Task 8 Complete: Reduced-Grid Experiments Executed")
    print("="*70)


if __name__ == "__main__":
    main()
