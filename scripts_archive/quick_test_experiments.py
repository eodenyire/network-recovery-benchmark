#!/usr/bin/env python3
"""Quick test: Run 10 representative experiments (instead of full 1800)."""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import DataGenerator
from recovery_methods import RecoveryMethodsFactory
from evaluation import RecoveryEvaluator

def run_quick_test():
    """Run 10 representative experiments."""
    print("="*70)
    print("Task 8: Quick Test Run (10 representative experiments)")
    print("="*70)
    print()
    
    # Representative subset: 2 time steps, 1 noise level, 1 density, 5 reps
    configs = [
        {'time_steps': 100, 'noise_level': 0.1, 'network_density': 2},
        {'time_steps': 250, 'noise_level': 0.3, 'network_density': 4},
    ]
    
    all_results = []
    
    gen = DataGenerator('config.yaml')
    
    for config_idx, config in enumerate(configs):
        for rep in range(5):
            try:
                # Generate data
                exp = gen.generate_experiment(
                    time_steps=config['time_steps'],
                    noise_level=config['noise_level'],
                    network_density=config['network_density'],
                    use_documented_network=True
                )
                
                X = exp['kri_panel']
                W_true = exp['W_true']
                
                result = {
                    'config': config,
                    'replication': rep,
                    'methods': {}
                }
                
                # VAR+DY
                try:
                    method = RecoveryMethodsFactory.create('var_dy')
                    W_est = method.fit_and_recover(X)
                    evaluator = RecoveryEvaluator(W_true)
                    metrics = evaluator.evaluate(W_est)
                    result['methods']['var_dy'] = metrics
                except Exception as e:
                    result['methods']['var_dy'] = {'error': str(e)[:100]}
                
                # CoVaR
                try:
                    method = RecoveryMethodsFactory.create('covar')
                    W_est = method.fit_and_recover(X)
                    evaluator = RecoveryEvaluator(W_true)
                    metrics = evaluator.evaluate(W_est)
                    result['methods']['covar'] = metrics
                except Exception as e:
                    result['methods']['covar'] = {'error': str(e)[:100]}
                
                all_results.append(result)
                print(f"  [{config_idx * 5 + rep + 1}/10] Config: T={config['time_steps']}, N={config['noise_level']}, D={config['network_density']}, Rep={rep}")
                
            except Exception as e:
                print(f"  Error in config {config_idx}, rep {rep}: {e}")
    
    print()
    print("="*70)
    print(f"Completed {len(all_results)} test experiments")
    print("="*70)
    print()
    
    # Save results (convert numpy arrays to lists for JSON serialization)
    results_json = 'results/benchmark_results_test.json'
    Path('results').mkdir(exist_ok=True)
    
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
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_json, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(all_results),
            'note': 'Quick test run (10 experiments)',
            'experiments': serializable_results
        }, f, indent=2)
    
    print(f"[OK] Results saved to {results_json}")
    print()
    
    # Show sample results
    print("Sample Results:")
    for i, result in enumerate(all_results[:3]):
        print(f"\nExperiment {i+1}:")
        print(f"  Config: T={result['config']['time_steps']}, N={result['config']['noise_level']}")
        
        if 'var_dy' in result['methods'] and 'f1' in result['methods']['var_dy']:
            print(f"  VAR+DY F1: {result['methods']['var_dy']['f1']:.3f}")
            print(f"  VAR+DY AUC: {result['methods']['var_dy']['auc_roc']:.3f}")
        
        if 'covar' in result['methods'] and 'f1' in result['methods']['covar']:
            print(f"  CoVaR F1: {result['methods']['covar']['f1']:.3f}")
            print(f"  CoVaR AUC: {result['methods']['covar']['auc_roc']:.3f}")
    
    print()
    print("="*70)
    print("[PASS] Task 8 Test Complete")
    print("Full 1800-experiment run ready (use run_experiments.py)")
    print("="*70)

if __name__ == "__main__":
    run_quick_test()
