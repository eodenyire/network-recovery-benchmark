#!/usr/bin/env python3
"""
Full End-to-End SRTG-20 Benchmark Execution (v2)
Runs 180-configuration × 100-replication grid for publication-ready M6 capstone draft.

Experimental grid:
    Time steps: 50, 100, 150, 200, 250 (5 levels)
    Noise levels: 0.1, 0.3, 0.5 (3 levels)
    t-copula df: 3, 5, 10 (3 levels)
    Density: 2, 4 (2 levels)
    Network: designed, random (2 levels)
    Replications: 100 per config
    Total: 5 × 3 × 3 × 2 × 2 × 100 = 18,000 runs

Usage:
    python run_full_benchmark_v2.py
    python run_full_benchmark_v2.py --quick      (5 configs × 5 reps for testing)
    python run_full_benchmark_v2.py --medium     (30 configs × 20 reps)
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from synthetic_generator import DataGenerator
from recovery_methods import VARDieboldYilmaz, CoVaR
from propagation import PropagationSimulator
from evaluation import ConfusionMetrics


def setup_logging(log_file="results/logs/full_benchmark.log"):
    """Configure logging."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_trivial_baselines(W_true, density=2):
    """Compute trivial baseline metrics (predict-all, random-k)."""
    n = W_true.shape[0]
    n_edges_true = (W_true > 0).sum()
    n_possible = n * (n - 1)  # directed, no self-loops
    
    # Predict-all: declare every edge present
    predict_all_precision = n_edges_true / n_possible
    predict_all_recall = 1.0
    predict_all_f1 = 2 * predict_all_precision * predict_all_recall / (predict_all_precision + predict_all_recall)
    predict_all_auc = 0.5  # no discriminative signal
    
    return {
        'predict_all': {
            'precision': predict_all_precision,
            'recall': predict_all_recall,
            'f1_score': predict_all_f1,
            'auc_roc': predict_all_auc
        }
    }


def aggregate_kri_to_prt(W_kri_100x100):
    """Aggregate KRI-level (100×100) to PRT-level (20×20) by summing blocks."""
    W_prt = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            i_start, i_end = i * 5, (i + 1) * 5
            j_start, j_end = j * 5, (j + 1) * 5
            W_prt[i, j] = W_kri_100x100[i_start:i_end, j_start:j_end].sum()
    
    # Row-normalize
    row_sums = W_prt.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W_prt = W_prt / row_sums
    
    return W_prt


def run_single_config(data_gen, time_steps, noise_level, t_df, density, net_source, num_reps, logger):
    """Run a single configuration and return aggregated results."""
    
    config_name = f"T{time_steps}_N{noise_level:.1f}_DF{t_df:.0f}_D{density}_{net_source}"
    logger.info(f"Configuration: {config_name}")
    
    results_per_rep = {
        'var_dy': [],
        'covar': [],
    }
    
    for rep in range(num_reps):
        try:
            # Generate synthetic data
            exp = data_gen.generate_experiment(
                time_steps=time_steps,
                noise_level=noise_level,
                network_density=density,
                t_df=t_df,
                use_documented_network=(net_source == 'designed')
            )
            
            X = exp['kri_panel']
            W_true_prt = exp['W_true']  # (20, 20)
            
            # VAR+DY recovery
            try:
                var_dy = VARDieboldYilmaz(lag_order=1)
                W_var_dy_kri = var_dy.fit_and_recover(X, threshold=0.1)  # (100, 100)
                
                # Aggregate to PRT level
                W_var_dy_prt = aggregate_kri_to_prt(W_var_dy_kri)
                
                # Evaluate
                metrics_var = ConfusionMetrics.compute_at_threshold(W_var_dy_prt, W_true_prt, threshold=0.1)
                results_per_rep['var_dy'].append(metrics_var)
            except Exception as e:
                logger.warning(f"  VAR+DY failed rep {rep}: {e}")
            
            # CoVaR recovery
            try:
                covar = CoVaR(quantile=0.05)
                W_covar_kri = covar.fit_and_recover(X, threshold=0.1)  # (100, 100)
                
                # Aggregate to PRT level
                W_covar_prt = aggregate_kri_to_prt(W_covar_kri)
                
                # Evaluate
                metrics_covar = ConfusionMetrics.compute_at_threshold(W_covar_prt, W_true_prt, threshold=0.1)
                results_per_rep['covar'].append(metrics_covar)
            except Exception as e:
                logger.warning(f"  CoVaR failed rep {rep}: {e}")
        
        except Exception as e:
            logger.error(f"  Data generation failed rep {rep}: {e}")
    
    # Aggregate
    aggregated = {
        'config': {
            'time_steps': time_steps,
            'noise_level': noise_level,
            't_df': t_df,
            'density': density,
            'network_source': net_source
        },
        'var_dy': None,
        'covar': None,
    }
    
    # Aggregate VAR+DY
    if results_per_rep['var_dy']:
        var_dy_data = pd.DataFrame(results_per_rep['var_dy'])
        aggregated['var_dy'] = {
            'mean': var_dy_data.mean().to_dict(),
            'std': var_dy_data.std().to_dict(),
        }
    
    # Aggregate CoVaR
    if results_per_rep['covar']:
        covar_data = pd.DataFrame(results_per_rep['covar'])
        aggregated['covar'] = {
            'mean': covar_data.mean().to_dict(),
            'std': covar_data.std().to_dict(),
        }
    
    return aggregated


def run_full_benchmark(mode='full'):
    """Execute full benchmark according to mode."""
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("SRTG-20 FULL BENCHMARK EXECUTION")
    logger.info("=" * 80)
    logger.info(f"Mode: {mode.upper()}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_gen = DataGenerator('config.yaml')
    
    # Define grid based on mode
    if mode == 'quick':
        time_steps_list = [250]
        noise_levels_list = [0.1]
        t_dfs_list = [5]
        densities_list = [2]
        net_sources_list = ['designed']
        num_reps = 5
        logger.info("QUICK TEST: 1×1×1×1×1×5 = 5 total runs")
    
    elif mode == 'medium':
        time_steps_list = [50, 100, 150, 200, 250]
        noise_levels_list = [0.1, 0.3, 0.5]
        t_dfs_list = [5]
        densities_list = [2, 4]
        net_sources_list = ['designed', 'random']
        num_reps = 20
        logger.info("MEDIUM GRID: 5×3×1×2×2×20 = 1,200 total runs")
    
    else:  # full
        time_steps_list = [50, 100, 150, 200, 250]
        noise_levels_list = [0.1, 0.3, 0.5]
        t_dfs_list = [3, 5, 10]
        densities_list = [2, 4]
        net_sources_list = ['designed', 'random']
        num_reps = 100
        logger.info("FULL GRID: 5×3×3×2×2×100 = 18,000 total runs")
    
    num_configs = (len(time_steps_list) * len(noise_levels_list) * len(t_dfs_list) *
                   len(densities_list) * len(net_sources_list))
    total_runs = num_configs * num_reps
    
    logger.info(f"Total configurations: {num_configs}")
    logger.info(f"Replications per config: {num_reps}")
    logger.info(f"Total experiments: {total_runs}")
    logger.info(f"Estimated time (5 sec/exp): {total_runs * 5 / 3600:.1f} hours\n")
    
    # Execute grid
    all_results = []
    start_time = time.time()
    config_idx = 0
    
    for T in time_steps_list:
        for sigma in noise_levels_list:
            for df in t_dfs_list:
                for density in densities_list:
                    for net_src in net_sources_list:
                        config_idx += 1
                        
                        try:
                            result = run_single_config(
                                data_gen, T, sigma, df, density, net_src, num_reps, logger
                            )
                            all_results.append(result)
                            logger.info(f"  ✓ Completed [{config_idx}/{num_configs}]")
                        except Exception as e:
                            logger.error(f"  ✗ Failed: {e}")
                            raise
                        
                        # Progress
                        elapsed = time.time() - start_time
                        rate = (elapsed / config_idx) if config_idx > 0 else 0
                        eta_sec = (num_configs - config_idx) * rate
                        logger.info(f"  Progress: {config_idx}/{num_configs} | ETA: {eta_sec/3600:.1f}h\n")
    
    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_file = results_dir / f"benchmark_results_{mode}.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results: {json_file}")
    
    # Create CSV summary
    summary_data = []
    for result in all_results:
        row = result['config'].copy()
        
        if result['var_dy']:
            row['var_dy_precision_mean'] = result['var_dy']['mean'].get('precision', np.nan)
            row['var_dy_f1_mean'] = result['var_dy']['mean'].get('f1_score', np.nan)
            row['var_dy_auc_roc_mean'] = result['var_dy']['mean'].get('auc_roc', np.nan)
        
        if result['covar']:
            row['covar_precision_mean'] = result['covar']['mean'].get('precision', np.nan)
            row['covar_f1_mean'] = result['covar']['mean'].get('f1_score', np.nan)
            row['covar_auc_roc_mean'] = result['covar']['mean'].get('auc_roc', np.nan)
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    csv_file = results_dir / f"benchmark_summary_{mode}.csv"
    summary_df.to_csv(csv_file, index=False)
    logger.info(f"Summary: {csv_file}")
    
    total_time = time.time() - start_time
    logger.info(f"\n{'=' * 80}")
    logger.info(f"BENCHMARK COMPLETE")
    logger.info(f"Total time: {total_time / 3600:.2f} hours")
    logger.info(f"Avg per config: {total_time / num_configs:.1f} seconds")
    logger.info(f"{'=' * 80}")
    
    return all_results, summary_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SRTG-20 benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 runs)")
    parser.add_argument("--medium", action="store_true", help="Medium grid (1,200 runs)")
    args = parser.parse_args()
    
    mode = 'quick' if args.quick else ('medium' if args.medium else 'full')
    
    try:
        results, summary = run_full_benchmark(mode=mode)
        sys.exit(0)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
