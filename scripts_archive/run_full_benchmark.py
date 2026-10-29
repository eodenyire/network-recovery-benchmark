#!/usr/bin/env python
"""
Full End-to-End SRTG-20 Benchmark Execution
Runs complete 180-configuration × 100-replication grid for publication-ready M6 capstone draft.

Usage:
    python run_full_benchmark.py [--quick] [--resume]
    
Options:
    --quick     Run reduced grid (for testing)
    --resume    Resume from last checkpoint if available
"""

import sys
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiment_runner import ExperimentRunner
from synthetic_generator import PlantedNetworkGenerator, TCopulaVARProcess
from random_networks import RandomNetworkGenerator
from recovery_methods import VARDieboldYilmaz, CoVaREstimator, GATGraphNeuralNetwork
from propagation import PropagationSimulator
from evaluation import compute_all_metrics, trivial_baselines


def setup_logging(log_file="results/logs/full_benchmark.log"):
    """Configure logging for full benchmark run."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_full_benchmark(quick=False, resume=False):
    """Execute full experimental grid."""
    logger = setup_logging()
    config = load_config()
    
    logger.info("=" * 80)
    logger.info("SRTG-20 FULL BENCHMARK EXECUTION")
    logger.info("=" * 80)
    logger.info(f"Study: {config['study']['name']}")
    logger.info(f"Version: {config['study']['version']}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Determine grid parameters
    if quick:
        logger.info("MODE: Quick test (reduced grid)")
        time_steps = [250]
        noise_levels = [0.1]
        tail_dfs = [5]
        densities = [2]
        network_sources = ["designed"]
        replications = 5
    else:
        logger.info("MODE: Full production grid")
        time_steps = config['experimental_grid']['time_steps']
        noise_levels = config['experimental_grid']['noise_levels']
        tail_dfs = config['experimental_grid'].get('tail_dependence_nu', [3, 5, 10])
        densities = config['experimental_grid']['network_density']
        network_sources = config['experimental_grid'].get('network_sources', ['designed', 'random'])
        replications = config['experimental_grid']['replications']
    
    # Calculate grid size
    num_configs = len(time_steps) * len(noise_levels) * len(tail_dfs) * len(densities) * len(network_sources)
    total_runs = num_configs * replications
    
    logger.info(f"Experimental grid: {len(time_steps)} × {len(noise_levels)} × {len(tail_dfs)} × {len(densities)} × {len(network_sources)} = {num_configs} configurations")
    logger.info(f"Replications per config: {replications}")
    logger.info(f"Total runs: {total_runs}")
    logger.info(f"Estimated time (assuming 2 sec/run): {total_runs * 2 / 3600:.1f} hours")
    
    # Initialize experiment runner
    runner = ExperimentRunner(config_path="config.yaml")
    
    # Check for checkpoint
    checkpoint_dir = Path(config['study']['checkpoint_dir'])
    checkpoint_file = checkpoint_dir / "full_benchmark_checkpoint.json"
    
    start_config = 0
    if resume and checkpoint_file.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        start_config = checkpoint.get('last_config_idx', 0)
        logger.info(f"Resuming from config {start_config}/{num_configs}")
    
    # Collect results
    all_results = []
    start_time = time.time()
    
    config_idx = 0
    for T in time_steps:
        for sigma in noise_levels:
            for nu in tail_dfs:
                for density in densities:
                    for net_src in network_sources:
                        if config_idx < start_config:
                            config_idx += 1
                            continue
                        
                        config_spec = {
                            'time_steps': T,
                            'noise_level': sigma,
                            'tail_dependence_nu': nu,
                            'network_density': density,
                            'network_source': net_src,
                            'replications': replications
                        }
                        
                        logger.info(f"\n[{config_idx+1}/{num_configs}] Configuration: T={T}, σ={sigma}, ν={nu}, density={density}, source={net_src}")
                        logger.info(f"  Replications: {replications}")
                        
                        # Run configuration
                        try:
                            config_results = runner.run_single_configuration(
                                time_steps=T,
                                noise_level=sigma,
                                tail_dependence_nu=nu,
                                network_density=density,
                                network_source=net_src,
                                num_replications=replications
                            )
                            
                            all_results.extend(config_results)
                            logger.info(f"  ✓ Completed: {len(config_results)} replications")
                            
                        except Exception as e:
                            logger.error(f"  ✗ Failed: {e}")
                            raise
                        
                        # Save checkpoint
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        with open(checkpoint_file, 'w') as f:
                            json.dump({'last_config_idx': config_idx + 1, 'timestamp': datetime.now().isoformat()}, f)
                        
                        # Print progress
                        elapsed = time.time() - start_time
                        rate = (config_idx + 1) / (elapsed / 3600) if elapsed > 0 else 0
                        eta_hours = (num_configs - config_idx - 1) / rate if rate > 0 else 0
                        logger.info(f"  Progress: {config_idx+1}/{num_configs} ({100*(config_idx+1)/num_configs:.1f}%) | Rate: {rate:.1f} configs/hr | ETA: {eta_hours:.1f}h")
                        
                        config_idx += 1
    
    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_file = results_dir / "benchmark_results_full.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved: {json_file}")
    
    # Generate summary
    logger.info("Generating summary statistics...")
    summary = runner.summary_table()
    csv_file = results_dir / "benchmark_summary_full.csv"
    summary.to_csv(csv_file, index=False)
    logger.info(f"Summary saved: {csv_file}")
    
    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    total_time = time.time() - start_time
    logger.info(f"\n{'=' * 80}")
    logger.info(f"BENCHMARK COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total experiments: {len(all_results)}")
    logger.info(f"Total time: {total_time / 3600:.2f} hours ({total_time / 60:.1f} minutes)")
    logger.info(f"Average time per config: {total_time / num_configs:.1f} seconds")
    logger.info(f"Results file: {json_file}")
    logger.info(f"Summary file: {csv_file}")
    
    return all_results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full SRTG-20 benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick test (reduced grid)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    try:
        results, summary = run_full_benchmark(quick=args.quick, resume=args.resume)
        sys.exit(0)
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
