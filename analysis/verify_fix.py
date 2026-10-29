#!/usr/bin/env python3
"""
Verification: Single experiment to test aggregation fix.
Run at T=250, N=0.1, D=2, replication=0
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_generator import DataGenerator
from recovery_methods import RecoveryMethodsFactory
from evaluation import RecoveryEvaluator

def verify_aggregation():
    """Run one experiment and verify the aggregation fix."""
    print("="*70)
    print("Verification: Single Experiment with Aggregation Fix")
    print("="*70)
    print()
    
    # Configuration
    t, sigma, density = 250, 0.1, 2
    
    print(f"Config: T={t}, N={sigma}, D={density}")
    print()
    
    # Generate synthetic data
    gen = DataGenerator('config.yaml')
    exp = gen.generate_experiment(
        time_steps=t,
        noise_level=sigma,
        network_density=density,
        use_documented_network=True
    )
    
    X = exp['kri_panel']
    W_true = exp['W_true']
    
    print(f"Data generated: X.shape={X.shape}, W_true.shape={W_true.shape}")
    print()
    
    # VAR+DY Recovery
    print("Running VAR+DY recovery...")
    method = RecoveryMethodsFactory.create('var_dy')
    W_est = method.fit_and_recover(X, threshold=0.1)
    
    print(f"W_est.shape={W_est.shape}, std={W_est.std():.6f}")
    print(f"W_est min={W_est.min():.6f}, max={W_est.max():.6f}")
    print()
    
    # Evaluate
    print("Evaluating with aggregation...")
    evaluator = RecoveryEvaluator(W_true)
    metrics = evaluator.evaluate(W_est)
    
    print()
    print("Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  Edge weight correlation: {metrics['edge_weight_correlation']:.4f}")
    print(f"  IR-MSE: {metrics['impulse_response_mse']:.6f}")
    print()
    
    # Verification
    print("Verification:")
    print(f"  ✓ Recall is NOT 1.0 (should be {metrics['recall']:.4f})")
    print(f"  ✓ Precision is NOT trivial (should be {metrics['precision']:.4f})")
    print(f"  ✓ AUC-ROC > 0.60? {metrics['auc_roc'] > 0.60} (AUC={metrics['auc_roc']:.4f})")
    print(f"  ✓ F1 > 0.2? {metrics['f1_score'] > 0.2} (F1={metrics['f1_score']:.4f})")
    print()
    
    if metrics['auc_roc'] > 0.60 and metrics['f1_score'] > 0.2:
        print("="*70)
        print("[PASS] Aggregation fix is working!")
        print("="*70)
        return True
    else:
        print("="*70)
        print("[FAIL] Metrics still poor. Check aggregation or FEVD computation.")
        print("="*70)
        return False

if __name__ == "__main__":
    success = verify_aggregation()
    sys.exit(0 if success else 1)
