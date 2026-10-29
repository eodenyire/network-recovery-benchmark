import json
import numpy as np

# Read the reduced benchmark results
with open('results/task4_reduced_benchmark.json', 'r') as f:
    results = json.load(f)

print(f'Total configurations: {len(results)}')

# Find easiest cell: T=250, noise=0.1, density=2
easiest = None
for r in results:
    cfg = r['config']
    if cfg['time_steps'] == 250 and cfg['noise_level'] == 0.1 and cfg['network_density'] == 2:
        easiest = r
        break

if easiest:
    print(f'\n=== EASIEST CELL: T=250, noise=0.1, density=2 ===\n')
    
    # Extract metrics for VAR+DY
    print('VAR+DY:')
    for metric in ['auc_roc', 'auprc', 'f1_score', 'edge_weight_correlation', 'impulse_response_mse']:
        if metric in easiest['var_dy']:
            val = easiest['var_dy'][metric]
            if isinstance(val, dict):
                print(f'  {metric}: {val["mean"]:.4f} ± {val["std"]:.4f}')
            else:
                print(f'  {metric}: {val:.4f}')
    
    # Extract metrics for CoVaR
    print('\nCoVaR:')
    for metric in ['auc_roc', 'auprc', 'f1_score', 'edge_weight_correlation', 'impulse_response_mse']:
        if metric in easiest['covar']:
            val = easiest['covar'][metric]
            if isinstance(val, dict):
                print(f'  {metric}: {val["mean"]:.4f} ± {val["std"]:.4f}')
            else:
                print(f'  {metric}: {val:.4f}')
    
    # Extract metrics for GNN if available
    print('\nGNN:')
    if 'gnn' in easiest and easiest['gnn'] is not None:
        for metric in ['auc_roc', 'auprc', 'f1_score', 'edge_weight_correlation', 'impulse_response_mse']:
            if metric in easiest['gnn']:
                val = easiest['gnn'][metric]
                if isinstance(val, dict):
                    print(f'  {metric}: {val["mean"]:.4f} ± {val["std"]:.4f}')
                else:
                    print(f'  {metric}: {val:.4f}')
    else:
        print('  [No GNN results in this benchmark run]')
    
    print('\n=== TRIVIAL BASELINES ===')
    print('Predict-all (declare all 400 edges present):')
    print('  Precision: 0.0975 (40 true / 400 predicted)')
    print('  Recall: 1.0000')
    print('  F1@oracle: 0.1777')
    print('  F1@calibrated: 0.1777')
    print('  AUC-ROC: 0.5000')
    
    print('\nRandom-k (randomly select 40 edges):')
    print('  Mean AUC-ROC: 0.5000 (by definition)')
    print('  Mean F1@oracle: ~0.11')
    
else:
    print('\nEasiest cell (T=250, noise=0.1, density=2) NOT FOUND')
