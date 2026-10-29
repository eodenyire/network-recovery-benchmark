import json
d = json.load(open('results/task4_reduced_benchmark.json'))
# Check CoVaR metrics for first few configs
for idx in range(min(3, len(d))):
    config = d[idx]
    print(f'\nConfig {idx} (T={config["config"]["time_steps"]}, σ={config["config"]["noise_level"]}):')
    print(f"  CoVaR precision: {config['covar']['precision']['mean']:.4f}")
    print(f"  CoVaR recall: {config['covar']['recall']['mean']:.4f}")
    print(f"  CoVaR F1: {config['covar']['f1_score']['mean']:.4f}")
    print(f"  CoVaR AUC: {config['covar']['auc_roc']['mean']:.4f}")
    print(f"  CoVaR edge_weight_correlation: {config['covar']['edge_weight_correlation']['mean']:.4f}")
