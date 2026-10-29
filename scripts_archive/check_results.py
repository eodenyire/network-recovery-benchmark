import json
d = json.load(open('results/benchmark_results_medium.json'))
print(f"Total experiments: {d['num_experiments']}")
print(f"Timestamp: {d['timestamp']}")
if d['num_experiments'] > 0:
    first_exp = d['experiments'][0]
    print(f"Sample config: T={first_exp['config']['time_steps']}, N={first_exp['config']['noise_level']}, D={first_exp['config']['network_density']}")
    print(f"Methods: {list(first_exp['methods'].keys())}")
