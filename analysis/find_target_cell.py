import json

with open('results/task4_reduced_benchmark.json', 'r') as f:
    data = json.load(f)

print('All configs in task4_reduced_benchmark.json:')
for i, item in enumerate(data):
    cfg = item['config']
    T = cfg['time_steps']
    sigma = cfg['noise_level']
    D = cfg['network_density']
    print(f'{i}: T={T}, sigma={sigma}, D={D}')

# Find T=250, σ=0.1, D=2
target = None
for item in data:
    cfg = item['config']
    if cfg['time_steps'] == 250 and cfg['noise_level'] == 0.1 and cfg['network_density'] == 2:
        target = item
        break

if target:
    print(f'\n*** Found T=250, sigma=0.1, D=2 ***')
    print(f'VAR+DY AUC-ROC: {target["var_dy"]["auc_roc"]["mean"]:.6f}')
    print(f'CoVaR AUC-ROC: {target["covar"]["auc_roc"]["mean"]:.6f}')
    print(f'DebtRank AUC-ROC PRESENT: {"auc_roc" in target["debtrank"]}  <-- VIOLATION OF B3')
    print(f'DebtRank F1 PRESENT: {"f1_score" in target["debtrank"]}  <-- VIOLATION OF B3')
    print(f'DebtRank IR-MSE: {target["debtrank"]["impulse_response_mse"]["mean"]:.10f}')
else:
    print('\nT=250, sigma=0.1, D=2 NOT FOUND')
