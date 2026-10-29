#!/usr/bin/env python3
"""
Validate robustness check: compare designed vs. random network results.
"""

import json

with open('results/task4_random_robustness_check.json') as f:
    data = json.load(f)

print("=" * 80)
print("ROBUSTNESS CHECK VALIDATION")
print("=" * 80)
print()

# Group by config_type
by_config_type = {}
for item in data:
    cfg_type = item.get('config_type', 'unknown')
    net_type = item.get('network_type', 'unknown')
    
    if cfg_type not in by_config_type:
        by_config_type[cfg_type] = {}
    
    by_config_type[cfg_type][net_type] = item

# Compare designed vs. random for each config type
for cfg_type in sorted(by_config_type.keys()):
    print(f"\n{cfg_type} Configuration:")
    print("-" * 80)
    
    configs = by_config_type[cfg_type]
    
    if 'designed' in configs and 'random' in configs:
        designed = configs['designed']
        random_net = configs['random']
        
        print(f"{'Estimator':<15} {'Designed AUC':<15} {'Random AUC':<15} {'Difference':<15}")
        print("-" * 60)
        
        for method in ['var_dy', 'covar', 'gnn']:
            if method in designed and method in random_net:
                des_data = designed.get(method)
                ran_data = random_net.get(method)
                
                if des_data and ran_data:
                    des_auc = des_data.get('auc_roc', {}).get('mean', None)
                    ran_auc = ran_data.get('auc_roc', {}).get('mean', None)
                    
                    if des_auc is not None and ran_auc is not None:
                        diff = des_auc - ran_auc
                        print(f"{method.upper():<15} {des_auc:<15.4f} {ran_auc:<15.4f} {diff:+.4f}")
                    else:
                        print(f"{method.upper():<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
                else:
                    print(f"{method.upper():<15} {'None':<15} {'None':<15} {'None':<15}")
    else:
        available = list(configs.keys())
        print(f"  Available network types: {available}")
        print(f"  (Designed: {'yes' if 'designed' in configs else 'NO'}, Random: {'yes' if 'random' in configs else 'NO'})")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)

# Check if we have both designed and random for comparison
has_both = False
for cfg_type in by_config_type:
    if 'designed' in by_config_type[cfg_type] and 'random' in by_config_type[cfg_type]:
        has_both = True
        break

if has_both:
    print("✓ Robustness check DATA PRESENT: Both designed and random networks compared")
else:
    print("✗ Robustness check INCOMPLETE: Missing comparison between designed and random networks")

print()
print("Per Ivan's requirement:")
print("  'add a robustness check with random sparse networks of the same density'")
print("  'so the results don't hinge on your design'")
print()

# Check if results show designed > random (or approximately equal)
if has_both:
    for cfg_type in by_config_type:
        configs = by_config_type[cfg_type]
        if 'designed' in configs and 'random' in configs:
            print(f"\n{cfg_type}:")
            
            for method in ['var_dy', 'covar']:
                des_data = configs['designed'].get(method)
                ran_data = configs['random'].get(method)
                
                if des_data and ran_data:
                    des_auc = des_data.get('auc_roc', {}).get('mean')
                    ran_auc = ran_data.get('auc_roc', {}).get('mean')
                    
                    if des_auc and ran_auc:
                        diff = des_auc - ran_auc
                        if abs(diff) < 0.02:
                            print(f"  {method.upper()}: Designed ≈ Random (diff: {diff:+.4f}) - NO DESIGN EFFECT")
                        elif des_auc > ran_auc:
                            print(f"  {method.upper()}: Designed > Random (diff: {diff:+.4f}) - Design carries information")
                        else:
                            print(f"  {method.upper()}: Random > Designed (diff: {diff:+.4f}) - UNEXPECTED")

print()
print("=" * 80)
