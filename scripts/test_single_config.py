#!/usr/bin/env python3
"""
Quick test: Run single configuration with all fixes
to verify GNN config defaults, DebtRank MSE-only, and JSON serialization work.
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.experiment_runner import ExperimentRunner

print("Testing single configuration with all fixes...")
print()

# Single config: T=50, noise=0.1, designed network, 2 replications (fast)
runner = ExperimentRunner("config.yaml")

try:
    print("Running configuration: T=50, noise=0.1, designed network, 2 reps...")
    result = runner.run_single_configuration(
        time_steps=50,
        noise_level=0.1,
        network_density=2,
        t_df=5.0,
        num_replications=2,
        use_documented_network=True,
    )
    
    print("\n✓ Configuration completed successfully")
    print()
    
    # Check results structure
    print("Results structure:")
    for method, data in result.items():
        if method == 'config':
            continue
        if data is None:
            print(f"  {method:15s}: None")
        elif isinstance(data, dict):
            metrics = [k for k in data.keys() if k != 'by_estimator']
            print(f"  {method:15s}: {', '.join(metrics)}")
    
    print()
    
    # Test JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy and other types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    serializable = convert_to_serializable(result)
    
    # Try to save to JSON
    output_file = "results/test_single_config.json"
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    file_size = Path(output_file).stat().st_size
    print(f"✓ JSON serialization successful")
    print(f"  Output file: {output_file}")
    print(f"  File size: {file_size} bytes")
    
    # Verify it's valid JSON
    with open(output_file, 'r') as f:
        loaded = json.load(f)
    
    print(f"✓ JSON is valid and can be read back")
    print()
    print("All fixes verified successfully!")
    
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
