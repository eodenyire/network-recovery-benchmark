#!/usr/bin/env python3
"""Minimal test to find the hang."""

import sys
import os
import time

print(f"[{time.time()}] Starting minimal test")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print(f"[{time.time()}] Path set, importing modules...")

try:
    print(f"[{time.time()}] Importing numpy...")
    import numpy as np
    print(f"[{time.time()}] ✓ numpy imported")
    
    print(f"[{time.time()}] Importing pandas...")
    import pandas as pd
    print(f"[{time.time()}] ✓ pandas imported")
    
    print(f"[{time.time()}] Importing synthetic_generator...")
    from synthetic_generator import PlantedNetworkGenerator
    print(f"[{time.time()}] ✓ synthetic_generator imported")
    
    print(f"[{time.time()}] Importing propagation...")
    from propagation import PropagationSimulator
    print(f"[{time.time()}] ✓ propagation imported")
    
    print(f"[{time.time()}] Creating PlantedNetworkGenerator...")
    gen = PlantedNetworkGenerator(num_prts=20, edge_file='data/srtg20_edges.csv')
    print(f"[{time.time()}] ✓ PlantedNetworkGenerator created")
    
    print(f"[{time.time()}] Calling generate()...")
    W_true = gen.generate()
    print(f"[{time.time()}] ✓ generate() returned, shape={W_true.shape}")
    
    print(f"[{time.time()}] Creating PropagationSimulator...")
    sim = PropagationSimulator(W_true)
    print(f"[{time.time()}] ✓ PropagationSimulator created")
    
    print(f"[{time.time()}] ✓ All imports and setup complete!")
    
except Exception as e:
    print(f"[{time.time()}] ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
