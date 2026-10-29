#!/usr/bin/env python3
"""Debug the AUC calculation."""

import numpy as np
from sklearn.metrics import roc_auc_score

# Create test data
W_true = np.array([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]])  # 4 true edges out of 9

W_est_all = np.ones((3, 3))  # Predict all edges
np.fill_diagonal(W_est_all, 0)  # No self-loops

# Flatten
w_est_flat = W_est_all.flatten()
w_true_flat = (W_true > 0).astype(int).flatten()

print(f"W_true:\n{W_true}")
print(f"W_est (all 1s):\n{W_est_all}")
print()
print(f"w_true_flat: {w_true_flat}")
print(f"w_est_flat: {w_est_flat}")
print()

# Compute AUC
try:
    auc = roc_auc_score(w_true_flat, w_est_flat)
    print(f"AUC-ROC (sklearn): {auc:.6f}")
except Exception as e:
    print(f"Error: {e}")

# Also try with actual 20x20
print()
print("Testing with 20x20 network:")

from src.synthetic_generator import PlantedNetworkGenerator

gen = PlantedNetworkGenerator(num_prts=20, edge_file='data/srtg20_edges.csv')
W_true_20 = gen.generate()

W_est_all_20 = np.ones((20, 20))
np.fill_diagonal(W_est_all_20, 0)

w_est_flat_20 = W_est_all_20.flatten()
w_true_flat_20 = (W_true_20 > 0).astype(int).flatten()

print(f"True edges: {w_true_flat_20.sum()}")
print(f"Predicted edges: {(w_est_flat_20 > 0).sum()}")

try:
    auc_20 = roc_auc_score(w_true_flat_20, w_est_flat_20)
    print(f"AUC-ROC (sklearn): {auc_20:.6f}")
except Exception as e:
    print(f"Error: {e}")
