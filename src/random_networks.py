"""
Random Sparse Network Generator
================================
Generates random directed weighted networks with controlled density.

Purpose: Robustness check for the recovery benchmark.
- Run recovery methods on data generated from documented W_true
- Also run on data generated from random sparse networks of same density
- Compare: if recovery performance depends on W_true design (not just density),
  methods are learning network structure, not just exploiting statistical artifacts

Design:
- Match density of srtg20_edges.csv (41 edges / 400 possible = 10.25%)
- Generate random adjacencies with same sparsity pattern
- Vary random seed across replications for different network realizations
"""

import numpy as np
from typing import List


class RandomNetworkGenerator:
    """Generate random directed weighted networks with controlled density."""
    
    def __init__(self, num_nodes: int = 20, density: float = None, edges_per_node: int = 2, seed: int = 42):
        """
        Args:
            num_nodes: Number of nodes (20 PRTs)
            density: Target edge density (alternative to edges_per_node).
                     If provided, edges_per_node is ignored.
                     density = num_edges / (num_nodes * num_nodes)
            edges_per_node: Average number of edges per node (if density not specified)
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.seed = seed
        np.random.seed(seed)
        
        if density is not None:
            # Calculate edges_per_node from density
            self.density = density
            self.target_edges = int(density * num_nodes * num_nodes)
            self.edges_per_node = self.target_edges / num_nodes
        else:
            self.edges_per_node = edges_per_node
            self.density = (edges_per_node * num_nodes) / (num_nodes * num_nodes)
            self.target_edges = int(edges_per_node * num_nodes)
    
    def generate_erdos_renyi(self) -> np.ndarray:
        """
        Generate random network using Erdos-Renyi model.
        
        Each edge (i,j) is included independently with probability p.
        
        Returns:
            W (num_nodes, num_nodes): Random weighted adjacency
        """
        # Calculate edge probability to achieve target density
        p = self.density
        
        # Generate random edges
        W = np.random.binomial(1, p, size=(self.num_nodes, self.num_nodes)).astype(float)
        
        # Remove self-loops
        np.fill_diagonal(W, 0)
        
        # Assign random weights from (0.25, 0.75)
        nonzero_mask = W > 0
        W[nonzero_mask] = np.random.uniform(0.25, 0.75, size=np.sum(nonzero_mask))
        
        return W
    
    def generate_configuration_model(self) -> np.ndarray:
        """
        Generate random network with fixed out-degree sequence.
        
        Each node has approximately edges_per_node outgoing edges.
        Edges are then randomly rewired to create the network.
        
        Returns:
            W (num_nodes, num_nodes): Random weighted adjacency
        """
        W = np.zeros((self.num_nodes, self.num_nodes))
        
        for source in range(self.num_nodes):
            # Number of outgoing edges from this node
            num_edges = int(self.edges_per_node)
            if np.random.rand() < (self.edges_per_node - num_edges):
                num_edges += 1
            
            # Select random target nodes
            targets = np.random.choice(
                self.num_nodes, 
                size=min(num_edges, self.num_nodes - 1), 
                replace=False
            )
            
            # Remove self-loops if accidentally selected
            targets = targets[targets != source]
            
            # Assign weights
            for target in targets:
                W[source, target] = np.random.uniform(0.25, 0.75)
        
        return W
    
    def generate_scale_free(self, power_law_exp: float = 2.0) -> np.ndarray:
        """
        Generate random scale-free network (power-law degree distribution).
        
        Simulate preferential attachment: high-degree nodes attract more edges.
        
        Args:
            power_law_exp: Exponent of power-law distribution (typical 2.0-3.0)
            
        Returns:
            W (num_nodes, num_nodes): Random weighted adjacency
        """
        W = np.zeros((self.num_nodes, self.num_nodes))
        
        # Out-degree follows power-law distribution
        degrees = np.random.pareto(power_law_exp, size=self.num_nodes) + 1
        degrees = (degrees / degrees.sum() * self.target_edges).astype(int)
        
        for source in range(self.num_nodes):
            # Number of outgoing edges (at most num_nodes - 1)
            num_edges = min(degrees[source], self.num_nodes - 1)
            if num_edges == 0:
                continue
            
            # Select target nodes with probability proportional to in-degree
            in_degrees = W.sum(axis=0)
            in_degrees[source] = 0  # Avoid self-loops
            
            # Preferential attachment
            if in_degrees.sum() > 0:
                probs = in_degrees / in_degrees.sum()
            else:
                probs = np.ones(self.num_nodes)
                probs[source] = 0
                probs = probs / probs.sum()
            
            # Avoid choosing already-connected targets
            available = np.where(W[source, :] == 0)[0]
            num_edges = min(num_edges, len(available))
            
            if num_edges > 0:
                probs_available = probs[available]
                probs_available = probs_available / probs_available.sum()
                
                targets = np.random.choice(
                    available,
                    size=num_edges,
                    replace=False,
                    p=probs_available
                )
                
                # Assign weights
                for target in targets:
                    W[source, target] = np.random.uniform(0.25, 0.75)
        
        return W
    
    def generate_small_world(self, rewire_prob: float = 0.3) -> np.ndarray:
        """
        Generate small-world network (Watts-Strogatz model).
        
        Start with ring topology, then rewire edges with probability p.
        
        Args:
            rewire_prob: Probability of rewiring each edge
            
        Returns:
            W (num_nodes, num_nodes): Random weighted adjacency
        """
        W = np.zeros((self.num_nodes, self.num_nodes))
        
        # Ring topology: each node connects to nearest neighbors
        k = int(self.edges_per_node)  # Number of neighbors on each side
        
        for i in range(self.num_nodes):
            for j in range(1, k + 1):
                target = (i + j) % self.num_nodes
                
                # Rewire with probability rewire_prob
                if np.random.rand() < rewire_prob:
                    # Connect to random node instead
                    target = np.random.randint(0, self.num_nodes)
                    while target == i or W[i, target] > 0:
                        target = np.random.randint(0, self.num_nodes)
                
                W[i, target] = np.random.uniform(0.25, 0.75)
        
        # Remove self-loops
        np.fill_diagonal(W, 0)
        
        return W
    
    def generate(self, model: str = "erdos_renyi") -> np.ndarray:
        """
        Generate random network using specified model.
        
        Args:
            model: 'erdos_renyi', 'configuration', 'scale_free', or 'small_world'
            
        Returns:
            W (num_nodes, num_nodes): Random weighted adjacency
        """
        np.random.seed(self.seed)
        
        if model.lower() == "erdos_renyi":
            return self.generate_erdos_renyi()
        elif model.lower() == "configuration":
            return self.generate_configuration_model()
        elif model.lower() == "scale_free":
            return self.generate_scale_free()
        elif model.lower() == "small_world":
            return self.generate_small_world()
        else:
            raise ValueError(f"Unknown model: {model}")


def generate_random_networks_ensemble(
    num_nodes: int = 20,
    density: float = 0.1025,
    num_networks: int = 10,
    models: List[str] = None,
    seed_base: int = 1000
) -> dict:
    """
    Generate an ensemble of random networks for robustness check.
    
    Args:
        num_nodes: Number of nodes
        density: Target edge density (matches srtg20_edges.csv)
        num_networks: Number of random networks to generate per model
        models: List of models to use (default: all)
        seed_base: Base seed for random number generation
        
    Returns:
        Dictionary with networks keyed by (model, replication_id)
    """
    if models is None:
        models = ["erdos_renyi", "configuration", "scale_free", "small_world"]
    
    networks = {}
    
    for model in models:
        for rep in range(num_networks):
            seed = seed_base + rep
            gen = RandomNetworkGenerator(
                num_nodes=num_nodes,
                density=density,
                seed=seed
            )
            W = gen.generate(model=model)
            networks[(model, rep)] = W
    
    return networks


if __name__ == "__main__":
    print("="*60)
    print("Task 7: Random Sparse Network Generator")
    print("="*60)
    print()
    
    # Generate random networks
    print("[1] Generating random networks (density=0.1025, 20 nodes)...")
    gen = RandomNetworkGenerator(num_nodes=20, density=0.1025, seed=42)
    
    for model in ["erdos_renyi", "configuration", "small_world"]:
        W = gen.generate(model=model)
        num_edges = np.count_nonzero(W)
        density = num_edges / 400
        print(f"    {model:20s}: {num_edges:3d} edges (density={density:.4f})")
    
    print()
    print("[2] Generating ensemble (10 networks per model)...")
    networks = generate_random_networks_ensemble(
        num_nodes=20,
        density=0.1025,
        num_networks=10,
        models=["erdos_renyi", "configuration", "small_world"],
        seed_base=1000
    )
    print(f"    Total networks: {len(networks)}")
    
    for (model, rep), W in list(networks.items())[:4]:
        print(f"      ({model}, rep={rep}): {np.count_nonzero(W)} edges")
    
    print()
    print("="*60)
    print("[PASS] Task 7 Complete")
    print("      Random networks ready for robustness benchmarking")
    print("="*60)
