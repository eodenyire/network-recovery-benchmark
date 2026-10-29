"""
Synthetic Data Generator for Capstone Methods Study
=====================================================
Generates controlled synthetic KRI data with a planted risk-propagation network.

Process:
1. Plant a known directed weighted network on 20 PRTs
2. Generate VAR(1) latent dynamics with persistent AR coefficients
3. Apply Student's t-copula to impose tail dependence (not Gaussian)
4. Map latent values to observable KRI scales
5. Add measurement noise
6. Return both the synthetic KRI panel and the ground-truth network

Key Features:
- Ground-truth network is fixed and known (no circularity)
- t-copula with ν=5 captures simultaneous extreme co-movements
- 100 KRI series from 20 PRTs (5 KRIs per PRT)
- 250 monthly observations per series
- Experimental factors: data volume, noise, tail dependence, network density
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t, pearsonr
from scipy.linalg import cholesky
import networkx as nx
from typing import Tuple, Dict, Any
import yaml


class PlantedNetworkGenerator:
    """
    Load or generate a fixed, known directed weighted network on PRT nodes.
    
    If edge_file is provided, reads documented edges from CSV.
    Otherwise, generates a random sparse network.
    """
    
    def __init__(self, num_prts: int = 20, edges_per_node: int = 2, seed: int = 42, edge_file: str = None):
        """
        Args:
            num_prts: Number of PRT nodes (default 20)
            edges_per_node: Average number of incoming/outgoing edges per node (if generating random)
            seed: Random seed for reproducibility
            edge_file: Path to CSV with documented edges (e.g., srtg20_edges.csv)
        """
        self.num_prts = num_prts
        self.edges_per_node = edges_per_node
        self.seed = seed
        self.edge_file = edge_file
        np.random.seed(seed)
        
        # Build PRT name mapping
        self.prt_names = [
            'Business_Continuity_Risk', 'Capital_Risk', 'Compliance_Risk', 'Concentration_Risk',
            'Conduct_Risk', 'Credit_Risk', 'Currency_Risk', 'Cyber_Risk', 'External_Fraud_Risk',
            'Financial_Crime_Risk', 'Interest_Rate_Risk', 'Internal_Fraud_Risk', 'Liquidity_Risk',
            'Market_Risk', 'Model_Risk', 'Operational_Risk', 'Reputational_Risk', 'Sovereign_Risk',
            'Strategic_Risk', 'Technology_Risk'
        ]
        self.prt_to_idx = {name: i for i, name in enumerate(self.prt_names)}
        
    def generate(self) -> np.ndarray:
        """
        Generate a sparse directed weighted adjacency matrix.
        
        If edge_file is provided, reads from that; otherwise generates random.
        
        Returns:
            W_true (num_prts, num_prts): Weighted adjacency matrix where
                W_true[i,j] = weight if there is an edge from i to j, else 0.
                Weights are in (0.3, 0.9) or as specified in edge_file.
        """
        W_true = np.zeros((self.num_prts, self.num_prts))
        
        if self.edge_file is not None:
            # Load documented edges from CSV
            try:
                edges_df = pd.read_csv(self.edge_file)
                for _, row in edges_df.iterrows():
                    source_name = row['source_prt']
                    target_name = row['target_prt']
                    weight = float(row['weight'])
                    
                    if source_name in self.prt_to_idx and target_name in self.prt_to_idx:
                        source_idx = self.prt_to_idx[source_name]
                        target_idx = self.prt_to_idx[target_name]
                        W_true[source_idx, target_idx] = weight
                print(f"[PlantedNetworkGenerator] Loaded {len(edges_df)} edges from {self.edge_file}")
            except Exception as e:
                print(f"[PlantedNetworkGenerator] Warning: Could not load {self.edge_file}: {e}")
                print(f"                         Falling back to random network generation")
                W_true = self._generate_random()
        else:
            # Generate random sparse network
            W_true = self._generate_random()
        
        return W_true
    
    def _generate_random(self) -> np.ndarray:
        """Generate a random sparse directed weighted network."""
        W_true = np.zeros((self.num_prts, self.num_prts))
        
        for target in range(self.num_prts):
            num_incoming = np.random.randint(1, self.edges_per_node + 1)
            sources = np.random.choice(self.num_prts, size=num_incoming, replace=False)
            
            for source in sources:
                if source != target:
                    weight = np.random.uniform(0.3, 0.9)
                    W_true[source, target] = weight
        
        return W_true


class TCopulaVARProcess:
    """
    Generate synthetic data from a VAR(1) process with Student's t-copula.
    
    The process captures:
    1. Temporal persistence (AR structure)
    2. Cross-sectional dependence via a planted network
    3. Tail dependence via t-copula (not Gaussian)
    """
    
    def __init__(
        self,
        num_prts: int,
        kris_per_prt: int,
        time_steps: int,
        t_df: float = 5.0,
        contagion_strength: float = 0.4,
        seed: int = 123,
    ):
        """
        Args:
            num_prts: Number of PRT nodes (20)
            kris_per_prt: Number of KRIs per PRT (5 → 100 total)
            time_steps: Length of time series (250)
            t_df: Degrees of freedom for t-copula (5 → fatter tails)
            contagion_strength: Scaling factor λ for network effect (0.4)
            seed: Random seed
        """
        self.num_prts = num_prts
        self.kris_per_prt = kris_per_prt
        self.num_kris = num_prts * kris_per_prt
        self.time_steps = time_steps
        self.t_df = t_df
        self.contagion_strength = contagion_strength
        self.seed = seed
        np.random.seed(seed)
        
    def _build_copula_correlation(self, W_true: np.ndarray) -> np.ndarray:
        """
        Construct the copula correlation matrix from the planted network.
        
        The planted network W_true is on PRT nodes (20 × 20).
        Each of 100 KRIs inherits correlation from its parent PRT.
        
        Correlation formula (for t-copula):
            Σ = (I - λ*W_true)^{-1} D (I - λ*W_true)^{-T}
        where λ is contagion strength and D is diagonal idiosyncratic variance.
        
        Then expand to 100 × 100 by PRT membership.
        
        Args:
            W_true (20, 20): Planted adjacency matrix
            
        Returns:
            Sigma_100 (100, 100): Correlation matrix for t-copula of 100 KRIs
        """
        # Build 20×20 correlation from network
        I = np.eye(self.num_prts)
        lam_W = self.contagion_strength * W_true
        
        # Check if (I - λW) is invertible
        try:
            inv_term = np.linalg.inv(I - lam_W)
        except np.linalg.LinAlgError:
            # If singular, fall back to regularized version
            inv_term = np.linalg.inv(I - 0.9 * lam_W)
        
        # Idiosyncratic variances (diagonal)
        D = np.diag(np.ones(self.num_prts) * 0.5)
        
        # Σ_PRT = (I - λW)^{-1} D (I - λW)^{-T}
        Sigma_prt = inv_term @ D @ inv_term.T
        
        # Ensure symmetric and positive definite
        Sigma_prt = (Sigma_prt + Sigma_prt.T) / 2
        
        # Clip eigenvalues to ensure PD
        eigvals, eigvecs = np.linalg.eigh(Sigma_prt)
        eigvals = np.maximum(eigvals, 1e-6)
        Sigma_prt = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Expand from 20×20 PRT to 100×100 KRI
        Sigma_kri = np.zeros((self.num_kris, self.num_kris))
        for i in range(self.num_prts):
            for j in range(self.num_prts):
                # Block (i,j) of the expanded matrix
                i_start, i_end = i * self.kris_per_prt, (i + 1) * self.kris_per_prt
                j_start, j_end = j * self.kris_per_prt, (j + 1) * self.kris_per_prt
                
                # All KRIs within the same PRT pair get the same correlation
                Sigma_kri[i_start:i_end, j_start:j_end] = np.ones((self.kris_per_prt, self.kris_per_prt)) * Sigma_prt[i, j]
        
        # Diag(Sigma_kri) should be 1 (correlation matrix)
        for k in range(self.num_kris):
            Sigma_kri[k, k] = 1.0
        
        # Ensure symmetric
        Sigma_kri = (Sigma_kri + Sigma_kri.T) / 2
        
        # Clip to correlation range [-1, 1]
        Sigma_kri = np.clip(Sigma_kri, -0.99, 0.99)
        
        return Sigma_kri
    
    def _sample_t_copula(self, Sigma: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Sample from a multivariate t-copula.
        
        Process:
        1. Sample from multivariate normal with cov Σ
        2. Sample chi-squared(ν) and scale by ν/χ²
        3. Transform to uniform marginals via t-CDF
        4. Invert to Gaussian via inverse normal CDF
        
        Args:
            Sigma (100, 100): Correlation matrix
            num_samples: Number of samples
            
        Returns:
            Z (num_samples, 100): Samples with t-copula dependence
        """
        # Cholesky decomposition
        L = cholesky(Sigma, lower=True)
        
        # Standard normal samples (100, num_samples)
        Z_normal = np.random.randn(self.num_kris, num_samples)
        
        # Scale by t-copula
        chi2_samples = np.random.chisquare(self.t_df, num_samples)
        scaling = np.sqrt(self.t_df / chi2_samples)
        
        # Apply Cholesky and scaling
        Z_scaled = L @ (Z_normal * scaling[np.newaxis, :])
        
        return Z_scaled.T  # (num_samples, 100)
    
    def generate_data(self, W_true: np.ndarray, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic KRI panel data.
        
        Args:
            W_true (20, 20): Planted network
            noise_level: Standard deviation of measurement noise (0.1, 0.3, or 0.5)
            
        Returns:
            kri_panel (time_steps, 100): Observable KRI values
            z_latent (time_steps, 100): Latent VAR process
        """
        # Step 1: Build copula correlation from planted network
        Sigma = self._build_copula_correlation(W_true)
        
        # Step 2: Sample innovations with t-copula dependence
        innovations = self._sample_t_copula(Sigma, self.time_steps)  # (T, 100)
        
        # Step 3: Generate VAR(1) with persistence driven by W_true
        z_latent = np.zeros((self.time_steps, self.num_kris))
        
        # Expand W_true from PRT (20×20) to KRI (100×100) level by block replication
        W_kri = np.kron(W_true, np.ones((self.kris_per_prt, self.kris_per_prt)))
        
        # Diagonal persistence (one per KRI)
        phi = np.random.uniform(0.5, 0.7, self.num_kris)  # Moderate persistence
        
        # VAR(1) lag coefficient: A1 = diag(φ) + λ * W_kri
        # where λ is the lag-matrix transmission strength
        # Use stronger signal even if it requires stationarity rescaling
        lambda_lag = self.contagion_strength * 0.70  # Back to ~0.28 for stronger signal
        A1 = np.diag(phi) + lambda_lag * W_kri
        
        # Check stationarity: all eigenvalues must be < 1
        eigvals = np.linalg.eigvals(A1)
        max_eig = np.abs(eigvals).max()
        if max_eig >= 1.0:
            # Rescale to ensure stationarity
            # Use a smaller buffer (1.01 instead of 1.05) to minimize signal loss
            A1 = A1 / (max_eig * 1.01)
        
        # Initialize
        z_latent[0] = innovations[0]
        
        # VAR(1): z_t = A1 * z_{t-1} + ε_t
        for t in range(1, self.time_steps):
            z_latent[t] = A1 @ z_latent[t - 1] + innovations[t]
        
        # Step 4: Map to observable KRI scale [10, 40] via sigmoid + scaling
        kri_panel = 10 + 30 * (1 / (1 + np.exp(-z_latent / np.std(z_latent))))
        
        # Step 5: Add measurement noise
        noise = np.random.normal(0, noise_level, kri_panel.shape)
        kri_panel = kri_panel + noise
        
        return kri_panel, z_latent


class DataGenerator:
    """High-level interface for generating synthetic data and ground-truth network."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def generate_experiment(
        self,
        time_steps: int = 250,
        noise_level: float = 0.1,
        network_density: int = 2,
        t_df: float = 5.0,
        use_documented_network: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a single experimental run.
        
        Args:
            time_steps: Length of KRI time series (50–250)
            noise_level: Measurement noise std dev (0.1, 0.3, 0.5)
            network_density: Edges per PRT node (2 or 4, only used if use_documented_network=False)
            t_df: t-copula degrees of freedom (3, 5, or 10)
            use_documented_network: If True, load srtg20_edges.csv; else generate random
            
        Returns:
            Dictionary with:
                - kri_panel: (time_steps, 100) synthetic KRI data
                - W_true: (20, 20) planted network (ground truth)
                - z_latent: (time_steps, 100) latent VAR process
                - metadata: experiment configuration
        """
        # Generate planted network
        edge_file = self.config['data']['edge_definitions'] if use_documented_network else None
        net_gen = PlantedNetworkGenerator(
            num_prts=20,
            edges_per_node=network_density,
            seed=self.config['seeds']['numpy_seed'],
            edge_file=edge_file
        )
        W_true = net_gen.generate()
        
        # Generate synthetic data with t-copula VAR
        var_gen = TCopulaVARProcess(
            num_prts=20,
            kris_per_prt=5,
            time_steps=time_steps,
            t_df=t_df,
            contagion_strength=self.config['dgp']['var']['contemporaneous_contagion'],
            seed=self.config['seeds']['torch_seed']
        )
        kri_panel, z_latent = var_gen.generate_data(W_true, noise_level=noise_level)
        
        return {
            'kri_panel': kri_panel,
            'W_true': W_true,
            'z_latent': z_latent,
            'metadata': {
                'time_steps': time_steps,
                'noise_level': noise_level,
                'network_density': network_density,
                't_df': t_df,
                'num_prts': 20,
                'num_kris': 100,
                'use_documented_network': use_documented_network,
            }
        }
    
    def generate_monte_carlo(
        self,
        time_steps: int,
        noise_level: float,
        network_density: int,
        t_df: float,
        num_replications: int = 100,
        use_documented_network: bool = True,
    ) -> list:
        """
        Generate multiple experimental runs for Monte Carlo analysis.
        
        Args:
            num_replications: Number of runs per configuration (default 100)
            use_documented_network: If True, load srtg20_edges.csv for each replication
            
        Returns:
            List of experiment dictionaries
        """
        experiments = []
        for rep in range(num_replications):
            # Vary seed slightly for each replication
            np.random.seed(self.config['seeds']['numpy_seed'] + rep)
            exp = self.generate_experiment(
                time_steps=time_steps,
                noise_level=noise_level,
                network_density=network_density,
                t_df=t_df,
                use_documented_network=use_documented_network,
            )
            exp['metadata']['replication'] = rep
            experiments.append(exp)
        
        return experiments


if __name__ == "__main__":
    # Test the generator
    gen = DataGenerator("config.yaml")
    
    # Single run
    exp = gen.generate_experiment(
        time_steps=250,
        noise_level=0.1,
        network_density=2,
        t_df=5.0
    )
    
    print("Generated synthetic data:")
    print(f"  KRI panel shape: {exp['kri_panel'].shape}")
    print(f"  Planted network edges: {np.count_nonzero(exp['W_true'])}")
    print(f"  KRI means: {exp['kri_panel'].mean(axis=0)[:5]} (first 5)")
    print(f"  KRI stds: {exp['kri_panel'].std(axis=0)[:5]} (first 5)")
