"""
DebtRank evaluation — propagation accuracy against reference cascade.
Per Iván's taxonomy: DebtRank is evaluated ONLY on impulse-response MSE.
No precision / recall / F1 / AUC.
"""
import json
import numpy as np
from pathlib import Path

from src.synthetic_generator import PlantedNetworkGenerator
from src.propagation import PropagationSimulator   # or wherever DebtRank lives
from src.recovery_methods import VARDieboldYilmaz, CoVaR


def run_debtrank_cascade(W: np.ndarray, simulator, shock_node: int = 0,
                         shock_size: float = 0.30, n_periods: int = 13) -> np.ndarray:
    """Return cascade vector C(t) for t=0..n_periods-1."""
    return simulator.simulate(W, shock_node=shock_node,
                              shock_size=shock_size,
                              n_periods=n_periods)


def cascade_mse(c_ref: np.ndarray, c_est: np.ndarray) -> float:
    return float(np.mean((c_ref - c_est) ** 2))


def main():
    results_file = Path("results/task4_enhanced_benchmark_with_random.json")
    with open(results_file) as f:
        configs = json.load(f)

    # Ground truth network
    generator = PlantedNetworkGenerator(edge_file="data/srtg20_edges.csv")
    W_true = generator.W_true   # 20x20

    # Reference cascade
    simulator = PropagationSimulator()
    C_ref = run_debtrank_cascade(W_true, simulator)

    out = []
    for cfg in configs:
        # Regenerate the same data this config used
        np.random.seed(cfg["config"].get("seed", 42))
        data = generator.generate(
            time_steps=cfg["config"]["time_steps"],
            noise_level=cfg["config"]["noise_level"],
            network_type=cfg["network_type"],
        )

        # Fit each recovery estimator on the data
        est_results = {}

        # VAR+DY
        var = VARDieboldYilmaz(num_prts=20, kris_per_prt=5)
        W_var = var.fit_and_recover(data)
        C_var = run_debtrank_cascade(W_var, simulator)
        est_results["var_dy"] = {
            "cascade_mse": cascade_mse(C_ref, C_var),
            "cascade_peak": float(np.max(np.abs(C_var))),
        }

        # CoVaR — needs a threshold to become a sparse graph
        cov = CoVaR(num_prts=20, kris_per_prt=5, tail=0.05)
        C_scores = cov.fit_and_recover(data)   # dense 20x20
        # TODO: pick a threshold (e.g., quantile matching true density)
        W_cov = (C_scores > np.quantile(C_scores, 1 - 41 / 400)).astype(float)
        C_cov = run_debtrank_cascade(W_cov, simulator)
        est_results["covar"] = {
            "cascade_mse": cascade_mse(C_ref, C_cov),
            "cascade_peak": float(np.max(np.abs(C_cov))),
        }

        # Reference (trivially zero)
        est_results["debtrank_ref"] = {
            "cascade_mse": 0.0,  # by construction
            "cascade_peak": float(np.max(np.abs(C_ref))),
        }

        out.append({
            "config": cfg["config"],
            "network_type": cfg["network_type"],
            "estimators": est_results,
        })

    output_file = "results/task4_debtrank_propagation_mse.json"
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()