# SRTG-20: Synthetic Risk Transmission Graph Recovery Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Capstone](https://img.shields.io/badge/Status-Capstone%20Submission-brightgreen)](https://wqu.edu)

**World Quant University | MScFE Capstone Project | Group 17869**  
**Track:** Practical Track, Risk Management  
**Deadline:** November 3, 2026

---

## Executive Summary

This is a **reproducible methods study** benchmarking four network-recovery techniques on synthetic financial risk data. We plant a known 20-node risk-propagation network, generate 100 synthetic Key Risk Indicators (KRIs) with tail dependence, and rigorously measure how accurately competing estimators recover the network structure.

**Core Question:** Which network-recovery methods work best? Under what data conditions? With what accuracy?

**Key Deliverables:**
- ✅ Reproducible 100-experiment benchmark (9,000 total runs)
- ✅ Four recovery methods: VAR+Diebold-Yilmaz, CoVaR, DebtRank, Graph Neural Networks
- ✅ Six performance metrics: Precision, Recall, F1, AUC-ROC, Edge-Weight Correlation, Impulse-Response MSE
- ✅ Full results in JSON + CSV with visualizations
- ✅ Production-ready code with pinned dependencies

---

## Problem Context

Financial institutions track interconnected risks through Key Risk Indicators (KRIs) grouped into Principal Risk Types (PRTs). While individual risk dashboards answer "Is KRI X in green/amber/red?", they cannot answer:

- **Which PRT shocks propagate to other PRTs?**
- **What are the speed and magnitude of contagion?**
- **Where are the critical contagion pathways?**

### The Data Constraint

African financial institutions face severe data constraints due to regulatory privacy requirements. This study reframes the problem:

1. **Plant a known network** on 20 PRTs with 41 documented edges
2. **Generate synthetic KRI data** with that embedded structure (250 months, 100 KRIs)
3. **Benchmark recovery methods** to measure their accuracy
4. **Publish empirical evidence** of performance under data scarcity

This avoids circularity (planting-then-discovering) by treating it as a **methods study**, not a real-network discovery.

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/WorldQuantCapstone/SRTG-20.git
cd SRTG-20

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (pinned versions for reproducibility)
pip install -r requirements.txt
```

**Note:** PyTorch and torch-geometric are included for GNN support. On CPU, they are large (~2 GB); on GPU, they use CUDA if available.

### 2. Run Quick Benchmark (2 minutes)

Test a single configuration with 5 replications:

```bash
python main.py --quick --replicates 5
```

Expected output:
```
Configuration: T=250, σ=0.1, ν=5, density=2
VAR+DY       F1=0.68  AUC=0.82
CoVaR        F1=0.61  AUC=0.76
GNN          F1=0.74  AUC=0.85
```

### 3. Run Full Benchmark (4–6 hours)

Execute all 20 configurations with 100 replications each (9,000 experiments):

```bash
python main.py --full --replicates 100
```

Output: `results/task4_enhanced_benchmark_with_random.json` (complete results)

### 4. Generate Visualizations

```bash
python scripts/generate_visualizations.py
```

Output: PNG plots in `results/figures/`

---

## Technical Approach

### Network Taxonomy

**20 Principal Risk Types (PRTs):**

| Category | Risk Types |
|----------|-----------|
| Financial (7) | Credit, Market, Liquidity, Capital, Interest Rate, Currency, Concentration |
| Non-Financial (13) | Operational, Compliance, Conduct, Financial Crime, External Fraud, Internal Fraud, Model, Technology, Cyber, Strategic, Reputational, Sovereign, Business Continuity |

**100 Key Risk Indicators (KRIs):** 5 per PRT (fixed mapping in `data/KRI_Definitions_v2.csv`)

### Synthetic Data Generating Process (DGP)

The DGP embeds three essential features for risk contagion:

#### Step 1: Plant a Known Network
- **Source:** `data/srtg20_edges.csv` (41 edges documented from Basel/supervisory frameworks)
- **Format:** Sparse directed weighted adjacency matrix W_true ∈ ℝ^(20×20)
- **Edge weights:** [0.3, 0.9] (log-normal distribution)

#### Step 2: VAR(1) Latent Dynamics with t-Copula
```
z_t = diag(φ) × z_{t-1} + ε_t
```
where:
- z_t ∈ ℝ^100 are latent KRI values
- φ ∈ [0.6, 0.85] are autoregressive coefficients
- ε_t ~ Student's t-copula(ν ∈ {3, 5, 10}) with correlation Σ = (I - λW_true)^(-1) D (I - λW_true)^(-T)

**Why t-copula?** Gaussian copulas have zero tail dependence and cannot generate simultaneous extreme co-movements—the defining characteristic of contagion. t-copulas with lower degrees of freedom capture this.

#### Step 3: Observable KRI Scale + Measurement Noise
```
KRI_observed = sigmoid(latent) + noise,  where noise ~ N(0, σ²)
```

### Recovery Methods

| Method | Type | Principle | Strengths | Weaknesses |
|--------|------|-----------|-----------|-----------|
| **VAR+DY** | Baseline 1 | Variance decomposition spillovers | Linear, transparent, stable | Captures averages, not tails |
| **CoVaR** | Baseline 2 | Tail-risk regression at 5th percentile | Direct tail focus | Pairwise only, no multi-node consistency |
| **DebtRank** | Propagation Operator | Cascading distress simulation | Models real contagion | Requires known exposures |
| **GNN** | Proposed Advanced | Message-passing graph neural network | Learns nonlinear dependencies | Hyperparameter sensitive |

### Evaluation Metrics (6 Metrics)

All measured at optimal threshold (maximizing F1):

| Metric | Definition | Interpretation |
|--------|-----------|---|
| **Precision** | TP / (TP + FP) | Fraction of recovered edges that truly exist |
| **Recall** | TP / (TP + FN) | Fraction of true edges recovered |
| **F1 Score** | 2·Precision·Recall / (Precision + Recall) | Harmonic mean; optimal = 1.0 |
| **AUC-ROC** | Area under ROC curve | Threshold-agnostic discrimination ability |
| **Edge-Weight Correlation** | Pearson r(estimated, true) | Accuracy of edge strength estimates |
| **Impulse-Response MSE** | MSE of shock propagation | Accuracy of network dynamics |

### Experimental Design

**Monte Carlo Framework:** 100 independent replications per configuration.

| Factor | Levels | Count | Rationale |
|--------|--------|-------|-----------|
| **Data Volume (T)** | 50, 100, 150, 200, 250 | 5 | African banks often have <250 monthly observations |
| **Noise (σ)** | 0.1, 0.3, 0.5 | 3 | Measurement error in risk systems |
| **Tail Dependence (ν)** | 3, 5, 10 | 3 | Lower ν = fatter tails = more extreme co-movements |
| **Network Density** | 2, 4 edges/node | 2 | Sparse vs. dense contagion networks |

**Total:** 5 × 3 × 3 × 2 = 20 configurations → 20 × 100 = 2,000 core experiments

**Robustness Check:** Compare designed network against 5 random sparse networks (additional 1,000 experiments)

**Grand Total:** ~3,000 experiments

---

## Project Structure

```
SRTG-20/
├── README.md                      # This file (project entry point)
├── LICENSE                        # MIT License
├── requirements.txt               # Pinned dependency versions
├── config.yaml                    # Experiment configuration (seeds, parameters)
├── main.py                        # Entry point: orchestrates all experiments
│
├── docs/                          # Public documentation
│   ├── README.md                  # (same as root README)
│   └── CONTRIBUTORS.md            # Team attribution and roles
│
├── src/                           # Core library code (production)
│   ├── __init__.py
│   ├── synthetic_generator.py     # DGP: plant network, generate KRIs with t-copula
│   ├── recovery_methods.py        # VAR+DY, CoVaR, DebtRank, GNN implementations
│   ├── evaluation.py              # 6-metric evaluation framework
│   ├── experiment_runner.py       # Orchestration and Monte Carlo loop
│   ├── propagation.py             # DebtRank impulse-response propagation
│   ├── random_networks.py         # Random network generation (robustness check)
│   └── visualization.py           # Heatmaps, degradation curves
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_suite.py              # Full test battery
│   ├── test_quick.py              # Quick smoke tests
│   └── test_data_gen.py           # Data generation verification
│
├── scripts/                       # Utility scripts (secondary entry points)
│   ├── generate_results_tables.py # Convert JSON → CSV summary
│   ├── generate_visualizations.py # Render PNG plots from results
│   └── validate_network.py        # Verify edge definitions
│
├── analysis/                      # Diagnostic and exploratory scripts
│   ├── audit_complete_project.py  # Full project audit
│   ├── task4_enhanced_benchmark_with_random.py
│   ├── inspect_all_results*.py    # Result inspection tools
│   └── ... (other analysis scripts)
│
├── data/                          # Input data (static, versioned)
│   ├── srtg20_edges.csv           # 41 documented edges (ground truth network)
│   ├── PRT_Definitions_v2.csv     # 20 PRTs with descriptions
│   ├── KRI_Definitions_v2.csv     # 100 KRIs with PRT mappings
│   ├── Contagion_Propagation.csv  # Reference cascade (sanity check)
│   ├── Observation_Data.csv       # Sample KRI observations
│   └── (other data files)
│
├── results/                       # Output directory (generated, not versioned)
│   ├── task4_enhanced_benchmark_with_random.json
│   ├── benchmark_summary.csv
│   ├── figures/
│   │   ├── f1_vs_data_volume.png
│   │   ├── f1_vs_noise.png
│   │   ├── auc_roc_vs_data_volume.png
│   │   ├── f1_heatmap.png
│   │   └── auc_roc_heatmap.png
│   └── logs/
│
├── archive/                       # Old drafts, logs, intermediate work
│   ├── Draft report.docx
│   ├── CAPSTONE_SUBMISSION_SUMMARY.txt
│   └── (other archived files)
│
└── .git/                          # Version control
```

---

## Usage Examples

### Example 1: Run Single Configuration (Quick)

```python
from src.experiment_runner import ExperimentRunner

runner = ExperimentRunner('config.yaml')
results = runner.run_single_configuration(
    time_steps=250,
    noise_level=0.1,
    network_density=2,
    t_df=5,
    num_reps=10
)
print(f"VAR+DY F1: {results['var_dy']['f1']:.3f}")
print(f"GNN F1: {results['gnn']['f1']:.3f}")
```

### Example 2: Benchmark Data Volume Effect

```python
from src.experiment_runner import ExperimentRunner

runner = ExperimentRunner('config.yaml')
results_list = []
for T in [50, 100, 150, 200, 250]:
    r = runner.run_single_configuration(
        time_steps=T,
        noise_level=0.1,
        num_reps=20
    )
    results_list.append(r)

# Visualize degradation
import json
with open('results/data_volume_benchmark.json', 'w') as f:
    json.dump(results_list, f, indent=2)
```

### Example 3: Generate Synthetic Data Only

```python
from src.synthetic_generator import TCopulaVARProcess, PlantedNetworkGenerator

# Plant network
net_gen = PlantedNetworkGenerator(edge_file='data/srtg20_edges.csv')
W_true = net_gen.generate()

# Generate KRIs
dgp = TCopulaVARProcess(
    W_true=W_true,
    T=250,
    noise_level=0.1,
    t_df=5
)
X = dgp.generate()  # Shape: (250, 100) — 250 time steps, 100 KRIs

print(f"Network planted: {W_true.shape}, {np.count_nonzero(W_true)} edges")
print(f"Data generated: {X.shape}")
```

### Example 4: Evaluate Recovery Method

```python
from src.recovery_methods import VARDieboldYilmaz, GNNRecovery
from src.evaluation import ConfusionMetrics

# Generate data (as above)
X = dgp.generate()

# Recover network
var_dy = VARDieboldYilmaz()
W_est = var_dy.fit_and_recover(X)

# Evaluate
metrics = ConfusionMetrics.compute_at_threshold(W_est, W_true, threshold=0.1)
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
```

---

## Expected Results

### Headline Findings (at optimal conditions: T=250, σ=0.1)

| Method | F1 | AUC-ROC | Edge Corr | Notes |
|--------|-----|---------|-----------|-------|
| **Trivial Baseline** | ~0.18 | 0.50 | — | Predict-all edges at 10% base rate |
| **VAR+DY** | 0.65–0.75 | 0.75–0.85 | 0.25–0.45 | Beats baseline; captures averages |
| **CoVaR** | 0.60–0.70 | 0.70–0.80 | 0.20–0.35 | Diagnostic only; weak tail signal |
| **GNN** | 0.70–0.80 | 0.80–0.88 | 0.35–0.55 | Best with sufficient data |

### Degradation as Data Becomes Scarce

**Data Volume Effect (T: 250 → 50):**
- VAR+DY AUC-ROC: 0.80 → 0.58 (−28%)
- GNN AUC-ROC: 0.85 → 0.52 (−39%)

**Noise Effect (σ: 0.1 → 0.5):**
- VAR+DY F1: 0.70 → 0.48 (−31%)
- GNN F1: 0.75 → 0.54 (−28%)

**Central Finding:** No single estimator dominates all conditions. Trade-offs depend on data scarcity, noise, and tail dependence.

---

## Key Contributions

1. **Methodological:** Reframes the circularity problem via planted networks and methods-study design
2. **Technical:** Implements and benchmarks four methods (including novel GNN) in unified framework
3. **Empirical:** Quantifies when recovery works and when it fails (critical for practitioners)
4. **Open:** Fully reproducible with no proprietary claims

---

## Reproducibility Guarantees

All results are fully reproducible:

- ✅ **Fixed random seeds** in `config.yaml` (NumPy, PyTorch)
- ✅ **Deterministic YAML configuration** — no hardcoded parameters
- ✅ **No external API calls** — entirely self-contained
- ✅ **Versioned dependencies** in `requirements.txt`
- ✅ **Network documented** in `data/srtg20_edges.csv` with sources
- ✅ **Results in machine-readable format** (JSON + CSV)

To reproduce any result:
```bash
# Modify config.yaml if desired, then:
python main.py --full --replicates 100

# Results saved to results/
# Regenerate visualizations:
python scripts/generate_visualizations.py
```

---

## File Guide

| File | Purpose | Audience |
|------|---------|----------|
| **README.md** | Project overview, quick start, examples | Everyone |
| **SETUP.md** | Detailed installation, environment setup | New users |
| **CONTRIBUTORS.md** | Team roles, attribution, funding | Collaborators |
| **config.yaml** | Experiment parameters, random seeds | Developers |
| **main.py** | Entry point, CLI orchestration | End users |
| **src/** | Production library code | Developers |
| **tests/** | Test suite for CI/CD | Developers |
| **scripts/** | Utility scripts (visualization, reporting) | Analysts |
| **analysis/** | Diagnostic and exploratory scripts | Researchers |

---

## Installation Troubleshooting

### Issue: PyTorch Installation Fails
**Solution:** torch-geometric requires a matching PyTorch version. Install torch first, then torch-geometric:
```bash
pip install torch==2.0.1
pip install torch-geometric==2.3.1
```

### Issue: Out of Memory (OOM) on CPU
**Solution:** The full benchmark (9,000 runs) requires ~16 GB RAM. For testing:
```bash
# Run quick benchmark instead (5–10 GB)
python main.py --quick --replicates 5
```

### Issue: Missing CUDA for GPU Acceleration
**Solution:** torch will auto-detect GPU. To force CPU:
```bash
CUDA_VISIBLE_DEVICES="" python main.py --quick
```

---

## Citation

If you use this work, please cite:

```bibtex
@thesis{anyira2026srtg,
  author = {Anyira, Emmanuel Odenyire},
  title = {SRTG-20: Synthetic Risk Transmission Graph Recovery Benchmark},
  school = {World Quant University},
  year = {2026},
  type = {Capstone Project},
  supervisor = {Iván Blanco}
}
```

---

## References

- **Network Recovery:** Diebold & Yilmaz (2014), *Measuring Financial Asset Return and Volatility Spillovers with Application to Global Equity Markets*
- **Tail Risk:** Adrian & Brunnermeier (2016), *CoVaR*, American Economic Review
- **Contagion:** Battiston et al. (2016), *DebtRank: A Microscopic Foundation for Shock Propagation*, Journal of Economic Dynamics & Control
- **Copulas:** Demarta & McNeil (2005), *The t Copula and Related Copulas*, International Statistical Review
- **GNNs:** Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR

See `docs/` for full bibliography and methodology.

---

## License

MIT License. See `LICENSE` file for details.

---

## Contact & Attribution

**Project Team:**
- **Emmanuel Odenyire Anyira** — Lead Researcher & Implementation
- **Supervisor:** Iván Blanco (World Quant University)

**Institution:** World Quant University | MScFE Program  
**Submission:** October 2026  
**Deadline:** November 3, 2026

---

## Status

**Status:** ✅ **Submission Ready** (Oct 2026)

- [x] Data generation verified
- [x] Recovery methods implemented and benchmarked
- [x] Full 9,000-experiment benchmark completed
- [x] Results visualized and summarized
- [x] Code documented and tested
- [x] Reproducibility verified
- [x] Submission-ready structure

**Next steps:** Deploy to GitHub, submit to WQU capstone evaluation.

---

*Last Updated: October 29, 2026*  
*Version: 1.0 (Production Ready)*
#   n e t w o r k - r e c o v e r y - b e n c h m a r k  
 
# network-recovery-benchmark
