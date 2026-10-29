# Contributors

## Project: SRTG-20 — Synthetic Risk Transmission Graph Recovery Benchmark
**World Quant University Capstone | M6 Submission**

---

## Core Team

### Emmanuel Odenyire Anyira
- **Role:** Lead Researcher & Implementation
- **Responsibilities:**
  - Study design and methodology development
  - Network architecture (srtg20_edges.csv)
  - Synthetic data generation (DGP implementation)
  - Recovery method implementation (VAR+DY, CoVaR, GNN)
  - Experimental execution and validation
  - Results analysis and visualization
  - Report writing and documentation

---

## Advisors & Guidance

### Iván Blanco (World Quant University)
- **Role:** Project Supervisor
- **Contributions:**
  - Methodology guidance and validation
  - Network design confirmation
  - Recovery method taxonomy (3 estimators + 1 operator)
  - Robustness check recommendations (random sparse networks)
  - Clarifications on edge directionality and VAR lag matrix integration

---

## Acknowledgments

### Literature & Frameworks Referenced

- **RAMSI (2012):** Bank of England solvency-and-funding-liquidity network
- **BEAST (2009):** ECB credit-profitability-solvency feedback model
- **IMF (2010):** Solvency-liquidity linkage analysis
- **Basel PSMOR (2013):** Operational and technology risk framework
- **Basel Step-in Risk (2015):** Reputational risk transmission
- **Basel IRB (2006):** Concentration risk modeling

### Tools & Libraries

- **Python 3.10+**
- **NumPy, Pandas:** Data manipulation
- **SciPy:** Statistical distributions and Cholesky decomposition
- **scikit-learn:** Metrics and preprocessing
- **PyTorch:** Graph Neural Network (GAT) implementation
- **Matplotlib, Seaborn:** Visualization
- **YAML:** Configuration management

---

## Study Contributions & Responsibilities

| Component | Contributor | Status |
|-----------|-------------|--------|
| Network design (41 edges) | Emmanuel Anyira | ✅ Complete |
| DGP implementation | Emmanuel Anyira | In Progress |
| VAR+DY recovery | Emmanuel Anyira | In Progress |
| CoVaR recovery | Emmanuel Anyira | In Progress |
| GNN training & recovery | Emmanuel Anyira | In Progress |
| DebtRank propagation | Emmanuel Anyira | In Progress |
| Evaluation metrics | Emmanuel Anyira | In Progress |
| Experiment orchestration | Emmanuel Anyira | In Progress |
| Results aggregation | Emmanuel Anyira | In Progress |
| Visualization & figures | Emmanuel Anyira | In Progress |
| Report writing | Emmanuel Anyira | In Progress |
| Peer review & feedback | Iván Blanco | Ongoing |

---

## Timeline

| Phase | Task | Start | End | Owner |
|-------|------|-------|-----|-------|
| 1 | Network design & configuration | Oct 1 | Oct 7 | Emmanuel |
| 2 | DGP & synthetic data generation | Oct 7 | Oct 14 | Emmanuel |
| 3 | Recovery method implementation | Oct 14 | Oct 21 | Emmanuel |
| 4 | Experiment execution | Oct 21 | Oct 28 | Emmanuel |
| 5 | Analysis & visualization | Oct 28 | Nov 2 | Emmanuel |
| 6 | Report writing & submission | Nov 2 | Nov 8 | Emmanuel |

---

## Data Attribution

**Network Edge Definitions (`srtg20_edges.csv`):**
- 41 edges sourced from 9 supervisory and academic frameworks
- Each edge includes documented source, section, and justification
- Full attribution in edge metadata and bibliography

**KRI & PRT Definitions:**
- Taxonomy derived from Basel III operational risk framework
- 20 Primary Risk Types
- 100 Key Risk Indicators (5 per PRT)

---

## Code & Reproducibility

**Repository Structure:**
```
data/
  ├── srtg20_edges.csv           # Planted network (41 edges)
  ├── PRT_Definitions_v2.csv     # 20 PRTs (taxonomy)
  ├── KRI_Definitions_v2.csv     # 100 KRIs (5 per PRT)
src/
  ├── synthetic_generator.py     # DGP & data generation
  ├── recovery_methods.py        # VAR+DY, CoVaR, GNN
  ├── evaluation.py              # Metrics computation
  ├── experiment_runner.py       # Orchestration
  └── visualization.py           # Figures & plots
results/
  ├── benchmark_results.json     # Raw results
  ├── benchmark_summary.csv      # Aggregated metrics
  └── figures/                   # Generated plots
config.yaml                      # Experiment configuration
requirements.txt                 # Python dependencies
```

**Reproducibility Checklist:**
- ✅ Seeds fixed for all RNGs (NumPy, PyTorch)
- ✅ Configuration versioned in config.yaml
- ✅ Network sourced from documented frameworks
- ✅ Methods deterministic or seeded
- ✅ Results saved as JSON (machine-readable) + CSV (human-readable)
- ✅ Visualizations regenerable from saved results

---

## Funding & Institutional Support

- **Funding:** World Quant University Capstone Program
- **Institution:** World Quant University
- **Program:** Online Quantitative Finance & Machine Learning
- **Submission Date:** October 2026

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

## Contact

**Project Lead:** Emmanuel Odenyire Anyira  
**Supervisor:** Iván Blanco (World Quant University)  
**Submission:** World Quant University Capstone Program

---

*Last updated: October 2026*
