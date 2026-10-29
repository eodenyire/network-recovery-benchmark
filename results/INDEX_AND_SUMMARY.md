# SRTG-20 Capstone: Complete Results Index

## 📋 Document Structure

This directory contains the complete results, analysis, and reproducible code for the SRTG-20 (Synthetic Risk Transmission Graph) network recovery benchmark study.

---

## 📄 Primary Documents

### 1. **RESULTS_AND_DISCUSSION.md** ← START HERE
**Comprehensive analysis of benchmark findings**
- Executive summary
- Performance tables (designed vs. random networks)
- Effect of data volume and noise
- Discussion of direction identifiability constraint
- Interpretation of CoVaR robustness and GNN limitations
- DebtRank cascade validation
- Methodological contributions
- Raw data and statistical summaries

**Key Finding:** VAR+DY achieves 0.50 AUC (random chance) due to fundamental direction identifiability constraints, while CoVaR achieves 0.60 AUC via quantile-based tail dependence.

### 2. **CONCLUSIONS_AND_RECOMMENDATIONS.md** ← THEN READ THIS
**Actionable conclusions and practical guidance**
- 5 substantive conclusions (direction identifiability, CoVaR trade-offs, GNN needs, cascade dynamics)
- 3 methodological contributions (operationalized diagnostics, benchmark suite, design vs. random comparison)
- Practical recommendations for practitioners (expectation setting, method selection, diagnostic workflow)
- Research directions for network science community
- Study limitations and caveats
- Appendix with figures, tables, and glossary

**Target Audience:** Practitioners in financial risk, macroeconomic contagion; researchers in network science and causal inference.

---

## 📊 Visualizations (Publication-Ready PNG)

Located in `results/figures/`

| Figure | Content | Interpretation |
|--------|---------|-----------------|
| **01_recovery_methods_comparison.png** | AUC vs. time steps (T=50–250), side-by-side designed/random | VAR+DY flat at 0.50; CoVaR improves with data; GNN gains from random networks |
| **02_designed_vs_random_boxplot.png** | Distribution comparison for VAR+DY, CoVaR, GNN, DebtRank | CoVaR spreads wider on random; GNN highly variable; VAR+DY tight around 0.50 |
| **03_effect_of_noise.png** | AUC vs. noise level (σ ∈ {0.1, 0.5}), designed/random | CoVaR invariant; VAR+DY unchanged; designed networks less resilient than random |
| **04_performance_heatmap.png** | Color-coded AUC for all methods × network types | Heatmap: green (high AUC), red (low AUC) |
| **05_debtrank_performance.png** | Cascade MSE distribution; means ± std | Designed networks 14% more stable (lower MSE) |

**Usage:** Copy figures to presentations, papers, or reports. All saved at 300 DPI for print-quality output.

---

## 📈 Data Tables (CSV Format)

Located in `results/` directory

| File | Rows | Columns | Content |
|------|------|---------|---------|
| **detailed_benchmark_results.csv** | 20 | 16 | Each configuration: network type, time steps, noise level, AUC/metrics for VAR+DY, CoVaR, GNN, DebtRank |
| **summary_by_time_steps.csv** | 5 | × | Aggregated metrics grouped by T ∈ {50, 100, 150, 200, 250} |
| **summary_by_noise_level.csv** | 2 | × | Aggregated metrics grouped by σ ∈ {0.1, 0.5} |
| **designed_vs_random_comparison.csv** | 4 | 4 | Head-to-head: Method, Designed Mean, Random Mean, Difference |

**Usage:** Import into Excel/R for further analysis or present in manuscript tables.

---

## 📦 Raw Results (JSON)

### **task4_enhanced_benchmark_with_random_FINAL.json**
- **Size:** 175 KB
- **Records:** 20 configurations × 5 replications = 100 experiments
- **Hash:** 5EA098BA2F8F151003107D8FD1B5E5177084E3B012C52766D98191BE3589815C (verify with `Get-FileHash`)
- **Schema:** Each record contains:
  - `config`: time_steps, noise_level, network_density, t_df
  - `var_dy`: precision, recall, f1_score, auc_roc, edge_weight_correlation, impulse_response_mse, optimal_threshold, roc_curve, confusion_matrix, weight_stats, ir_stats
  - `covar`: Same as var_dy
  - `gnn`: Same as var_dy (or null if training failed)
  - `debtrank`: impulse_response_mse, by_estimator
  - `network_type`: "designed" or "random"

**Usage:** Load with `json.load()` in Python or `fromJSON()` in R for custom analysis.

---

## 🔬 Reproducible Code

Located in repository root and `src/` directory

### Core Benchmark Script
**`task4_enhanced_benchmark_with_random.py`** (executable)
- Generates 100 experiments (20 configs × 5 reps)
- Runs VAR+DY, CoVaR, GNN, DebtRank on each
- Saves results to `results/task4_enhanced_benchmark_with_random.json`
- Runtime: ~4 minutes (optimized; full GNN would take hours)

**To reproduce:**
```bash
python task4_enhanced_benchmark_with_random.py
```

### Supporting Modules
- **`src/synthetic_generator.py`** – VAR(1) + t-copula data generation
- **`src/recovery_methods.py`** – VAR+DY, CoVaR, GNN, DebtRank implementations
- **`src/evaluation.py`** – Evaluation metrics (AUC, precision, recall, F1, correlation, IR-MSE)
- **`src/experiment_runner.py`** – Orchestration and aggregation

### Analysis Scripts
- **`generate_results_tables.py`** – Extract and aggregate benchmark results
- **`generate_visualizations.py`** – Create 5 publication-ready PNG figures

**To regenerate all artifacts:**
```bash
python generate_results_tables.py    # Produces CSV files
python generate_visualizations.py    # Produces PNG figures
```

---

## 🎯 Quick Reference: Key Numbers

| Metric | Designed | Random | Interpretation |
|--------|----------|--------|-----------------|
| **VAR+DY AUC** | 0.4998 | 0.5007 | Random chance (identifiability constraint) |
| **CoVaR AUC** | 0.5982 | 0.7025 | Robust baseline; tail dependence signal |
| **GNN AUC** | 0.2166 | 0.3341 | Poor performance; needs architecture improvement |
| **DebtRank MSE** | 0.1647 | 0.1877 | Designed network 14% more stable |

---

## 📋 Benchmark Specifications

- **Network:** 20 nodes (PRTs), 5 KRIs each = 100 total variables
- **True edges:** 41 (directed, weighted)
- **Data generation:** VAR(1) + t-copula (ν=5) + Gaussian measurement noise
- **Configurations:** 5 time steps × 2 noise levels × 2 network types = 20
- **Replications:** 5 per configuration = 100 total experiments
- **Recovery methods:** VAR+DY, CoVaR, GNN, DebtRank
- **Evaluation metrics:** Precision, recall, F1, AUC-ROC, AUPRC, edge-weight correlation, impulse-response MSE
- **Random seed:** 42 (reproducible)

---

## ✅ Verification Checklist

Before citing or using these results, verify:

- [ ] JSON file hash: 5EA098BA2F8F151003107D8FD1B5E5177084E3B012C52766D98191BE3589815C
  ```powershell
  Get-FileHash results/FINAL/task4_enhanced_benchmark_with_random_FINAL.json -Algorithm SHA256
  ```
- [ ] All 5 figures present in `results/figures/`
- [ ] All 4 CSV files present in `results/`
- [ ] Both markdown documents (RESULTS_AND_DISCUSSION.md, CONCLUSIONS_AND_RECOMMENDATIONS.md) present
- [ ] Python code runs without error: `python task4_enhanced_benchmark_with_random.py --validate`
- [ ] Results reproducible within ±0.0005 AUC (minor floating-point variations expected)

---

## 🚀 How to Use This Package

### Scenario 1: **Academic Paper**
1. Read RESULTS_AND_DISCUSSION.md (Methods, Results sections)
2. Read CONCLUSIONS_AND_RECOMMENDATIONS.md (Discussion, Conclusion)
3. Copy figures 1–5 to your paper
4. Include CSV tables in appendix
5. Cite JSON hash for reproducibility

### Scenario 2: **Presentation/Defense**
1. Use figures 1–5 as slides
2. Reference key numbers from "Quick Reference" table
3. Prepare talking points from CONCLUSIONS_AND_RECOMMENDATIONS.md
4. Be ready to discuss identifiability constraint and CoVaR trade-offs

### Scenario 3: **Risk Manager (Practitioner)**
1. Read "Practical Recommendations" section in CONCLUSIONS_AND_RECOMMENDATIONS.md
2. Implement diagnostic workflow (Granger test → CoVaR → DebtRank)
3. Compare methods on your own network data
4. Adapt method selection guidance to your risk tolerance

### Scenario 4: **Researcher (Future Work)**
1. Read all of CONCLUSIONS_AND_RECOMMENDATIONS.md
2. Focus on "Advancing Network Recovery" section
3. Consider building GNN improvements or copula extensions
4. Fork/clone code and modify for your experiments

---

## 📞 Support and Questions

**Code Issues:**
- Check `src/recovery_methods.py` line numbers referenced in error messages
- Verify config.yaml is present and valid YAML
- Ensure PyTorch, pandas, numpy, scipy, scikit-learn are installed

**Methodological Questions:**
- Refer to CONCLUSIONS_AND_RECOMMENDATIONS.md "Substantive Conclusions" section
- See "Limitations and Caveats" for known issues
- Contact research advisors (Iván Blanco et al.)

**Reproducibility:**
- All code open-source; all results deterministic (fixed seed=42)
- Floating-point arithmetic may cause ±0.0001 variation across systems
- Benchmark runtime ~4 minutes on typical CPU (Windows 10, Python 3.14)

---

## 📅 Timeline and Milestones

| Date | Milestone | Status |
|------|-----------|--------|
| 2026-10-15 | Benchmark script completed | ✅ Done |
| 2026-10-20 | Direction identifiability diagnostic run | ✅ Done |
| 2026-10-28 | Full 100-experiment benchmark completed | ✅ Done |
| 2026-10-28 | Results, Discussion, Conclusions written | ✅ Done |
| 2026-11-03 | **SUBMISSION DEADLINE** | ⏰ Pending |

---

## 📚 References

### Primary Papers
- Blanco, I., et al. (2024). "Direction Identifiability in Network Recovery from VAR Models." *Journal of Econometrics*. [To be cited]
- Diebold, F. X., & Yilmaz, K. (2009). "Measuring financial asset return and volatility spillovers, with application to global equity markets." *Economic Journal*, 119(534), 158–171.
- Adrian, T., & Brunnermeier, M. K. (2016). "CoVaR." *American Economic Review*, 106(7), 1705–41.

### Dataset
- SRTG-20: Synthetic 20-node risk transmission graph (this study)

### Reproducibility
- Code: [Repository Link]
- Results: This directory
- Hash: 5EA098BA2F8F151003107D8FD1B5E5177084E3B012C52766D98191BE3589815C

---

## 📄 Document Metadata

- **Generated:** 2026-10-28 23:21:47 UTC
- **Generator:** Python 3.14 + pandas + matplotlib + seaborn
- **Benchmark Version:** task4_enhanced_benchmark_with_random.py v1.0
- **Status:** Final, ready for submission
- **Maintainers:** Group 17869 (Emmanuel Odenyire, Francis Kwami Dzikpe)

---

**Last Updated:** 2026-10-28  
**Next Review:** Upon feedback from advisors/reviewers
