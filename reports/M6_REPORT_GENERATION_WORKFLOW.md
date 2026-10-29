# M6 Capstone Report Generation Workflow

## Overview

This document describes the complete automated workflow for generating the M6 capstone project report. The workflow is designed to:

1. **Execute benchmark experiments** (1,200 runs across 60 configurations)
2. **Extract and aggregate results** into publication-ready metrics
3. **Generate all tables and figures** with concrete numbers
4. **Fill Discussion and Conclusion** sections with empirical findings
5. **Produce final M6_CAPSTONE_DRAFT_FINAL.txt** ready for submission

## Current Status

**Last Updated:** 2026-10-28 20:04 UTC
**Benchmark Progress:** 11/60 configs complete (18.3%)
**ETA to Completion:** ~1.1 hours (approx 21:15 UTC)

---

## Architecture

### Core Components

#### 1. Benchmark Execution (`run_full_benchmark_v2.py`)

```bash
python run_full_benchmark_v2.py --medium
```

**Inputs:**
- `config.yaml` - Experimental grid definition
- `src/*.py` - Estimator implementations (VAR+DY, CoVaR, GNN)
- Synthetic data generator (TCopulaVARProcess)

**Execution:**
- Runs 60 configurations × 20 replications = 1,200 experiments
- Each config: T∈{50,100,150,200,250} × σ∈{0.1,0.3,0.5} × density∈{2,4} × network∈{designed,random}
- Each replication generates synthetic KRI data, fits recovery estimators, computes metrics
- Time: ~4-5 minutes per configuration × 60 configs ≈ 4-5 hours total

**Outputs:**
- `results/benchmark_results_medium.json` - Full results (list of config result objects)
- `results/benchmark_summary_medium.csv` - Aggregated summary statistics
- `results/logs/full_benchmark.log` - Execution log with progress tracking

#### 2. Completion Detection (`wait_for_new_results.py`)

```bash
python wait_for_new_results.py
```

**Function:** Monitors `results/benchmark_results_medium.json` for:
- File size growth (should exceed 50KB when complete)
- Modification time (must be recent, after script started)

**Polling Interval:** 60 seconds
**Max Wait Time:** 12 hours
**Output:** Exits with success code when new results detected

#### 3. Results Extraction & Aggregation (`generate_filled_report.py`)

```bash
python generate_filled_report.py results/benchmark_results_medium.json
```

**Function:**
- Loads JSON results
- Groups by configuration (T, σ, ν, density, network_source)
- Aggregates metrics across replications (mean, std)
- Extracts metrics for specific cells (e.g., easiest cell)

**Key Extractions:**
- Table 8: Recovery metrics at T=250, σ=0.1, ν=5, density=2, designed
- Table 9: Continuous metrics (edge-weight correlation, impulse-response MSE)
- Table 11: Density effect (mean F1 at density=2 vs 4)
- Table 12: Robustness check (designed vs random networks)

**Outputs:**
- `FILLED_TABLES.txt` - Formatted markdown tables ready for report

#### 4. Smart Report Filler (`smart_report_filler.py`)

```bash
python smart_report_filler.py results/benchmark_results_medium.json "Complete project report"
```

**Function:**
- Reads benchmark results JSON
- Extracts all metrics needed for report
- Fills Discussion section with empirical findings
- Substitutes all [X.XXX] placeholders with concrete numbers

**Key Features:**
- Honest diagnostic narratives (e.g., VAR+DY stationarity trade-off)
- Automated table generation with proper formatting
- Figure generation (F1 vs T, F1 vs σ degradation curves)

**Outputs:**
- `M6_CAPSTONE_DRAFT_FINAL.txt` - Complete filled report

#### 5. End-to-End Automation (`FINAL_REPORT_GENERATOR.py`)

```bash
python FINAL_REPORT_GENERATOR.py
```

**Function:** Master orchestration script that:
1. Waits for benchmark completion (max 12 hours)
2. Loads and aggregates results
3. Generates all tables and figures
4. Fills Discussion and Conclusion sections
5. Outputs final M6_CAPSTONE_DRAFT_FINAL.txt

**Advantage:** Single script, no manual intervention needed

---

## Workflow Timeline

| Time | Action | Status |
|------|--------|--------|
| 19:50 UTC | Benchmark start | ✓ In progress |
| 20:04 UTC | Benchmark 18% complete (11/60 configs) | ✓ Running |
| ~21:15 UTC | Benchmark completes | ⏳ Waiting |
| 21:16 UTC | wait_for_new_results.py detects completion | ⏳ Waiting |
| 21:17-21:30 UTC | smart_report_filler.py generates report | ⏳ Waiting |
| 21:31 UTC | M6_CAPSTONE_DRAFT_FINAL.txt created | ⏳ Waiting |
| 21:32 UTC | PDF export (if pandoc available) | ⏳ Waiting |
| 21:33 UTC | Git commit and push | ⏳ Waiting |
| **21:35 UTC** | **READY FOR SUBMISSION** | ⏳ Waiting |

---

## Expected Report Outputs

### Files Generated

#### Primary Report
- **M6_CAPSTONE_DRAFT_FINAL.txt** (120-150 KB plaintext)
  - Complete report with all placeholders filled
  - Ready for copy-paste into LMS or for PDF conversion
  
- **M6_CAPSTONE_DRAFT_FINAL.pdf** (if pandoc installed)
  - Publication-ready PDF format
  - Command: `pandoc M6_CAPSTONE_DRAFT_FINAL.txt -o M6_CAPSTONE_DRAFT_FINAL.pdf`

#### Data & Figures
- **results/benchmark_results_medium.json** - Raw results (60 configs × 20 reps)
- **results/benchmark_summary_medium.csv** - Aggregated statistics
- **results/figures/figure_1_f1_vs_time.png** - F1 degradation vs time steps
- **results/figures/figure_2_f1_vs_noise.png** - F1 degradation vs measurement noise

#### Documentation
- This file (M6_REPORT_GENERATION_WORKFLOW.md)
- REPORT_TEMPLATE_SECTIONS.md (breakdown of all sections)
- DISCUSSION_AND_CONCLUSION_TEMPLATE.md (narrative templates)

---

## Concrete Numbers to Expect

Based on the Expectations document and earlier verification runs:

### Trivial Baselines (Deterministic)
```
Predict-all precision: 40 edges / 400 pairs = 0.0975
Predict-all recall: 1.0000
Predict-all F1: 0.1777
Predict-all AUC-ROC: 0.5000
Random-k F1: ~0.11 ± 0.05
```

### VAR+DY Easiest Cell (T=250, σ=0.1, ν=5, density=2)
```
Expected (from Expectations doc): AUC-ROC 0.65–0.80, F1 0.20–0.35
Actual (likely): AUC-ROC ~0.50-0.63, F1 ~0.15-0.25
Reason: Stationarity rescaling dampens lag matrix signal (honest diagnostic)
```

### CoVaR Easiest Cell
```
Expected: AUC-ROC 0.50–0.60 (near-random), F1 0.15–0.22
Role: Diagnostic (pairwise tail-risk, not graph recovery)
```

### Robustness Check (Designed vs. Random)
```
Expected: Outcome A or B (both publishable)
Outcome A: Designed ≈ Random (density matters more than topology)
Outcome B: Designed > Random (documented topology carries signal)
```

---

## Quality Assurance

### Validation Checks

When report is complete, verify:

- [ ] All [X.XXX] placeholders filled with concrete numbers
- [ ] Propagation sanity check values match Contagion_Propagation.csv (MAE ≈ 0.0177)
- [ ] Trivial baseline metrics match deterministic expectations (F1 = 0.1777)
- [ ] All tables have proper formatting and markdown syntax
- [ ] All figures referenced in text with proper captions
- [ ] Discussion contains honest diagnostics (VAR+DY stationarity trade-off, CoVaR limitations)
- [ ] Conclusion addresses limitations and practical implications
- [ ] All citations present and formatted per references section
- [ ] No dangling reference markers or incomplete sentences
- [ ] File size reasonable (120-150 KB for complete report)

### Numerical Alignment Checks

- VAR+DY AUC-ROC should degrade as T decreases
- CoVaR AUC-ROC should consistently hover near 0.50-0.60
- GNN should show steeper degradation than VAR+DY at low T
- Designed network F1 should be ≥ Random network F1 (or approximately equal)

---

## Commands Reference

### Quick Start (After Benchmark Complete)

```bash
# Single-line end-to-end automation
python FINAL_REPORT_GENERATOR.py

# Or step-by-step with individual scripts
python wait_for_new_results.py
python smart_report_filler.py results/benchmark_results_medium.json "Complete project report"
```

### Individual Components

```bash
# Start benchmark
python run_full_benchmark_v2.py --medium

# Monitor progress
python wait_for_new_results.py

# Extract and format results
python generate_filled_report.py results/benchmark_results_medium.json

# Generate report (detailed)
python FINAL_REPORT_GENERATOR.py

# Convert to PDF
pandoc M6_CAPSTONE_DRAFT_FINAL.txt -o M6_CAPSTONE_DRAFT_FINAL.pdf

# Git operations
git add M6_CAPSTONE_DRAFT_FINAL.txt M6_CAPSTONE_DRAFT_FINAL.pdf
git add results/benchmark_results_medium.json results/benchmark_summary_medium.csv
git add results/figures/*.png
git commit -m "Add M6 capstone report final submission with benchmark results"
git push origin main
```

---

## Troubleshooting

### Problem: "wait_for_new_results.py hangs indefinitely"

**Diagnosis:** Benchmark may have crashed or results file not being written
**Solution:**
1. Check `results/logs/full_benchmark.log` for errors
2. Verify `results/benchmark_results_medium.json` is being updated: `ls -la results/benchmark_results_medium.json`
3. Check benchmark process: `list_processes` or `ps aux | grep python`
4. Restart benchmark if needed: `python run_full_benchmark_v2.py --medium`

### Problem: "smart_report_filler.py crashes with 'configuration not found'"

**Diagnosis:** Results file format differs from expected
**Solution:**
1. Verify results file is valid JSON: `python -m json.tool results/benchmark_results_medium.json > /dev/null`
2. Check file size: should be > 50KB when complete
3. Try with smaller test results: use `results/benchmark_results_quick.json` if available
4. Inspect first record: `python -c "import json; f=open('results/benchmark_results_medium.json'); d=json.load(f); print(d[0] if isinstance(d,list) else d['experiments'][0])"`

### Problem: "Unicode encoding errors in output"

**Diagnosis:** PowerShell console encoding issue
**Solution:**
1. Save output to file: `python script.py > output.txt 2>&1`
2. Use explicit encoding: `python -c "import sys; sys.stdout.reconfigure(encoding='utf-8')"`
3. Check Python version: ensure Python 3.7+
4. On Windows: set `PYTHONIOENCODING=utf-8` environment variable

---

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `run_full_benchmark_v2.py` | Execute benchmark | ✓ Created |
| `wait_for_new_results.py` | Monitor completion | ✓ Created |
| `generate_filled_report.py` | Extract metrics | ✓ Created |
| `smart_report_filler.py` | Fill placeholders | ✓ Created |
| `assemble_final_report.py` | Final assembly | ✓ Created |
| `FINAL_REPORT_GENERATOR.py` | End-to-end automation | ✓ Created |
| `DISCUSSION_AND_CONCLUSION_TEMPLATE.md` | Narrative templates | ✓ Created |
| `REPORT_TEMPLATE_SECTIONS.md` | Section breakdown | ✓ Created |
| `Complete project report` | Template (input) | ✓ Exists |
| `config.yaml` | Experimental grid | ✓ Updated |
| `src/synthetic_generator.py` | DGP implementation | ✓ Working |
| `src/recovery_methods.py` | Estimators | ✓ Working |

---

## Success Criteria

Report generation is complete when:

1. ✓ `M6_CAPSTONE_DRAFT_FINAL.txt` exists and is readable
2. ✓ All [X.XXX] placeholders have been replaced with concrete numbers
3. ✓ All tables (8, 9, 11, 12, 13) are filled with real results
4. ✓ All figures (1, 2) are present in `results/figures/`
5. ✓ Discussion section contains empirical findings from all 60 configurations
6. ✓ Conclusion section addresses limitations and practical implications
7. ✓ File passes basic validation (>100KB, UTF-8 encoding, valid markdown)
8. ✓ Git commit includes report, results, and figures

---

## Timeline to Submission

- **Oct 28, 2026 (~21:35 UTC):** Report generation complete
- **Oct 28-Nov 3:** Review, minor revisions, formatting
- **Nov 3, 2026 (23:59):** Submit to WorldQuant University (DEADLINE)

**Days to deadline:** ~6 days
**Slack time:** Ample for review and revision

---

## Authors & Contact

- **Emmanuel Odenyire Anyira** (implementation, benchmarking)
- **Francis Dzikpe** (methodology, oversight)

---

**Last Updated:** 2026-10-28 20:04 UTC
**Next Update:** Upon benchmark completion (~21:15 UTC)
