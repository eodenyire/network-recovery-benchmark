# Installation & Setup Guide

This guide walks through installing SRTG-20 from scratch, configuring the environment, and verifying everything works.

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux, macOS, or Windows (PowerShell/CMD) |
| **Python** | 3.8+ (3.10+ recommended for performance) |
| **RAM** | 8 GB minimum (16 GB recommended for full benchmark) |
| **Disk** | 2 GB free (for dependencies + results) |
| **GPU** | Optional (CUDA 11.8+ for torch acceleration) |

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/WorldQuantCapstone/SRTG-20.git
cd SRTG-20
```

---

## Step 2: Create a Virtual Environment

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

Verify activation:
```bash
which python  # Linux/macOS
# or
where python  # Windows
```

Should point to `/path/to/SRTG-20/venv/...`

---

## Step 3: Install Dependencies

```bash
# Upgrade pip, setuptools, wheel (important!)
pip install --upgrade pip setuptools wheel

# Install pinned dependencies
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed numpy-1.24.3 pandas-2.0.3 scipy-1.11.2 ... (13 packages)
```

**Installation time:** 5–10 minutes (varies by internet speed)

### Troubleshooting Installation

#### Issue: `ERROR: Could not find a version that satisfies the requirement torch==2.0.1`

**Cause:** torch may not be available for your OS/Python version.

**Solution:** Install PyTorch from official source:
```bash
# For CPU-only:
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.8):
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Then install torch-geometric:
pip install torch-geometric==2.3.1
```

#### Issue: `ImportError: No module named 'torch'` after installation

**Cause:** Virtual environment not activated.

**Solution:** Activate venv (see Step 2) and try again.

#### Issue: `PermissionError: [Errno 13] Permission denied`

**Cause:** Pip cache issue or insufficient permissions.

**Solution:**
```bash
pip install --no-cache-dir -r requirements.txt
# or
pip install --user -r requirements.txt
```

---

## Step 4: Verify Installation

```bash
python -c "import numpy, pandas, scipy, sklearn, networkx, torch; print('✓ All core packages imported successfully')"
```

Expected output:
```
✓ All core packages imported successfully
```

### Quick Import Test

```bash
python -c "from src.synthetic_generator import TCopulaVARProcess; from src.recovery_methods import VARDieboldYilmaz; print('✓ SRTG-20 modules imported')"
```

---

## Step 5: Run Quick Smoke Test

```bash
# Test data generation
python -c "
from src.synthetic_generator import PlantedNetworkGenerator, TCopulaVARProcess
import numpy as np

net_gen = PlantedNetworkGenerator(edge_file='data/srtg20_edges.csv')
W_true = net_gen.generate()
print(f'Network: {W_true.shape}, {np.count_nonzero(W_true)} edges')

dgp = TCopulaVARProcess(W_true, T=50, noise_level=0.1, t_df=5, seed=42)
X = dgp.generate()
print(f'Data: {X.shape}')
print('✓ Smoke test passed')
"
```

Expected output:
```
Network: (20, 20), 41 edges
Data: (50, 100)
✓ Smoke test passed
```

---

## Step 6: Configuration

All experiments are controlled via `config.yaml`. Review and modify as needed:

```yaml
# config.yaml

# Data Generation
num_prts: 20
num_kris_per_prt: 5
edge_file: "data/srtg20_edges.csv"

# Experiment Grid
time_steps: [50, 100, 150, 200, 250]
noise_levels: [0.1, 0.3, 0.5]
t_df_values: [3, 5, 10]
network_density: [2, 4]  # edges per node

# Random Seed (for reproducibility)
seed: 42

# Output
output_dir: "results"
benchmark_file: "task4_enhanced_benchmark_with_random.json"
summary_file: "benchmark_summary.csv"
```

### Custom Experiment

To run a single configuration:

```python
from src.experiment_runner import ExperimentRunner

runner = ExperimentRunner('config.yaml')
results = runner.run_single_configuration(
    time_steps=250,
    noise_level=0.1,
    network_density=2,
    t_df=5,
    num_reps=10  # 10 replications (not 100) for quick testing
)
```

---

## Step 7: Run Experiments

### Option A: Quick Benchmark (Testing, 2–5 minutes)

```bash
python main.py --quick --replicates 5
```

- Runs 1 configuration with 5 replications
- Memory: ~2 GB
- Output: `results/quick_benchmark.json`

### Option B: Full Benchmark (Production, 4–6 hours)

```bash
python main.py --full --replicates 100
```

- Runs 20 configurations with 100 replications each
- Memory: ~16 GB
- Output: `results/task4_enhanced_benchmark_with_random.json`

### Option C: Custom Run

```bash
python main.py --time-steps 250 --noise 0.1 --replicates 50
```

---

## Step 8: Generate Results

### Convert Results to CSV Summary

```bash
python scripts/generate_results_tables.py
```

Output: `results/benchmark_summary.csv` (one row per configuration)

### Generate Visualizations

```bash
python scripts/generate_visualizations.py
```

Output: `results/figures/`
- `f1_vs_data_volume.png`
- `f1_vs_noise.png`
- `auc_roc_vs_data_volume.png`
- `f1_heatmap.png`
- `auc_roc_heatmap.png`

---

## Step 9: Run Tests

```bash
# Full test suite
python -m pytest tests/ -v

# Or individual tests
python tests/test_quick.py          # Quick smoke tests
python tests/test_data_gen.py       # Data generation verification
python tests/test_suite.py          # Full suite
```

Expected: All tests pass (✓)

---

## Deactivate Virtual Environment

When finished:

```bash
deactivate
```

---

## Environment Variables (Optional)

### GPU Acceleration (if available)

```bash
# Force GPU (CUDA):
export CUDA_VISIBLE_DEVICES=0
python main.py --quick

# Force CPU:
export CUDA_VISIBLE_DEVICES=""
python main.py --quick
```

### Memory Management

```bash
# Limit PyTorch memory (useful for limited RAM):
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=256
python main.py --quick
```

---

## Docker Setup (Alternative)

For containerized reproducibility:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py", "--quick"]
```

Build and run:
```bash
docker build -t srtg20 .
docker run --rm srtg20
```

---

## Troubleshooting: Common Issues

### Issue 1: `ModuleNotFoundError: No module named 'src'`

**Cause:** Working directory is not the repo root.

**Solution:**
```bash
cd /path/to/SRTG-20
python main.py --quick
```

### Issue 2: `FileNotFoundError: config.yaml`

**Cause:** config.yaml not found in working directory.

**Solution:**
```bash
ls config.yaml  # Verify file exists
# or
python main.py --config /path/to/config.yaml --quick
```

### Issue 3: `MemoryError` during full benchmark

**Cause:** Insufficient RAM (need 16 GB for full benchmark).

**Solution:**
```bash
# Run quick benchmark instead:
python main.py --quick --replicates 10

# Or run one configuration at a time:
python -c "
from src.experiment_runner import ExperimentRunner
runner = ExperimentRunner('config.yaml')
results = runner.run_single_configuration(
    time_steps=250,
    noise_level=0.1,
    num_reps=20
)
"
```

### Issue 4: CUDA Out of Memory (OOM) on GPU

**Cause:** Batch size too large for GPU memory.

**Solution:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=256
python main.py --quick
```

### Issue 5: `RuntimeError: CUDA out of memory`

**Cause:** GNN model too large for GPU.

**Solution:** Use CPU or reduce batch size in `config.yaml`:
```yaml
gnn:
  batch_size: 32  # Reduce from default
```

---

## Performance Benchmarks

Expected runtime on different hardware:

| Config | CPU (i7-10700) | GPU (RTX 3080) | Memory |
|--------|---|---|---|
| Quick (5 reps) | 2–5 min | 1–2 min | 2 GB |
| Medium (50 reps) | 30–45 min | 15–20 min | 6 GB |
| Full (100 reps) | 4–6 hours | 2–3 hours | 16 GB |

---

## Next Steps

Once installed and verified:

1. **Run quick benchmark** to verify setup: `python main.py --quick`
2. **Read README.md** for usage examples
3. **Review `docs/CONTRIBUTORS.md`** for team roles
4. **Check `scripts/`** for utility tools (visualization, reporting)
5. **Explore `analysis/`** for exploratory and diagnostic scripts

---

## Getting Help

- **Installation issues:** Check troubleshooting section above
- **Code issues:** Review docstrings in `src/` (in-code comments)
- **Results questions:** See `README.md` under "Expected Results"
- **Contributing:** Contact Emmanuel Odenyire or Iván Blanco

---

*Last Updated: October 29, 2026*
