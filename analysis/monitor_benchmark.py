#!/usr/bin/env python3
"""
Monitor benchmark progress and auto-generate report when complete.

Usage:
    python monitor_benchmark.py
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess


def check_benchmark_complete(results_file="results/benchmark_results_medium.json"):
    """Check if benchmark has completed by looking for updated results file."""
    path = Path(results_file)
    
    if not path.exists():
        return False, 0
    
    # Check file size and modification time
    size = path.stat().st_size
    mtime = path.stat().st_mtime
    
    return size > 100000, size  # results file should be > 100KB when complete


def wait_for_benchmark(check_interval=60, max_wait=43200):  # 12 hour max
    """Wait for benchmark to complete."""
    start_time = time.time()
    
    print("=" * 80)
    print("MONITORING BENCHMARK PROGRESS")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Check interval: {check_interval} seconds")
    print(f"Max wait time: {max_wait / 3600:.1f} hours\n")
    
    last_size = 0
    checks = 0
    
    while time.time() - start_time < max_wait:
        checks += 1
        elapsed = time.time() - start_time
        
        is_complete, file_size = check_benchmark_complete()
        size_change = file_size - last_size
        
        status = "✓ COMPLETE" if is_complete else "● RUNNING"
        eta_hours = (elapsed / checks) * (60 - checks) / 3600 if checks < 60 else 0
        
        print(f"[{status}] Check #{checks} | Elapsed: {elapsed/3600:.1f}h | "
              f"File: {file_size/1024:.1f}KB (Δ{size_change/1024:+.1f}KB) | ETA: {eta_hours:.1f}h")
        
        if is_complete:
            print(f"\n✓✓✓ BENCHMARK COMPLETE ✓✓✓")
            print(f"Total time: {elapsed / 3600:.2f} hours")
            return True
        
        last_size = file_size
        time.sleep(check_interval)
    
    print(f"\n✗ TIMEOUT: Benchmark did not complete within {max_wait/3600:.0f} hours")
    return False


def generate_report(results_json_file):
    """Generate filled report from results."""
    print("\n" + "=" * 80)
    print("GENERATING FILLED REPORT")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            ["python", "generate_filled_report.py", results_json_file],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("\n✓ Report generation successful")
            return True
        else:
            print(f"\n✗ Report generation failed (exit code {result.returncode})")
            return False
    
    except Exception as e:
        print(f"✗ Error generating report: {e}")
        return False


if __name__ == "__main__":
    # Wait for benchmark
    if wait_for_benchmark():
        # Generate report
        time.sleep(5)  # Brief pause to ensure file is fully written
        generate_report("results/benchmark_results_medium.json")
    
    sys.exit(0)
