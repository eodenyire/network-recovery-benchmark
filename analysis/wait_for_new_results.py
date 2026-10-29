#!/usr/bin/env python3
"""
Wait for new benchmark results and extract metrics when ready.

This script checks if benchmark_results_medium.json has been updated
(to distinguish it from the old file) and then extracts and displays
the key results tables.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime


RESULTS_FILE = Path("results/benchmark_results_medium.json")
STARTED_AT = datetime.now()
CHECK_INTERVAL = 60  # seconds


def get_file_info():
    """Get file size and modification time."""
    if not RESULTS_FILE.exists():
        return None, None
    
    mtime = RESULTS_FILE.stat().st_mtime
    size = RESULTS_FILE.stat().st_size
    return mtime, size


def is_new_file(mtime):
    """Check if file was modified after script started."""
    return mtime > STARTED_AT.timestamp()


def main():
    """Main polling loop."""
    print("="*80)
    print("WAITING FOR NEW BENCHMARK RESULTS")
    print("="*80)
    print(f"Started at: {STARTED_AT.isoformat()}")
    print(f"Results file: {RESULTS_FILE}")
    print(f"Check interval: {CHECK_INTERVAL} seconds\n")
    
    last_size = 0
    checks = 0
    
    while True:
        checks += 1
        mtime, size = get_file_info()
        
        if mtime is None:
            status = "NOT FOUND"
            size_str = "0KB"
        else:
            is_new = is_new_file(mtime)
            status = "✓ NEW" if is_new else "● OLD"
            size_str = f"{size/1024:.1f}KB"
            
            if is_new and size > 50000:  # Results file should be > 50KB when meaningful
                print(f"\n{'='*80}")
                print(f"✓✓✓ NEW RESULTS DETECTED ✓✓✓")
                print(f"File size: {size_str}")
                print(f"Modification time: {datetime.fromtimestamp(mtime).isoformat()}")
                print(f"{'='*80}\n")
                return True
        
        elapsed = time.time() - STARTED_AT.timestamp()
        size_delta = size - last_size if size else 0
        
        print(f"[{status}] Check #{checks} | Elapsed: {elapsed/60:.1f}m | "
              f"File: {size_str} (Δ{size_delta/1024:+.1f}KB)")
        
        last_size = size
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        if main():
            print("Now running: python generate_filled_report.py results/benchmark_results_medium.json")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
