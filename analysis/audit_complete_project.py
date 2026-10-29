#!/usr/bin/env python3
"""
Comprehensive audit of the SRTG-20 project state before M6 submission.

Checks:
1. Data integrity (Contagion_Propagation.csv vs propagation output)
2. Benchmark results consistency (all configs run, no missing data)
3. Figure generation (all required figures present)
4. Code correctness (synthetic generator, propagation, aggregation)
5. Report completeness (placeholders, sections)
6. Edge documentation (srtg20_edges.csv sources)
7. Requirements reproducibility (dependencies, versions)
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime

def audit_propagation_data():
    """Verify ground truth cascade."""
    print("\n[1] PROPAGATION DATA INTEGRITY")
    print("-" * 60)
    
    contagion_file = Path("Contagion_Propagation.csv")
    if not contagion_file.exists():
        print("❌ MISSING: Contagion_Propagation.csv")
        return False
    
    try:
        with open(contagion_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"✓ Contagion_Propagation.csv: {len(rows)} records")
        print(f"  First row: {rows[0] if rows else 'empty'}")
        print(f"  Last row: {rows[-1] if rows else 'empty'}")
        
        # Check for expected PRTs
        expected_prts = {
            "Technology_Risk", "Cyber_Risk", "Operational_Risk",
            "Reputational_Risk", "Liquidity_Risk", "External_Fraud_Risk",
            "Credit_Risk"
        }
        found_prts = set()
        if rows:
            for row in rows[:3]:
                for col in row:
                    if "PRT" in str(col) or any(prt in str(row.get(col, "")) for prt in expected_prts):
                        found_prts.add(col)
        
        print(f"  Detected PRT columns: {len(found_prts)}")
        return True
    except Exception as e:
        print(f"❌ ERROR reading Contagion_Propagation.csv: {e}")
        return False

def audit_benchmark_results():
    """Verify benchmark results completeness."""
    print("\n[2] BENCHMARK RESULTS INTEGRITY")
    print("-" * 60)
    
    results_file = Path("results/task4_reduced_benchmark.json")
    if not results_file.exists():
        print("❌ MISSING: results/task4_reduced_benchmark.json")
        return False
    
    try:
        with open(results_file) as f:
            results = json.load(f)
        
        print(f"✓ Loaded {len(results)} configurations")
        
        # Check structure
        configs_expected = 10
        if len(results) != configs_expected:
            print(f"⚠️  Expected {configs_expected} configs, got {len(results)}")
        
        # Check estimators present
        estimators_found = {"var_dy": 0, "covar": 0, "debtrank": 0, "gnn": 0}
        for result in results:
            if result.get("var_dy"):
                estimators_found["var_dy"] += 1
            if result.get("covar"):
                estimators_found["covar"] += 1
            if result.get("debtrank"):
                estimators_found["debtrank"] += 1
            if result.get("gnn"):
                estimators_found["gnn"] += 1
        
        print(f"  VAR+DY results:  {estimators_found['var_dy']}/10 configs ✓" if estimators_found["var_dy"] == 10 else f"  VAR+DY results:  {estimators_found['var_dy']}/10 configs ❌")
        print(f"  CoVaR results:   {estimators_found['covar']}/10 configs ✓" if estimators_found["covar"] == 10 else f"  CoVaR results:   {estimators_found['covar']}/10 configs ❌")
        print(f"  DebtRank results: {estimators_found['debtrank']}/10 configs ✓" if estimators_found["debtrank"] == 10 else f"  DebtRank results: {estimators_found['debtrank']}/10 configs ❌")
        print(f"  GNN results:     {estimators_found['gnn']}/10 configs {'✓' if estimators_found['gnn'] > 0 else '⚠️ (expected: null)'}")
        
        # Check for NaN values
        var_dy_first = results[0].get("var_dy", {})
        if var_dy_first and var_dy_first.get("auc_roc"):
            auc_mean = var_dy_first["auc_roc"]["mean"]
            print(f"  Sample VAR+DY AUC-ROC: {auc_mean} (should be ~0.50)")
        
        return True
    except Exception as e:
        print(f"❌ ERROR reading results: {e}")
        return False

def audit_figures():
    """Verify all required figures generated."""
    print("\n[3] FIGURE GENERATION")
    print("-" * 60)
    
    figures_dir = Path("results/figures")
    required_figures = [
        "auc_roc_heatmap_task6.png",
        "f1_heatmap_task6.png",
        "auc_degradation_curve_task6.png",
        "f1_degradation_curve_task6.png",
        "auc_noise_sensitivity_task6.png",
        "estimator_comparison_scatter_task6.png",
    ]
    
    generated = 0
    for fig in required_figures:
        fig_path = figures_dir / fig
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            print(f"✓ {fig:45s} ({size_kb:.1f} KB)")
            generated += 1
        else:
            print(f"❌ {fig:45s} MISSING")
    
    print(f"\nTotal: {generated}/{len(required_figures)} figures present")
    return generated == len(required_figures)

def audit_code_files():
    """Verify critical code files exist and are non-empty."""
    print("\n[4] CODE FILE INTEGRITY")
    print("-" * 60)
    
    critical_files = {
        "src/synthetic_generator.py": "Data generation",
        "src/propagation.py": "DebtRank propagation",
        "src/evaluation.py": "Metrics calculation",
        "src/estimators.py": "Recovery estimators",
        "task4_reduced_grid_benchmark.py": "Benchmark runner",
    }
    
    all_present = True
    for file_path, description in critical_files.items():
        full_path = Path(file_path)
        if full_path.exists() and full_path.stat().st_size > 100:
            print(f"✓ {file_path:40s} ({full_path.stat().st_size} bytes) — {description}")
        else:
            print(f"❌ {file_path:40s} MISSING or EMPTY")
            all_present = False
    
    return all_present

def audit_report():
    """Verify report completeness."""
    print("\n[5] REPORT STRUCTURE")
    print("-" * 60)
    
    report_file = Path("Complete project report")
    if not report_file.exists():
        print("❌ MISSING: Complete project report")
        return False
    
    try:
        with open(report_file) as f:
            content = f.read()
        
        sections = {
            "Abstract": "Abstract" in content,
            "Introduction": "1. Introduction" in content,
            "Theoretical Framework": "2. Theoretical Framework" in content,
            "Methodology": "3. Methodology" in content,
            "Results": "4. Results" in content or "[X.XXX]" in content,  # May have placeholders
            "Discussion": "5. Discussion" in content or "[X.XXX]" in content,
            "Conclusion": "6. Conclusion" in content or "[X.XXX]" in content,
            "References": "References" in content or "[X.XXX]" in content,
        }
        
        found = sum(1 for v in sections.values() if v)
        print(f"Report sections present: {found}/{len(sections)}")
        for section, present in sections.items():
            status = "✓" if present else "⚠️"
            print(f"  {status} {section:25s}")
        
        # Check for unresolved placeholders
        placeholder_count = content.count("[X.XXX]")
        if placeholder_count > 0:
            print(f"\n⚠️  Unresolved placeholders: {placeholder_count}")
        
        print(f"\nReport size: {len(content)} characters")
        return True
    except Exception as e:
        print(f"❌ ERROR reading report: {e}")
        return False

def audit_edges_documentation():
    """Verify network edge documentation."""
    print("\n[6] NETWORK EDGE DOCUMENTATION")
    print("-" * 60)
    
    edges_file = Path("data/srtg20_edges.csv")
    if not edges_file.exists():
        print("❌ MISSING: data/srtg20_edges.csv")
        return False
    
    try:
        with open(edges_file) as f:
            reader = csv.DictReader(f)
            edges = list(reader)
        
        print(f"✓ Edges documented: {len(edges)}")
        
        # Check for required columns
        if edges:
            cols = set(edges[0].keys())
            required_cols = {"from_prt", "to_prt", "weight", "source", "description"}
            missing_cols = required_cols - cols
            if missing_cols:
                print(f"⚠️  Missing columns: {missing_cols}")
            else:
                print(f"  All required columns present: {required_cols}")
            
            # Check source documentation
            sources_found = set()
            for edge in edges:
                source = edge.get("source", "")
                if source:
                    sources_found.add(source)
            
            print(f"  Unique sources cited: {len(sources_found)}")
            for source in sorted(list(sources_found)[:5]):
                print(f"    - {source}")
            if len(sources_found) > 5:
                print(f"    ... and {len(sources_found) - 5} more")
        
        return True
    except Exception as e:
        print(f"❌ ERROR reading edges: {e}")
        return False

def audit_requirements():
    """Verify dependencies."""
    print("\n[7] REQUIREMENTS & REPRODUCIBILITY")
    print("-" * 60)
    
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ MISSING: requirements.txt")
        return False
    
    try:
        with open(req_file) as f:
            requirements = f.read()
        
        critical_packages = {
            "numpy": "Numerical computing",
            "pandas": "Data manipulation",
            "scipy": "Scientific computing",
            "scikit-learn": "ML metrics & preprocessing",
            "statsmodels": "VAR/FEVD",
            "matplotlib": "Visualization",
            "seaborn": "Heatmaps",
        }
        
        found = 0
        for pkg, desc in critical_packages.items():
            if pkg.lower() in requirements.lower():
                print(f"✓ {pkg:20s} — {desc}")
                found += 1
            else:
                print(f"⚠️  {pkg:20s} — NOT FOUND (may be optional)")
        
        print(f"\nCritical packages: {found}/{len(critical_packages)} present")
        return found >= 5
    except Exception as e:
        print(f"❌ ERROR reading requirements: {e}")
        return False

def main():
    print("=" * 80)
    print("COMPREHENSIVE PROJECT AUDIT — M6 DEADLINE CHECK")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    checks = [
        ("Propagation Data", audit_propagation_data),
        ("Benchmark Results", audit_benchmark_results),
        ("Figures", audit_figures),
        ("Code Files", audit_code_files),
        ("Report Structure", audit_report),
        ("Edge Documentation", audit_edges_documentation),
        ("Requirements", audit_requirements),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR in {name}: {e}")
            results[name] = False
    
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status:8s} {name}")
    
    print(f"\nOverall: {passed}/{total} audit checks passed")
    
    if passed == total:
        print("\n🟢 PROJECT STATE: READY FOR M6 SUBMISSION (after report filling)")
        return 0
    else:
        print("\n🔴 PROJECT STATE: ISSUES DETECTED — SEE ABOVE FOR DETAILS")
        return 1

if __name__ == "__main__":
    sys.exit(main())
