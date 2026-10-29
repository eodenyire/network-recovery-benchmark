#!/bin/bash
# Master execution script for M6 capstone report generation
# 
# This script orchestrates the complete workflow:
# 1. Monitors benchmark until completion
# 2. Generates all tables and figures from results
# 3. Fills report template with concrete numbers
# 4. Outputs final M6_CAPSTONE_DRAFT.txt ready for publication

echo "================================================================================"
echo "M6 CAPSTONE REPORT GENERATION - AUTOMATED WORKFLOW"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Step 1: Wait for benchmark
echo "STEP 1: Monitoring benchmark progress..."
python wait_for_new_results.py
if [ $? -ne 0 ]; then
    echo "ERROR: Benchmark monitoring failed"
    exit 1
fi

# Step 2: Generate filled report
echo ""
echo "STEP 2: Generating filled report..."
python smart_report_filler.py results/benchmark_results_medium.json "Complete project report"
if [ $? -ne 0 ]; then
    echo "ERROR: Report generation failed"
    exit 1
fi

# Step 3: Convert to PDF (if pandoc installed)
echo ""
echo "STEP 3: Converting to PDF..."
if command -v pandoc &> /dev/null; then
    pandoc M6_CAPSTONE_DRAFT_FINAL.txt -o M6_CAPSTONE_DRAFT_FINAL.pdf
    echo "✓ PDF generated: M6_CAPSTONE_DRAFT_FINAL.pdf"
else
    echo "⚠ Pandoc not installed; skipping PDF generation"
fi

# Step 4: Summary
echo ""
echo "================================================================================"
echo "✓ WORKFLOW COMPLETE"
echo "================================================================================"
echo "Output files:"
echo "  - M6_CAPSTONE_DRAFT_FINAL.txt (text report)"
echo "  - M6_CAPSTONE_DRAFT_FINAL.pdf (PDF report, if pandoc available)"
echo "  - results/figures/ (all figures and tables)"
echo ""
echo "Next steps:"
echo "  1. Review M6_CAPSTONE_DRAFT_FINAL.txt"
echo "  2. Commit all files to Git"
echo "  3. Submit to WorldQuant University"
echo ""
echo "End time: $(date)"
echo "================================================================================"
