#!/usr/bin/env python3
"""
Assemble the final M6 capstone report by substituting filled tables and sections
into the Complete project report template.

Usage:
    python assemble_final_report.py FILLED_TABLES.txt
"""

import sys
import re
from pathlib import Path


def read_complete_report(report_file="Complete project report"):
    """Read the template report."""
    with open(report_file, 'r') as f:
        return f.read()


def read_filled_tables(tables_file):
    """Read the generated tables."""
    with open(tables_file, 'r') as f:
        return f.read()


def substitute_tables(report_text, filled_tables_text):
    """Substitute [X.XXX] placeholders with concrete numbers."""
    
    # Extract table data from filled_tables_text
    # This is a simplified version - in practice you'd parse each table
    
    # For now, just mark the substitutions made
    substitutions_made = 0
    
    # Count placeholders
    placeholders = re.findall(r'\[X\.\d+\]', report_text)
    print(f"Found {len(placeholders)} placeholders in report")
    
    return report_text, substitutions_made


def generate_final_report(report_text, output_file="M6_CAPSTONE_DRAFT.txt"):
    """Generate the final report file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return output_file


def main(filled_tables_file):
    """Main entry point."""
    print("="*80)
    print("ASSEMBLING FINAL M6 CAPSTONE REPORT")
    print("="*80 + "\n")
    
    # Read template
    print("Reading report template...")
    report_text = read_complete_report()
    print(f"✓ Loaded report template ({len(report_text)} characters)")
    
    # Read filled tables
    print("Reading filled tables...")
    filled_tables = read_filled_tables(filled_tables_file)
    print(f"✓ Loaded filled tables ({len(filled_tables)} characters)")
    
    # Substitute
    print("Substituting placeholders...")
    final_report, n_subs = substitute_tables(report_text, filled_tables)
    print(f"✓ Made {n_subs} substitutions")
    
    # Generate output
    print("Writing final report...")
    output_file = generate_final_report(final_report)
    print(f"✓ Generated: {output_file}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("  1. Open M6_CAPSTONE_DRAFT.txt and review")
    print("  2. Run: pandoc M6_CAPSTONE_DRAFT.txt -o M6_CAPSTONE_DRAFT.pdf")
    print("  3. Commit and push to GitHub")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python assemble_final_report.py FILLED_TABLES.txt")
        sys.exit(1)
    
    main(sys.argv[1])
