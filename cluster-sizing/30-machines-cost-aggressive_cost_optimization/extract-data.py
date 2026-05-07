#!/usr/bin/env python3
"""
Extract statistics from Bayesian Optimization results
Calculates standard deviation and other useful metrics for paper tables
"""

import json
import numpy as np
from pathlib import Path
import sys


def load_results(filepath):
    """Load optimization results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_statistics(results):
    """Calculate comprehensive statistics from optimization results"""
    evaluations = results['optimization_results']['all_evaluations']
    
    # Filter successful evaluations (those with makespan_minutes)
    successful_evals = [e for e in evaluations 
                       if 'makespan_minutes' in e and 'total_cost' in e]
    
    if not successful_evals:
        return None
    
    # Extract final costs (the primary metric we care about)
    costs = [e['total_cost'] for e in successful_evals]
    makespans = [e['makespan_minutes'] for e in successful_evals]
    
    # Calculate statistics
    stats = {
        'n_trials': len(successful_evals),
        'cost': {
            'mean': float(np.mean(costs)),
            'median': float(np.median(costs)),
            'std': float(np.std(costs, ddof=1)),  # Sample std dev
            'min': float(np.min(costs)),
            'max': float(np.max(costs)),
            'q1': float(np.percentile(costs, 25)),
            'q3': float(np.percentile(costs, 75)),
            'iqr': float(np.percentile(costs, 75) - np.percentile(costs, 25)),
            'range': float(np.max(costs) - np.min(costs)),
            'p10': float(np.percentile(costs, 10)),
            'p90': float(np.percentile(costs, 90)),
        },
        'makespan': {
            'mean': float(np.mean(makespans)),
            'median': float(np.median(makespans)),
            'std': float(np.std(makespans, ddof=1)),
            'min': float(np.min(makespans)),
            'max': float(np.max(makespans)),
            'q1': float(np.percentile(makespans, 25)),
            'q3': float(np.percentile(makespans, 75)),
            'iqr': float(np.percentile(makespans, 75) - np.percentile(makespans, 25)),
        }
    }
    
    return stats


def format_table_row(algo_name, stats):
    """Format statistics as a LaTeX table row"""
    c = stats['cost']
    row = f"{algo_name} & " \
          f"\\${c['mean']:.2f} & " \
          f"\\${c['median']:.2f} & " \
          f"\\${c['std']:.2f} & " \
          f"\\${c['iqr']:.2f} & " \
          f"\\${c['min']:.2f} & " \
          f"\\${c['max']:.2f} & " \
          f"\\${c['range']:.2f} \\\\"
    return row


def format_quantile_table(stats_dict):
    """Format quantile comparison table"""
    quantiles = [10, 25, 50, 75, 90]
    rows = []
    
    for q in quantiles:
        if q == 10:
            key = 'p10'
        elif q == 25:
            key = 'q1'
        elif q == 50:
            key = 'median'
        elif q == 75:
            key = 'q3'
        elif q == 90:
            key = 'p90'
        
        classic_val = stats_dict['bo-classic']['cost'][key]
        enhanced_val = stats_dict['bo-enhanced']['cost'][key]
        diff = classic_val - enhanced_val
        pct_improvement = (diff / classic_val) * 100
        
        row = f"{q}th & " \
              f"\\${classic_val:.2f} & " \
              f"\\${enhanced_val:.2f} & " \
              f"\\${diff:.2f} & " \
              f"{pct_improvement:.1f}\\% \\\\"
        rows.append(row)
    
    return rows


def print_summary(algo_name, stats):
    """Print human-readable summary"""
    print(f"\n{'='*60}")
    print(f"Algorithm: {algo_name}")
    print(f"{'='*60}")
    print(f"Number of trials: {stats['n_trials']}")
    print(f"\nCost Statistics:")
    print(f"  Mean:   ${stats['cost']['mean']:.2f}")
    print(f"  Median: ${stats['cost']['median']:.2f}")
    print(f"  Std:    ${stats['cost']['std']:.2f}")
    print(f"  Min:    ${stats['cost']['min']:.2f}")
    print(f"  Max:    ${stats['cost']['max']:.2f}")
    print(f"  IQR:    ${stats['cost']['iqr']:.2f} ({stats['cost']['q1']:.2f} - {stats['cost']['q3']:.2f})")
    print(f"  Range:  ${stats['cost']['range']:.2f}")
    print(f"  P10:    ${stats['cost']['p10']:.2f}")
    print(f"  P90:    ${stats['cost']['p90']:.2f}")
    print(f"\nMakespan Statistics:")
    print(f"  Mean:   {stats['makespan']['mean']:.2f} min")
    print(f"  Median: {stats['makespan']['median']:.2f} min")
    print(f"  Std:    {stats['makespan']['std']:.2f} min")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python extract_stats.py <bo-classic-results.json> <bo-enhanced-results.json>")
        print("   or: python extract_stats.py <results.json>")
        sys.exit(1)
    
    # Load results
    results_files = sys.argv[1:]
    stats_dict = {}
    
    for filepath in results_files:
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File not found: {filepath}")
            continue
        
        # Infer algorithm name from filename
        filename = path.stem
        if 'classic' in filename.lower():
            algo_name = 'bo-classic'
        elif 'enhanced' in filename.lower():
            algo_name = 'bo-enhanced'
        else:
            algo_name = filename
        
        print(f"\nProcessing: {filepath}")
        results = load_results(filepath)
        stats = calculate_statistics(results)
        
        if stats:
            stats_dict[algo_name] = stats
            print_summary(algo_name, stats)
        else:
            print(f"No successful evaluations found in {filepath}")
    
    # If we have both algorithms, print comparison
    if len(stats_dict) >= 2:
        print(f"\n{'='*60}")
        print("LATEX TABLE 1: SUMMARY STATISTICS")
        print(f"{'='*60}")
        print("""
\\begin{table}[t]
\\centering
\\caption{Summary statistics for cluster sizing optimization across 30 independent trials}
\\label{tab:cluster_sizing_summary}
\\begin{tabular}{lccccccc}
\\toprule
\\textbf{Algorithm} & \\textbf{Mean} & \\textbf{Median} & \\textbf{Std Dev} & \\textbf{IQR} & \\textbf{Min} & \\textbf{Max} & \\textbf{Range} \\\\
\\midrule""")
        
        for algo_name in sorted(stats_dict.keys()):
            print(format_table_row(algo_name, stats_dict[algo_name]))
        
        print("""\\bottomrule
\\end{tabular}
\\end{table}
""")
        
        # Print quantile table if we have both algorithms
        if 'bo-classic' in stats_dict and 'bo-enhanced' in stats_dict:
            print(f"\n{'='*60}")
            print("LATEX TABLE 2: QUANTILE COMPARISON")
            print(f"{'='*60}")
            print("""
\\begin{table}[t]
\\centering
\\caption{Quantile comparison showing cost distribution differences between algorithms}
\\label{tab:quantile_comparison}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Quantile} & \\textbf{BO-Classic} & \\textbf{BO-Enhanced} & \\textbf{Difference} & \\textbf{Improvement} \\\\
\\midrule""")
            
            for row in format_quantile_table(stats_dict):
                print(row)
            
            print("""\\bottomrule
\\end{tabular}
\\end{table}
""")
            
            # Print improvement summary
            classic = stats_dict['bo-classic']['cost']
            enhanced = stats_dict['bo-enhanced']['cost']
            
            print(f"\n{'='*60}")
            print("IMPROVEMENT SUMMARY")
            print(f"{'='*60}")
            print(f"Mean cost reduction: ${classic['mean'] - enhanced['mean']:.2f} "
                  f"({((classic['mean'] - enhanced['mean']) / classic['mean'] * 100):.1f}%)")
            print(f"Median cost reduction: ${classic['median'] - enhanced['median']:.2f} "
                  f"({((classic['median'] - enhanced['median']) / classic['median'] * 100):.1f}%)")
            print(f"IQR reduction: ${classic['iqr'] - enhanced['iqr']:.2f} "
                  f"({((classic['iqr'] - enhanced['iqr']) / classic['iqr'] * 100):.1f}% tighter distribution)")
            print(f"Std Dev reduction: ${classic['std'] - enhanced['std']:.2f} "
                  f"({((classic['std'] - enhanced['std']) / classic['std'] * 100):.1f}%)")


if __name__ == "__main__":
    main()