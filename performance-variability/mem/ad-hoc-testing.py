#!/usr/bin/env python3
"""
Analyze memory allocation strategies for GenScale motivation section.
Compares naive baselines against actual memory requirements.

Usage:
    python analyze_allocation_strategies.py path/to/your/database.db
"""

import sys
import sqlite3
import pandas as pd
import numpy as np

def load_data(db_path):
    """Load memory data from SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT tool_name, input_size, peak_mem, exec_time, time_to_peak
        FROM memory
        WHERE peak_mem > 0 AND tool_name != 'unknown'
    """, conn)
    conn.close()
    print(f"Loaded {len(df)} executions across {df['tool_name'].nunique()} tools\n")
    return df


def compute_tool_stats(df):
    """Compute per-tool memory statistics."""
    stats = df.groupby('tool_name')['peak_mem'].agg([
        'count', 'min', 'max', 'mean', 'std'
    ])
    stats['p95'] = df.groupby('tool_name')['peak_mem'].quantile(0.95)
    stats['p50'] = df.groupby('tool_name')['peak_mem'].quantile(0.50)
    
    print("=== Per-Tool Memory Statistics (MB) ===")
    print(stats.round(1).to_string())
    print()
    return stats


def evaluate_strategy(df, tool_stats, strategy_name, allocation_fn):
    """
    Evaluate a memory allocation strategy.
    
    Returns: (total_allocated_mb, total_waste_mb, waste_pct, num_failures, failure_pct)
    """
    total_allocated = 0
    total_actual = 0
    failures = 0
    
    for _, row in df.iterrows():
        tool = row['tool_name']
        actual = row['peak_mem']
        
        # Skip if tool not in stats (shouldn't happen)
        if tool not in tool_stats.index:
            continue
            
        allocated = allocation_fn(tool, tool_stats)
        
        total_allocated += allocated
        total_actual += actual
        
        if allocated < actual:
            failures += 1
    
    waste = total_allocated - total_actual
    waste_pct = (waste / total_allocated) * 100 if total_allocated > 0 else 0
    failure_pct = (failures / len(df)) * 100
    
    return {
        'Strategy': strategy_name,
        'Total Allocated (GB)': total_allocated / 1024,
        'Total Actual (GB)': total_actual / 1024,
        'Waste (GB)': waste / 1024,
        'Waste %': waste_pct,
        'OOM Failures': failures,
        'Failure Rate %': failure_pct
    }


def main(db_path):
    # Load data
    df = load_data(db_path)
    
    # Compute statistics
    tool_stats = compute_tool_stats(df)
    
    # Define allocation strategies
    strategies = {
        # Static allocations (common ad-hoc practice)
        'Static-8GB': lambda t, s: 8 * 1024,
        'Static-16GB': lambda t, s: 16 * 1024,
        'Static-32GB': lambda t, s: 32 * 1024,
        'Static-64GB': lambda t, s: 64 * 1024,
        
        # Per-tool strategies
        'Tool-Max': lambda t, s: s.loc[t, 'max'],
        'Tool-Mean': lambda t, s: s.loc[t, 'mean'],
        '2×-Mean': lambda t, s: 2 * s.loc[t, 'mean'],
        '1.5×-Mean': lambda t, s: 1.5 * s.loc[t, 'mean'],
        'P95': lambda t, s: s.loc[t, 'p95'],
        'P50 (Median)': lambda t, s: s.loc[t, 'p50'],
    }
    
    # Evaluate each strategy
    results = []
    for name, fn in strategies.items():
        result = evaluate_strategy(df, tool_stats, name, fn)
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by waste percentage
    results_df = results_df.sort_values('Waste %', ascending=True)
    
    print("\n=== Allocation Strategy Comparison ===")
    print(results_df.round(2).to_string(index=False))
    
    # Print LaTeX table
    print("\n\n=== LaTeX Table ===")
    print(generate_latex_table(results_df))
    
    # Print key insights
    print("\n=== Key Insights for Paper ===")
    print_insights(df, tool_stats, results_df)
    
    return results_df


def generate_latex_table(results_df):
    """Generate LaTeX table for paper."""
    
    # Select most interesting strategies for paper (pick 4-5)
    interesting = ['Static-32GB', 'Tool-Max', '2×-Mean', 'P95', 'Tool-Mean']
    subset = results_df[results_df['Strategy'].isin(interesting)].copy()
    subset = subset.sort_values('Waste %', ascending=False)
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Comparison of ad-hoc memory allocation strategies against actual requirements across %d workflow executions. Static allocation ignores tool-specific patterns; Tool-Max eliminates failures but wastes resources; percentile-based approaches balance waste and reliability.}
\label{tab:adhoc}
\vspace{-2.5mm}
\begin{tabular}{l|rrr}
\hline
\textbf{Strategy} & \textbf{Waste (GB)} & \textbf{Waste \%%} & \textbf{Failures} \\
\hline
""" % len(results_df)
    
    for _, row in subset.iterrows():
        latex += f"{row['Strategy']} & {row['Waste (GB)']:.1f} & {row['Waste %']:.1f}\\% & {int(row['OOM Failures'])} \\\\\n"
    
    latex += r"""\hline
\end{tabular}
\vspace{-1em}
\end{table}
"""
    return latex


def print_insights(df, tool_stats, results_df):
    """Print key statistics for paper narrative."""
    
    # Memory range across tools
    print(f"\n1. Memory requirements vary widely:")
    for tool in tool_stats.index:
        min_mem = tool_stats.loc[tool, 'min']
        max_mem = tool_stats.loc[tool, 'max']
        if max_mem > 1024:  # Only show tools with >1GB variation
            print(f"   - {tool}: {min_mem/1024:.1f}GB to {max_mem/1024:.1f}GB")
    
    # Static-32GB analysis
    static_row = results_df[results_df['Strategy'] == 'Static-32GB'].iloc[0]
    print(f"\n2. Static-32GB allocation:")
    print(f"   - Wastes {static_row['Waste (GB)']:.1f}GB ({static_row['Waste %']:.1f}%)")
    print(f"   - Causes {int(static_row['OOM Failures'])} OOM failures ({static_row['Failure Rate %']:.1f}%)")
    
    # Tool-Max analysis  
    max_row = results_df[results_df['Strategy'] == 'Tool-Max'].iloc[0]
    print(f"\n3. Tool-Max (conservative) allocation:")
    print(f"   - Wastes {max_row['Waste (GB)']:.1f}GB ({max_row['Waste %']:.1f}%)")
    print(f"   - Zero failures but significant over-provisioning")
    
    # Best tradeoff
    # Find strategy with <5% failures and lowest waste
    viable = results_df[results_df['Failure Rate %'] < 5]
    if len(viable) > 0:
        best = viable.loc[viable['Waste %'].idxmin()]
        print(f"\n4. Best tradeoff ({best['Strategy']}):")
        print(f"   - Wastes {best['Waste (GB)']:.1f}GB ({best['Waste %']:.1f}%)")
        print(f"   - {int(best['OOM Failures'])} failures ({best['Failure Rate %']:.1f}%)")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python analyze_allocation_strategies.py <database.db>")
        sys.exit(1)
    
    main(sys.argv[1])