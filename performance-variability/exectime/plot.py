#!/usr/bin/env python3
"""CPU comparison: Execution time vs Input size across different CPU models"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Pastel colors for CPUs (expanded palette)
PASTEL_COLORS = [
    '#E07A84', '#5FB0DA', '#8FC3A9', '#F28C52', '#C38BC3', '#F2C265',
    '#78AEE6', '#C6A27E', '#88C97A', '#E6A8C4', '#7ED6D6', '#E9C57C',
    '#A8DADC', '#F1FAEE', '#E63946', '#457B9D', '#1D3557', '#F4A261',
]

plt.rcParams.update({
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Load data
db_path = sys.argv[1] if len(sys.argv) > 1 else 'metrics.db'
if not os.path.exists(db_path):
    print(f"Error: Database file '{db_path}' not found")
    print(f"Usage: python3 {sys.argv[0]} <path_to_metrics.db>")
    sys.exit(1)

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("""
    SELECT tool_name, input_size, execution_time_seconds, cpu_model 
    FROM execution_times_with_machines
""", conn)
conn.close()

print(f"Loaded {len(df)} records")

# Filter to only positive values
df = df[(df['input_size'] > 0) & (df['execution_time_seconds'] > 0)].copy()

# Clean up CPU model names (remove extra whitespace)
df['cpu_model'] = df['cpu_model'].str.strip()

print(f"After filtering: {len(df)} records")
print(f"Unique CPU models: {df['cpu_model'].nunique()}")
print(f"CPU models found:")
for cpu in sorted(df['cpu_model'].unique()):
    count = len(df[df['cpu_model'] == cpu])
    print(f"  - {cpu}: {count} records")
print(f"Tools: {df['tool_name'].nunique()}")

def create_bins(data, num_bins=20):
    """Create logarithmic bins for input size"""
    data_positive = data[data > 0]
    if len(data_positive) == 0:
        return None
    log_min = np.log10(data_positive.min())
    log_max = np.log10(data_positive.max())
    return np.logspace(log_min, log_max, num_bins)

def plot_cpu_comparison(tool_data, tool_name, ax=None, return_handles=False):
    """Compare execution times across different CPU models with error bands"""
    
    # Get unique CPU models for this tool
    cpu_models = tool_data['cpu_model'].unique()
    
    # Only plot if we have at least 2 CPU models with sufficient data
    valid_cpus = []
    for cpu in cpu_models:
        cpu_data = tool_data[tool_data['cpu_model'] == cpu]
        if len(cpu_data) >= 5:  # Minimum data points required
            valid_cpus.append(cpu)
    
    if len(valid_cpus) < 2:
        print(f"Skipping {tool_name}: need at least 2 CPUs with sufficient data (found {len(valid_cpus)})")
        return False
    
    # Create figure if ax not provided
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    # Assign colors to CPU models
    color_map = {cpu: PASTEL_COLORS[i % len(PASTEL_COLORS)] 
                 for i, cpu in enumerate(sorted(valid_cpus))}
    
    legend_handles = []
    legend_labels = []
    
    for cpu_model in sorted(valid_cpus):
        cpu_data = tool_data[tool_data['cpu_model'] == cpu_model].copy()
        
        # Create bins
        bins = create_bins(cpu_data['input_size'], num_bins=15)
        if bins is None:
            continue
        
        cpu_data['size_bin'] = pd.cut(cpu_data['input_size'], bins=bins)
        
        # Calculate statistics for each bin
        stats_data = cpu_data.groupby('size_bin', observed=True)['execution_time_seconds'].agg([
            ('p10', lambda x: np.percentile(x, 10)),
            ('p25', lambda x: np.percentile(x, 25)),
            ('median', 'median'),
            ('p75', lambda x: np.percentile(x, 75)),
            ('p90', lambda x: np.percentile(x, 90)),
            ('mean', 'mean'),
            ('count', 'count')
        ]).reset_index()
        
        stats_data = stats_data[stats_data['count'] >= 2]  # Need at least 2 points
        
        if len(stats_data) == 0:
            continue
        
        # Extract bin midpoints
        bin_midpoints = np.array([interval.mid for interval in stats_data['size_bin']])
        
        if len(bin_midpoints) == 0:
            continue
        
        color = color_map[cpu_model]
        
        # Shorten CPU name for legend (keep key info)
        cpu_short = cpu_model \
            .replace('(R)', '') \
            .replace('(TM)', '') \
            .replace('  ', ' ') \
            .replace('24-Core Processor', ' ') \
            .replace('Intel', '') \
            .replace('AMD', '') \
            .replace('Gold', '') \
            .strip()
        
        # Plot percentile bands
        band = ax.fill_between(bin_midpoints, stats_data['p10'], stats_data['p90'],
                        alpha=0.12, color=color, label=f'{cpu_short} (10-90%ile)')
        ax.fill_between(bin_midpoints, stats_data['p25'], stats_data['p75'],
                        alpha=0.20, color=color)
        
        # Plot median line
        line, = ax.plot(bin_midpoints, stats_data['median'],
                color=color, linewidth=2.5, label=f'{cpu_short} (median)',
                marker='o', markersize=5, markeredgecolor='#333333', markeredgewidth=0.5)
        
        # Plot mean as dashed
        ax.plot(bin_midpoints, stats_data['mean'],
                color=color, linewidth=1.5, linestyle='--', alpha=0.7,
                marker='s', markersize=4)
        
        # Collect handles for legend
        legend_handles.append((line, band))
        legend_labels.append(cpu_short)
    
    ax.set_xlabel('Input Size (MB)', fontsize=16)
    ax.set_ylabel('Execution Time (Seconds)', fontsize=16)
    # ax.set_title(f'{tool_name}\nCPU Performance Comparison', fontsize=18)
    ax.set_xscale('linear')  # Both used to be log
    ax.set_yscale('linear')
    ax.tick_params(axis='both', labelsize=16, length=0)
    
    # Adjust legend to handle potentially many CPUs
    if standalone:
        # legend = ax.legend(frameon=False, loc='best', fontsize=12, ncol=1 if len(valid_cpus) <= 3 else 2)
        legend = ax.legend(frameon=False, loc='best', fontsize=13, ncol=1 if len(valid_cpus) <= 3 else 2)
    
    ax.grid(True, alpha=0.5, linestyle=':', linewidth=0.5)
    
    if standalone:
        plt.tight_layout()
        safe_name = tool_name.replace('/', '_').replace(' ', '_')
        plt.savefig(f'cpu_compare_{safe_name}.svg', format='svg', bbox_inches='tight')
        plt.savefig(f'cpu_compare_{safe_name}.pdf', format='pdf', bbox_inches='tight')
        plt.close()
    
    if return_handles:
        return True, legend_handles, legend_labels
    return True

def plot_combined_comparison(df, tool1, tool2, output_name):
    """Create side-by-side comparison plots for two tools"""
    
    tool1_data = df[df['tool_name'] == tool1].copy()
    tool2_data = df[df['tool_name'] == tool2].copy()
    
    if len(tool1_data) == 0 or len(tool2_data) == 0:
        print(f"Skipping combined plot: missing data for {tool1} or {tool2}")
        return False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    
    # Plot first tool
    result1 = plot_cpu_comparison(tool1_data, tool1, ax=ax1, return_handles=True)
    if not result1:
        plt.close()
        return False
    
    success1, handles1, labels1 = result1
    
    # Plot second tool
    result2 = plot_cpu_comparison(tool2_data, tool2, ax=ax2, return_handles=True)
    if not result2:
        plt.close()
        return False
    
    success2, handles2, labels2 = result2
    
    # Add legends to each subplot with the same style as standalone plots
    num_cpus1 = len(labels1)
    num_cpus2 = len(labels2)

    ax1.set_title("(a) PCWM", fontsize=14)
    ax2.set_title("(b) BWA-MEM", fontsize=14)

    # ax1.legend().remove()
    # ax2.legend().remove()
    
    ax1.legend(frameon=False, loc='best', fontsize=10, ncol=1 if num_cpus1 <= 3 else 2)
    # ax2.legend(frameon=False, loc='best', fontsize=10, ncol=1 if num_cpus2 <= 3 else 2)

    plt.tight_layout()
    
    # plt.savefig(output_name, format='svg', bbox_inches='tight')
    plt.savefig(output_name, format='pdf', bbox_inches='tight')
    plt.close()
    
    return True

# Generate plots for each tool
tools = sorted(df['tool_name'].unique())
success_count = 0

for tool in tools:
    tool_data = df[df['tool_name'] == tool].copy()
    
    try:
        if plot_cpu_comparison(tool_data, tool):
            success_count += 1
            safe_name = tool.replace('/', '_').replace(' ', '_')
            print(f"[Success] Saved: cpu_compare_{safe_name}.svg")
    except Exception as e:
        print(f"[Error] {tool}: {e}")
        plt.close('all')

print(f"\n=== Generation Complete ===")
# print(f"Successfully generated {success_count} CPU comparison plots")

# Create combined plot for picard-collect-wgs-metrics and burrows-wheeler-aligner

# print(f"\n=== Creating Combined Plot ===")
# if plot_combined_comparison(df, 'picard-collect-wgs-metrics', 'burrows-wheeler-aligner', 
#                             'cpu_compare_combined_picard_bwa.pdf'):
#     print("[Success] Saved: cpu_compare_combined_picard_bwa.svg")
# else:
#     print("[Error] Could not create combined plot")