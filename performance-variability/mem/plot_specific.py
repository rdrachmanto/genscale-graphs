#!/usr/bin/env python3
"""Box plot with jitter for specified tools"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

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

PASTEL_COLORS = [
    '#5FB0DA', '#E07A84', '#F28C52', '#8FC3A9', '#C38BC3', '#F2C265',
    '#78AEE6', '#C6A27E', '#88C97A', '#E6A8C4', '#7ED6D6', '#E9C57C',
]

if len(sys.argv) < 3:
    print(f"Usage: python3 {sys.argv[0]} <path_to_metrics.db> <tool1> <tool2> ...")
    sys.exit(1)

db_path = sys.argv[1]
tools_to_plot = sys.argv[2:]

if not os.path.exists(db_path):
    print(f"Error: Database file '{db_path}' not found")
    sys.exit(1)

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT tool_name, input_size, peak_mem FROM memory", conn)
conn.close()

def create_bins(data, num_bins=15):
    data_positive = data[data > 0]
    if len(data_positive) == 0:
        return None
    log_min = np.log10(data_positive.min())
    log_max = np.log10(data_positive.max())
    return np.logspace(log_min, log_max, num_bins)

for i, tool in enumerate(tools_to_plot):
    if tool not in df['tool_name'].values:
        print(f"Warning: Tool '{tool}' not found in database")
        continue
    
    tool_data = df[df['tool_name'] == tool].copy()
    tool_data = tool_data[(tool_data['input_size'] > 0) & (tool_data['peak_mem'] > 0)]
    
    if len(tool_data) < 3:
        print(f"Skipping {tool}: insufficient data ({len(tool_data)} points)")
        continue
    
    bins = create_bins(tool_data['input_size'], num_bins=15)
    if bins is None:
        continue
    
    tool_data['size_bin'] = pd.cut(tool_data['input_size'], bins=bins)
    bin_counts = tool_data.groupby('size_bin', observed=True).size()
    valid_bins = bin_counts[bin_counts >= 3].index
    tool_data = tool_data[tool_data['size_bin'].isin(valid_bins)]
    
    if len(tool_data) == 0:
        continue
    
    bin_labels = sorted(tool_data['size_bin'].unique())
    if len(bin_labels) == 0:
        continue
    
    color = PASTEL_COLORS[i % len(PASTEL_COLORS)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data_by_bin = [tool_data[tool_data['size_bin'] == bin_label]['peak_mem'].values 
                   for bin_label in bin_labels]
    
    bp = ax.boxplot(data_by_bin, 
                    positions=range(len(bin_labels)),
                    widths=0.5,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='#333333', linewidth=2),
                    boxprops=dict(facecolor=color, alpha=0.6, linewidth=1),
                    whiskerprops=dict(color='#333333', linewidth=1),
                    capprops=dict(color='#333333', linewidth=1))
    
    for j, bin_label in enumerate(bin_labels):
        bin_data = tool_data[tool_data['size_bin'] == bin_label]['peak_mem'].values
        x_jitter = np.random.normal(j, 0.15, size=len(bin_data))
        ax.scatter(x_jitter, bin_data, 
                  alpha=0.4, s=20, color=color, 
                  edgecolor='#333333', linewidth=0.3,
                  zorder=3)
    
    bin_midpoints = [interval.mid for interval in bin_labels]
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels([f'{x:.1e}' for x in bin_midpoints], 
                       rotation=45, ha='right')
    
    ax.set_xlabel('Input Size (megabytes)')
    ax.set_ylabel('Peak Memory (megabytes)')
    ax.set_title(f'{tool}\nMemory Distribution by Input Size')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    safe_name = tool.replace('/', '_').replace(' ', '_')
    plt.savefig(f'boxplot_{safe_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: boxplot_{safe_name}.pdf")

print("Done.")