#!/usr/bin/env python3
"""Violin plot with jitter for specified tools"""

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
    
    color = PASTEL_COLORS[i + 1 % len(PASTEL_COLORS)]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    data_by_bin = [tool_data[tool_data['size_bin'] == bin_label]['peak_mem'].values 
                   for bin_label in bin_labels]
    
    parts = ax.violinplot(data_by_bin,
                         positions=range(len(bin_labels)),
                         widths=0.7,
                         showmeans=False,
                         showmedians=True,
                         showextrema=True)
    
    # Customize violin appearance
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor('#333333')
        pc.set_linewidth(1)
    
    # Customize median, min, max lines
    parts['cmedians'].set_edgecolor('#333333')
    parts['cmedians'].set_linewidth(2)
    parts['cmins'].set_edgecolor('#333333')
    parts['cmins'].set_linewidth(1)
    parts['cmaxes'].set_edgecolor('#333333')
    parts['cmaxes'].set_linewidth(1)
    parts['cbars'].set_edgecolor('#333333')
    parts['cbars'].set_linewidth(1)
    
    # Add jittered scatter points
    for j, bin_label in enumerate(bin_labels):
        bin_data = tool_data[tool_data['size_bin'] == bin_label]['peak_mem'].values
        x_jitter = np.random.normal(j, 0.08, size=len(bin_data))
        ax.scatter(x_jitter, bin_data, 
                  alpha=0.4, s=15, color=color, 
                  edgecolor='#333333', linewidth=0.3,
                  zorder=3)
    
    bin_midpoints_mb = [interval.mid for interval in bin_labels]
    bin_midpoints_gb = [x / 1000 for x in bin_midpoints_mb]
    
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(
        [f'{x:.2f}' if x < 10 else f'{x:.0f}' for x in bin_midpoints_gb],
        rotation=0,
        # ha='right'
    )
    
    ax.set_xlabel('Input Size (GB)', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    # ax.set_title(f'{tool} Memory Distribution by Input Size', fontsize=18)
    ax.set_yscale('linear')
    ax.grid(True, alpha=0.6)

    ax.tick_params(axis='both', labelsize=13, length=0)
    
    plt.tight_layout()
    safe_name = tool.replace('/', '_').replace(' ', '_')
    plt.savefig(f'violinplot_{safe_name}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'violinplot_{safe_name}.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: violinplot_{safe_name}.pdf")
    print(f"Saved: violinplot_{safe_name}.svg")

print("Done.")