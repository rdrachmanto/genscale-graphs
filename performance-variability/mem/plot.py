#!/usr/bin/env python3
"""Hexbin heatmap + CV + Error bands: Input Size vs Peak Memory per tool"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy import stats

# Pastel colors for heatmaps
PASTEL_COLORS = [
    '#5FB0DA', '#E07A84', '#F28C52', '#8FC3A9', '#C38BC3', '#F2C265',
    '#78AEE6', '#C6A27E', '#88C97A', '#E6A8C4', '#7ED6D6', '#E9C57C',
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
df = pd.read_sql_query("SELECT tool_name, input_size, peak_mem FROM memory", conn)
conn.close()

print(f"Loaded {len(df)} records for {df['tool_name'].nunique()} tools")

def create_bins(data, num_bins=15):
    """Create logarithmic bins for input size"""
    data_positive = data[data > 0]
    if len(data_positive) == 0:
        return None
    log_min = np.log10(data_positive.min())
    log_max = np.log10(data_positive.max())
    return np.logspace(log_min, log_max, num_bins)

def plot_cv_heatmap(tool_data, tool_name, color):
    """Coefficient of Variation heatmap"""
    tool_data = tool_data[(tool_data['input_size'] > 0) & (tool_data['peak_mem'] > 0)].copy()
    
    if len(tool_data) < 10:
        return False
    
    bins = create_bins(tool_data['input_size'], num_bins=20)
    if bins is None:
        return False
    
    tool_data['size_bin'] = pd.cut(tool_data['input_size'], bins=bins)
    
    # Calculate CV for each bin
    cv_data = tool_data.groupby('size_bin', observed=True)['peak_mem'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    cv_data['cv'] = (cv_data['std'] / cv_data['mean']) * 100  # CV as percentage
    cv_data = cv_data[cv_data['count'] >= 3]  # Need at least 3 points
    
    if len(cv_data) == 0:
        return False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract bin midpoints for x-axis
    bin_midpoints = [interval.mid for interval in cv_data['size_bin']]
    
    # Create bar plot with CV values
    bars = ax.bar(range(len(cv_data)), cv_data['cv'], 
                  color=color, alpha=0.8, edgecolor='#333333', linewidth=0.8)
    
    # Color bars by CV intensity
    norm = plt.Normalize(vmin=cv_data['cv'].min(), vmax=cv_data['cv'].max())
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
    for bar, cv_val in zip(bars, cv_data['cv']):
        bar.set_facecolor(sm.to_rgba(cv_val))
    
    ax.set_xlabel('Input Size (bytes)')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title(f'{tool_name}\nMemory Variability (CV)')
    
    # Format x-axis labels
    ax.set_xticks(range(len(cv_data)))
    labels = [f'{x:.1e}' for x in bin_midpoints]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, label='CV (%)')
    
    # Add horizontal line at CV=20% (common threshold for high variability)
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(cv_data)-1, 20, ' High variability threshold', 
            va='bottom', ha='right', fontsize=8, color='red')
    
    plt.tight_layout()
    safe_name = tool_name.replace('/', '_').replace(' ', '_')
    plt.savefig(f'cv_{safe_name}.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    return True

def plot_error_bands(tool_data, tool_name, color):
    """Error band plot showing percentiles"""
    tool_data = tool_data[(tool_data['input_size'] > 0) & (tool_data['peak_mem'] > 0)].copy()
    
    if len(tool_data) < 10:
        return False
    
    bins = create_bins(tool_data['input_size'], num_bins=25)
    if bins is None:
        return False
    
    tool_data['size_bin'] = pd.cut(tool_data['input_size'], bins=bins)
    
    # Calculate statistics for each bin
    stats_data = tool_data.groupby('size_bin', observed=True)['peak_mem'].agg([
        ('p10', lambda x: np.percentile(x, 10)),
        ('p25', lambda x: np.percentile(x, 25)),
        ('median', 'median'),
        ('p75', lambda x: np.percentile(x, 75)),
        ('p90', lambda x: np.percentile(x, 90)),
        ('mean', 'mean'),
        ('count', 'count')
    ]).reset_index()
    
    stats_data = stats_data[stats_data['count'] >= 3]
    
    if len(stats_data) == 0:
        return False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract bin midpoints
    bin_midpoints = np.array([interval.mid for interval in stats_data['size_bin']])
    
    if len(bin_midpoints) == 0:
        plt.close()
        return False
    
    # Plot percentile bands
    ax.fill_between(bin_midpoints, stats_data['p10'], stats_data['p90'],
                    alpha=0.2, color=color, label='10th-90th percentile')
    ax.fill_between(bin_midpoints, stats_data['p25'], stats_data['p75'],
                    alpha=0.4, color=color, label='25th-75th percentile (IQR)')
    
    # Plot median and mean lines
    ax.plot(bin_midpoints, stats_data['median'], 
            color='#333333', linewidth=2, label='Median', marker='o', markersize=4)
    ax.plot(bin_midpoints, stats_data['mean'], 
            color=color, linewidth=2, linestyle='--', label='Mean', marker='s', markersize=4)
    
    ax.set_xlabel('Input Size (bytes)')
    ax.set_ylabel('Peak Memory (bytes)')
    ax.set_title(f'{tool_name}\nMemory Usage with Variability Bands')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False, loc='best')
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    safe_name = tool_name.replace('/', '_').replace(' ', '_')
    plt.savefig(f'errorband_{safe_name}.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    return True

def plot_boxplot_with_jitter(tool_data, tool_name, color):
    """Box plot with jittered points showing distribution at each input size bin"""
    tool_data = tool_data[(tool_data['input_size'] > 0) & (tool_data['peak_mem'] > 0)].copy()
    
    if len(tool_data) < 10:
        return False
    
    bins = create_bins(tool_data['input_size'], num_bins=15)
    if bins is None:
        return False
    
    tool_data['size_bin'] = pd.cut(tool_data['input_size'], bins=bins)
    
    # Filter bins with sufficient data
    bin_counts = tool_data.groupby('size_bin', observed=True).size()
    valid_bins = bin_counts[bin_counts >= 3].index
    tool_data = tool_data[tool_data['size_bin'].isin(valid_bins)]
    
    if len(tool_data) == 0:
        return False
    
    # Get bin labels
    bin_labels = sorted(tool_data['size_bin'].unique())
    if len(bin_labels) == 0:
        return False
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for box plot
    data_by_bin = [tool_data[tool_data['size_bin'] == bin_label]['peak_mem'].values 
                   for bin_label in bin_labels]
    
    # Create box plot
    bp = ax.boxplot(data_by_bin, 
                    positions=range(len(bin_labels)),
                    widths=0.5,
                    patch_artist=True,
                    showfliers=False,  # We'll add jittered points instead
                    medianprops=dict(color='#333333', linewidth=2),
                    boxprops=dict(facecolor=color, alpha=0.6, linewidth=1),
                    whiskerprops=dict(color='#333333', linewidth=1),
                    capprops=dict(color='#333333', linewidth=1))
    
    # Add jittered scatter points
    for i, bin_label in enumerate(bin_labels):
        bin_data = tool_data[tool_data['size_bin'] == bin_label]['peak_mem'].values
        
        # Add jitter to x-axis
        jitter_strength = 0.15
        x_jitter = np.random.normal(i, jitter_strength, size=len(bin_data))
        
        ax.scatter(x_jitter, bin_data, 
                  alpha=0.4, s=20, color=color, 
                  edgecolor='#333333', linewidth=0.3,
                  zorder=3)
    
    # Format x-axis labels
    bin_midpoints = [interval.mid for interval in bin_labels]
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels([f'{x:.1e}' for x in bin_midpoints], 
                       rotation=45, ha='right')
    
    ax.set_xlabel('Input Size (bytes)')
    ax.set_ylabel('Peak Memory (bytes)')
    ax.set_title(f'{tool_name}\nMemory Distribution by Input Size')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    safe_name = tool_name.replace('/', '_').replace(' ', '_')
    plt.savefig(f'boxplot_{safe_name}.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    return True

# Create plots for each tool
tools = sorted(df['tool_name'].unique())

hexbin_count = 0
cv_count = 0
errorband_count = 0
quantile_count = 0

for i, tool in enumerate(tools):
    tool_data = df[df['tool_name'] == tool].copy()
    
    # Skip if insufficient data
    if len(tool_data) < 3:
        print(f"Skipping {tool}: insufficient data ({len(tool_data)} points)")
        continue
    
    color = PASTEL_COLORS[i % len(PASTEL_COLORS)]
    
    # 1. Hexbin plot
    tool_data_positive = tool_data[(tool_data['input_size'] > 0) & (tool_data['peak_mem'] > 0)]
    
    if len(tool_data_positive) >= 3:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            hexbin = ax.hexbin(tool_data_positive['input_size'], tool_data_positive['peak_mem'],
                               gridsize=20, cmap='YlOrRd', mincnt=1,
                               xscale='log', yscale='log', 
                               edgecolors='#333333', linewidths=0.3)
            
            ax.set_xlabel('Input Size (bytes)')
            ax.set_ylabel('Peak Memory (bytes)')
            ax.set_title(f'{tool}\nMemory Usage Density')
            
            cb = plt.colorbar(hexbin, ax=ax, label='Observation Count')
            
            plt.tight_layout()
            
            safe_name = tool.replace('/', '_').replace(' ', '_')
            plt.savefig(f'hexbin_{safe_name}.svg', format='svg', bbox_inches='tight')
            plt.close()
            
            hexbin_count += 1
            print(f"[Hexbin] Saved: hexbin_{safe_name}.svg")
        
        except Exception as e:
            print(f"[Hexbin] Error for {tool}: {e}")
            plt.close('all')
    
    # 2. CV Heatmap
    try:
        if plot_cv_heatmap(tool_data, tool, color):
            cv_count += 1
            safe_name = tool.replace('/', '_').replace(' ', '_')
            print(f"[CV] Saved: cv_{safe_name}.svg")
    except Exception as e:
        print(f"[CV] Error for {tool}: {e}")
        plt.close('all')
    
    # 3. Error band plot
    try:
        if plot_error_bands(tool_data, tool, color):
            errorband_count += 1
            safe_name = tool.replace('/', '_').replace(' ', '_')
            print(f"[ErrorBand] Saved: errorband_{safe_name}.svg")
    except Exception as e:
        print(f"[ErrorBand] Error for {tool}: {e}")
        plt.close('all')
    
    # 4. Box plot with jitter
    try:
        if plot_boxplot_with_jitter(tool_data, tool, color):
            quantile_count += 1
            safe_name = tool.replace('/', '_').replace(' ', '_')
            print(f"[BoxPlot] Saved: boxplot_{safe_name}.svg")
    except Exception as e:
        print(f"[BoxPlot] Error for {tool}: {e}")
        plt.close('all')

print(f"\n=== Generation Complete ===")
print(f"Hexbin plots: {hexbin_count}")
print(f"CV plots: {cv_count}")
print(f"Error band plots: {errorband_count}")
print(f"Box plot with jitter: {quantile_count}")

# ============================================================================
# SPECIAL: Create 2x2 grid of selected plots
# ============================================================================

def create_combined_grid():
    """Create a 2x2 grid combining 4 different plot types from different tools"""
    
    # Define the 4 tools and their plot types
    selections = [
        ('fastqc', 'boxplot'),
        ('picard-validate-sam', 'cv'),
        ('picard-markduplicate', 'errorband'),
        ('picard-collect-wgs-metrics', 'hexbin')
    ]
    
    titles = [
        'FastQC - Box Plot with Jitter',
        'Picard Validate SAM - Coefficient of Variation',
        'Picard MarkDuplicate - Error Bands',
        'Picard Collect WGS Metrics - Hexbin Density'
    ]
    
    # Check if all tools exist in dataset
    available_tools = df['tool_name'].unique()
    missing = [tool for tool, _ in selections if tool not in available_tools]
    
    if missing:
        print(f"\nSkipping combined grid: missing tools {missing}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, ((tool_name, plot_type), title) in enumerate(zip(selections, titles)):
        ax = axes[idx]
        tool_data = df[df['tool_name'] == tool_name].copy()
        tool_data = tool_data[(tool_data['input_size'] > 0) & (tool_data['peak_mem'] > 0)]
        
        color = PASTEL_COLORS[idx % len(PASTEL_COLORS)]
        
        try:
            if plot_type == 'hexbin':
                hexbin = ax.hexbin(tool_data['input_size'], tool_data['peak_mem'],
                                   gridsize=20, cmap='YlOrRd', mincnt=1,
                                   xscale='log', yscale='log',
                                   edgecolors='#333333', linewidths=0.3)
                ax.set_xlabel('Input Size (megabytes)')
                ax.set_ylabel('Peak Memory (megabytes)')
                plt.colorbar(hexbin, ax=ax, label='Count')
                
            elif plot_type == 'cv':
                bins = create_bins(tool_data['input_size'], num_bins=20)
                if bins is not None:
                    tool_data['size_bin'] = pd.cut(tool_data['input_size'], bins=bins)
                    cv_data = tool_data.groupby('size_bin', observed=True)['peak_mem'].agg([
                        ('mean', 'mean'),
                        ('std', 'std'),
                        ('count', 'count')
                    ]).reset_index()
                    cv_data['cv'] = (cv_data['std'] / cv_data['mean']) * 100
                    cv_data = cv_data[cv_data['count'] >= 3]
                    
                    bin_midpoints = [interval.mid for interval in cv_data['size_bin']]
                    bars = ax.bar(range(len(cv_data)), cv_data['cv'],
                                  color=color, alpha=0.8, edgecolor='#333333', linewidth=0.8)
                    
                    # Color bars by intensity
                    from matplotlib import cm
                    norm = plt.Normalize(vmin=cv_data['cv'].min(), vmax=cv_data['cv'].max())
                    sm = cm.ScalarMappable(cmap='YlOrRd', norm=norm)
                    for bar, cv_val in zip(bars, cv_data['cv']):
                        bar.set_facecolor(sm.to_rgba(cv_val))
                    
                    ax.set_xlabel('Input Size Bin')
                    ax.set_ylabel('CV (%)')
                    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    
            elif plot_type == 'errorband':
                bins = create_bins(tool_data['input_size'], num_bins=25)
                if bins is not None:
                    tool_data['size_bin'] = pd.cut(tool_data['input_size'], bins=bins)
                    stats_data = tool_data.groupby('size_bin', observed=True)['peak_mem'].agg([
                        ('p10', lambda x: np.percentile(x, 10)),
                        ('p25', lambda x: np.percentile(x, 25)),
                        ('median', 'median'),
                        ('p75', lambda x: np.percentile(x, 75)),
                        ('p90', lambda x: np.percentile(x, 90)),
                        ('mean', 'mean'),
                        ('count', 'count')
                    ]).reset_index()
                    stats_data = stats_data[stats_data['count'] >= 3]
                    
                    bin_midpoints = np.array([interval.mid for interval in stats_data['size_bin']])
                    
                    ax.fill_between(bin_midpoints, stats_data['p10'], stats_data['p90'],
                                    alpha=0.2, color=color)
                    ax.fill_between(bin_midpoints, stats_data['p25'], stats_data['p75'],
                                    alpha=0.4, color=color)
                    ax.plot(bin_midpoints, stats_data['median'],
                            color='#333333', linewidth=2, marker='o', markersize=4)
                    ax.plot(bin_midpoints, stats_data['mean'],
                            color=color, linewidth=2, linestyle='--', marker='s', markersize=4)
                    
                    ax.set_xlabel('Input Size (megabytes)')
                    ax.set_ylabel('Peak Memory (megabytes)')
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
                    
            elif plot_type == 'boxplot':
                bins = create_bins(tool_data['input_size'], num_bins=15)
                if bins is not None:
                    tool_data['size_bin'] = pd.cut(tool_data['input_size'], bins=bins)
                    bin_counts = tool_data.groupby('size_bin', observed=True).size()
                    valid_bins = bin_counts[bin_counts >= 3].index
                    tool_data = tool_data[tool_data['size_bin'].isin(valid_bins)]
                    
                    bin_labels = sorted(tool_data['size_bin'].unique())
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
                    
                    # Add jitter
                    for i, bin_label in enumerate(bin_labels):
                        bin_data = tool_data[tool_data['size_bin'] == bin_label]['peak_mem'].values
                        x_jitter = np.random.normal(i, 0.15, size=len(bin_data))
                        ax.scatter(x_jitter, bin_data,
                                  alpha=0.4, s=20, color=color,
                                  edgecolor='#333333', linewidth=0.3, zorder=3)
                    
                    ax.set_xlabel('Input Size Bin')
                    ax.set_ylabel('Peak Memory (megabytes)')
                    ax.set_yscale('log')
                    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, axis='y')
            
            ax.set_title(title, fontsize=13, pad=15, weight='normal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=13, pad=15)
    
    plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
    plt.savefig('combined_2x2_grid.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.savefig('combined_2x2_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n[Combined Grid] Saved: combined_2x2_grid.svg and combined_2x2_grid.png")

create_combined_grid()