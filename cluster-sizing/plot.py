#!/usr/bin/env python3
"""
Plot optimization results with multiple visualization styles.
Generate different plot types to compare algorithms.
Each subplot is saved as a separate file.
Bayesian Optimization Cluster Sizing Evaluation
Publication-quality styling matching Lotaru plotter
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import wesanderson

# ============================================================================
# GLOBAL STYLE CONFIGURATION FOR RESEARCH PAPER
# ============================================================================

# Define sophisticated pastel color palette (matching Lotaru plotter)
PASTEL_COLORS = [
    "#E06672",  # Strong Rose
    "#48A7D6",  # Bright Sky Blue
    '#C38BC3',  # Rich Lavender
    '#F28C52',  # Vibrant Coral
    '#8FC3A9',  # Fresh Sage
    '#F2C265',  # Warm Sand
    '#78AEE6',  # Clear Periwinkle
    '#C6A27E',  # Deep Taupe
    '#88C97A',  # Lively Mint
    '#E6A8C4',  # Bright Mauve
    '#7ED6D6',  # Crisp Aqua
    '#E9C57C',  # Sunny Wheat
]

# Set publication-quality matplotlib parameters
plt.rcParams.update({
    # Figure
    'figure.facecolor': 'white',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    
    # Font settings - Times-like for academic feel
    # 'font.family': 'serif',
    # 'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    
    # Axes styling - minimal and clean
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#000000',
    'axes.labelcolor': '#000000',
    'axes.titleweight': 'normal',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    
    # Ticks
    'xtick.color': '#000000',
    'ytick.color': '#000000',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    
    # Legend
    'legend.frameon': False,
    'legend.loc': 'best',
})


def load_optimization_results(directory: Path) -> Dict[str, List[Dict]]:
    """Load all optimization results from a directory, recursively searching subdirectories."""
    results = {}
    
    # Recursively find all directories that contain optimization_results_*.json files
    def find_algo_dirs(base_path: Path):
        algo_dirs = {}
        
        for item in base_path.rglob("*"):
            if item.is_file() and item.name.startswith("optimization_results_") and item.name.endswith(".json"):
                # The parent directory is the algorithm directory
                algo_dir = item.parent
                algo_name = algo_dir.name
                
                if algo_name not in algo_dirs:
                    algo_dirs[algo_name] = []
                algo_dirs[algo_name].append(item)
        
        return algo_dirs
    
    algo_dirs = find_algo_dirs(directory)
    
    for algo_name, result_files in algo_dirs.items():
        algo_results = []
        
        for result_file in sorted(result_files):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    algo_results.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load {result_file}: {e}")
                continue
        
        if algo_results:
            results[algo_name] = algo_results
    
    return results


def extract_metrics(results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Extract makespan and total_cost from optimization results."""
    data = []
    
    for algo_name, algo_results in results.items():
        for run_idx, result in enumerate(algo_results, 1):
            try:
                best_config = result['optimization_results']['best_configuration']
                makespan = best_config['makespan_minutes']
                total_cost = best_config['total_cost']
                
                data.append({
                    'algorithm': algo_name,
                    'run': run_idx,
                    'makespan_minutes': makespan,
                    'total_cost': total_cost
                })
            except KeyError as e:
                print(f"Warning: Missing key in {algo_name} run {run_idx}: {e}")
                continue
    
    return pd.DataFrame(data)


def plot_style_1_scatter(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 1: Simple scatter plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    algorithms = df['algorithm'].unique()
    colors = PASTEL_COLORS[:len(algorithms)]
    color_map = dict(zip(algorithms, colors))
    
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo]
        ax.scatter(algo_data['makespan_minutes'], 
                  algo_data['total_cost'],
                  label=algo,
                  s=100,
                  alpha=0.7,
                  color=color_map[algo],
                  edgecolors='#333333',
                  linewidth=0.5)
    
    ax.set_xlabel('Makespan (minutes)', fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontweight='bold')
    ax.set_title(f'Makespan vs Total Cost - {objective_name.title()} Objective', 
                fontweight='normal', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_style_2_bar_comparison(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 2: Bar charts with error bars - saved separately"""
    algorithms = df['algorithm'].unique()
    colors = PASTEL_COLORS[:len(algorithms)]
    
    # Compute means and std
    makespan_stats = df.groupby('algorithm')['makespan_minutes'].agg(['mean', 'std']).reindex(algorithms)
    cost_stats = df.groupby('algorithm')['total_cost'].agg(['mean', 'std']).reindex(algorithms)
    
    # Makespan bar chart
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(algorithms))
    ax1.bar(x_pos, makespan_stats['mean'], yerr=makespan_stats['std'],
            color=colors, alpha=0.8, capsize=5, edgecolor='#333333', linewidth=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(algorithms, rotation=15, ha='right')
    ax1.set_ylabel('Makespan (minutes)', fontweight='bold')
    ax1.set_title(f'Makespan Comparison - {objective_name.title()} Objective', fontweight='normal', pad=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(makespan_stats['mean'], makespan_stats['std'])):
        ax1.text(i, mean + std, f'{mean:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    makespan_path = output_path.parent / f"{output_path.stem}_makespan.png"
    plt.savefig(makespan_path, dpi=300, bbox_inches='tight')
    plt.savefig(makespan_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {makespan_path}")
    plt.close()
    
    # Cost bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(x_pos, cost_stats['mean'], yerr=cost_stats['std'],
            color=colors, alpha=0.8, capsize=5, edgecolor='#333333', linewidth=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(algorithms, rotation=15, ha='right')
    ax2.set_ylabel('Total Cost ($)', fontweight='bold')
    ax2.set_title(f'Cost Comparison - {objective_name.title()} Objective', fontweight='normal', pad=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(cost_stats['mean'], cost_stats['std'])):
        ax2.text(i, mean + std, f'${mean:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    cost_path = output_path.parent / f"{output_path.stem}_cost.png"
    plt.savefig(cost_path, dpi=300, bbox_inches='tight')
    plt.savefig(cost_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {cost_path}")
    plt.close()


def plot_style_3_violin(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 3: Violin plots showing distribution - saved separately"""
    algorithms = df['algorithm'].unique()
    algo_str = ["BO-Classic", "BnB+BO"]
    colors = PASTEL_COLORS[:len(algorithms)]
    
    # Makespan violin
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=df, x='algorithm', y='makespan_minutes', ax=ax1, 
                   palette=colors, inner='quartile', order=algorithms)
    ax1.set_xlabel('Algorithm', fontweight='bold')
    ax1.set_ylabel('Makespan (minutes)', fontweight='bold')
    ax1.set_title(f'Makespan Distribution\n{objective_name.title()} Objective', fontweight='normal', pad=15, fontsize=18)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout()
    makespan_path = output_path.parent / f"{output_path.stem}_makespan.png"
    plt.savefig(makespan_path, dpi=300, bbox_inches='tight')
    plt.savefig(makespan_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {makespan_path}")
    plt.close()
    
    # Cost violin
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    sns.violinplot(
        data=df,
        x='algorithm',
        y='total_cost',
        ax=ax2,
        palette=wesanderson.film_palette('The Royal Tenenbaums', 0),
        inner='quartile',
        order=algorithms
    )
    
    ax2.set_xticklabels(algo_str)
    
    ax2.set_xlabel(None)
    ax2.set_ylabel('Total Cost ($)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', labelsize=11, length=0)
    
    plt.tight_layout()
    cost_path = output_path.parent / f"{output_path.stem}_cost.png"
    plt.savefig(cost_path, dpi=300, bbox_inches='tight')
    plt.savefig(cost_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    plt.savefig(cost_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {cost_path}")
    plt.close()


def plot_style_4_pareto_front(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 4: Scatter with Pareto front highlighted"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    algorithms = df['algorithm'].unique()
    colors = PASTEL_COLORS[:len(algorithms)]
    color_map = dict(zip(algorithms, colors))
    
    # Plot points
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo]
        ax.scatter(algo_data['makespan_minutes'], 
                  algo_data['total_cost'],
                  label=algo,
                  s=120,
                  alpha=0.7,
                  color=color_map[algo],
                  edgecolors='#333333',
                  linewidth=0.8,
                  zorder=3)
    
    # Find and plot Pareto front
    points = df[['makespan_minutes', 'total_cost']].values
    pareto_indices = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i != j:
                # Check if other dominates point (lower makespan AND lower cost)
                if other[0] <= point[0] and other[1] <= point[1] and (other[0] < point[0] or other[1] < point[1]):
                    dominated = True
                    break
        if not dominated:
            pareto_indices.append(i)
    
    if pareto_indices:
        pareto_points = points[pareto_indices]
        # Sort by makespan for line plot
        sorted_indices = np.argsort(pareto_points[:, 0])
        pareto_sorted = pareto_points[sorted_indices]
        
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 
               'r--', linewidth=2, alpha=0.6, label='Pareto Front', zorder=2)
        ax.scatter(pareto_sorted[:, 0], pareto_sorted[:, 1],
                  s=200, facecolors='none', edgecolors='red', 
                  linewidth=2.5, zorder=4, marker='o')
    
    ax.set_xlabel('Makespan (minutes)', fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontweight='bold')
    ax.set_title(f'Pareto Front Analysis - {objective_name.title()} Objective', 
                fontweight='normal', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_style_5_strip_with_means(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 5: Strip plot with mean lines - saved separately"""
    algorithms = df['algorithm'].unique()
    colors = PASTEL_COLORS[:len(algorithms)]
    
    # Makespan strip plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.stripplot(data=df, x='algorithm', y='makespan_minutes', ax=ax1,
                  palette=colors, size=8, alpha=0.7, edgecolor='#333333', linewidth=0.5,
                  order=algorithms)
    
    # Add mean lines
    for i, algo in enumerate(algorithms):
        mean_val = df[df['algorithm'] == algo]['makespan_minutes'].mean()
        ax1.hlines(mean_val, i-0.3, i+0.3, colors='red', linewidth=3, alpha=0.8, zorder=10)
    
    ax1.set_xlabel('Algorithm', fontweight='bold')
    ax1.set_ylabel('Makespan (minutes)', fontweight='bold')
    ax1.set_title(f'Makespan (red line = mean) - {objective_name.title()} Objective', fontweight='normal', pad=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout()
    makespan_path = output_path.parent / f"{output_path.stem}_makespan.png"
    plt.savefig(makespan_path, dpi=300, bbox_inches='tight')
    plt.savefig(makespan_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {makespan_path}")
    plt.close()
    
    # Cost strip plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.stripplot(data=df, x='algorithm', y='total_cost', ax=ax2,
                  palette=colors, size=8, alpha=0.7, edgecolor='#333333', linewidth=0.5,
                  order=algorithms)
    
    # Add mean lines
    for i, algo in enumerate(algorithms):
        mean_val = df[df['algorithm'] == algo]['total_cost'].mean()
        ax2.hlines(mean_val, i-0.3, i+0.3, colors='red', linewidth=3, alpha=0.8, zorder=10)
    
    ax2.set_xlabel('Algorithm', fontweight='bold')
    ax2.set_ylabel('Total Cost ($)', fontweight='bold')
    ax2.set_title(f'Cost (red line = mean) - {objective_name.title()} Objective', fontweight='normal', pad=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    cost_path = output_path.parent / f"{output_path.stem}_cost.png"
    plt.savefig(cost_path, dpi=300, bbox_inches='tight')
    plt.savefig(cost_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {cost_path}")
    plt.close()


def plot_style_6_heatmap_table(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 6: Table-style heatmap of mean values"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create summary table
    algorithms = df['algorithm'].unique()
    metrics = ['makespan_minutes', 'total_cost']
    metric_names = ['Makespan (min)', 'Total Cost ($)']
    
    summary_data = []
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo]
        row = [
            algo_data['makespan_minutes'].mean(),
            algo_data['total_cost'].mean()
        ]
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data, columns=metric_names, index=algorithms)
    
    # Normalize for coloring
    summary_normalized = summary_df.copy()
    for col in summary_df.columns:
        min_val = summary_df[col].min()
        max_val = summary_df[col].max()
        if max_val > min_val:
            summary_normalized[col] = (summary_df[col] - min_val) / (max_val - min_val)
    
    # Create heatmap
    sns.heatmap(summary_normalized.T, annot=summary_df.T, fmt='.2f', 
                cmap='Reds', ax=ax, cbar_kws={'label': 'Normalized Score'},
                linewidths=2, linecolor='white', square=True)
    
    ax.set_title(f'Performance Summary - {objective_name.title()} Objective', 
                fontweight='normal', pad=10)
    ax.set_xlabel('Algorithm', fontweight='bold')
    ax.set_ylabel('Metric', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_style_7_combined_scatter_means(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 7: Scatter with mean points emphasized"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    algorithms = df['algorithm'].unique()
    colors = PASTEL_COLORS[:len(algorithms)]
    color_map = dict(zip(algorithms, colors))
    
    # Plot individual runs (light)
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo]
        ax.scatter(algo_data['makespan_minutes'], 
                  algo_data['total_cost'],
                  s=60,
                  alpha=0.3,
                  color=color_map[algo],
                  edgecolors='gray',
                  linewidth=0.5)
    
    # Plot means (emphasized)
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo]
        mean_makespan = algo_data['makespan_minutes'].mean()
        mean_cost = algo_data['total_cost'].mean()
        
        ax.scatter(mean_makespan, mean_cost,
                  label=f'{algo} (mean)',
                  s=300,
                  alpha=0.9,
                  color=color_map[algo],
                  edgecolors='#333333',
                  linewidth=1.5,
                  marker='D',
                  zorder=10)
        
        # Add error bars
        std_makespan = algo_data['makespan_minutes'].std()
        std_cost = algo_data['total_cost'].std()
        ax.errorbar(mean_makespan, mean_cost,
                   xerr=std_makespan, yerr=std_cost,
                   color=color_map[algo], alpha=0.5,
                   linewidth=2, capsize=5, capthick=2, zorder=5)
    
    ax.set_xlabel('Makespan (minutes)', fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontweight='bold')
    ax.set_title(f'Mean Performance with Error Bars - {objective_name.title()} Objective', 
                fontweight='normal', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_style_8_stacked_normalized_bar(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 8: Stacked normalized bar chart showing relative performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = df['algorithm'].unique()
    colors = PASTEL_COLORS[:len(algorithms)]
    
    # Calculate mean values
    stats = df.groupby('algorithm').agg({
        'makespan_minutes': 'mean',
        'total_cost': 'mean'
    }).reindex(algorithms)
    
    # Normalize to percentages (what % of total does each metric contribute)
    makespan_min = stats['makespan_minutes'].min()
    makespan_max = stats['makespan_minutes'].max()
    cost_min = stats['total_cost'].min()
    cost_max = stats['total_cost'].max()
    
    # Normalize each to [0, 1]
    if makespan_max > makespan_min:
        makespan_norm = (stats['makespan_minutes'] - makespan_min) / (makespan_max - makespan_min)
    else:
        makespan_norm = pd.Series([0.5] * len(algorithms), index=algorithms)
    
    if cost_max > cost_min:
        cost_norm = (stats['total_cost'] - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = pd.Series([0.5] * len(algorithms), index=algorithms)
    
    # Convert to percentages
    total = makespan_norm + cost_norm
    makespan_pct = (makespan_norm / total) * 100
    cost_pct = (cost_norm / total) * 100
    
    x_pos = np.arange(len(algorithms))
    
    # Create stacked bars
    p1 = ax.bar(x_pos, makespan_pct, color='steelblue', alpha=0.8, 
                edgecolor='#333333', linewidth=0.8, label='Makespan')
    p2 = ax.bar(x_pos, cost_pct, bottom=makespan_pct, color='coral', 
                alpha=0.8, edgecolor='#333333', linewidth=0.8, label='Cost')
    
    # Add percentage labels
    for i, (m_pct, c_pct) in enumerate(zip(makespan_pct, cost_pct)):
        ax.text(i, m_pct/2, f'{m_pct:.1f}%', ha='center', va='center', 
                fontweight='bold', fontsize=8)
        ax.text(i, m_pct + c_pct/2, f'{c_pct:.1f}%', ha='center', va='center', 
                fontweight='bold', fontsize=8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=15, ha='right')
    ax.set_ylabel('Relative Performance Contribution (%)', fontweight='bold')
    ax.set_title(f'Normalized Performance Breakdown - {objective_name.title()} Objective\n' +
                'Higher % = worse relative performance for that metric', 
                fontweight='normal', pad=10)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_style_9_performance_profile(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 9: Performance profile showing cumulative distribution - saved separately"""
    algorithms = df['algorithm'].unique()
    colors = PASTEL_COLORS[:len(algorithms)]
    color_map = dict(zip(algorithms, colors))
    
    # Performance profile for makespan
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]['makespan_minutes'].values
        sorted_data = np.sort(algo_data)
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
        
        ax1.plot(sorted_data, y_vals, marker='o', markersize=6, linewidth=2,
                label=algo, color=color_map[algo], alpha=0.8)
    
    ax1.set_xlabel('Makespan (minutes)')
    ax1.set_ylabel('Cumulative Probability (%)')
    # ax1.set_title(f'Makespan Performance Profile - {objective_name.title()} Objective', # \nCurves shifted left are better
    #              fontweight='normal', pad=10, fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout()
    makespan_path = output_path.parent / f"{output_path.stem}_makespan.png"
    plt.savefig(makespan_path, dpi=300, bbox_inches='tight')
    plt.savefig(makespan_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {makespan_path}")
    plt.close()
    
    # Performance profile for cost
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]['total_cost'].values
        sorted_data = np.sort(algo_data)
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
        
        ax2.plot(sorted_data, y_vals, marker='o', markersize=6, linewidth=2,
                label=algo, color=wesanderson.film_palette('The Royal Tenenbaums', 0)[i], alpha=0.8)
    
    ax2.set_xlabel('Total Cost ($)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability (%)', fontsize=11)
    # ax2.set_title(f'Cost Performance Profile\n{objective_name.title()} Objective', # \nCurves shifted left are better 
    #              fontweight='normal', pad=15, fontsize=18)
    ax2.grid(True, alpha=0.3)
    ax2.legend(["BO-Classic", "BnB+BO"], fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.tick_params(axis='both', labelsize=11, length=0)
    
    plt.tight_layout()
    cost_path = output_path.parent / f"{output_path.stem}_cost.png"
    plt.savefig(cost_path, dpi=300, bbox_inches='tight')
    plt.savefig(cost_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    plt.savefig(cost_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {cost_path}")
    plt.close()


def plot_style_10_radar_chart(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 10: Radar chart comparing algorithms across metrics"""
    algorithms = df['algorithm'].unique()
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Calculate global normalization values
    makespan_min_global = df['makespan_minutes'].min()
    makespan_max_global = df['makespan_minutes'].max()
    cost_min_global = df['total_cost'].min()
    cost_max_global = df['total_cost'].max()
    
    metrics = ['Makespan', 'Cost', 'Std(Makespan)', 'Std(Cost)']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = PASTEL_COLORS[:len(algorithms)]
    color_map = dict(zip(algorithms, colors))
    
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo]
        
        # Calculate means and stds
        mean_makespan = algo_data['makespan_minutes'].mean()
        mean_cost = algo_data['total_cost'].mean()
        std_makespan = algo_data['makespan_minutes'].std()
        std_cost = algo_data['total_cost'].std()
        
        # Normalize (0 = best, 1 = worst)
        if makespan_max_global > makespan_min_global:
            makespan_norm = (mean_makespan - makespan_min_global) / (makespan_max_global - makespan_min_global)
        else:
            makespan_norm = 0.5
        
        if cost_max_global > cost_min_global:
            cost_norm = (mean_cost - cost_min_global) / (cost_max_global - cost_min_global)
        else:
            cost_norm = 0.5
        
        # Normalize standard deviations
        std_makespan_max = df.groupby('algorithm')['makespan_minutes'].std().max()
        std_cost_max = df.groupby('algorithm')['total_cost'].std().max()
        
        std_makespan_norm = std_makespan / std_makespan_max if std_makespan_max > 0 else 0
        std_cost_norm = std_cost / std_cost_max if std_cost_max > 0 else 0
        
        values = [makespan_norm, cost_norm, std_makespan_norm, std_cost_norm]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=color_map[algo])
        ax.fill(angles, values, alpha=0.15, color=color_map[algo])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title(f'Algorithm Comparison Radar - {objective_name.title()} Objective\n' +
                'Smaller area = better (closer to center = lower values)', 
                fontweight='normal', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_style_11_scatter_matrix(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 11: Scatter matrix/pair plot showing relationships"""
    # Add algorithm as a categorical column for coloring
    algorithms = df['algorithm'].unique()
    colors = PASTEL_COLORS[:len(algorithms)]
    
    # Create pair plot
    pair_data = df[['makespan_minutes', 'total_cost', 'algorithm']].copy()
    
    g = sns.pairplot(pair_data, hue='algorithm', palette=colors,
                     diag_kind='kde', plot_kws={'alpha': 0.7, 's': 80, 'edgecolor': '#333333', 'linewidth': 0.5},
                     diag_kws={'alpha': 0.7, 'linewidth': 2})
    
    g.fig.suptitle(f'Scatter Matrix - {objective_name.title()} Objective', 
                   fontweight='normal', y=1.00)
    
    # Adjust labels
    g.axes[0, 0].set_ylabel('Makespan (min)', fontweight='bold')
    g.axes[1, 0].set_ylabel('Total Cost ($)', fontweight='bold')
    g.axes[1, 0].set_xlabel('Makespan (min)', fontweight='bold')
    g.axes[1, 1].set_xlabel('Total Cost ($)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_style_12_stacked_area(df: pd.DataFrame, objective_name: str, output_path: Path):
    """Style 12: Normalized stacked area chart showing metric contributions"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = df['algorithm'].unique()
    
    # Calculate normalized values for each algorithm
    all_data = []
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo]
        
        mean_makespan = algo_data['makespan_minutes'].mean()
        mean_cost = algo_data['total_cost'].mean()
        
        all_data.append({
            'algorithm': algo,
            'makespan': mean_makespan,
            'cost': mean_cost
        })
    
    data_df = pd.DataFrame(all_data)
    
    # Normalize to get proportions
    makespan_min = data_df['makespan'].min()
    makespan_max = data_df['makespan'].max()
    cost_min = data_df['cost'].min()
    cost_max = data_df['cost'].max()
    
    if makespan_max > makespan_min:
        data_df['makespan_norm'] = (data_df['makespan'] - makespan_min) / (makespan_max - makespan_min)
    else:
        data_df['makespan_norm'] = 0.5
    
    if cost_max > cost_min:
        data_df['cost_norm'] = (data_df['cost'] - cost_min) / (cost_max - cost_min)
    else:
        data_df['cost_norm'] = 0.5
    
    # Create stacked area
    x_pos = np.arange(len(algorithms))
    
    ax.fill_between(x_pos, 0, data_df['makespan_norm'], 
                    alpha=0.7, color='steelblue', label='Makespan (normalized)')
    ax.fill_between(x_pos, data_df['makespan_norm'], 
                    data_df['makespan_norm'] + data_df['cost_norm'],
                    alpha=0.7, color='coral', label='Cost (normalized)')
    
    # Add algorithm labels
    for i, algo in enumerate(algorithms):
        makespan_val = data_df.iloc[i]['makespan_norm']
        cost_val = data_df.iloc[i]['cost_norm']
        
        # Label in the middle of each section
        ax.text(i, makespan_val/2, f'{makespan_val:.2f}', 
               ha='center', va='center', fontweight='bold', fontsize=8)
        ax.text(i, makespan_val + cost_val/2, f'{cost_val:.2f}', 
               ha='center', va='center', fontweight='bold', fontsize=8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=15, ha='right')
    ax.set_ylabel('Normalized Performance Score', fontweight='bold')
    ax.set_title(f'Cumulative Normalized Performance - {objective_name.title()} Objective\n' +
                'Lower total height = better overall performance', 
                fontweight='normal', pad=10)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_violin_perf_profile_cost(df: pd.DataFrame, objective_name: str, output_path: Path):
    algorithms = df['algorithm'].unique()
    algo_str = ["BnB+BO", "BO-Classic"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

    colors = [
        "#899DA4",
        "#C93312",
    ]
    
    # Violin plot
    sns.violinplot(
        data=df,
        x='algorithm',
        y='total_cost',
        ax=ax1,
        palette=colors[::-1],
        inner='quartile',
        order=algorithms[::-1]
    )
    
    ax1.set_xticks(range(len(algo_str)))
    ax1.set_xticklabels(algo_str)
    ax1.set_xlabel(None)
    ax1.set_ylabel('Total Cost ($)', fontsize=16)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', labelsize=16, length=0)
    ax1.set_title("(a) Cost distribution", fontsize=16)
    
    # Performance profile
    palette = wesanderson.film_palette('The Royal Tenenbaums', 0)
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]['total_cost'].values
        sorted_data = np.sort(algo_data)
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    
        ax2.plot(
            sorted_data,
            y_vals,
            marker='o',
            markersize=6,
            linewidth=2,
            label=algo_str[::-1][i],   # use pretty labels directly
            color=palette[i],
            alpha=0.8
        )
    
    ax2.set_xlabel('Total Cost ($)', fontsize=16)
    ax2.set_ylabel('Cum. Probability (%)', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14, frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', labelsize=14, length=0)
    ax2.set_title("(b) Budget-meeting probability", fontsize=16)
    
    plt.tight_layout()
    
    cost_path = output_path.parent / f"{output_path.stem}_cost.png"
    plt.savefig(cost_path, dpi=300, bbox_inches='tight')
    plt.savefig(cost_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    plt.savefig(cost_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {cost_path}")
    plt.close()


def main():
    """Generate all plot styles for comparison."""
    base_dir = Path.cwd()
    
    objective_dirs = [d for d in base_dir.iterdir() 
                     if d.is_dir() and d.name.__contains__('-machines-')]
    
    if not objective_dirs:
        print("No objective directories found")
        return
    
    plot_functions = [
        # ('style1_scatter', plot_style_1_scatter),
        # ('style2_bars', plot_style_2_bar_comparison),
        # ('style3_violin', plot_style_3_violin),
        # ('style4_pareto', plot_style_4_pareto_front),
        # ('style5_strip', plot_style_5_strip_with_means),
        # ('style6_heatmap', plot_style_6_heatmap_table),
        # ('style7_scatter_means', plot_style_7_combined_scatter_means),
        # ('style8_stacked_norm', plot_style_8_stacked_normalized_bar),
        # ('style9_perf_profile', plot_style_9_performance_profile),
        # ('style10_radar', plot_style_10_radar_chart),
        # ('style11_scatter_matrix', plot_style_11_scatter_matrix),
        # ('style12_stacked_area', plot_style_12_stacked_area),
        ('violin_and_perf', plot_combined_violin_perf_profile_cost),
    ]
    
    for obj_dir in sorted(objective_dirs):
        objective_name = obj_dir.name.split('-')[-1]
        
        print(f"\n{'='*60}")
        print(f"Processing: {objective_name.upper()} objective")
        print(f"{'='*60}")
        
        results = load_optimization_results(obj_dir)
        if not results:
            print(f"No results found in {obj_dir}")
            continue
        
        df = extract_metrics(results)
        if df.empty:
            print(f"No valid data extracted")
            continue
        
        print(f"Found {len(df)} results across {len(df['algorithm'].unique())} algorithms")
        print(f"Generating {len(plot_functions)} plot styles...")
        
        for style_name, plot_func in plot_functions:
            output_path = obj_dir / f"{objective_name}_{style_name}.png"
            try:
                plot_func(df, objective_name, output_path)
            except Exception as e:
                print(f"Error generating {style_name}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()