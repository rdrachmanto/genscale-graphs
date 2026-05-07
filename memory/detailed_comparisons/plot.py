#!/usr/bin/env python3
"""
Memory prediction analysis tool for baseline and FAMR results
Publication-quality styling version
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
from pathlib import Path
import wesanderson

# ============================================================================
# GLOBAL STYLE CONFIGURATION FOR RESEARCH PAPER
# ============================================================================
BASELINE_COLORS = {
    'Sizey':           '#E07A84',  # Strong Rose
    'WittPercentile':  '#F28C52',  # Vibrant Coral
    'WittRegression':  '#C38BC3',  # Rich Lavender
    'Statistical_P95': '#F2C265',  # Warm Sand
}

FAMR_COLORS = {
    'FAMR_1':    '#5FB0DA',  # Bright Sky Blue
    'FAMR_1.15': '#78AEE6',  # Clear Periwinkle
    'FAMR_2':    '#7ED6D6',  # Crisp Aqua
}

FAMR_HATCH = '///'
FAMR_HATCH_COLOR = '#ffffff80'  # Semi-transparent white

def get_method_color(method):
    if method in FAMR_COLORS:
        return FAMR_COLORS[method]
    return BASELINE_COLORS.get(method, '#C6A27E')

def is_famr(method):
    return method.startswith('FAMR')

PASTEL_COLORS = [
    '#E07A84',  # Strong Rose
    '#F28C52',  # Vibrant Coral
    '#8FC3A9',  # Fresh Sage
    '#5FB0DA',  # Bright Sky Blue
    '#C38BC3',  # Rich Lavender
    '#F2C265',  # Warm Sand
    '#78AEE6',  # Clear Periwinkle
    '#C6A27E',  # Deep Taupe
    '#88C97A',  # Lively Mint
    '#E6A8C4',  # Bright Mauve
    '#7ED6D6',  # Crisp Aqua
    '#E9C57C',  # Sunny Wheat
]

replaceToolsWithAbbr = {
    'fastqc': 'FQC',
    'fastq-cleaner': 'FQ-C',
    'burrows-wheeler-aligner': 'BWA',
    'picard-markduplicate': 'PMd',
    'samtools-sort-markduplicate': 'SSMd',
    'samtools-index-markduplicate': 'SIMd',
    'gatk-base-recalibrator': 'BQSR',
    'gatk-apply-bqsr': 'ABQSR',
    'picard-validate-sam': 'PVS',
    'picard-collect-wgs-metrics': 'PCWM',
}

# Set publication-quality matplotlib parameters
plt.rcParams.update({
    # Figure
    'figure.facecolor': 'white',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    
    # Font settings - Times-like for academic feel
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
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

# ============================================================================
# CONFIGURATION
# ============================================================================

IGNORE_ESTIMATORS = [
    'Tovar',
    'FAMR_1.05',
    'FAMR_1.1',
    'FAMR_1.75',
    'Statistical_P90',
    # 'Statistical_P95',
    # 'WittRegression',
]

ACCURACY_METRICS_IGNORED_TOOLS = []  # e.g. ['featurecounts']

WATERFALL_IGNORED_TOOLS = []  # e.g. ['fastqc', 'fastq-cleaner']

# Use pastel colors for scatter plots
SCATTER_COLORS = PASTEL_COLORS

def save_plot(filename):
    """Save plot in both PNG and PDF formats"""
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def get_tool_pipeline_order():
    """Define the logical order of tools in the memory analysis workflow"""
    return [
        'fastqc',
        'fastq-cleaner',
        'burrows-wheeler-aligner',
        'picard-markduplicate',
        'samtools-sort-markduplicate',
        'samtools-index-markduplicate',
        'gatk-base-recalibrator',
        'gatk-apply-bqsr',
        'picard-validate-sam',
        'picard-collect-wgs-metrics'
    ]

def load_and_transform_data():
    """Load CSV files and transform to match original plotter format"""
    
    baseline_file = "baseline_detailed_results.csv"
    famr_file = "famr_detailed_results.csv"
    
    dfs = []
    
    if Path(baseline_file).exists():
        df_baseline = pd.read_csv(baseline_file)
        df_baseline['Source_File'] = baseline_file
        dfs.append(df_baseline)
        print(f"Loaded {len(df_baseline)} baseline records")
    
    if Path(famr_file).exists():
        df_famr = pd.read_csv(famr_file)
        df_famr['Source_File'] = famr_file
        dfs.append(df_famr)
        print(f"Loaded {len(df_famr)} FAMR records")
    
    if not dfs:
        print("No data files found. Looking for: baseline_detailed_results.csv, famr_detailed_results.csv")
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    
    df = df.rename(columns={
        'tool_name': 'TaskName',
        'method_name': 'Estimator', 
        'actual_memory_bytes': 'Real',
        'predicted_memory_bytes': 'Predicted',
        'input_size': 'SizeInput'
    })
    
    # Handle failed predictions (where predicted = 0)
    df = df[df['Predicted'] > 0].copy()
    
    # Calculate derived metrics
    df['Absolute_Error'] = abs(df['Real'] - df['Predicted'])
    df['Squared_Error'] = (df['Real'] - df['Predicted']) ** 2
    df['Deviation'] = (df['Real'] - df['Predicted']) / df['Real']
    df['APE'] = abs(df['Deviation']) * 100
    
    # Calculate time-based metrics
    df['Total_Time'] = df['exec_time_s'] + df['retry_penalty_time_s']
    df['Time_Wasted_Due_To_Prediction'] = df['retry_penalty_time_s']
    df['Time_Efficiency'] = df['exec_time_s'] / df['Total_Time']
    
    print(f"\nTime Statistics Debug:")
    print(f"  Exec time range: {df['exec_time_s'].min():.0f} - {df['exec_time_s'].max():.0f} seconds")
    print(f"  Penalty time range: {df['retry_penalty_time_s'].min():.0f} - {df['retry_penalty_time_s'].max():.0f} seconds") 
    print(f"  Total time range: {df['Total_Time'].min():.0f} - {df['Total_Time'].max():.0f} seconds")
    print(f"  Average penalty time: {df['retry_penalty_time_s'].mean():.1f} seconds")
    
    df['Machine'] = 'CPU:Local_Machine,RAM:16GB'
    df['Workflow'] = 'Memory_Prediction'
    
    if IGNORE_ESTIMATORS:
        print(f"Ignoring estimators: {IGNORE_ESTIMATORS}")
        df = df[~df['Estimator'].isin(IGNORE_ESTIMATORS)]
    
    print(f"Final dataset: {len(df)} records")
    print(f"Estimators: {', '.join(df['Estimator'].unique())}")
    print(f"Tools: {', '.join(df['TaskName'].unique())}")
    
    return df

def plot_predicted_vs_actual(df):
    """Scatter plot: Predicted vs Actual memory usage by estimator"""
    estimators = df['Estimator'].unique()
    n_estimators = len(estimators)
    
    cols = min(3, n_estimators)
    rows = (n_estimators + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_estimators == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, estimator in enumerate(estimators):
        est_data = df[df['Estimator'] == estimator]
        
        color = SCATTER_COLORS[i % len(SCATTER_COLORS)]
        axes[i].scatter(est_data['Real'], est_data['Predicted'], 
                       alpha=0.7, color=color, s=50, edgecolor='#333333', linewidth=0.5)
        
        min_val = min(est_data['Real'].min(), est_data['Predicted'].min())
        max_val = max(est_data['Real'].max(), est_data['Predicted'].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
        
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        
        axes[i].set_xlabel('Actual Memory (MB)')
        axes[i].set_ylabel('Predicted Memory (MB)')
        axes[i].set_title(f'{estimator}')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
    
    for i in range(n_estimators, len(axes)):
        axes[i].set_visible(False)
    
    save_plot('predicted_vs_actual')

def plot_error_distribution(df):
    """Box plot: Error distribution by estimator"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bp = ax.boxplot([df[df['Estimator'] == est]['Deviation'].values 
                      for est in df['Estimator'].unique()],
                     labels=df['Estimator'].unique(),
                     patch_artist=True,
                     widths=0.6)
    
    for patch, color in zip(bp['boxes'], SCATTER_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(0.8)
    
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='#333333', linewidth=0.8)
    
    ax.set_title('Memory Prediction Error Distribution by Method')
    ax.set_ylabel('Relative Error\n(Negative = Over-prediction, Positive = Under-prediction)')
    ax.set_xlabel('Prediction Method')
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=0, ha='center')
    
    save_plot('error_distribution')

def plot_performance_by_task(df):
    """Performance comparison across tools"""
    task_stats = df.groupby(['TaskName', 'Estimator'])['Deviation'].mean().unstack()
    
    pipeline_order = get_tool_pipeline_order()
    ordered = [t for t in pipeline_order if t in task_stats.index]
    remaining = [t for t in task_stats.index if t not in pipeline_order]
    task_stats = task_stats.reindex(ordered + remaining)
    
    from matplotlib.colors import LinearSegmentedColormap
    pastel_cmap_diverging = LinearSegmentedColormap.from_list(
        'pastel_diverging', ['#5FB0DA', '#7ED6D6', '#FFFFFF', '#F2C265', '#E07A84'], N=256
    )
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.heatmap(task_stats, annot=True, fmt='.3f', cmap=pastel_cmap_diverging, 
                center=0, linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean Relative Error'})
    
    ax.set_title('Mean Relative Error by Tool and Prediction Method')
    ax.set_ylabel('Tool')
    ax.set_xlabel('Prediction Method')
    plt.xticks(rotation=0, ha='center')
    plt.yticks(
        ticks=[i + 0.5 for i in range(len(task_stats.index))],
        labels=[replaceToolsWithAbbr.get(label, label) for label in task_stats.index],
        rotation=0,
        ha='right'
    )
    
    save_plot('performance_by_task')

def plot_input_size_vs_error(df):
    """Error vs input size"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    estimators = df['Estimator'].unique()
    
    for i, estimator in enumerate(estimators):
        est_data = df[df['Estimator'] == estimator]
        color = SCATTER_COLORS[i % len(SCATTER_COLORS)]
        
        ax.scatter(est_data['SizeInput'], est_data['Deviation'], 
                   alpha=0.7, label=estimator, color=color, s=50, 
                   edgecolor='#333333', linewidth=0.5)
    
    ax.set_xscale('log')
    ax.set_xlabel('Input Size')
    ax.set_ylabel('Relative Error')
    ax.set_title('Prediction Error vs Input Size')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_plot('error_vs_input_size')

def plot_accuracy_metrics(df):
    """Plot MAE, RMSE, MAPE by tool-estimator combination"""
    metrics = df.groupby(['TaskName', 'Estimator']).agg({
        'Absolute_Error': 'mean',
        'Squared_Error': lambda x: np.sqrt(x.mean()),
        'APE': 'mean'
    }).round(3)
    
    metrics.columns = ['MAE', 'RMSE', 'MAPE']

    from matplotlib.colors import LinearSegmentedColormap
    cmap_map = {
        'MAE': LinearSegmentedColormap.from_list('pastel_reds', ['#FFFFFF', '#F2C265', '#F28C52', '#E07A84'], N=256),
        'RMSE': LinearSegmentedColormap.from_list('pastel_oranges', ['#FFFFFF', '#F2C265', '#F28C52', '#C6A27E'], N=256),
        'MAPE': LinearSegmentedColormap.from_list('pastel_blues', ['#FFFFFF', '#7ED6D6', '#5FB0DA', '#78AEE6'], N=256),
    }

    pipeline_order = get_tool_pipeline_order()
    
    for metric in ['MAE', 'RMSE', 'MAPE']:
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_data = metrics[metric].unstack(fill_value=0)

        ordered = [t for t in pipeline_order if t in pivot_data.index and t not in ACCURACY_METRICS_IGNORED_TOOLS]
        remaining = [t for t in pivot_data.index if t not in pipeline_order and t not in ACCURACY_METRICS_IGNORED_TOOLS]
        pivot_data = pivot_data.reindex(ordered + remaining)

        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap=cmap_map[metric],
                   linewidths=0.5, cbar_kws={'label': f'{metric} (MB)'}, ax=ax)
        ax.set_title(f'{metric} by Tool and Prediction Method')
        ax.set_ylabel('Tool')
        ax.set_xlabel('Prediction Method')
        plt.xticks(rotation=0, ha='center')
        plt.yticks(
            ticks=[i + 0.5 for i in range(len(pivot_data.index))],
            labels=[replaceToolsWithAbbr.get(label, label) for label in pivot_data.index],
            rotation=0,
            ha='right'
        )
        
        save_plot(f'{metric.lower()}_by_task')

def get_tool_workflow_order():
    """Define logical order of tools (alias for compatibility)"""
    return get_tool_pipeline_order()

def prepare_workflow_analysis(df):
    """Prepare data for workflow analysis"""
    tool_analysis = df.groupby(['TaskName', 'Estimator']).agg({
        'Real': 'mean',
        'Absolute_Error': 'mean',
        'Deviation': 'mean',
        'APE': 'mean',
        'Total_Time': 'mean',
        'exec_time_s': 'mean',
        'Time_Wasted_Due_To_Prediction': 'mean',
        'Time_Efficiency': 'mean',
        'num_retries': 'mean'
    }).reset_index()
    
    for estimator in tool_analysis['Estimator'].unique():
        est_data = tool_analysis[tool_analysis['Estimator'] == estimator].copy()
        total_memory = est_data['Real'].sum()
        total_time = est_data['Total_Time'].sum()
        
        tool_analysis.loc[tool_analysis['Estimator'] == estimator, 'Memory_Weight'] = \
            est_data['Real'] / total_memory * 100
        tool_analysis.loc[tool_analysis['Estimator'] == estimator, 'Time_Weight'] = \
            est_data['Total_Time'] / total_time * 100
    
    tool_analysis['Weighted_Memory_Error_Contribution'] = (
        tool_analysis['Absolute_Error'] * tool_analysis['Memory_Weight'] / 100
    )
    
    tool_analysis['Time_Efficiency_Loss_Percent'] = (1 - tool_analysis['Time_Efficiency']) * 100
    tool_analysis['Slowdown_Factor'] = tool_analysis['Total_Time'] / tool_analysis['exec_time_s']
    tool_analysis['Waste_Ratio_Percent'] = (tool_analysis['Time_Wasted_Due_To_Prediction'] / tool_analysis['exec_time_s']) * 100
    tool_analysis['Weighted_Time_Waste_Contribution'] = tool_analysis['Time_Efficiency_Loss_Percent']
    tool_analysis['Weighted_Error_Contribution'] = tool_analysis['Weighted_Memory_Error_Contribution']
    
    return tool_analysis

def plot_error_waterfall(tool_analysis):
    """Waterfall chart showing cumulative memory error through tools"""
    if tool_analysis is None or tool_analysis.empty:
        print("No tool analysis data for waterfall chart")
        return
    
    estimators = tool_analysis['Estimator'].unique()
    tool_order = get_tool_pipeline_order()
    available_tools = [tool for tool in tool_order if tool in tool_analysis['TaskName'].values]
    
    if not available_tools:
        available_tools = sorted(tool_analysis['TaskName'].unique())
    
    n_estimators = len(estimators)
    cols = min(2, n_estimators)
    rows = (n_estimators + cols - 1) // cols
    
    global_max = 0
    for estimator in estimators:
        est_data = tool_analysis[tool_analysis['Estimator'] == estimator]
        contributions = []
        for tool in available_tools:
            tool_row = est_data[est_data['TaskName'] == tool]
            if not tool_row.empty:
                contributions.append(abs(tool_row['Weighted_Error_Contribution'].iloc[0]))
        if contributions:
            cumulative = np.cumsum([0] + contributions)
            global_max = max(global_max, cumulative[-1])
    
    fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 7*rows))
    if n_estimators == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for est_idx, estimator in enumerate(estimators):
        ax = axes[est_idx]
        est_data = tool_analysis[tool_analysis['Estimator'] == estimator]
        
        ordered_data = []
        for tool in available_tools:
            tool_row = est_data[est_data['TaskName'] == tool]
            if not tool_row.empty:
                ordered_data.append({
                    'tool': tool,
                    'error_contrib': tool_row['Weighted_Error_Contribution'].iloc[0]
                })
        
        if not ordered_data:
            continue
            
        contributions = [abs(d['error_contrib']) for d in ordered_data]
        cumulative = np.cumsum([0] + contributions)
        
        colors = [PASTEL_COLORS[i % len(PASTEL_COLORS)] for i in range(len(contributions))]
        
        bars = ax.bar(range(len(contributions)), contributions, 
                     bottom=cumulative[:-1], color=colors, alpha=0.8, edgecolor='#333333', linewidth=0.8)
        
        ax.plot(range(len(cumulative)), cumulative, 'ko-', linewidth=1.5, markersize=5)
        
        for i, (bar, contrib, cum) in enumerate(zip(bars, contributions, cumulative[1:])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                   f'+{contrib:.1f}', ha='center', va='center', fontsize=8, color='white')
            ax.text(i, cum + cum*0.01, f'{cum:.1f}', ha='center', va='bottom', fontsize=8)
        
        tool_names = [replaceToolsWithAbbr.get(d['tool'], d['tool']) for d in ordered_data]
        ax.set_xticks(range(len(tool_names)))
        ax.set_xticklabels(tool_names, rotation=0, ha='center')
        ax.set_ylabel('Memory Error Contribution (MB)')
        ax.set_title(f'{estimator} - Memory Error Accumulation Across Tools')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, global_max * 1.1)
    
    for i in range(n_estimators, len(axes)):
        axes[i].set_visible(False)
    
    save_plot('error_waterfall')

def plot_time_waste_waterfall(tool_analysis):
    """Waterfall chart showing cumulative time waste through tools"""
    if tool_analysis is None or tool_analysis.empty:
        print("No tool analysis data for time waste waterfall chart")
        return
    
    estimators = tool_analysis['Estimator'].unique()
    tool_order = get_tool_pipeline_order()
    available_tools = [tool for tool in tool_order if tool in tool_analysis['TaskName'].values]
    
    if not available_tools:
        available_tools = sorted(tool_analysis['TaskName'].unique())
    
    fig, axes = plt.subplots(1, len(estimators), figsize=(10*len(estimators), 7))
    if len(estimators) == 1:
        axes = [axes]
    
    for est_idx, estimator in enumerate(estimators):
        ax = axes[est_idx]
        est_data = tool_analysis[tool_analysis['Estimator'] == estimator]
        
        ordered_data = []
        for tool in available_tools:
            tool_row = est_data[est_data['TaskName'] == tool]
            if not tool_row.empty:
                ordered_data.append({
                    'tool': tool,
                    'time_waste_contrib': tool_row['Weighted_Time_Waste_Contribution'].iloc[0],
                    'avg_penalty_time': tool_row['Time_Wasted_Due_To_Prediction'].iloc[0],
                    'avg_retries': tool_row['num_retries'].iloc[0],
                    'time_efficiency': tool_row['Time_Efficiency'].iloc[0]
                })
        
        if not ordered_data:
            continue
            
        contributions = [d['time_waste_contrib'] for d in ordered_data]
        cumulative = np.cumsum([0] + contributions)
        
        colors = []
        for d in ordered_data:
            if d['time_efficiency'] > 0.95:
                colors.append(PASTEL_COLORS[2])
            elif d['time_efficiency'] > 0.8:
                colors.append(PASTEL_COLORS[5])
            else:
                colors.append(PASTEL_COLORS[1])
        
        bars = ax.bar(range(len(contributions)), contributions, 
                     bottom=cumulative[:-1], color=colors, alpha=0.8, edgecolor='#333333', linewidth=0.8)
        
        ax.plot(range(len(cumulative)), cumulative, 'ko-', linewidth=1.5, markersize=5)
        
        for i, (bar, contrib, cum, data) in enumerate(zip(bars, contributions, cumulative[1:], ordered_data)):
            if contrib > 3600:
                contrib_str = f'+{contrib/3600:.1f}h'
            elif contrib > 60:
                contrib_str = f'+{contrib/60:.1f}m'
            else:
                contrib_str = f'+{contrib:.0f}s'
                
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                   contrib_str, ha='center', va='center', fontsize=8, color='white')
            
            if cum > 3600:
                cum_str = f'{cum/3600:.1f}h'
            elif cum > 60:
                cum_str = f'{cum/60:.1f}m'
            else:
                cum_str = f'{cum:.0f}s'
                
            ax.text(i, cum + cum*0.01, cum_str, ha='center', va='bottom', fontsize=8)
            ax.text(i, -max(contributions)*0.1, f'{data["time_efficiency"]*100:.0f}%', 
                   ha='center', va='top', fontsize=8, color='#666666')
        
        tool_names = [replaceToolsWithAbbr.get(d['tool'], d['tool']) for d in ordered_data]
        ax.set_xticks(range(len(tool_names)))
        ax.set_xticklabels(tool_names, rotation=0, ha='center')
        ax.set_ylabel('Time Waste Contribution (seconds)')
        ax.set_title(f'{estimator} - Time Waste Accumulation')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        
    
    save_plot('time_waste_waterfall')

def plot_time_vs_memory_analysis(df):
    """Combined analysis showing relationship between memory prediction errors and time waste"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    estimators = df['Estimator'].unique()
    
    # Penalty time distribution
    ax2 = axes[0]
    bp = ax2.boxplot([df[df['Estimator'] == est]['Time_Wasted_Due_To_Prediction'].values 
                      for est in estimators],
                     labels=estimators,
                     patch_artist=True,
                     widths=0.6)
    
    for patch, color in zip(bp['boxes'], SCATTER_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(0.8)
    
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='#333333', linewidth=0.8)
    
    ax2.set_xticklabels(estimators, rotation=0, ha='center')
    ax2.set_title('Time Penalty Distribution by Method')
    ax2.set_ylabel('Penalty Time (seconds)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Retry frequency analysis
    ax3 = axes[1]
    retry_stats = df.groupby(['TaskName', 'Estimator'])['num_retries'].mean().unstack(fill_value=0)
    pipeline_order = get_tool_pipeline_order()
    ordered = [t for t in pipeline_order if t in retry_stats.index]
    remaining = [t for t in retry_stats.index if t not in pipeline_order]
    retry_stats = retry_stats.reindex(ordered + remaining)
    from matplotlib.colors import LinearSegmentedColormap
    pastel_cmap_reds = LinearSegmentedColormap.from_list('pastel_reds', ['#FFFFFF', '#F2C265', '#F28C52', '#E07A84'], N=256)
    sns.heatmap(retry_stats, annot=True, fmt='.2f', cmap=pastel_cmap_reds, ax=ax3, linewidths=0.5)
    ax3.set_title('Average Number of Retries by Tool and Method')
    ax3.set_ylabel('Tool')
    ax3.set_xlabel('Prediction Method')
    plt.sca(ax3)
    plt.xticks(rotation=0, ha='center')
    plt.yticks(
        ticks=[i + 0.5 for i in range(len(retry_stats.index))],
        labels=[replaceToolsWithAbbr.get(label, label) for label in retry_stats.index],
        rotation=0,
        ha='right'
    )
    
    save_plot('time_vs_memory_analysis')

def plot_combined_time_waterfall_comparison(tool_analysis):
    """Combined time waste waterfall comparison - all estimators"""
    if tool_analysis is None or tool_analysis.empty:
        print("No tool analysis data for combined time waterfall")
        return
    
    IGNORE = [
        'samtools-index-markduplicate',
        'samtools-sort-markduplicate',
    ]
    
    estimators = tool_analysis['Estimator'].unique()
    tool_order = get_tool_pipeline_order()
    available_tools = [tool for tool in tool_order if tool in tool_analysis['TaskName'].values and tool not in IGNORE]    
    
    if not available_tools:
        available_tools = sorted(tool_analysis['TaskName'].unique())
    
    if len(available_tools) == 0:
        print("No tools found for time waterfall analysis")
        return
    
    fig, axes = plt.subplots(figsize=(8, 4))
    
    # 1. Cumulative time efficiency loss comparison
    # ax1 = axes[0]
    ax1 = axes
    for est_idx, estimator in enumerate(estimators):
        est_data = tool_analysis[tool_analysis['Estimator'] == estimator]
        
        contributions = []
        for tool in available_tools:
            tool_row = est_data[est_data['TaskName'] == tool]
            contrib = tool_row['Time_Efficiency_Loss_Percent'].iloc[0] if not tool_row.empty else 0
            contributions.append(contrib)
        
        if sum(contributions) == 0:
            continue
        
        cumulative = np.cumsum([0] + contributions)
        
        # color = get_method_color(estimator)
        color = [
            "#b6534d",
            "#a24d7b",
            "#6f5891",
            "#375e86",
            "#1e5a66",
            "#305148",
            "#3d4539"
        ]
        linestyle = ['-', '--', '-.', ':'][est_idx % 4]
        # linewidth = max(1, 2.5 - est_idx * 0.3)
        linewidth = 1.6
        
        ax1.plot(range(len(cumulative)), cumulative, color=color[est_idx], linestyle=linestyle, 
                linewidth=linewidth, marker='o', markersize=9, label=estimator, alpha=0.8)
        
        final_loss = cumulative[-1]
        ax1.annotate(f'{final_loss:.1f}%', xy=(len(cumulative)-1, final_loss),
                    xytext=(5, 5), textcoords='offset points', fontsize=13, color="#222")
    
    # ax1.set_xlabel('Tool Stage', fontsize=16)
    ax1.set_ylabel('Cumulative Time Efficiency Loss (%)', fontsize=13)
    # ax1.set_title('Cumulative Time Efficiency Loss', fontsize=18)
    # ax1.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.23), fontsize=12, ncols=4)
    ax1.legend(frameon=False, loc='upper left', fontsize=13, ncols=2)
    ax1.set_xticks(range(len(available_tools) + 1))
    ax1.set_xticklabels(
        ['Start'] + [replaceToolsWithAbbr.get(tool, tool) for tool in available_tools],
        rotation=0, ha='center'
    )
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.tick_params(axis='both', labelsize=13)
    ax1.tick_params(axis='x', rotation=0)

    ax1.tick_params(axis='both', length=0)
    ax1.set_axisbelow(True)
    ax1.grid(True, alpha=0.3)
    
    # 2. Penalty time per tool
    # ax3 = axes[1]
    # width = 0.35 / len(estimators) if len(estimators) > 0 else 0.35

    # for est_idx, estimator in enumerate(estimators):
    #     est_data = tool_analysis[tool_analysis['Estimator'] == estimator]

    #     penalty_times = []
    #     for tool in available_tools:
    #         tool_row = est_data[est_data['TaskName'] == tool]
    #         penalty = tool_row['Time_Wasted_Due_To_Prediction'].iloc[0] if not tool_row.empty else 0
    #         penalty_times.append(penalty)

    #     color = get_method_color(estimator)
    #     # Shift bars to positions 1..n so position 0 = 'Start'
    #     ax3.bar([i + 1 + est_idx * width for i in range(len(penalty_times))],
    #             penalty_times, width=width, alpha=0.8,
    #             color=color, label=estimator, edgecolor='none')

    # ax3.set_xlabel('Tool')
    # ax3.set_ylabel('Average Penalty Time (seconds)')
    # ax3.set_title('Penalty Time per Tool')
    # ax3.legend(frameon=False)
    # ax3.set_xticks(range(len(available_tools) + 1))
    # ax3.set_xticklabels(
    #     ['Start'] + [replaceToolsWithAbbr.get(tool, tool) for tool in available_tools],
    #     rotation=0, ha='center'
    # )
    # ax3.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    
    save_plot('combined_time_waterfall_comparison')
    
    print("\n=== TIME EFFICIENCY ANALYSIS INSIGHTS ===")
    
    for estimator in estimators:
        est_data = tool_analysis[tool_analysis['Estimator'] == estimator]
        
        total_efficiency_loss = 0
        worst_efficiency_loss = 0
        worst_tool = ""
        avg_slowdown = 0
        
        for tool in available_tools:
            tool_row = est_data[est_data['TaskName'] == tool]
            if not tool_row.empty:
                eff_loss = tool_row['Time_Efficiency_Loss_Percent'].iloc[0]
                slowdown = tool_row['Slowdown_Factor'].iloc[0]
                
                total_efficiency_loss += eff_loss
                avg_slowdown += slowdown
                
                if eff_loss > worst_efficiency_loss:
                    worst_efficiency_loss = eff_loss
                    worst_tool = tool
        
        avg_slowdown = avg_slowdown / len(available_tools) if available_tools else 1
        overall_efficiency = est_data['Time_Efficiency'].mean() * 100
        
        print(f"\n{estimator}:")
        print(f"  - Total efficiency loss across workflow: {total_efficiency_loss:.1f}%")
        print(f"  - Worst performing tool: {worst_tool} ({worst_efficiency_loss:.1f}% loss)")
        print(f"  - Average slowdown factor: {avg_slowdown:.1f}x")
        print(f"  - Overall time efficiency: {overall_efficiency:.1f}%")

def plot_combined_waterfall_comparison(tool_analysis):
    """Combined waterfall comparison - all estimators on one plot"""
    if tool_analysis is None or tool_analysis.empty:
        print("No tool analysis data for combined waterfall")
        return

    estimators = tool_analysis['Estimator'].unique()
    tool_order = get_tool_pipeline_order()
    available_tools = [
        tool for tool in tool_order
        if tool in tool_analysis['TaskName'].values
        and tool not in WATERFALL_IGNORED_TOOLS
    ]

    if not available_tools:
        available_tools = [t for t in sorted(tool_analysis['TaskName'].unique()) if t not in WATERFALL_IGNORED_TOOLS]

    if len(available_tools) == 0:
        print("No tools found for waterfall analysis")
        return

    fig, axes = plt.subplots(2, figsize=(9, 10))

    # 1. Multi-line cumulative comparison
    ax1 = axes[0]
    for est_idx, estimator in enumerate(estimators):
        est_data = tool_analysis[tool_analysis['Estimator'] == estimator]

        contributions = []
        for tool in available_tools:
            tool_row = est_data[est_data['TaskName'] == tool]
            contrib = abs(tool_row['Weighted_Error_Contribution'].iloc[0]) if not tool_row.empty else 0
            contributions.append(contrib)

        if sum(contributions) == 0:
            continue

        cumulative = np.cumsum([0] + contributions)

        color = SCATTER_COLORS[est_idx % len(SCATTER_COLORS)]
        linestyle = ['-', '--', '-.', ':'][est_idx % 4]
        linewidth = max(1, 2.5 - est_idx * 0.3)

        ax1.plot(range(len(cumulative)), cumulative, color=color, linestyle=linestyle,
                linewidth=linewidth, marker='o', markersize=5, label=estimator, alpha=0.8)

        final_error = cumulative[-1]
        ax1.annotate(f'{final_error:.1f}', xy=(len(cumulative)-1, final_error),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, color=color)

    ax1.set_xlabel('Tool Stage')
    ax1.set_ylabel('Cumulative Absolute Error (MB)')
    ax1.set_title('Cumulative Memory Error Comparison')
    ax1.legend(frameon=False, loc='upper left')
    ax1.set_xticks(range(len(available_tools) + 1))
    ax1.set_xticklabels(
        ['Start'] + [replaceToolsWithAbbr.get(tool, tool) for tool in available_tools],
        rotation=0, ha='center'
    )
    ax1.spines[['top', 'right']].set_visible(False)

    # 2. Error slope / acceleration analysis
    ax2 = axes[1]

    for est_idx, estimator in enumerate(estimators):
        est_data = tool_analysis[tool_analysis['Estimator'] == estimator]

        contributions = []
        for tool in available_tools:
            tool_row = est_data[est_data['TaskName'] == tool]
            contrib = abs(tool_row['Weighted_Error_Contribution'].iloc[0]) if not tool_row.empty else 0
            contributions.append(contrib)

        if sum(contributions) == 0:
            continue

        cumulative = np.cumsum([0] + contributions)
        slopes = np.diff(cumulative)

        color = SCATTER_COLORS[est_idx % len(SCATTER_COLORS)]
        linestyle = ['-', '--', '-.', ':'][est_idx % 4]
        linewidth = max(1, 2.5 - est_idx * 0.3)

        # Anchor at (0, 0) so lines start from Start
        ax2.plot(
            [0] + list(range(1, len(slopes) + 1)),
            [0] + list(slopes),
            color=color, marker='s', linestyle=linestyle,
            linewidth=linewidth, markersize=6, label=estimator, alpha=0.8
        )

    ax2.set_xlabel('Tool Stage')
    ax2.set_ylabel('Error Rate (Change per Stage, MB)')
    ax2.set_title('Error Acceleration Analysis')
    ax2.legend(frameon=False)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax2.set_xticks(range(len(available_tools) + 1))
    ax2.set_xticklabels(
        ['Start'] + [replaceToolsWithAbbr.get(tool, tool) for tool in available_tools],
        rotation=0, ha='center'
    )
    ax2.spines[['top', 'right']].set_visible(False)

    save_plot('combined_waterfall_comparison')

    print("\n=== WATERFALL COMPARISON INSIGHTS ===")

    for estimator in estimators:
        est_data = tool_analysis[tool_analysis['Estimator'] == estimator]
        contributions = []
        tool_names = []
        for tool in available_tools:
            tool_row = est_data[est_data['TaskName'] == tool]
            if not tool_row.empty:
                contributions.append(abs(tool_row['Weighted_Error_Contribution'].iloc[0]))
                tool_names.append(tool)

        if contributions:
            max_contrib_idx = np.argmax(contributions)
            worst_tool = tool_names[max_contrib_idx]
            final_error = sum(contributions)

            print(f"\n{estimator}:")
            print(f"  - Worst error contributor: {worst_tool} ({contributions[max_contrib_idx]:.1f} MB)")
            print(f"  - Total cumulative error: {final_error:.1f} MB")

def print_summary_stats(df):
    """Print summary statistics"""
    print("\n=== MEMORY PREDICTION ANALYSIS ===")
    print("Note: Negative relative errors indicate over-prediction (predicted > actual)")
    print("      Positive relative errors indicate under-prediction (predicted < actual)")
    
    summary = df.groupby('Estimator').agg({
        'Deviation': ['count', 'mean', 'std', 'min', 'max'],
        'Absolute_Error': 'mean',
        'APE': 'mean'
    }).round(4)
    
    summary.columns = ['Count', 'Mean_Rel_Error', 'Std_Rel_Error', 'Min_Rel_Error', 'Max_Rel_Error', 'MAE_MB', 'MAPE_%']
    print("\nPerformance by Prediction Method:")
    print(summary)
    
    best_estimator = summary['Mean_Rel_Error'].abs().idxmin()
    print(f"\nBest performing method (lowest absolute mean error): {best_estimator}")
    
    print("\nTools with highest mean prediction error:")
    tool_errors = df.groupby('TaskName')['APE'].mean().sort_values(ascending=False)
    print(tool_errors.round(2))
    
    print(f"\nMemory Usage Statistics:")
    print(f"  Average actual memory: {df['Real'].mean():.1f} MB")
    print(f"  Memory range: {df['Real'].min():.1f} - {df['Real'].max():.1f} MB")
    print(f"  Average absolute error: {df['Absolute_Error'].mean():.1f} MB")

def main():
    df = load_and_transform_data()
    if df is None:
        return
    
    print(f"\nDataset overview:")
    print(f"- {len(df)} successful predictions")
    print(f"- Estimators: {', '.join(df['Estimator'].unique())}")
    print(f"- Tools analyzed: {', '.join(df['TaskName'].unique())}")
    
    print("\nGenerating core visualizations...")
    plot_predicted_vs_actual(df)
    plot_error_distribution(df) 
    plot_performance_by_task(df)
    plot_input_size_vs_error(df)
    plot_accuracy_metrics(df)
    
    print("\nGenerating time vs memory analysis...")
    plot_time_vs_memory_analysis(df)
    
    print("\nGenerating memory waterfall analysis...")
    tool_analysis = prepare_workflow_analysis(df)
    plot_error_waterfall(tool_analysis)
    plot_combined_waterfall_comparison(tool_analysis)
    
    print("\nGenerating TIME WASTE waterfall analysis...")
    plot_time_waste_waterfall(tool_analysis)
    plot_combined_time_waterfall_comparison(tool_analysis)
    
    print_summary_stats(df)
    
    print("\n" + "="*60)
    print("ALL PLOTS SAVED IN PNG AND PDF FORMATS:")
    print("="*60)
    print("\n📊 CORE ANALYSIS:")
    print("- predicted_vs_actual")
    print("- error_distribution")
    print("- performance_by_task")
    print("- error_vs_input_size")
    print("- mae_by_task, rmse_by_task, mape_by_task")
    print("\n💾 MEMORY-FOCUSED ANALYSIS:")
    print("- error_waterfall")
    print("- combined_waterfall_comparison")
    print("\n⏱️ TIME-FOCUSED ANALYSIS:")
    print("- time_vs_memory_analysis")
    print("- time_waste_waterfall")
    print("- combined_time_waterfall_comparison")

if __name__ == "__main__":
    main()