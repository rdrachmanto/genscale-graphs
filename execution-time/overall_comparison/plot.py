#!/usr/bin/env python3
"""
Enhanced visualization tool for Lotaru prediction results
Publication-quality styling version
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import wesanderson

# ============================================================================
# GLOBAL STYLE CONFIGURATION FOR RESEARCH PAPER
# ============================================================================

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

# Custom pastel colormaps for heatmaps
PASTEL_CMAP_REDS = LinearSegmentedColormap.from_list('pastel_reds', ['#FFFFFF', '#F2C265', '#F28C52', '#E07A84'], N=256)
PASTEL_CMAP_BLUES = LinearSegmentedColormap.from_list('pastel_blues', ['#FFFFFF', '#7ED6D6', '#5FB0DA', '#78AEE6'], N=256)
PASTEL_CMAP_ORANGES = LinearSegmentedColormap.from_list('pastel_oranges', ['#FFFFFF', '#F2C265', '#F28C52', '#C6A27E'], N=256)
PASTEL_CMAP_WARM = LinearSegmentedColormap.from_list('pastel_warm', ['#FFFFFF', '#F2C265', '#F28C52', '#E07A84', '#C38BC3'], N=256)

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
    'figure.facecolor': 'white',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#000000',
    'axes.labelcolor': '#000000',
    'axes.titleweight': 'normal',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'xtick.color': '#000000',
    'ytick.color': '#000000',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
    'legend.loc': 'best',
})

# ============================================================================
# CONFIGURATION
# ============================================================================

IGNORE_ESTIMATORS = [
    'Perfect',
    'Naive',
]

SCATTER_COLORS = PASTEL_COLORS

def save_plot(filename):
    """Save plot in both PNG and PDF formats"""
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def load_all_results():
    """Load all tasks_lotaru_*.csv files and filter out ignored estimators"""
    files = glob.glob("tasks_lotaru_*.csv")
    if not files:
        print("No tasks_lotaru_*.csv files found")
        return None
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df['Source_File'] = file
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    if IGNORE_ESTIMATORS:
        print(f"Ignoring estimators: {IGNORE_ESTIMATORS}")
        df = df[~df['Estimator'].isin(IGNORE_ESTIMATORS)]
        print(f"Remaining estimators: {df['Estimator'].unique()}")
    
    return df

def calculate_accuracy_metrics(df):
    """Calculate MAE, RMSE, MAPE for each estimator and task"""
    df['Absolute_Error'] = abs(df['Real'] - df['Predicted'])
    df['Squared_Error'] = (df['Real'] - df['Predicted']) ** 2
    df['APE'] = abs((df['Real'] - df['Predicted']) / df['Real']) * 100
    return df

def plot_predicted_vs_actual(df):
    """Scatter plot: Predicted vs Actual values by estimator"""
    estimators = df['Estimator'].unique()
    n_estimators = len(estimators)
    
    cols = 2
    rows = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_estimators == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if cols > 1 else [axes]
    else:
        axes = axes.flatten()
        
    fig.suptitle('Predicted vs Actual by Estimator')
    
    for i, estimator in enumerate(estimators):
        est_data = df[df['Estimator'] == estimator]
        color = SCATTER_COLORS[i % len(SCATTER_COLORS)]
        axes[i].scatter(est_data['Real'], est_data['Predicted'], 
                       alpha=0.7, color=color, s=50, edgecolor='#333333', linewidth=0.5)
        
        min_val = min(est_data['Real'].min(), est_data['Predicted'].min())
        max_val = max(est_data['Real'].max(), est_data['Predicted'].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
        
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'{estimator}')
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
    
    for i in range(n_estimators, len(axes)):
        axes[i].set_visible(False)
    
    save_plot('predicted_vs_actual')

def plot_error_distribution(df, top_min = 1, bottom_max = 0.65):
    """Box plot: Error distribution by estimator"""

    estimators = ["PTS-Hybrid", "Lotaru-G", "OnlineP", "Lotaru-A"]

    # Keep only estimators that actually exist in the dataframe
    estimators = [est for est in estimators if est in df["Estimator"].unique()]

    box_data = [
        df[df["Estimator"] == est]["Deviation"].dropna().values
        for est in estimators
    ]

    # Make sure colors are in the same order as estimators
    # Example:
    # PTS-Hybrid, Lotaru-G, OnlineP, Lotaru-A
    colors = SCATTER_COLORS[:len(estimators)]

    # ---- layout: 3 vertical panels ----
    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1, 1, 1.2],   # top, middle, bottom(CDF)
        hspace=0.22
    )

    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
    ax_cdf = fig.add_subplot(gs[2])

    def draw_boxplot(ax):
        bp = ax.boxplot(
            box_data,
            labels=estimators,
            patch_artist=True,
            widths=0.55,
            showfliers=True,
            medianprops=dict(color="#333333", linewidth=1.4),
            whiskerprops=dict(color="#666666", linewidth=0.9),
            capprops=dict(color="#666666", linewidth=0.9),
            boxprops=dict(linewidth=1.0, color="#555555"),
            flierprops=dict(
                marker='o',
                markerfacecolor='none',
                markeredgecolor='black',
                markersize=5,
                linestyle='none'
            )
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_edgecolor(color)
            patch.set_linewidth(2)

        return bp

    # ---- broken-axis boxplot ----
    draw_boxplot(ax_top)
    draw_boxplot(ax_bottom)

    max_dev = df["Deviation"].max()
    ax_bottom.set_ylim(0, bottom_max)
    ax_top.set_ylim(top_min, max_dev * 1.05)

    # Hide touching spines
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_bottom.xaxis.tick_bottom()

    # Labels
    ax_top.set_ylabel("Outliers", fontsize=12)
    ax_bottom.set_ylabel("Box body", fontsize=12)

    # x tick labels for boxplot
    plt.setp(ax_bottom.get_xticklabels(), rotation=0)

    # Tick control
    ax_bottom.set_yticks(np.arange(0, bottom_max + 0.001, 0.2))

    if top_min <= 1:
        ax_top.set_yticks([1, 5, 10, 14])
    elif top_min <= 2:
        ax_top.set_yticks([2, 5, 10, 14])

    # Break symbol: left side only
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.4)

    ax_top.plot(
        [-0.015, 0.000, 0.015],
        [-0.020, 0.000, -0.020],
        **kwargs
    )

    kwargs.update(transform=ax_bottom.transAxes)

    ax_bottom.plot(
        [-0.015, 0.000, 0.015],
        [1.020, 1.000, 1.020],
        **kwargs
    )

    # Clean up boxplot axes
    for ax in (ax_top, ax_bottom):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis='both', labelsize=11, width=1, length=0)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    # ---- CDF subplot ----
    for est, color in zip(estimators, colors):
        values = df[df["Estimator"] == est]["Deviation"].dropna().values
        values = np.sort(values)

        if len(values) == 0:
            continue

        cumulative_probability = np.arange(1, len(values) + 1) / len(values)

        ax_cdf.plot(
            values,
            cumulative_probability,
            label=est,
            color=color,
            linewidth=1.8
        )

    ax_cdf.set_xlabel("Relative Error", fontsize=12)
    ax_cdf.set_ylabel("Cumulative Probability", fontsize=12)
    ax_cdf.set_ylim(0, 1.01)
    ax_cdf.set_xlim(0, max_dev * 1.05)
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.set_axisbelow(True)
    ax_cdf.legend(frameon=False, fontsize=10)

    ax_cdf.spines["right"].set_visible(False)
    ax_cdf.spines["top"].set_visible(False)
    ax_cdf.tick_params(axis='both', labelsize=11, width=1)

    fig.subplots_adjust(hspace=0.22)
    save_plot("error_dist_broken_axis")
    plt.close()

    
def plot_performance_by_task(df):
    """Performance comparison across tasks"""
    task_stats = df.groupby(['TaskName', 'Estimator'])['Deviation'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.heatmap(task_stats, annot=True, fmt='.3f', cmap=PASTEL_CMAP_WARM, 
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean Error'})
    
    ax.set_title('Mean Error by Task and Estimator')
    ax.set_ylabel('Task')
    ax.set_xlabel('Estimator')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    save_plot('performance_by_task')

# def plot_performance_by_machine(df):
#     """Performance comparison across machines"""
#     df['Machine_Short'] = df['Machine'].str.extract(r'CPU:([^,]+)')
    
#     machine_stats = df.groupby(['Machine_Short', 'Estimator'])['Deviation'].mean().unstack()
    
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     sns.heatmap(machine_stats, annot=True, fmt='.3f', cmap=PASTEL_CMAP_BLUES, 
#                 linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean Error'})
    
#     ax.set_title('Mean Error by Machine and Estimator')
#     ax.set_ylabel('Machine')
#     ax.set_xlabel('Estimator')
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
    
#     save_plot('performance_by_machine')

def plot_performance_by_machine(df):
    """Performance comparison across machines"""
    df['Machine_Short'] = df['Machine'].str.extract(r'CPU:([^,]+)')
    
    machine_stats = df.groupby(['Machine_Short', 'Estimator'])['Deviation'].mean().unstack()
    
    # estimators = machine_stats.columns.tolist()
    estimators = ['OnlineP', 'Lotaru-A', 'Lotaru-G', 'PTS-Hybrid']
    machines = machine_stats.index.tolist()
    n_estimators = len(estimators)
    n_machines = len(machines)
    
    x = np.arange(n_machines)
    width = 0.22
    offsets = np.linspace(-(n_estimators - 1) / 2, (n_estimators - 1) / 2, n_estimators) * width
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    replaceMap = {
        'Intel(R) Xeon(R) Gold 6240R': 'Xeon 6240R',
        'Intel(R) Xeon(R) Gold 6126': 'Xeon 6126',
        'AMD EPYC 7352 24-Core Processor': 'Epyc 7352',
    }
    
    for i, (estimator, offset) in enumerate(zip(estimators, offsets)):
        values = machine_stats[estimator].values
        bars = ax.bar(x + offset, values, width=width,
                      label=estimator,
                      # color=PASTEL_COLORS[i],
                      color=wesanderson.film_palette('Moonrise Kingdom', 1)[i],
                      hatch='//' if estimator == 'PTS-Hybrid' else None,
                      # edgecolor='white' if estimator != 'PTS-Hybrid' else '#666666',
                      edgecolor='#000',
                      linewidth=0.8)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.2f}',
                    ha='center', va='bottom', rotation=0,
                    fontsize=12, color='#000', fontweight='semibold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([replaceMap.get(m, m) for m in machines], rotation=0, ha='center')
    ax.set_ylabel('Mean Error', fontsize=14)
    ax.set_xlabel(None, fontsize=14)
    # ax.set_title('Mean Error by Machine and Estimator', fontsize=16, pad=15)
    ax.legend(bbox_to_anchor=(0.5, -0.25), loc='lower center', framealpha=0.9, ncols=4, fontsize=14)
    ax.set_ylim(0, machine_stats.values.max() * 1.18)
    ax.yaxis.grid(True, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    # ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', labelsize=14, length=0)
    
    plt.tight_layout()
    save_plot('performance_by_machine')

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
    
    save_plot('error_vs_input_size')

def plot_accuracy_metrics_by_task(df):
    """Separate plots for MAE, RMSE, MAPE by task-estimator combination"""
    
    metrics = df.groupby(['TaskName', 'Estimator']).agg({
        'Absolute_Error': 'mean',
        'Squared_Error': lambda x: np.sqrt(x.mean()),
        'APE': 'mean'
    }).round(3)
    
    metrics.columns = ['MAE', 'RMSE', 'MAPE']
    
    cmap_map = {
        'MAE': PASTEL_CMAP_REDS,
        'RMSE': PASTEL_CMAP_ORANGES,
        'MAPE': PASTEL_CMAP_BLUES,
    }
    
    for metric in ['MAE', 'RMSE', 'MAPE']:
        fig, ax = plt.subplots(figsize=(8, 7))
        pivot_data = metrics[metric].unstack(fill_value=0)
        
        IGNORE=['featurecounts']
        
        # Reorder rows: pipeline order first, then any remaining tasks (RNA tools etc.) at the end
        pipeline_order = get_pipeline_order()
        ordered = [t for t in pipeline_order if t in pivot_data.index and t not in IGNORE]
        remaining = [t for t in pivot_data.index if t not in pipeline_order and t not in IGNORE]
        pivot_data = pivot_data.reindex(ordered + remaining)
        
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap=cmap_map[metric],
                linewidths=0.5, cbar_kws={'label': metric}, ax=ax)
        ax.set_title(f'{metric} by Task and Estimator')
        ax.set_ylabel('Task')
        ax.set_xlabel('Estimator')
        plt.xticks(rotation=0, ha='center')
        plt.yticks(
            ticks=[i + 0.5 for i in range(len(pivot_data.index))],
            labels=[replaceToolsWithAbbr.get(label, label) for label in pivot_data.index],
            rotation=0,
            ha='right'
        )
        
        save_plot(f'{metric.lower()}_by_task')

def get_pipeline_order():
    """Define the logical order of tools in the DNA analysis workflow"""
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

def plot_error_waterfall(task_analysis):
    """Waterfall chart showing cumulative error through pipeline stages"""
    if task_analysis is None or task_analysis.empty:
        print("No task analysis data for waterfall chart")
        return
    
    estimators = task_analysis['Estimator'].unique()
    pipeline_order = get_pipeline_order()
    available_tasks = [task for task in pipeline_order if task in task_analysis['TaskName'].values]
    
    if not available_tasks:
        return
    
    n_estimators = len(estimators)
    cols = min(2, n_estimators)
    rows = (n_estimators + cols - 1) // cols
    
    global_max = 0
    for estimator in estimators:
        est_data = task_analysis[task_analysis['Estimator'] == estimator]
        contributions = []
        for task in available_tasks:
            task_row = est_data[est_data['TaskName'] == task]
            if not task_row.empty:
                contributions.append(task_row['Weighted_Error_Contribution'].iloc[0])
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
        est_data = task_analysis[task_analysis['Estimator'] == estimator]
        
        ordered_data = []
        for task in available_tasks:
            task_row = est_data[est_data['TaskName'] == task]
            if not task_row.empty:
                ordered_data.append({
                    'task': task,
                    'error_contrib': task_row['Weighted_Error_Contribution'].iloc[0]
                })
        
        if not ordered_data:
            continue
            
        contributions = [d['error_contrib'] for d in ordered_data]
        cumulative = np.cumsum([0] + contributions)
        
        colors = [PASTEL_COLORS[i % len(PASTEL_COLORS)] for i in range(len(contributions))]
        
        bars = ax.bar(range(len(contributions)), contributions, 
                     bottom=cumulative[:-1], color=colors, alpha=0.8, 
                     edgecolor='#333333', linewidth=0.8)
        
        ax.plot(range(len(cumulative)), cumulative, 'ko-', linewidth=1.5, markersize=5)
        
        for i, (bar, contrib, cum) in enumerate(zip(bars, contributions, cumulative[1:])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                   f'+{contrib:.3f}', ha='center', va='center', fontsize=8, color='white')
            ax.text(i, cum + cum*0.01, f'{cum:.3f}', ha='center', va='bottom', fontsize=8)
        
        task_names = [d['task'].replace('-', '\n') for d in ordered_data]
        ax.set_xticks(range(len(task_names)))
        ax.set_xticklabels(task_names, rotation=45, ha='right')
        ax.set_ylabel('Error Contribution')
        ax.set_title(f'{estimator} - Error Accumulation Through Pipeline')
        ax.set_ylim(0, global_max * 1.1)
    
    for i in range(n_estimators, len(axes)):
        axes[i].set_visible(False)
    
    save_plot('error_waterfall')
    
def plot_combined_waterfall_comparison(task_analysis):
    """Combined waterfall comparison - all estimators on one plot"""
    if task_analysis is None or task_analysis.empty:
        print("No task analysis data for combined waterfall")
        return
    
    WATERFALL_IGNORED_TOOLS = ['samtools-index-markduplicate', 'samtools-sort-markduplicate']

    # estimators = task_analysis['Estimator'].unique()

    estimator_order = ["OnlineP", "Lotaru-A", "Lotaru-G", "PTS-Hybrid"]
    estimators = [
        est for est in estimator_order
        if est in task_analysis["Estimator"].unique()
    ]

    palette = wesanderson.film_palette("Moonrise Kingdom", 1)

    estimator_colors = {
        "OnlineP": palette[0],
        "Lotaru-A": palette[1],
        "Lotaru-G": palette[2],
        "PTS-Hybrid": palette[3],
    }
    
    pipeline_order = get_pipeline_order()
    available_tasks = [task for task in pipeline_order if task in task_analysis['TaskName'].values and task not in WATERFALL_IGNORED_TOOLS]

    if not available_tasks:
        return

    fig, axes = plt.subplots(1, figsize=(8, 3))

    # 1. Multi-line cumulative comparison
    # ax1 = axes[0]
    ax1 = axes
    for est_idx, estimator in enumerate(estimators):
        est_data = task_analysis[task_analysis['Estimator'] == estimator]

        contributions = []
        for task in available_tasks:
            task_row = est_data[est_data['TaskName'] == task]
            contrib = task_row['Weighted_Error_Contribution'].iloc[0] if not task_row.empty else 0
            contributions.append(contrib)

        cumulative = np.cumsum([0] + contributions)

        color = SCATTER_COLORS[est_idx % len(SCATTER_COLORS)]
        linestyle = [':', '-.', '--', '-'][est_idx % 4]
        # linewidth = max(1, 1 + est_idx * 0.1)
        linewidth = 1.6

        ax1.plot(range(len(cumulative)), cumulative, color=estimator_colors[estimator], linestyle=linestyle,
                linewidth=linewidth, marker='o', markersize=9, label=estimator, alpha=0.8)

        final_error = cumulative[-1]
        ax1.annotate(f'{final_error:.2f}', xy=(len(cumulative)-1, final_error),
                    xytext=(5, 5), textcoords='offset points', fontsize=13, color="#222")

    # ax1.set_xlabel('Pipeline Stage', fontsize=14)
    ax1.set_ylabel('Cumulative Error', fontsize=13)
    # ax1.set_title('Cumulative Error Comparison', fontsize=18)
    ax1.legend(frameon=True, loc='upper left', fontsize=12)
    ax1.set_xticks(range(len(available_tasks) + 1))
    ax1.set_xticklabels(
        ['Start'] + [replaceToolsWithAbbr.get(task, task) for task in available_tasks],
        rotation=0, ha='center'
    )
    ax1.tick_params(axis='both', labelsize=14, length=0)
    ax1.tick_params(axis='x', rotation=0)

    ax1.grid(True, alpha=0.5)
        
    # # 2. Error slope analysis
    # ax4 = axes[1]

    # for est_idx, estimator in enumerate(estimators):
    #     est_data = task_analysis[task_analysis['Estimator'] == estimator]

    #     contributions = []
    #     for task in available_tasks:
    #         task_row = est_data[est_data['TaskName'] == task]
    #         contrib = task_row['Weighted_Error_Contribution'].iloc[0] if not task_row.empty else 0
    #         contributions.append(contrib)

    #     cumulative = np.cumsum([0] + contributions)
    #     slopes = np.diff(cumulative)

    #     color = SCATTER_COLORS[est_idx % len(SCATTER_COLORS)]
    #     linestyle = [':', '-.', '--', '-'][est_idx % 4]
    #     linewidth = max(1, 1 + est_idx * 0.1)

    #     # Anchor line at (0, 0) then plot slopes at positions 1..n
    #     ax4.plot([0] + list(range(1, len(slopes) + 1)),
    #              [0] + list(slopes),
    #              color=color, marker='s', linestyle=linestyle,
    #              linewidth=linewidth, markersize=6, label=estimator, alpha=0.8)

    # ax4.set_xlabel('Pipeline Stage')
    # ax4.set_ylabel('Error Rate (Change per Stage)')
    # ax4.set_title('Error Acceleration Analysis')
    # ax4.legend(frameon=False)
    # ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    # ax4.set_xticks(range(len(available_tasks) + 1))
    # ax4.set_xticklabels(
    #     ['Start'] + [replaceToolsWithAbbr.get(task, task) for task in available_tasks],
    #     rotation=0, ha='center'
    # )

    save_plot('combined_waterfall_comparison')

    # Print insights
    print("\n=== WATERFALL COMPARISON INSIGHTS ===")

    for estimator in estimators:
        est_data = task_analysis[task_analysis['Estimator'] == estimator]
        contributions = []
        for task in available_tasks:
            task_row = est_data[est_data['TaskName'] == task]
            contrib = task_row['Weighted_Error_Contribution'].iloc[0] if not task_row.empty else 0
            contributions.append(contrib)

        if contributions:
            max_contrib_idx = np.argmax(contributions)
            worst_stage = available_tasks[max_contrib_idx]
            final_error = sum(contributions)

            print(f"\n{estimator}:")
            print(f"  - Worst error spike at: {worst_stage} ({contributions[max_contrib_idx]:.3f})")
            print(f"  - Final cumulative error: {final_error:.3f}")

def plot_enhanced_critical_path_analysis(task_analysis):
    """Enhanced critical path analysis with pipeline context"""
    if task_analysis is None or task_analysis.empty:
        return
    
    impact_matrix = task_analysis.pivot_table(
        index='TaskName', 
        columns='Estimator', 
        values='Weighted_Error_Contribution',
        fill_value=0
    )
    
    pipeline_order = get_pipeline_order()
    IGNORE = ['featurecounts']
    ordered = [t for t in pipeline_order if t in impact_matrix.index and t not in IGNORE]
    remaining = [t for t in impact_matrix.index if t not in pipeline_order and t not in IGNORE]
    impact_matrix = impact_matrix.reindex(ordered + remaining)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    
    # 1. Heatmap
    sns.heatmap(impact_matrix, annot=True, fmt='.2f', cmap=PASTEL_CMAP_REDS,
                linewidths=0.5, cbar_kws={'label': 'Weighted Error Contribution'}, ax=ax1)
    ax1.set_title('Critical Path Analysis\n(Pipeline Order Preserved)')
    ax1.set_ylabel('Task')
    ax1.set_xlabel('Estimator')
    plt.sca(ax1)
    plt.xticks(rotation=0, ha='center')
    plt.yticks(
        ticks=[i + 0.5 for i in range(len(impact_matrix.index))],
        labels=[replaceToolsWithAbbr.get(label, label) for label in impact_matrix.index],
        rotation=0,
        ha='right'
    )
    
    # 2. Critical bottlenecks bar chart
    task_criticality = task_analysis.groupby('TaskName')['Weighted_Error_Contribution'].mean()
    ordered_criticality = task_criticality.reindex([t for t in pipeline_order if t in task_criticality.index])
    
    colors = [PASTEL_COLORS[i % len(PASTEL_COLORS)] for i in range(len(ordered_criticality))]
    bars = ax2.barh(range(len(ordered_criticality)), ordered_criticality.values, 
                    color=colors, alpha=0.8, edgecolor='none')
    ax2.set_yticks(range(len(ordered_criticality)))
    ax2.set_yticklabels([replaceToolsWithAbbr.get(task, task) for task in ordered_criticality.index])
    ax2.set_xlabel('Average Error Contribution')
    ax2.set_title('Pipeline Bottlenecks\n(Ordered by Workflow Stage)')
    
    for i, (bar, val) in enumerate(zip(bars, ordered_criticality.values)):
        ax2.text(val + val*0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8)
    
    save_plot('enhanced_critical_path')

def plot_workflow_pipeline_analysis(df):
    """Combined workflow impact analysis with pipeline-focused visualizations"""
    
    dna_workflow_tasks = get_pipeline_order()
    workflow_df = df[df['TaskName'].isin(dna_workflow_tasks)].copy()

    if workflow_df.empty:
        print("Warning: No tasks found matching the DNA workflow. Available tasks:")
        print(df['TaskName'].unique())
        return None

    print(f"Workflow analysis using {len(workflow_df)} records from tasks: {workflow_df['TaskName'].unique()}")

    task_analysis = workflow_df.groupby(['TaskName', 'Estimator']).agg({
        'Real': 'mean',
        'Absolute_Error': 'mean',
        'Deviation': 'mean',
        'APE': 'mean'
    }).reset_index()

    for estimator in task_analysis['Estimator'].unique():
        est_data = task_analysis[task_analysis['Estimator'] == estimator].copy()
        total_workflow_time = est_data['Real'].sum()
        task_analysis.loc[task_analysis['Estimator'] == estimator, 'Time_Weight'] = \
            est_data['Real'] / total_workflow_time * 100

    task_analysis['Weighted_Error_Contribution'] = (
        task_analysis['Absolute_Error'] * task_analysis['Time_Weight'] / 100
    )

    print("\nGenerating Error Waterfall Chart...")
    plot_error_waterfall(task_analysis)

    print("Generating Enhanced Critical Path Analysis...")
    plot_enhanced_critical_path_analysis(task_analysis)

    print("Generating Combined Waterfall Comparison...")
    plot_combined_waterfall_comparison(task_analysis)

    return task_analysis

def print_summary_stats(df):
    """Print summary statistics"""
    print("\n=== SUMMARY STATISTICS ===")
    
    summary = df.groupby('Estimator')['Deviation'].agg(['count', 'mean', 'std', 'min', 'max'])
    print("\nPerformance by Estimator:")
    print(summary.round(4))
    
    best_estimator = summary['mean'].idxmin()
    print(f"\nBest performing estimator (lowest mean error): {best_estimator}")
    
    print("\nTasks with highest mean error:")
    task_errors = df.groupby('TaskName')['Deviation'].mean().sort_values(ascending=False)
    print(task_errors.head().round(4))
    
    df['Machine_Short'] = df['Machine'].str.extract(r'CPU:([^,]+)')
    print("\nMachines with highest mean error:")
    machine_errors = df.groupby('Machine_Short')['Deviation'].mean().sort_values(ascending=False)
    print(machine_errors.round(4))

def print_enhanced_summary_stats(df, task_analysis):
    """Print enhanced summary statistics including workflow-level metrics"""
    print("\n=== ENHANCED WORKFLOW ANALYSIS ===")
    
    if task_analysis is None:
        print("No workflow analysis data available")
        return
    
    print("\nWorkflow Performance Summary by Estimator:")
    workflow_summary = task_analysis.groupby('Estimator').agg({
        'Weighted_Error_Contribution': 'sum',
        'Time_Weight': 'sum',
        'Absolute_Error': 'mean',
        'Real': 'sum'
    }).round(4)
    workflow_summary.columns = ['Total_Workflow_Impact', 'Total_Time_Weight', 'Avg_Task_Error', 'Total_Duration']
    print(workflow_summary)
    
    print("\nTask Criticality Analysis (by duration and error impact):")
    task_criticality = task_analysis.groupby('TaskName').agg({
        'Time_Weight': 'mean',
        'Absolute_Error': 'mean',
        'Weighted_Error_Contribution': 'mean',
        'Real': 'mean'
    }).round(4)
    task_criticality = task_criticality.sort_values('Weighted_Error_Contribution', ascending=False)
    print(task_criticality)

def main():
    df = load_all_results()
    if df is None:
        return
    
    df = calculate_accuracy_metrics(df)
    
    print(f"Loaded {len(df)} predictions from {df['Source_File'].nunique()} files")
    print(f"Estimators: {', '.join(df['Estimator'].unique())}")
    print(f"Tasks: {', '.join(df['TaskName'].unique())}")
    print(f"Workflows: {', '.join(df['Workflow'].unique())}")
    
    print("\nGenerating visualizations...")
    # plot_predicted_vs_actual(df)
    plot_error_distribution(df)
    # plot_performance_by_task(df)
    plot_performance_by_machine(df)
    # plot_input_size_vs_error(df)
    # plot_accuracy_metrics_by_task(df)
    
    workflow_metrics = plot_workflow_pipeline_analysis(df)
    
    # print_summary_stats(df)
    # print_enhanced_summary_stats(df, workflow_metrics)
    
    print("\n" + "="*60)
    print("ALL PLOTS SAVED IN PNG AND PDF FORMATS")
    print("="*60)
    print("\nPipeline-focused plots:")
    print("- error_waterfall - Cumulative error buildup through stages") 
    print("- combined_waterfall_comparison - All estimators compared in one view")
    print("- enhanced_critical_path - Pipeline bottlenecks and critical paths")
    print("- mae_by_task, rmse_by_task, mape_by_task - Accuracy metrics")

if __name__ == "__main__":
    main()