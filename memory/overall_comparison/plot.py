import pandas as pd
import matplotlib.pyplot as plt
import io
from matplotlib.ticker import FuncFormatter
import numpy as np

try:
    from adjustText import adjust_text
except ImportError:
    print("Please install adjustText for optimal label placement: pip install adjustText")
    adjust_text = None

# ============================================================================
# GLOBAL STYLE CONFIGURATION FOR RESEARCH PAPER
# ============================================================================

# Color families: warm tones for baselines, cool tones for FAMR
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
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
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
    'hatch.linewidth': 0.8,
})

# ============================================================================
# DATA PREPARATION
# ============================================================================

csv_data = """method_name,total_predictions,failure_rate,total_wastage_mb,avg_wastage_mb,total_exec_time_ms,total_retry_time_s,total_time_with_penalties_s,efficiency_score
Sizey,2000,0.0,3721957.5004305453,1860.9787502152726,1893763,242160,2135923,0.886625126467574
WittPercentile,2000,0.0,4182267.750000001,2091.1338750000004,1893763,70840,1964603,0.9639418243787676
WittRegression,2000,0.0,6824956.358246249,3412.4781791231244,1893763,105220,1998983,0.947363234204593
Statistical_P95,2000,0.0,3928676.650000001,1964.3383250000004,1893763,77613,1971376,0.9606300370908442
FAMR_1,2000,0.075,2335433,1167.7165,1893763,93816,1987579,0.9527988573032821
FAMR_1.15,2000,0.075,2450450,1225.225,1893763,66846,1960609,0.9659054916100048
FAMR_2,2000,0.075,3187698,1593.849,1893763,9363,1903126,0.9950801996294517
"""

df = pd.read_csv(io.StringIO(csv_data))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_plot(filename):
    """Save plot in both PNG and PDF formats"""
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def style_bars(ax, bars, methods):
    """Apply FAMR hatching and bold edges to bars"""
    for bar, method in zip(bars, methods):
        if is_famr(method):
            bar.set_hatch(FAMR_HATCH)
            bar.set_edgecolor(FAMR_HATCH_COLOR)
            bar.set_linewidth(0.8)
        else:
            bar.set_edgecolor('none')

def add_bar_labels(ax, bars, values, methods, max_val, direction='horizontal', fmt='{:,.0f}'):
    """Add value labels to bars - always outside for FAMR (hatching obscures text)"""
    for bar, val, method in zip(bars, values, methods):
        label = fmt.format(val)
        if direction == 'horizontal':
            if is_famr(method):
                # Always place outside for hatched bars
                ax.text(val + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                        label, va='center', ha='left', fontsize=8, color='#333333')
            elif val > max_val * 0.3:
                ax.text(val - max_val * 0.02, bar.get_y() + bar.get_height()/2,
                        label, va='center', ha='right', fontsize=8, color='white')
            else:
                ax.text(val + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                        label, va='center', ha='left', fontsize=8, color='#333333')

def add_famr_baseline_legend(ax, loc='upper right'):
    """Add a legend distinguishing FAMR from baseline methods"""
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#5FB0DA', edgecolor='#2a6e96', hatch=FAMR_HATCH, label='FAMR'),
        Patch(facecolor='#F28C52', edgecolor='none', label='Baseline'),
    ]
    ax.legend(handles=legend_elements, loc=loc, frameon=False, fontsize=8)

# ============================================================================
# FIGURE 1: MEMORY WASTAGE HISTOGRAM
# ============================================================================

def plot_memory_wastage(df, ax=None, standalone=True):
    df_sorted = df.sort_values('total_wastage_mb', ascending=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(
        y=range(len(df_sorted)),
        width=df_sorted['total_wastage_mb'],
        color=[get_method_color(m) for m in df_sorted['method_name']],
        height=0.7
    )
    style_bars(ax, bars, df_sorted['method_name'].values)

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['method_name'])
    ax.set_title('Total Memory Wastage by Method', pad=10)
    ax.set_xlabel('Total Memory Wasted (MB)')

    add_bar_labels(ax, bars, df_sorted['total_wastage_mb'].values,
                   df_sorted['method_name'].values,
                   df_sorted['total_wastage_mb'].max())

    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    ax.set_xlim(0, df_sorted['total_wastage_mb'].max() * 1.12)

    if standalone:
        save_plot('memory_wastage_histogram')

# ============================================================================
# FIGURE 2: TIME WASTAGE HISTOGRAM
# ============================================================================

def plot_time_wastage(df, ax=None, standalone=True):
    df_sorted = df.sort_values('total_retry_time_s', ascending=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(
        y=range(len(df_sorted)),
        width=df_sorted['total_retry_time_s'],
        color=[get_method_color(m) for m in df_sorted['method_name']],
        height=0.7
    )
    style_bars(ax, bars, df_sorted['method_name'].values)

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['method_name'])
    ax.set_title('Total Retry Time by Method', pad=10)
    ax.set_xlabel('Total Retry Time (seconds)')

    add_bar_labels(ax, bars, df_sorted['total_retry_time_s'].values,
                   df_sorted['method_name'].values,
                   df_sorted['total_retry_time_s'].max())

    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, p: f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'))
    ax.set_xlim(0, df_sorted['total_retry_time_s'].max() * 1.12)

    if standalone:
        save_plot('time_wastage_histogram')

# ============================================================================
# FIGURE 3: SCATTER PLOT - MEMORY VS TIME
# ============================================================================

def plot_memory_vs_time(df, ax=None, standalone=True, x_split=None, y_split=None):
    # if ax is None:
    #     fig, ax = plt.subplots(figsize=(6, 5)) # Old: (7, 5)

    # for method in df['method_name'].unique():
    #     row = df[df['method_name'] == method].iloc[0]
    #     color = get_method_color(method)
    #     edgecolor = '#2a6e96' if is_famr(method) else '#333333'
    #     size = 80

    #     ax.scatter(
    #         row['total_retry_time_s'],
    #         row['total_wastage_mb'],
    #         s=size, color=color, edgecolor=edgecolor,
    #         linewidth=1.0 if is_famr(method) else 0.5,
    #         alpha=0.85, label=method, zorder=3, marker='o'
    #     )

    # ax.set_title('Memory Wastage vs. Time Wastage', pad=10, fontsize=18)
    # ax.set_xlabel('Total Retry Time (seconds)', fontsize=16)
    # ax.set_ylabel('Total Memory Wasted (MB)', fontsize=16)
    # ax.set_yscale('log')

    # ax.xaxis.set_major_formatter(FuncFormatter(
    #     lambda x, p: f'{int(x/1000)}K' if x >= 1000 else f'{int(x)}'))
    # ax.yaxis.set_major_formatter(FuncFormatter(
    #     lambda y, p: f'{y/1e6:.1f}M' if y >= 1e6 else f'{y/1e3:.0f}K'))

    # # Annotate key outliers
    # min_time_idx = df['total_retry_time_s'].idxmin()
    # max_time_idx = df['total_retry_time_s'].idxmax()
    # min_mem_idx = df['total_wastage_mb'].idxmin()
    # max_mem_idx = df['total_wastage_mb'].idxmax()
    # annotate_indices = list(set([min_time_idx, max_time_idx, min_mem_idx, max_mem_idx]))

    # for idx in annotate_indices:
    #     row = df.loc[idx]
    #     ax.annotate(
    #         row['method_name'],
    #         xy=(row['total_retry_time_s'], row['total_wastage_mb']),
    #         xytext=(10, 10), textcoords='offset points', fontsize=8, color='#333333',
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
    #                   edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5),
    #         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
    #                        color='#666666', lw=0.5, alpha=0.6)
    #     )

    # # Per-method legend
    # from matplotlib.lines import Line2D
    # legend_elements = []
    # for m in df['method_name'].unique():
    #     legend_elements.append(Line2D([0], [0], marker='o', color='w',
    #         markerfacecolor=get_method_color(m),
    #         markeredgecolor='#2a6e96' if is_famr(m) else '#333333',
    #         markersize=7, linewidth=0, label=m))

    # ax.tick_params(axis='both', labelsize=16)
    # ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=10,
    #           handletextpad=0.4, labelspacing=0.3)

    # if standalone:
    #     save_plot('memory_vs_time_scatter')

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    df_plot = df.copy()

    # Convert units once and use these everywhere
    df_plot['total_retry_time_h'] = df_plot['total_retry_time_s'] / 3600
    df_plot['total_wastage_tb'] = df_plot['total_wastage_mb'] / 1e6

    marker_map = {
        'Sizey': 'o',
        'WittPercentile': 's',
        'WittRegression': 'D',
        'Statistical_P95': '^',
        'FAMR_1': 'P',
        'FAMR_1.15': 'X',
        'FAMR_2': 'v',
    }

    fallback_markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', '<', '>']

    label_offsets = {
        'Sizey': (10, 8),
        'WittPercentile': (8, 10),
        'WittRegression': (10, 8),
        'Statistical_P95': (8, -12),
        'FAMR_1': (10, -3),
        'FAMR_1.15': (8, 14),
        'FAMR_2': (10, 8),
    }

    methods = df_plot['method_name'].unique()

    # ----------------------------
    # Quadrant split
    # ----------------------------
    if x_split is None:
        x_split = df_plot['total_retry_time_h'].median()
    else:
        # If passed in seconds, convert to hours
        x_split = x_split / 3600

    if y_split is None:
        y_split = df_plot['total_wastage_tb'].median()
    else:
        # If passed in MB, convert to TB
        y_split = y_split / 1e6

    x_min = 0
    x_max = df_plot['total_retry_time_h'].max() * 1.08
    y_min = df_plot['total_wastage_tb'].min() * 0.92
    y_max = df_plot['total_wastage_tb'].max() * 1.08

    # ----------------------------
    # Highlight lower-left best region
    # ----------------------------
    ax.axvspan(
        x_min,
        x_split,
        color='green',
        alpha=0.05,
        zorder=0
    )

    ax.axhspan(
        y_min,
        y_split,
        xmin=0,
        xmax=(x_split - x_min) / (x_max - x_min),
        color='green',
        alpha=0.10,
        zorder=0
    )

    # Quadrant divider lines
    ax.axvline(
        x_split,
        color='red',
        linestyle='--',
        linewidth=1.2,
        zorder=1
    )

    ax.axhline(
        y_split,
        color='red',
        linestyle='--',
        linewidth=1.2,
        zorder=1
    )

    # ----------------------------
    # Annotate divider lines
    # ----------------------------
    ax.text(
        x_split + 0.015 * (x_max - x_min),
        y_max - 0.03 * (y_max - y_min),
        f'Low Retry Time',
        rotation=90,
        fontsize=12,
        color='red',
        ha='left',
        va='top',
        bbox=dict(
            boxstyle='round,pad=0.2',
            facecolor='white',
            edgecolor='none',
            alpha=0.8
        )
    )

    ax.text(
        x_max - 0.10 * (x_max - x_min),
        y_split + 0.015 * (y_max - y_min),
        f'Low Wasted Memory',
        fontsize=12,
        color='red',
        ha='right',
        va='bottom',
        bbox=dict(
            boxstyle='round,pad=0.2',
            facecolor='white',
            edgecolor='none',
            alpha=0.8
        )
    )

    # ----------------------------
    # Plot points and direct labels
    # ----------------------------
    for i, method in enumerate(methods):
        row = df_plot[df_plot['method_name'] == method].iloc[0]

        x = row['total_retry_time_h']
        y = row['total_wastage_tb']

        color = get_method_color(method)
        edgecolor = '#2a6e96' if is_famr(method) else '#333333'
        marker = marker_map.get(method, fallback_markers[i % len(fallback_markers)])

        ax.scatter(
            x,
            y,
            s=80,
            color=color,
            edgecolor=edgecolor,
            linewidth=1.0 if is_famr(method) else 0.5,
            alpha=0.85,
            zorder=3,
            marker=marker
        )

        dx, dy = label_offsets.get(method, (10, 8))

        ax.annotate(
            method,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=12,
            color='#333333',
            ha='left',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.22',
                facecolor='white',
                edgecolor='#CCCCCC',
                alpha=0.85,
                linewidth=0.6
            ),
            zorder=4
        )

    # ----------------------------
    # Axis labels and formatting
    # ----------------------------
    ax.set_xlabel('Total Retry Time (Hours)', fontsize=13)
    ax.set_ylabel('Total Memory Wasted (TB)', fontsize=13)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f'{x:.0f}')
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, p: f'{y:.1f}')
    )

    ax.tick_params(axis='both', labelsize=14)

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if standalone:
        save_plot('memory_vs_time_scatter')

# ============================================================================
# FIGURE 4: EFFICIENCY SCORE COMPARISON
# ============================================================================

def plot_efficiency_score(df, ax=None, standalone=True):
    df_sorted = df.sort_values('efficiency_score', ascending=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(
        y=range(len(df_sorted)),
        width=df_sorted['efficiency_score'],
        color=[get_method_color(m) for m in df_sorted['method_name']],
        height=0.7
    )
    style_bars(ax, bars, df_sorted['method_name'].values)

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['method_name'])
    ax.set_title('Efficiency Score by Method', pad=10)
    ax.set_xlabel('Efficiency Score')

    for bar, (idx, row) in zip(bars, df_sorted.iterrows()):
        val = row['efficiency_score']
        method = row['method_name']
        if is_famr(method):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', ha='left', fontsize=8, color='#333333')
        else:
            ax.text(val - 0.008, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', ha='right', fontsize=8, color='white')

    ax.set_xlim(0.85, 1.0)

    if standalone:
        save_plot('efficiency_score_comparison')

# ============================================================================
# COMBINED 2x2 FIGURE
# ============================================================================

def plot_combined(df):
    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    fig.suptitle('Resource Efficiency Analysis', fontsize=13, fontweight='normal', y=0.98)

    plot_memory_wastage(df, ax=axes[0], standalone=False)
    plot_time_wastage(df, ax=axes[1], standalone=False)
    plot_memory_vs_time(df, ax=axes[2], standalone=False)
    plot_efficiency_score(df, ax=axes[3], standalone=False)

    # Add subplot labels (a), (b), (c), (d)
    for i, (ax, label) in enumerate(zip(axes.flatten(), ['(a)', '(b)', '(c)', '(d)'])):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

    save_plot('combined_resource_analysis')

# ============================================================================
# MAIN
# ============================================================================

# Individual plots
plot_memory_wastage(df)
plot_time_wastage(df)
plot_memory_vs_time(df)
plot_efficiency_score(df)

# Combined 2x2
plot_combined(df)

print("✓ All plots saved in PNG and PDF formats!")
print("  Individual plots:")
print("    - memory_wastage_histogram.{png,pdf}")
print("    - time_wastage_histogram.{png,pdf}")
print("    - memory_vs_time_scatter.{png,pdf}")
print("    - efficiency_score_comparison.{png,pdf}")
print("  Combined:")
print("    - combined_resource_analysis.{png,pdf}")