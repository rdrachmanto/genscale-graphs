#!/usr/bin/env python3
"""
Usage: python3 mem_variability_all_tools.py <path_to_metrics.db> [output_base]
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import make_interp_spline
import sys
import os

# ── CLI ───────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)

db_path = sys.argv[1]
if not os.path.exists(db_path):
    print(f"Error: '{db_path}' not found"); sys.exit(1)

input_stem   = os.path.splitext(os.path.basename(db_path))[0]
input_dir    = os.path.dirname(os.path.abspath(db_path))
default_base = os.path.join(input_dir, input_stem + '_variability')
output_base  = sys.argv[2] if len(sys.argv) >= 3 else default_base

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#FAFAFA',
    'axes.facecolor':    '#FAFAFA',
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'font.family':       'serif',
    'font.size':         11,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
})

# Carefully chosen, perceptually distinct colors
COLORS = [
    '#2176AE',   # fastqc              — steel blue
    '#E07A5F',   # fastq-cleaner       — terra cotta
    '#3D9970',   # burrows-wheeler     — muted green
    '#C0392B',   # picard-markdup      — crimson
    '#8E44AD',   # samtools-sort       — purple
    '#F39C12',   # samtools-index      — amber
    '#1ABC9C',   # gatk-base-recal     — teal
    '#2C3E50',   # gatk-apply-bqsr     — dark slate
    '#D35400',   # picard-validate     — burnt orange
    '#27AE60',   # picard-collect-wgs  — emerald
]

TOOLS = [
    'fastqc',
    'fastq-cleaner',
    'burrows-wheeler-aligner',
    'picard-markduplicate',
    'samtools-sort-markduplicate',
    'samtools-index-markduplicate',
    'gatk-base-recalibrator',
    'gatk-apply-bqsr',
    'picard-validate-sam',
    'picard-collect-wgs-metrics',
]

# ── Load ──────────────────────────────────────────────────────────────────
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT tool_name, input_size, peak_mem FROM memory", conn)
conn.close()

df = df[df['tool_name'].isin(TOOLS)]
df = df[(df['input_size'] > 0) & (df['peak_mem'] > 0)].copy()

print(f"Loaded {len(df)} records across {df['tool_name'].nunique()} tools")

if df.empty:
    print("No data after filtering."); sys.exit(1)

# ── Bin & compute CV ──────────────────────────────────────────────────────
N_BINS = 35
global_bins = np.logspace(
    np.log10(df['input_size'].min()),
    np.log10(df['input_size'].max()),
    N_BINS
)
df['size_bin'] = pd.cut(df['input_size'], bins=global_bins)

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6.5))

# Subtle background grid only on y
ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.5, linestyle='--', alpha=0.6)
ax.xaxis.grid(True, color='#CCCCCC', linewidth=0.5, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BBBBBB')
ax.spines['bottom'].set_color('#BBBBBB')

def smooth_line(x, y, n_smooth=300):
    """Cubic spline smoothing; falls back to raw if too few points."""
    if len(x) < 4:
        return x, y
    try:
        x_log = np.log10(x)
        x_new = np.linspace(x_log.min(), x_log.max(), n_smooth)
        spl   = make_interp_spline(x_log, y, k=3)
        y_new = spl(x_new)
        y_new = np.clip(y_new, 0, None)   # CV can't be negative
        return 10**x_new, y_new
    except Exception:
        return x, y

MIN_TOOLS = 5  # minimum tools with data required to show that x region

# First pass: compute CV per tool
tool_grps = {}
for tool in TOOLS:
    td = df[df['tool_name'] == tool]
    if td.empty:
        continue
    grp = td.groupby('size_bin', observed=True)['peak_mem'].agg(
        mean='mean', std='std', count='count'
    ).reset_index()
    grp = grp[grp['count'] >= 3].copy()
    grp['cv'] = (grp['std'] / grp['mean']) * 100
    if len(grp) >= 3:
        tool_grps[tool] = grp

# Find x range where >= MIN_TOOLS tools have coverage
all_bin_mids = sorted(set(
    iv.mid
    for grp in tool_grps.values()
    for iv in grp['size_bin']
))
covered = [
    m for m in all_bin_mids
    if sum(1 for grp in tool_grps.values()
           if grp['size_bin'].apply(lambda iv: abs(iv.mid - m) < 1).any()) >= MIN_TOOLS
]
if not covered:
    print("No x range has enough tool coverage."); sys.exit(1)
x_min, x_max = min(covered), max(covered)

# Second pass: plot clipped to [x_min, x_max]
plotted = 0
for i, tool in enumerate(TOOLS):
    if tool not in tool_grps:
        continue

    grp   = tool_grps[tool]
    mids  = np.array([iv.mid for iv in grp['size_bin']])
    cvs   = grp['cv'].values
    color = COLORS[i % len(COLORS)]

    xs, ys = smooth_line(mids, cvs)

    # Clip to shared coverage range
    mask     = (xs   >= x_min) & (xs   <= x_max)
    dot_mask = (mids >= x_min) & (mids <= x_max)
    if mask.sum() < 2:
        continue

    cv_std = cvs.std()
    ax.fill_between(xs[mask], np.clip(ys[mask] - cv_std * 0.4, 0, None),
                    ys[mask] + cv_std * 0.4, color=color, alpha=0.07)

    ax.plot(xs[mask], ys[mask],
            color=color, linewidth=2.0, alpha=0.92,
            label=tool, zorder=3 + i)

    ax.scatter(mids[dot_mask], cvs[dot_mask],
               color=color, s=22, zorder=10, alpha=0.7,
               edgecolors='white', linewidths=0.6)

    plotted += 1

ax.set_xlim(x_min * 0.98, x_max * 1.02)

# Readable x-axis tick labels
def fmt_mb(x, _):
    if x >= 1e3:  return f'{x/1e3:.1f}GB'
    elif x >= 1:  return f'{x:.0f}MB'
    else:         return f'{x*1e3:.0f}KB'
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_mb))
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, prune='both'))
plt.xticks(rotation=0, ha='center', fontsize=9)

print(f"Plotted {plotted} tools")

# ── Threshold line ────────────────────────────────────────────────────────
xlim = ax.get_xlim()
ax.axhline(20, color='#C0392B', linestyle=(0, (6, 4)), linewidth=1.1,
           alpha=0.75, zorder=2)
ax.text(10**((np.log10(global_bins[0]) + np.log10(global_bins[-1])) / 2),
        20.8, 'High variability threshold (CV = 20%)',
        color='#C0392B', fontsize=8, ha='center', va='bottom', style='italic')

# ── Axes labels & title ───────────────────────────────────────────────────
ax.set_xlabel('Input Size, Single File (MB)', fontsize=11, labelpad=10, color='#333333')
ax.set_ylabel('Coefficient of Variation (%)', fontsize=11, labelpad=10, color='#333333')
ax.set_title('Peak Memory Variability by Input Size',
             fontsize=14, fontweight='normal', pad=16, color='#1A1A1A',
             loc='left')
ax.text(0, 1.01,
        'Coefficient of Variation (CV%) per log-spaced input size bin (MB) · 10 bioinformatics tools',
        transform=ax.transAxes, fontsize=8.5, color='#777777', va='bottom')

# ── Legend ────────────────────────────────────────────────────────────────
leg = ax.legend(
    loc='upper right',
    fontsize=8.5,
    frameon=True,
    facecolor='#FAFAFA',
    edgecolor='#DDDDDD',
    framealpha=0.95,
    borderpad=0.9,
    labelspacing=0.45,
    handlelength=1.6,
    title='Tool',
    title_fontsize=9,
    ncol=2,
)
leg.get_title().set_color('#444444')

plt.tight_layout(pad=1.8)

# ── Save ──────────────────────────────────────────────────────────────────
out_pdf = output_base + '.pdf'
out_png = output_base + '.png'

with pdf_backend.PdfPages(out_pdf) as pdf:
    pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
    d = pdf.infodict()
    d['Title']   = 'Peak Memory Variability by Input Size'
    d['Subject'] = 'CV% by log-spaced input size bin per bioinformatics tool'

plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")