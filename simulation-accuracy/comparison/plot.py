#!/usr/bin/env python3
"""
Cluster Sizing Accuracy — Normalized View
==========================================
Actual makespan is normalized to 1.0 (horizontal baseline).
Bars show simulated/actual ratio; values above 1 = over-prediction,
values below 1 = under-prediction.

Usage:
  # Single file
  python plot_accuracy_normalized.py results.json
  python plot_accuracy_normalized.py results.json -o my_plot

  # Dual file comparison (side-by-side bars, common config IDs)
  python plot_accuracy_normalized.py file1.json file2.json --labels "BB" "BO"
  python plot_accuracy_normalized.py file1.json file2.json --labels "BB" "BO" -o compare

Outputs: <base>.pdf  +  <base>.svg   (no PNG)
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── rcParams ──────────────────────────────────────────────────────────────────
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

# ── palette ───────────────────────────────────────────────────────────────────
C_GOOD    = '#2A9D5C'   # |err| ≤ 10%
C_BAD     = '#C0392B'   # |err| >  10%
C_BASE_A  = '#5FB0DA'   # dataset A bar fill
C_BASE_B  = '#E3A965'   # dataset B bar fill
C_ACTUAL  = '#111111'   # baseline (actual = 1.0)
C_BAND    = '#2A9D5C'   # ±10% tolerance band

THRESHOLD = 0.10        # ±10 %

# ── helpers ───────────────────────────────────────────────────────────────────

def load_evals(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {
        e["config_id"]: e
        for e in data["evaluations"]
        if e["status"] == "completed"
        and e["actual_makespan_min"] is not None
        and e["simulated_makespan_min"] is not None
    }


def ratio(sim, act):
    """simulated / actual  (actual = 1.0 baseline)"""
    return sim / act


def err_color(r, c_good=C_GOOD, c_bad=C_BAD):
    return c_good if abs(r - 1.0) <= THRESHOLD else c_bad


# ── draw one normalized bar ───────────────────────────────────────────────────

def draw_norm_bar(ax, xi, bw, sim, act, c_fill, c_good=C_GOOD, c_bad=C_BAD):
    r      = ratio(sim, act)
    ec     = err_color(r, c_good, c_bad)
    height = r - 1.0          # positive = over-predict, negative = under-predict

    # bar from 1.0 (= y=0 in transformed coords, but we work in ratio space)
    ax.bar(xi, height, bw,
           bottom=1.0,
           color=c_fill, alpha=0.80,
           edgecolor='#333333', linewidth=0.7,
           zorder=3)

    # hatch overlay using error colour
    hatch = '////' if height >= 0 else r'\\\\'
    ax.bar(xi, height, bw,
           bottom=1.0,
           color='none', edgecolor=ec,
           hatch=hatch, linewidth=0, alpha=0.80,
           zorder=4)

    # percentage label
    pct = (r - 1.0) * 100
    label = f'+{pct:.1f}%' if pct >= 0 else f'{pct:.1f}%'

    label_y  = r + 0.012  if height >= 0 else r - 0.012
    label_va = 'bottom'   if height >= 0 else 'top'

    ax.text(xi, label_y, label,
            ha='center', va=label_va,
            fontsize=14, color=ec, fontweight='bold', zorder=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75,
                      boxstyle='round,pad=0.15'))


# ── shared axis finish ────────────────────────────────────────────────────────

def finish_norm_axes(ax, config_ids, all_ratios, title, legend_handles):
    # ±10% tolerance band
    ax.axhspan(1.0 - THRESHOLD, 1.0 + THRESHOLD,
               color=C_BAND, alpha=0.08, zorder=1, label='_nolegend_')
    ax.axhline(1.0 - THRESHOLD, color=C_BAND,
               linewidth=0.9, linestyle=':', alpha=0.55, zorder=2)
    ax.axhline(1.0 + THRESHOLD, color=C_BAND,
               linewidth=0.9, linestyle=':', alpha=0.55, zorder=2)

    # actual = 1.0 baseline
    ax.axhline(1.0, color=C_ACTUAL, linewidth=1.6, zorder=6, alpha=0.85)

    # grid
    ax.yaxis.grid(True, alpha=0.6, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # y limits — give some breathing room beyond the tallest bar
    pad   = 0.08
    y_lo  = min(min(all_ratios) - pad, 1.0 - THRESHOLD - pad)
    y_hi  = max(max(all_ratios) + pad, 1.0 + THRESHOLD + pad)
    ax.set_ylim(y_lo, y_hi)

    # y-axis: show as percentage deviation from actual
    def fmt_ratio(val, _):
        pct = (val - 1.0) * 100
        return f'+{pct:.0f}%' if pct > 0 else f'{pct:.0f}%'

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_ratio))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.10))

    ax.set_xticks(range(len(config_ids)))
    ax.set_xticklabels(config_ids, fontsize=6)
    ax.set_ylabel('Simulated / Actual  (actual = 0%)', fontsize=15)
    ax.set_xlabel('Configuration ID', fontsize=15)
    # ax.set_title(title, fontsize=13, fontweight='normal', pad=14)

    ax.tick_params(axis='both', length=0, labelsize=15)
    
    # annotations for band edges
    n = len(config_ids)
    for yv, txt in [(1.0 + THRESHOLD, '+10%'), (1.0 - THRESHOLD, '−10%')]:
        ax.text(n - 0.5, yv, txt,
                ha='right', va='bottom' if yv > 1 else 'top',
                fontsize=7, color=C_BAND, alpha=0.7)

    fig = ax.get_figure()
    fig.legend(handles=legend_handles, fontsize=15, frameon=False,
               loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.05),
               facecolor='white', edgecolor='#cccccc', framealpha=0.92,
               borderpad=0.7, labelspacing=0.5, columnspacing=1.2,
               handlelength=1.4)


# ── single-file normalized plot ───────────────────────────────────────────────

def plot_single(evals: dict) -> plt.Figure:
    ids  = sorted(evals.keys())
    sims = [evals[c]['simulated_makespan_min'] for c in ids]
    acts = [evals[c]['actual_makespan_min']    for c in ids]
    rats = [ratio(s, a) for s, a in zip(sims, acts)]
    n    = len(ids)

    fig, ax = plt.subplots(figsize=(8, 4))

    bw = 0.5
    for i, (s, a) in enumerate(zip(sims, acts)):
        draw_norm_bar(ax, i, bw, s, a, C_BASE_A)

    finish_norm_axes(ax, [f"C{c}" for c in ids], rats,
                     'Cluster Sizing Accuracy: GenScale Simulation vs Actual (Normalized)',
                     [
        mpatches.Patch(facecolor=C_ACTUAL, label='Actual makespan  (baseline = 0%)'),
        mpatches.Patch(facecolor=C_BASE_A, alpha=0.80, edgecolor='#333333',
                       linewidth=0.7, label='Simulated makespan  (bar = deviation)'),
        mpatches.Patch(facecolor=C_GOOD, alpha=0.5, hatch='////',
                       edgecolor=C_GOOD, label=None),
        mpatches.Patch(facecolor=C_BAD,  alpha=0.5, hatch='////',
                       edgecolor=C_BAD,  label=None),
        mpatches.Patch(facecolor=C_BAND, alpha=0.15,
                       label='±10% tolerance band'),
    ])
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ── dual-file normalized plot ─────────────────────────────────────────────────

def plot_dual(evals_a: dict, evals_b: dict,
              label_a: str, label_b: str) -> plt.Figure:
    common = sorted(set(evals_a) & set(evals_b))
    if not common:
        sys.exit("ERROR: No common config IDs found between the two files.")

    n   = len(common)
    bw  = 0.36
    gap = 0.06

    rats_a = [ratio(evals_a[c]['simulated_makespan_min'],
                    evals_a[c]['actual_makespan_min']) for c in common]
    rats_b = [ratio(evals_b[c]['simulated_makespan_min'],
                    evals_b[c]['actual_makespan_min']) for c in common]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, cid in enumerate(common):
        xa = i - (bw / 2 + gap / 2)
        xb = i + (bw / 2 + gap / 2)
        draw_norm_bar(ax, xa, bw,
                      evals_a[cid]['simulated_makespan_min'],
                      evals_a[cid]['actual_makespan_min'],
                      C_BASE_A)
        draw_norm_bar(ax, xb, bw,
                      evals_b[cid]['simulated_makespan_min'],
                      evals_b[cid]['actual_makespan_min'],
                      C_BASE_B)

    xlabels = [f"C{c}" for c in range(1, 10)]
    
    finish_norm_axes(ax, xlabels, rats_a + rats_b,
                     f'Cluster Simulation Accuracy — {label_a} vs {label_b} (Normalized)',
                     [
        # mpatches.Patch(facecolor=C_ACTUAL, label='Actual makespan  (baseline = 0%)'),
        mpatches.Patch(facecolor=C_BASE_A, alpha=0.80, edgecolor='#333333',
                       linewidth=0.7, label=f'{label_a}'),
        mpatches.Patch(facecolor=C_BASE_B, alpha=0.80, edgecolor='#333333',
                       linewidth=0.7, label=f'{label_b}'),
        # mpatches.Patch(facecolor=C_GOOD, alpha=0.5, hatch='////',
        #                edgecolor=C_GOOD, label=None),
        # mpatches.Patch(facecolor=C_BAD,  alpha=0.5, hatch='////',
        #                edgecolor=C_BAD,  label=None),
        mpatches.Patch(facecolor=C_BAND, alpha=0.15,
                       label='±10% tolerance band'),
    ])
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ── save PDF + SVG ────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, base: str):
    for ext, kwargs in [
        ('.pdf', dict(format='pdf')),
        ('.svg', dict(format='svg')),
    ]:
        path = base + ext
        fig.savefig(path, bbox_inches='tight', **kwargs)
        print(f"Saved: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot GenScale cluster sizing accuracy (normalized view).")
    parser.add_argument("files", nargs="+", metavar="FILE",
                        help="One or two JSON result files.")
    parser.add_argument("--labels", nargs=2, default=["File A", "File B"],
                        metavar=("LABEL_A", "LABEL_B"),
                        help="Labels for dual-file mode.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output base path (no extension). "
                             "Defaults to same dir as first input, named after it.")
    args = parser.parse_args()

    if args.output:
        out_base = args.output
    else:
        stem     = os.path.splitext(os.path.basename(args.files[0]))[0]
        out_dir  = os.path.dirname(os.path.abspath(args.files[0]))
        out_base = os.path.join(out_dir, stem + "_normalized")

    if len(args.files) == 1:
        fig = plot_single(load_evals(args.files[0]))
    elif len(args.files) == 2:
        fig = plot_dual(load_evals(args.files[0]),
                        load_evals(args.files[1]),
                        args.labels[0], args.labels[1])
    else:
        parser.error("Provide 1 or 2 JSON files.")

    save_fig(fig, out_base)


if __name__ == "__main__":
    main()