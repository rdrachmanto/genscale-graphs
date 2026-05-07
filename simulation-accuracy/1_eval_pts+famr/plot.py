#!/usr/bin/env python3
"""
Usage: python3 prediction_histogram.py <path_to_opt.json> [output_path]

  <path_to_opt.json>  : required — path to the optimization results JSON
  [output_path]       : optional — output file path without extension
                        defaults to same directory as input JSON,
                        named after the input file (e.g. opt.pdf / opt.png)
"""

import sys
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── CLI args ──────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

json_path = sys.argv[1]
if not os.path.exists(json_path):
    print(f"Error: file not found: {json_path}")
    sys.exit(1)

# Output base path: same dir as input, stem of input filename
input_stem   = os.path.splitext(os.path.basename(json_path))[0]
input_dir    = os.path.dirname(os.path.abspath(json_path))
default_base = os.path.join(input_dir, input_stem)
output_base  = sys.argv[2] if len(sys.argv) >= 3 else default_base

# ── Load JSON ─────────────────────────────────────────────────────────────
with open(json_path) as f:
    data = json.load(f)

evaluations = [
    e for e in data["evaluations"]
    if e["status"] == "completed"
    and e["simulated_makespan_min"] is not None
    and e["actual_makespan_min"] is not None
]

if not evaluations:
    print("No completed evaluations found in JSON.")
    sys.exit(1)

config_ids = [f"C{e['config_id']}" for e in evaluations]
simulated  = np.array([e['simulated_makespan_min'] for e in evaluations])
actual     = np.array([e['actual_makespan_min']    for e in evaluations])

print(f"Loaded {len(evaluations)} completed configs from '{json_path}'")

# ── Styling from reference ────────────────────────────────────────────────
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

C_BASE = '#2176AE'   # deep blue — high contrast bar
C_GOOD = '#2A9D5C'   # strong green — |err| <= 10%
C_BAD  = '#C0392B'   # strong red   — |err| >  10%
C_DASH = '#333333'
THRESHOLD = 10.0

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(12, len(config_ids) * 0.8), 7))

x  = np.arange(len(config_ids))
bw = 0.6

for i, (sim, act) in enumerate(zip(simulated, actual)):
    error     = act - sim
    error_pct = error / sim * 100
    hcolor    = C_BAD if abs(error_pct) > THRESHOLD else C_GOOD

    ax.bar(x[i], act, bw, color=C_BASE, alpha=0.85, zorder=3,
           edgecolor='#333333', linewidth=0.8)

    if abs(error) < 0.01:
        ax.text(x[i], act + 0.25, '~0%', ha='center', va='bottom',
                fontsize=7, color='#333333', zorder=7)
        continue

    if error > 0:
        hatch_bottom = sim
        hatch_height = error
        hatch_pat    = '////'
    else:
        hatch_bottom = act
        hatch_height = -error
        hatch_pat    = r'\\\\'

    ax.bar(x[i], hatch_height, bw, bottom=hatch_bottom,
           color=hcolor, alpha=0.25, zorder=4, edgecolor='none')
    ax.bar(x[i], hatch_height, bw, bottom=hatch_bottom,
           color='none', edgecolor=hcolor, hatch=hatch_pat,
           linewidth=0, alpha=0.9, zorder=5)
    ax.plot([x[i] - bw/2, x[i] + bw/2], [sim, sim],
            color=C_DASH, linewidth=1.4, linestyle='--', zorder=6, alpha=0.7)

    label = f'+{error_pct:.1f}%' if error > 0 else f'{error_pct:.1f}%'
    if error > 0:
        # underestimate — label above bar top, clear of hatch
        label_y  = act + 0.3
        label_va = 'bottom'
    else:
        # overestimate — label just below bar top, sitting inside the hatch
        label_y  = act - 0.3
        label_va = 'top'
    ax.text(x[i], label_y, label,
            ha='center', va=label_va,
            fontsize=7.5, color=hcolor, fontweight='bold', zorder=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75,
                      boxstyle='round,pad=0.15'))

ax.yaxis.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

y_min = max(0, actual.min() - 5)
y_max = actual.max() + 5
ax.set_ylim(y_min, y_max)
ax.set_xticks(x)
ax.set_xticklabels(config_ids, fontsize=9)
ax.set_ylabel('Makespan (minutes)', fontsize=10)
ax.set_xlabel('Configuration ID', fontsize=10)
ax.set_title('Cluster Sizing Accuracy: GenScale Simulation vs Actual',
             fontsize=13, fontweight='normal', pad=14)

legend_elements = [
    mpatches.Patch(facecolor=C_BASE, alpha=0.85, edgecolor='#333333', linewidth=0.8,
                   label='Actual makespan  (bar top = actual)'),
    mpatches.Patch(facecolor=C_GOOD, alpha=0.5, hatch='////',
                   edgecolor=C_GOOD, label='|Error| ≤ 10%  — under error threshold'),
    mpatches.Patch(facecolor=C_BAD,  alpha=0.5, hatch='////',
                   edgecolor=C_BAD,  label='|Error| > 10%  — over error threshold'),
    mpatches.Patch(facecolor='none', edgecolor=C_DASH,
                   linestyle='--', linewidth=1.4, label='Simulated makespan  (dashed line)'),
]
ax.legend(handles=legend_elements, fontsize=8, frameon=True, loc='lower right',
          facecolor='white', edgecolor='#cccccc', framealpha=0.92,
          borderpad=0.8, labelspacing=0.5)

plt.tight_layout()

out_png = output_base + '.png'
out_pdf = output_base + '.pdf'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")