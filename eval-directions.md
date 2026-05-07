# RESEARCH PAPER PLOTTING STYLE GUIDE

Use this prompt when requesting plots for research papers:

---

## PROMPT FOR CLAUDE:

"Create publication-quality plots using the following styling requirements:

**Typography & Fonts:**
- Use serif fonts (DejaVu Serif, Times New Roman, or Times)
- Font sizes: body text 10pt, axis labels 10pt, tick labels 9pt, legend 9pt, title 11pt
- Title weight: normal (not bold)
- All text in black (#000000)

**Visual Design:**
- White background, no grid lines
- Remove top and right spines (frame only left and bottom)
- Axis line width: 0.8pt
- Tick marks: 4pt outward, 0.8pt width
- No borders/edges on bars or filled elements (edgecolor='none')
- Minimal padding and spacing

**Colors:**
- Use soft, muted pastel colors:
  * Soft Rose: #E8B4B8
  * Soft Coral: #F4A582
  * Soft Sage: #B8D4C8
  * Soft Sky Blue: #92C5DE
  * Soft Lavender: #D4B5D4
  * Soft Sand: #F4D9A6
  * Soft Periwinkle: #A8C8E1
  * Soft Taupe: #D9C5B2
  * Soft Mint: #B5D4A8
  * Soft Mauve: #E8C8D4
  * Soft Aqua: #A8D8D8
  * Soft Wheat: #E8D4B8
- Colors should be consistent across all plots for the same data series
- Use alpha=0.8 for transparency when appropriate

**Plot Proportions:**
- Default figure size: 8×5 inches (or 7×5 for scatter plots)
- Bar height: 0.7 (for horizontal bars)
- Scatter point size: 100
- Use tight_layout() and bbox_inches='tight'

**Data Labels & Formatting:**
- Place labels inside bars when space permits (white text), otherwise outside (dark text)
- Use abbreviated numbers: 'M' for millions, 'K' for thousands
- Format: `lambda x: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'`
- Label font size: 8pt
- Minimal annotations on scatter plots (only key points)

**Legend:**
- No frame (frameon=False)
- Position: 'best' or 'upper left'
- 2 columns if >8 items
- Minimal spacing (columnspacing=1, handletextpad=0.5)

**Output:**
- Save in both PNG and SVG formats
- PNG: 300 DPI resolution
- SVG: Vector format for scalability
- Both with tight bounding box (bbox_inches='tight')
- No excessive whitespace

**Overall Aesthetic:**
- Clean, minimal, professional
- Publication-ready for academic journals
- Inspired by IEEE/ACM/Nature style guides"

---

## QUICK REFERENCE - MATPLOTLIB RCPARAMS:

```python
plt.rcParams.update({
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'png',  # Default, but also save as SVG
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
    'axes.titleweight': 'normal',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'legend.frameon': False,
})
```

## COLOR PALETTE ARRAY:

```python
PASTEL_COLORS = [
    '#E8B4B8',  # Soft Rose
    '#F4A582',  # Soft Coral
    '#B8D4C8',  # Soft Sage
    '#92C5DE',  # Soft Sky Blue
    '#D4B5D4',  # Soft Lavender
    '#F4D9A6',  # Soft Sand
    '#A8C8E1',  # Soft Periwinkle
    '#D9C5B2',  # Soft Taupe
    '#B5D4A8',  # Soft Mint
    '#E8C8D4',  # Soft Mauve
    '#A8D8D8',  # Soft Aqua
    '#E8D4B8',  # Soft Wheat
]
```

---

## SAVING PLOTS IN BOTH FORMATS:

```python
# Save each plot in both PNG and SVG
plt.savefig('plot_name.png', dpi=300, bbox_inches='tight')
plt.savefig('plot_name.svg', format='svg', bbox_inches='tight')
plt.close()
```

---

## USAGE EXAMPLES:

**For new plots:**
"Create a bar chart showing X vs Y using the research paper plotting style."

**For converting existing plots:**
"Convert this plotting code to use the research paper plotting style guide."

**For quick reference:**
"Apply research paper plot styling" or "Use publication-quality plot style"

---

## KEY PRINCIPLES:

1. **Less is more** - Remove unnecessary elements
2. **Consistency** - Same colors/fonts across all figures
3. **Readability** - Clear labels, appropriate sizes
4. **Professional** - Clean, minimal, academic aesthetic
5. **Publication-ready** - High DPI, proper formatting