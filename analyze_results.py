"""
=============================================================================
GA-OBN Results Analyzer
=============================================================================
PURPOSE:
    Reads results CSV files, produces charts, comparison tables, and a
    summary HTML report. Works on any results file produced by the
    comparison framework — small test runs or full overnight experiments.

USAGE:
    python3 analyze_results.py                    # uses results_light.csv
    python3 analyze_results.py results_full.csv   # specify a file

OUTPUT:
    results_analysis/
    ├── charts/          ← all PNG charts
    ├── tables/          ← all CSV summary tables
    └── report.html      ← single-file interactive report
=============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE   = sys.argv[1] if len(sys.argv) > 1 else "results_light.csv"
OUT_DIR      = "results_analysis"
CHARTS_DIR   = os.path.join(OUT_DIR, "charts")
TABLES_DIR   = os.path.join(OUT_DIR, "tables")

os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ── Colors ───────────────────────────────────────────────────────────────────
PALETTE = {
    "GA-OBN (v1)": "#1A3A5C",
    "GA-OBN (v2)": "#1E8449",
    "GA-OBN (v0)": "#5D6D7E",
    "MLP-ReLU":    "#2980B9",
    "MLP-Sin":     "#7D3C98",
    "ChebyKAN":    "#E67E22",
    "FourierKAN":  "#C0392B",
}
DEFAULT_COLOR = "#888888"

FUNC_CATEGORIES = {
    "sphere":          "polynomial",
    "rosenbrock":      "polynomial",
    "rastrigin":       "mixed",
    "griewank":        "periodic",
    "ackley":          "mixed",
    "sine_composite":  "periodic",
    "fourier_mixture": "periodic",
}

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.titlesize':     11,
    'axes.labelsize':     10,
})

# ════════════════════════════════════════════════════════════════════════════
# LOAD AND VALIDATE DATA
# ════════════════════════════════════════════════════════════════════════════

print(f"\nLoading: {INPUT_FILE}")
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found.")
    print("Run the comparison script first to generate results.")
    sys.exit(1)

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows")
print(f"Models:     {sorted(df['Model'].unique())}")
print(f"Functions:  {sorted(df['Function'].unique())}")
print(f"Dimensions: {sorted(df['Dims'].unique())}")
print(f"Runs:       {df['Run'].max() + 1}")

# Clean up
df['Function'] = df['Function'].str.lower()
df['Category'] = df['Function'].map(FUNC_CATEGORIES).fillna('unknown')

# ── Summary table ────────────────────────────────────────────────────────────
summary = df.groupby(['Function', 'Dims', 'Model']).agg(
    R2_mean   = ('R2',         'mean'),
    R2_std    = ('R2',         'std'),
    R2_min    = ('R2',         'min'),
    R2_max    = ('R2',         'max'),
    MSE_mean  = ('MSE',        'mean'),
    MAE_mean  = ('MAE',        'mean'),
    RMSE_mean = ('RMSE',       'mean'),
    Time_mean = ('train_time', 'mean'),
    Params    = ('n_params',   'first'),
    N_runs    = ('Run',        'count'),
).reset_index()

summary['R2_std'] = summary['R2_std'].fillna(0)
summary.to_csv(os.path.join(TABLES_DIR, "summary.csv"), index=False)
print(f"\nSummary table saved → {TABLES_DIR}/summary.csv")

# ── Best model per function/dim ──────────────────────────────────────────────
best = summary.loc[summary.groupby(['Function', 'Dims'])['R2_mean'].idxmax()].copy()
best.to_csv(os.path.join(TABLES_DIR, "best_per_function.csv"), index=False)

# ── GA-OBN rank per function/dim ─────────────────────────────────────────────
ranks = []
for (fn, dim), grp in summary.groupby(['Function', 'Dims']):
    ranked = grp.sort_values('R2_mean', ascending=False).reset_index(drop=True)
    for i, row in ranked.iterrows():
        if 'GA-OBN' in row['Model']:
            ranks.append({
                'Function': fn, 'Dims': dim, 'Model': row['Model'],
                'R2_mean': row['R2_mean'], 'Rank': i + 1,
                'Total_models': len(ranked),
            })
rank_df = pd.DataFrame(ranks)
rank_df.to_csv(os.path.join(TABLES_DIR, "ga_obn_rankings.csv"), index=False)

all_models   = sorted(df['Model'].unique())
all_funcs    = sorted(df['Function'].unique())
all_dims     = sorted(df['Dims'].unique())
ga_models    = [m for m in all_models if 'GA-OBN' in m]
base_models  = [m for m in all_models if 'GA-OBN' not in m]

def get_color(model):
    return PALETTE.get(model, DEFAULT_COLOR)


# ════════════════════════════════════════════════════════════════════════════
# CHART 1: R² Overview — all functions and dimensions
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Chart 1: R² Overview...")

n_funcs = len(all_funcs)
n_dims  = len(all_dims)
fig, axes = plt.subplots(n_dims, n_funcs,
                         figsize=(max(16, n_funcs * 3), n_dims * 4),
                         squeeze=False)
fig.suptitle("R² by Function, Dimension, and Model\n(mean ± std across runs)",
             fontsize=13, fontweight='bold', color='#1A3A5C', y=1.01)

for di, dim in enumerate(all_dims):
    for fi, fn in enumerate(all_funcs):
        ax = axes[di][fi]
        sub = summary[(summary['Function'] == fn) & (summary['Dims'] == dim)]
        if sub.empty:
            ax.set_visible(False)
            continue

        sub = sub.sort_values('R2_mean', ascending=False)
        models = sub['Model'].tolist()
        r2s    = sub['R2_mean'].tolist()
        stds   = sub['R2_std'].tolist()
        colors = [get_color(m) for m in models]

        x = range(len(models))
        bars = ax.bar(x, r2s, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
        ax.errorbar(x, r2s, yerr=stds, fmt='none', color='black',
                    capsize=3, linewidth=1, zorder=4)
        ax.axhline(0, color='#CCCCCC', linewidth=0.8, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\n').replace('GA-OBN\n', 'GA-OBN\n')
                            for m in models], fontsize=7)
        ymin = min(min(r2s) - 0.15, -0.1)
        ax.set_ylim(ymin, 1.15)

        for bar, val in zip(bars, r2s):
            ypos = max(val, 0) + 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold')

        title_color = '#1E8449' if any('GA-OBN' in m for m in
                      sub.head(1)['Model'].tolist()) else '#1A3A5C'
        ax.set_title(f"{fn.upper()}\n{dim}D", fontsize=9,
                     fontweight='bold', color=title_color)
        ax.set_ylabel("R²" if fi == 0 else "", fontsize=9)

# Legend
legend_handles = [mpatches.Patch(color=get_color(m), label=m) for m in all_models]
fig.legend(handles=legend_handles, loc='lower center', ncol=len(all_models),
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))

plt.tight_layout()
path = os.path.join(CHARTS_DIR, "01_r2_overview.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# CHART 2: GA-OBN vs Best Baseline — head to head
# ════════════════════════════════════════════════════════════════════════════
print("Generating Chart 2: GA-OBN vs Best Baseline...")

combos = [(fn, dim) for fn in all_funcs for dim in all_dims]
n_combos = len(combos)

fig, ax = plt.subplots(figsize=(max(14, n_combos * 1.1), 6))
fig.suptitle("GA-OBN vs Best Baseline — Head to Head",
             fontsize=12, fontweight='bold', color='#1A3A5C')

x       = np.arange(n_combos)
w       = 0.35
labels  = []
ga_vals, ga_stds   = [], []
bl_vals, bl_stds   = [], []
bl_names           = []
ga_model_names     = []

for fn, dim in combos:
    sub = summary[(summary['Function'] == fn) & (summary['Dims'] == dim)]
    if sub.empty:
        ga_vals.append(0); ga_stds.append(0)
        bl_vals.append(0); bl_stds.append(0)
        bl_names.append(''); ga_model_names.append('')
        labels.append(f"{fn}\n{dim}D")
        continue

    ga_sub = sub[sub['Model'].str.contains('GA-OBN')]
    bl_sub = sub[~sub['Model'].str.contains('GA-OBN')]

    if ga_sub.empty or bl_sub.empty:
        ga_vals.append(0); ga_stds.append(0)
        bl_vals.append(0); bl_stds.append(0)
        bl_names.append(''); ga_model_names.append('')
        labels.append(f"{fn}\n{dim}D")
        continue

    best_ga = ga_sub.loc[ga_sub['R2_mean'].idxmax()]
    best_bl = bl_sub.loc[bl_sub['R2_mean'].idxmax()]

    ga_vals.append(best_ga['R2_mean'])
    ga_stds.append(best_ga['R2_std'])
    ga_model_names.append(best_ga['Model'])
    bl_vals.append(best_bl['R2_mean'])
    bl_stds.append(best_bl['R2_std'])
    bl_names.append(best_bl['Model'])
    labels.append(f"{fn}\n{dim}D")

b1 = ax.bar(x - w/2, ga_vals, w, label='GA-OBN (best)',
            color='#1E8449', alpha=0.9, edgecolor='white')
b2 = ax.bar(x + w/2, bl_vals, w, label='Best baseline',
            color='#2980B9', alpha=0.9, edgecolor='white')
ax.errorbar(x - w/2, ga_vals, yerr=ga_stds, fmt='none',
            color='black', capsize=3, linewidth=1)
ax.errorbar(x + w/2, bl_vals, yerr=bl_stds, fmt='none',
            color='black', capsize=3, linewidth=1)
ax.axhline(0, color='#CCCCCC', linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("R²", fontsize=11)
ax.legend(fontsize=10, frameon=False)
ymin = min(min(ga_vals + bl_vals) - 0.1, -0.1)
ax.set_ylim(ymin, 1.35)

# Annotate GA-OBN version name above each green bar
for i, (val, name) in enumerate(zip(ga_vals, ga_model_names)):
    if name:
        short = name.replace('GA-OBN ', '')  # e.g. "(v2)"
        ax.text(i - w/2, max(val, 0) + 0.06, short,
                ha='center', va='bottom', fontsize=7,
                color='#1E8449', fontweight='bold', rotation=45)

# Annotate best baseline model name above each blue bar
for i, (val, name) in enumerate(zip(bl_vals, bl_names)):
    if name and val > -0.5:
        ax.text(i + w/2, max(val, 0) + 0.06, name,
                ha='center', va='bottom', fontsize=7,
                color='#2980B9', fontweight='bold', rotation=45)

# Annotate wins
for i, (gv, bv) in enumerate(zip(ga_vals, bl_vals)):
    if gv >= bv and gv > 0:
        ax.text(i, max(gv, bv) + 0.22, '★', ha='center',
                fontsize=12, color='#1E8449', fontweight='bold')

ax.text(0.99, 0.97, '★ = GA-OBN wins', transform=ax.transAxes,
        ha='right', va='top', fontsize=9, color='#1E8449', style='italic')

plt.tight_layout()
path = os.path.join(CHARTS_DIR, "02_ga_vs_best_baseline.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# CHART 3: Parameter efficiency scatter
# ════════════════════════════════════════════════════════════════════════════
print("Generating Chart 3: Parameter efficiency...")

n_dims = len(all_dims)
fig, axes = plt.subplots(1, n_dims, figsize=(7 * n_dims, 6), squeeze=False)
fig.suptitle("Parameter Count vs R² — Efficiency Analysis",
             fontsize=12, fontweight='bold', color='#1A3A5C')

for di, dim in enumerate(all_dims):
    ax = axes[0][di]
    sub = summary[summary['Dims'] == dim]
    if sub.empty:
        ax.set_visible(False)
        continue

    for model in all_models:
        msub = sub[sub['Model'] == model]
        if msub.empty:
            continue
        params  = msub['Params'].values
        r2_vals = msub['R2_mean'].values
        color   = get_color(model)
        size    = 220 if 'GA-OBN' in model else 120
        marker  = '*' if 'GA-OBN' in model else 'o'
        zorder  = 5 if 'GA-OBN' in model else 3

        ax.scatter(params, r2_vals, c=color, s=size, marker=marker,
                   zorder=zorder, edgecolors='white', linewidth=1,
                   label=model, alpha=0.9)

    ax.axhline(0, color='#CCCCCC', linewidth=0.8)
    ax.set_xlabel("Number of Parameters / Genes", fontsize=10)
    ax.set_ylabel("R² (mean across functions and runs)", fontsize=10)
    ax.set_title(f"{dim}D", fontsize=11, fontweight='bold', color='#1A3A5C')

    ax.text(0.02, 0.98,
            "★ = GA-OBN\n● = Baseline",
            transform=ax.transAxes, va='top', fontsize=8,
            color='#555555', style='italic')

handles = [mpatches.Patch(color=get_color(m), label=m) for m in all_models]
fig.legend(handles=handles, loc='lower center', ncol=len(all_models),
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
path = os.path.join(CHARTS_DIR, "03_parameter_efficiency.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# CHART 4: R² distribution (box plots) per model
# ════════════════════════════════════════════════════════════════════════════
print("Generating Chart 4: R² distribution box plots...")

fig, axes = plt.subplots(1, n_dims, figsize=(7 * n_dims, 6), squeeze=False)
fig.suptitle("R² Distribution Across All Functions — Consistency Analysis",
             fontsize=12, fontweight='bold', color='#1A3A5C')

for di, dim in enumerate(all_dims):
    ax = axes[0][di]
    sub = df[df['Dims'] == dim]
    if sub.empty:
        ax.set_visible(False)
        continue

    data_by_model = [sub[sub['Model'] == m]['R2'].dropna().values
                     for m in all_models]
    colors = [get_color(m) for m in all_models]

    bp = ax.boxplot(data_by_model, patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color='#CCCCCC', linewidth=0.8)
    ax.set_xticks(range(1, len(all_models) + 1))
    ax.set_xticklabels([m.replace(' ', '\n') for m in all_models], fontsize=8)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title(f"{dim}D — All Functions Combined",
                 fontsize=11, fontweight='bold', color='#1A3A5C')

plt.tight_layout()
path = os.path.join(CHARTS_DIR, "04_r2_distribution.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# CHART 5: Per-category analysis
# ════════════════════════════════════════════════════════════════════════════
print("Generating Chart 5: Category analysis...")

categories = df['Category'].unique()
n_cats     = len(categories)
fig, axes  = plt.subplots(1, n_cats, figsize=(6 * n_cats, 6), squeeze=False)
fig.suptitle("R² by Function Category",
             fontsize=12, fontweight='bold', color='#1A3A5C')

for ci, cat in enumerate(sorted(categories)):
    ax   = axes[0][ci]
    csub = summary[summary['Function'].map(FUNC_CATEGORIES) == cat]
    if csub.empty:
        ax.set_visible(False)
        continue

    cat_avg = csub.groupby('Model')['R2_mean'].mean().reset_index()
    cat_avg = cat_avg.sort_values('R2_mean', ascending=False)
    colors  = [get_color(m) for m in cat_avg['Model']]

    bars = ax.bar(range(len(cat_avg)), cat_avg['R2_mean'],
                  color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='#CCCCCC', linewidth=0.8)
    ax.set_xticks(range(len(cat_avg)))
    ax.set_xticklabels([m.replace(' ', '\n') for m in cat_avg['Model']], fontsize=8)
    ax.set_ylabel("Mean R² across all functions in category", fontsize=9)
    ax.set_title(f"{cat.upper()} functions", fontsize=11,
                 fontweight='bold', color='#1A3A5C')

    for bar, val in zip(bars, cat_avg['R2_mean']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(val, 0) + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
path = os.path.join(CHARTS_DIR, "05_category_analysis.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# CHART 6: Consistency — R² std across runs
# ════════════════════════════════════════════════════════════════════════════
print("Generating Chart 6: Consistency (std) analysis...")

fig, axes = plt.subplots(1, n_dims, figsize=(7 * n_dims, 5), squeeze=False)
fig.suptitle("Result Consistency — Lower Std = More Stable Across Runs",
             fontsize=12, fontweight='bold', color='#1A3A5C')

for di, dim in enumerate(all_dims):
    ax   = axes[0][di]
    sub  = summary[summary['Dims'] == dim]
    if sub.empty:
        ax.set_visible(False)
        continue

    std_avg = sub.groupby('Model')['R2_std'].mean().reset_index()
    std_avg = std_avg.sort_values('R2_std')
    colors  = [get_color(m) for m in std_avg['Model']]

    bars = ax.bar(range(len(std_avg)), std_avg['R2_std'],
                  color=colors, edgecolor='white')
    ax.set_xticks(range(len(std_avg)))
    ax.set_xticklabels([m.replace(' ', '\n') for m in std_avg['Model']], fontsize=8)
    ax.set_ylabel("Mean R² std deviation", fontsize=10)
    ax.set_title(f"{dim}D", fontsize=11, fontweight='bold', color='#1A3A5C')
    ax.text(0.98, 0.97, 'Lower = more consistent',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, style='italic', color='#666666')

plt.tight_layout()
path = os.path.join(CHARTS_DIR, "06_consistency.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ════════════════════════════════════════════════════════════════════════════
print("Generating HTML report...")

def img_tag(filename, caption, width="100%"):
    rel = os.path.join("charts", filename)
    return f'''
    <figure>
      <img src="{rel}" style="width:{width}; border-radius:6px;
           box-shadow:0 2px 8px rgba(0,0,0,0.12);">
      <figcaption>{caption}</figcaption>
    </figure>'''

# Best results table HTML
best_html_rows = ""
for _, row in best.sort_values(['Function', 'Dims']).iterrows():
    ga_winner = 'GA-OBN' in row['Model']
    style = 'background:#D5F5E3; font-weight:bold;' if ga_winner else ''
    best_html_rows += f"""
    <tr style="{style}">
      <td>{row['Function']}</td><td>{int(row['Dims'])}D</td>
      <td>{row['Model']}</td><td>{row['R2_mean']:.4f}</td>
      <td>±{row['R2_std']:.4f}</td><td>{int(row['Params'])}</td>
    </tr>"""

# Rankings table HTML
rank_html_rows = ""
for _, row in rank_df.sort_values(['Function', 'Dims']).iterrows():
    rank_color = '#1E8449' if row['Rank'] == 1 else \
                 '#E67E22' if row['Rank'] == 2 else '#888888'
    medal = '🥇' if row['Rank'] == 1 else '🥈' if row['Rank'] == 2 else \
            '🥉' if row['Rank'] == 3 else f"#{int(row['Rank'])}"
    rank_html_rows += f"""
    <tr>
      <td>{row['Function']}</td><td>{int(row['Dims'])}D</td>
      <td>{row['Model']}</td>
      <td style="color:{rank_color}; font-weight:bold; font-size:1.1em">
          {medal} {int(row['Rank'])} of {int(row['Total_models'])}</td>
      <td>{row['R2_mean']:.4f}</td>
    </tr>"""

# Full summary table HTML
summary_html_rows = ""
for _, row in summary.sort_values(['Function', 'Dims', 'R2_mean'],
                                   ascending=[True, True, False]).iterrows():
    ga = 'GA-OBN' in row['Model']
    style = 'background:#EBF5FB;' if ga else ''
    summary_html_rows += f"""
    <tr style="{style}">
      <td>{row['Function']}</td><td>{int(row['Dims'])}D</td>
      <td><b>{row['Model']}</b></td>
      <td>{row['R2_mean']:.4f}</td><td>±{row['R2_std']:.4f}</td>
      <td>{row['R2_min']:.4f}</td><td>{row['R2_max']:.4f}</td>
      <td>{row['MSE_mean']:.5f}</td>
      <td>{int(row['Params'])}</td>
      <td>{row['Time_mean']:.1f}s</td>
      <td>{int(row['N_runs'])}</td>
    </tr>"""

timestamp = datetime.now().strftime("%d %B %Y, %H:%M")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GA-OBN Results Analysis</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, sans-serif; background: #F8F9FA;
          color: #222; line-height: 1.6; }}
  .header {{ background: #1A3A5C; color: white; padding: 32px 40px; }}
  .header h1 {{ font-size: 2em; margin-bottom: 6px; }}
  .header p  {{ opacity: 0.8; font-size: 0.95em; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px; }}
  h2 {{ color: #1A3A5C; border-left: 4px solid #1E8449;
        padding-left: 12px; margin: 40px 0 16px; font-size: 1.4em; }}
  h3 {{ color: #C0392B; margin: 24px 0 10px; }}
  figure {{ margin: 16px 0 24px; }}
  figcaption {{ text-align:center; font-size:0.85em; color:#666;
                margin-top:6px; font-style:italic; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                 gap: 16px; margin: 24px 0; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
  .stat-card .value {{ font-size: 2em; font-weight: bold; color: #1A3A5C; }}
  .stat-card .label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 16px 0; }}
  th {{ background: #1A3A5C; color: white; padding: 10px 14px;
        text-align: left; font-size: 0.9em; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #F0F0F0;
        font-size: 0.88em; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #F8FBFF; }}
  .win {{ background: #D5F5E3; font-weight: bold; }}
  .note {{ background: #FEF9E7; border-left: 4px solid #E67E22;
           padding: 12px 16px; border-radius: 4px; margin: 16px 0;
           font-size: 0.9em; }}
  .footer {{ text-align:center; padding: 32px; color: #999;
             font-size: 0.85em; border-top: 1px solid #EEE; margin-top: 40px; }}
</style>
</head>
<body>

<div class="header">
  <h1>GA-OBN Results Analysis</h1>
  <p>Source: {INPUT_FILE} &nbsp;|&nbsp; Generated: {timestamp}</p>
  <p>Models: {", ".join(all_models)} &nbsp;|&nbsp;
     Functions: {", ".join(all_funcs)} &nbsp;|&nbsp;
     Dimensions: {", ".join(str(d) for d in all_dims)}</p>
</div>

<div class="container">

<!-- STATS -->
<div class="stats-grid">
  <div class="stat-card">
    <div class="value">{len(df)}</div>
    <div class="label">Total experiment runs</div>
  </div>
  <div class="stat-card">
    <div class="value">{len(all_models)}</div>
    <div class="label">Models compared</div>
  </div>
  <div class="stat-card">
    <div class="value">{len(all_funcs)}</div>
    <div class="label">Benchmark functions</div>
  </div>
  <div class="stat-card">
    <div class="value">{int(rank_df[rank_df['Rank']==1].shape[0])}</div>
    <div class="label">GA-OBN first-place finishes</div>
  </div>
  <div class="stat-card">
    <div class="value">{int(rank_df[rank_df['Rank']<=2].shape[0])}</div>
    <div class="label">GA-OBN top-2 finishes</div>
  </div>
  <div class="stat-card">
    <div class="value">{summary[summary['Model'].str.contains('GA-OBN')]['Params'].min():.0f}</div>
    <div class="label">Min GA-OBN genes (2D)</div>
  </div>
</div>

<!-- CHART 1 -->
<h2>1. R² Overview — All Functions and Dimensions</h2>
{img_tag("01_r2_overview.png", "R² for every model across all tested functions and dimensions. Green star (★) marks where GA-OBN placed first.")}

<!-- CHART 2 -->
<h2>2. GA-OBN vs Best Baseline — Head to Head</h2>
{img_tag("02_ga_vs_best_baseline.png", "Best GA-OBN version vs best competing baseline per function/dimension. ★ marks GA-OBN wins.")}

<!-- BEST TABLE -->
<h2>3. Best Model Per Function and Dimension</h2>
<div class="note">Green highlighting indicates GA-OBN was the top model for that function/dimension combination.</div>
<table>
  <tr>
    <th>Function</th><th>Dim</th><th>Best Model</th>
    <th>R² Mean</th><th>R² Std</th><th>Params</th>
  </tr>
  {best_html_rows}
</table>

<!-- RANKINGS TABLE -->
<h2>4. GA-OBN Rankings</h2>
<table>
  <tr>
    <th>Function</th><th>Dim</th><th>Model</th><th>Rank</th><th>R² Mean</th>
  </tr>
  {rank_html_rows}
</table>

<!-- CHART 3 -->
<h2>5. Parameter Efficiency</h2>
{img_tag("03_parameter_efficiency.png", "R² vs parameter count. Stars (★) are GA-OBN. Better models appear top-left (high R², low params).")}

<!-- CHART 4 -->
<h2>6. Result Consistency Across Runs</h2>
{img_tag("04_r2_distribution.png", "Box plots showing R² distribution across all runs and functions. Tighter boxes = more consistent.")}
{img_tag("06_consistency.png", "Mean R² standard deviation per model. Lower = more consistent results across runs.")}

<!-- CHART 5 -->
<h2>7. Performance by Function Category</h2>
{img_tag("05_category_analysis.png", "Mean R² grouped by function type: polynomial, mixed, and periodic. Reveals which model families suit which problem types.")}

<!-- FULL TABLE -->
<h2>8. Complete Results Table</h2>
<table>
  <tr>
    <th>Function</th><th>Dim</th><th>Model</th>
    <th>R² Mean</th><th>R² Std</th><th>R² Min</th><th>R² Max</th>
    <th>MSE Mean</th><th>Params</th><th>Time</th><th>Runs</th>
  </tr>
  {summary_html_rows}
</table>

</div>
<div class="footer">
  GA-OBN Dissertation Analysis &nbsp;|&nbsp; Sahar Alwadei G201901730 &nbsp;|&nbsp;
  KFUPM &nbsp;|&nbsp; {timestamp}
</div>
</body>
</html>"""

report_path = os.path.join(OUT_DIR, "report.html")
with open(report_path, 'w') as f:
    f.write(html)
print(f"  Saved → {report_path}")

# ════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nOutput folder: {OUT_DIR}/")
print(f"  Charts:      {CHARTS_DIR}/ (6 charts)")
print(f"  Tables:      {TABLES_DIR}/ (3 CSV files)")
print(f"  Report:      {report_path}")
print(f"\nOpen the report: open {report_path}")
print()

print("GA-OBN RANKING SUMMARY:")
print(f"{'Function':<18} {'Dim':<6} {'Model':<18} {'Rank':<12} {'R²'}")
print("-" * 65)
for _, row in rank_df.sort_values(['Dims', 'Function']).iterrows():
    medal = '🥇' if row['Rank'] == 1 else '🥈' if row['Rank'] == 2 else \
            '🥉' if row['Rank'] == 3 else f"  #{int(row['Rank'])}"
    print(f"{row['Function']:<18} {str(int(row['Dims']))+'D':<6} "
          f"{row['Model']:<18} {medal} of {int(row['Total_models'])}      "
          f"{row['R2_mean']:.4f}")
