"""Shared figure style (2026-11-16, second version). Color scheme taken from the trim-sensitivity figure of the manuscript
(navy solid, crimson dashed, dark green dash-dot, pale blue and pale green fills), which the author set as the reference.
Conventions: closed axes box, inward ticks, Times-like serif (STIX), light dashed grid, series distinguishable without color by line style
and marker, framed legends placed where no data lie. save() writes a PDF and an SVG whose text stays editable text (Inkscape)."""
import os
import matplotlib as mpl
K, NAVY, CRIMSON, GREEN, GREY, LGREY = "#000000", "#0b1a6e", "#dc143c", "#0b6623", "#6e6e6e", "#c8c8c8"
LBLUE, LGREEN, LRED = "#dfe5f6", "#dff0df", "#f9dfe4"
RED, BLUE = CRIMSON, NAVY   # names used by older code
def apply(size=9):
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": size, "axes.labelsize": size, "axes.titlesize": size, "legend.fontsize": size - 1,
        "xtick.labelsize": size - 0.5, "ytick.labelsize": size - 0.5, "axes.linewidth": 0.8, "axes.edgecolor": K, "axes.labelcolor": K, "text.color": K,
        "axes.spines.top": True, "axes.spines.right": True, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.color": K, "ytick.color": K, "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": GREY, "legend.framealpha": 1.0,
        "patch.linewidth": 0.8, "axes.grid": True, "grid.color": "#d9d9d9", "grid.linestyle": (0, (4, 3)), "grid.linewidth": 0.5, "axes.axisbelow": True,
        "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none", "hatch.linewidth": 0.6, "errorbar.capsize": 2.5})
def save(fig, pdf_path, **kw):
    fig.savefig(pdf_path, **kw); d = os.path.join(os.path.dirname(pdf_path), "svg"); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, os.path.basename(pdf_path).replace(".pdf", ".svg")), **kw)
def box(ax, data, pos, width=0.55, face=LBLUE, edge=NAVY, med=CRIMSON, vert=True):
    return ax.boxplot(data, positions=pos, widths=width, patch_artist=True, vert=vert, showfliers=True,
                      flierprops=dict(marker="o", ms=3.2, mfc="white", mec=edge, mew=0.8), medianprops=dict(color=med, lw=1.8),
                      boxprops=dict(facecolor=face, edgecolor=edge, lw=1.0), whiskerprops=dict(color=edge, lw=1.0), capprops=dict(color=edge, lw=1.0))

AMBER, LAMBER, LGREYF = "#b36b00", "#fbe8cc", "#ececec"
def sbox(ax, data, pos, color, fill, width=0.6, seed=0, pts=True):
    """Box plot in the manner of Suo et al. (Engineering Reports, 2026, Fig. 8): one color per group, pale filled box, median line,
    whiskers to the extremes, and every run drawn as a small dot of the same color."""
    import numpy as np
    b = ax.boxplot([data], positions=[pos], widths=width, patch_artist=True, showfliers=False, whis=(0, 100), medianprops=dict(color=color, lw=1.6),
                   boxprops=dict(facecolor=fill, edgecolor=color, lw=1.0), whiskerprops=dict(color=color, lw=1.0), capprops=dict(color=color, lw=1.0))
    if pts:
        r = np.random.default_rng(seed); ax.plot(pos + r.uniform(-0.2, 0.2, len(data)), data, ls="", marker="o", ms=2.6, mfc=color, mec="none", alpha=0.9, zorder=3)
    return b
def bracket(ax, x1, x2, y, text, h=0.8, fontsize=8):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=K, lw=0.8, clip_on=False); ax.text((x1 + x2) / 2, y + h * 1.15, text, ha="center", va="bottom", fontsize=fontsize)
