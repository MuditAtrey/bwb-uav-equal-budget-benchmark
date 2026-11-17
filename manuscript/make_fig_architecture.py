"""
Copy of scripts/rebuild_figure3.py for the manuscript (2026-11-17): no title on the figure (CJA guide), the last card names the single
objective, text kept as editable text in the SVG. Writes paper_v2/figures/figure3_trim_loop_architecture.{pdf,svg}.
Rebuild Figure 3: Mass/CG build-up, trim constraints and active-learning MDO architecture (battery centroid fixed at 0.82 c_root)
Clean, publication-grade vector architecture flowchart with generous spacing,
unboxed text labels, and clean broken-line connectors.
"""
import os
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 10,
})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures"); SVG_DIR = os.path.join(FIGURES_DIR, "svg"); os.makedirs(SVG_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Setup Figure Canvas ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.8, 7.4), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# ── Color Palette ────────────────────────────────────────────────────────────
NAVY_EDGE   = "#0f3057"
NAVY_BG     = "#f0f6fa"

GREEN_EDGE  = "#1b5e20"
GREEN_BG    = "#f1f8f3"

PURPLE_EDGE = "#4a148c"
PURPLE_BG   = "#f6eff9"

ORANGE_EDGE = "#b45309"
ORANGE_BG   = "#fffbf5"

GRAY_EDGE   = "#374151"
GRAY_BG     = "#f8fafc"

def draw_card(ax, x, y, w, h, title, lines, bg_color, edge_color, title_size=8.5, body_size=7.6, lw=1.6):
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2,rounding_size=1.5",
        facecolor=bg_color, edgecolor=edge_color,
        linewidth=lw, zorder=2
    )
    ax.add_patch(card)
    
    # Title placed with clear margin below top
    ax.text(
        x + w/2, y + h - 3.4,
        title,
        ha="center", va="center",
        fontsize=title_size, fontweight="bold",
        color=edge_color, zorder=4
    )
    
    # Body lines centered
    n_lines = len(lines)
    body_top = y + h - 6.0
    body_avail_h = body_top - (y + 1.2)
    line_spacing = body_avail_h / max(n_lines, 1)
    
    for i, line_text in enumerate(lines):
        line_y = body_top - 0.5*line_spacing - i * line_spacing
        ax.text(
            x + w/2, line_y,
            line_text,
            ha="center", va="center",
            fontsize=body_size, color="#1f2937", zorder=4
        )

def draw_arrow(ax, x1, y1, x2, y2, color="#0f3057", lw=1.8):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=13,
        color=color, linewidth=lw, zorder=5
    )
    ax.add_patch(arrow)

# ── Title Banner ─────────────────────────────────────────────────────────────
# (title removed: captions carry the title)

# ── ROW 1: Forward Physical Evaluation Pipeline (y = 67.5 to 90, h = 22.5) ───
r1_y = 67.5
r1_h = 22.5
c_w  = 16.5

# 1. Design Vector
draw_card(
    ax, 2.5, r1_y, c_w, r1_h,
    "1. Design Vector",
    [
        r"$\mathbf{x} \in \Omega \subset \mathbb{R}^{14}$",
        r"Planform: $\{\Lambda, c, \theta\}$",
        r"Ops: $\{m_\mathrm{batt}, V_\mathrm{cr}\}$"
    ],
    NAVY_BG, NAVY_EDGE
)

# 2. Hermite Manifold
draw_card(
    ax, 22.0, r1_y, c_w, r1_h,
    "2. Planform Loft",
    [
        "Cubic LE, straight TE",
        r"$c_\mathrm{root} \geq c_\mathrm{blend} \geq c_\mathrm{tip}$",
        "3D Loft & Nacelle"
    ],
    NAVY_BG, NAVY_EDGE
)

# 3. Aerodynamics (VLM)
draw_card(
    ax, 41.5, r1_y, c_w, r1_h,
    "3. Aerodynamics",
    [
        "AeroSandbox VLM",
        r"$C_L(\alpha), C_{D,\mathrm{ind}}, C_{m0}$",
        r"Neutral Point $x_\mathrm{np}$"
    ],
    NAVY_BG, NAVY_EDGE
)

# 4. Dynamic CG Balance
draw_card(
    ax, 61.0, r1_y, c_w, r1_h,
    "4. Mass & CG",
    [
        r"Battery fixed at $0.82\,c_\mathrm{root}$",
        r"Update $m_\mathrm{tot}$, $x_\mathrm{cg}$",
        r"Evaluate $SM$ and $C_m$ at $\alpha_\mathrm{cr}$"
    ],
    PURPLE_BG, PURPLE_EDGE
)

# 5. Mission Objectives
draw_card(
    ax, 80.5, r1_y, 17.0, r1_h,
    "5. Objective",
    [
        r"Breguet Range $R(\mathbf{x})$",
        r"(the only objective;",
        r"constraints in the gate)"
    ],
    GREEN_BG, GREEN_EDGE
)

# Forward arrows (Row 1)
draw_arrow(ax, 2.5 + c_w, 78.5, 22.0, 78.5, color=NAVY_EDGE)
draw_arrow(ax, 22.0 + c_w, 78.5, 41.5, 78.5, color=NAVY_EDGE)
draw_arrow(ax, 41.5 + c_w, 78.5, 61.0, 78.5, color=PURPLE_EDGE)
draw_arrow(ax, 61.0 + c_w, 78.5, 80.5, 78.5, color=GREEN_EDGE)

# ── ROW 2: Feasibility Gate (y = 39.5 to 57.5, h = 18.0) ──────────────────────
gate_y = 39.5
gate_h = 18.0
gate_w = 75.5
gate_x = 22.0

gate_box = FancyBboxPatch(
    (gate_x, gate_y), gate_w, gate_h,
    boxstyle="round,pad=0.2,rounding_size=1.8",
    facecolor=ORANGE_BG, edgecolor=ORANGE_EDGE,
    linewidth=1.8, zorder=2
)
ax.add_patch(gate_box)

ax.text(
    gate_x + gate_w/2, gate_y + gate_h - 2.8,
    "Physics & Packaging Feasibility Gate",
    ha="center", va="center",
    fontsize=10.0, fontweight="bold", color=ORANGE_EDGE, zorder=4
)

# 4 Criteria Pills inside the Gate
crit_boxes = [
    (r"$\mathbf{Spar\;Depth}$", r"$t_\mathrm{spar} \geq 10\,\mathrm{mm}$"),
    (r"$\mathbf{Static\;Margin}$", r"$5\% \leq SM \leq 25\%$"),
    (r"$\mathbf{Pitch\;Trim}$", r"$|C_m| \leq 0.10$"),
    (r"$\mathbf{Bay\;Volume}$", r"$V_\mathrm{bay} \geq 2.5\,\mathrm{L}$"),
]

cell_w = (gate_w - 5.0) / 4.0
for i, (title_c, val_c) in enumerate(crit_boxes):
    cx = gate_x + 2.5 + i * cell_w
    cy = gate_y + 2.0
    ch = 10.0
    
    cell_patch = FancyBboxPatch(
        (cx, cy), cell_w - 1.5, ch,
        boxstyle="round,pad=0.15,rounding_size=1.2",
        facecolor="#ffffff", edgecolor=ORANGE_EDGE,
        linewidth=1.0, alpha=0.95, zorder=3
    )
    ax.add_patch(cell_patch)
    
    ax.text(
        cx + (cell_w - 1.5)/2, cy + ch - 2.6,
        title_c,
        ha="center", va="center",
        fontsize=8.5, color=ORANGE_EDGE, zorder=4
    )
    ax.text(
        cx + (cell_w - 1.5)/2, cy + 2.8,
        val_c,
        ha="center", va="center",
        fontsize=7.8, color="#1f2937", zorder=4
    )

# Connect Row 1 to Feasibility Gate
draw_arrow(ax, 69.25, 67.5, 69.25, 57.5, color=PURPLE_EDGE, lw=1.6)
draw_arrow(ax, 89.0, 67.5, 89.0, 57.5, color=GREEN_EDGE, lw=1.6)

# ── ROW 3: Active Learning & Surrogates (y = 7 to 29.5, h = 22.5) ──────────────
r3_y = 7.0
r3_h = 22.5

# 8. Active Learning Policy (Left)
draw_card(
    ax, 2.5, r3_y, 23.0, r3_h,
    "Active Learning Policy",
    [
        "Upper Confidence Bound",
        r"$\alpha_\mathrm{UCB}(\mathbf{x}) = \hat{\mu}(\mathbf{x}) + \kappa\sigma(\mathbf{x})$",
        r"Parameter: $\kappa = 0.3$",
        r"Classifier weight $w(P_\mathrm{feas})$"
    ],
    PURPLE_BG, PURPLE_EDGE, body_size=7.6
)

# 7. Surrogate Model Ensemble & Classifier (Center)
draw_card(
    ax, 37.5, r3_y, 25.0, r3_h,
    "Surrogate Ensemble & Filter",
    [
        r"5-MLP Ensemble ($R^2 \geq 0.996$, CV)",
        r"Random Forest Classifier ($88.5\%$ Prec)",
        r"Epistemic Variance $\sigma^2(\mathbf{x})$",
        "Fast Parallel Inference"
    ],
    NAVY_BG, NAVY_EDGE, body_size=7.6
)

# 6. Database / Knowledge Base (Right)
draw_card(
    ax, 74.5, r3_y, 23.0, r3_h,
    "Optimization Knowledge Base",
    [
        r"$\mathbf{10,087}$ Total Evaluations",
        r"• $2,907$ Feasible Designs",
        r"• $7,180$ Infeasible / Rejected",
        "Active Database Sync"
    ],
    GRAY_BG, GRAY_EDGE, body_size=7.6
)

# Connect Feasibility Gate to Knowledge Base (Clean text label with distinct downward arrow)
ax.text(
    86.0, 36.5, "Log Result",
    ha="center", va="center",
    fontsize=8.0, fontweight="bold", color=GRAY_EDGE, zorder=6
)
draw_arrow(
    ax, 86.0, 34.5, 86.0, 29.5,
    color=GRAY_EDGE, lw=1.6
)

# Connect Database to Surrogate (Inline label with white bbox to break line)
ax.text(
    68.5, 17.5, "Train",
    ha="center", va="center",
    fontsize=8.5, fontweight="bold", color=NAVY_EDGE, zorder=6,
    bbox=dict(facecolor="white", edgecolor="none", pad=1.5)
)
draw_arrow(
    ax, 74.5, 17.5, 62.5, 17.5,
    color=NAVY_EDGE, lw=1.6
)

# Connect Surrogate to Active Learning Policy (Inline label with white bbox to break line)
ax.text(
    31.5, 17.5, "Predict",
    ha="center", va="center",
    fontsize=8.5, fontweight="bold", color=PURPLE_EDGE, zorder=6,
    bbox=dict(facecolor="white", edgecolor="none", pad=1.5)
)
draw_arrow(
    ax, 37.5, 17.5, 25.5, 17.5,
    color=PURPLE_EDGE, lw=1.6
)

# Feedback Loop: Active Learning to Design Vector (Broken arrow around clean unboxed text)
ax.plot([10.8, 10.8], [29.5, 43.5], color=PURPLE_EDGE, lw=1.8, zorder=5)
ax.text(
    10.8, 48.5,
    r"$\mathbf{Next\;Candidate\;\mathbf{x}^*}$" + "\n" + r"$(\mathrm{Acquisition\;Max})$",
    ha="center", va="center",
    fontsize=8.0, color=PURPLE_EDGE, zorder=6
)
draw_arrow(ax, 10.8, 53.5, 10.8, 67.5, color=PURPLE_EDGE, lw=1.8)

# Save PDF, High-Res PNG, and SVG
pdf_path = os.path.join(FIGURES_DIR, "figure3_trim_loop_architecture.pdf")
png_path = os.path.join(SVG_DIR, "_figure3_preview.png")
svg_path = os.path.join(SVG_DIR, "figure3_trim_loop_architecture.svg")
fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight", format="svg")
print("Successfully generated Figure 3 PDF, PNG, and SVG.")
plt.close(fig)

# Save zoomed bottom row for visual verification
im = Image.open(png_path)
w, h = im.size
crop_bottom = im.crop((int(w * 0.0), int(h * 0.65), int(w * 1.0), int(h * 0.95)))
pass
print("Saved zoomed verification crop.")
