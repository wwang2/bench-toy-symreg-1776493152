"""Generate results.png and narrative.png for the sin+quad fit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

COL_DATA = "#4C72B0"
COL_FIT = "#C44E52"
COL_SIN = "#55A868"
COL_QUAD = "#DD8452"

# Load training data
train = np.loadtxt(ROOT / "research" / "eval" / "train_data.csv", delimiter=",", skiprows=1)
x_tr, y_tr = train[:, 0], train[:, 1]

x_dense = np.linspace(-4, 4, 400)
f_fit = np.sin(x_dense) + 0.1 * x_dense**2
sin_part = np.sin(x_dense)
quad_part = 0.1 * x_dense**2

# Residuals at training points
y_pred_tr = np.sin(x_tr) + 0.1 * x_tr**2
resid = y_tr - y_pred_tr
rms = float(np.sqrt(np.mean(resid**2)))

# MSE on clean test grid (same as evaluator)
x_te = np.linspace(-4, 4, 400)
y_te = np.sin(x_te) + 0.1 * x_te**2
mse_clean = float(np.mean((np.sin(x_te) + 0.1 * x_te**2 - y_te) ** 2))

# ---------- results.png ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1.0]})

ax = axes[0]
ax.scatter(x_tr, y_tr, s=28, color=COL_DATA, alpha=0.85, zorder=3, label="training data (n=40, σ=0.03)")
ax.plot(x_dense, f_fit, color=COL_FIT, lw=2.0, zorder=2, label=r"$f(x)=\sin(x)+0.1\,x^2$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Closed-form fit vs. training data")
ax.legend(loc="upper center")
ax.text(0.02, 0.97, f"train RMS residual = {rms:.3f}\ntest MSE = {mse_clean:.2e}",
        transform=ax.transAxes, va="top", fontsize=10, color="#333333")

ax = axes[1]
ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=1)
ax.scatter(x_tr, resid, s=28, color=COL_DATA, alpha=0.85, zorder=3)
ax.axhspan(-0.03, 0.03, color=COL_DATA, alpha=0.08, zorder=0, label=r"±σ = ±0.03 noise band")
ax.set_xlabel("x")
ax.set_ylabel(r"$y - f(x)$")
ax.set_title("Residuals")
ax.legend(loc="upper right")

fig.savefig(HERE / "figures" / "results.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------- narrative.png ----------
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=True)

panels = [
    (axes[0], sin_part, COL_SIN, r"$\sin(x)$", "odd component", "(a)"),
    (axes[1], quad_part, COL_QUAD, r"$0.1\,x^2$", "even envelope", "(b)"),
    (axes[2], f_fit, COL_FIT, r"$\sin(x) + 0.1\,x^2$", "sum matches data", "(c)"),
]

for ax, curve, color, eqn, caption, tag in panels:
    ax.plot(x_dense, curve, color=color, lw=2.2, zorder=2)
    ax.axhline(0, color="#cccccc", lw=0.7, zorder=1)
    ax.set_xlabel("x")
    ax.set_title(f"{caption}\n{eqn}")
    ax.text(-0.08, 1.10, tag, transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_xlim(-4.5, 4.5)

# Expand shared y-limits so annotations + panel tags sit in whitespace
axes[0].set_ylim(-2.1, 3.0)
axes[0].set_ylabel("y")

# Overlay data on the rightmost panel
axes[2].scatter(x_tr, y_tr, s=22, color=COL_DATA, alpha=0.9, zorder=3,
                label="train data (n=40)")
axes[2].legend(loc="lower right")

# Annotations placed in clear whitespace on each panel
axes[0].annotate("odd about x=0\n(flips sign)",
                 xy=(-np.pi/2, -1.0), xytext=(-3.8, -1.9),
                 fontsize=10, color="#333333",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
axes[1].annotate(r"$0.1\cdot 4^2 = 1.6$",
                 xy=(4, 1.6), xytext=(-1.2, 2.5),
                 fontsize=10, color="#333333",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
axes[2].annotate("asymmetry from sin,\ncurvature from $0.1\\,x^2$",
                 xy=(-4, 2.36), xytext=(-2.6, -1.8),
                 fontsize=10, color="#333333",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

fig.suptitle("Decomposing the toy symbolic-regression target", fontsize=15)
fig.savefig(HERE / "figures" / "narrative.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Wrote figures; RMS residual on train = {rms:.4f}")
