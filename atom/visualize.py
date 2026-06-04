"""Turn a training flip-book into something a human can read at a glance.

The unit of display is a **dashboard cell** with four panels:

  * network diagram — circles are neurons (coloured by their value, with the
    number printed inside and the bias below); lines are weights (red +, blue −,
    thick when large, with the value printed on the line).
  * loss curve — how wrong the network is, falling over epochs.
  * "what the network learned" — the function it represents right now.
  * "the truth" — the function it is trying to match.

Below the network we also print the cell's **starting weights** — the actual
numbers it drew from its seed. The live weights on the diagram animate away from
those; the block is the fixed starting point, and it shows why two cells under
the same grid seed still begin differently (each drew a different slice).

`draw_dashboard()` fills one cell. The single-network view is exactly one cell;
the capacity grid (see atom/grid.py) is a whole grid of identical cells. The
drawing is generic over architecture, so a 2-2-1 and a 2-3-3-1 render the same
way — only bigger.
"""

import matplotlib
matplotlib.use("Agg")  # headless: render straight to files, no window needed

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation, PillowWriter

WEIGHT_CMAP = plt.get_cmap("coolwarm")   # weights: blue (−) … red (+)
ACT_CMAP = plt.get_cmap("RdBu_r")        # neuron activation, diverging at 0

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _sub(i: int) -> str:
    return str(i).translate(_SUB)


def _text_color(rgba):
    r, g, b, _ = rgba
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def scales_from_frames(frames):
    """Fixed weight/activation colour scales across every frame (and every cell)
    so a colour means the same thing everywhere."""
    w_max, a_max = 1e-6, 1e-6
    for f in frames:
        for (W, _b) in f["snap"]:
            w_max = max(w_max, W.abs().max().item())
        for act in f["trace"]:
            a_max = max(a_max, act.abs().max().item())
    return Normalize(-w_max, w_max), Normalize(-a_max, a_max)


# --------------------------------------------------------------------------- #
# network diagram — generic over any width/depth, fully labelled
# --------------------------------------------------------------------------- #
def _positions(sizes, y_lo=0.05, y_hi=0.95):
    L = len(sizes)
    xs = [l / (L - 1) for l in range(L)] if L > 1 else [0.5]
    dy = (y_hi - y_lo) / max(max(sizes), 1)
    pos = []
    for l, n in enumerate(sizes):
        ys = [0.5 + (i - (n - 1) / 2) * dy for i in range(n)]
        pos.append([(xs[l], y) for y in ys])
    return pos, dy


def draw_network(ax, sizes, snap, trace, w_norm, a_norm):
    ax.clear()
    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(-0.16, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")

    pos, dy = _positions(sizes)
    node_r = min(0.40 * dy, 0.06)
    L = len(sizes)

    # edges = weights (with value labels)
    for l in range(L - 1):
        W = snap[l][0]  # (out, in)
        for j, (x1, y1) in enumerate(pos[l + 1]):
            for i, (x0, y0) in enumerate(pos[l]):
                w = W[j, i].item()
                width = 0.5 + 4.5 * min(abs(w) / max(w_norm.vmax, 1e-6), 1.0)
                ax.plot([x0, x1], [y0, y1], color=WEIGHT_CMAP(w_norm(w)), lw=width,
                        solid_capstyle="round", zorder=1)
                mx, my = 0.58 * x0 + 0.42 * x1, 0.58 * y0 + 0.42 * y1
                ax.text(mx, my, f"{w:+.2f}", fontsize=6, ha="center", va="center",
                        color="#222", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.08", fc="white",
                                  ec="none", alpha=0.72))

    # nodes = neurons (value inside, label above, bias below)
    for l, layer in enumerate(pos):
        act = trace[l][0]
        for i, (x, y) in enumerate(layer):
            value = act[i].item()
            rgba = ACT_CMAP(a_norm(value))
            ax.add_patch(Circle((x, y), node_r, fc=rgba, ec="#333", lw=1.1, zorder=4))
            ax.text(x, y, f"{value:.2f}", fontsize=6.5, ha="center", va="center",
                    color=_text_color(rgba), zorder=5, weight="bold")
            if l == 0:
                label = "x" + _sub(i + 1)
            elif l == L - 1:
                label = "ŷ" if sizes[-1] == 1 else "ŷ" + _sub(i + 1)
            else:
                label = "h" + _sub(i + 1)
            ax.text(x, y + node_r + 0.012, label, fontsize=7, ha="center",
                    va="bottom", color="#444", zorder=5)
            if l > 0:  # bias feeding this node
                bias = snap[l - 1][1][i].item()
                ax.text(x, y - node_r - 0.012, f"b={bias:+.2f}", fontsize=5.5,
                        ha="center", va="top", color="#777", style="italic", zorder=5)

    # column captions
    n_hidden = L - 2
    for l in range(L):
        if l == 0:
            cap = "Inputs"
        elif l == L - 1:
            cap = "Output"
        else:
            cap = "Hidden · ReLU" if n_hidden == 1 else f"Hidden {l} · ReLU"
        ax.text(pos[l][0][0], 1.10, cap, fontsize=7.5, ha="center", va="bottom",
                color="#222", weight="bold")

    inp = trace[0][0]
    probe = ", ".join(f"{v:.0f}" for v in inp.tolist())
    ax.text(0.5, -0.13, f"tracing input  x = ({probe})", fontsize=6.5,
            ha="center", va="bottom", color="#666")


# --------------------------------------------------------------------------- #
# loss + surfaces
# --------------------------------------------------------------------------- #
def draw_loss(ax, frame, total_epochs):
    ax.clear()
    losses = frame["losses"]
    ax.plot(losses, color="#1f77b4", lw=1.6)
    ax.scatter([len(losses) - 1], [losses[-1]], color="#d62728", zorder=5, s=20)
    ax.set_yscale("log")
    ax.set_xlim(0, total_epochs)
    ax.set_title("loss — how wrong the network is", fontsize=9.5)
    ax.set_xlabel("epoch", fontsize=8)
    ax.set_ylabel("loss (log)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.25)
    info = f"epoch {frame['epoch']}\nloss {frame['loss']:.4f}"
    # what trained this particular cell — matters once --rerun gives cells
    # their own lr / seed (see atom/grid.py)
    if frame.get("lr") is not None and frame.get("seed") is not None:
        info += f"\nlr {frame['lr']:g}  ·  seed {frame['seed']}"
    ax.text(0.96, 0.93, info,
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="#fff7e6", ec="#e0c080"))


def draw_surface(ax, data, lim, title, vmin, vmax):
    ax.clear()
    ax.imshow(data, origin="lower", extent=[-lim, lim, -lim, lim],
              cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x₁", fontsize=8)
    ax.set_ylabel("x₂", fontsize=8)
    ax.tick_params(labelsize=7)


# --------------------------------------------------------------------------- #
# the cell's epoch-0 weights, printed verbatim — the *actual numbers* it drew
# from the seed's stream. Lifted out of the loss info box (too bulky) into its
# own block under the network, so two "seed 42" cells visibly start different.
# --------------------------------------------------------------------------- #
def _init_lines(snap):
    """One monospaced line per layer: the weight rows then the bias vector."""
    lines = []
    for l, (W, b) in enumerate(snap):
        rows = ["[" + " ".join(f"{W[i, j].item():+.2f}" for j in range(W.shape[1])) + "]"
                for i in range(W.shape[0])]
        b_str = " ".join(f"{b[i].item():+.2f}" for i in range(b.shape[0]))
        lines.append(f"W{l + 1} {' '.join(rows)}   b{l + 1} [{b_str}]")
    return lines


def draw_init_block(ax_net, frame):
    """Print the cell's starting weights below its network diagram (outside the
    plot). Static across frames — the live weights animate on the diagram; this
    is the fixed starting point they came from."""
    init = frame.get("init")
    if not init:
        return
    head = "init weights — the actual numbers this cell drew"
    seed = frame.get("seed")
    if seed is not None:
        head += f"  (seed {seed}'s slice)"
    text = head + "\n" + "\n".join(_init_lines(init))
    ax_net.text(-0.10, -0.20, text, transform=ax_net.transAxes,
                ha="left", va="top", fontsize=6, family="monospace",
                color="#555", clip_on=False, zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f6f6f4", ec="#cfcfcf",
                          alpha=0.95))


# --------------------------------------------------------------------------- #
# one full dashboard cell (network + loss + learned + truth)
# --------------------------------------------------------------------------- #
def draw_dashboard(axes, frame, ctx, w_norm, a_norm, s_vmin, s_vmax,
                   net_title=None):
    ax_net, ax_loss, ax_pred, ax_true = axes
    lim = ctx["grid_lim"]
    draw_network(ax_net, frame["sizes"], frame["snap"], frame["trace"], w_norm, a_norm)
    if net_title is None:
        net_title = "the network  ·  lines = weights, circles = neurons"
    ax_net.set_title(net_title, fontsize=9.5, pad=6)
    draw_init_block(ax_net, frame)
    draw_loss(ax_loss, frame, ctx["total_epochs"])
    draw_surface(ax_pred, frame["pred"].numpy(), lim, "what the network learned",
                 s_vmin, s_vmax)
    draw_surface(ax_true, ctx["true_surface"].numpy(), lim, "the truth (target)",
                 s_vmin, s_vmax)


def _surface_range(ctx):
    if ctx["task"] == "classification":
        return 0.0, 1.0
    return 0.0, float(ctx["true_surface"].max())


def _make_cell_axes(fig, spec):
    """Carve a (2,4) dashboard out of one gridspec slot; return the 4 axes."""
    sub = spec.subgridspec(2, 4, width_ratios=[1.1, 1.1, 1.0, 1.0],
                           wspace=0.5, hspace=0.5)
    ax_net = fig.add_subplot(sub[:, 0:2])
    ax_loss = fig.add_subplot(sub[0, 2:4])
    ax_pred = fig.add_subplot(sub[1, 2])
    ax_true = fig.add_subplot(sub[1, 3])
    return ax_net, ax_loss, ax_pred, ax_true


def _shared_colorbars(fig, w_norm, s_vmin, s_vmax):
    cax_w = fig.add_axes([0.935, 0.55, 0.012, 0.30])
    fig.colorbar(ScalarMappable(w_norm, WEIGHT_CMAP), cax=cax_w).set_label(
        "weight (− blue / + red)", fontsize=8)
    cax_s = fig.add_axes([0.935, 0.14, 0.012, 0.30])
    fig.colorbar(ScalarMappable(Normalize(s_vmin, s_vmax), "viridis"),
                 cax=cax_s).set_label("output value", fontsize=8)


# --------------------------------------------------------------------------- #
# single-network render (one cell)
# --------------------------------------------------------------------------- #
def render(history, context, out_gif="outputs/training.gif",
           out_png="outputs/final.png", fps=12):
    task = context["task"]
    total_epochs = context["total_epochs"]
    w_norm, a_norm = scales_from_frames(history)
    s_vmin, s_vmax = _surface_range(context)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 1, left=0.05, right=0.90, top=0.88, bottom=0.07)
    axes = _make_cell_axes(fig, gs[0, 0])
    _shared_colorbars(fig, w_norm, s_vmin, s_vmax)

    def update(i):
        draw_dashboard(axes, history[i], context, w_norm, a_norm, s_vmin, s_vmax)
        f = history[i]
        fig.suptitle(
            f"atom_network — watching a {f['nparams']}-parameter brain learn  ·  "
            f"{task}  ·  epoch {f['epoch']}/{total_epochs}  ·  loss {f['loss']:.4f}",
            fontsize=14, weight="bold")
        return []

    anim = FuncAnimation(fig, update, frames=len(history), blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    update(len(history) - 1)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_gif, out_png
