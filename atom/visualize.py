"""Turn a training flip-book into something a human can read at a glance.

The dashboard has four panels:

  * left  — the network itself. Circles are neurons coloured by the value
            they currently hold; lines are weights, red for positive, blue
            for negative, thick when large. You are literally looking at the
            nine numbers being tuned.
  * top-right    — the loss falling as the network learns.
  * bottom-mid   — the function the network represents *right now*.
  * bottom-right — the function it is trying to match (the ground truth).

Watching the bottom-middle morph into the bottom-right, while the weights on
the left shift and the loss on the top-right drops, is the whole story of
learning in one picture.
"""

import matplotlib
matplotlib.use("Agg")  # headless: render straight to files, no window needed

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation, PillowWriter

# Neuron positions, laid out left to right: inputs, hidden, output.
POS = {
    "input": [(0.0, 0.6), (0.0, -0.6)],
    "hidden": [(1.0, 0.6), (1.0, -0.6)],
    "output": [(2.0, 0.0)],
}
NODE_R = 0.17
WEIGHT_CMAP = plt.get_cmap("coolwarm")   # blue (−)  ...  red (+)
ACT_CMAP = plt.get_cmap("RdBu_r")        # neuron activation, diverging at 0


def _text_color(rgba):
    """Pick black or white text for readability over a coloured node."""
    r, g, b, _ = rgba
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.6 else "white"


def _global_scales(history):
    """Fixed colour scales across the whole animation so colours mean the
    same thing in every frame."""
    w_max, a_max = 1e-6, 1e-6
    for f in history:
        w = f["weights"]
        for key in ("W1", "W2"):
            w_max = max(w_max, w[key].abs().max().item())
        t = f["trace"]
        for key in ("input", "hidden_post", "output"):
            a_max = max(a_max, t[key].abs().max().item())
    return w_max, a_max


def _draw_network(ax, frame, w_norm, a_norm):
    ax.clear()
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the network  ·  lines = weights, circles = neurons",
                 fontsize=11, pad=8)

    W = frame["weights"]
    t = frame["trace"]
    inp = t["input"][0]
    hid = t["hidden_post"][0]
    out = t["output"][0]

    # --- weights as edges -------------------------------------------------
    def edge(p0, p1, value):
        color = WEIGHT_CMAP(w_norm(value))
        width = 1.0 + 5.0 * min(abs(value) / max(w_norm.vmax, 1e-6), 1.0)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=width,
                solid_capstyle="round", zorder=1)
        mx, my = 0.55 * p0[0] + 0.45 * p1[0], 0.55 * p0[1] + 0.45 * p1[1]
        ax.text(mx, my, f"{value:+.2f}", fontsize=7, ha="center", va="center",
                color="#222", zorder=3,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))

    for j, hp in enumerate(POS["hidden"]):
        for i, ip in enumerate(POS["input"]):
            edge(ip, hp, W["W1"][j, i].item())
    for j, hp in enumerate(POS["hidden"]):
        edge(hp, POS["output"][0], W["W2"][0, j].item())

    # --- neurons as nodes -------------------------------------------------
    def node(pos, value, label, bias=None):
        rgba = ACT_CMAP(a_norm(value))
        ax.add_patch(Circle(pos, NODE_R, fc=rgba, ec="#333", lw=1.4, zorder=4))
        ax.text(pos[0], pos[1], f"{value:.2f}", fontsize=8.5, ha="center",
                va="center", color=_text_color(rgba), zorder=5, weight="bold")
        ax.text(pos[0], pos[1] + NODE_R + 0.1, label, fontsize=8,
                ha="center", va="bottom", color="#444")
        if bias is not None:
            ax.text(pos[0], pos[1] - NODE_R - 0.1, f"b={bias:+.2f}", fontsize=7,
                    ha="center", va="top", color="#777", style="italic")

    node(POS["input"][0], inp[0].item(), "x₁")
    node(POS["input"][1], inp[1].item(), "x₂")
    node(POS["hidden"][0], hid[0].item(), "h₁", W["b1"][0].item())
    node(POS["hidden"][1], hid[1].item(), "h₂", W["b1"][1].item())
    node(POS["output"][0], out[0].item(), "ŷ", W["b2"][0].item())

    # column captions + the probe input being traced
    for x, name in [(0, "Inputs"), (1, "Hidden · ReLU"), (2, "Output")]:
        ax.text(x, 1.32, name, fontsize=9.5, ha="center", va="bottom",
                color="#222", weight="bold")
    ax.text(1.0, -1.4, f"tracing input  x = ({inp[0]:.0f}, {inp[1]:.0f})",
            fontsize=8, ha="center", va="bottom", color="#666")


def _draw_loss(ax, frame, total_epochs):
    ax.clear()
    losses = frame["losses"]
    ax.plot(losses, color="#1f77b4", lw=1.8)
    ax.scatter([len(losses) - 1], [losses[-1]], color="#d62728", zorder=5, s=24)
    ax.set_yscale("log")
    ax.set_xlim(0, total_epochs)
    ax.set_title("loss — how wrong the network is", fontsize=11)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log)")
    ax.grid(True, which="both", alpha=0.25)
    ax.text(0.97, 0.92, f"epoch {frame['epoch']}\nloss {frame['loss']:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#fff7e6", ec="#e0c080"))


def _draw_surface(ax, data, ctx, title, vmin, vmax):
    ax.clear()
    lim = ctx["grid_lim"]
    im = ax.imshow(data, origin="lower", extent=[-lim, lim, -lim, lim],
                   cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    return im


def render(history, context, out_gif="outputs/training.gif",
           out_png="outputs/final.png", fps=12):
    """Render the flip-book to an animated GIF and a final-state PNG."""
    task = context["task"]
    total_epochs = context["total_epochs"]
    w_max, a_max = _global_scales(history)
    w_norm = Normalize(-w_max, w_max)
    a_norm = Normalize(-a_max, a_max)

    true = context["true_surface"]
    if task == "classification":
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, float(true.max())

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1.1, 1.0, 1.0],
                          height_ratios=[1, 1], hspace=0.28, wspace=0.32)
    ax_net = fig.add_subplot(gs[:, 0:2])
    ax_loss = fig.add_subplot(gs[0, 2:4])
    ax_pred = fig.add_subplot(gs[1, 2])
    ax_true = fig.add_subplot(gs[1, 3])

    # one shared colourbar for the two surface panels
    sm = ScalarMappable(norm=Normalize(vmin, vmax), cmap="viridis")
    cbar = fig.colorbar(sm, ax=[ax_pred, ax_true], fraction=0.046, pad=0.04)
    cbar.set_label("output value", fontsize=8)

    # one colourbar explaining the weight colours on the network
    sm_w = ScalarMappable(norm=w_norm, cmap=WEIGHT_CMAP)
    cbar_w = fig.colorbar(sm_w, ax=ax_net, fraction=0.04, pad=0.02,
                          location="left")
    cbar_w.set_label("weight value  (blue −  /  red +)", fontsize=8)

    true_np = true.numpy()

    def update(i):
        frame = history[i]
        _draw_network(ax_net, frame, w_norm, a_norm)
        _draw_loss(ax_loss, frame, total_epochs)
        _draw_surface(ax_pred, frame["pred_surface"].numpy(), context,
                      "what the network learned", vmin, vmax)
        _draw_surface(ax_true, true_np, context,
                      "the truth (target)", vmin, vmax)
        fig.suptitle(
            f"atom_network — watching a 9-parameter brain learn  ·  "
            f"{task}  ·  epoch {frame['epoch']}/{total_epochs}  ·  "
            f"loss {frame['loss']:.4f}",
            fontsize=14, weight="bold")
        return []

    anim = FuncAnimation(fig, update, frames=len(history), blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))

    # final frame as a crisp still
    update(len(history) - 1)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_gif, out_png
