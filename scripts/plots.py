"""Generate all figures in plots/ from results/*.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qahvit.model import ViT
from qahvit.hw import predict_latency

# ---- palette (validated: scripts/validate_palette.js, light surface) --------
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GROUP = {  # color, marker
    "baseline": (INK2, "*"),
    "quant only": (S[1], "s"),
    "prune only": (S[2], "o"),
    "prune+quant (sequential)": (MUTED, "D"),
    "QAH-ViT (joint)": (S[0], "o"),
}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 130,
})
os.makedirs("plots", exist_ok=True)


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"plots/{name}", bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/" + name)


# ---------------------------------------------------------------- fig 1
def fig_pruning_methods(search):
    sw = search["sweep"]
    base = [r for r in sw if r["ratio"] == 1.0][0]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    names = {"evit+fuse": "CLS-attention + fuse (EViT)", "evit": "CLS-attention, drop",
             "norm": "Token L2-norm", "random": "Random"}
    for i, m in enumerate(["evit+fuse", "evit", "norm", "random"]):
        rows = sorted([base] + [r for r in sw if r["method"] == m and r["ratio"] < 1], key=lambda r: r["gmacs"])
        x, y = [r["gmacs"] for r in rows], [r["acc"] for r in rows]
        ax.plot(x, y, color=S[i], marker="o", ms=5, label=names[m], zorder=3)
        ax.annotate(names[m], (x[0], y[0]), xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=INK2)
    for r in sorted([r for r in sw if r["method"] == "evit+fuse" and r["ratio"] >= 0.5], key=lambda r: r["gmacs"]):
        ax.annotate(f"r={r['ratio']}", (r["gmacs"], r["acc"]), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7, color=MUTED)
    ax.set_xlim(0.15, 1.33)
    ax.set_xlabel("GMACs per image")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Training-free token pruning: which tokens to keep matters")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "fig1_pruning_methods.png")


# ---------------------------------------------------------------- fig 2
def fig_lut(lut):
    N = np.array(lut["N"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lab = {"fp32": "FP32", "int8_dyn": "INT8 dynamic", "int8_static": "INT8 static (W8A8)"}
    ax = axes[0]
    for i, k in enumerate(["fp32", "int8_dyn", "int8_static"]):
        tot = np.array(lut[k]["attn"]) + np.array(lut[k]["mlp"])
        ax.plot(N, tot, color=S[i], lw=1.6, label=lab[k])
    ax.set_xlabel("Sequence length N (tokens)")
    ax.set_ylabel("Latency of one block (ms)")
    ax.set_title("Latency of one transformer block")
    ax.legend(loc="upper left")
    ax = axes[1]
    fp = np.array(lut["fp32"]["attn"]) + np.array(lut["fp32"]["mlp"])
    for i, k in enumerate(["int8_dyn", "int8_static"], start=1):
        tot = np.array(lut[k]["attn"]) + np.array(lut[k]["mlp"])
        ax.plot(N, fp / tot, color=S[i], lw=1.4, label=lab[k])
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Sequence length N (tokens)")
    ax.set_ylabel("Speed-up over FP32 (x)")
    ax.set_title("INT8 speed-up over FP32 (<1 = slower)")
    ax.legend(loc="upper left")
    fig.suptitle("Measured on Intel i5-10210U, PyTorch 2.4 (fbgemm), 4 threads, batch 1",
                 x=0.01, ha="left", fontsize=9, color=INK2, y=1.0)
    save(fig, "fig2_latency_lut.png")

    # zoom: fine-grained structure (hardware-friendly lengths)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sel = (N >= 40) & (N <= 140)
    m8 = sel & (N % 8 == 0)
    for i, (k, lab_) in enumerate([("fp32", "FP32 block"), ("int8_static", "INT8 static block")]):
        tot = np.array(lut[k]["attn"]) + np.array(lut[k]["mlp"])
        ax.plot(N[sel], tot[sel], color=S[i * 2], lw=1.3, marker="o", ms=2.2, label=lab_)
        ax.scatter(N[m8], tot[m8], s=36, facecolor="none", edgecolor=INK2, lw=1.2, zorder=4,
                   label="N multiple of 8" if i == 0 else None)
    ax.set_xlabel("Sequence length N (tokens)")
    ax.set_ylabel("Latency of one block (ms)")
    ax.set_title("Fine-grained LUT: latency is not linear in N")
    ax.legend(loc="upper left")
    save(fig, "fig3_lut_staircase.png")


# ---------------------------------------------------------------- fig 4
def fig_search(search):
    sp = [c for c in search["space"] if c.get("val_acc_1k") is not None]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter([c["pred_ms"] for c in sp], [c["val_acc_1k"] for c in sp], s=14, color=GRID,
               edgecolor=MUTED, lw=0.5, label="evaluated candidate schedules", zorder=2)
    for j, s in enumerate(search["selected"]):
        ax.axvline(s["budget_ms"], color=MUTED, ls="--", lw=1, zorder=1)
        b, u = s["best"], s["uniform"]
        ax.scatter(b["pred_ms"], b["val_acc_1k"], s=70, color=S[0], edgecolor=SURF, lw=1.5, zorder=4,
                   label="selected (best within budget)" if j == 0 else None)
        ax.scatter(u["pred_ms"], u["val_acc_1k"], s=55, marker="s", color=S[1], edgecolor=SURF, lw=1.5,
                   zorder=4, label="best uniform schedule" if j == 0 else None)
        ax.annotate(f"{s['speedup']}x budget\n" + ",".join(str(r) for r in b["ratios"]) + f" (m={b['mult']})",
                    (b["pred_ms"], b["val_acc_1k"]), xytext=(6, 6), textcoords="offset points",
                    fontsize=7.5, color=INK2)
    lo, hi = ax.get_ylim()
    for s in search["selected"]:
        ax.text(s["budget_ms"], lo + 0.02 * (hi - lo), f" {s['budget_ms']:.0f} ms", fontsize=7.5, color=MUTED)
    ax.set_xlabel("Predicted INT8 latency from LUT (ms, batch 1)")
    ax.set_ylabel("Val accuracy, 1k images (%)")
    ax.set_title("Hardware-aware schedule search (keep ratios at blocks 3/6/9)")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig4_hw_search.png")


# ---------------------------------------------------------------- fig 5-7
def short(n):
    return n.replace(" (dyn deploy)", " dyn")


LABEL_OFF = {
    "FP32 baseline": (-8, 4, "right"), "PTQ INT8 dynamic": (0, 9, "center"),
    "QAT INT8 static": (-2, -13, "right"), "PTQ INT8 static": (4, -13, "center"),
    "QAH-ViT 1.25x (dyn deploy)": (-7, -3, "right"), "QAH-ViT 1.25x": (7, -3, "left"),
}


def fig_final(final, lut):
    fig, ax = plt.subplots(figsize=(8, 5))
    seen = set()
    for r in final:
        c, mk = GROUP[r["group"]]
        ax.scatter(r["lat_b1_ms"], r["test_acc"], s=150 if mk == "*" else 60, marker=mk, color=c,
                   edgecolor=SURF, lw=1.5, zorder=4, label=r["group"] if r["group"] not in seen else None)
        seen.add(r["group"])
        off = LABEL_OFF.get(r["name"], (6, -3, "left"))
        ax.annotate(short(r["name"]), (r["lat_b1_ms"], r["test_acc"]), xytext=off[:2],
                    textcoords="offset points", fontsize=7.5, color=INK2, ha=off[2])
    q = sorted([r for r in final if r["group"] == "QAH-ViT (joint)" and "dyn" not in r["name"]],
               key=lambda r: r["lat_b1_ms"])
    ax.plot([r["lat_b1_ms"] for r in q], [r["test_acc"] for r in q], color=S[0], lw=1.2, zorder=3)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.3)
    ax.set_xlabel("Latency, batch 1, 4 threads (ms)  -  lower is better")
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("Accuracy vs. measured CPU latency")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig5_accuracy_vs_latency.png")

    metrics = [("lat_b1_ms", "Latency b=1 (ms)"), ("throughput_img_s", "Throughput b=32 (img/s)"),
               ("gmacs", "Compute (GMACs)"), ("size_mb", "Model size (MB)"),
               ("act_traffic_mb", "Activation traffic per image (MB)"), ("test_acc", "Test accuracy (%)")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
    names = [short(r["name"]) for r in final]
    y = np.arange(len(final))
    for ax, (k, t) in zip(axes.flat, metrics):
        vals = [r[k] for r in final]
        ax.barh(y, vals, color=[GROUP[r["group"]][0] for r in final], height=0.7, edgecolor=SURF, lw=2)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.1f}" if v < 100 else f" {v:.0f}", va="center", fontsize=7.5, color=INK2)
        ax.set_title(t)
        ax.grid(axis="y", visible=False)
        if k == "test_acc":
            ax.set_xlim(min(vals) - 3, 100)
        else:
            ax.set_xlim(0, max(vals) * 1.18)
    axes[0, 0].set_yticks(y, names, fontsize=8)
    axes[1, 0].set_yticks(y, names, fontsize=8)
    axes[0, 0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP[g][0]) for g in GROUP]
    fig.legend(handles, list(GROUP), loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
    save(fig, "fig6_all_metrics.png")

    # LUT prediction vs measured
    if lut:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        errs = []
        for r in final:
            m = ViT(num_classes=10)
            m.prune_cfg = {int(k): v for k, v in r.get("prune", {}).items()}
            m.token_multiple = r.get("mult", 1)
            prec = {"fp32": "fp32", "dyn": "int8_dyn", "static": "int8_static"}[r["deploy"]]
            p = predict_latency(m, lut, prec)
            c, mk = GROUP[r["group"]]
            ax.scatter(p, r["lat_b1_ms"], color=c, marker=mk, s=160 if mk == "*" else 60, edgecolor=SURF, zorder=3)
            errs.append(abs(p - r["lat_b1_ms"]) / r["lat_b1_ms"])
        ax.text(0.04, 0.93, f"mean abs. error {100*np.mean(errs):.1f}%", transform=ax.transAxes, fontsize=9, color=INK2)
        lim = [30, 56]
        ax.plot(lim, lim, color=MUTED, ls="--", lw=1)
        ax.text(lim[1] * 0.97, lim[1] * 0.9, "y = x", ha="right", fontsize=8, color=MUTED)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("LUT-predicted latency (ms)")
        ax.set_ylabel("Measured end-to-end latency (ms)")
        ax.set_title("Latency model check")
        save(fig, "fig7_lut_prediction.png")


def fig_training():
    runs = [("baseline_fp32", "FP32 fine-tune (from ImageNet)", S[0]), ("qat_only", "QAT only", S[1]),
            ("prune_ft_1.25", "Prune 1.25x + FT (FP32)", S[2]), ("qah_1.25", "QAH-ViT 1.25x (prune + QAT)", S[3])]
    fig, ax = plt.subplots(figsize=(7, 4))
    for f, lab, c in runs:
        d = load(f"checkpoints/{f}.json")
        if not d:
            continue
        st = np.array([l["step"] for l in d["log"]]); lo = np.array([l["loss"] for l in d["log"]])
        k = 5
        sm = np.convolve(lo, np.ones(k) / k, mode="valid")
        ax.plot(st[k - 1:], sm, color=c, label=f"{lab} (val {d['val_acc']:.1f}%)")
    ax.set_xlabel("Training step (batch 16)")
    ax.set_ylabel("Train loss (label smoothing 0.1, smoothed)")
    ax.set_title("Fine-tuning curves")
    ax.legend(fontsize=8)
    save(fig, "fig8_training.png")


if __name__ == "__main__":
    search, lut, final = load("results/search.json"), load("results/latency_lut.json"), load("results/final.json")
    if search:
        fig_pruning_methods(search)
        if "selected" in search:
            fig_search(search)
    if lut:
        fig_lut(lut)
    if final:
        fig_final(final, lut)
    fig_training()
