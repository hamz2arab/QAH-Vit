"""Show which image patches survive each pruning stage (QAH-ViT 2x model)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qahvit.model import load_finetuned
from qahvit.data import splits, preprocess
from qahvit.quant import set_quant_mode

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def traced_forward(m, x):
    """Same computation as ViT.forward, but tracks original patch positions."""
    x = m.patch_embed(x)
    B = x.shape[0]
    x = torch.cat([m.cls_token.expand(B, -1, -1), x], 1) + m.pos_embed
    pos = torch.arange(196).expand(B, -1)  # original patch index of each patch token (-1 = fused)
    stages = []
    for i, blk in enumerate(m.blocks):
        if i not in m.prune_cfg:
            x = blk(x)
            continue
        a, cls_attn = blk.attn(blk.norm1(x), need_cls_attn=True)
        x = x + a
        k = m.keep_count(m.prune_cfg[i], x.shape[1] - 1)
        tok = x[:, 1:]
        idx = cls_attn.topk(k, dim=1).indices
        kept = tok.gather(1, idx.unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
        w = cls_attn.scatter(1, idx, 0.0); w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
        fused = (w.unsqueeze(-1) * tok).sum(1, keepdim=True)
        pos = torch.cat([pos.gather(1, idx), torch.full((B, 1), -1)], 1)
        x = torch.cat([x[:, :1], kept, fused], 1)
        stages.append((i, pos.clone()))
        x = x + blk.mlp(blk.norm2(x))
    return m.head(m.norm(x)[:, 0]), stages


def main():
    sel = json.load(open("results/search.json"))
    best = [s for s in sel["selected"] if s["speedup"] == 1.25][0]["best"]
    m = load_finetuned("checkpoints/qah_1.25.pt").eval()
    set_quant_mode(m, "fake")
    m.prune_cfg = {l: r for l, r in zip((3, 6, 9), best["ratios"]) if r < 1}
    m.token_multiple = best["mult"]
    _, _, (xte, yte) = splits()
    ids = [3, 10, 25, 36, 52, 70]
    xb = preprocess(xte[ids])
    logits, stages = traced_forward(m, xb)
    pred = logits.argmax(1)
    img = F.interpolate(xte[ids].float() / 255, size=224, mode="bilinear", align_corners=False)
    ncol = 1 + len(stages)
    fig, axes = plt.subplots(len(ids), ncol, figsize=(2.1 * ncol, 2.1 * len(ids)))
    for r in range(len(ids)):
        im = img[r].permute(1, 2, 0).numpy()
        axes[r, 0].imshow(im)
        axes[r, 0].set_title(f"{CLASSES[yte[ids[r]]]} -> {CLASSES[pred[r]]}", fontsize=8)
        for c, (blk, pos) in enumerate(stages, start=1):
            mask = np.zeros(196)
            p = pos[r][pos[r] >= 0].numpy()
            mask[p] = 1
            mask = np.kron(mask.reshape(14, 14), np.ones((16, 16)))[..., None]
            axes[r, c].imshow(im * (0.15 + 0.85 * mask) + (1 - mask) * 0.0)
            if r == 0:
                axes[r, c].set_title(f"after block {blk}\n{len(p)} / 196 patches", fontsize=8)
    for a in axes.flat:
        a.axis("off")
    fig.suptitle("Patches kept by attentive token pruning (QAH-ViT 1.25x)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig("plots/fig9_token_visualization.png", dpi=120, facecolor="#fcfcfb")
    print("wrote plots/fig9_token_visualization.png")


if __name__ == "__main__":
    main()
