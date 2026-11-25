"""Hardware profiling: build a latency look-up table (LUT) on *this* CPU.

For every sequence length N in [2, 197] we time (batch 1, 4 threads):
  * the attention half of a transformer block   x + Attn(LN(x))
  * the MLP half of a transformer block          x + MLP(LN(x))
  * the token-pruning operator (top-k + gather + fuse)
for FP32, INT8-dynamic and INT8-static (W8A8) linears.

The LUT lets the search predict end-to-end latency of any pruning schedule
without running it, and reveals hardware effects (e.g. sequence lengths that
map well/badly onto the GEMM micro-kernels).
"""
import copy, json, os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from qahvit.model import deit_tiny, prune_tokens
from qahvit.quant import convert_dynamic, convert_static, set_quant_mode, set_calibrating
from qahvit.metrics import latency_ms

torch.set_num_threads(4)
WARM, ITERS = 10, 60


@torch.no_grad()
def t(fn, x):
    for _ in range(WARM):
        fn(x)
    ts = []
    for _ in range(ITERS):
        s = time.perf_counter(); fn(x); ts.append(time.perf_counter() - s)
    return statistics.median(ts) * 1e3


def make_variants():
    base = deit_tiny(10, "checkpoints/deit_tiny_patch16_224.pth").eval()
    calib = copy.deepcopy(base); set_quant_mode(calib, "fake"); set_calibrating(calib, True)
    with torch.no_grad():
        calib(torch.randn(4, 3, 224, 224))
    return {"fp32": base, "int8_dyn": convert_dynamic(copy.deepcopy(base)),
            "int8_static": convert_static(calib)}


def main():
    variants = make_variants()
    Ns = list(range(2, 198))
    lut = {"N": Ns}
    for name, m in variants.items():
        blk = m.blocks[5]
        attn = lambda x: x + blk.attn(blk.norm1(x))[0]
        mlp = lambda x: x + blk.mlp(blk.norm2(x))
        lut[name] = {"attn": [], "mlp": []}
        for n in Ns:
            x = torch.randn(1, n, 192)
            lut[name]["attn"].append(t(attn, x))
            lut[name]["mlp"].append(t(mlp, x))
        # fixed costs: patch embedding + pos-embed, final norm + head
        pe = lambda x: m.patch_embed(x)
        lut[name]["patch_embed"] = t(pe, torch.randn(1, 3, 224, 224))
        lut[name]["head"] = t(lambda x: m.head(m.norm(x)[:, 0]), torch.randn(1, 2, 192))
        lut[name]["end2end_unpruned"] = latency_ms(m, 1, WARM, ITERS)[0]
        print(name, "attn@197 %.2f mlp@197 %.2f e2e %.1f ms" %
              (lut[name]["attn"][-1], lut[name]["mlp"][-1], lut[name]["end2end_unpruned"]), flush=True)
    # pruning operator overhead
    lut["prune_op"] = []
    for n in Ns:
        x = torch.randn(1, n, 192); a = torch.rand(1, n - 1).softmax(-1)
        k = max(1, int(0.6 * (n - 1)))
        lut["prune_op"].append(t(lambda x: prune_tokens(x, k, "evit", a, True), x))
    os.makedirs("results", exist_ok=True)
    json.dump(lut, open("results/latency_lut.json", "w"))
    print("saved results/latency_lut.json")


if __name__ == "__main__":
    main()
