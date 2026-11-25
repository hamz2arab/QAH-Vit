"""Final benchmark of every variant on the full CIFAR-10 test set (10k).

Metrics: top-1 accuracy, GMACs, parameters, serialized model size, analytical
activation memory, measured peak process memory (separate subprocess),
latency (batch 1 @ 4 threads, batch 1 @ 1 thread) and throughput (batch 32).
"""
import argparse, copy, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from qahvit.model import load_finetuned
from qahvit.data import splits, batches
from qahvit import metrics as M
from qahvit.quant import (convert_dynamic, convert_static, set_quant_mode,
                          set_calibrating, reset_observers)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def variants():
    sel = json.load(open(os.path.join(ROOT, "results/search.json")))
    mult = sel["hw_mult"]
    cfg = {s["speedup"]: s["best"] for s in sel["selected"]}

    def pc(sp):
        c = cfg[sp]
        return {"prune": {l: r for l, r in zip((3, 6, 9), c["ratios"]) if r < 1.0}, "mult": c["mult"]}

    B = "checkpoints/baseline_fp32.pt"
    V = [
        dict(name="FP32 baseline", group="baseline", ckpt=B, deploy="fp32"),
        dict(name="PTQ INT8 dynamic", group="quant only", ckpt=B, deploy="dyn"),
        dict(name="PTQ INT8 static", group="quant only", ckpt=B, deploy="static", calibrate=True),
        dict(name="QAT INT8 static", group="quant only", ckpt="checkpoints/qat_only.pt", deploy="static"),
        dict(name="Prune 1.25x (no FT)", group="prune only", ckpt=B, deploy="fp32", **pc(1.25)),
        dict(name="Prune 1.25x + FT", group="prune only", ckpt="checkpoints/prune_ft_1.25.pt", deploy="fp32", **pc(1.25)),
        dict(name="Prune 1.25x + FT + PTQ", group="prune+quant (sequential)", ckpt="checkpoints/prune_ft_1.25.pt",
             deploy="static", calibrate=True, **pc(1.25)),
    ]
    for sp in (1.15, 1.25, 1.35):
        V.append(dict(name=f"QAH-ViT {sp}x", group="QAH-ViT (joint)", ckpt=f"checkpoints/qah_{sp}.pt",
                      deploy="static", **pc(sp)))
    V.append(dict(name="QAH-ViT 1.25x (dyn deploy)", group="QAH-ViT (joint)", ckpt="checkpoints/qah_1.25.pt",
                  deploy="dyn", **pc(1.25)))
    return V


def build(v, xcal=None):
    m = load_finetuned(os.path.join(ROOT, v["ckpt"]))
    m.prune_cfg = {int(k): r for k, r in v.get("prune", {}).items()}
    m.token_multiple = v.get("mult", 1)
    if v["deploy"] == "static":
        set_quant_mode(m, "fake")
        if v.get("calibrate"):
            reset_observers(m); set_calibrating(m, True); m.eval()
            if xcal is None:
                (xtr, ytr), _, _ = splits()
                xcal = xtr[:512]
            with torch.no_grad():
                for xb, _ in batches(xcal, torch.zeros(len(xcal)), 32):
                    m(xb)
            set_calibrating(m, False)
        m = convert_static(m)
    elif v["deploy"] == "dyn":
        m = convert_dynamic(m)
    return m.eval()


def mem_probe(idx):
    """Run in a fresh process. A sampler thread records RSS every ~0.2 ms while
    the model runs, so we get the extra memory used *by inference itself*
    (activations + kernel workspaces), separate from weights and data loading."""
    import gc, threading, time, psutil
    torch.set_num_threads(4)
    proc = psutil.Process()
    v = variants()[idx]
    xcal = None
    if v.get("calibrate"):
        (xtr, _), _, _ = splits()
        xcal = xtr[:512].clone(); del xtr
    rss_a = proc.memory_info().rss
    m = build(v, xcal); del xcal; gc.collect()
    rss_model = proc.memory_info().rss - rss_a
    res = {"weights_rss_mb": rss_model / 2**20}
    for b in (1, 32):
        x = torch.randn(b, 3, 224, 224)
        with torch.no_grad():
            m(x)  # warm-up (allocator caches, packed weights)
        gc.collect()
        base = proc.memory_info().rss
        peak, stop = [base], [False]

        def sampler():
            while not stop[0]:
                peak[0] = max(peak[0], proc.memory_info().rss)
                time.sleep(0.0002)
        th = threading.Thread(target=sampler); th.start()
        with torch.no_grad():
            for _ in range(3):
                m(x)
        stop[0] = True; th.join()
        res[f"infer_mem_b{b}_mb"] = (peak[0] - base) / 2**20 + 0.0
    # torch's CPU allocator caches freed blocks, so "base" after warm-up already
    # holds the arena; report also the warm-up-inclusive total
    res["total_rss_mb"] = proc.memory_info().rss / 2**20
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem_probe", type=int, default=-1)
    ap.add_argument("--only", default="")
    ap.add_argument("--mem_only", action="store_true", help="re-measure memory for existing rows")
    args = ap.parse_args()
    if args.mem_probe >= 0:
        return mem_probe(args.mem_probe)

    if args.mem_only:
        out_path = os.path.join(ROOT, "results/final.json")
        rows = json.load(open(out_path))
        names = [v["name"] for v in variants()]
        for r in rows:
            for k in ("peak_rss_mb", "rss_model_mb"):
                r.pop(k, None)
            probe = subprocess.run([PY, os.path.abspath(__file__), "--mem_probe", str(names.index(r["name"]))],
                                   capture_output=True, text=True, cwd=ROOT)
            r.update(json.loads(probe.stdout.strip().splitlines()[-1]))
            print(r["name"], {k: round(r[k], 1) for k in ("weights_rss_mb", "infer_mem_b1_mb", "infer_mem_b32_mb", "total_rss_mb")}, flush=True)
        json.dump(rows, open(out_path, "w"), indent=1)
        return
    torch.set_num_threads(4)
    (xtr, _), _, (xte, yte) = splits()
    xcal = xtr[:512].clone(); del xtr
    out_path = os.path.join(ROOT, "results/final.json")
    rows = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {r["name"] for r in rows}
    for i, v in enumerate(variants()):
        if v["name"] in done or (args.only and args.only not in v["name"]):
            continue
        m = build(v, xcal)
        r = dict(v)
        r["test_acc"] = M.accuracy(m, xte, yte)
        r["gmacs"] = M.macs(m)
        r["tokens_mlp"] = m.token_schedule()[1]
        r["params_m"] = M.params(m) / 1e6 if v["deploy"] == "fp32" else M.params(load_finetuned(os.path.join(ROOT, v["ckpt"]))) / 1e6
        r["size_mb"] = M.model_size_mb(m)
        r["act_mem_mb_b1"] = M.activation_memory_mb(m, 1)
        r["act_mem_mb_b32"] = M.activation_memory_mb(m, 32)
        r["lat_b1_ms"], r["lat_b1_std"] = M.latency_ms(m, 1, 20, 100)
        torch.set_num_threads(1)
        r["lat_b1_1t_ms"], _ = M.latency_ms(m, 1, 5, 30)
        torch.set_num_threads(4)
        r["lat_b32_ms"], _ = M.latency_ms(m, 32, 2, 8)
        r["throughput_img_s"] = 32 * 1000 / r["lat_b32_ms"]
        probe = subprocess.run([PY, os.path.abspath(__file__), "--mem_probe", str(i)],
                               capture_output=True, text=True, cwd=ROOT)
        r.update(json.loads(probe.stdout.strip().splitlines()[-1]))
        rows.append(r)
        print({k: r[k] for k in ("name", "test_acc", "gmacs", "size_mb", "lat_b1_ms", "lat_b32_ms", "peak_rss_mb")},
              flush=True)
        json.dump(rows, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
