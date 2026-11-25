"""(A) Training-free token-pruning study and (B) hardware-aware schedule search.

(A) uniform keep-ratio r at blocks {3,6,9} for several token scoring methods
    -> accuracy vs GMACs (on the 2k validation split, FP32 baseline weights).
(B) search space: keep ratio r_i in {0.4..1.0} at each of blocks 3, 6, 9
    (343 schedules) x token rounding {1 (none), hardware multiple}.
    Every schedule's batch-1 INT8 latency is *predicted* from the LUT (free);
    for each latency budget only the "maximal" schedules (any larger ratio would
    break the budget) are evaluated for accuracy; the best one is selected.
"""
import itertools, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from qahvit.model import load_finetuned
from qahvit.data import splits
from qahvit.metrics import accuracy, macs
from qahvit.hw import load_lut, predict_latency

torch.set_num_threads(4)
LOCS = (3, 6, 9)
RATIOS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SPEEDUPS = [1.15, 1.25, 1.35]  # latency budgets = FP32-unpruned latency / speed-up
PRECISION = "int8_static"


def set_cfg(m, ratios, mult=1, method="evit", fuse=True):
    m.prune_cfg = {l: r for l, r in zip(LOCS, ratios) if r < 1.0}
    m.token_multiple, m.prune_method, m.fuse = mult, method, fuse


def main():
    hw_mult = int(os.environ.get("HW_MULT", 8))
    _, (xva, yva), _ = splits()
    model = load_finetuned("checkpoints/baseline_fp32.pt").eval()
    lut = load_lut()
    out = {"hw_mult": hw_mult}

    # ---------------- (A) pruning method sweep ----------------
    sweep = []
    prev = json.load(open("results/search.json")) if os.path.exists("results/search.json") else {}
    if prev.get("sweep"):
        sweep = prev["sweep"]
        print("re-using pruning sweep from results/search.json")
    for method, fuse in [("evit", True), ("evit", False), ("norm", False), ("random", False)]:
        for r in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
            if (r == 1.0 and not (method == "evit" and fuse)) or prev.get("sweep"):
                continue
            torch.manual_seed(0)
            set_cfg(model, (r, r, r), 1, method, fuse)
            acc = accuracy(model, xva, yva)
            row = dict(method=method + ("+fuse" if fuse else ""), ratio=r, acc=acc, gmacs=macs(model),
                       tokens=model.token_schedule()[1])
            sweep.append(row); print("sweep", row, flush=True)
    out["sweep"] = sweep
    json.dump(out, open("results/search.json", "w"), indent=1)

    # ---------------- (B) hardware-aware search ----------------
    fp32_lat = lut["fp32"]["end2end_unpruned"]
    space = []
    for ratios in itertools.product(RATIOS, repeat=3):
        for mult in (1, hw_mult):
            set_cfg(model, ratios, mult)
            space.append(dict(ratios=ratios, mult=mult, gmacs=macs(model),
                              pred_ms=predict_latency(model, lut, PRECISION),
                              pred_fp32_ms=predict_latency(model, lut, "fp32")))
    cache = {}

    def evaluate(c):
        key = (c["ratios"], c["mult"])
        if key not in cache:
            set_cfg(model, c["ratios"], c["mult"])
            cache[key] = accuracy(model, xva, yva, limit=1000)
        return cache[key]

    selected = []
    for sp in SPEEDUPS:
        budget = fp32_lat / sp
        feas = [c for c in space if c["pred_ms"] <= budget]

        def maximal(c):
            for j in range(3):
                i = RATIOS.index(c["ratios"][j])
                if i + 1 < len(RATIOS):
                    up = list(c["ratios"]); up[j] = RATIOS[i + 1]
                    if any(d["ratios"] == tuple(up) and d["mult"] == c["mult"] for d in feas):
                        return False
            return True

        cands = [c for c in feas if maximal(c)]
        t0 = time.time()
        for c in cands:
            c["val_acc_1k"] = evaluate(c)
        best = max(cands, key=lambda c: (c["val_acc_1k"], -c["pred_ms"]))
        # uniform reference within the same budget (no rounding)
        uni = [c for c in feas if c["mult"] == 1 and len(set(c["ratios"])) == 1]
        if uni:
            uni_best = max(uni, key=lambda c: c["ratios"][0])
        else:  # no uniform schedule fits: take the fastest uniform one
            uni_best = min((c for c in space if c["mult"] == 1 and len(set(c["ratios"])) == 1),
                           key=lambda c: c["pred_ms"])
        uni_best["val_acc_1k"] = evaluate(uni_best)
        selected.append(dict(speedup=sp, budget_ms=budget, n_candidates=len(cands), best=best,
                             uniform=uni_best, minutes=(time.time() - t0) / 60))
        print(f"budget {sp}x ({budget:.1f} ms): {len(cands)} candidates, best {best}, uniform {uni_best}",
              flush=True)
    for c in space:
        c["val_acc_1k"] = cache.get((c["ratios"], c["mult"]))
    out["space"] = space
    out["selected"] = selected
    out["fp32_ms"] = fp32_lat
    json.dump(out, open("results/search.json", "w"), indent=1)
    print("saved results/search.json")


if __name__ == "__main__":
    main()
