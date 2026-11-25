"""Fine-tune DeiT-Tiny on CIFAR-10 (224px), optionally with token pruning
and/or quantization-aware training (fake INT8) active during training.

  baseline : python scripts/train.py --out checkpoints/baseline_fp32.pt
  QAH-ViT  : python scripts/train.py --init checkpoints/baseline_fp32.pt \
                 --prune "3:0.6,6:0.6,9:0.6" --qat --steps 400 --lr 2e-5 --out ...
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

from qahvit.model import deit_tiny, load_finetuned
from qahvit.data import splits, batches
from qahvit.metrics import accuracy
from qahvit.quant import set_quant_mode, reset_observers, set_calibrating


def parse_prune(s):
    if not s:
        return {}
    return {int(k): float(v) for k, v in (p.split(":") for p in s.split(","))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="pretrained")
    ap.add_argument("--prune", default="")
    ap.add_argument("--token_multiple", type=int, default=1)
    ap.add_argument("--qat", action="store_true")
    ap.add_argument("--w_bits", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    (xtr, ytr), (xva, yva), _ = splits()
    if args.init == "pretrained":
        model = deit_tiny(10, "checkpoints/deit_tiny_patch16_224.pth")
    else:
        model = load_finetuned(args.init)
    model.prune_cfg = parse_prune(args.prune)
    model.token_multiple = args.token_multiple
    if args.qat:
        set_quant_mode(model, "fake", w_bits=args.w_bits)
        reset_observers(model)
        # calibrate activation ranges on a few batches before training
        set_calibrating(model, True)
        model.eval()
        with torch.no_grad():
            for i, (xb, _) in enumerate(batches(xtr, ytr, 32, shuffle=True)):
                model(xb)
                if i == 7:
                    break
        set_calibrating(model, False)

    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "pos_embed" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim > 1 and "pos_embed" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    log = []
    step, t0 = 0, time.time()
    model.train()
    while step < args.steps:
        for xb, yb in batches(xtr, ytr, args.bs, shuffle=True, train=True):
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 25 == 0:
                el = time.time() - t0
                print(f"step {step}/{args.steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{el/60:.1f}min eta {(args.steps-step)*el/step/60:.1f}min", flush=True)
                log.append({"step": step, "loss": loss.item()})
            if step >= args.steps:
                break

    model.eval()
    va = accuracy(model, xva, yva, limit=int(os.environ.get("VAL_LIMIT", 0)) or None)
    print(f"val acc {va:.2f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    json.dump({"args": vars(args), "val_acc": va, "train_minutes": (time.time() - t0) / 60, "log": log},
              open(args.out.replace(".pt", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
