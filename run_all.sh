#!/usr/bin/env bash
# Full QAH-ViT pipeline. Each stage is skipped if its output already exists.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p logs results plots checkpoints
STEPS=${STEPS:-300}

# 0. ImageNet DeiT-Tiny -> CIFAR-10 FP32 baseline
[ -f checkpoints/baseline_fp32.pt ] || $PY scripts/train.py --steps 1500 --lr 1e-4 --threads 4 \
    --out checkpoints/baseline_fp32.pt 2>&1 | tee logs/train_baseline.log

# 1. hardware profiling (latency LUT)
[ -f results/latency_lut.json ] || $PY scripts/latency_lut.py 2>&1 | tee logs/latency_lut.log

# 2. pruning study + hardware-aware search
grep -q '"selected"' results/search.json 2>/dev/null || $PY scripts/search.py 2>&1 | tee logs/search.log

cfg() {  # prune string + token multiple of the schedule selected for budget $1
  $PY -c "
import json; s=[x for x in json.load(open('results/search.json'))['selected'] if x['speedup']==$1][0]['best']
print(','.join(f'{l}:{r}' for l,r in zip((3,6,9),s['ratios']) if r<1), s['mult'])"
}

# 3. fine-tuning: ablations + QAH-ViT (joint pruning + QAT)
[ -f checkpoints/qat_only.pt ] || $PY scripts/train.py --init checkpoints/baseline_fp32.pt --qat \
    --steps $STEPS --lr 2e-5 --warmup 30 --threads 4 --out checkpoints/qat_only.pt 2>&1 | tee logs/train_qat_only.log
read P M < <(cfg 1.25)
[ -f checkpoints/prune_ft_1.25.pt ] || $PY scripts/train.py --init checkpoints/baseline_fp32.pt --prune "$P" \
    --token_multiple $M --steps $STEPS --lr 2e-5 --warmup 30 --threads 4 --out checkpoints/prune_ft_1.25.pt \
    2>&1 | tee logs/train_prune_ft_1.25.log
for SP in 1.15 1.25 1.35; do
  read P M < <(cfg $SP)
  [ -f checkpoints/qah_$SP.pt ] || $PY scripts/train.py --init checkpoints/baseline_fp32.pt --prune "$P" \
      --token_multiple $M --qat --steps $STEPS --lr 2e-5 --warmup 30 --threads 4 \
      --out checkpoints/qah_$SP.pt 2>&1 | tee logs/train_qah_$SP.log
done

# 4. final benchmark on the 10k test set
$PY scripts/evaluate.py 2>&1 | tee logs/evaluate.log

# 5. figures
$PY scripts/plots.py
$PY scripts/visualize_tokens.py
