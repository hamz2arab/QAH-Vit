"""LUT-based latency predictor (hardware-aware component)."""
import json


def load_lut(path="results/latency_lut.json"):
    return json.load(open(path))


def _sum(model, lut, precision):
    L = lut[precision]
    idx = {n: i for i, n in enumerate(lut["N"])}
    attn_n, mlp_n = model.token_schedule()
    total = L["patch_embed"] + L["head"]
    for i, (na, nm) in enumerate(zip(attn_n, mlp_n)):
        total += L["attn"][idx[na]] + L["mlp"][idx[nm]]
        if i in model.prune_cfg and nm != na:
            total += lut["prune_op"][idx[na]]
    return total


def predict_latency(model, lut, precision="int8_static"):
    """Predicted batch-1 latency (ms) of `model`'s current pruning schedule:
    sum of LUT entries + a constant framework overhead, calibrated so that the
    unpruned prediction equals the measured end-to-end latency."""
    saved = (dict(model.prune_cfg), model.token_multiple)
    model.prune_cfg = {}
    offset = lut[precision]["end2end_unpruned"] - _sum(model, lut, precision)
    model.prune_cfg, model.token_multiple = saved
    return _sum(model, lut, precision) + offset
