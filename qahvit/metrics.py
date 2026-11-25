"""Accuracy, complexity (MACs), model size, memory and latency measurement."""
import io
import os
import time
import statistics
import torch

from .data import batches


@torch.no_grad()
def accuracy(model, x, y, bs=100, limit=None):
    model.eval()
    if limit:
        x, y = x[:limit], y[:limit]
    correct = 0
    for xb, yb in batches(x, y, bs):
        correct += (model(xb).argmax(1) == yb).sum().item()
    return 100.0 * correct / len(x)


def macs(model):
    """Analytical multiply-accumulate count for one 224x224 image (GMACs).
    Counts patch-embed conv, every Linear, and the two attention matmuls."""
    D, Hd = model.dim, int(model.dim * model.mlp_ratio)
    P = model.patch_embed.num_patches
    total = P * (3 * 16 * 16) * D
    attn_n, mlp_n = model.token_schedule()
    for na, nm in zip(attn_n, mlp_n):
        total += na * D * 3 * D        # qkv
        total += 2 * na * na * D       # QK^T and AV
        total += na * D * D            # proj
        total += nm * 2 * D * Hd       # MLP (runs on pruned tokens)
    total += D * 10
    return total / 1e9


def linear_macs_fraction(model):
    D, Hd = model.dim, int(model.dim * model.mlp_ratio)
    attn_n, mlp_n = model.token_schedule()
    lin = sum(na * 4 * D * D + nm * 2 * D * Hd for na, nm in zip(attn_n, mlp_n))
    return lin / (macs(model) * 1e9)


def params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    """Serialized size of the state_dict (what you would ship)."""
    buf = io.BytesIO()
    sd = {k: v for k, v in model.state_dict().items() if not k.endswith(("_lo", "_hi", "initialized"))}
    torch.save(sd, buf)
    return buf.getbuffer().nbytes / 2 ** 20


def activation_memory_mb(model, batch=1, bytes_per=4):
    """Analytical peak live activation memory of one block (largest tensor set)."""
    D, Hd, H = model.dim, int(model.dim * model.mlp_ratio), model.heads
    attn_n, mlp_n = model.token_schedule()
    peak = 0
    for na, nm in zip(attn_n, mlp_n):
        attn = na * D + na * 3 * D + H * na * na + na * D  # x, qkv, scores, out
        mlp = nm * D + nm * Hd * 2                          # x, fc1 out, gelu out
        peak = max(peak, attn, mlp)
    return batch * peak * bytes_per / 2 ** 20


@torch.no_grad()
def latency_ms(model, batch=1, warmup=10, iters=50, size=224):
    model.eval()
    x = torch.randn(batch, 3, size, size)
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts), statistics.stdev(ts)


def peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@torch.no_grad()
def peak_activation_mb(model, batch=1, size=224):
    """Exact peak of live tensor memory allocated *during* one forward pass
    (activations + intermediate buffers, excluding weights and the input).
    Replays the profiler's allocator events (alloc > 0, free < 0) in time order."""
    model.eval()
    x = torch.randn(batch, 3, size, size)
    model(x)
    with torch.profiler.profile(profile_memory=True) as p:
        model(x)
    ev = sorted((e.start_ns(), e.nbytes()) for e in p.profiler.kineto_results.events()
                if e.name() == "[memory]")
    cur = peak = 0
    for _, b in ev:
        cur += b
        peak = max(peak, cur)
    return peak / 2 ** 20


@torch.no_grad()
def activation_traffic_mb(model, batch=1, size=224):
    """Total bytes allocated for intermediate tensors in one forward pass, per
    image (a proxy for activation memory traffic)."""
    model.eval()
    x = torch.randn(batch, 3, size, size)
    model(x)
    with torch.profiler.profile(profile_memory=True) as p:
        model(x)
    tot = sum(e.nbytes() for e in p.profiler.kineto_results.events() if e.name() == "[memory]" and e.nbytes() > 0)
    return tot / 2 ** 20 / batch
