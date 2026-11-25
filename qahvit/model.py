"""DeiT-Tiny Vision Transformer with built-in token pruning.

Parameter names match the official DeiT / timm checkpoints so the ImageNet
weights load directly.

Token pruning (EViT-style "attentive token pruning"):
  at selected blocks, after multi-head self-attention, the patch tokens are
  ranked by the attention they receive from the [CLS] token (averaged over
  heads). The top-k are kept; the remaining ones are optionally *fused* into a
  single token (attention-weighted average) so their information is not lost.
  The block's MLP and every following block then operate on fewer tokens, and
  cost scales roughly linearly (MLP/QKV) or quadratically (attention) with N.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import QLinear


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = QLinear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = QLinear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.scale = self.hd ** -0.5
        self.qkv = QLinear(dim, dim * 3)
        self.proj = QLinear(dim, dim)

    def forward(self, x, need_cls_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B,H,N,hd
        out = F.scaled_dot_product_attention(q, k, v)
        cls_attn = None
        if need_cls_attn:  # only the CLS row is needed -> O(N) extra work
            a = (q[:, :, :1] @ k.transpose(-2, -1)) * self.scale  # B,H,1,N
            cls_attn = a.softmax(-1)[:, :, 0, 1:].mean(1)  # B,N-1
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out), cls_attn


def prune_tokens(x, k, method="evit", cls_attn=None, fuse=True):
    """Keep CLS + top-k patch tokens (+1 fused token)."""
    B, N, C = x.shape
    cls, tok = x[:, :1], x[:, 1:]
    Nt = tok.shape[1]
    if k >= Nt:
        return x
    if method == "evit":
        score = cls_attn
    elif method == "random":
        score = torch.rand(B, Nt, device=x.device)
    elif method == "norm":
        score = tok.norm(dim=-1)
    else:
        raise ValueError(method)
    idx = score.topk(k, dim=1, sorted=False).indices
    kept = tok.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
    if not fuse:
        return torch.cat([cls, kept], 1)
    w = score if method == "evit" else torch.ones_like(score)
    w = w.scatter(1, idx, 0.0)
    w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
    fused = (w.unsqueeze(-1) * tok).sum(1, keepdim=True)
    return torch.cat([cls, kept, fused], 1)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, keep=None, method="evit", fuse=True):
        a, cls_attn = self.attn(self.norm1(x), need_cls_attn=(keep is not None and method == "evit"))
        x = x + a
        if keep is not None:
            x = prune_tokens(x, keep, method, cls_attn, fuse)
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, img=224, patch=16, dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)
        self.num_patches = (img // patch) ** 2

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ViT(nn.Module):
    def __init__(self, img=224, patch=16, dim=192, depth=12, heads=3, mlp_ratio=4.0, num_classes=10):
        super().__init__()
        self.dim, self.depth, self.heads, self.mlp_ratio = dim, depth, heads, mlp_ratio
        self.patch_embed = PatchEmbed(img, patch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = QLinear(dim, num_classes)
        self.head._deploy_quant = False  # classifier stays FP32
        # pruning configuration: {block_index: keep_ratio}
        self.prune_cfg = {}
        self.prune_method = "evit"
        self.fuse = True
        self.token_multiple = 1  # hardware-aware rounding of kept tokens

    # ------------------------------------------------------------------
    def keep_count(self, ratio, n_patch_tokens):
        k = int(math.ceil(ratio * n_patch_tokens))
        m = self.token_multiple
        if m > 1:  # make total sequence length (CLS + k + fused) a multiple of m
            extra = 2 if self.fuse else 1
            k = max(m - extra, int(round((k + extra) / m)) * m - extra)
        return max(1, min(k, n_patch_tokens))

    def token_schedule(self):
        """Sequence length entering each block's MLP (len = depth) and attention."""
        n = self.patch_embed.num_patches + 1
        attn_n, mlp_n = [], []
        for i in range(self.depth):
            attn_n.append(n)
            if i in self.prune_cfg:
                k = self.keep_count(self.prune_cfg[i], n - 1)
                if k < n - 1:
                    n = 1 + k + (1 if self.fuse else 0)
            mlp_n.append(n)
        return attn_n, mlp_n

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embed
        for i, blk in enumerate(self.blocks):
            keep = None
            if i in self.prune_cfg:
                keep = self.keep_count(self.prune_cfg[i], x.shape[1] - 1)
            x = blk(x, keep, self.prune_method, self.fuse)
        return self.head(self.norm(x)[:, 0])


def deit_tiny(num_classes=10, pretrained_path=None):
    model = ViT(num_classes=num_classes)
    if pretrained_path:
        sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not (m.startswith("head") or m.endswith(("_lo", "_hi", "initialized")))]
        assert not missing and not unexpected, (missing, unexpected)
        nn.init.trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)
    return model


def load_finetuned(path):
    model = ViT(num_classes=10)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model
