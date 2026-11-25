"""Quantization utilities.

* ``QLinear``  – drop-in replacement for ``nn.Linear`` that can run in
  - ``fp``   mode: plain float32
  - ``fake`` mode: quantization-aware training (QAT) with fake-quantized
    weights (symmetric, per-output-channel) and fake-quantized input
    activations (asymmetric, per-tensor, EMA min/max observer).
  Gradients flow through the rounding with the straight-through estimator.
* ``convert_dynamic`` – deploys a model with real INT8 kernels using PyTorch's
  fbgemm dynamic quantization (int8 weights, activations quantized on the fly).
* ``convert_static``  – deploys with real INT8 kernels using the activation
  ranges *learned during QAT* (static W8A8 ``quantized.Linear``).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _round_ste(x):
    return x + (torch.round(x) - x).detach()


def fake_quant_weight(w, bits=8):
    """Symmetric per-output-channel fake quantization."""
    qmax = 2 ** (bits - 1) - 1
    scale = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return torch.clamp(_round_ste(w / scale), -qmax, qmax) * scale


def act_qparams(lo, hi, qmax):
    lo = torch.clamp(lo, max=0.0)
    hi = torch.clamp(hi, min=0.0)
    scale = (hi - lo).clamp_min(1e-8) / qmax
    zp = torch.clamp(torch.round(-lo / scale), 0, qmax)
    return scale, zp


def fake_quant_act(x, lo, hi, bits=8):
    qmax = 2 ** bits - 1
    scale, zp = act_qparams(lo, hi, qmax)
    q = torch.clamp(_round_ste(x / scale) + zp, 0, qmax)
    return (q - zp) * scale


class QLinear(nn.Linear):
    # activation range is 7 bit on x86/fbgemm (reduce_range) to avoid int16
    # accumulation overflow on CPUs without VNNI (e.g. this i5-10210U).
    def __init__(self, in_f, out_f, bias=True, w_bits=8, a_bits=7):
        super().__init__(in_f, out_f, bias)
        self.mode = "fp"
        self.w_bits, self.a_bits = w_bits, a_bits
        self.momentum = 0.95
        self.register_buffer("in_lo", torch.tensor(0.0))
        self.register_buffer("in_hi", torch.tensor(0.0))
        self.register_buffer("out_lo", torch.tensor(0.0))
        self.register_buffer("out_hi", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))
        self.calibrating = False  # update observers in eval mode as well
        self.sim_out_quant = True  # simulate output requantization of static kernels

    @torch.no_grad()
    def _ema(self, lo_buf, hi_buf, t, first):
        lo, hi = t.min(), t.max()
        if first:
            lo_buf.copy_(lo); hi_buf.copy_(hi)
        else:
            m = self.momentum
            lo_buf.mul_(m).add_((1 - m) * lo); hi_buf.mul_(m).add_((1 - m) * hi)

    def forward(self, x):
        if self.mode == "fp":
            return F.linear(x, self.weight, self.bias)
        observe = self.training or self.calibrating
        first = not bool(self.initialized)
        if observe:
            self._ema(self.in_lo, self.in_hi, x, first)
        xq = fake_quant_act(x, self.in_lo, self.in_hi, self.a_bits) if self.a_bits < 32 else x
        wq = fake_quant_weight(self.weight, self.w_bits) if self.w_bits < 32 else self.weight
        y = F.linear(xq, wq, self.bias)
        if observe:
            self._ema(self.out_lo, self.out_hi, y, first)
            self.initialized.fill_(True)
        if self.sim_out_quant:  # static INT8 kernels emit quint8 outputs
            y = fake_quant_act(y, self.out_lo, self.out_hi, 8)
        return y


def set_quant_mode(model, mode, w_bits=None, a_bits=None):
    for m in model.modules():
        if isinstance(m, QLinear) and getattr(m, "_deploy_quant", True):
            m.mode = mode
            if w_bits is not None:
                m.w_bits = w_bits
            if a_bits is not None:
                m.a_bits = a_bits


def set_calibrating(model, flag):
    for m in model.modules():
        if isinstance(m, QLinear):
            m.calibrating = flag


def reset_observers(model):
    for m in model.modules():
        if isinstance(m, QLinear):
            m.initialized.fill_(False)


# ----------------------------------------------------------------------------
# Deployment: real INT8 kernels (CPU, fbgemm / x86 backend)
# ----------------------------------------------------------------------------
def _quantize_weight(w):
    scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    zp = torch.zeros_like(scale, dtype=torch.long)
    return torch.quantize_per_channel(w, scale.double(), zp, 0, torch.qint8)


class DynamicInt8Linear(nn.Module):
    """int8 weight (per-channel), activations quantized on the fly per batch."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.m = torch.ao.nn.quantized.dynamic.Linear(lin.in_features, lin.out_features, dtype=torch.qint8)
        self.m.set_weight_bias(_quantize_weight(lin.weight.detach()), lin.bias.detach())

    def forward(self, x):
        return self.m(x)


class StaticInt8Linear(nn.Module):
    """W8A8 static: uses activation ranges learned/calibrated in QAT."""

    def __init__(self, lin: QLinear):
        super().__init__()
        qmax = 2 ** lin.a_bits - 1
        s_in, zp_in = act_qparams(lin.in_lo, lin.in_hi, qmax)
        s_out, zp_out = act_qparams(lin.out_lo, lin.out_hi, 255)
        self.s_in, self.zp_in = float(s_in), int(zp_in)
        self.m = torch.ao.nn.quantized.Linear(lin.in_features, lin.out_features, dtype=torch.qint8)
        self.m.set_weight_bias(_quantize_weight(lin.weight.detach()), lin.bias.detach())
        self.m.scale, self.m.zero_point = float(s_out), int(zp_out)

    def forward(self, x):
        xq = torch.quantize_per_tensor(x, self.s_in, self.zp_in, torch.quint8)
        return self.m(xq).dequantize()


def _swap_linears(model, fn):
    for name, child in list(model.named_children()):
        if isinstance(child, QLinear) and getattr(child, "_deploy_quant", True):
            setattr(model, name, fn(child))
        else:
            _swap_linears(child, fn)
    return model


def convert_dynamic(model):
    torch.backends.quantized.engine = "fbgemm"
    return _swap_linears(model, DynamicInt8Linear)


def convert_static(model):
    torch.backends.quantized.engine = "fbgemm"
    return _swap_linears(model, StaticInt8Linear)
