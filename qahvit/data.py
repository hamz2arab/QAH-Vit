"""CIFAR-10 kept in memory as uint8; resizing to 224 is done per batch with
F.interpolate (much faster on CPU than per-image PIL transforms)."""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_cifar(root="data"):
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    to_t = lambda d: torch.from_numpy(np.ascontiguousarray(d.data)).permute(0, 3, 1, 2).contiguous()
    return (to_t(tr), torch.tensor(tr.targets)), (to_t(te), torch.tensor(te.targets))


def splits(root="data", n_val=2000, seed=0):
    """train (48k) / val (2k, held out from train, used for the search) / test (10k)."""
    (xtr, ytr), (xte, yte) = load_cifar(root)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(xtr), generator=g)
    vi, ti = perm[:n_val], perm[n_val:]
    return (xtr[ti], ytr[ti]), (xtr[vi], ytr[vi]), (xte, yte)


def preprocess(xb, size=224, train=False):
    x = xb.float().div_(255)
    if train:  # random crop (pad 4) + horizontal flip at native res
        B = x.shape[0]
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")
        i, j = np.random.randint(0, 9, 2)
        x = x[:, :, i:i + 32, j:j + 32]
        flip = torch.rand(B) < 0.5
        x[flip] = x[flip].flip(-1)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


def batches(x, y, bs, shuffle=False, train=False, size=224):
    idx = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for s in range(0, len(x), bs):
        b = idx[s:s + bs]
        yield preprocess(x[b], size, train), y[b]
