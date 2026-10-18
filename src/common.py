"""数据、增强、模型、评估等公共组件。所有方法共用同一骨干与同一数据划分，保证对比公平。"""
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


# ============================================================================ 基础工具
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_result(csv_path, row: dict):
    """每次实验追加一行到结果 CSV，summarize.py 汇总均值±标准差。"""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new = not os.path.exists(csv_path)
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ============================================================================ 数据
def load_domain(root, name):
    d = np.load(os.path.join(root, f"{name}.npz"))
    with open(os.path.join(root, f"{name}.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return d, meta


def znorm(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)


class SigDS(Dataset):
    """返回 (x, y, idx)；idx 供无源方法维护记忆库。"""

    def __init__(self, X, y):
        self.X = torch.from_numpy(znorm(X.astype(np.float32))).unsqueeze(1)  # (N,1,L)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], i


def split_val(X, y, ratio=0.1, seed=0):
    """从源域训练集分出验证集（分层），用于早停与选模型——不碰目标域标签。"""
    rng = np.random.RandomState(seed)
    va = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        va += idx[: max(1, int(len(idx) * ratio))].tolist()
    mask = np.zeros(len(y), bool); mask[va] = True
    return X[~mask], y[~mask], X[mask], y[mask]


def loader(X, y, bs, shuffle, drop_last=False, workers=0):
    return DataLoader(SigDS(X, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


# ============================================================================ 振动信号增强（物理上合理）
def aug_shift(x, max_frac=0.25):
    s = int(x.shape[-1] * max_frac)
    k = torch.randint(-s, s + 1, (1,)).item()
    return torch.roll(x, k, dims=-1)


def aug_noise(x, snr_db=(4, 20)):
    snr = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(*snr_db)
    p = x.pow(2).mean(-1, keepdim=True)
    return x + torch.randn_like(x) * torch.sqrt(p / 10 ** (snr / 10))


def aug_scale(x, lo=0.8, hi=1.2):
    return x * torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(lo, hi)


def aug_bandmask(x, max_frac=0.15, n_bands=2):
    """频带掩码：随机抹去若干频带，模拟传递路径/工况变化导致的频响差异。"""
    X = torch.fft.rfft(x, dim=-1)
    B, nf = x.shape[0], X.shape[-1]
    k = torch.arange(nf, device=x.device)[None, :]
    keep = torch.ones(B, nf, dtype=torch.bool, device=x.device)
    for _ in range(n_bands):
        w = (nf * torch.empty(B, 1, device=x.device).uniform_(0.02, max_frac)).long()
        s = (torch.rand(B, 1, device=x.device) * (nf - w - 1)).long() + 1
        keep &= ~((k >= s) & (k < s + w))
    return torch.fft.irfft(X * keep[:, None, :], n=x.shape[-1], dim=-1)


def weak_aug(x):
    return aug_scale(aug_shift(x))


def strong_aug(x):
    return aug_noise(aug_bandmask(aug_scale(aug_shift(x))))


VOTE_AUGS = [lambda x: x, aug_shift, lambda x: aug_noise(x, (8, 20)), aug_scale, aug_bandmask]


# ============================================================================ 模型
class LiteWDCNN(nn.Module):
    """轻量一维 CNN：宽首层卷积（抑制高频噪声）+ 4 个小卷积块 + 可选 SE 通道注意力。
    默认约 8 万参数，适合树莓派 / STM32H7 部署。width 可调骨干宽度。"""

    def __init__(self, width=16, se=True):
        super().__init__()
        c = [width, width * 2, width * 4, width * 4, width * 8]
        layers = [nn.Conv1d(1, c[0], 64, stride=8, padding=28), nn.BatchNorm1d(c[0]), nn.ReLU(), nn.MaxPool1d(2)]
        for i in range(4):
            layers += [nn.Conv1d(c[i], c[i + 1], 3, padding=1), nn.BatchNorm1d(c[i + 1]), nn.ReLU()]
            if se:
                layers += [SE(c[i + 1])]
            if i < 3:
                layers += [nn.MaxPool1d(2)]
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.out_dim = c[-1]

    def forward(self, x):
        return self.net(x)


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class Net(nn.Module):
    """骨干 F + 瓶颈 B（256 维 + BN）+ 权重归一化分类器 C —— 与 SHOT 系列方法的结构一致。"""

    def __init__(self, n_cls, width=16, bottleneck=256, se=True):
        super().__init__()
        self.F = LiteWDCNN(width, se)
        # 与 SHOT 官方实现一致：Linear + BN，不加 ReLU/Dropout（非负特征会让余弦聚类退化）
        self.B = nn.Sequential(nn.Linear(self.F.out_dim, bottleneck), nn.BatchNorm1d(bottleneck))
        self.C = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, n_cls), dim=0)

    def feat(self, x):
        return self.B(self.F(x))

    def forward(self, x, return_feat=False):
        f = self.feat(x)
        out = self.C(f)
        return (out, f) if return_feat else out


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ============================================================================ 评估
@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    P, Y = [], []
    for x, y, _ in dl:
        P.append(model(x.to(dev)).argmax(1).cpu()); Y.append(y)
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    return {"acc": 100 * accuracy_score(Y, P), "f1": 100 * f1_score(Y, P, average="macro")}


def group_of(tag):
    """样本来源标签 -> 分组名。PU: 'K001_N15_M07_F10_15' -> 'K001'（按轴承）；CWRU/JNU: 按原始文件。"""
    import re
    t = str(tag).split("[")[0]
    m = re.match(r"^(K[AIB]?\d+)_N\d", t)
    return m.group(1) if m else t


@torch.no_grad()
def per_group_report(model, X, y, tags, dev, classes, csv_path=None, info=None):
    """逐轴承（或逐文件）准确率：用来定位“整个轴承被判错”这类错误。只用于诊断，不参与训练或选模型。"""
    model.eval()
    P = []
    for x, _, _ in loader(X, y, 256, False):
        P.append(model(x.to(dev)).argmax(1).cpu())
    P = torch.cat(P).numpy()
    groups = np.array([group_of(t) for t in tags])
    lines = []
    for g in sorted(set(groups)):
        m = groups == g
        true = int(np.bincount(y[m]).argmax())
        pred_major = int(np.bincount(P[m], minlength=len(classes)).argmax())
        acc = 100 * (P[m] == y[m]).mean()
        flag = f"  <-- 整体被判为 {classes[pred_major]}" if pred_major != true else ""
        lines.append(f"    {g:<10} 真实 {classes[true]:<8} n={m.sum():<4} acc {acc:6.2f}{flag}")
        if csv_path:
            append_result(csv_path, {**(info or {}), "group": g, "true": classes[true],
                                     "pred_major": classes[pred_major], "n": int(m.sum()), "acc": f"{acc:.2f}"})
    print("  逐组准确率：\n" + "\n".join(lines))


@torch.no_grad()
def collect(model, dl, dev):
    """返回整个数据集的特征、softmax 输出、标签（标签仅用于记录伪标签准确率，不参与训练）。"""
    model.eval()
    Fs, Ps, Ys = [], [], []
    for x, y, _ in dl:
        o, f = model(x.to(dev), return_feat=True)
        Fs.append(f.cpu()); Ps.append(o.softmax(1).cpu()); Ys.append(y)
    return torch.cat(Fs), torch.cat(Ps), torch.cat(Ys)


# ============================================================================ 常用损失
def entropy(p, eps=1e-5):
    return -(p * torch.log(p + eps)).sum(1)


def im_loss(logits):
    """信息最大化：个体熵最小 + 平均预测熵最大（防止塌缩到单类）。"""
    p = logits.softmax(1)
    pm = p.mean(0)
    return entropy(p).mean() + (pm * torch.log(pm + 1e-5)).sum()


class LabelSmoothCE(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__(); self.eps = eps

    def forward(self, logits, y):
        logp = logits.log_softmax(1)
        t = torch.full_like(logp, self.eps / logits.shape[1]).scatter_(1, y[:, None], 1 - self.eps + self.eps / logits.shape[1])
        return -(t * logp).sum(1).mean()


def per_group_path(results_csv):
    """逐轴承结果文件：主结果写 results/per_group.csv；其他结果文件（如参数网格）各自对应 <名字>_per_group.csv，互不混淆。"""
    if os.path.normpath(results_csv) == os.path.normpath("results/results.csv"):
        return "results/per_group.csv"
    return os.path.splitext(results_csv)[0] + "_per_group.csv"
