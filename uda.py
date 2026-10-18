"""
经典无监督域适应基线（需要源数据，作“参考上界”）：MK-MMD(DAN) / Deep CORAL / DANN / CDAN。
与本文方法使用同一骨干、同一数据划分。目标域只用训练部分的无标签样本，在目标测试部分评估。

  python uda.py --method dann --src pu_0 --tgt pu_1 --seed 0
"""
import argparse
import math

import torch
import torch.nn as nn

from src.common import (LabelSmoothCE, Net, append_result, device, evaluate, load_domain, loader, set_seed,
                        weak_aug)


# ---------------------------------------------------------------------------- 对齐损失
def mk_mmd(xs, xt, mults=(0.25, 0.5, 1.0, 2.0, 4.0)):
    x = torch.cat([xs, xt])
    d2 = torch.cdist(x, x).pow(2)
    bw = d2.detach().sum() / (len(x) ** 2 - len(x))
    K = sum(torch.exp(-d2 / (bw * m)) for m in mults)
    n = len(xs)
    return K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()


def coral(xs, xt):
    d = xs.shape[1]
    cs = torch.cov(xs.T); ct = torch.cov(xt.T)
    return (cs - ct).pow(2).sum() / (4 * d * d)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Disc(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))

    def forward(self, x, lam):
        return self.net(GRL.apply(x, lam))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["mmd", "coral", "dann", "cdan"])
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--trade_off", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--results", default="results/results.csv")
    a = ap.parse_args()
    set_seed(a.seed); dev = device()

    ds, meta = load_domain(a.data_root, a.src)
    dt, _ = load_domain(a.data_root, a.tgt)
    n_cls = len(meta["classes"])
    dl_s = loader(ds["X_train"], ds["y_train"], a.bs, True, drop_last=True)
    dl_t = loader(dt["X_train"], dt["y_train"], a.bs, True, drop_last=True)  # 标签不使用
    dl_te = loader(dt["X_test"], dt["y_test"], 256, False)

    model = Net(n_cls, a.width).to(dev)
    params = list(model.parameters())
    disc = None
    if a.method in ("dann", "cdan"):
        disc = Disc(256 if a.method == "dann" else 256 * n_cls).to(dev)
        params += list(disc.parameters())
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=1e-4)
    crit, bce = LabelSmoothCE(0.1), nn.BCEWithLogitsLoss()

    iters = a.epochs * len(dl_s)
    it = 0
    for ep in range(a.epochs):
        model.train()
        t_iter = iter(dl_t)
        for xs, ys, _ in dl_s:
            try:
                xt, _, _ = next(t_iter)
            except StopIteration:
                t_iter = iter(dl_t); xt, _, _ = next(t_iter)
            xs, ys, xt = weak_aug(xs.to(dev)), ys.to(dev), weak_aug(xt.to(dev))
            p = it / iters
            lam = 2 / (1 + math.exp(-10 * p)) - 1  # DANN 的渐进权重
            out, f = model(torch.cat([xs, xt]), return_feat=True)
            n = len(xs)
            loss_cls = crit(out[:n], ys)
            if a.method == "mmd":
                loss_da = mk_mmd(f[:n], f[n:])
            elif a.method == "coral":
                loss_da = coral(f[:n], f[n:])
            else:
                if a.method == "dann":
                    h = f
                else:  # CDAN：特征与预测的外积作为判别器输入
                    sm = out.softmax(1).detach()
                    h = torch.bmm(sm.unsqueeze(2), f.unsqueeze(1)).flatten(1)
                dlab = torch.cat([torch.ones(n, 1), torch.zeros(len(xt), 1)]).to(dev)
                loss_da = bce(disc(h, lam), dlab)
            loss = loss_cls + a.trade_off * (lam if a.method in ("mmd", "coral") else 1.0) * loss_da
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1
        if ep % 10 == 0 or ep == a.epochs - 1:
            r = evaluate(model, dl_te, dev)
            print(f"ep {ep:3d} cls {loss_cls.item():.3f} da {loss_da.item():.3f} | tgt-test acc {r['acc']:.2f}")

    r = evaluate(model, dl_te, dev)  # 取最后一轮，不用目标标签选模型
    print(f"{a.method} {a.src}->{a.tgt}: acc {r['acc']:.2f} f1 {r['f1']:.2f}")
    append_result(a.results, dict(method=a.method, src=a.src, tgt=a.tgt, seed=a.seed,
                                  acc=f"{r['acc']:.2f}", f1=f"{r['f1']:.2f}", note=""))


if __name__ == "__main__":
    main()
