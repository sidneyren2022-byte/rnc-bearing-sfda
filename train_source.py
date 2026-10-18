"""
第 1 步：源域训练（所有无源方法共用同一个源模型），并给出 Source-only 下界。

  python train_source.py --src pu_0 --tgt pu_1 pu_2 pu_3 --seed 0

模型保存到 ckpt/<src>_w<width>_s<seed>.pt；Source-only 结果追加到 results/results.csv。
"""
import argparse
import copy
import os

import torch

from src.common import (LabelSmoothCE, Net, append_result, device, evaluate, load_domain, loader, n_params,
                        per_group_path, per_group_report, set_seed, split_val, weak_aug)


def ckpt_path(a):
    return os.path.join(a.ckpt, f"{a.src}_w{a.width}_s{a.seed}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--results", default="results/results.csv")
    a = ap.parse_args()
    set_seed(a.seed); dev = device()

    d, meta = load_domain(a.data_root, a.src)
    n_cls = len(meta["classes"])
    Xtr, ytr, Xva, yva = split_val(d["X_train"], d["y_train"], 0.1, a.seed)
    dl_tr = loader(Xtr, ytr, a.bs, True, drop_last=True)
    dl_va = loader(Xva, yva, 256, False)

    model = Net(n_cls, a.width).to(dev)
    print(f"params: {n_params(model) / 1e3:.1f}K  classes: {n_cls}  train: {len(ytr)}  val: {len(yva)}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    crit = LabelSmoothCE(0.1)  # 标签平滑让源模型输出更适合后续伪标签（SHOT 的做法）

    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        for x, y, _ in dl_tr:
            x, y = weak_aug(x.to(dev)), y.to(dev)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        va = evaluate(model, dl_va, dev)["acc"]
        if va >= best:
            best, best_state = va, copy.deepcopy(model.state_dict())
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"ep {ep:3d} loss {loss.item():.4f} src-val {va:.2f} (best {best:.2f})")

    model.load_state_dict(best_state)
    os.makedirs(a.ckpt, exist_ok=True)
    torch.save({"state": best_state, "n_cls": n_cls, "width": a.width, "src": a.src}, ckpt_path(a))
    src_te = evaluate(model, loader(d["X_test"], d["y_test"], 256, False), dev)
    print(f"source test: {src_te}")

    for t in a.tgt:
        dt, _ = load_domain(a.data_root, t)
        r = evaluate(model, loader(dt["X_test"], dt["y_test"], 256, False), dev)
        print(f"Source-only {a.src}->{t}: acc {r['acc']:.2f} f1 {r['f1']:.2f}")
        if "src_test" in dt.files:
            per_group_report(model, dt["X_test"], dt["y_test"], dt["src_test"], dev, meta["classes"],
                             per_group_path(a.results), dict(method="source_only", src=a.src, tgt=t, seed=a.seed, note=""))
        append_result(a.results, dict(method="source_only", src=a.src, tgt=t, seed=a.seed,
                                      acc=f"{r['acc']:.2f}", f1=f"{r['f1']:.2f}", note=""))


if __name__ == "__main__":
    main()
