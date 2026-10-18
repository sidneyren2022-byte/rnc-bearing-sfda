"""
第 2 步：无源域自适应。只加载源模型 + 目标域无标签训练样本，在目标域测试样本上评估。

方法：
  shot  —— Liang et al., ICML 2020：信息最大化 + 聚类伪标签
  nrc   —— Yang et al., NeurIPS 2021：邻域互惠聚类（简化实现）
  ours  —— 本文方法 RPF-SFDA（Reliability-guided Prototype and Frequency-consistency SFDA）：
           (V) 多增强投票 + 类别均衡的可靠样本选择
           (P) 可靠样本建原型，不可靠样本用原型相似度软标签监督（而非 SDALR 的熵最大化）
           (F) 弱增强/频带掩码强增强之间的一致性约束
           (IM) 信息最大化防塌缩
           消融开关：--no_vote --no_proto --no_freq --no_im

  python sfda.py --method ours --src pu_0 --tgt pu_1 --seed 0
  python sfda.py --method ours --src pu_0 --tgt pu_1 --seed 0 --no_freq     # 消融
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.common import (VOTE_AUGS, LabelSmoothCE, Net, append_result, device, evaluate, im_loss, load_domain, per_group_path,
                        loader, per_group_report, set_seed, strong_aug, weak_aug)


# ============================================================================ SHOT 伪标签
@torch.no_grad()
def shot_pseudo_labels(model, dl_full, dev, n_cls):
    model.eval()
    Fs, Ps = [], []
    for x, _, _ in dl_full:
        o, f = model(x.to(dev), return_feat=True)
        Fs.append(f); Ps.append(o.softmax(1))
    Fe, P = torch.cat(Fs), torch.cat(Ps)
    Fe = torch.cat([Fe, torch.ones(len(Fe), 1, device=dev)], 1)
    Fe = F.normalize(Fe, dim=1)
    c = (P.T @ Fe) / (P.sum(0)[:, None] + 1e-8)
    # 官方实现的 labelset：只在模型实际预测到的类别里聚类，避免空类中心错配
    labelset = torch.where(torch.bincount(P.argmax(1), minlength=n_cls) > 0)[0]
    for _ in range(2):
        sim = Fe @ F.normalize(c, dim=1).T
        pred = labelset[sim[:, labelset].argmax(1)]
        onehot = F.one_hot(pred, n_cls).float()
        c = (onehot.T @ Fe) / (onehot.sum(0)[:, None] + 1e-8)
    sim = Fe @ F.normalize(c, dim=1).T
    return labelset[sim[:, labelset].argmax(1)]


# ============================================================================ 本文：可靠性划分 + 原型
@torch.no_grad()
def reliability_split(model, dl_full, dev, n_cls, rho, use_vote, T=0.1, n_proto=1, seed=0):
    """返回：伪标签 pl、可靠掩码 rel、原型软标签 q、原型软标签置信度 w。
    n_proto>1 时每类用 k-means 求多个子原型：同一故障类别包含不同轴承 / 损伤程度，特征分布是多峰的，
    单个类中心会把离它最远的那一簇（常常是整个轴承）推向别的类别。"""
    model.eval()
    preds, probs, feats = [], [], []
    augs = VOTE_AUGS if use_vote else VOTE_AUGS[:1]
    for x, _, _ in dl_full:
        x = x.to(dev)
        pv, pp = [], []
        for k, aug in enumerate(augs):
            o, f = model(aug(x), return_feat=True)
            p = o.softmax(1)
            pv.append(p.argmax(1)); pp.append(p)
            if k == 0:
                feats.append(F.normalize(f, dim=1))
        preds.append(torch.stack(pv)); probs.append(torch.stack(pp).mean(0))
    V = torch.cat(preds, 1)          # (n_aug, N)
    P = torch.cat(probs)             # (N, C) 各视图平均概率
    Fe = torch.cat(feats)            # (N, D)
    pl = V[0]                        # 原始视图预测
    agree = (V == pl[None]).all(0)   # 所有增强视图投票一致
    conf = P.gather(1, pl[:, None]).squeeze(1)

    # 类别均衡选择：每个预测类别内部只取置信度前 rho 比例，避免强势类别独占可靠集
    rel = torch.zeros_like(agree)
    for c in range(n_cls):
        idx = torch.where((pl == c) & agree)[0]
        if len(idx) == 0:
            continue
        k = max(1, int(round(rho * (pl == c).sum().item())))
        top = idx[conf[idx].argsort(descending=True)[:k]]
        rel[top] = True

    # 原型：可靠样本特征的类均值（n_proto>1 时为 k-means 子原型）；某类没有可靠样本时用概率加权兜底
    from sklearn.cluster import KMeans
    protos = torch.zeros(n_cls, n_proto, Fe.shape[1], device=dev)
    for c in range(n_cls):
        m = rel & (pl == c)
        if not m.any():
            protos[c] = ((P[:, c:c + 1] * Fe).sum(0) / (P[:, c].sum() + 1e-8))[None]
        elif n_proto == 1 or m.sum() < 5 * n_proto:
            protos[c] = Fe[m].mean(0)[None]
        else:
            km = KMeans(n_proto, n_init=3, random_state=seed).fit(Fe[m].cpu().numpy())
            protos[c] = torch.from_numpy(km.cluster_centers_).float().to(dev)
    protos = F.normalize(protos, dim=2)
    sim = torch.einsum("nd,ckd->nck", Fe, protos).max(2).values  # 与每类最近子原型的相似度
    q = (sim / T).softmax(1)             # 原型软标签
    w = q.max(1).values                  # 越接近某原型，监督权重越大
    return pl, rel, q, w


# ============================================================================ 无标签模型选择：SND（Saito et al., ICCV 2021）
@torch.no_grad()
def snd_score(model, dl_full, dev, tau=0.05):
    """Soft Neighborhood Density：目标样本 softmax 输出两两相似度 → 软邻居分布的平均熵，越大越好。只用无标签目标数据。"""
    model.eval()
    P = torch.cat([model(x.to(dev)).softmax(1) for x, _, _ in dl_full])
    P = F.normalize(P, dim=1)
    S = P @ P.T / tau
    S.fill_diagonal_(-float("inf"))
    Q = S.softmax(1)
    return float(-(Q * torch.log(Q + 1e-12)).sum(1).mean())


def session_peers(tags, S):
    """采集会话先验：同一段录音中连续 S 个窗口视为同一“会话”，健康状态相同（这只用到采集元数据，不用故障标签）。
    返回 (N, S-1) 的同会话样本下标矩阵（不足处用自身填充）以及有效掩码。"""
    tags = np.asarray([str(t) for t in tags])
    sid = np.empty(len(tags), dtype=object)
    pos = {}
    for i, t in enumerate(tags):           # 同一 tag 内的样本在文件中按时间顺序排列（prepare.py 保证）
        k = pos.get(t, 0); pos[t] = k + 1
        sid[i] = f"{t}#{k // S}"
    groups = {}
    for i, s in enumerate(sid):
        groups.setdefault(s, []).append(i)
    peers = np.tile(np.arange(len(tags))[:, None], (1, max(1, S - 1)))
    mask = np.zeros_like(peers, dtype=np.float32)
    for g in groups.values():
        for i in g:
            others = [j for j in g if j != i][: S - 1]
            peers[i, :len(others)] = others
            mask[i, :len(others)] = 1.0
    return torch.from_numpy(peers), torch.from_numpy(mask), len(groups)


# ============================================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["shot", "nrc", "aad", "rnc", "ours"])
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    # SHOT
    ap.add_argument("--cls_par", type=float, default=0.3)
    # NRC
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--KK", type=int, default=3)
    # AaD / RNC（新方法）
    ap.add_argument("--disp_beta", type=float, default=1.0, help="AaD 分散项权重的衰减指数")
    ap.add_argument("--session", type=int, default=0, help="RNC：采集会话长度（连续窗口数），0 关闭")
    ap.add_argument("--w_sess", type=float, default=1.0, help="RNC：会话一致性权重")
    ap.add_argument("--freq_w", type=float, default=0.0, help="RNC：弱/强（频带掩码）一致性权重，0 关闭")
    # 无标签模型选择（任何方法都可用）
    ap.add_argument("--select", default="last", choices=["last", "snd"],
                    help="snd：按 SND 在各轮（含适应前）中选模型，额外写一行 note 加 _snd 的结果")
    # ours
    ap.add_argument("--rho0", type=float, default=0.5, help="初始可靠比例")
    ap.add_argument("--rho1", type=float, default=0.8, help="最终可靠比例（线性增长）")
    ap.add_argument("--beta", type=float, default=0.5, help="不可靠样本原型软标签损失权重")
    ap.add_argument("--gamma", type=float, default=1.0, help="频带一致性损失权重")
    ap.add_argument("--alpha", type=float, default=1.0, help="信息最大化损失权重")
    ap.add_argument("--T", type=float, default=0.1, help="原型温度")
    ap.add_argument("--n_proto", type=int, default=1, help="每类子原型个数（多原型，处理类内多峰）")
    ap.add_argument("--no_vote", action="store_true")
    ap.add_argument("--no_proto", action="store_true")
    ap.add_argument("--no_freq", action="store_true")
    ap.add_argument("--no_im", action="store_true")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="源模型锚定权重（KL 到冻结源模型的预测），0 表示关闭；防止整簇样本被翻转")
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--curve_dir", default="results/curves", help="伪标签准确率曲线（论文作图用）")
    ap.add_argument("--save_feats", action="store_true", help="保存适应前后目标测试特征，供 t-SNE")
    ap.add_argument("--test_snr", type=int, nargs="*", default=[], help="如 -4 0 4 8，噪声鲁棒性实验")
    ap.add_argument("--save_model", action="store_true", help="保存适应后的模型（部署实验用）")
    ap.add_argument("--timing", default="", help="记录适应训练耗时的 CSV（如 results/timing.csv），默认不记录")
    a = ap.parse_args()
    set_seed(a.seed); dev = device()

    ck = torch.load(os.path.join(a.ckpt, f"{a.src}_w{a.width}_s{a.seed}.pt"), map_location=dev, weights_only=False)
    n_cls = ck["n_cls"]
    model = Net(n_cls, a.width).to(dev)
    model.load_state_dict(ck["state"])
    src_model = None
    if a.method == "ours" and a.anchor > 0:  # 冻结的源模型副本，只用于锚定，不参与训练
        import copy
        src_model = copy.deepcopy(model).eval()
        for p in src_model.parameters():
            p.requires_grad = False
    for p in model.C.parameters():  # 冻结分类器（假设迁移），只调特征提取器
        p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)

    dt, tmeta = load_domain(a.data_root, a.tgt)
    Xt, yt = dt["X_train"], dt["y_train"]   # yt 只用于记录伪标签准确率，绝不参与训练
    dl_t = loader(Xt, yt, a.bs, True, drop_last=len(yt) > a.bs)
    dl_full = loader(Xt, yt, 256, False)
    dl_te = loader(dt["X_test"], dt["y_test"], 256, False)
    yt_t = torch.from_numpy(yt).to(dev)

    tag = "full" if a.method == "ours" else ""
    if a.method == "ours":
        off = [n for n in ("vote", "proto", "freq", "im") if getattr(a, f"no_{n}")]
        tag = "full" if not off else "-" + "-".join(off)
        if a.n_proto > 1:
            tag += f"_mp{a.n_proto}"
        if a.anchor > 0:
            tag += f"_anc{a.anchor:g}"
    if a.method == "rnc":
        tag = rnc_tag(a)
    r0 = evaluate(model, dl_te, dev)
    print(f"[before adaptation] tgt-test acc {r0['acc']:.2f} f1 {r0['f1']:.2f}")
    if a.save_feats:
        dump_feats(model, dl_te, dev, a, "before")

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    _sync(); t_start = time.perf_counter(); t_eval = 0.0  # 计时从这里开始：含记忆库初始化、伪标签更新和全部训练迭代
    crit = LabelSmoothCE(0.1)
    curve = []
    N = len(yt)
    if a.method == "rnc" and a.session > 1:
        peers, pmask, n_sess = session_peers(dt["src_train"], a.session)
        peers, pmask = peers.to(dev), pmask.to(dev)
        print(f"  会话先验：{n_sess} 个会话，每个最多 {a.session} 个连续窗口")
    total_it, it = a.epochs * len(dl_t), 0
    snd_hist, best = [], None
    if a.select == "snd":
        import copy
        s0 = snd_score(model, dl_full, dev)
        snd_hist.append((-1, s0, r0["acc"]))
        best = (s0, -1, copy.deepcopy(model.state_dict()))
    if a.method in ("nrc", "aad", "rnc"):
        model.eval()
        with torch.no_grad():
            fb, sb = torch.zeros(N, 256, device=dev), torch.zeros(N, n_cls, device=dev)
            for x, _, i in dl_full:
                o, f = model(x.to(dev), return_feat=True)
                fb[i] = F.normalize(f, dim=1); sb[i] = o.softmax(1)

    for ep in range(a.epochs):
        # ---- 每轮开始：更新伪标签
        if a.method == "shot":
            pl = shot_pseudo_labels(model, dl_full, dev, n_cls)
            curve.append((ep, (pl == yt_t).float().mean().item() * 100, 1.0))
        elif a.method == "ours":
            rho = a.rho0 + (a.rho1 - a.rho0) * ep / max(1, a.epochs - 1)
            pl, rel, q, w = reliability_split(model, dl_full, dev, n_cls, rho, not a.no_vote, a.T,
                                              a.n_proto, a.seed)
            pl_acc = (pl[rel] == yt_t[rel]).float().mean().item() * 100 if rel.any() else 0
            curve.append((ep, pl_acc, rel.float().mean().item()))
        model.train()
        for x, _, idx in dl_t:
            x, idx = x.to(dev), idx.to(dev)
            if a.method == "shot":
                out = model(weak_aug(x))
                loss = a.cls_par * F.cross_entropy(out, pl[idx]) + im_loss(out)
            elif a.method == "nrc":
                out, f = model(weak_aug(x), return_feat=True)
                p = out.softmax(1)
                with torch.no_grad():
                    fn = F.normalize(f, dim=1)
                    fb[idx] = fn.detach(); sb[idx] = p.detach()
                    sim = fn @ fb.T
                    sim[torch.arange(len(idx)), idx] = -1
                    near = sim.topk(a.K, dim=1).indices                  # (B,K)
                    sim2 = fb[near] @ fb.T                               # (B,K,N)
                    near2 = sim2.topk(a.KK + 1, dim=2).indices[:, :, 1:]  # 去掉自身
                    # 互为近邻（邻居的近邻中包含自己）权重为 1，否则 0.1
                    recip = (near2 == idx[:, None, None]).any(2).float()
                    w1 = torch.where(recip > 0, torch.ones_like(recip), 0.1 * torch.ones_like(recip))
                    s1 = sb[near]                                        # (B,K,C)
                    s2 = sb[near2.flatten(1)]                            # (B,K*KK,C)
                loss = -(w1 * (p[:, None] * s1).sum(2)).sum(1).mean() \
                       - 0.1 * (p[:, None] * s2).sum(2).sum(1).mean()
                pm = p.mean(0)
                loss = loss + (pm * torch.log(pm + 1e-5)).sum()
            elif a.method in ("aad", "rnc"):
                # (N) AaD：吸引特征空间近邻的预测，分散批内其他样本的预测（Yang et al., NeurIPS 2022）
                xw = weak_aug(x)
                out, f = model(xw, return_feat=True)
                p = out.softmax(1)
                with torch.no_grad():
                    fn = F.normalize(f, dim=1)
                    fb[idx] = fn.detach(); sb[idx] = p.detach()
                    sim = fn @ fb.T
                    sim[torch.arange(len(idx)), idx] = -1
                    near = sim.topk(a.K, dim=1).indices
                lam = (1 + 10 * it / max(1, total_it)) ** (-a.disp_beta)
                attract = -(p[:, None] * sb[near]).sum(2).sum(1).mean()
                pp = p @ p.T
                disperse = (pp.sum(1) - pp.diag()).mean()
                loss = attract + lam * disperse
                if a.method == "rnc" and a.session > 1:
                    # (S) 会话一致性：同一采集会话（同一段录音中相邻的窗口）预测应一致
                    sp = sb[peers[idx]]                           # (B, S-1, C)，来自记忆库
                    m = pmask[idx]
                    loss = loss - a.w_sess * ((p[:, None] * sp).sum(2) * m).sum(1).div(m.sum(1).clamp(min=1)).mean()
                if a.method == "rnc" and a.freq_w > 0:
                    # (F) 弱增强 / 频带掩码强增强之间的一致性
                    out_s = model(strong_aug(x))
                    loss = loss + a.freq_w * F.kl_div(out_s.log_softmax(1), p.detach(), reduction="batchmean")
            else:  # ours
                xw, xs = weak_aug(x), strong_aug(x)
                out_w = model(xw)
                loss = out_w.new_zeros(())
                r = rel[idx]
                if r.any():                                   # (V) 可靠样本：硬伪标签
                    loss = loss + crit(out_w[r], pl[idx][r])
                if not a.no_proto and (~r).any():             # (P) 不可靠样本：原型软标签（加权 KL）
                    logp = out_w[~r].log_softmax(1)
                    kl = F.kl_div(logp, q[idx][~r], reduction="none").sum(1)
                    loss = loss + a.beta * (w[idx][~r] * kl).mean()
                if not a.no_freq:                             # (F) 弱/强（频带掩码）一致性
                    out_s = model(xs)
                    tgt_p = out_w.detach().softmax(1)
                    loss = loss + a.gamma * F.kl_div(out_s.log_softmax(1), tgt_p, reduction="batchmean")
                if not a.no_im:                               # (IM) 防塌缩
                    loss = loss + a.alpha * im_loss(out_w)
                if a.anchor > 0:                              # (A) 源模型锚定：阻止整簇样本被翻转到错误类别
                    with torch.no_grad():
                        p_src = src_model(xw).softmax(1)
                    loss = loss + a.anchor * F.kl_div(out_w.log_softmax(1), p_src, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1
        if a.select == "snd":
            s = snd_score(model, dl_full, dev)
            snd_hist.append((ep, s, float("nan")))
            if s > best[0]:
                best = (s, ep, copy.deepcopy(model.state_dict()))
        if ep % 5 == 0 or ep == a.epochs - 1:
            _sync(); te0 = time.perf_counter()
            r = evaluate(model, dl_te, dev)
            extra = f" | pl-acc {curve[-1][1]:.1f} rel {curve[-1][2]:.2f}" if curve else ""
            print(f"ep {ep:3d} loss {loss.item():.4f} tgt-test acc {r['acc']:.2f}{extra}")
            _sync(); t_eval += time.perf_counter() - te0  # 中途的测试集评估只是打印日志，不计入适应耗时

    _sync(); adapt_sec = time.perf_counter() - t_start - t_eval
    print(f"  适应训练耗时 {adapt_sec:.1f} s（{a.epochs} 轮，每轮 {adapt_sec / max(1, a.epochs):.2f} s，不含中途评估）")
    if a.timing:
        os.makedirs(os.path.dirname(a.timing) or ".", exist_ok=True)
        new = not os.path.exists(a.timing)
        with open(a.timing, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["method", "note", "src", "tgt", "seed", "epochs", "n_target", "device", "adapt_sec"])
            w.writerow([a.method, tag, a.src, a.tgt, a.seed, a.epochs, N,
                        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", f"{adapt_sec:.2f}"])

    r = evaluate(model, dl_te, dev)  # 固定轮数取最后一轮，不用目标标签挑模型
    name = a.method if a.method != "ours" else "ours"
    print(f"{name}[{tag}] {a.src}->{a.tgt}: acc {r['acc']:.2f} f1 {r['f1']:.2f}")
    append_result(a.results, dict(method=name, src=a.src, tgt=a.tgt, seed=a.seed,
                                  acc=f"{r['acc']:.2f}", f1=f"{r['f1']:.2f}", note=tag))
    if "src_test" in dt.files:
        per_group_report(model, dt["X_test"], dt["y_test"], dt["src_test"], dev, tmeta["classes"],
                         per_group_path(a.results), dict(method=name, src=a.src, tgt=a.tgt, seed=a.seed, note=tag))
    if a.select == "snd":  # 额外记录：按 SND（无标签）选出的模型
        last_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best[2])
        rs = evaluate(model, dl_te, dev)
        stag = (tag + "_snd") if tag else "snd"
        print(f"{name}[{stag}] {a.src}->{a.tgt}: acc {rs['acc']:.2f} f1 {rs['f1']:.2f}  (SND 选中第 {best[1]} 轮，-1 = 不适应)")
        append_result(a.results, dict(method=name, src=a.src, tgt=a.tgt, seed=a.seed,
                                      acc=f"{rs['acc']:.2f}", f1=f"{rs['f1']:.2f}", note=stag))
        if "src_test" in dt.files:
            per_group_report(model, dt["X_test"], dt["y_test"], dt["src_test"], dev, tmeta["classes"],
                             per_group_path(a.results), dict(method=name, src=a.src, tgt=a.tgt, seed=a.seed, note=stag))
        model.load_state_dict(last_state)
    if a.save_model:
        os.makedirs("ckpt/adapted", exist_ok=True)
        torch.save({"state": model.state_dict(), "n_cls": n_cls, "width": a.width},
                   f"ckpt/adapted/{name}{tag}_{a.src}_{a.tgt}_s{a.seed}.pt")
    for snr in a.test_snr:  # 噪声鲁棒性实验：在目标测试样本上叠加指定信噪比的高斯白噪声
        rs = evaluate_noisy(model, dl_te, dev, snr)
        print(f"  SNR {snr:+d} dB: acc {rs['acc']:.2f} f1 {rs['f1']:.2f}")
        append_result(a.results, dict(method=name, src=a.src, tgt=a.tgt, seed=a.seed,
                                      acc=f"{rs['acc']:.2f}", f1=f"{rs['f1']:.2f}", note=f"{tag}|snr={snr}"))
    if curve:
        os.makedirs(a.curve_dir, exist_ok=True)
        np.savetxt(os.path.join(a.curve_dir, f"{name}{tag}_{a.src}_{a.tgt}_s{a.seed}.csv"), np.array(curve),
                   delimiter=",", header="epoch,pseudo_label_acc,reliable_ratio", comments="")
    if a.save_feats:
        dump_feats(model, dl_te, dev, a, "after")


def rnc_tag(a):
    """RNC 的结果标签：S<会话长度> / F<频带一致性权重>，全关时为空。run_all.py 用同样规则判断是否已跑过。"""
    parts = []
    if a.K != 5:
        parts.append(f"K{a.K}")
    if a.session > 1:
        parts.append(f"S{a.session}")
    if a.freq_w > 0:
        parts.append(f"F{a.freq_w:g}")
    return "_".join(parts)


@torch.no_grad()
def evaluate_noisy(model, dl, dev, snr_db, seed=123):
    from sklearn.metrics import accuracy_score, f1_score
    g = torch.Generator(device="cpu").manual_seed(seed)  # 固定噪声，方法间可比
    model.eval()
    P, Y = [], []
    for x, y, _ in dl:
        pw = x.pow(2).mean(-1, keepdim=True)
        xn = x + torch.randn(x.shape, generator=g) * torch.sqrt(pw / 10 ** (snr_db / 10))
        xn = (xn - xn.mean(-1, keepdim=True)) / (xn.std(-1, keepdim=True) + 1e-8)
        P.append(model(xn.to(dev)).argmax(1).cpu()); Y.append(y)
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    return {"acc": 100 * accuracy_score(Y, P), "f1": 100 * f1_score(Y, P, average="macro")}


@torch.no_grad()
def dump_feats(model, dl, dev, a, stage):
    model.eval()
    Fs, Ys = [], []
    for x, y, _ in dl:
        _, f = model(x.to(dev), return_feat=True)
        Fs.append(f.cpu()); Ys.append(y)
    os.makedirs("results/feats", exist_ok=True)
    np.savez(f"results/feats/{a.method}_{a.src}_{a.tgt}_s{a.seed}_{stage}.npz",
             feat=torch.cat(Fs).numpy(), y=torch.cat(Ys).numpy())


if __name__ == "__main__":
    main()
