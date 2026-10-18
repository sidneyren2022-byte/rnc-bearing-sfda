"""
可行性检查：按转速归一化的“阶次包络谱”能否解决跨工况的外圈/内圈混淆？

直接读 PU 原始 .mat（长窗 0.256 s），一次提取三种特征，用最简单的逻辑回归做跨工况 Source-only：
  hz    —— 普通包络谱（横轴为 Hz，5–300 Hz）
  order —— 阶次包络谱（横轴为转频的倍数，0.5–12 阶；外圈约 3.05 阶、内圈约 4.95 阶，不随转速漂移）
  both  —— 两者拼接
输出每个任务的准确率、逐类别准确率，以及外圈轴承的逐轴承准确率。
只用于决定“要不要做物理分支”，不是论文实验本身。

  python tools/order_envelope_check.py --raw data/raw/PU
  python tools/order_envelope_check.py --raw data/raw/PU --set 15
"""
import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from prepare import PU_CONDS, PU_GROUPS, PU_GROUPS15, load_pu_vib  # noqa: E402

FS = 64000
RPM = {"N15": 1500, "N09": 900}  # 工况名中的标称转速
KINDS = ("hz", "order", "both", "hz_w", "order_w")
NAMES = {"hz": "普通包络谱（Hz）", "order": "阶次包络谱（按转速归一化）", "both": "两者拼接",
         "hz_w": "预白化包络谱（Hz）", "order_w": "预白化阶次包络谱"}


def envelope_spectrum(x, fs, hp=1000.0):
    """高通（去掉低频轴频成分，保留共振带）→ 解析信号取包络 → 包络的幅值谱。"""
    n = len(x)
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(n, 1 / fs)
    X[f < hp] = 0
    full = np.zeros(n, complex); full[:len(X)] = X
    full[1:(n + 1) // 2] *= 2                     # 单边谱 → 解析信号
    env = np.abs(np.fft.ifft(full))
    E = np.abs(np.fft.rfft(env - env.mean())) / n
    return np.fft.rfftfreq(n, 1 / fs), E


def _norm(v):
    v = np.log(v + 1e-8)
    return (v - v.mean()) / (v.std() + 1e-8)


def prewhiten(f, E, fmax=400.0, win_hz=60.0):
    """谱预白化：在 Hz 域用中值滤波估计平滑背景（与转速无关的部分），从对数谱中减去，只保留高出背景的谱峰。"""
    from scipy.ndimage import median_filter
    m = f <= fmax + win_hz
    logE = np.log(E[m] + 1e-8)
    k = max(3, int(win_hz / (f[1] - f[0])) | 1)
    peaks = np.maximum(logE - median_filter(logE, size=k, mode="nearest"), 0.0)
    return f[m], peaks


def features(x, fs, rpm, nbins=480):
    f, E = envelope_spectrum(x, fs)
    fr = rpm / 60.0
    hz_grid, ord_grid = np.linspace(5.0, 300.0, nbins), np.linspace(0.5, 12.0, nbins)
    hz = _norm(np.interp(hz_grid, f, E))
    order = _norm(np.interp(ord_grid, f / fr, E))
    fw, P = prewhiten(f, E)
    # 白化后不再逐样本标准化：保留谱峰的绝对显著程度，健康样本应接近全 0
    hz_w = np.interp(hz_grid, fw, P)
    order_w = np.interp(ord_grid, fw / fr, P)
    return {"hz": hz, "order": order, "both": np.concatenate([hz, order]), "hz_w": hz_w, "order_w": order_w}


def build(raw, groups, W):
    data = {}
    classes = list(groups)
    for ci, cond in enumerate(PU_CONDS):
        rpm = RPM[cond[:3]]
        X = {k: {"tr": [], "te": []} for k in KINDS}
        y = {"tr": [], "te": []}; g = {"tr": [], "te": []}
        for lab, cls in enumerate(classes):
            for b in groups[cls]:
                for r in range(1, 21):
                    p = os.path.join(raw, b, f"{cond}_{b}_{r}.mat")
                    if not os.path.exists(p):
                        continue
                    sig = load_pu_vib(p)
                    part = "tr" if r <= 14 else "te"  # 与主实验相同：按录音划分，无泄漏
                    for s in range(0, len(sig) - W + 1, W):
                        feats = features(sig[s:s + W], FS, rpm)
                        for k in KINDS:
                            X[k][part].append(feats[k])
                        y[part].append(lab); g[part].append(b)
        data[ci] = {k: {pt: (np.array(X[k][pt]), np.array(y[pt]), np.array(g[pt])) for pt in ("tr", "te")}
                    for k in KINDS}
        print(f"  pu_{ci} {cond} ({rpm} rpm): train {len(y['tr'])} test {len(y['te'])}")
    return data, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/PU")
    ap.add_argument("--set", type=int, default=9, choices=[9, 15])
    ap.add_argument("--W", type=int, default=16384, help="窗长（点）；16384 点 = 0.256 s，频率分辨率约 3.9 Hz")
    a = ap.parse_args()
    groups = PU_GROUPS15 if a.set == 15 else PU_GROUPS
    or_bearings = groups["OR"]

    print("读取并提取特征（三种特征一次完成）…")
    data, classes = build(a.raw, groups, a.W)
    summary = {}
    for kind in KINDS:
        print(f"\n=== 特征：{NAMES[kind]} ===")
        accs, wrong, cls_acc_900, cls_acc_1500 = [], 0, [], []
        for s in range(4):
            Xs, ys, _ = data[s][kind]["tr"]
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)).fit(Xs, ys)
            for t in range(4):
                if s == t:
                    continue
                Xt, yt, gt = data[t][kind]["te"]
                p = clf.predict(Xt)
                acc = 100 * (p == yt).mean(); accs.append(acc)
                per_cls = [100 * (p[yt == c] == c).mean() for c in range(len(classes))]
                (cls_acc_900 if 1 in (s, t) else cls_acc_1500).append(per_cls)
                detail = []
                for b in sorted(set(gt)):
                    m = gt == b
                    ab = 100 * (p[m] == yt[m]).mean()
                    wrong += ab < 50
                    if b in or_bearings:
                        detail.append(f"{b} {ab:5.1f}")
                cls_txt = " ".join(f"{c} {v:5.1f}" for c, v in zip(classes, per_cls))
                print(f"  pu_{s}->pu_{t}: acc {acc:6.2f} | 各类 {cls_txt} | 外圈逐轴承 " + ", ".join(detail))
        summary[kind] = (np.mean(accs), wrong, np.mean(cls_acc_900, 0), np.mean(cls_acc_1500, 0))
        print(f"  平均 {np.mean(accs):.2f}%，判错轴承共 {wrong} 个")

    print("\n=== 结论 ===")
    hdr = " ".join(f"{c:>8}" for c in classes)
    print(f"  {'特征':<6} {'平均':>7} {'判错轴承':>6} | 涉及 900 rpm 的 6 个任务各类准确率 {hdr} | 1500 rpm 之间 6 个任务 {hdr}")
    for kind, (m, w, c9, c15) in summary.items():
        print(f"  {kind:<6} {m:7.2f} {w:6d}     | {' '.join(f'{v:8.1f}' for v in c9)} | {' '.join(f'{v:8.1f}' for v in c15)}")


if __name__ == "__main__":
    main()
