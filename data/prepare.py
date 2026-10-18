"""
把原始 .mat/.csv 切成样本，按“无泄漏”规则划分训练/测试，每个域存一个 npz。

无泄漏规则（论文“评测协议”一节照此描述）：
  * PU：每个轴承每个工况有 20 段独立录音。第 1-14 段只进训练集，第 15-20 段只进测试集，
        测试样本与训练样本来自不同录音，不存在重叠窗口。
  * CWRU / JNU：每个类别每个工况只有一段录音，无法按文件划分，
        因此按时间切分：信号前 70% 切训练样本，留出一个窗长的间隔，后 30% 切测试样本。
        训练区内部可以 50% 重叠增广，测试区不重叠，两区之间没有任何共享采样点。

输出：data/processed/<域名>.npz，键 X_train y_train X_test y_test src_train src_test
      以及 data/processed/<域名>.json（类别名、采样率、窗长）。

用法：
  python data/prepare.py cwru --raw data/raw/CWRU                 # cwru_0..cwru_3 (10 类)
  python data/prepare.py cwru --raw data/raw/CWRU --classes 4     # cwru4_0..3 (正常/内圈/滚动体/外圈)
  python data/prepare.py pu   --raw data/raw/PU                   # pu_0..pu_3 (3 类)
  python data/prepare.py jnu  --raw data/raw/JNU                  # jnu_600/800/1000 (4 类, 重采样到 12 kHz)
  python data/prepare.py jnu_sdalr --raw data/raw/JNU_sdalr       # SDALR 网盘版（已切好的 txt），自动拼回再划分
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.io import loadmat

OUT = "data/processed"

# ----------------------------------------------------------------------------- 通用
def windows(sig, L, stride, max_n=None):
    n = (len(sig) - L) // stride + 1
    if n <= 0:
        return np.zeros((0, L), np.float32)
    idx = np.arange(n) * stride
    if max_n is not None and n > max_n:  # 均匀抽取，避免只取开头
        idx = idx[np.linspace(0, n - 1, max_n).astype(int)]
    return np.stack([sig[i:i + L] for i in idx]).astype(np.float32)


def time_split(sig, L, train_ratio=0.7, train_stride=None, max_train=None, max_test=None):
    """前 train_ratio 训练，间隔 L，后面测试。"""
    cut = int(len(sig) * train_ratio)
    tr = windows(sig[:cut], L, train_stride or L // 2, max_train)
    te = windows(sig[cut + L:], L, L, max_test)
    return tr, te


def save(name, Xtr, ytr, Xte, yte, str_, ste, meta):
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), X_train=Xtr, y_train=ytr,
                        X_test=Xte, y_test=yte, src_train=np.array(str_), src_test=np.array(ste))
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    cnt = lambda y: np.bincount(y, minlength=len(meta["classes"])).tolist()
    print(f"[{name}] train {Xtr.shape} {cnt(ytr)} | test {Xte.shape} {cnt(yte)}")
    # 泄漏自检：训练与测试的来源录音段不能有交集（PU），CWRU/JNU 用时间区间保证
    if meta.get("split") == "by_recording":
        assert not (set(str_) & set(ste)), "训练/测试录音重叠！"


# ----------------------------------------------------------------------------- CWRU
CWRU_FILES = {
    0: [97, 105, 118, 130, 169, 185, 197, 209, 222, 234],
    1: [98, 106, 119, 131, 170, 186, 198, 210, 223, 235],
    2: [99, 107, 120, 132, 171, 187, 199, 211, 224, 236],
    3: [100, 108, 121, 133, 172, 188, 200, 212, 225, 237],
}
CWRU10 = ["Normal", "IR007", "B007", "OR007", "IR014", "B014", "OR014", "IR021", "B021", "OR021"]
CWRU4 = ["Normal", "IR", "Ball", "OR"]          # 与 JNU 对齐的 4 类
TO4 = [0, 1, 2, 3, 1, 2, 3, 1, 2, 3]


def load_cwru_de(path, n):
    m = loadmat(path)
    keys = [k for k in m if k.endswith("_DE_time")]
    # 98.mat 等文件里同时有 X098 和 X099，优先选与文件号一致的通道
    pref = [k for k in keys if k.startswith(f"X{n:03d}")]
    return m[(pref or keys)[0]].ravel().astype(np.float32)


def prep_cwru(raw, L, classes):
    names = CWRU10 if classes == 10 else CWRU4
    for load, files in CWRU_FILES.items():
        Xtr, ytr, Xte, yte, str_, ste = [], [], [], [], [], []
        for c, n in enumerate(files):
            sig = load_cwru_de(os.path.join(raw, f"{n}.mat"), n)
            lab = c if classes == 10 else TO4[c]
            # 正常信号约为故障信号 2-4 倍长，限制每个文件的样本数以平衡类别
            cap_tr, cap_te = (200, 80) if classes == 10 else (120, 50)
            tr, te = time_split(sig, L, max_train=cap_tr, max_test=cap_te)
            Xtr.append(tr); ytr += [lab] * len(tr); str_ += [f"{n}.mat[train]"] * len(tr)
            Xte.append(te); yte += [lab] * len(te); ste += [f"{n}.mat[test]"] * len(te)
        name = f"cwru_{load}" if classes == 10 else f"cwru4_{load}"
        save(name, np.concatenate(Xtr), np.array(ytr), np.concatenate(Xte), np.array(yte), str_, ste,
             dict(dataset="CWRU", load_hp=load, fs=12000, L=L, classes=names, split="by_time_70_30"))


# ----------------------------------------------------------------------------- PU
PU_CONDS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
PU_GROUPS = {  # 3 类：健康 / 外圈真实损伤 / 内圈真实损伤（每类 3 个轴承）
    "Healthy": ["K001", "K002", "K003"],
    "OR": ["KA04", "KA15", "KA16"],
    "IR": ["KI14", "KI16", "KI18"],
}
PU_GROUPS15 = {  # 每类 5 个轴承：单个轴承只占 1/15，评估更稳定（需额外下载 K004 K005 KA22 KA30 KI04 KI21）
    "Healthy": ["K001", "K002", "K003", "K004", "K005"],
    "OR": ["KA04", "KA15", "KA16", "KA22", "KA30"],
    "IR": ["KI04", "KI14", "KI16", "KI18", "KI21"],
}


def load_pu_vib(path):
    m = loadmat(path)
    key = [k for k in m if not k.startswith("__")][0]
    s = m[key][0, 0]
    Y = s["Y"][0]
    for ch in Y:  # 按通道名找 vibration_1，比硬编码下标更稳妥
        if str(np.squeeze(ch["Name"])) == "vibration_1":
            return np.squeeze(ch["Data"]).astype(np.float32)
    return np.squeeze(Y[6]["Data"]).astype(np.float32)  # 兜底：与 UDTL 相同的下标


def prep_pu(raw, L, train_recs, test_recs, per_rec_train, per_rec_test, down, groups=None, prefix="pu"):
    PU_GROUPS = groups or globals()["PU_GROUPS"]
    classes = list(PU_GROUPS)
    for ci, cond in enumerate(PU_CONDS):
        Xtr, ytr, Xte, yte, str_, ste = [], [], [], [], [], []
        for lab, g in enumerate(classes):
            for b in PU_GROUPS[g]:
                for r in list(train_recs) + list(test_recs):
                    f = os.path.join(raw, b, f"{cond}_{b}_{r}.mat")
                    if not os.path.exists(f):
                        print(f"  缺文件 {f}，跳过")
                        continue
                    sig = load_pu_vib(f)[::down]
                    is_tr = r in train_recs
                    w = windows(sig, L, L, per_rec_train if is_tr else per_rec_test)
                    tag = f"{b}_{cond}_{r}"
                    if is_tr:
                        Xtr.append(w); ytr += [lab] * len(w); str_ += [tag] * len(w)
                    else:
                        Xte.append(w); yte += [lab] * len(w); ste += [tag] * len(w)
        if not Xtr or not Xte:
            print(f"[{prefix}_{ci}] 工况 {cond} 没有找到数据，跳过")
            continue
        save(f"{prefix}_{ci}", np.concatenate(Xtr), np.array(ytr), np.concatenate(Xte), np.array(yte), str_, ste,
             dict(dataset="PU", condition=cond, fs=64000 // down, L=L, classes=classes,
                  bearings=PU_GROUPS, split="by_recording", train_recordings=list(train_recs),
                  test_recordings=list(test_recs)))


# ----------------------------------------------------------------------------- JNU
def prep_jnu(raw, L, fs_out):
    """JNU 原始文件为 csv：n600_3_2.csv, ib600_2.csv, ob600_2.csv, tb600_2.csv 等。"""
    from scipy.signal import resample_poly
    order = [("n", 0), ("ib", 1), ("tb", 2), ("ob", 3)]  # 与 CWRU4 顺序一致：Normal IR Ball OR
    for speed in (600, 800, 1000):
        Xtr, ytr, Xte, yte, str_, ste = [], [], [], [], [], []
        for prefix, lab in order:
            fs = [p for p in glob.glob(os.path.join(raw, "*.csv"))
                  if re.match(rf"^{prefix}{speed}[_\.]", os.path.basename(p))]
            if not fs:
                print(f"  未找到 {prefix}{speed}*.csv，跳过")
                continue
            sig = np.loadtxt(fs[0], delimiter=",", dtype=np.float32).ravel()
            sig = resample_poly(sig, fs_out, 50000).astype(np.float32)
            tr, te = time_split(sig, L, max_train=120, max_test=50)
            Xtr.append(tr); ytr += [lab] * len(tr); str_ += [os.path.basename(fs[0])] * len(tr)
            Xte.append(te); yte += [lab] * len(te); ste += [os.path.basename(fs[0])] * len(te)
        if Xtr:
            save(f"jnu_{speed}", np.concatenate(Xtr), np.array(ytr), np.concatenate(Xte), np.array(yte),
                 str_, ste, dict(dataset="JNU", speed=speed, fs=fs_out, L=L, classes=CWRU4,
                                 split="by_time_70_30"))


# ----------------------------------------------------------------------------- JNU（SDALR 预切分版本）
def _detect_stride(a, b):
    """若 b 是 a 向后平移 s 个点的窗口（a[s:] == b[:L-s]），返回 s；否则 None。"""
    L = len(a)
    for s in np.where(a == b[0])[0]:
        if s > 0 and np.array_equal(a[s:], b[:L - s]):
            return int(s)
    return None


def _load_txt(path):
    return np.loadtxt(path, dtype=np.float64)  # 保持文本精度，便于精确比对重叠


def stitch_windows(files):
    """把按时间顺序、相互重叠的窗口拼回连续信号。无法确认重叠时返回 None。"""
    first = [_load_txt(f) for f in files[:6]]
    strides = [_detect_stride(first[i], first[i + 1]) for i in range(5)]
    if None in strides or len(set(strides)) != 1:
        return None, strides
    s = strides[0]
    sig = [first[0]]
    prev = first[0]
    for k, f in enumerate(files[1:], 1):
        cur = first[k] if k < len(first) else _load_txt(f)
        if k % 200 == 0 and not np.array_equal(prev[s:], cur[:len(cur) - s]):  # 抽查：顺序是否始终连续
            return None, [s, f"断在第 {k} 个文件"]
        sig.append(cur[-s:])
        prev = cur
    return np.concatenate(sig), s


def reconstruct_by_overlap(W, min_overlap=64):
    """窗口顺序未知时：找出所有互相重叠的窗口对，推算每个窗口在原始时间轴上的位置，
    把信号重建为若干段互不重叠的连续片段（不同片段之间没有共享采样点）。
    W: (N, Lw) 原始窗口。返回 (片段列表, 诊断信息)。"""
    N, Lw = W.shape
    big = W.ravel()
    owner = np.repeat(np.arange(N), Lw)
    local = np.tile(np.arange(Lw), N)
    adj = [[] for _ in range(N)]              # (邻居, 偏移)：pos[邻居] = pos[自己] + 偏移
    has_overlap = np.zeros(N, bool)
    for j in range(N):
        w = W[j]
        cand = np.where(big == w[0])[0]
        cand = cand[(local[cand] <= Lw - min_overlap)]
        if len(cand):  # 用前 4 个点快速筛选
            ok = np.ones(len(cand), bool)
            for k in range(1, 4):
                ok &= big[cand + k] == w[k]
            cand = cand[ok]
        for p in cand:
            i, s = owner[p], local[p]
            if i == j:
                continue
            if np.array_equal(W[i, s:], w[:Lw - s]):
                adj[i].append((j, s)); adj[j].append((i, -s))
                has_overlap[i] = has_overlap[j] = True
    frac = has_overlap.mean()
    if frac < 0.5:
        return None, f"只有 {frac:.0%} 的样本检测到与其他样本重叠（可能每个样本被单独归一化过），无法排除泄漏"
    # BFS 求每个连通分量内的相对位置，再把窗口写回时间轴
    pos = np.full(N, np.iinfo(np.int64).min, dtype=np.int64)
    segments, n_comp, conflicts = [], 0, 0
    for root in range(N):
        if pos[root] != np.iinfo(np.int64).min:
            continue
        n_comp += 1
        pos[root] = 0
        comp, stack = [root], [root]
        while stack:
            u = stack.pop()
            for v, off in adj[u]:
                if pos[v] == np.iinfo(np.int64).min:
                    pos[v] = pos[u] + off; comp.append(v); stack.append(v)
        lo = pos[comp].min()
        span = pos[comp].max() - lo + Lw
        sig = np.full(span, np.nan)
        for u in comp:
            a = pos[u] - lo
            seg = sig[a:a + Lw]
            m = ~np.isnan(seg)
            if m.any() and not np.array_equal(seg[m], W[u][m]):
                conflicts += 1
            sig[a:a + Lw] = W[u]
        # 按覆盖情况切成连续片段（理论上一个分量内没有空洞）
        valid = ~np.isnan(sig)
        edges = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            segments.append(sig[a:b])
    if conflicts:
        return None, f"重建时有 {conflicts} 个窗口与已放置的数据冲突，位置推算不可靠"
    lens = sorted((len(s) for s in segments), reverse=True)
    info = (f"{frac:.0%} 的样本有重叠；重建出 {n_comp} 个互不重叠的连续片段，"
            f"最长 {lens[0]} 点，共 {sum(lens)} 点 ≈ {sum(lens) / 50000:.1f} s")
    return segments, info


def prep_jnu_sdalr(raw, L, fs_out):
    """读取 SDALR 仓库提供的 JNU_1d_2048_2000：data/<NO|IR|B|OR>/<类>_<转速>_<序号>.txt，每个文件 2048 点。
    先检测相邻样本的重叠并拼回连续信号，再按时间 70/30 重新切分，彻底避免重叠窗口泄漏。"""
    from scipy.signal import resample_poly
    order = [("NO", 0), ("IR", 1), ("B", 2), ("OR", 3)]  # 与 CWRU4 一致：Normal IR Ball OR
    for speed in (600, 800, 1000):
        Xtr, ytr, Xte, yte, str_, ste = [], [], [], [], [], []
        for folder, lab in order:
            files = glob.glob(os.path.join(raw, "data", folder, f"{folder}_{speed}_*.txt"))
            if not files:
                print(f"  未找到 {folder}_{speed}_*.txt，跳过")
                continue
            files.sort(key=lambda p: int(re.search(r"_(\d+)\.txt$", p).group(1)))
            sig, info = stitch_windows(files)
            if sig is not None:
                print(f"  {folder}_{speed}: {len(files)} 个窗口按时间顺序排列，步长 {info} 点，"
                      f"拼回连续信号 {len(sig)} 点 ≈ {len(sig) / 50000:.1f} s")
                segments = [sig]
            else:  # 顺序被打乱：按重叠关系重建时间轴
                W = np.stack([_load_txt(f) for f in files])
                segments, info = reconstruct_by_overlap(W)
                if segments is None:
                    raise SystemExit(f"[{folder}_{speed}] {info}。为避免泄漏已停止，请把这行信息发给我。")
                print(f"  {folder}_{speed}: 样本顺序被打乱，按重叠关系重建 —— {info}")
            segs = [resample_poly(s.astype(np.float32), fs_out, 50000).astype(np.float32) for s in segments]
            segs = [s for s in segs if len(s) >= L]
            trs, tes = [], []
            if len(segs) == 1:  # 单段连续信号：时间 70/30 切分
                a, b = time_split(segs[0], L)
                trs.append(a); tes.append(b)
            else:
                # 多个互不重叠的片段：整段分配给训练或测试（片段之间没有共享采样点，因此无泄漏）
                # 从短到长交替分配，使测试集约占总时长的 30%
                total = sum(len(s) for s in segs)
                te_len = 0
                for k, s in enumerate(sorted(segs, key=len)):
                    if te_len < 0.3 * total and k % 3 == 0:
                        tes.append(windows(s, L, L)); te_len += len(s)
                    else:
                        trs.append(windows(s, L, L // 2))
            trs = [t for t in trs if len(t)]; tes = [t for t in tes if len(t)]
            if not trs or not tes:
                raise SystemExit(f"[{folder}_{speed}] 重建出的连续片段太短，无法切出 {L} 点的训练/测试样本，请把输出发给我。")
            tr, te = np.concatenate(trs), np.concatenate(tes)
            tr = tr[np.linspace(0, len(tr) - 1, min(120, len(tr))).astype(int)]
            te = te[np.linspace(0, len(te) - 1, min(50, len(te))).astype(int)]
            Xtr.append(tr); ytr += [lab] * len(tr); str_ += [f"{folder}_{speed}[train]"] * len(tr)
            Xte.append(te); yte += [lab] * len(te); ste += [f"{folder}_{speed}[test]"] * len(te)
        if Xtr:
            save(f"jnu_{speed}", np.concatenate(Xtr), np.array(ytr), np.concatenate(Xte), np.array(yte),
                 str_, ste, dict(dataset="JNU", source="SDALR JNU_1d_2048_2000 (re-stitched)", speed=speed,
                                 fs=fs_out, L=L, classes=CWRU4, split="by_time_70_30"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["cwru", "pu", "jnu", "jnu_sdalr"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--L", type=int, default=None, help="窗长；默认 CWRU/JNU 1024，PU 2048")
    ap.add_argument("--classes", type=int, default=10, choices=[10, 4], help="CWRU 类别数")
    ap.add_argument("--pu_down", type=int, default=1, help="PU 降采样倍数（64k/down）")
    ap.add_argument("--pu_set", type=int, default=9, choices=[9, 15], help="PU 用 9 个还是 15 个轴承（15 → pu15_*）")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    OUT = a.out
    if a.dataset == "cwru":
        prep_cwru(a.raw, a.L or 1024, a.classes)
    elif a.dataset == "pu":
        prep_pu(a.raw, a.L or 2048, range(1, 15), range(15, 21), 20, 20, a.pu_down,
                PU_GROUPS15 if a.pu_set == 15 else PU_GROUPS, "pu15" if a.pu_set == 15 else "pu")
    elif a.dataset == "jnu":
        prep_jnu(a.raw, a.L or 1024, 12000)
    else:
        prep_jnu_sdalr(a.raw, a.L or 1024, 12000)
