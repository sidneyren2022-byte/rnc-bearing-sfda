"""核对 SDALR 的 JNU txt 窗口是否就是官方 CSV 原始信号的切片。
  python tools/jnu_compare.py --csv_dir data/raw/JNU_csv --txt_dir data/raw/JNU_sdalr
输出每个类别/转速：CSV 长度、txt 窗口在 CSV 中能精确找到的比例、覆盖的时间范围。"""
import argparse
import glob
import os
import re

import numpy as np

PAIRS = [("n", "NO"), ("ib", "IR"), ("tb", "B"), ("ob", "OR")]


def locate(sig, w, tol):
    cand = np.where(np.abs(sig[: len(sig) - len(w) + 1] - w[0]) <= tol)[0]
    for c in cand:
        if np.allclose(sig[c:c + len(w)], w, atol=tol, rtol=0):
            return c
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--txt_dir", required=True, help="SDALR 的 JNU 目录（其下有 data/NO, data/IR ...）")
    ap.add_argument("--n_check", type=int, default=200, help="每类抽查多少个 txt 窗口")
    ap.add_argument("--tol", type=float, default=1e-4, help="数值容差（txt 与 csv 的小数位数可能不同）")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for speed in (600, 800, 1000):
        for cp, tp in PAIRS:
            cf = [p for p in glob.glob(os.path.join(a.csv_dir, "*.csv"))
                  if re.match(rf"^{cp}{speed}[_\.]", os.path.basename(p))]
            tf = glob.glob(os.path.join(a.txt_dir, "data", tp, f"{tp}_{speed}_*.txt"))
            if not cf or not tf:
                print(f"{tp}_{speed}: 缺少文件（csv {len(cf)}，txt {len(tf)}）"); continue
            sig = np.loadtxt(cf[0], delimiter=",", dtype=np.float64).ravel()
            pick = rng.choice(len(tf), size=min(a.n_check, len(tf)), replace=False)
            pos = np.array([locate(sig, np.loadtxt(tf[i]), a.tol) for i in pick])
            ok = pos >= 0
            span = (pos[ok].min(), pos[ok].max() + 2048) if ok.any() else (0, 0)
            print(f"{tp}_{speed}: CSV {len(sig)} 点 = {len(sig) / 50000:.1f} s；txt 窗口精确命中 "
                  f"{ok.mean() * 100:.0f}%；命中窗口覆盖 {span[0] / 50000:.2f}–{span[1] / 50000:.2f} s")


if __name__ == "__main__":
    main()
