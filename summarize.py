"""
汇总结果：每个 任务 × 方法 计算各种子的 均值±标准差。不同数据集（cwru / pu / jnu-cwru4 跨设备…）分开出表，
各自计算平均值——不同数据集混在一起求平均没有意义。

  python summarize.py                              # 主对比表（准确率），每个数据集一张
  python summarize.py --metric f1                  # macro-F1 表
  python summarize.py --only_ours                  # 消融表（本文方法各变体）
  python summarize.py --csv results/noise.csv --noise   # 噪声鲁棒性表
"""
import argparse
import os

import pandas as pd

ORDER = ["source_only", "mmd", "coral", "dann", "cdan", "shot", "nrc", "aad", "rnc", "ours"]


def dataset_of(src, tgt):
    a, b = src.split("_")[0], tgt.split("_")[0]
    return a if a == b else f"{a}-{b}" if a < b else f"{b}-{a}"


def ours_col(note):
    return "ours" if note == "full" else f"ours[{note}]"


def col_name(m, n):
    if m == "ours":
        return ours_col(n)
    return m if n == "" else f"{m}[{n}]"


def make_table(df, metric, noise):
    g = df.groupby(["row", "col"])[metric].agg(["mean", "std", "count"]).reset_index()
    cell = g.assign(v=g["mean"].map("{:.2f}".format) + "±" + g["std"].fillna(0).map("{:.2f}".format))
    table = cell.pivot(index="row", columns="col", values="v")
    means = g.pivot(index="row", columns="col", values="mean")
    if not noise:
        # 平均值只在“所有方法都有结果”的任务上计算，保证公平
        full_rows = means.dropna().index
        table.loc["Average"] = means.loc[full_rows].mean(axis=0).map("{:.2f}".format)
        if len(full_rows) < len(means):
            print(f"  注意：{len(means) - len(full_rows)} 个任务缺少部分方法的结果，未计入 Average")
    cols = [c for c in ORDER if c in table.columns] + sorted(c for c in table.columns if c not in ORDER)
    table = table[cols]
    md = table.copy()
    for r in md.index:
        vals = {c: float(str(md.loc[r, c]).split("±")[0]) for c in cols if isinstance(md.loc[r, c], str)}
        if vals:
            best = max(vals, key=vals.get)
            md.loc[r, best] = f"**{md.loc[r, best]}**"
    return table, md, sorted(g["count"].unique().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--metric", default="acc", choices=["acc", "f1"])
    ap.add_argument("--only_ours", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--dataset", default=None, help="只看某个数据集，如 pu")
    ap.add_argument("--out", default="results/summary")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, keep_default_na=False)
    df[a.metric] = df[a.metric].astype(float)
    df["ds"] = [dataset_of(s, t) for s, t in zip(df.src, df.tgt)]
    if a.noise:
        df = df[df.note.str.contains("snr=")].copy()
        base = df.note.str.split("|").str[0]
        df["col"] = [col_name(m, n) for m, n in zip(df.method, base)]
        df["row"] = df.note.str.extract(r"snr=(-?\d+)")[0].astype(int)
    else:
        df = df[~df.note.str.contains("snr=")].copy()
        is_abl = df.note.str.startswith("-")
        if a.only_ours:
            df = df[df.method == "ours"].copy()
            df["col"] = "ours[" + df.note + "]"
        else:
            df = df[(df.method != "ours") | ~is_abl].copy()
            df["col"] = [col_name(m, n) for m, n in zip(df.method, df.note)]
        df["row"] = df.src + "→" + df.tgt

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ds in sorted(df.ds.unique()):
        if a.dataset and ds != a.dataset:
            continue
        table, md, cnt = make_table(df[df.ds == ds], a.metric, a.noise)
        suffix = ("_noise" if a.noise else "_ablation" if a.only_ours else "") + f"_{ds}_{a.metric}"
        table.to_csv(a.out + suffix + ".csv", encoding="utf-8-sig")
        with open(a.out + suffix + ".md", "w", encoding="utf-8") as f:
            f.write(md.to_markdown())
        print(f"\n### {ds}  ({a.metric}，每格种子数 {cnt})")
        print(md.to_markdown())
        print(f"-> {a.out + suffix}.csv/.md")


if __name__ == "__main__":
    main()
