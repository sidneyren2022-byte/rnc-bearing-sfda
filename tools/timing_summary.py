"""汇总适应训练耗时：python tools/timing_summary.py results/timing.csv"""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/timing.csv")
df["per_epoch"] = df["adapt_sec"] / df["epochs"]
df["label"] = df["method"] + df["note"].fillna("").map(lambda n: f"[{n}]" if n else "")
g = df.groupby("label").agg(runs=("adapt_sec", "size"), total_mean=("adapt_sec", "mean"),
                            total_std=("adapt_sec", "std"), per_epoch=("per_epoch", "mean"))
base = g.loc[[i for i in g.index if i.startswith("aad")][0], "total_mean"] if any(i.startswith("aad") for i in g.index) else None
if base:
    g["vs_AaD"] = g["total_mean"] / base
print("设备:", ", ".join(df["device"].unique()), "| 目标样本数:", ", ".join(map(str, sorted(df["n_target"].unique()))))
print(g.round(2).to_string())
