"""论文结果图（Fig. 2 逐轴承热力图、Fig. 3 敏感性）。用法：
  python tools/make_figures.py --grid_pg results/grid_pu15_lr5e-4_ep30_per_group.csv \
      --noise results/noise.csv --sens results/sens_pu15.csv --out figs
图 2 的超参数网格汇总数值直接写在下面（来自 summarize.py 的结果表）。"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 固定的类别色顺序（已做色盲校验），颜色跟着方法走
C = {"RNC (ours)": "#2a78d6", "SHOT": "#eb6834", "NRC": "#1baf7a", "AaD": "#eda100",
     "RNC N+S": "#e87ba4", "RNC N+F": "#008300", "Source-only": "#8a8984"}
MK = {"RNC (ours)": "o", "SHOT": "s", "NRC": "^", "AaD": "D", "RNC N+S": "v", "RNC N+F": "P", "Source-only": "x"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=600 if ext == "png" else None)
    plt.close(fig)


def end_labels(ax, xs, series, dy=None):
    """线末直接标注（避免只靠颜色区分）；dy 用于手动错开重叠的标签"""
    for k, ys in series.items():
        ax.annotate(k, (xs[-1], ys[-1]), xytext=(6, (dy or {}).get(k, 0)), textcoords="offset points",
                    va="center", fontsize=7, color=INK)


def fig_robustness(out):
    st = ["1e-4/15", "1e-4/30", "5e-4/15", "5e-4/30"]
    acc = {"Source-only": [76.19] * 4, "SHOT": [83.39, 83.36, 80.78, 80.03], "NRC": [89.25, 87.67, 59.19, 42.17],
           "AaD": [89.29, 86.66, 66.52, 49.38], "RNC (ours)": [92.13, 92.46, 90.24, 91.12]}
    err = {"Source-only": [126] * 4, "SHOT": [85, 85, 100, 106], "NRC": [49, 58, 215, 318],
           "AaD": [48, 62, 167, 269], "RNC (ours)": [35, 36, 47, 43]}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    x = np.arange(4)
    for ax, d, yl, t in [(axs[0], acc, "Mean target accuracy (%)", "(a) Accuracy"),
                         (axs[1], err, "Bearing-level errors (of 540)", "(b) Bearing-level errors")]:
        for k, ys in d.items():
            ax.plot(x, ys, color=C[k], marker=MK[k], ms=5, lw=2 if k != "Source-only" else 1.2,
                    ls="--" if k == "Source-only" else "-", mec="white", mew=0.8, label=k)
        ax.set_xticks(x, st); ax.set_xlabel("Learning rate / epochs"); ax.set_ylabel(yl)
        ax.set_title(t, loc="left", fontsize=8.5, color=INK); ax.set_xlim(-0.2, 3.9)
    end_labels(axs[0], x, acc, {"NRC": 0, "AaD": 0, "SHOT": 3, "Source-only": -2})
    axs[0].set_ylim(35, 96)
    end_labels(axs[1], x, err, {"NRC": 0, "AaD": 0})
    axs[1].legend(loc="upper left", fontsize=7, ncol=1)
    fig.tight_layout(); save(fig, out, "fig2_robustness")


def fig_noise(path, out):
    d = pd.read_csv(path); d["note"] = d["note"].fillna("")
    d["cfg"] = d["note"].str.split("|").str[0]
    d["snr"] = d["note"].str.extract(r"snr=(-?\d+)")[0].fillna("clean")
    name = {("shot", ""): "SHOT", ("nrc", ""): "NRC", ("aad", ""): "AaD", ("rnc", "S10"): "RNC N+S",
            ("rnc", "F0.5"): "RNC N+F", ("rnc", "S10_F0.5"): "RNC (ours)"}
    d["m"] = [name.get((a, b)) for a, b in zip(d["method"], d["cfg"])]
    piv = d.dropna(subset=["m"]).pivot_table(index="m", columns="snr", values="acc", aggfunc="mean")
    cols = ["-4", "0", "4", "8", "clean"]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = np.arange(len(cols))
    order = ["SHOT", "NRC", "AaD", "RNC N+S", "RNC N+F", "RNC (ours)"]
    for k in order:
        ax.plot(x, piv.loc[k, cols].values, color=C[k], marker=MK[k], ms=5, lw=2, mec="white", mew=0.8, label=k)
    ax.set_xticks(x, ["−4", "0", "4", "8", "Clean"]); ax.set_xlabel("Test SNR (dB)")
    ax.set_ylabel("Target accuracy (%)"); ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); save(fig, out, "fig4_noise")
    piv[cols].round(2).to_csv(os.path.join(out, "fig4_noise_table.csv"))


def fig_sens(path, out):
    d = pd.read_csv(path).groupby("note")["acc"].mean()
    panels = [("Session length S", ["5", "10", "20"], ["S5_F0.5", "S10_F0.5", "S20_F0.5"]),
              ("Spectral weight γ", ["0.25", "0.5", "1"], ["S10_F0.25", "S10_F0.5", "S10_F1"]),
              ("Neighbours K", ["3", "5", "10"], ["K3_S10_F0.5", "S10_F0.5", "K10_S10_F0.5"])]
    fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.0), sharey=True)
    for ax, (t, lab, keys) in zip(axs, panels):
        ys = [d[k] for k in keys]
        ax.plot(range(3), ys, color=C["RNC (ours)"], marker="o", ms=5, lw=2, mec="white", mew=0.8)
        ax.scatter([1], [ys[1]], s=60, facecolor="none", edgecolor=INK, lw=0.8, zorder=3)
        for i, y in enumerate(ys):
            ax.annotate(f"{y:.2f}", (i, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.5)
        ax.set_xticks(range(3), lab); ax.set_xlabel(t); ax.set_xlim(-0.4, 2.4)
    axs[0].set_ylim(89, 95); axs[0].set_ylabel("Mean accuracy (%)")
    fig.tight_layout(); save(fig, out, "fig3_sensitivity")


def fig_bearing(path, out):
    d = pd.read_csv(path); d["note"] = d["note"].fillna("")
    sel = {"AaD": ("aad", ""), "RNC (ours)": ("rnc", "S10_F0.5")}
    order = d.drop_duplicates("group").sort_values(["true", "group"])[["group", "true"]]
    mats = {}
    for k, (m, n) in sel.items():
        s = d[(d.method == m) & (d.note == n)]
        mats[k] = s.pivot_table(index="group", columns="tgt", values="acc", aggfunc="mean").reindex(order["group"])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 4.2), sharey=True)
    for ax, (k, M) in zip(axs, mats.items()):
        im = ax.imshow(M.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.grid(False)
        for (i, j), v in np.ndenumerate(M.values):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if v > 60 else INK)
        ax.set_xticks(range(M.shape[1]), [c.replace("pu15_", "P") for c in M.columns])
        ax.set_xlabel("Target condition"); ax.set_title(f"{k} (lr 5e-4, 30 epochs)", fontsize=8.5, color=INK)
        ax.set_xticks(np.arange(-.5, M.shape[1]), minor=True); ax.set_yticks(np.arange(-.5, M.shape[0]), minor=True)
        ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    axs[0].set_yticks(range(len(order)), [f"{g} ({t})" for g, t in order.values])
    cb = fig.colorbar(im, ax=axs, fraction=0.04, pad=0.02); cb.set_label("Bearing accuracy (%)")
    cb.outline.set_visible(False)
    save(fig, out, "fig2_bearing_heatmap")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_pg", default="results/grid_pu15_lr5e-4_ep30_per_group.csv")
    ap.add_argument("--noise", default="results/noise.csv")
    ap.add_argument("--sens", default="results/sens_pu15.csv")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--extra", action="store_true", help="另外画超参数鲁棒性和噪声曲线（论文未用）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # 论文用图：Fig. 2 逐轴承热力图，Fig. 3 敏感性。鲁棒性曲线和噪声曲线与表 3–5 重复，论文中未使用，需要时加 --extra
    fig_bearing(a.grid_pg, a.out); fig_sens(a.sens, a.out)
    if a.extra:
        fig_robustness(a.out); fig_noise(a.noise, a.out)
    print("figures written to", a.out)
