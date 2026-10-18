"""Fig. 1：RNC 流程图（矢量 PDF + 600 dpi PNG）。python tools/fig1_pipeline.py --out figs"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, LINE = "#0b0b0b", "#52514e", "#8a8984"
BLUE, BLUE_T = "#2a78d6", "#e8f1fc"
GREY_T, ORANGE_T, AQUA_T, MAG_T = "#f3f2ef", "#fdebe3", "#e2f5ee", "#fbe9f0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42})


def box(ax, x, y, w, h, text, fc=GREY_T, ec=LINE, bold=False, fs=7.5, lw=0.9, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, ls="-", color=INK2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def main(out):
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_xlim(0, 18); ax.set_ylim(1.15, 10.2); ax.axis("off")
    for x, w, t in [(0.1, 3.6, "(1) Source side"), (3.9, 11.0, "(2) Source-free target adaptation"),
                    (15.1, 2.8, "(3) Deployment")]:
        ax.add_patch(FancyBboxPatch((x, 1.35), w, 8.45, boxstyle="round,pad=0,rounding_size=0.2", fc="none",
                                    ec="#c3c2b7", lw=0.8, ls=(0, (4, 3))))
        ax.text(x + 0.2, 9.45, t, ha="left", va="center", fontsize=8, color=INK, fontweight="bold")
    R1, R2, R3 = (7.3, 1.6), (4.5, 1.8), (1.9, 1.7)   # 三行的 (y, 高)

    # (1) 源端
    box(ax, 0.4, R1[0], 2.9, R1[1], "Labelled source\nrecordings $(x^s, y^s)$")
    box(ax, 0.4, R2[0], 2.9, R2[1], "Supervised training\nLiteWDCNN\n$g$ + $h$")
    box(ax, 0.4, R3[0], 2.9, R3[1], "Released model\n$h\\circ g$ (source\ndata not shared)", fs=7, fc=BLUE_T, ec=BLUE)
    arrow(ax, (1.85, R1[0]), (1.85, R2[0] + R2[1])); arrow(ax, (1.85, R2[0]), (1.85, R3[0] + R3[1]))

    # (2) 第 1 行：输入 → 两种视图 → g → h
    box(ax, 4.2, R1[0], 2.4, R1[1], "Unlabelled\ntarget windows\n+ recording ID")
    box(ax, 7.0, 8.15, 2.5, 0.75, "Weak view")
    box(ax, 7.0, 7.3, 2.5, 0.75, "Strong view\n(band mask + noise)", fs=6.5)
    box(ax, 9.9, R1[0], 1.7, R1[1], "Feature\nextractor $g$\n(trained)", fc=BLUE_T, ec=BLUE)
    box(ax, 12.0, R1[0], 1.6, R1[1], "Classifier\n$h$\n(frozen)", fc="#ffffff", ec=LINE, ls=(0, (3, 2)))
    arrow(ax, (6.6, 8.35), (7.0, 8.52)); arrow(ax, (6.6, 7.85), (7.0, 7.68))
    arrow(ax, (9.5, 8.52), (9.9, 8.35)); arrow(ax, (9.5, 7.68), (9.9, 7.85))
    arrow(ax, (11.6, 8.1), (12.0, 8.1))

    # 输出总线：h 的输出（特征 + 弱/强视图预测）送到记忆库和三个损失
    bus = 6.85
    ax.plot([12.8, 12.8], [R1[0], bus], color=INK2, lw=0.9)
    ax.plot([5.4, 13.5], [bus, bus], color=INK2, lw=0.9)
    ax.text(10.9, bus + 0.12, "features $f$, predictions $p_w$, $p_s$", fontsize=6.3, color=INK2, ha="center", va="bottom")

    # 第 2 行：三个损失
    L = [(7.0, "Neighbourhood\nterm $\\mathcal{L}_{\\mathrm{nb}}$\nattract $K$-NN,\ndisperse the rest", AQUA_T, "#1baf7a"),
         (9.55, "Session term $\\mathcal{L}_{\\mathrm{sess}}$\n$S$ consecutive\nwindows of one\nrecording agree", MAG_T, "#e87ba4"),
         (12.1, "Spectral term $\\mathcal{L}_{\\mathrm{freq}}$\nKL($p_w \\Vert p_s$)\nbetween weak and\nstrong views", ORANGE_T, "#eb6834")]
    for x, t, fc, ec in L:
        box(ax, x, R2[0], 2.35, R2[1], t, fc=fc, ec=ec, fs=6.4)
        arrow(ax, (x + 1.175, bus), (x + 1.175, R2[0] + R2[1]))
        arrow(ax, (x + 1.175, R2[0]), (x + 1.175, R3[0] + R3[1]))

    # 第 3 行：记忆库 + 总损失
    box(ax, 4.2, R3[0], 2.4, R3[1], "Memory banks\n$\\mathcal{F}$ (features)\n$\\mathcal{P}$ (predictions)")
    arrow(ax, (5.4, bus), (5.4, R3[0] + R3[1]))
    ax.text(5.3, 5.4, "refresh\neach step", fontsize=6.2, color=INK2, ha="right", va="center")
    # 记忆库 → 邻域项：直角折线箭头（先向上，再向右进入 L_nb 左侧中点）
    ax.plot([6.3, 6.3, 6.75], [R3[0] + R3[1], R2[0] + R2[1] / 2, R2[0] + R2[1] / 2], color=INK2, lw=0.9)
    arrow(ax, (6.7, R2[0] + R2[1] / 2), (7.0, R2[0] + R2[1] / 2))
    ax.text(5.5, 4.25, "$K$-NN,\npeers", fontsize=6.2, color=INK2, ha="left", va="center")
    box(ax, 7.0, R3[0], 7.45, R3[1], "$\\mathcal{L} = \\mathcal{L}_{\\mathrm{nb}} + w_{\\mathrm{sess}}\\,\\mathcal{L}_{\\mathrm{sess}}"
        " + \\gamma\\,\\mathcal{L}_{\\mathrm{freq}}$\nback-propagate to $g$ only; fixed number of epochs;\nno target labels used",
        fc=BLUE_T, ec=BLUE, fs=7.2)
    arrow(ax, (3.3, 2.75), (4.2, 2.75))
    ax.text(3.75, 2.95, "init.", fontsize=6.2, color=INK2, ha="center", va="bottom")

    # (3) 部署
    box(ax, 15.35, R1[0], 2.3, R1[1], "Adapted model\n$h\\circ g$")
    box(ax, 15.35, R2[0], 2.3, R2[1], "ONNX export\n94.0K parameters\n3.37 MFLOPs", fs=7)
    box(ax, 15.35, R3[0], 2.3, R3[1], "Edge inference\n0.11 ms / window", fs=7)
    arrow(ax, (13.6, 8.1), (15.35, 8.1))
    ax.text(14.25, 8.3, "after\nadaptation", fontsize=5.8, color=INK2, ha="center", va="bottom")
    arrow(ax, (16.5, R1[0]), (16.5, R2[0] + R2[1])); arrow(ax, (16.5, R2[0]), (16.5, R3[0] + R3[1]))

    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig1_pipeline.{ext}"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figs"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); main(a.out)
