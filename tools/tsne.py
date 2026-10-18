"""
画适应前后的 t-SNE 对比图（先运行 sfda.py ... --save_feats）。

  python tools/tsne.py --prefix results/feats/ours_pu_0_pu_1_s0 --classes Healthy OR IR
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True)
ap.add_argument("--classes", nargs="*", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, stage, title in zip(axes, ["before", "after"], ["(a) Before adaptation", "(b) After adaptation"]):
    d = np.load(f"{a.prefix}_{stage}.npz")
    z = TSNE(2, init="pca", random_state=0, perplexity=30).fit_transform(d["feat"])
    for c in np.unique(d["y"]):
        m = d["y"] == c
        ax.scatter(z[m, 0], z[m, 1], s=6, label=a.classes[c] if a.classes else str(c))
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
axes[1].legend(markerscale=3, fontsize=8, loc="best")
plt.tight_layout()
out = a.out or a.prefix + "_tsne.png"
plt.savefig(out, dpi=300)
print("saved", out)
