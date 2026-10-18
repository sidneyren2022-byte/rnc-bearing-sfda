"""核对逐轴承热力图（论文 Fig. 2）和轴承级错误数（Table 4）。
  python tools/check_bearing.py results/grid_pu15_lr5e-4_ep30_per_group.csv
打印：每个方法在每个测试轴承 × 目标工况上的平均准确率（= 热力图格子里的数），
以及准确率 < 50% 的轴承实例数（= Table 4 对应列）。"""
import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "results/grid_pu15_lr5e-4_ep30_per_group.csv"
d = pd.read_csv(path)
d["note"] = d["note"].fillna("")
d["name"] = d["method"] + d["note"].map(lambda n: f"[{n}]" if n else "")
order = d.drop_duplicates("group").sort_values(["true", "group"])
for name, g in d.groupby("name"):
    piv = g.pivot_table(index="group", columns="tgt", values="acc", aggfunc="mean").reindex(order["group"])
    piv.columns = [c.replace("pu15_", "P") for c in piv.columns]
    piv.insert(0, "class", order.set_index("group")["true"])
    print(f"\n===== {name}：每格 = 3 个源工况 × 3 个种子的平均准确率（%） =====")
    print(piv.round(0).to_string())
    print(f"轴承实例总数 {len(g)}，其中准确率 < 50% 的 {int((g['acc'] < 50).sum())} 个（对应 Table 4）")
