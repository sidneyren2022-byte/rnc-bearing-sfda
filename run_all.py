"""
批量跑实验（可断点续跑：结果 CSV 里已有的组合会自动跳过）。Windows / Linux 通用。

  # 1. 训练所有源模型 + Source-only（4 个源域 × 5 个种子）
  python run_all.py --dataset pu --stage source
  # 2. 经典 UDA 基线（12 任务 × 4 方法 × 5 种子）
  python run_all.py --dataset pu --stage uda
  # 3. 无源方法：SHOT / NRC / 本文
  python run_all.py --dataset pu --stage sfda
  # 4. 消融实验
  python run_all.py --dataset pu --stage ablation
  # 5. 噪声鲁棒性
  python run_all.py --dataset pu --stage noise
  # 只看命令不执行
  python run_all.py --dataset pu --stage sfda --dry
"""
import argparse
import csv
import itertools
import os
import subprocess
import sys

DOMAINS = {
    "cwru": [f"cwru_{i}" for i in range(4)],
    "cwru4": [f"cwru4_{i}" for i in range(4)],
    "pu": [f"pu_{i}" for i in range(4)],
    "pu15": [f"pu15_{i}" for i in range(4)],
    "jnu": ["jnu_600", "jnu_800", "jnu_1000"],
    "fake": [f"fake_{i}" for i in range(3)],
}
CROSS = {  # 跨设备任务：CWRU(4 类) <-> JNU
    "cross": [("cwru4_0", "jnu_1000"), ("cwru4_1", "jnu_800"), ("jnu_1000", "cwru4_0"), ("jnu_600", "cwru4_2")],
}


def done_set(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["method"], r["src"], r["tgt"], r["seed"], r["note"]) for r in csv.DictReader(f)}


def run(cmd, dry):
    print(">>", " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DOMAINS) + list(CROSS))
    ap.add_argument("--stage", required=True, choices=["source", "uda", "sfda", "ablation", "noise"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--uda_methods", nargs="+", default=["mmd", "coral", "dann", "cdan"])
    ap.add_argument("--sfda_methods", nargs="+", default=["shot", "nrc", "ours"])
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--extra", default="", help="透传给子脚本的额外参数，如 \"--epochs 5\"")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    extra = a.extra.split() if a.extra else []

    if a.dataset in CROSS:
        pairs = CROSS[a.dataset]
        srcs = sorted({s for s, _ in pairs})
    else:
        doms = DOMAINS[a.dataset]
        pairs = [(s, t) for s, t in itertools.permutations(doms, 2)]
        srcs = doms
    done = done_set(a.results)
    R = ["--results", a.results]
    # 本文方法的结果标签要与 sfda.py 生成的一致（多原型时带 _mpK 后缀），否则断点续跑会误判
    mp = ""
    if "--n_proto" in extra:
        k = int(extra[extra.index("--n_proto") + 1])
        mp = f"_mp{k}" if k > 1 else ""
    # RNC 的标签规则与 sfda.py 的 rnc_tag 一致
    def _val(flag, default):
        return float(extra[extra.index(flag) + 1]) if flag in extra else default
    rnc_parts = []
    if _val("--K", 5) != 5:
        rnc_parts.append(f"K{int(_val('--K', 5))}")
    if _val("--session", 0) > 1:
        rnc_parts.append(f"S{int(_val('--session', 0))}")
    if _val("--freq_w", 0) > 0:
        rnc_parts.append(f"F{_val('--freq_w', 0):g}")
    rnc_note = "_".join(rnc_parts)
    if "--anchor" in extra:
        v = float(extra[extra.index("--anchor") + 1])
        mp += f"_anc{v:g}" if v > 0 else ""

    for seed in a.seeds:
        if a.stage == "source":
            for s in srcs:
                tg = [t for ss, t in pairs if ss == s]
                if all(("source_only", s, t, str(seed), "") in done for t in tg):
                    continue
                run([py, "train_source.py", "--src", s, "--tgt", *tg, "--seed", str(seed), *R, *extra], a.dry)
        elif a.stage == "uda":
            for (s, t), m in itertools.product(pairs, a.uda_methods):
                if (m, s, t, str(seed), "") not in done:
                    run([py, "uda.py", "--method", m, "--src", s, "--tgt", t, "--seed", str(seed), *R, *extra], a.dry)
        elif a.stage == "sfda":
            for (s, t), m in itertools.product(pairs, a.sfda_methods):
                note = ("full" + mp) if m == "ours" else (rnc_note if m == "rnc" else "")
                if (m, s, t, str(seed), note) not in done:
                    run([py, "sfda.py", "--method", m, "--src", s, "--tgt", t, "--seed", str(seed), *R, *extra],
                        a.dry)
        elif a.stage == "ablation":
            variants = {"-vote": ["--no_vote"], "-proto": ["--no_proto"], "-freq": ["--no_freq"],
                        "-im": ["--no_im"], "-proto-freq": ["--no_proto", "--no_freq"]}
            for (s, t), (note, flags) in itertools.product(pairs, variants.items()):
                if ("ours", s, t, str(seed), note + mp) not in done:
                    run([py, "sfda.py", "--method", "ours", "--src", s, "--tgt", t, "--seed", str(seed), *flags,
                         *R, *extra], a.dry)
        elif a.stage == "noise":
            done = done_set("results/noise.csv")  # 噪声实验写在单独文件里
            for (s, t), m in itertools.product(pairs[:3], a.sfda_methods):  # 选 3 个代表任务即可
                note = (("full" + mp) if m == "ours" else (rnc_note if m == "rnc" else "")) + "|snr=-4"
                if (m, s, t, str(seed), note) not in done:
                    run([py, "sfda.py", "--method", m, "--src", s, "--tgt", t, "--seed", str(seed),
                         "--test_snr", "-4", "0", "4", "8", "--results", "results/noise.csv", *extra], a.dry)


if __name__ == "__main__":
    main()
