#!/bin/bash
# 适应训练耗时测量（pu15 的 3 个任务 × 3 个种子 × 4 种方法 = 36 次，主配置 lr=1e-4, 30 轮）。
# 运行时 GPU 上不要有其他任务，否则计时偏大。约 30–60 分钟。
#   bash timing.sh 2>&1 | tee logs_timing.txt
set -e
LR="--lr 1e-4 --epochs 30"
OUT=results/timing_runs.csv     # 精度结果单独写，不污染主结果
T=results/timing.csv            # 耗时记录
for seed in 0 1 2; do
  for pair in "pu15_0 pu15_1" "pu15_1 pu15_0" "pu15_2 pu15_3"; do
    set -- $pair
    python sfda.py --method shot --src $1 --tgt $2 --seed $seed $LR --results $OUT --timing $T
    python sfda.py --method nrc  --src $1 --tgt $2 --seed $seed $LR --results $OUT --timing $T
    python sfda.py --method aad  --src $1 --tgt $2 --seed $seed $LR --results $OUT --timing $T
    python sfda.py --method rnc  --src $1 --tgt $2 --seed $seed $LR --session 10 --freq_w 0.5 --results $OUT --timing $T
  done
done
python tools/timing_summary.py $T
