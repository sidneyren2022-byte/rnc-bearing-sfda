#!/bin/bash
# 在任意数据集上：训练源模型（3 个种子），再在 4 组参数上跑基线和 RNC 的消融变体（各 3 个种子）。
# 每组参数一个结果文件：results/grid_<数据集>_lr<学习率>_ep<轮数>.csv。已完成的组合自动跳过，可断点续跑。
#   bash grid.sh cwru  2>&1 | tee logs_grid_cwru.txt
#   BASE="aad" bash grid.sh pu15 2>&1 | tee logs_grid_pu15.txt   # pu15 的 SHOT/NRC 已跑过，只补 AaD
set -e
ds=${1:?用法: bash grid.sh <数据集，如 cwru / cross / pu15>}
BASE=${BASE:-"shot nrc aad"}
python run_all.py --dataset $ds --stage source --seeds 0 1 2
for cfg in "1e-4 15" "1e-4 30" "5e-4 15" "5e-4 30"; do
  set -- $cfg
  out=results/grid_${ds}_lr$1_ep$2.csv
  python run_all.py --dataset $ds --stage sfda --seeds 0 1 2 --sfda_methods $BASE --extra "--lr $1 --epochs $2" --results $out
  # RNC 完整版 + 消融：只加会话(S10)、只加频带一致性(F0.5)
  python run_all.py --dataset $ds --stage sfda --seeds 0 1 2 --sfda_methods rnc --extra "--lr $1 --epochs $2 --session 10 --freq_w 0.5" --results $out
  python run_all.py --dataset $ds --stage sfda --seeds 0 1 2 --sfda_methods rnc --extra "--lr $1 --epochs $2 --session 10" --results $out
  python run_all.py --dataset $ds --stage sfda --seeds 0 1 2 --sfda_methods rnc --extra "--lr $1 --epochs $2 --freq_w 0.5" --results $out
done
for cfg in "1e-4 15" "1e-4 30" "5e-4 15" "5e-4 30"; do
  set -- $cfg
  echo; echo "########## $ds  lr=$1 epochs=$2 ##########"
  python summarize.py --csv results/grid_${ds}_lr$1_ep$2.csv
done
echo "########## Source-only ##########"
python summarize.py --dataset $([ "$ds" = cross ] && echo cwru4-jnu || echo $ds)
echo "$ds 全部完成"
