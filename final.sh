#!/bin/bash
# 论文剩余实验（主配置统一为 lr=1e-4, epochs=30），全部可断点续跑。约 6–8 小时。
#   bash final.sh 2>&1 | tee logs_final.txt
set -e
LR="--lr 1e-4 --epochs 30"

# 1) 有源数据的经典 UDA（参考上界，需要源数据，不与无源方法直接比较）
for ds in pu15 cwru jnu; do
  python run_all.py --dataset $ds --stage uda --seeds 0 1 2
done

# 2) RNC 超参数敏感性（pu15）：会话长度 S、频带一致性权重、近邻数 K，各改一个，其余保持默认（S=10, F=0.5, K=5）
out=results/sens_pu15.csv
for v in "--session 5 --freq_w 0.5" "--session 20 --freq_w 0.5" \
         "--session 10 --freq_w 0.25" "--session 10 --freq_w 1" \
         "--session 10 --freq_w 0.5 --K 3" "--session 10 --freq_w 0.5 --K 10" \
         "--session 10 --freq_w 0.5"; do
  python run_all.py --dataset pu15 --stage sfda --seeds 0 1 2 --sfda_methods rnc --extra "$LR $v" --results $out
done

# 3) 噪声鲁棒性（pu15 的 3 个任务，SNR -4/0/4/8 dB）
python run_all.py --dataset pu15 --stage noise --seeds 0 1 2 --sfda_methods shot nrc aad rnc \
       --extra "$LR --session 10 --freq_w 0.5"
# RNC 消融变体的噪声测试（Table 5 的 N+S、N+F 两行）
python run_all.py --dataset pu15 --stage noise --seeds 0 1 2 --sfda_methods rnc --extra "$LR --session 10"
python run_all.py --dataset pu15 --stage noise --seeds 0 1 2 --sfda_methods rnc --extra "$LR --freq_w 0.5"

# 4) t-SNE 特征（论文 Fig. 4：P2 -> P1，激进设置 lr=5e-4、30 轮，种子 0）
python sfda.py --method rnc --src pu15_2 --tgt pu15_1 --seed 0 --lr 5e-4 --epochs 30 --session 10 --freq_w 0.5 --save_feats --results results/tsne_runs.csv
python sfda.py --method nrc --src pu15_2 --tgt pu15_1 --seed 0 --lr 5e-4 --epochs 30 --save_feats --results results/tsne_runs.csv
python tools/tsne.py --prefix results/feats/rnc_pu15_2_pu15_1_s0 --classes Healthy OR IR
python tools/tsne.py --prefix results/feats/nrc_pu15_2_pu15_1_s0 --classes Healthy OR IR

# 5) 效率（参数量、FLOPs、延迟）与 ONNX 导出
python tools/efficiency.py --ckpt ckpt/pu15_0_w16_s0.pt --L 2048 --onnx deploy/rnc_pu15.onnx

echo; echo "########## 汇总 ##########"
for ds in pu15 cwru jnu; do python summarize.py --dataset $ds; done
python summarize.py --csv results/sens_pu15.csv --dataset pu15
python summarize.py --csv results/noise.csv --noise
echo "final 全部完成"
