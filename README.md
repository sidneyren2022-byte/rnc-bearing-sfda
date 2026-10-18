# RNC: Robust Neighborhood Consistency for Source-Free Cross-Condition Bearing Fault Diagnosis

Code for the paper

> Z. Ren, Y. Bao. *Robust source-free domain adaptation for cross-condition bearing fault diagnosis via neighborhood, acquisition-session and spectral consistency.* 

The repository contains the leakage-free data splits for PU, CWRU and JNU, re-implementations of the source-free baselines (SHOT, NRC, AaD) and the UDA references (DAN/MMD, Deep CORAL, DANN, CDAN), the proposed RNC, and every script used for the tables and figures in the paper. All methods share the same backbone (LiteWDCNN, 94.0K parameters), the same source models and the same splits.

```
├── data/
│   ├── download_cwru.py     download CWRU drive-end files
│   ├── download_pu.py       download Paderborn (PU) bearings (--set 15 for the 15 bearings used in the paper)
│   └── prepare.py           windowing + leakage-free splits -> data/processed/*.npz
├── src/common.py            backbone, augmentations, evaluation, per-bearing report
├── train_source.py          source model training + source-only results
├── sfda.py                  SHOT / NRC / AaD / RNC (source-free adaptation)
├── uda.py                   MMD / CORAL / DANN / CDAN (use source data; reference only)
├── run_all.py               batch runner with resume (finished runs are skipped)
├── summarize.py             result tables (mean ± std over seeds)
├── grid.sh                  all source-free methods under 4 learning-rate / epoch settings
├── final.sh                 UDA references, sensitivity, noise, t-SNE, efficiency
├── timing.sh                adaptation-time measurement
└── tools/
    ├── fig1_pipeline.py     Fig. 1 (method overview)
    ├── make_figures.py      Fig. 2 (per-bearing heatmap) and Fig. 3 (sensitivity)
    ├── tsne.py              Fig. 4 (t-SNE)
    ├── check_bearing.py     per-bearing accuracy table and bearing-level error count (Table 4)
    ├── jnu_compare.py       checks the pre-segmented JNU release against the official CSV files
    ├── timing_summary.py    summary of results/timing.csv
    ├── efficiency.py        parameters, FLOPs, CPU latency, ONNX export
    └── make_fake_data.py    synthetic data for a quick smoke test
```

Code comments are partly in Chinese; command-line options and outputs are self-explanatory.

## 1. Environment

Tested with Python 3.10, PyTorch 2.5.1 (CUDA 12.1) on a single NVIDIA RTX A2000 12 GB.

```bash
conda create -n rnc python=3.10 -y && conda activate rnc
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the build for your CUDA version
pip install -r requirements.txt
```

Smoke test with synthetic data (about 5 minutes; the numbers are meaningless):

```bash
python tools/make_fake_data.py
python run_all.py --dataset fake --stage source --seeds 0 --extra "--epochs 3"
python run_all.py --dataset fake --stage sfda --seeds 0 --sfda_methods shot nrc aad rnc --extra "--epochs 3"
rm -rf results ckpt data/processed/fake_*
```

## 2. Data

| Dataset | Source | Command |
|---|---|---|
| PU (15 bearings) | [KAt-DataCenter, Paderborn University](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter) | `python data/download_pu.py --out data/raw/PU --set 15`, then `python data/prepare.py pu --raw data/raw/PU --pu_set 15` |
| CWRU (drive end, 12 kHz) | [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) | `python data/download_cwru.py --out data/raw/CWRU`, then `python data/prepare.py cwru --raw data/raw/CWRU` |
| JNU | [JNU-Bearing-Dataset](https://github.com/ClarkGableWang/JNU-Bearing-Dataset) | see below |

PU bearings: healthy K001–K005, outer race KA04, KA15, KA16, KA22, KA30, inner race KI04, KI14, KI16, KI18, KI21. If the download script is slow, download the `.rar` files in a browser and extract each to `data/raw/PU/<bearing>/`.

Splits (fixed for all methods):

- **PU**: recordings 1–14 of every bearing and condition for training, 15–20 for testing (disjoint recordings).
- **CWRU / JNU**: first 70% of each signal for training, one window gap, last 30% for testing.

**JNU.** The paper uses the widely distributed pre-segmented release (`JNU_1d_2048_2000`, 2,048-point windows in shuffled order with about 88% overlap). `prepare.py jnu_sdalr` rebuilds the continuous signals by matching overlapping samples and then splits them by time:

```bash
python data/prepare.py jnu_sdalr --raw data/raw/JNU_sdalr    # layout: data/raw/JNU_sdalr/data/{NO,IR,B,OR}/*.txt
```

`tools/jnu_compare.py` verifies that every window of that release is an exact excerpt of the official CSV recordings (the reconstruction covers 99.98% of each recording). The official CSV files can also be used directly with `python data/prepare.py jnu --raw data/raw/JNU_csv`; window positions then differ slightly from the paper, so expect small numerical differences on JNU.

## 3. Reproducing the paper

Default setting: learning rate 1e-4, 30 epochs, seeds 0/1/2. RNC uses `--session 10 --freq_w 0.5` (K = 5).

```bash
# Source models + SHOT/NRC/AaD/RNC and its ablations under 4 settings (Tables 2-4)
bash grid.sh pu15  2>&1 | tee logs_grid_pu15.txt
bash grid.sh cwru  2>&1 | tee logs_grid_cwru.txt
bash grid.sh jnu   2>&1 | tee logs_grid_jnu.txt

# UDA references, sensitivity (Fig. 3), noise (Table 5), t-SNE (Fig. 4), efficiency (Section 6.8)
bash final.sh 2>&1 | tee logs_final.txt

# Adaptation time (Section 6.8); run with an otherwise idle GPU
bash timing.sh 2>&1 | tee logs_timing.txt
```

A single run:

```bash
python train_source.py --src pu15_2 --tgt pu15_0 pu15_1 pu15_3 --seed 0
python sfda.py --method rnc --src pu15_2 --tgt pu15_1 --seed 0 --lr 1e-4 --epochs 30 --session 10 --freq_w 0.5
```

Where the numbers come from:

| Paper item | Result file | Command |
|---|---|---|
| Table 2 (source-free) | `results/grid_<ds>_lr1e-4_ep30.csv` | `python summarize.py --csv results/grid_pu15_lr1e-4_ep30.csv` |
| Table 2 (source-only, UDA) | `results/results.csv` | `python summarize.py --dataset pu15` |
| Table 3 | `results/grid_pu15_lr*_ep*.csv` | `summarize.py --csv` on each file |
| Table 4, Fig. 2 | `results/grid_pu15_lr*_ep*_per_group.csv` | `python tools/check_bearing.py <file>` |
| Table 5 | `results/noise.csv` | `python summarize.py --csv results/noise.csv --noise` |
| Fig. 3 | `results/sens_pu15.csv` | `python tools/make_figures.py` |
| Fig. 1 | – | `python tools/fig1_pipeline.py` |

Runs are deterministic on the same hardware and software: repeating 36 runs for the timing measurement reproduced the original accuracies exactly.

Target labels are never used for training, early stopping or model selection; every method runs for a fixed number of epochs and the last model is evaluated.

Note: `sfda.py --method ours` is an earlier prototype that is kept for completeness and is not reported in the paper.

## 4. Citation

If you use this code, please cite the paper (details will be added after publication).

## License

MIT, see `LICENSE`.
