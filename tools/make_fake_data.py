"""
生成合成轴承信号（3 个“工况”域，4 类），只用于检查代码能否跑通，不能用于论文。

  python tools/make_fake_data.py
  python run_all.py --dataset fake --stage source --seeds 0 --extra "--epochs 3"
"""
import json
import os

import numpy as np

OUT = "data/processed"
FS, L = 12000, 1024
CLASSES = ["Normal", "IR", "Ball", "OR"]
FAULT_HZ = [0, 162, 141, 107]  # 示意性的故障特征频率


def signal(cls, rpm, gain, n, rng):
    t = np.arange(L) / FS
    x = np.zeros((n, L), np.float32)
    for i in range(n):
        s = gain * np.sin(2 * np.pi * rpm / 60 * t + rng.uniform(0, 6.28))
        if cls:
            f = FAULT_HZ[cls] * rpm / 1797
            imp = (np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) > 0.97).astype(float)
            s += 2.0 * gain * np.convolve(imp, np.exp(-np.arange(60) / 8) * np.sin(np.arange(60) * 1.3), "same")
        x[i] = s + rng.normal(0, 0.3 + 0.2 * cls / 3, L)
    return x


def main():
    rng = np.random.RandomState(0)
    os.makedirs(OUT, exist_ok=True)
    for d, (rpm, gain) in enumerate([(1797, 1.0), (1500, 1.6), (1200, 0.7)]):
        Xtr, ytr, Xte, yte = [], [], [], []
        for c in range(4):
            Xtr.append(signal(c, rpm, gain, 120, rng)); ytr += [c] * 120
            Xte.append(signal(c, rpm, gain, 50, rng)); yte += [c] * 50
        np.savez_compressed(f"{OUT}/fake_{d}.npz", X_train=np.concatenate(Xtr), y_train=np.array(ytr),
                            X_test=np.concatenate(Xte), y_test=np.array(yte),
                            src_train=np.array(["fake"] * len(ytr)), src_test=np.array(["fake"] * len(yte)))
        with open(f"{OUT}/fake_{d}.json", "w") as f:
            json.dump(dict(dataset="FAKE", fs=FS, L=L, classes=CLASSES, split="synthetic"), f)
        print(f"fake_{d}: rpm {rpm} gain {gain}")


if __name__ == "__main__":
    main()
