"""
效率分析 + 边缘部署准备：参数量、FLOPs、CPU 单样本推理延迟，并导出 ONNX。

  python tools/efficiency.py --n_cls 3 --L 2048 --onnx deploy/model.onnx --ckpt ckpt/pu_0_w16_s0.pt

在树莓派上测延迟：把 model.onnx 拷过去，pip install onnxruntime，然后
  python tools/efficiency.py --onnx_only deploy/model.onnx --L 2048
STM32：用 STM32CubeMX 的 X-CUBE-AI 插件导入 model.onnx（可先 INT8 量化），
         它会给出 Flash/RAM 占用和单次推理周期数，直接写进论文的部署实验表。
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bench_onnx(path, L, runs=200):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 1, L).astype(np.float32)
    name = s.get_inputs()[0].name
    for _ in range(20):
        s.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(runs):
        s.run(None, {name: x})
    print(f"ONNXRuntime CPU 单样本延迟: {(time.perf_counter() - t) / runs * 1000:.3f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cls", type=int, default=10)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--onnx_only", default=None)
    a = ap.parse_args()
    if a.onnx_only:
        return bench_onnx(a.onnx_only, a.L)

    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from src.common import Net, n_params
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False) if a.ckpt else None
    m = Net(ck["n_cls"] if ck else a.n_cls, ck["width"] if ck else a.width).eval()  # 有权重时以权重为准
    if ck:
        m.load_state_dict(ck["state"])
    x = torch.randn(1, 1, a.L)
    with FlopCounterMode(display=False) as fc:
        m(x)
    print(f"参数量: {n_params(m) / 1e3:.1f} K   FLOPs: {fc.get_total_flops() / 1e6:.2f} M (乘加按 2 次计)")
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(20):
            m(x)
        t = time.perf_counter()
        for _ in range(200):
            m(x)
    print(f"PyTorch CPU 单线程单样本延迟: {(time.perf_counter() - t) / 200 * 1000:.3f} ms")
    if a.onnx:
        os.makedirs(os.path.dirname(a.onnx) or ".", exist_ok=True)
        kw = dict(input_names=["signal"], output_names=["logits"], opset_version=17)
        try:  # PyTorch >= 2.5 才有 dynamo 参数；旧版本不传
            torch.onnx.export(m, x, a.onnx, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(m, x, a.onnx, **kw)
        print("ONNX 已导出:", a.onnx)
        try:
            bench_onnx(a.onnx, a.L)
        except ImportError:
            print("未安装 onnxruntime，跳过 ONNX 延迟测试")


if __name__ == "__main__":
    main()
