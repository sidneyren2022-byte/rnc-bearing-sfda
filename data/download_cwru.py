"""
下载 CWRU 12k 驱动端轴承数据（10 类 × 4 种负载 = 40 个 .mat 文件，约 160 MB）。

用法：
    python data/download_cwru.py --out data/raw/CWRU

文件编号与 UDTL 框架 (github.com/ZhaoZhibin/UDTL) 一致：
    类别顺序：正常, IR007, B007, OR007@6, IR014, B014, OR014@6, IR021, B021, OR021@6
"""
import argparse
import os
import time
import urllib.request

# 负载(HP) -> 10 个文件编号，顺序即类别标签 0..9
CWRU_FILES = {
    0: [97, 105, 118, 130, 169, 185, 197, 209, 222, 234],   # 1797 rpm
    1: [98, 106, 119, 131, 170, 186, 198, 210, 223, 235],   # 1772 rpm
    2: [99, 107, 120, 132, 171, 187, 199, 211, 224, 236],   # 1750 rpm
    3: [100, 108, 121, 133, 172, 188, 200, 212, 225, 237],  # 1730 rpm
}
CLASS_NAMES = ["Normal", "IR007", "B007", "OR007", "IR014", "B014",
               "OR014", "IR021", "B021", "OR021"]

# 官网当前地址在前，旧地址作为备用
URL_PATTERNS = [
    "https://engineering.case.edu/sites/default/files/{n}.mat",
    "http://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/{n}.mat",
]


def download(n, out_dir, retries=3):
    path = os.path.join(out_dir, f"{n}.mat")
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        print(f"[skip] {n}.mat 已存在")
        return True
    for pattern in URL_PATTERNS:
        url = pattern.format(n=n)
        for k in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as f:
                    f.write(r.read())
                os.replace(path + ".part", path)
                print(f"[ok]   {n}.mat  <- {url}")
                return True
            except Exception as e:  # noqa: BLE001
                print(f"[retry {k + 1}] {url}: {e}")
                time.sleep(2)
    print(f"[fail] {n}.mat，请手动从 https://engineering.case.edu/bearingdatacenter/download-data-file 下载")
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/CWRU")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ok = all(download(n, args.out) for files in CWRU_FILES.values() for n in files)
    print("全部完成" if ok else "有文件下载失败，见上方 [fail]")
