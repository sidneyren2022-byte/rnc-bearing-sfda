"""
下载 Paderborn (PU) 轴承数据。每个轴承一个 .rar（约 150-180 MB），解压后每个工况 20 段录音。

默认只下载 3 类实验需要的 9 个轴承（约 1.4 GB）：
    健康:          K001 K002 K003
    外圈真实损伤:   KA04 KA15 KA16
    内圈真实损伤:   KI14 KI16 KI18

用法：
    python data/download_pu.py --out data/raw/PU
    python data/download_pu.py --out data/raw/PU --bearings K001 KA04 KI04   # 自定义

解压：
    脚本会尝试用 rarfile 库（需系统装 unrar 或 7-Zip）自动解压；
    失败时请手动用 7-Zip / WinRAR 把每个 rar 解压到 data/raw/PU/<轴承名>/ 下。
数据许可：CC BY-NC 4.0，论文中需引用 Lessmeier et al., PHM Europe 2016。
"""
import argparse
import os
import time
import urllib.request

BASE = "https://groups.uni-paderborn.de/kat/BearingDataCenter/{b}.rar"
DEFAULT = ["K001", "K002", "K003", "KA04", "KA15", "KA16", "KI14", "KI16", "KI18"]
EXTRA15 = ["K004", "K005", "KA22", "KA30", "KI04", "KI21"]


def download(b, out_dir, retries=3):
    rar = os.path.join(out_dir, f"{b}.rar")
    if os.path.exists(rar) and os.path.getsize(rar) > 50_000_000:
        print(f"[skip] {b}.rar 已存在")
        return rar
    url = BASE.format(b=b)
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(rar + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(rar + ".part", rar)
            print(f"[ok]   {b}.rar")
            return rar
        except Exception as e:  # noqa: BLE001
            print(f"[retry {k + 1}] {url}: {e}")
            time.sleep(3)
    print(f"[fail] {b}.rar，请浏览器手动下载 {url}")
    return None


def extract(rar, out_dir):
    target = os.path.join(out_dir, os.path.basename(rar)[:-4])
    if os.path.isdir(target) and len(os.listdir(target)) > 10:
        return True
    try:
        import rarfile
        with rarfile.RarFile(rar) as rf:
            rf.extractall(out_dir)
        print(f"[unrar] {rar}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[manual] 无法自动解压 {rar}（{e}），请手动解压到 {out_dir}")
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/PU")
    ap.add_argument("--bearings", nargs="+", default=None)
    ap.add_argument("--set", type=int, default=9, choices=[9, 15], help="15：另加 K004 K005 KA22 KA30 KI04 KI21")
    ap.add_argument("--no_extract", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    bearings = args.bearings or (DEFAULT + (EXTRA15 if args.set == 15 else []))
    for b in bearings:
        rar = download(b, args.out)
        if rar and not args.no_extract:
            extract(rar, args.out)
