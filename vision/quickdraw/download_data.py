"""
下载 QuickDraw .npy 数据文件
来源: Google Cloud Storage
每个类别一个 .npy 文件，28×28 灰度位图
并发下载，最大化带宽利用率

用法:
    python -m vision.quickdraw.download_data              # 默认 8 并发
    python -m vision.quickdraw.download_data --workers 4  # 4 并发
    python -m vision.quickdraw.download_data --retry      # 重试失败的
"""

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision.quickdraw.config import CATEGORIES, DATA_DIR, category_url

_print_lock = Lock()


def download_one(cat: str) -> tuple:
    """下载单个类别的 .npy 文件，返回 (cat, success, size_mb)"""
    filepath = os.path.join(DATA_DIR, f"{cat}.npy")
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        with _print_lock:
            print(f"  [SKIP] {cat}.npy 已存在 ({size_mb:.1f} MB)")
        return (cat, True, size_mb)

    url = category_url(cat)
    try:
        urllib.request.urlretrieve(url, filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        with _print_lock:
            print(f"  [OK] {cat}.npy ({size_mb:.1f} MB)")
        return (cat, True, size_mb)
    except Exception as e:
        with _print_lock:
            print(f"  [FAIL] {cat}: {e}")
        # 删除可能的不完整文件
        if os.path.exists(filepath):
            os.remove(filepath)
        return (cat, False, 0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QuickDraw 数据下载")
    parser.add_argument("--workers", type=int, default=8, help="并发下载数（默认 8）")
    parser.add_argument("--retry", action="store_true", help="仅重试之前失败的类别")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # 确定要下载的类别
    if args.retry:
        pending = [c for c in CATEGORIES if not os.path.exists(os.path.join(DATA_DIR, f"{c}.npy"))]
    else:
        pending = CATEGORIES

    print(f"=" * 50)
    print(f"QuickDraw 数据下载")
    print(f"目标目录: {DATA_DIR}")
    print(f"待下载: {len(pending)}/{len(CATEGORIES)} 个类别")
    print(f"并发数: {args.workers}")
    print(f"=" * 50)

    success = 0
    failed = []
    total_size = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, cat): cat for cat in pending}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            cat, ok, size_mb = future.result()
            if ok:
                success += 1
                total_size += size_mb
            else:
                failed.append(cat)
            # 每 10 个输出一次总进度
            if done_count % 10 == 0 or done_count == len(pending):
                elapsed = time.time() - start
                speed = total_size / elapsed if elapsed > 0 else 0
                print(f"  [{done_count}/{len(pending)}] 进度: {done_count/len(pending)*100:.0f}% | "
                      f"已下载: {total_size:.0f} MB | 速度: {speed:.2f} MB/s | 耗时: {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"下载完成: {success}/{len(pending)} 成功, 耗时 {elapsed:.0f}s")
    print(f"数据总量: {total_size:.0f} MB, 平均速度: {total_size/elapsed if elapsed > 0 else 0:.2f} MB/s")

    if failed:
        print(f"失败类别 ({len(failed)}):")
        for cat in failed:
            print(f"  - {cat}")
        print(f"\n重试: python -m vision.quickdraw.download_data --retry")


if __name__ == "__main__":
    main()
