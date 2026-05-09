"""
QuickDraw 数据集加载
.npy 文件：shape (N, 784)，值 0-255 uint8，28×28 灰度位图
使用内存映射避免一次性加载全部数据到 RAM
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DATA_DIR, CATEGORIES, SAMPLES_PER_CLASS, CATEGORY_TO_IDX


class QuickDrawDataset(Dataset):
    """内存映射方式加载，按需读取样本，避免 OOM"""

    def __init__(self, train: bool = True):
        self.samples = []  # [(filepath, offset_in_file), ...]
        self.labels = []   # [class_idx, ...]
        self._file_cache = {}  # path -> mmap array

        rng = np.random.RandomState(42)

        for cat in CATEGORIES:
            filepath = f"{DATA_DIR}/{cat}.npy"
            try:
                fp = np.load(filepath, mmap_mode='r')
            except FileNotFoundError:
                print(f"  [WARN] 数据文件缺失: {filepath}，跳过")
                continue

            n_total = fp.shape[0]
            n_use = min(n_total, SAMPLES_PER_CLASS)

            # 随机选取 n_use 个索引（不用 randint 因为要确定性，用 arange + shuffle）
            indices = np.arange(n_total)
            rng.shuffle(indices)
            indices = indices[:n_use]
            indices.sort()  # 排序以利用顺序读取

            # 划分 train/val（85/15，按索引位置）
            split = int(n_use * 0.85)
            if train:
                indices = indices[:split]
            else:
                indices = indices[split:]

            label = CATEGORY_TO_IDX[cat]
            for idx in indices:
                self.samples.append((filepath, int(idx)))
                self.labels.append(label)

        # 再次打乱样本顺序
        idx = rng.permutation(len(self.samples))
        self.samples = [self.samples[i] for i in idx]
        self.labels = [self.labels[i] for i in idx]

        print(f"  {'训练' if train else '验证'}集: {len(self.samples)} 样本, "
              f"{len(set(self.labels))} 类")

    def _get_array(self, filepath: str):
        """懒加载内存映射数组"""
        if filepath not in self._file_cache:
            self._file_cache[filepath] = np.load(filepath, mmap_mode='r')
        return self._file_cache[filepath]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, offset = self.samples[idx]
        arr = self._get_array(filepath)
        # uint8 → float32 [0,1]，reshape 为 (1, 28, 28)
        img = arr[offset].astype(np.float32) / 255.0
        img = img.reshape(1, 28, 28)
        label = self.labels[idx]
        return torch.from_numpy(img.copy()), label  # .copy() 确保返回连续 tensor
