"""
QuickDraw CNN 分类模型
输入: (B, 1, 28, 28) 灰度位图
输出: (B, NUM_CLASSES) logits
"""

import torch
import torch.nn as nn

from .config import NUM_CLASSES


class QuickDrawCNN(nn.Module):
    """轻量 2 层 CNN，适合 28×28 草图分类"""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28×28 → 14×14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14×14 → 5×5
        )

        # 64 通道 × 4×4 = 1024
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
