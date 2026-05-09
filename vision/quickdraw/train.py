"""
训练 QuickDraw 分类模型 + 导出 ONNX

用法:
    python -m vision.quickdraw.train           # 完整训练
    python -m vision.quickdraw.train --resume  # 从 checkpoint 恢复
    python -m vision.quickdraw.train --export-only  # 仅导出 ONNX（已有 checkpoint）

输出:
    vision/models/quickdraw_mobilenet.onnx     # 供 sketch_recognizer.py 加载
    vision/quickdraw/checkpoint.pth            # 训练断点，可恢复
"""

import os
import sys
import argparse
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision.quickdraw.config import (
    CATEGORIES, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MODEL_PATH, CHECKPOINT_PATH, IDX_TO_CATEGORY,
)
from vision.quickdraw.model import QuickDrawCNN
from vision.quickdraw.dataset import QuickDrawDataset


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(loader), correct / total


def export_onnx(model, device):
    """导出 ONNX 模型"""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model.eval()
    dummy = torch.randn(1, 1, 28, 28).to(device)

    torch.onnx.export(
        model,
        dummy,
        MODEL_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    print(f"\nONNX 模型已导出: {MODEL_PATH}")

    # 同时保存类别映射
    mapping_path = os.path.join(os.path.dirname(MODEL_PATH), "quickdraw_classes.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(IDX_TO_CATEGORY, f, ensure_ascii=False, indent=2)
    print(f"类别映射已保存: {mapping_path}")


def main():
    parser = argparse.ArgumentParser(description="QuickDraw 模型训练")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复训练")
    parser.add_argument("--export-only", action="store_true", help="仅导出 ONNX（需已有 checkpoint）")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"类别数: {NUM_CLASSES}")

    model = QuickDrawCNN(num_classes=NUM_CLASSES).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 仅导出模式
    if args.export_only:
        if os.path.exists(CHECKPOINT_PATH):
            print(f"加载 checkpoint: {CHECKPOINT_PATH}")
            state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            export_onnx(model, device)
            return
        else:
            print(f"checkpoint 不存在: {CHECKPOINT_PATH}")
            sys.exit(1)

    # ---- 加载数据 ----
    print("\n" + "=" * 50)
    print("加载数据...")
    train_ds = QuickDrawDataset(train=True)
    val_ds = QuickDrawDataset(train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # ---- 优化器 & 损失 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_acc = 0.0

    # 恢复训练
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        print(f"恢复 checkpoint: {CHECKPOINT_PATH}")
        state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_acc = state.get("best_acc", 0.0)

    # ---- 训练循环 ----
    print("\n" + "=" * 50)
    print(f"开始训练 (epochs={args.epochs}, lr={args.lr}, batch={args.batch_size})")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"{elapsed:.0f}s")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "num_classes": NUM_CLASSES,
            }, CHECKPOINT_PATH)

    print(f"\n训练完成，最佳验证准确率: {best_acc:.3f}")

    # ---- 加载最佳模型并导出 ONNX ----
    if os.path.exists(CHECKPOINT_PATH):
        state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print("已加载最佳 checkpoint")

    export_onnx(model, device)


if __name__ == "__main__":
    main()
