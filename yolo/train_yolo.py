#!/usr/bin/env python3
"""
YOLO 颜色牌检测模型训练脚本
训练 6 类颜色牌检测模型，用于第一幕"择色"

使用方法:
    python yolo/train_yolo.py

训练前准备:
    1. python yolo/collect_data.py --output dataset/images/train --prefix color_card
    2. 用 LabelImg 标注每张图片，导出 YOLO txt 到 dataset/labels/train/
    3. 确保 yolo/dataset.yaml 中 path 正确
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, '..')

from ultralytics import YOLO


def main():
    print("=" * 50)
    print("YOLO 颜色牌检测模型训练 (6 类)")
    print("=" * 50)

    print("加载预训练模型 yolov8n.pt...")
    model = YOLO('yolov8n.pt')

    print("\n数据集: yolo/dataset.yaml (6 类颜色牌)")
    print("  0: 岳麓绿  1: 书院红  2: 西迁黄")
    print("  3: 湘江蓝  4: 校徽金  5: 墨色")
    print()

    results = model.train(
        data=os.path.join(os.path.dirname(__file__), 'dataset.yaml'),
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        project=os.path.join(os.path.dirname(__file__), '..', 'runs', 'detect'),
        name='color_card_detector',
        exist_ok=True,
        verbose=True,
        save=True,
        save_period=10,
        amp=False,
        workers=0,
        deterministic=True,
    )

    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"模型保存在: ../runs/detect/color_card_detector/")
    print("=" * 50)

    print("\n验证模型性能...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    print("\n部署说明:")
    print("  1. 将 best.pt 复制为 yolo/color_card.pt")
    print("  2. 在 test_integrated.py 中设置 model_path='yolo/color_card.pt'")
    print("  3. 取消 _run_camera_loop 中 YOLO 检测的注释")

    return results


if __name__ == "__main__":
    main()
