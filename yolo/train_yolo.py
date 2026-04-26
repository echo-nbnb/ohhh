#!/usr/bin/env python3
"""
YOLO 模型训练脚本
用于训练模块识别模型

使用方法:
    python train_yolo.py
"""

import sys
sys.path.insert(0, '..')

from ultralytics import YOLO


def main():
    print("=" * 50)
    print("YOLO 模块识别模型训练")
    print("=" * 50)

    # 加载预训练模型（n=轻量版，s=标准版）
    # yolov8n.pt - 最轻量，适合边缘设备
    # yolov8s.pt - 标准版，精度更高
    print("加载预训练模型 yolov8n.pt...")
    model = YOLO('yolov8n.pt')

    # 训练
    print("\n开始训练...")
    print("参数说明:")
    print("  data    - 数据集配置文件")
    print("  epochs  - 训练轮数")
    print("  imgsz   - 输入图片尺寸")
    print("  batch   - 批次大小（根据显存调整）")
    print("  device   - 运行设备（0=GPU0, cpu=CPU）")
    print()

    results = model.train(
        data='dataset.yaml',      # 数据集配置
        epochs=100,               # 训练轮数，可根据效果调整
        imgsz=640,                # 输入图片尺寸
        batch=16,                 # 批次大小（4060建议16-32）
        device=0,                 # 使用GPU 0
        project='../runs/detect', # 输出目录
        name='module_detector',   # 项目名称
        exist_ok=True,            # 允许覆盖已有结果
        verbose=True,             # 显示详细输出
        save=True,                # 保存模型
        save_period=10,           # 每10轮保存一次
    )

    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"模型保存在: ../runs/detect/module_detector/")
    print("=" * 50)

    # 验证模型
    print("\n验证模型性能...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    return results


if __name__ == "__main__":
    main()
