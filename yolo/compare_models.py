#!/usr/bin/env python3
"""
模型对比测试脚本
对比两个训练模型的检测效果
"""

import os
import sys
os.chdir(os.path.dirname(__file__) or '.')

from ultralytics import YOLO

# 验证图片目录
VAL_IMG = "../dataset/images/val"
MODEL_1 = "runs/detect/train_20260427_213530/best.pt"
MODEL_2 = "runs/detect/train_20260427_214155/best.pt"

def test_model(model_path, name):
    print(f"\n{'='*50}")
    print(f"测试模型: {name}")
    print(f"路径: {model_path}")
    print(f"{'='*50}")

    model = YOLO(model_path)

    # 列出验证图片
    imgs = [f for f in os.listdir(VAL_IMG) if f.endswith(('.jpg', '.png'))]
    print(f"验证图片数: {len(imgs)}")

    results = []
    for img in imgs:
        result = model.predict(os.path.join(VAL_IMG, img), imgsz=640, conf=0.3, verbose=False)
        count = len(result[0].boxes)
        results.append((img, count))
        print(f"  {img}: 检测到 {count} 个目标")

    total = sum(r[1] for r in results)
    avg = total / max(len(results), 1)
    print(f"\n总计: {total} 个检测, 平均 {avg:.1f} 个/图")
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("YOLO 模型对比测试")
    print("=" * 50)

    r1 = test_model(MODEL_1, "模型1 (train_20260427_213530)")
    r2 = test_model(MODEL_2, "模型2 (train_20260427_214155)")

    print("\n" + "=" * 50)
    print("对比结果")
    print("=" * 50)
    print(f"模型1 总检测: {sum(r[1] for r in r1)}")
    print(f"模型2 总检测: {sum(r[1] for r in r2)}")