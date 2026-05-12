#!/usr/bin/env python3
"""
YOLO 颜色牌检测 — 实时预测与跟踪
使用训练好的颜色牌检测模型（6 类），内置 BoT-SORT 跟踪器保持跨帧 ID 一致

使用方法:
    python yolo/predict_yolo.py                           # 默认摄像头 + 默认模型
    python yolo/predict_yolo.py --model yolo/color_card.pt  # 指定模型
    python yolo/predict_yolo.py --source image.jpg        # 检测图片
    python yolo/predict_yolo.py --no-track               # 仅检测，不使用跟踪

模型部署到集成系统后请使用 vision/color_card_detector.py 替代本脚本。
"""

import sys
import cv2
import argparse

sys.path.insert(0, '..')

from ultralytics import YOLO
from vision import IPCamera

try:
    from config_ipcam import CAMERA_URL as DEFAULT_CAMERA
except ImportError:
    DEFAULT_CAMERA = "http://10.54.71.31:8080/video"


def main():
    parser = argparse.ArgumentParser(description='YOLO 颜色牌检测 (6 类)')
    parser.add_argument('--model', type=str,
                       default='../runs/detect/color_card_detector/weights/best.pt',
                       help='模型路径')
    parser.add_argument('--source', type=str, default='0',
                       help='图片来源: 0=摄像头, 文件路径, 或视频路径')
    parser.add_argument('--camera', type=str,
                       default=DEFAULT_CAMERA,
                       help='IP摄像头地址')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--save', action='store_true',
                       help='保存检测结果')
    parser.add_argument('--no-track', action='store_true',
                       help='禁用多目标跟踪（仅逐帧检测）')

    args = parser.parse_args()

    CLASS_NAMES = ["岳麓绿", "书院红", "西迁黄", "湘江蓝", "校徽金", "墨色"]

    print("=" * 50)
    print("YOLO 颜色牌检测 (6 类)")
    print("=" * 50)

    print(f"加载模型: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请先运行 train_yolo.py 训练模型")
        return

    if args.source == '0':
        print(f"连接摄像头: {args.camera}")
        camera = IPCamera(args.camera)
        if not camera.connect():
            print("摄像头连接失败!")
            return
        print("摄像头已连接，按 ESC 退出")

        while camera.is_connected():
            frame = camera.read_frame()
            if frame is None:
                continue

            if args.no_track:
                results = model.predict(frame, conf=args.conf, verbose=False)
            else:
                results = model.track(frame, conf=args.conf, persist=True, verbose=False)

            annotated = results[0].plot()
            cv2.imshow("YOLO Color Card Detection", annotated)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        camera.release()
        cv2.destroyAllWindows()

    else:
        print(f"检测来源: {args.source}")
        if args.no_track:
            results = model.predict(args.source, conf=args.conf, save=args.save)
        else:
            results = model.track(args.source, conf=args.conf, save=args.save)

        if args.save:
            print(f"结果已保存到: {results[0].save_dir}")


if __name__ == "__main__":
    main()
