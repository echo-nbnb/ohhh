#!/usr/bin/env python3
"""
YOLO 模型预测脚本
使用训练好的模型检测模块

使用方法:
    python predict_yolo.py                           # 使用默认摄像头
    python predict_yolo.py --model best.pt          # 指定模型
    python predict_yolo.py --source image.jpg        # 检测图片
    python predict_yolo.py --source 0                # 使用默认摄像头
"""

import sys
import cv2
import argparse

sys.path.insert(0, '..')

from ultralytics import YOLO
from vision import IPCamera

# 默认摄像头地址
try:
    from config_ipcam import CAMERA_URL as DEFAULT_CAMERA
except ImportError:
    DEFAULT_CAMERA = "http://10.54.71.31:8080/video"


def main():
    parser = argparse.ArgumentParser(description='YOLO 模块检测')
    parser.add_argument('--model', type=str,
                       default='../runs/detect/module_detector/weights/best.pt',
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

    args = parser.parse_args()

    print("=" * 50)
    print("YOLO 模块检测")
    print("=" * 50)

    # 加载模型
    print(f"加载模型: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请先运行 train_yolo.py 训练模型")
        return

    # 判断数据来源
    if args.source == '0':
        # 使用IP摄像头
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

            # 检测
            results = model.predict(frame, conf=args.conf, verbose=False)

            # 绘制结果
            annotated = results[0].plot()

            cv2.imshow("YOLO Detection", annotated)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        camera.release()
        cv2.destroyAllWindows()

    else:
        # 检测图片或视频
        print(f"检测来源: {args.source}")
        results = model.predict(args.source, conf=args.conf, save=args.save)

        if args.save:
            print(f"结果已保存到: {results[0].save_dir}")


if __name__ == "__main__":
    main()
