#!/usr/bin/env python3
"""
数据集采集脚本
通过IP摄像头采集模块图片，用于YOLO训练

使用方法:
    python collect_data.py                    # 默认保存到 ../dataset/images/train
    python collect_data.py --output my_folder # 保存到指定文件夹
    python collect_data.py --val             # 保存到验证集
"""

import sys
import os
import cv2
import argparse
import time

sys.path.insert(0, '..')

from vision import IPCamera


def main():
    parser = argparse.ArgumentParser(description='采集模块图片用于YOLO训练')
    parser.add_argument('--output', type=str, default='../dataset/images/train',
                       help='输出文件夹路径')
    parser.add_argument('--val', action='store_true',
                       help='保存到验证集 ../dataset/images/val')
    parser.add_argument('--camera', type=str,
                       default='http://10.7.46.53:8080/video',
                       help='摄像头地址')
    parser.add_argument('--interval', type=int, default=0,
                       help='自动拍摄间隔（秒），0=手动拍摄')
    parser.add_argument('--prefix', type=str, default='module',
                       help='图片文件名前缀')

    args = parser.parse_args()

    # 如果使用 --val 参数
    if args.val:
        args.output = '../dataset/images/val'

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    print("=" * 50)
    print("数据集采集工具")
    print("=" * 50)
    print(f"输出目录: {args.output}")
    print(f"摄像头: {args.camera}")
    print(f"自动拍摄间隔: {'关闭' if args.interval == 0 else f'{args.interval}秒'}")
    print()
    print("操作说明:")
    print("  SPACE   - 拍摄当前帧")
    print("  S       - 开始/停止自动拍摄")
    print("  R       - 查看已采集数量")
    print("  ESC/q   - 退出")
    print("=" * 50)

    # 连接摄像头
    camera = IPCamera(args.camera)
    if not camera.connect():
        print("错误: 无法连接到摄像头!")
        return

    print(f"\n摄像头已连接，分辨率: 640x480")
    print(f"当前输出目录: {args.output}")

    # 获取当前已采集数量
    existing_files = [f for f in os.listdir(args.output) if f.endswith(('.jpg', '.png'))]
    count = len(existing_files)
    print(f"已采集图片: {count} 张")

    auto_capture = False
    last_capture_time = time.time()
    window_name = "Data Collection"

    while camera.is_connected():
        frame = camera.read_frame()
        if frame is None:
            continue

        display = frame.copy()

        # 显示提示
        cv2.putText(display, f"Output: {args.output}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, f"Saved: {count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        status = "AUTO" if auto_capture else "MANUAL"
        status_color = (0, 255, 255) if auto_capture else (255, 255, 255)
        cv2.putText(display, f"Mode: {status}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # 拍摄提示
        cv2.putText(display, "SPACE: Capture | S: Auto | R: Count | ESC: Exit",
                   (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # 添加边框指示
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, 0), (w-1, h-1), (0, 255, 0), 2)

        cv2.imshow(window_name, display)

        # 自动拍摄
        if auto_capture and args.interval > 0:
            if time.time() - last_capture_time >= args.interval:
                save_image(args.output, args.prefix, frame)
                count += 1
                last_capture_time = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC 或 q
            break
        elif key == 32:  # SPACE
            save_image(args.output, args.prefix, frame)
            count += 1
        elif key == ord('s'):  # S 切换自动拍摄
            auto_capture = not auto_capture
            print(f"自动拍摄: {'开启' if auto_capture else '关闭'}")
        elif key == ord('r'):  # R 查看数量
            print(f"已采集: {count} 张")

    camera.release()
    cv2.destroyAllWindows()
    print(f"\n采集完成，共 {count} 张图片保存在 {args.output}")


def save_image(output_dir: str, prefix: str, frame) -> str:
    """保存图片"""
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"已保存: {filename}")
    return filepath


if __name__ == "__main__":
    main()
