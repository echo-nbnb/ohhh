#!/usr/bin/env python3
"""
区域标定Demo - 四点透视方式
通过鼠标点击4个点确定任意四边形区域，适应透视变换

使用方法:
    python demo_region_calibration.py
    python demo_region_calibration.py
"""

import sys
import cv2
import time
import numpy as np

sys.path.insert(0, '.')

from vision import IPCamera, HandDetector, RegionCalibrator, ManualRegionSelector


def demo_physical_calibration(camera_url: str):
    """物理标定物方式Demo"""
    print("=" * 50)
    print("物理标定物方式")
    print("=" * 50)
    print("请在摄像头前放置一个红色矩形框作为标定物")
    print("按 'r' 重新标定")
    print("按 'c' 切换标定颜色 (红/蓝/黄/绿)")
    print("按 ESC 退出")
    print("=" * 50)

    camera = IPCamera(camera_url)
    if not camera.connect():
        print("错误: 无法连接到摄像头!")
        return

    calibrator = RegionCalibrator(calibrate_color="red", min_area=2000)
    hand_detector = HandDetector(num_hands=1)

    colors = ["red", "blue", "yellow", "green"]
    color_idx = 0
    timestamp_ms = 0

    print("\n请将标定物（红色矩形框）放置在希望检测的区域...")

    while camera.is_connected():
        frame = camera.read_frame()
        if frame is None:
            continue

        if not calibrator.is_calibrated():
            calibrator.calibrate(frame)

        display = calibrator.draw_calibration_visualization(frame)

        region = calibrator.get_region()
        if region and calibrator.is_calibrated():
            x, y, w, h = region
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                annotated_roi, results = hand_detector.detect_and_draw(roi, timestamp_ms, mode="area")
                display[y:y+h, x:x+w] = annotated_roi
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status = f"Color: {colors[color_idx].upper()}"
        cv2.putText(display, status, (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Physical Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            calibrator.reset()
        elif key == ord('c'):
            color_idx = (color_idx + 1) % len(colors)
            calibrator = RegionCalibrator(calibrate_color=colors[color_idx], min_area=2000)

        timestamp_ms += 33

    hand_detector.close()
    camera.release()
    cv2.destroyAllWindows()


def demo_four_point_calibration(camera_url: str):
    """
    四点透视标定Demo
    鼠标左键点击4个点确定区域，右键删除上一个点
    """
    print("=" * 50)
    print("四点透视标定方式")
    print("=" * 50)
    print("鼠标左键点击4个点确定区域（左上->右上->右下->左下）")
    print("右键删除上一个点")
    print("按 'r' 重置")
    print("按 ESC 退出")
    print("=" * 50)

    camera = IPCamera(camera_url)
    if not camera.connect():
        print("错误: 无法连接到摄像头!")
        return

    selector = ManualRegionSelector()
    hand_detector = HandDetector(num_hands=1)

    window_name = "Four-Point Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, selector.mouse_callback)

    timestamp_ms = 0

    while camera.is_connected():
        frame = camera.read_frame()
        if frame is None:
            continue

        display = selector.draw_selection(frame)

        # 如果已完成4点选择，执行透视变换并检测
        if selector.is_complete():
            quad = selector.get_quad()
            if quad is not None:
                x, y, w, h = cv2.boundingRect(quad.astype(np.int32))

                # 透视变换
                warped, M = selector.get_warped_region(frame)
                if warped is not None:
                    # 在变换后的矩形区域检测手部
                    annotated_warped, results = hand_detector.detect_and_draw(warped, timestamp_ms, mode="area")

                    # 创建输出图像
                    output = np.ones((h, w, 3), dtype=np.uint8) * 255
                    output = annotated_warped.copy()

                    # 将检测结果贴回主画面（放在角落）
                    display_small = cv2.resize(display, (640, 480))
                    # 在左上角显示透视变换后的区域
                    cv2.namedWindow("Warped Region", cv2.WINDOW_NORMAL)
                    cv2.imshow("Warped Region", annotated_warped)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            selector.reset()
            cv2.destroyWindow("Warped Region")

        timestamp_ms += 33

    hand_detector.close()
    camera.release()
    cv2.destroyAllWindows()
    cv2.destroyWindow("Warped Region")


def main():
    if len(sys.argv) > 1:
        camera_url = sys.argv[1]
    else:
        try:
            from config_ipcam import CAMERA_URL
            camera_url = CAMERA_URL
        except ImportError:
            camera_url = "http://10.54.71.31:8080/video"

    print("\n" + "=" * 50)
    print("区域标定Demo - 选择标定方式")
    print("=" * 50)
    print("1. 物理标定物方式（红色矩形框）")
    print("2. 四点透视标定（鼠标点击4个点）")
    print("=" * 50)

    choice = input("请选择 (1/2): ").strip()

    if choice == "1":
        demo_physical_calibration(camera_url)
    else:
        demo_four_point_calibration(camera_url)


if __name__ == "__main__":
    main()
