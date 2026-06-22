"""
独立摄像头测试程序 — 物件颜色检测器 V2

用法:
    python vision/test_color_detector_live.py

操作:
    将物品放入画面中央 ROI 框内 → 按 空格 识别单帧
    按 Q 退出
"""

import sys
import logging
import cv2
import numpy as np

sys.path.insert(0, ".")
from vision.color_detector import ObjectColorDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

# ── 固定颜色定义 ──
ROI_COLOR = (0, 255, 255)       # 黄色 ROI 框
MATCH_COLOR = (0, 255, 0)       # 绿色 — 识别成功
FAIL_COLOR = (0, 0, 255)        # 红色 — 识别失败
INFO_COLOR = (255, 255, 255)    # 白色 — 信息文字
DEBUG_COLOR = (200, 200, 200)   # 灰色 — debug 信息


def draw_roi(display, region, color=ROI_COLOR, thickness=2):
    """绘制 ROI 框"""
    x1, y1, x2, y2 = region
    cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
    # 对角线强调
    cv2.line(display, (x1, y1), (x2, y2), color, 1)
    cv2.line(display, (x1, y2), (x2, y1), color, 1)


def draw_result(display, detector, result, y_start=30):
    """在画面上叠加识别结果"""
    y = y_start
    line_h = 28

    if result is not None:
        # 成功
        cv2.putText(display, f"MATCH: {result.color_name}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, MATCH_COLOR, 2)
        y += line_h + 5
        cv2.putText(display, f"  基础色系: {result.base_family or '?'}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, INFO_COLOR, 1)
        y += line_h
        cv2.putText(display, f"  置信度:   {result.confidence:.3f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, INFO_COLOR, 1)
        y += line_h
        cv2.putText(display, f"  HSV:      ({result.dominant_hsv[0]}, {result.dominant_hsv[1]}, {result.dominant_hsv[2]})",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INFO_COLOR, 1)
        y += line_h
        if result.dominant_lab:
            cv2.putText(display, f"  Lab:      ({result.dominant_lab[0]:.1f}, {result.dominant_lab[1]:.1f}, {result.dominant_lab[2]:.1f})",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INFO_COLOR, 1)
            y += line_h
        cv2.putText(display, f"  有效像素: {result.valid_pixel_count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, INFO_COLOR, 1)
        y += line_h
        cv2.putText(display, f"  主簇占比: {result.cluster_ratio:.3f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, INFO_COLOR, 1)
    else:
        # 失败
        reason = detector.last_debug.get("failure_reason", "unknown")
        cv2.putText(display, f"NO MATCH", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, FAIL_COLOR, 2)
        y += line_h + 5
        cv2.putText(display, f"  原因: {reason}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, FAIL_COLOR, 1)
        # 显示部分 debug 信息
        if "dominant_hsv" in detector.last_debug:
            hsv = detector.last_debug["dominant_hsv"]
            y += line_h
            cv2.putText(display, f"  提取 HSV: ({hsv[0]}, {hsv[1]}, {hsv[2]})",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DEBUG_COLOR, 1)
        if "base_family" in detector.last_debug:
            y += line_h
            cv2.putText(display, f"  疑似色系: {detector.last_debug['base_family']}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DEBUG_COLOR, 1)
        if "confidence" in detector.last_debug:
            y += line_h
            cv2.putText(display, f"  置信度:   {detector.last_debug['confidence']:.3f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DEBUG_COLOR, 1)


def main():
    print("=" * 60)
    print("  物件颜色检测器 — 独立摄像头测试 (V2)")
    print("=" * 60)
    print()
    print("  操作:")
    print("    将物品放入黄色 ROI 框内 → 按 空格 识别")
    print("    按 Q 退出")
    print("    不会连续自动识别，每次空格只识别当前单帧")
    print()

    detector = ObjectColorDetector()
    print(f"[OK] 检测器已就绪，颜色配置数: {len(detector.color_profiles)}")
    print()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_result = None
    last_debug = {}

    print("[运行] 等待按键...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取帧失败")
            continue

        display = frame.copy()

        # 计算 ROI
        region = detector.get_default_region(frame)

        # 绘制 ROI 框
        draw_roi(display, region)

        # 显示上次结果
        if last_result is not None or last_debug:
            draw_result(display, detector, last_result)

        # 提示文字
        h, w = display.shape[:2]
        cv2.putText(display, "SPACE=识别 | Q=退出", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, DEBUG_COLOR, 1)

        cv2.imshow("Color Detector Live Test", display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == 32:  # 空格
            # 单帧识别
            result = detector.detect(frame, region)
            if result is not None:
                print(f"[识别] {result.color_name}  置信度={result.confidence:.3f}  "
                      f"色系={result.base_family}  HSV={result.dominant_hsv}  "
                      f"Lab=({result.dominant_lab[0]:.1f},{result.dominant_lab[1]:.1f},{result.dominant_lab[2]:.1f})  "
                      f"像素={result.valid_pixel_count}  簇比={result.cluster_ratio:.3f}")
                last_result = result
                last_debug = {}
            else:
                reason = detector.last_debug.get("failure_reason", "?")
                print(f"[识别] NO MATCH — {reason}")
                print(f"  debug: { {k: v for k, v in detector.last_debug.items() if k != 'candidates'} }")
                last_result = None
                last_debug = dict(detector.last_debug)

    cap.release()
    cv2.destroyAllWindows()
    print("\n[退出] 测试结束")


if __name__ == "__main__":
    main()
