"""
IP摄像头颜色识别独立测试

用法:
    python vision/test_ipcam_color.py [IP摄像头URL]

默认摄像头: config_ipcam.py 或 http://10.83.31.143:8080/video

操作:
    将物品放入画面中央 ROI 框内 → 自动连续识别
    按 空格 暂停/继续识别
    按 Q 退出
"""

import sys
import time
import cv2
import numpy as np

sys.path.insert(0, ".")
from vision.color_detector import ObjectColorDetector, COLOR_HEX_VALUES

# ── 常量 ──
ROI_COLOR = (0, 255, 255)
MATCH_COLOR = (0, 255, 0)
FAIL_COLOR = (0, 0, 255)
INFO_COLOR = (255, 255, 255)

# 所有 22 色（21 + 墨色）
ALL_COLORS = list(COLOR_HEX_VALUES.keys()) + ["墨色"]


def get_url():
    """获取 IP 摄像头 URL"""
    url = sys.argv[1] if len(sys.argv) > 1 else ""
    if not url:
        try:
            from config_ipcam import CAMERA_URL
            url = CAMERA_URL
        except ImportError:
            url = "http://10.83.31.143:8080/video"
    return url


def hsv_match(hue, sat, val):
    """HSV → 六色基色匹配"""
    if sat < 40 or val < 40:
        return "墨色"
    if hue <= 10 or hue >= 170:
        return "朱红"
    if 11 <= hue <= 25:
        return "灯橙"
    if 26 <= hue <= 34:
        return "梨黄"
    if 35 <= hue <= 85:
        return "叶绿"
    if 86 <= hue <= 125:
        return "瓷青" if sat < 180 else "海蓝"
    return "烟紫"


def draw_overlay(display, region, result, fps, paused):
    """绘制 ROI 框 + 识别结果"""
    h, w = display.shape[:2]
    x1, y1, x2, y2 = region

    # ROI 框
    cv2.rectangle(display, (x1, y1), (x2, y2), ROI_COLOR, 2)
    cv2.line(display, (x1, y1), (x2, y2), ROI_COLOR, 1)
    cv2.line(display, (x1, y2), (x2, y1), ROI_COLOR, 1)

    # 识别结果
    y = 30
    if result:
        cv2.putText(display, f"COLOR: {result.color_name}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, MATCH_COLOR, 2)
        y += 30
        cv2.putText(display, f"  family: {result.base_family or '?'}  conf: {result.confidence:.3f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INFO_COLOR, 1)
        y += 22
        cv2.putText(display, f"  HSV: ({result.dominant_hsv[0]},{result.dominant_hsv[1]},{result.dominant_hsv[2]})",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INFO_COLOR, 1)
        y += 22
        cv2.putText(display, f"  pixels: {result.valid_pixel_count}  ratio: {result.cluster_ratio:.3f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INFO_COLOR, 1)
        # 色块
        hex_str = COLOR_HEX_VALUES.get(result.color_name, {}).get("hsv_center")
        if hex_str:
            hsv_pixel = np.uint8([[[hex_str[0], hex_str[1], hex_str[2]]]])
            bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
            cv2.rectangle(display, (w - 80, 10), (w - 10, 60),
                          (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)
    else:
        cv2.putText(display, "NO MATCH", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, FAIL_COLOR, 2)
        if hasattr(detector, 'last_debug'):
            reason = detector.last_debug.get("failure_reason", "")
            y += 30
            cv2.putText(display, f"  {reason}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, FAIL_COLOR, 1)

    # 状态
    status = "PAUSED" if paused else f"FPS: {fps:.0f}"
    cv2.putText(display, status, (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255) if paused else INFO_COLOR, 1)
    cv2.putText(display, "SPACE=pause | Q=quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


def main():
    url = get_url()
    print("=" * 60)
    print("  IP摄像头颜色识别测试")
    print("=" * 60)
    print(f"  摄像头: {url}")
    print(f"  操作: 空格=暂停  Q=退出")
    print()

    # 连接 IP 摄像头
    try:
        from vision.ipcamera import IPCamera
        cam = IPCamera(url, target_width=1280, target_height=720)
        if not cam.connect():
            print("[ERROR] IP摄像头连接失败")
            return
        print("[OK] IP摄像头已连接")
    except Exception as e:
        print(f"[!] IPCamera 导入失败: {e}，尝试 OpenCV 直连...")
        cam = cv2.VideoCapture(url)
        if not cam.isOpened():
            print("[ERROR] 无法打开摄像头")
            return

    # 初始化检测器
    detector = ObjectColorDetector()
    print(f"[OK] 检测器就绪，颜色数: {len(detector.color_profiles)}")

    paused = False
    last_result = None
    fps = 0
    frame_count = 0
    t0 = time.time()

    print("\n[运行] 按 Q 退出...")

    while True:
        if not paused:
            if hasattr(cam, 'read'):
                ret, frame = cam.read()
            else:
                ret, frame = True, cam.read()[1]

            if not ret or frame is None:
                print("[WARN] 帧读取失败，重试...")
                time.sleep(0.1)
                continue

            frame_count += 1
            display = frame.copy()
            h, w = frame.shape[:2]

            # ROI: 中央 40% 区域
            region = detector.get_default_region(frame)

            # 连续识别
            result = detector.detect(frame, region)
            if result:
                last_result = result

            # 计算 FPS
            if frame_count % 30 == 0:
                t1 = time.time()
                fps = 30 / (t1 - t0 + 0.001)
                t0 = t1
        else:
            display = frame.copy() if 'frame' in dir() else np.zeros((720, 1280, 3), dtype=np.uint8)

        draw_overlay(display, region if not paused else (0, 0, w, h),
                     last_result if paused else (result if 'result' in dir() else None),
                     fps, paused)

        cv2.imshow("IP Camera Color Detection", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == 32:  # 空格
            paused = not paused
            if paused:
                print(f"\n[暂停] 当前颜色: {last_result.color_name if last_result else '无'}")
                if last_result:
                    print(f"  HSV: {last_result.dominant_hsv}")
                    print(f"  置信度: {last_result.confidence:.3f}")
                    print(f"  色系: {last_result.base_family}")
                    print(f"  像素: {last_result.valid_pixel_count}")
                    print(f"  簇比: {last_result.cluster_ratio:.3f}")
            else:
                print("[继续]")

    cv2.destroyAllWindows()
    print("\n[退出]")


if __name__ == "__main__":
    main()
