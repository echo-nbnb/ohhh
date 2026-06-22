#!/usr/bin/env python3
"""IP 摄像头 + MediaPipe 手部骨架测试"""

import sys, cv2, time
sys.path.insert(0, ".")

from config_ipcam import CAMERA_URL
from vision.ipcamera import IPCamera

# 连接 IP 摄像头
print(f"连接 IP 摄像头: {CAMERA_URL}")
cam = IPCamera(CAMERA_URL, target_width=1280, target_height=720)
if not cam.connect():
    print("连接失败")
    sys.exit(1)

# 初始化 MediaPipe 手部追踪
print("初始化 MediaPipe 手部追踪...")
from vision.hand_tracker import HandTracker, HAND_CONNECTIONS
tracker = HandTracker()

# 初始化颜色检测
print("初始化颜色检测...")
from vision.color_detector import ObjectColorDetector
color_detector = ObjectColorDetector()
last_color = ("--", 0.0)
fc = 0

print("运行中 — 按 ESC 退出")
cv2.namedWindow("IP Camera + MediaPipe Skeleton")

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        time.sleep(0.01)
        continue

    h, w = frame.shape[:2]
    display = frame.copy()
    fc += 1
    gesture_id = 0

    # 手部检测 + 骨架绘制 + 手势识别
    results = tracker._detect(frame, int(time.time() * 1000))
    if results.hand_landmarks:
        for hand_lm in results.hand_landmarks:
            for i, lm in enumerate(hand_lm):
                px, py = int(lm.x * w), int(lm.y * h)
                c = (0, 255, 0) if i % 4 == 0 else (0, 200, 0)
                cv2.circle(display, (px, py), 5, c, -1)
            for a, b in HAND_CONNECTIONS:
                pt1 = (int(hand_lm[a].x * w), int(hand_lm[a].y * h))
                pt2 = (int(hand_lm[b].x * w), int(hand_lm[b].y * h))
                cv2.line(display, pt1, pt2, (0, 255, 0), 2)

            # 手势识别：1=张手 2=食指 3=握拳
            get_y = lambda idx: hand_lm[idx].y
            tips = {8: 6, 12: 10, 16: 14, 20: 18}  # tip: mcp
            ext = [get_y(t) < get_y(m) - 0.05 for t, m in tips.items()]
            if all(ext):       gesture_id = 1  # 张手
            elif ext[0] and not any(ext[1:]): gesture_id = 2  # 食指
            elif not any(ext): gesture_id = 3  # 握拳

        # 画手势编号
        cv2.putText(display, str(gesture_id), (w - 80, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)

    # 颜色检测（每 30 帧），输出到控制台
    if fc % 30 == 0:
        region = color_detector.detect_dominant_region(frame)
        result = color_detector.detect(frame, region)
        if result:
            print(f"\r[颜色] {result.color_name}  置信度: {result.confidence:.2f}   HSV: {result.dominant_hsv}  帧: {fc}", end="", flush=True)
        else:
            print(f"\r[颜色] 未匹配六色  帧: {fc}", end="", flush=True)

    cv2.imshow("IP Camera + MediaPipe Skeleton", display)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cam.release()
cv2.destroyAllWindows()
print("退出")
