"""
电脑摄像头完整功能测试 — 颜色识别 + 握拳 + 手势绘画

用法:
    python vision/test_webcam_full.py

操作:
    握拳  → 触发颜色识别（画面中心区域取色）
    张手  → 取消/重置
    伸食指 → 绘画（指尖轨迹实时显示）
    握拳  → 确认绘画 → 识别物象
    Q     → 退出
    R     → 重置全部状态
    C     → 手动触发颜色识别
"""

import sys
import math
import time
import random
import cv2
import numpy as np

sys.path.insert(0, ".")

# ── MediaPipe 手部 ──
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode

PROJECT_ROOT = "."
MODEL_PATH = "hand_landmarker.task"

# ── 手势类型 ──
class Gesture:
    NONE = "none"
    FIST = "fist"
    OPEN = "open"
    INDEX = "index_pointing"

# ── 颜色定义 ──
ALL_COLORS = [
    "朱红","灯橙","梨黄","叶绿","瓷青","海蓝","烟紫",
    "枫红","暖橙","藤黄","玉绿","石青","澄蓝","影紫",
    "桃红","夕橙","桂黄","茶绿","湖青","沧蓝","黛紫","墨色"
]

OBJECTS = ["古树","书卷","碑刻","竹林","石阶","讲堂","爱晚亭","石桥",
           "长廊","流水","桥梁","湖面","林荫道","岳麓山","岳麓书院",
           "图书馆","校门","匾额","砚台","墨锭","窗格","屋脊"]

# ── HSV 六色匹配 ──
def hsv_to_color(hue, sat, val):
    if sat < 40 or val < 40:
        return "墨色"
    if hue <= 10 or hue >= 170:
        return random.choice(["朱红","枫红","桃红"])
    if 11 <= hue <= 25:
        return random.choice(["灯橙","暖橙","夕橙"])
    if 26 <= hue <= 34:
        return random.choice(["梨黄","藤黄","桂黄"])
    if 35 <= hue <= 85:
        return random.choice(["叶绿","玉绿","茶绿"])
    if 86 <= hue <= 125:
        return random.choice(["瓷青","石青","湖青","海蓝","澄蓝","沧蓝"])
    return random.choice(["烟紫","影紫","黛紫"])


# ── 手势识别（指尖-MCP y轴 + 角度）──
def recognize_gesture(landmarks):
    """返回 Gesture 类型"""
    def y(i):
        return landmarks[i].y if hasattr(landmarks[i], 'y') else landmarks[i][1]

    def pt(i):
        lm = landmarks[i]
        return (lm.x if hasattr(lm, 'x') else lm[0],
                lm.y if hasattr(lm, 'y') else lm[1])

    def pip_angle(mcp_i, pip_i, tip_i):
        a, b, c = pt(mcp_i), pt(pip_i), pt(tip_i)
        ba = (a[0]-b[0], a[1]-b[1])
        bc = (c[0]-b[0], c[1]-b[1])
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        m = ((ba[0]**2+ba[1]**2)**0.5) * ((bc[0]**2+bc[1]**2)**0.5)
        if m < 1e-6: return 0
        return math.degrees(math.acos(max(-1, min(1, dot/m))))

    # 四指: (tip, mcp, pip_angle_tuple)
    fingers = [
        (y(8),  y(6),  pip_angle(5, 6, 8)),     # index
        (y(12), y(10), pip_angle(9, 10, 12)),    # middle
        (y(16), y(14), pip_angle(13, 14, 16)),   # ring
        (y(20), y(18), pip_angle(17, 18, 20)),   # pinky
    ]

    pointing_up = [tip < mcp - 0.005 for tip, mcp, _ in fingers]
    is_bent = [ang < 135 for _, _, ang in fingers]

    n_up = sum(pointing_up)
    n_bent = sum(is_bent)

    # 食指朝上 + 最多2根朝上 → INDEX
    if pointing_up[0] and n_up <= 2:
        return Gesture.INDEX
    # 全朝上 → OPEN
    if n_up >= 4:
        return Gesture.OPEN
    # ≥3根朝下 或 ≥2根弯曲 → FIST
    if (4 - n_up) >= 3 and not pointing_up[0]:
        return Gesture.FIST
    if n_bent >= 2 and not pointing_up[0]:
        return Gesture.FIST
    return Gesture.NONE


def main():
    print("=" * 60)
    print("  电脑摄像头完整功能测试")
    print("=" * 60)
    print("  握拳=取色 | 伸食指=绘画 | 握拳=确认 | 张手=取消")
    print("  R=重置 | C=手动取色 | Q=退出")
    print()

    # ── 摄像头 ──
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    # ── MediaPipe 手部 ──
    base_opts = BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(opts)

    # ── 状态 ──
    mode = "color"          # color | drawing
    current_color = None
    current_gesture = Gesture.NONE
    prev_gesture = Gesture.NONE
    gesture_counter = {}    # 防抖
    DEBOUNCE = 3

    trajectory = []         # 绘画轨迹 [(x, y), ...]
    object_result = None    # 识别结果
    confirmed_objects = []

    # ── 主循环 ──
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # 镜像
        h, w = frame.shape[:2]
        display = frame.copy()
        frame_idx += 1

        # ── 手部检测 ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, frame_idx)

        hand_lm = None
        if result.hand_landmarks:
            hand_lm = result.hand_landmarks[0]
            # 手势识别
            raw = recognize_gesture(hand_lm)
            gesture_counter[raw] = gesture_counter.get(raw, 0) + 1
            for g in list(gesture_counter.keys()):
                if g != raw:
                    gesture_counter[g] = max(0, gesture_counter[g] - 1)
            if gesture_counter.get(raw, 0) >= DEBOUNCE:
                prev_gesture = current_gesture
                current_gesture = raw

            # 绘制手部骨架
            for lm in hand_lm:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(display, (px, py), 3, (0, 255, 0), -1)
            CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                          (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                          (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
            for a, b in CONNECTIONS:
                p1 = (int(hand_lm[a].x * w), int(hand_lm[a].y * h))
                p2 = (int(hand_lm[b].x * w), int(hand_lm[b].y * h))
                cv2.line(display, p1, p2, (0, 200, 0), 1)

            # 指尖高亮
            for i in [4, 8, 12, 16, 20]:
                px, py = int(hand_lm[i].x * w), int(hand_lm[i].y * h)
                cv2.circle(display, (px, py), 6, (0, 255, 255), -1)
        else:
            # 手离开 → 衰减
            for g in list(gesture_counter.keys()):
                gesture_counter[g] = max(0, gesture_counter[g] - 1)
            if all(v <= 0 for v in gesture_counter.values()):
                current_gesture = Gesture.NONE

        gesture_changed = (current_gesture != prev_gesture)

        # ── 模式切换 ──
        if gesture_changed:
            if mode == "color":
                if current_gesture == Gesture.FIST:
                    # 取画面中心颜色
                    cx, cy = w // 2, h // 2
                    rw, rh = int(w * 0.1), int(h * 0.1)
                    roi = frame[cy - rh:cy + rh, cx - rw:cx + rw]
                    avg = roi.mean(axis=0).mean(axis=0)
                    hsv_pixel = cv2.cvtColor(np.uint8([[avg]]), cv2.COLOR_BGR2HSV)[0, 0]
                    current_color = hsv_to_color(int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2]))
                    mode = "drawing"
                    trajectory = []
                    object_result = None
                    print(f"[取色] {current_color}  HSV=({int(hsv_pixel[0])},{int(hsv_pixel[1])},{int(hsv_pixel[2])})")

            elif mode == "drawing":
                if current_gesture == Gesture.INDEX:
                    # 开始绘画
                    print("[绘画] 食指伸出，开始作画")
                elif current_gesture == Gesture.FIST:
                    # 确认绘画 → 识别物象
                    if len(trajectory) > 5:
                        object_result = random.choice(OBJECTS)
                        confirmed_objects.append(object_result)
                        print(f"[确认] 物象: {object_result}  (轨迹{len(trajectory)}点)")
                    trajectory = []
                    mode = "color"
                    print("[重置] 回到取色模式")
                elif current_gesture == Gesture.OPEN:
                    # 取消
                    trajectory = []
                    print("[取消] 轨迹已清除")

        # ── 绘画录制 ──
        if mode == "drawing" and hand_lm and current_gesture == Gesture.INDEX:
            tip = hand_lm[8]
            tx, ty = int(tip.x * w), int(tip.y * h)
            trajectory.append((tx, ty))

        # ── 绘制轨迹 ──
        if mode == "drawing" and len(trajectory) > 1:
            pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], False, (255, 200, 50), 3)
        # 已确认物象的轨迹
        for i, obj in enumerate(confirmed_objects):
            cv2.putText(display, f"#{i+1}: {obj}", (w - 200, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── 状态显示 ──
        cv2.putText(display, f"MODE: {mode.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f"GESTURE: {current_gesture}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 0) if current_gesture == Gesture.INDEX else
                   (0, 0, 255) if current_gesture == Gesture.FIST else
                   (255, 0, 0) if current_gesture == Gesture.OPEN else (150, 150, 150), 1)
        if current_color:
            cv2.putText(display, f"COLOR: {current_color}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if object_result:
            cv2.putText(display, f"OBJECT: {object_result}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, f"trajectory: {len(trajectory)} pts", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # ── ROI 框 ──
        cx, cy = w // 2, h // 2
        rw, rh = int(w * 0.1), int(h * 0.1)
        roi_color = (0, 255, 255) if mode == "color" else (100, 100, 100)
        cv2.rectangle(display, (cx - rw, cy - rh), (cx + rw, cy + rh), roi_color, 2)

        # ── 提示 ──
        tips = {
            "color": "握拳取色 | Q=退出",
            "drawing": "伸食指绘画 | 握拳确认 | 张手取消 | Q=退出"
        }
        cv2.putText(display, tips.get(mode, ""), (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Webcam Full Test", display)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            mode = "color"
            current_color = None
            trajectory = []
            object_result = None
            confirmed_objects = []
            print("[重置] 全部状态已清除")
        elif key == ord('c'):
            cx, cy = w // 2, h // 2
            rw, rh = int(w * 0.1), int(h * 0.1)
            roi = frame[cy - rh:cy + rh, cx - rw:cx + rw]
            avg = roi.mean(axis=0).mean(axis=0)
            hsv_pixel = cv2.cvtColor(np.uint8([[avg]]), cv2.COLOR_BGR2HSV)[0, 0]
            current_color = hsv_to_color(int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2]))
            mode = "drawing"
            trajectory = []
            print(f"[手动取色] {current_color}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[退出]")


if __name__ == "__main__":
    main()
