"""
草图识别简单测试 —— 摄像头前用手势画画

操作：
  - 伸出食指 → 绘画（指尖轨迹实时显示）
  - 握拳 → 识别（显示 Top-3 候选物象）
  - 五指张开 → 清空，重画
  - 按 Q → 退出

依赖：摄像头 + MediaPipe（项目中已有 hand_tracker.py）
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.hand_tracker import HandTracker
from vision.hand_detector import HandLandmarkIndex
from vision.sketch_recognizer import create_sketch_recognizer


def is_index_pointing(landmarks) -> bool:
    """仅食指伸展，其余三指握起"""
    if landmarks is None:
        return False
    lm = landmarks
    idx_tip = lm[HandLandmarkIndex.INDEX_TIP]
    idx_mcp = lm[HandLandmarkIndex.INDEX_MCP]
    mid_tip = lm[HandLandmarkIndex.MIDDLE_TIP]
    mid_mcp = lm[HandLandmarkIndex.MIDDLE_MCP]
    ring_tip = lm[HandLandmarkIndex.RING_TIP]
    ring_mcp = lm[HandLandmarkIndex.RING_MCP]
    pinky_tip = lm[HandLandmarkIndex.PINKY_TIP]
    pinky_mcp = lm[HandLandmarkIndex.PINKY_MCP]

    index_extended = idx_tip.y < idx_mcp.y - 0.03
    others_curled = (
        mid_tip.y > mid_mcp.y - 0.02
        and ring_tip.y > ring_mcp.y - 0.02
        and pinky_tip.y > pinky_mcp.y - 0.02
    )
    return index_extended and others_curled


def is_fist(landmarks) -> bool:
    """五指全部握起"""
    if landmarks is None:
        return False
    lm = landmarks
    return not (
        lm[HandLandmarkIndex.INDEX_TIP].y < lm[HandLandmarkIndex.INDEX_MCP].y - 0.02
        or lm[HandLandmarkIndex.MIDDLE_TIP].y < lm[HandLandmarkIndex.MIDDLE_MCP].y - 0.02
        or lm[HandLandmarkIndex.RING_TIP].y < lm[HandLandmarkIndex.RING_MCP].y - 0.02
        or lm[HandLandmarkIndex.PINKY_TIP].y < lm[HandLandmarkIndex.PINKY_MCP].y - 0.02
    )


def is_open_hand(landmarks) -> bool:
    """五指全部伸展"""
    if landmarks is None:
        return False
    lm = landmarks
    return (
        lm[HandLandmarkIndex.INDEX_TIP].y < lm[HandLandmarkIndex.INDEX_MCP].y - 0.03
        and lm[HandLandmarkIndex.MIDDLE_TIP].y < lm[HandLandmarkIndex.MIDDLE_MCP].y - 0.03
        and lm[HandLandmarkIndex.RING_TIP].y < lm[HandLandmarkIndex.RING_MCP].y - 0.03
        and lm[HandLandmarkIndex.PINKY_TIP].y < lm[HandLandmarkIndex.PINKY_MCP].y - 0.03
    )


def main():
    print(__doc__)

    # 模拟颜色（按数字键切换）
    colors = ["岳麓绿", "书院红", "西迁黄", "湘江蓝", "校徽金", "墨色"]
    current_color_idx = 0
    current_color = colors[current_color_idx]

    # 初始化
    tracker = HandTracker()
    recognizer = create_sketch_recognizer()
    model_path = os.path.join(os.path.dirname(__file__), "models", "quickdraw_mobilenet.onnx")
    if os.path.exists(model_path):
        ok = recognizer.load_model(model_path)
        print(f"ONNX 模型: {'已加载' if ok else '加载失败，使用启发式降级'}")
    else:
        print(f"ONNX 模型不存在 ({model_path})，使用启发式降级")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 状态
    trajectory = []          # 当前轨迹
    is_drawing = False
    results_text = []        # 识别结果文本
    result_timer = 0         # 结果显示计时
    last_gesture = "none"
    frame_count = 0          # 帧计数（用作时间戳）

    # 轨迹可视化画布
    trail_canvas = None

    print(f"\n当前模拟颜色: {current_color}")
    print("数字键 1-6 切换颜色\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if trail_canvas is None:
            trail_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # 手部检测（只调用一次 _detect，避免时间戳重复）
        timestamp_ms = frame_count * 33  # 30fps ≈ 33ms/frame
        mp_results = tracker._detect(frame, timestamp_ms)
        hand_landmarks = None
        fingertip = None

        if mp_results.hand_landmarks:
            hand_landmarks = mp_results.hand_landmarks[0]
            idx_tip = hand_landmarks[HandLandmarkIndex.INDEX_TIP]
            fingertip = (int(idx_tip.x * w), int(idx_tip.y * h))

        # ---- 手势判断 ----
        pointing = is_index_pointing(hand_landmarks)
        fist = is_fist(hand_landmarks)
        open_hand = is_open_hand(hand_landmarks)

        # ---- 状态机 ----
        gesture_text = ""

        if pointing:
            gesture_text = "DRAWING - 食指绘画中"
            if not is_drawing:
                is_drawing = True
            if fingertip:
                trajectory.append(fingertip)
        elif fist and is_drawing:
            # 握拳 = 结束绘画，触发识别
            gesture_text = "RECOGNIZING..."
            is_drawing = False
            if len(trajectory) >= 5:
                results = recognizer.recognize(trajectory, color=current_color)
                results_text = [
                    f"{i+1}. {r.entity_name} ({r.score:.2f})"
                    for i, r in enumerate(results)
                ]
                result_timer = 120  # 显示约 4 秒（30fps × 4）
                print(f"\n识别结果 ({current_color}):")
                for line in results_text:
                    print(f"  {line}")
            trajectory = []
        elif open_hand:
            # 张手 = 清空
            gesture_text = "CLEARED - 已清空"
            is_drawing = False
            trajectory = []
            results_text = []
            result_timer = 0

        # ---- 绘制 ----
        # 轨迹画布
        if is_drawing and len(trajectory) >= 2:
            for i in range(1, len(trajectory)):
                cv2.line(trail_canvas, trajectory[i - 1], trajectory[i],
                         (0, 255, 255), 3, cv2.LINE_AA)
        elif not is_drawing and not trajectory:
            trail_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # 合并画布到画面
        mask = trail_canvas.astype(bool).any(axis=2)
        frame[mask] = cv2.addWeighted(frame, 0.4, trail_canvas, 0.6, 0)[mask]

        # 当前轨迹绘制
        if len(trajectory) >= 2:
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i],
                         (0, 255, 255), 2, cv2.LINE_AA)

        # 指尖光点
        if fingertip:
            cv2.circle(frame, fingertip, 8, (0, 255, 255), -1)
            cv2.circle(frame, fingertip, 10, (0, 200, 255), 2)

        # ---- UI 信息栏 ----
        # 顶部半透明条
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, f"Color: {current_color}  [1-6 to switch]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, gesture_text,
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 识别结果（右侧显示）
        if result_timer > 0:
            result_timer -= 1
            for i, text in enumerate(results_text):
                y = 25 + i * 25
                cv2.putText(frame, text, (w - 280, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # 底部操作提示
        tips = "Index finger: DRAW  |  Fist: RECOGNIZE  |  Open hand: CLEAR  |  Q: QUIT"
        cv2.putText(frame, tips, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow("Sketch Recognizer Test", frame)

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('1') <= key <= ord('6'):
            current_color_idx = key - ord('1')
            current_color = colors[current_color_idx]
            print(f"切换颜色: {current_color}")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
