#!/usr/bin/env python3
"""
手部跟踪服务器 - 双线程版
摄像头读取和手部检测分开，避免检测阻塞读取
"""

import cv2
import json
import socket
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, '.')
from vision.ipcamera import IPCamera
from vision.hand_tracker import HandTracker, HAND_CONNECTIONS

print("=" * 50)
print("手部跟踪服务器（双线程版）")
print("=" * 50)

# 使用高分辨率
camera = IPCamera("http://10.54.71.31:8080/video", target_width=1920, target_height=1080)
if not camera.connect():
    print("[错误] 摄像头连接失败")
    sys.exit(1)
print("[OK] 摄像头已连接")

try:
    hand_tracker = HandTracker()
    print("[OK] 手部检测已就绪")
except Exception as e:
    print(f"[错误] 手部检测初始化失败: {e}")
    sys.exit(1)

# 共享状态：检测线程写入，主线程读取
shared = SimpleNamespace()
shared.lock = threading.Lock()
shared.latest_frame = None
shared.latest_hand_data = None
shared.running = True

def detection_loop():
    """后台线程：不断取最新帧做手部检测"""
    frame_count = 0
    while shared.running:
        with shared.lock:
            frame = shared.latest_frame
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        hand_data = hand_tracker.get_data_for_unity(frame, frame_count * 33)

        with shared.lock:
            shared.latest_hand_data = hand_data

        # 检测本身是瓶颈，不要太密集，留点时间给摄像头读取
        # 降低检测频率到约15fps，但保证最新帧优先
        time.sleep(0.02)

t_detect = threading.Thread(target=detection_loop, daemon=True)
t_detect.start()
print("[OK] 检测线程已启动")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 8889))
server.listen(1)
print("[OK] 监听端口 8889")
print("[运行] 按ESC退出")
print()

unity_socket = None
frame_count = 0
last_sent_time = 0

while True:
    # accept
    if unity_socket is None:
        print("[Server] 等待Unity连接...")
        unity_socket, addr = server.accept()
        unity_socket.settimeout(0.5)
        unity_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[Server] Unity已连接: {addr}")

        test_data = {
            "type": "hand_tracking",
            "palm_center": [320, 240],
            "wrist": [320, 220],
            "contour": [320, 220, 350, 180, 340, 180, 350, 180, 360, 180, 290, 180, 320, 220],
            "bounding_box": [250, 150, 390, 330]
        }
        try:
            unity_socket.send((json.dumps(test_data) + "\n").encode('utf-8'))
            print("[Server] 测试数据已发送")
        except Exception as e:
            print(f"[Server] 发送测试数据失败: {e}")
            unity_socket.close()
            unity_socket = None
            continue

    # 主线程：不断读取最新帧（不会被检测阻塞）
    frame = camera.read_frame()
    if frame is not None:
        with shared.lock:
            shared.latest_frame = frame

    frame_count += 1

    # 从共享状态读取检测结果
    hand_data = None
    with shared.lock:
        hand_data = shared.latest_hand_data

    # 发送数据（限制发送频率，约30fps）
    current_time = time.time()
    if unity_socket and hand_data and (current_time - last_sent_time) > 0.033:
        last_sent_time = current_time
        try:
            fingertips = hand_data["fingertips"]
            wrist = hand_data["wrist"]

            contour = [wrist[0], wrist[1]]
            for tip in fingertips:
                contour.append(tip[0])
                contour.append(tip[1])
            contour.append(wrist[0])
            contour.append(wrist[1])

            all_x = [wrist[0]] + [t[0] for t in fingertips]
            all_y = [wrist[1]] + [t[1] for t in fingertips]
            x_min = max(0, min(all_x) - 30)
            y_min = max(0, min(all_y) - 30)
            x_max = min(frame.shape[1], max(all_x) + 30)
            y_max = min(frame.shape[0], max(all_y) + 30)

            data = {
                "type": "hand_tracking",
                "palm_center": hand_data["palm_center"],
                "wrist": wrist,
                "contour": contour,
                "bounding_box": [int(x_min), int(y_min), int(x_max), int(y_max)]
            }
            msg = (json.dumps(data, ensure_ascii=False) + "\n").encode('utf-8')
            unity_socket.send(msg)
            if frame_count % 60 == 0:
                print(f"[Server] 帧{frame_count} palm={hand_data['palm_center']} hand检测正常")
        except Exception as e:
            print(f"[Server] 发送失败: {e}")
            unity_socket.close()
            unity_socket = None

    # 预览窗口（降刷新率，不影响检测）
    if frame is not None and frame_count % 10 == 0:
        display = frame.copy()
        if hand_data:
            for i, (x, y) in enumerate(hand_data["landmarks"]):
                color = (0, 255, 0) if i % 4 == 0 else (0, 200, 0)
                cv2.circle(display, (x, y), 3, color, -1)
            for s, e in HAND_CONNECTIONS:
                cv2.line(display, hand_data["landmarks"][s], hand_data["landmarks"][e], (0, 255, 0), 1)
            cv2.circle(display, hand_data["palm_center"], 10, (0, 255, 255), -1)
            all_x = [hand_data["wrist"][0]] + [t[0] for t in hand_data["fingertips"]]
            all_y = [hand_data["wrist"][1]] + [t[1] for t in hand_data["fingertips"]]
            x_min = max(0, min(all_x) - 30)
            y_min = max(0, min(all_y) - 30)
            x_max = min(frame.shape[1], max(all_x) + 30)
            y_max = min(frame.shape[0], max(all_y) + 30)
            cv2.rectangle(display, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            cv2.putText(display, f"F:{frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(display, f"F:{frame_count} NoHand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if unity_socket:
            cv2.putText(display, "Unity: Connected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Unity: Disconnected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Hand Tracking", display)

    key = cv2.waitKey(33) & 0xFF  # 限制约30fps，减少CPU空转
    if key == 27:
        break

# 清理
shared.running = False
camera.release()
hand_tracker.close()
if unity_socket:
    unity_socket.close()
server.close()
cv2.destroyAllWindows()
print("[退出]")
