#!/usr/bin/env python3
"""
手部跟踪服务器
从IP摄像头获取视频 → 检测手部位置 → 发送到手部跟踪窗口
同时支持与Unity通信，发送手部坐标数据

使用方法:
    python hand_server.py
"""

import cv2
import json
import socket
import sys
import time
import numpy as np
import threading

sys.path.insert(0, '.')

from vision.ipcamera import IPCamera
from vision.hand_tracker import HandTracker, HAND_CONNECTIONS


class HandTrackingServer:
    """手部跟踪服务器"""

    def __init__(self, camera_url: str = "http://10.54.71.31:8080/video",
                 unity_port: int = 8889):
        self.camera_url = camera_url
        self.unity_port = unity_port

        self.camera = None
        self.hand_tracker = None
        self.server_socket = None
        self.unity_socket = None
        self.is_running = False

        # 标定状态
        self.calibration_points = []  # 用户点击的四点
        self.is_calibrated = False
        self.calibration_mode = True

        # 显示窗口
        self.window_name = "Hand Tracking"

    def start(self):
        """启动服务器"""
        print("=" * 50)
        print("手部跟踪服务器")
        print("=" * 50)

        # 初始化摄像头
        print(f"[1] 连接摄像头: {self.camera_url}")
        self.camera = IPCamera(self.camera_url)
        if not self.camera.connect():
            print("[错误] 摄像头连接失败!")
            return
        print("[OK] 摄像头已连接")

        # 初始化手部检测
        print("[2] 初始化手部检测...")
        try:
            self.hand_tracker = HandTracker()
            print("[OK] 手部检测已就绪")
        except Exception as e:
            print(f"[错误] 手部检测初始化失败: {e}")
            return

        # 启动Unity连接监听线程
        print(f"[3] 启动Unity监听端口 {self.unity_port}...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.unity_port))
        self.server_socket.listen(1)
        accept_thread = threading.Thread(target=self.accept_loop, daemon=True)
        accept_thread.start()
        print(f"[OK] 等待Unity连接...")

        # 创建显示窗口
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse_click)

        self.is_running = True
        print("\n[运行] 按ESC退出")
        print("[提示] 在画面中点击4个点进行标定（Unity画布四角的对应位置）")

        self.run_loop()

    def accept_loop(self):
        """接受Unity连接"""
        while self.is_running:
            try:
                self.server_socket.settimeout(1.0)
                try:
                    client_socket, addr = self.server_socket.accept()
                except socket.timeout:
                    continue
                print(f"[Unity] 已连接: {addr}")
                self.unity_socket = client_socket
                self.send_connected_message()
            except Exception as e:
                if self.is_running:
                    print(f"[Unity] accept错误: {e}")

    def send_connected_message(self):
        """发送连接确认消息"""
        try:
            msg = json.dumps({"type": "connected", "message": "hand_tracking_server_ready"}, ensure_ascii=False) + "\n"
            self.unity_socket.sendall(msg.encode('utf-8'))
            print("[Unity] 连接确认已发送")

            # 测试数据：发送一条假的手部数据
            test_data = {
                "type": "hand_tracking",
                "palm_center": [960, 540],
                "wrist": [960, 500],
                "landmarks": [i * 45 for i in range(42)],  # [0,0, 45,0, 90,0, ...]
                "fingertips": [960, 300, 1000, 300, 1040, 300, 1080, 300, 920, 300]
            }
            test_msg = json.dumps(test_data, ensure_ascii=False) + "\n"
            self.unity_socket.sendall(test_msg.encode('utf-8'))
            print("[Unity] 测试手部数据已发送")
        except Exception as e:
            print(f"[Unity] 发送失败: {e}")

    def on_mouse_click(self, event, x, y, flags, param):
        """鼠标点击回调，用于标定"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.calibration_mode and len(self.calibration_points) < 4:
                self.calibration_points.append((x, y))
                print(f"[标定] 点{len(self.calibration_points)}: ({x}, {y})")

                if len(self.calibration_points) == 4:
                    print("[标定] 开始标定...")
                    self.is_calibrated = True
                    self.calibration_mode = False

    def run_loop(self):
        """主循环"""
        timestamp_ms = 0

        while self.is_running:
            frame = self.camera.read_frame()
            if frame is None:
                continue

            hand_data = self.hand_tracker.get_data_for_unity(frame, timestamp_ms)

            display = frame.copy()

            if hand_data:
                for i, (x, y) in enumerate(hand_data["landmarks"]):
                    color = (0, 255, 0) if i % 4 == 0 else (0, 200, 0)
                    cv2.circle(display, (x, y), 3, color, -1)

                for start_idx, end_idx in HAND_CONNECTIONS:
                    pt1 = hand_data["landmarks"][start_idx]
                    pt2 = hand_data["landmarks"][end_idx]
                    cv2.line(display, pt1, pt2, (0, 255, 0), 1)

                palm_x, palm_y = hand_data["palm_center"]
                cv2.circle(display, (palm_x, palm_y), 10, (0, 255, 255), -1)

                fingertips = hand_data["fingertips"]
                wrist = hand_data["wrist"]
                contour_points = [wrist] + fingertips + [wrist]
                for i in range(len(contour_points) - 1):
                    cv2.line(display, contour_points[i], contour_points[i + 1], (255, 255, 0), 2)

                if self.unity_socket:
                    self.send_to_unity(hand_data)

            for i, (x, y) in enumerate(self.calibration_points):
                cv2.circle(display, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(display, str(i + 1), (x + 15, y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            status = "已标定" if self.is_calibrated else f"待标定 ({len(self.calibration_points)}/4)"
            cv2.putText(display, f"状态: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if self.unity_socket:
                cv2.putText(display, "Unity: 已连接", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Unity: 未连接", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('r'):
                self.calibration_points = []
                self.is_calibrated = False
                self.calibration_mode = True
                print("[标定] 已重置")

            timestamp_ms += 33

        self.cleanup()

    def send_to_unity(self, hand_data: dict):
        """发送数据到Unity（使用扁平数组，JsonUtility可解析）"""
        if not self.unity_socket:
            return
        try:
            # 扁平化landmarks: [x0,y0,x1,y1,...,x20,y20]
            landmarks_flat = []
            for x, y in hand_data["landmarks"]:
                landmarks_flat.extend([x, y])

            # 扁平化fingertips: [x0,y0,x1,y1,x2,y2,x3,y3,x4,y4]
            fingertips_flat = []
            for x, y in hand_data["fingertips"]:
                fingertips_flat.extend([x, y])

            data = {
                "type": "hand_tracking",
                "palm_center": hand_data["palm_center"],
                "wrist": hand_data["wrist"],
                "landmarks": landmarks_flat,
                "fingertips": fingertips_flat
            }
            msg = json.dumps(data, ensure_ascii=False) + "\n"
            self.unity_socket.sendall(msg.encode('utf-8'))
        except Exception as e:
            print(f"[Unity] 发送失败: {e}")
            self.unity_socket = None

    def cleanup(self):
        """清理"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        if self.hand_tracker:
            self.hand_tracker.close()
        if self.unity_socket:
            self.unity_socket.close()
        if self.server_socket:
            self.server_socket.close()
        cv2.destroyAllWindows()
        print("[退出] 服务器已停止")


def main():
    camera_url = sys.argv[1] if len(sys.argv) > 1 else None
    if camera_url is None:
        try:
            from config_ipcam import CAMERA_URL
            camera_url = CAMERA_URL
        except ImportError:
            camera_url = "http://10.54.71.31:8080/video"

    server = HandTrackingServer(camera_url=camera_url)

    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[中断] 收到Ctrl+C")
        server.cleanup()


if __name__ == "__main__":
    main()
