#!/usr/bin/env python3
"""
YOLO 检测服务器 - 双模块测试版
从 IP 摄像头读取画面，用 YOLO 检测两个相同的物块，
将检测结果通过 TCP 发送给 Unity。

物块统一标注为 "module"，服务器只关心有无检测到，不区分具体类别。
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

# ============ 配置 =============
CAMERA_URL = "http://10.54.71.31:8080/video"
MODEL_PATH = "../yolo/runs/detect/train_20260427_213530/best.pt"  # 训练好的模型
CONF_THRESHOLD = 0.4               # 置信度阈值（调低点提高召回）
SERVER_PORT = 8890                 # Unity 连接端口
# ==================================

from ultralytics import YOLO

# 共享状态
shared = SimpleNamespace()
shared.lock = threading.Lock()
shared.latest_frame = None
shared.latest_results = []         # 当前帧 YOLO 检测结果
shared.running = True


def load_trained_model(model_path):
    """加载自定义格式的模型权重"""
    import torch
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state' in checkpoint:
        # 自定义训练格式：提取 state_dict
        state_dict = checkpoint['model_state']
        # 用预训练模型作为骨架
        model = YOLO('yolov8n.pt')
        model.model.load_state_dict(state_dict, strict=False)
        return model
    else:
        # 标准 YOLO 格式
        return YOLO(model_path)


def detection_loop():
    """后台线程：不断取最新帧做 YOLO 检测"""
    print(f"[Detect] 加载模型 {MODEL_PATH}...")
    try:
        model = load_trained_model(MODEL_PATH)
    except Exception as e:
        print(f"[Detect] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        shared.running = False
        return
    print("[Detect] 模型加载成功")

    frame_count = 0
    while shared.running:
        with shared.lock:
            frame = shared.latest_frame
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

        # 提取检测结果
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "conf": round(conf, 3)
                })

        with shared.lock:
            shared.latest_results = detections

        # 控制检测频率，避免过载
        time.sleep(0.03)

    model = None  # 释放显存


def main():
    print("=" * 50)
    print("YOLO 检测服务器（双模块测试版）")
    print("=" * 50)

    # 连接摄像头
    camera = IPCamera(CAMERA_URL, target_width=1920, target_height=1080)
    if not camera.connect():
        print("[错误] 摄像头连接失败")
        sys.exit(1)
    print("[OK] 摄像头已连接")

    # 启动检测线程
    t_detect = threading.Thread(target=detection_loop, daemon=True)
    t_detect.start()

    # 启动 TCP 服务器
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', SERVER_PORT))
    server.listen(1)
    print(f"[OK] 监听端口 {SERVER_PORT}")
    print("[运行] 按 ESC 退出")
    print()

    unity_socket = None
    frame_count = 0
    last_send_time = 0

    while True:
        # accept
        if unity_socket is None:
            print("[Server] 等待 Unity 连接...")
            try:
                unity_socket, addr = server.accept()
                unity_socket.settimeout(0.5)
                unity_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[Server] Unity 已连接: {addr}")
            except Exception as e:
                print(f"[Server] accept 失败: {e}")
                continue

        # 主线程：不断读取最新帧
        frame = camera.read_frame()
        if frame is not None:
            with shared.lock:
                shared.latest_frame = frame

        frame_count += 1

        # 读取检测结果
        detections = []
        with shared.lock:
            detections = list(shared.latest_results)

        # 发送数据（约30fps）
        current_time = time.time()
        if unity_socket and (current_time - last_send_time) > 0.033:
            last_send_time = current_time
            try:
                data = {
                    "type": "yolo_detection",
                    "count": len(detections),
                    "detections": detections
                }
                msg = (json.dumps(data, ensure_ascii=False) + "\n").encode('utf-8')
                unity_socket.send(msg)

                if frame_count % 60 == 0:
                    print(f"[Server] 帧{frame_count} 检测到 {len(detections)} 个物块")
            except Exception as e:
                print(f"[Server] 发送失败: {e}")
                try:
                    unity_socket.close()
                except:
                    pass
                unity_socket = None

        # 预览窗口（降刷新率）
        if frame is not None and frame_count % 5 == 0:
            display = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                conf = det["conf"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"module {conf:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display, f"F:{frame_count} Det:{len(detections)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            status = "Unity: Connected" if unity_socket else "Unity: Disconnected"
            color = (0, 255, 0) if unity_socket else (0, 0, 255)
            cv2.putText(display, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("YOLO Detection", display)

        key = cv2.waitKey(33) & 0xFF
        if key == 27:
            break

    # 清理
    shared.running = False
    camera.release()
    if unity_socket:
        unity_socket.close()
    server.close()
    cv2.destroyAllWindows()
    print("[退出]")


if __name__ == "__main__":
    main()
