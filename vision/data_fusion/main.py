#!/usr/bin/env python3
"""
数据融合主程序
整合摄像头、YOLO检测、ESP32 Hub和数据融合

使用方式：
python -m vision.data_fusion.main
"""

import sys
import time
import json
import threading
from typing import Optional

sys.path.insert(0, '.')

from vision.ipcamera import IPCamera
from vision.data_fusion.fuse import DataFusion, FusedModuleData
from vision.data_fusion.esp32_hub import ESP32Hub


class DataFusionServer:
    """
    数据融合服务器
    整合所有数据源，提供统一的融合数据输出
    """

    def __init__(self,
                 camera_url: str = "http://10.54.71.31:8080/video",
                 esp32_port: int = 8888,
                 yolo_model: str = "yolo/yolov8n.pt",
                 max_distance: int = 100):
        """
        Args:
            camera_url: 摄像头地址
            esp32_port: ESP32 Hub端口
            yolo_model: YOLO模型路径
            max_distance: 模块匹配最大距离
        """
        self.camera_url = camera_url
        self.esp32_port = esp32_port
        self.yolo_model = yolo_model
        self.max_distance = max_distance

        # 组件
        self.camera: Optional[IPCamera] = None
        self.esp32_hub: Optional[ESP32Hub] = None
        self.fusion: Optional[DataFusion] = None
        self.yolo_model_handle = None

        # 状态
        self.running = False
        self.frame_count = 0

        # 共享数据
        self.shared_lock = threading.Lock()
        self.latest_yolo_results = []
        self.latest_fused_data = []

    def _init_yolo(self):
        """初始化YOLO模型"""
        try:
            from ultralytics import YOLO
            self.yolo_model_handle = YOLO(self.yolo_model)
            print(f"[OK] YOLO模型已加载: {self.yolo_model}")
            return True
        except Exception as e:
            print(f"[错误] YOLO模型加载失败: {e}")
            return False

    def _init_components(self) -> bool:
        """初始化所有组件"""
        print("=" * 50)
        print("数据融合服务器初始化")
        print("=" * 50)

        # 初始化YOLO
        if not self._init_yolo():
            return False

        # 初始化摄像头
        self.camera = IPCamera(self.camera_url)
        if not self.camera.connect():
            print("[错误] 摄像头连接失败")
            return False
        print(f"[OK] 摄像头已连接: {self.camera_url}")

        # 初始化ESP32 Hub
        self.esp32_hub = ESP32Hub(port=self.esp32_port)
        if not self.esp32_hub.start():
            return False
        print(f"[OK] ESP32 Hub已启动: {self.esp32_port}")

        # 初始化数据融合
        self.fusion = DataFusion(max_distance=self.max_distance)
        print(f"[OK] 数据融合器已就绪 (max_distance={self.max_distance})")

        return True

    def _process_yolo(self, frame) -> list:
        """处理YOLO检测"""
        if self.yolo_model_handle is None:
            return []

        try:
            results = self.yolo_model_handle.track(frame, conf=0.5, persist=True, verbose=False)
            detections = []

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = results[0].names[cls]
                    track_id = int(box.id[0]) if box.id is not None else None

                    detections.append({
                        'class_id': cls,
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        'track_id': track_id,
                        'timestamp': time.time()
                    })

            return detections

        except Exception as e:
            print(f"[错误] YOLO检测失败: {e}")
            return []

    def _process_frame(self, frame):
        """处理单帧"""
        self.frame_count += 1

        # 1. YOLO检测
        yolo_results = self._process_yolo(frame)

        # 2. 接收ESP32数据
        esp32_data = []
        while True:
            data = self.esp32_hub.receive_once()
            if data is None:
                break
            esp32_data.append(data)

        # 3. 更新融合器
        self.fusion.update_yolo_detections(yolo_results)
        self.fusion.update_esp32_modules(esp32_data)

        # 4. 执行融合
        fused_result = self.fusion.fuse()

        # 5. 更新共享数据
        with self.shared_lock:
            self.latest_yolo_results = yolo_results
            self.latest_fused_data = fused_result

        return fused_result

    def get_latest_data(self) -> dict:
        """获取最新融合数据"""
        with self.shared_lock:
            return self.fusion.to_dict() if self.fusion else {}

    def run_loop(self):
        """运行主循环"""
        if not self._init_components():
            return

        self.running = True
        print("\n[运行] 数据融合服务运行中...")
        print("[提示] 按ESC退出，按P打印当前状态\n")

        try:
            while self.running:
                # 读取帧
                frame = self.camera.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 处理帧
                fused = self._process_frame(frame)

                # 定期打印状态
                if self.frame_count % 30 == 0 and fused:
                    print(f"[状态] 帧:{self.frame_count} | 检测:{len(self.latest_yolo_results)} | 融合:{len(fused)}")

                # 简单预览（可注释掉）
                if self.frame_count % 10 == 0:
                    import cv2
                    # 在帧上绘制融合结果
                    display = frame.copy()
                    for det in self.latest_yolo_results:
                        x, y, w, h = det['bbox']
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{det['class_name']}"
                        if det.get('track_id') is not None:
                            label += f" ID:{det['track_id']}"
                        cv2.putText(display, label, (x, y-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("DataFusion Preview", display)

                # 按键处理
                import cv2
                key = cv2.waitKey(33) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('p'):
                    print("\n[打印] 当前融合数据:")
                    print(json.dumps(self.get_latest_data(), indent=2, ensure_ascii=False))

        except KeyboardInterrupt:
            print("\n[退出]")
        finally:
            self.stop()

    def stop(self):
        """停止服务"""
        self.running = False
        if self.camera:
            self.camera.release()
        if self.esp32_hub:
            self.esp32_hub.stop()
        import cv2
        cv2.destroyAllWindows()
        print("[停止] 数据融合服务已停止")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='数据融合服务器')
    parser.add_argument('--camera', type=str, default="http://10.54.71.31:8080/video",
                       help='摄像头地址')
    parser.add_argument('--port', type=int, default=8888,
                       help='ESP32 Hub端口')
    parser.add_argument('--model', type=str, default="yolo/yolov8n.pt",
                       help='YOLO模型路径')
    parser.add_argument('--max-dist', type=int, default=100,
                       help='最大匹配距离')

    args = parser.parse_args()

    server = DataFusionServer(
        camera_url=args.camera,
        esp32_port=args.port,
        yolo_model=args.model,
        max_distance=args.max_dist
    )
    server.run_loop()


if __name__ == "__main__":
    main()
