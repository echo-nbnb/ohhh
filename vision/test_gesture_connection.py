"""
手势连接模块测试
测试手势握拳检测和物块连接功能

使用方法:
    python -m vision.test_gesture_connection
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.gesture_connection import GestureConnection, Module, ConnectionState
from vision.ipcamera import IPCamera


# 配置
CAMERA_URL = "http://192.168.0.111:8080/video"  # IP摄像头地址
WINDOW_NAME = "Gesture Connection Test"


def create_test_modules():
    """创建测试用的物块"""
    return [
        Module("color_red", "color", (200, 200), (100, 100)),
        Module("color_green", "color", (400, 200), (100, 100)),
        Module("object_academy", "object", (600, 200), (100, 100)),
        Module("character_zhangshi", "character", (300, 400), (100, 100)),
        Module("character_zhu", "character", (500, 400), (100, 100)),
    ]


def on_connection(connection_type: str, module_a, module_b):
    """连接完成回调"""
    print(f"\n[连接完成] {module_a.id} ↔ {module_b.id}")
    print(f"         类型: {connection_type}")


def main():
    print("=" * 50)
    print("手势连接测试")
    print("=" * 50)

    # 连接摄像头
    print(f"[1] 连接摄像头: {CAMERA_URL}")
    camera = IPCamera(CAMERA_URL)
    if not camera.connect():
        print("[错误] 摄像头连接失败!")
        return

    w = int(camera._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(camera._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] 摄像头已连接 ({w}x{h})")

    # 初始化手势连接
    print("[2] 初始化手势连接...")
    gc = GestureConnection(canvas_size=(w, h))
    gc.set_modules(create_test_modules())
    gc.set_connection_callback(on_connection)
    print("[OK] 手势连接已就绪")

    # 显示操作说明
    print("\n[操作说明]")
    print("  1. 手悬停在物块上（显示浅蓝色）")
    print("  2. 握拳开始连接（物块变黄色）")
    print("  3. 保持握拳，移动到另一个物块（目标变绿色）")
    print("  4. 手张开完成连接")
    print("\n[按 ESC 退出]")

    # 创建窗口
    cv2.namedWindow(WINDOW_NAME)

    # 主循环
    timestamp_ms = 0
    while True:
        frame = camera.read_frame()
        if frame is None:
            continue

        # 处理帧
        result = gc.process_frame(frame, timestamp_ms)

        # 绘制调试画面
        display = gc.draw_debug_visualization(frame, result)

        # 显示状态信息
        state = result["state"]
        gesture = result.get("gesture", "未知")
        is_fist = result.get("is_fist", False)

        info_text = f"状态: {state} | 手势: {gesture} | 握拳: {is_fist}"
        cv2.putText(display, info_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        timestamp_ms += 33

    # 清理
    print("\n[退出] 清理资源...")
    gc.close()
    camera.release()
    cv2.destroyAllWindows()
    print("[完成]")


if __name__ == "__main__":
    # 检查模型文件
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hand_landmarker.task")
    if not os.path.exists(model_path):
        print(f"[警告] 模型文件不存在: {model_path}")
        print("[提示] 请下载 MediaPipe Hand Landmarker 模型并命名为 hand_landmarker.task")

    main()