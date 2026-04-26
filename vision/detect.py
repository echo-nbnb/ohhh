"""
视觉检测模块
支持从IP摄像头实时获取视频流并进行视觉识别
"""

import cv2
import numpy as np
from typing import Optional, Callable, Tuple
from .ipcamera import IPCamera, IPCameraManager


class VisionDetector:
    """视觉检测器基类"""

    def __init__(self, camera: IPCamera):
        self.camera = camera

    def detect(self, frame: np.ndarray) -> any:
        """
        在帧上执行检测

        Args:
            frame: 输入图像（BGR格式）

        Returns:
            检测结果
        """
        raise NotImplementedError


class FrameProcessor:
    """帧处理器"""

    @staticmethod
    def resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """调整图像大小"""
        return cv2.resize(frame, (width, height))

    @staticmethod
    def convert_to_rgb(frame: np.ndarray) -> np.ndarray:
        """BGR转RGB"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def convert_to_gray(frame: np.ndarray) -> np.ndarray:
        """转灰度图"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def draw_text(frame: np.ndarray, text: str, position: Tuple[int, int],
                  font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """在图像上绘制文本"""
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness)
        return frame


class StreamViewer:
    """实时流查看器"""

    def __init__(self, window_name: str = "IP Camera Stream"):
        self.window_name = window_name

    def show_frame(self, frame: np.ndarray) -> bool:
        """
        显示帧

        Args:
            frame: 要显示的图像

        Returns:
            是否继续显示（按ESC返回False）
        """
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != 27  # ESC键退出

    def run(self, camera: IPCamera, resize: Optional[Tuple[int, int]] = None):
        """
        运行实时显示循环

        Args:
            camera: IPCamera实例
            resize: 可选的调整大小参数 (width, height)
        """
        print(f"开始显示视频流，按ESC键退出...")

        while camera.is_connected():
            frame = camera.read_frame()
            if frame is None:
                continue

            if resize:
                frame = cv2.resize(frame, resize)

            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cv2.destroyAllWindows()


def test_camera_connection(url: str) -> bool:
    """
    测试摄像头连接

    Args:
        url: 摄像头地址

    Returns:
        连接是否成功
    """
    print(f"正在连接摄像头: {url}")

    camera = IPCamera(url)
    success = camera.connect()

    if success:
        print("连接成功!")
        # 读取一帧验证
        frame = camera.read_frame()
        if frame is not None:
            print(f"成功读取帧，尺寸: {frame.shape}")
        camera.release()
    else:
        print("连接失败!")

    return success


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        camera_url = sys.argv[1]
    else:
        # 默认地址，IPCam APP常用的地址格式
        camera_url = "http://192.168.1.100:8080/video"

    test_camera_connection(camera_url)
