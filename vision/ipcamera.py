"""
IP摄像头连接模块
支持通过RTSP/HTTP协议连接手机IP摄像头（如IPCam等APP）
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class IPCamera:
    """IP摄像头连接类"""

    def __init__(self, url: str, timeout: int = 5000, target_width: int = 640, target_height: int = 480):
        """
        初始化IP摄像头连接

        Args:
            url: 摄像头的RTSP或HTTP地址
            timeout: 连接超时时间（毫秒）
            target_width: 目标宽度（降低分辨率）
            target_height: 目标高度（降低分辨率）
        """
        self.url = url
        self.timeout = timeout
        self.target_width = target_width
        self.target_height = target_height
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_connected = False
        self._frame_count = 0

    def connect(self) -> bool:
        """
        连接到IP摄像头

        Returns:
            连接是否成功
        """
        try:
            self._cap = cv2.VideoCapture(self.url)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout)
            # 尝试降低分辨率（不一定所有摄像头都支持）
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            # 测试连接
            if self._cap.isOpened():
                self._is_connected = True
                actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"[IPCam] 分辨率: {actual_w}x{actual_h}")
                return True
            else:
                self._is_connected = False
                return False
        except Exception as e:
            print(f"连接失败: {e}")
            self._is_connected = False
            return False

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._is_connected and self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Optional[np.ndarray]:
        """
        读取一帧图像，自动缩放到目标分辨率

        Returns:
            numpy数组格式的图像（BGR格式），读取失败返回None
        """
        if not self.is_connected():
            return None

        ret, frame = self._cap.read()
        if ret:
            self._frame_count += 1
            # 如果实际分辨率比目标大，缩放
            if frame.shape[1] > self.target_width or frame.shape[0] > self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            return frame
        return None

    def get_frame(self, width: int = 640, height: int = 480) -> Optional[np.ndarray]:
        """
        获取调整大小后的帧

        Args:
            width: 目标宽度
            height: 目标高度

        Returns:
            调整大小后的图像
        """
        frame = self.read_frame()
        if frame is not None:
            return cv2.resize(frame, (width, height))
        return None

    def release(self):
        """释放连接"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class IPCameraManager:
    """IP摄像头管理器，支持多摄像头"""

    def __init__(self):
        self.cameras: dict[str, IPCamera] = {}

    def add_camera(self, name: str, url: str, timeout: int = 5000) -> bool:
        """
        添加一个摄像头

        Args:
            name: 摄像头名称/标识
            url: 摄像头地址
            timeout: 超时时间

        Returns:
            是否添加成功
        """
        camera = IPCamera(url, timeout)
        if camera.connect():
            self.cameras[name] = camera
            return True
        return False

    def get_camera(self, name: str) -> Optional[IPCamera]:
        """获取指定名称的摄像头"""
        return self.cameras.get(name)

    def remove_camera(self, name: str):
        """移除摄像头"""
        if name in self.cameras:
            self.cameras[name].release()
            del self.cameras[name]

    def read_all_frames(self) -> dict[str, Optional[np.ndarray]]:
        """读取所有摄像头的当前帧"""
        return {name: cam.read_frame() for name, cam in self.cameras.items()}

    def release_all(self):
        """释放所有摄像头"""
        for camera in self.cameras.values():
            camera.release()
        self.cameras.clear()


def create_camera_from_url(url: str) -> IPCamera:
    """
    工厂函数：从URL创建摄像头实例

    Args:
        url: 摄像头地址

    Returns:
        IPCamera实例（未连接）
    """
    return IPCamera(url)
