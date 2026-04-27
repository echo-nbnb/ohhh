"""
手部位置跟踪模块
检测手部21点关键点，通过透视变换映射到Unity画布坐标
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple, List
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode

# 模型文件路径（相对于项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'hand_landmarker.task')


class HandTracker:
    """手部关键点跟踪器"""

    def __init__(self):
        # 初始化MediaPipe Hands
        base_options = BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # 透视变换矩阵（待标定）
        self.transform_matrix = None
        self.calibration_points = []  # 标定用的四点
        self.output_size = (1920, 1080)  # Unity画布尺寸

    def calibrate(self, frame: np.ndarray, calibration_points: List[Tuple[int, int]]):
        """
        使用四点标定计算透视变换矩阵

        Args:
            frame: 当前帧
            calibration_points: 标定点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                             分别是：左上、右上、右下、左下
        """
        if len(calibration_points) != 4:
            return False

        # 在画面中检测手部，获取手腕位置作为参考
        results = self._detect(frame)
        if results.hand_landmarks:
            wrist = results.hand_landmarks[0][0]
            h, w = frame.shape[:2]
            wrist_pos = (int(wrist.x * w), int(wrist.y * h))

            # 用手腕位置作为第四个点（实际应用中应该用标定物）
            # 这里简化处理：假设标定点就是Unity画布的四个角
            src_points = np.array(calibration_points, dtype=np.float32)

            # Unity画布的四个角（目标点）
            dst_points = np.array([
                [0, 0],                          # 左上
                [self.output_size[0], 0],          # 右上
                [self.output_size[0], self.output_size[1]],  # 右下
                [0, self.output_size[1]]           # 左下
            ], dtype=np.float32)

            # 计算透视变换矩阵
            self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            self.calibration_points = calibration_points
            return True

        return False

    def _detect(self, frame: np.ndarray, timestamp_ms: int = 0):
        """检测手部关键点"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def get_hand_position(self, frame: np.ndarray, timestamp_ms: int = 0) -> Optional[dict]:
        """
        获取手部位置（透视变换后）

        Returns:
            {
                "landmarks": [(x, y), ...],  # 21个关键点Unity坐标
                "palm_center": (x, y),       # 手掌中心
                "wrist": (x, y),            # 手腕
            }
        """
        results = self._detect(frame, timestamp_ms)

        if not results.hand_landmarks:
            return None

        hand_landmarks = results.hand_landmarks[0]
        h, w = frame.shape[:2]

        # 转换为像素坐标
        pixel_landmarks = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pixel_landmarks.append((x, y))

        # 如果已标定，进行透视变换
        if self.transform_matrix is not None:
            # 批量变换关键点
            points = np.array(pixel_landmarks, dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(points, self.transform_matrix)
            unity_landmarks = [(int(p[0][0]), int(p[0][1])) for p in transformed]
        else:
            # 未标定，返回原像素坐标（需要手动缩放）
            unity_landmarks = pixel_landmarks

        # 计算手掌中心（所有点的平均）
        palm_x = sum(p[0] for p in unity_landmarks) // 21
        palm_y = sum(p[1] for p in unity_landmarks) // 21

        return {
            "landmarks": unity_landmarks,
            "palm_center": (palm_x, palm_y),
            "wrist": unity_landmarks[0],
            "raw_pixel": pixel_landmarks
        }

    def get_data_for_unity(self, frame: np.ndarray, timestamp_ms: int = 0) -> Optional[dict]:
        """
        获取发送给Unity的数据格式
        """
        hand_data = self.get_hand_position(frame, timestamp_ms)
        if hand_data is None:
            return None

        return {
            "type": "hand_tracking",
            "palm_center": hand_data["palm_center"],
            "wrist": hand_data["wrist"],
            "landmarks": hand_data["landmarks"],
            # 简化的5个指尖坐标（用于连线显示）
            "fingertips": [
                hand_data["landmarks"][4],   # 拇指
                hand_data["landmarks"][8],   # 食指
                hand_data["landmarks"][12],  # 中指
                hand_data["landmarks"][16],  # 无名指
                hand_data["landmarks"][20],  # 小指
            ]
        }

    def close(self):
        """关闭检测器"""
        self.detector.close()


# 手部21点索引
HAND_LANDMARKS = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_MCP",
    6: "INDEX_PIP",
    7: "INDEX_DIP",
    8: "INDEX_TIP",
    9: "MIDDLE_MCP",
    10: "MIDDLE_PIP",
    11: "MIDDLE_DIP",
    12: "MIDDLE_TIP",
    13: "RING_MCP",
    14: "RING_PIP",
    15: "RING_DIP",
    16: "RING_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}

# 手部骨架连接
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
    (5, 9), (9, 13), (13, 17),  # 手掌
]
