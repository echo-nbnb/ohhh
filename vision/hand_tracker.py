"""
手部位置跟踪模块（展览优化版）
- 四点透视标定（鼠标点击选取）
- 标定数据持久化
- 手部坐标平滑（移动平均）
- 提高的置信度阈值
"""

import cv2
import numpy as np
import os
import json
from typing import Optional, Tuple, List
from collections import deque

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'hand_landmarker.task')
CALIB_FILE = os.path.join(PROJECT_ROOT, 'calibration.json')


class HandTracker:
    """手部关键点跟踪器（展览优化版）"""

    def __init__(self,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7,
                 num_hands: int = 1,
                 output_size: Tuple[int, int] = (1920, 1080)):
        """
        Args:
            min_detection_confidence: 手部检测置信度（展览建议 0.7）
            min_tracking_confidence: 手部跟踪置信度（展览建议 0.7）
            num_hands: 同时检测手数
            output_size: 投影/画布分辨率
        """
        base_options = BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # 透视变换
        self.transform_matrix = None
        self.calibration_points = []      # 摄像头画面中四点 [(x,y),...]
        self.calibration_collecting = False
        self.calibration_order = ["左上", "右上", "右下", "左下"]
        self.output_size = output_size

        # 坐标平滑（移动平均窗口）
        self._smooth_window = 5
        self._palm_history = deque(maxlen=self._smooth_window)
        self._wrist_history = deque(maxlen=self._smooth_window)
        self._fingertips_history = [deque(maxlen=self._smooth_window) for _ in range(5)]

        # 手部检测状态
        self._hand_present = False
        self._consecutive_missing = 0
        self._max_missing = 5  # 连续丢失 N 帧后认为手离开

        # 加载已保存的标定
        self._load_calibration()

    # ================================================================
    # 标定
    # ================================================================

    def start_calibration(self):
        """开始标定模式，清除旧数据"""
        self.calibration_collecting = True
        self.calibration_points = []
        self.transform_matrix = None

    def add_calibration_point(self, x: int, y: int) -> bool:
        """
        添加一个标定点（从鼠标点击回调中调用）

        Returns:
            True 如果 4 个点都已收集完毕
        """
        if not self.calibration_collecting:
            return False

        self.calibration_points.append((x, y))
        idx = len(self.calibration_points)
        print(f"[标定] 点{idx} ({self.calibration_order[idx-1]}): ({x}, {y})")

        if len(self.calibration_points) >= 4:
            self._compute_transform()
            self._save_calibration()
            self.calibration_collecting = False
            return True
        return False

    def reset_calibration(self):
        """重置标定"""
        self.calibration_points = []
        self.transform_matrix = None
        self.calibration_collecting = True
        print("[标定] 已重置，请重新选点")

    def _compute_transform(self):
        """根据 4 个标定点计算透视变换矩阵"""
        if len(self.calibration_points) != 4:
            return

        src = np.array(self.calibration_points, dtype=np.float32)
        dst = np.array([
            [0, 0],                                          # 左上 → 投影 0,0
            [self.output_size[0], 0],                         # 右上
            [self.output_size[0], self.output_size[1]],       # 右下
            [0, self.output_size[1]]                          # 左下
        ], dtype=np.float32)

        self.transform_matrix = cv2.getPerspectiveTransform(src, dst)
        print(f"[标定] 透视变换矩阵已计算")
        print(f"  源点: {self.calibration_points}")
        print(f"  目标分辨率: {self.output_size}")

    def _save_calibration(self):
        """保存标定到文件"""
        data = {
            "calibration_points": self.calibration_points,
            "output_size": list(self.output_size),
            "transform_matrix": self.transform_matrix.tolist() if self.transform_matrix is not None else None
        }
        with open(CALIB_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[标定] 已保存到 {CALIB_FILE}")

    def _load_calibration(self):
        """从文件加载标定"""
        if not os.path.exists(CALIB_FILE):
            return
        try:
            with open(CALIB_FILE, 'r') as f:
                data = json.load(f)
            pts = data.get("calibration_points", [])
            mat = data.get("transform_matrix")
            if len(pts) == 4:
                self.calibration_points = pts
                self.output_size = tuple(data.get("output_size", [1920, 1080]))
                if mat:
                    self.transform_matrix = np.array(mat, dtype=np.float32)
                else:
                    self._compute_transform()
                print(f"[标定] 已从文件加载 ({CALIB_FILE})")
        except Exception as e:
            print(f"[标定] 加载失败: {e}")

    @property
    def is_calibrated(self) -> bool:
        return self.transform_matrix is not None

    def draw_calibration_overlay(self, frame: np.ndarray) -> np.ndarray:
        """在相机画面上绘制标定辅助信息"""
        display = frame.copy()
        h, w = display.shape[:2]

        # 绘制已选点
        for i, (px, py) in enumerate(self.calibration_points):
            cv2.circle(display, (px, py), 12, (0, 255, 0), -1)
            cv2.circle(display, (px, py), 14, (255, 255, 255), 2)
            cv2.putText(display, self.calibration_order[i],
                       (px + 20, py - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 如果已标定，绘制映射网格
        if self.is_calibrated:
            grid_size = 100
            for gx in range(0, self.output_size[0] + 1, grid_size):
                for gy in range(0, self.output_size[1] + 1, grid_size):
                    pt = np.array([[[gx, gy]]], dtype=np.float32)
                    inv = cv2.getPerspectiveTransform(
                        np.array([[0, 0], [self.output_size[0], 0],
                                  [self.output_size[0], self.output_size[1]],
                                  [0, self.output_size[1]]], dtype=np.float32),
                        self.transform_matrix
                    )
                    mapped = cv2.perspectiveTransform(pt, inv)
                    mx, my = int(mapped[0][0][0]), int(mapped[0][0][1])
                    if 0 <= mx < w and 0 <= my < h:
                        cv2.circle(display, (mx, my), 2, (0, 255, 255), -1)

        # 提示
        if self.calibration_collecting:
            idx = len(self.calibration_points)
            tip = f"请点击投影屏幕的{self.calibration_order[idx]}角 (共4点)"
            cv2.putText(display, tip, (20, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif self.is_calibrated:
            cv2.putText(display, "已标定 | 按 R 重新标定",
                       (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "未标定 | 按 C 开始标定",
                       (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return display

    # ================================================================
    # 手部检测
    # ================================================================

    def _detect(self, frame: np.ndarray, timestamp_ms: int = 0):
        """检测手部关键点（内部使用）"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect_for_video(mp_img, timestamp_ms)

    def get_hand_position(self, frame: np.ndarray,
                          timestamp_ms: int = 0) -> Optional[dict]:
        """
        获取手部位置（透视变换 + 平滑后）

        Returns:
            None 如果未检测到手，或
            {
                "landmarks": [(x,y), ...],     # 21个关键点（投影坐标）
                "raw_landmarks": [(x,y), ...],  # 原始像素坐标
                "palm_center": (x, y),          # 手掌中心（投影坐标，已平滑）
                "wrist": (x, y),               # 手腕（投影坐标，已平滑）
                "fingertips": [(x,y)*5],        # 5 指尖（投影坐标，已平滑）
                "hand_detected": bool
            }
        """
        results = self._detect(frame, timestamp_ms)

        if not results.hand_landmarks:
            self._consecutive_missing += 1
            if self._consecutive_missing > self._max_missing:
                self._hand_present = False
            return None

        self._consecutive_missing = 0
        self._hand_present = True
        hand_lm = results.hand_landmarks[0]
        h, w = frame.shape[:2]

        # 原始像素坐标
        raw = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]

        # 透视变换
        if self.is_calibrated:
            pts = np.array(raw, dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(pts, self.transform_matrix)
            landmarks = [(int(p[0][0]), int(p[0][1])) for p in transformed]
        else:
            # 未标定：简单缩放到输出尺寸
            landmarks = [(int(x * self.output_size[0] / w),
                          int(y * self.output_size[1] / h))
                         for (x, y) in raw]

        # 手掌中心 = 21点平均
        palm = (sum(p[0] for p in landmarks) // 21,
                sum(p[1] for p in landmarks) // 21)
        wrist = landmarks[0]
        fingertips = [landmarks[i] for i in (4, 8, 12, 16, 20)]

        # 平滑
        palm = self._smooth_point(self._palm_history, palm)
        wrist = self._smooth_point(self._wrist_history, wrist)
        fingertips = [self._smooth_point(self._fingertips_history[i], ft)
                      for i, ft in enumerate(fingertips)]

        return {
            "landmarks": landmarks,
            "raw_landmarks": raw,
            "palm_center": palm,
            "wrist": wrist,
            "fingertips": fingertips,
            "hand_detected": True
        }

    def _smooth_point(self, history: deque, pt: Tuple[int, int]) -> Tuple[int, int]:
        """移动平均平滑"""
        history.append(pt)
        if len(history) < 2:
            return pt
        avg_x = sum(p[0] for p in history) // len(history)
        avg_y = sum(p[1] for p in history) // len(history)
        return (avg_x, avg_y)

    def get_data_for_unity(self, frame: np.ndarray,
                           timestamp_ms: int = 0) -> Optional[dict]:
        """获取发送给 Unity/前端 的数据格式"""
        hand = self.get_hand_position(frame, timestamp_ms)
        if hand is None:
            return None

        return {
            "type": "hand_tracking",
            "palm_center": hand["palm_center"],
            "wrist": hand["wrist"],
            "landmarks": hand["landmarks"],
            "fingertips": hand["fingertips"],
            "is_calibrated": self.is_calibrated,
            "hand_detected": hand["hand_detected"]
        }

    @property
    def is_hand_present(self) -> bool:
        return self._hand_present

    def close(self):
        self.detector.close()


# ================================================================
# 手部关键点索引 & 骨架连接
# ================================================================

HAND_LANDMARKS = {
    0: "WRIST", 1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
    5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
    9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
    13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
    17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),   # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20), # 小指
    (5, 9), (9, 13), (13, 17),          # 手掌
]
