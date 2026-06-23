"""
颜色稳定检测器 — 框内取色 + V2 分类 + 3 秒稳定确认

画面中央方框 → V2 全管线 21 色匹配 → 同一颜色持续 3 秒不变化 → 确认。

用法:
    detector = ColorStabilityDetector()
    detector.start()
    while True:
        result = detector.update(frame)
        if result:
            print(f"确认: {result.color_name} ({result.confidence:.3f})")
            break
        detector.draw_roi(frame)
"""

import time
from typing import Optional

import cv2
import numpy as np

from vision.color_detector import ObjectColorDetector, ColorDetectionResult


class ColorStabilityDetector:
    """
    框内取色 + V2 分类 + 3 秒稳定确认

    参数:
        box_size:        方框半边长（像素），默认 120（→ 240×240 框）
        confirm_seconds:  稳定确认秒数，默认 3.0
    """

    def __init__(self, box_size: int = 120, confirm_seconds: float = 3.0):
        self.box_size = box_size
        self.confirm_seconds = confirm_seconds
        self.v2 = ObjectColorDetector()

        # 降低阈值，适配小框内检测
        self.v2.MIN_CLUSTER_RATIO = 0.08
        self.v2.MIN_CONFIDENCE = 0.22

        self._active = False
        self._stable_result: Optional[ColorDetectionResult] = None
        self._stable_since = 0.0
        self._confirmed: Optional[ColorDetectionResult] = None
        self._last_result: Optional[ColorDetectionResult] = None
        self._last_box = (0, 0, 0, 0)

    # ── 公共 API ──

    def start(self) -> None:
        self._active = True
        self._stable_result = None
        self._stable_since = time.time()
        self._last_result = None
        self._confirmed = None

    def stop(self) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def stable_color(self) -> Optional[str]:
        return self._stable_result.color_name if self._stable_result else None

    @property
    def confirmed(self) -> Optional[ColorDetectionResult]:
        return self._confirmed

    @property
    def elapsed(self) -> float:
        if not self._stable_result:
            return 0.0
        return time.time() - self._stable_since

    def _get_box(self, frame: np.ndarray) -> tuple:
        h, w = frame.shape[:2]
        s = self.box_size
        return (w // 2 - s, h // 2 - s, w // 2 + s, h // 2 + s)

    def update(self, frame: np.ndarray) -> Optional[ColorDetectionResult]:
        """
        每帧调用。画面中央方框内取色 → V2 全管线匹配 21 色。
        同一颜色稳定 confirm_seconds 后返回确认结果（仅返回一次，自动 stop）。
        """
        if not self._active or frame is None or frame.size == 0:
            return None

        region = self._get_box(frame)
        self._last_box = region
        result = self.v2.detect(frame, region=region)
        self._last_result = result
        now = time.time()

        if result is None:
            self._stable_result = None
            self._stable_since = now
            return None

        if self._stable_result is None or result.color_name != self._stable_result.color_name:
            self._stable_result = result
            self._stable_since = now
            return None

        if now - self._stable_since < self.confirm_seconds:
            return None

        # 确认！
        self._confirmed = result
        self._active = False
        return result

    def draw_roi(self, frame: np.ndarray) -> None:
        """画中央方框 + 倒计时 / 颜色名"""
        x1, y1, x2, y2 = self._last_box

        if self._confirmed:
            # 已确认：绿色实框 + 颜色名
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, self._confirmed.color_name,
                        (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif self._active and self._stable_result:
            # 有稳定色：黄色框 + 倒计时
            elapsed = time.time() - self._stable_since
            remain = max(0, self.confirm_seconds - elapsed)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, f"{self._stable_result.color_name} {remain:.1f}s",
                        (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif self._active:
            # 检测中：白虚线框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(frame, "R",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 调试 HSV
        if self._stable_result:
            h, s, v = self._stable_result.dominant_hsv
            cv2.putText(frame, f"H:{h} S:{s} V:{v}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
