"""
颜色检测器 — 第一幕"择色"

支持两种检测模式：
1. 物件颜色检测：分析画面中物件的主色
2. 衣物颜色检测：调用 webcam_color_detector.py（电脑端摄像头）

检测管线:
    摄像头帧 → 物件区域检测/直接分析
                              ↓
                    HSV颜色统计 → 主色提取
                              ↓
                    六色基调匹配 → color_type + confidence

使用方式:
    # 物件颜色检测
    detector = ObjectColorDetector()
    color_type, confidence = detector.detect_object_color(frame, region=None)

    # 衣物颜色检测
    from vision.webcam_color_detector import WebcamColorDetector
    webcam_detector = WebcamColorDetector()
    color_type, confidence = webcam_detector.detect(webcam_frame)
"""

import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger("ColorDetector")


# 六色HSV中心值
COLOR_HEX_VALUES = {
    "岳麓绿": {
        "hsv_center": (60, 120, 110),
        "hsv_range": ((35, 40, 20), (85, 255, 200))
    },
    "书院红": {
        "hsv_center": (8, 140, 110),
        "hsv_range": ((0, 40, 20), (15, 255, 200))
    },
    "西迁黄": {
        "hsv_center": (28, 160, 180),
        "hsv_range": ((15, 40, 40), (40, 255, 255))
    },
    "湘江蓝": {
        "hsv_center": (115, 130, 110),
        "hsv_range": ((95, 40, 20), (135, 255, 200))
    },
    "校徽金": {
        "hsv_center": (22, 180, 200),
        "hsv_range": ((10, 60, 60), (35, 200, 255))
    },
    "墨色": {
        "hsv_center": (0, 10, 30),
        "hsv_range": ((0, 0, 0), (180, 50, 60))
    }
}


@dataclass
class ColorDetectionResult:
    """颜色检测结果"""
    color_type: str          # 颜色名称
    color_name: str           # 中文颜色名
    confidence: float         # 匹配置信度 [0, 1]
    source: str              # 来源：object / clothing / fallback
    dominant_hsv: Tuple[int, int, int]  # 提取到的HSV主色


class ObjectColorDetector:
    """物件颜色检测器"""

    def __init__(self):
        pass

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """白平衡 + 直方图均衡，适应展厅光照"""
        # LAB 色彩空间的亮度均衡
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def detect(self, frame: np.ndarray,
               region: Optional[Tuple[int, int, int, int]] = None) -> Optional[ColorDetectionResult]:
        """
        检测物件颜色

        Args:
            frame: BGR格式图像
            region: 可选，检测区域 (x1, y1, x2, y2)

        Returns:
            ColorDetectionResult 或 None
        """
        if frame is None or frame.size == 0:
            return None

        # 裁剪区域
        if region is not None:
            x1, y1, x2, y2 = region
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame

        # 白平衡校正 + 直方图均衡（适应展厅复杂光照）
        roi = self._preprocess(roi)

        # 转换为HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 计算2D直方图 (H-S)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        # 找峰值
        peak_idx = np.unravel_index(np.argmax(hist), hist.shape)
        h_peak = peak_idx[0]
        s_peak = peak_idx[1]

        # 估计V为加权平均
        v_vals = hsv[:, :, 2]
        v_weights = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        v_peak = int(np.argmax(v_weights))

        dominant_hsv = (h_peak, s_peak, v_peak)

        # 与六色匹配
        color_name, confidence = self._match_to_color(dominant_hsv)

        if color_name is None:
            return None

        return ColorDetectionResult(
            color_type=color_name,
            color_name=color_name,
            confidence=confidence,
            source="object",
            dominant_hsv=dominant_hsv
        )

    def _match_to_color(self, hsv: Tuple[int, int, int]) -> Tuple[Optional[str], float]:
        """
        将HSV颜色与六色匹配

        Returns:
            (颜色名称, 置信度)
        """
        h, s, v = hsv

        best_color = None
        best_score = 0.0

        for color_name, color_info in COLOR_HEX_VALUES.items():
            hsv_center = color_info["hsv_center"]
            hsv_range = color_info["hsv_range"]

            # 计算距离（考虑HSV环特性）
            h_diff = abs(h - hsv_center[0])
            if h_diff > 90:
                h_diff = 180 - h_diff
            h_dist = h_diff / 90.0

            s_dist = abs(s - hsv_center[1]) / 128.0
            v_dist = abs(v - hsv_center[2]) / 128.0

            # 综合距离
            total_dist = h_dist * 0.4 + s_dist * 0.3 + v_dist * 0.3
            score = max(0, 1 - total_dist)

            # 检查是否在范围内
            min_vals, max_vals = hsv_range
            h_range = (min_vals[0], max_vals[0])
            s_range = (min_vals[1], max_vals[1])
            v_range = (min_vals[2], max_vals[2])
            in_range = (h_range[0] <= h <= h_range[1] and
                        s_range[0] <= s <= s_range[1] and
                        v_range[0] <= v <= v_range[1])

            if in_range:
                score = min(1.0, score + 0.1)  # 在范围内小幅加分，避免误匹配

            if score > best_score:
                best_score = score
                best_color = color_name

        if best_score < 0.25:
            return None, 0.0

        return best_color, round(best_score, 3)

    def detect_dominant_region(self, frame: np.ndarray,
                                min_saturation: int = 30,
                                min_area: int = 1000) -> Optional[Tuple[int, int, int, int]]:
        """
        检测画面中颜色最显著的区域

        Args:
            frame: BGR格式图像
            min_saturation: 最小饱和度阈值
            min_area: 最小区域面积

        Returns:
            (x1, y1, x2, y2) 或 None
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 饱和度掩码
        saturation_mask = hsv[:, :, 1] > min_saturation

        # 创建掩码
        mask = saturation_mask.astype(np.uint8) * 255

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 找最大轮廓
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < min_area:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        return (x, y, x + w, y + h)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("物件颜色检测器自测")
    print("=" * 50)

    detector = ObjectColorDetector()

    print("\n开启摄像头测试（按 Q 退出）...")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测显著区域
        region = detector.detect_dominant_region(frame)

        # 检测该区域颜色
        result = detector.detect(frame, region)

        # 可视化
        display = frame.copy()

        if region:
            x1, y1, x2, y2 = region
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if result:
            cv2.putText(display, f"{result.color_name} ({result.confidence:.2f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display, "未匹配到六色",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Object Color Detection", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n自测结束")
