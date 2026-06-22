"""
颜色检测器 — 第一幕"择色"

支持两种检测模式：
1. 物件颜色检测：分析画面中物件的主色
2. 衣物颜色检测：调用 webcam_color_detector.py（电脑端摄像头）

检测管线（V2 优化）:
    摄像头帧 → 灰度世界白平衡 + 高斯模糊
              → 饱和度/亮度双阈值掩码（排除灰白/阴影/高光）
              → 连通域分析，选中央物品
              → Lab 空间 K-Means 提取主色簇
              → 先判 7 大色系 → 再匹配 3 种文化颜色
              → 不确定时返回 None，由上层随机保底

使用方式:
    # 物件颜色检测
    detector = ObjectColorDetector()
    result = detector.detect(frame, region=None)
    # result.color_name, result.confidence, ...

    # 检测显著区域
    region = detector.detect_dominant_region(frame)
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("ColorDetector")


# ── 21色 HSV 配置（与前端 colors.js 对齐，三组各7色）────────────────────
# 红色系 lower hue 改为 170 以支持跨 0°/179° 的正红色

COLOR_HEX_VALUES: Dict[str, Dict[str, Any]] = {
    # ── 第一组 · 亮色系 ──
    "朱红": {"hsv_center": (0, 253, 221), "hsv_range": ((170, 80, 45), (10, 255, 255))},
    "灯橙": {"hsv_center": (11, 255, 254), "hsv_range": ((4, 70, 55), (18, 255, 255))},
    "梨黄": {"hsv_center": (27, 187, 240), "hsv_range": ((21, 55, 55), (35, 255, 255))},
    "叶绿": {"hsv_center": (50, 210, 255), "hsv_range": ((39, 45, 40), (62, 255, 255))},
    "瓷青": {"hsv_center": (87, 218, 255), "hsv_range": ((76, 40, 40), (98, 255, 255))},
    "海蓝": {"hsv_center": (109, 224, 255), "hsv_range": ((98, 45, 40), (119, 255, 255))},
    "烟紫": {"hsv_center": (133, 210, 235), "hsv_range": ((123, 40, 40), (145, 255, 255))},
    # ── 第二组 · 浓色系 ──
    "枫红": {"hsv_center": (0, 255, 196), "hsv_range": ((170, 80, 35), (10, 255, 245))},
    "暖橙": {"hsv_center": (18, 205, 255), "hsv_range": ((10, 60, 55), (24, 255, 255))},
    "藤黄": {"hsv_center": (26, 255, 246), "hsv_range": ((20, 70, 55), (34, 255, 255))},
    "玉绿": {"hsv_center": (61, 255, 235), "hsv_range": ((50, 45, 40), (73, 255, 255))},
    "石青": {"hsv_center": (79, 195, 253), "hsv_range": ((68, 40, 40), (91, 255, 255))},
    "澄蓝": {"hsv_center": (115, 219, 235), "hsv_range": ((104, 45, 35), (126, 255, 255))},
    "影紫": {"hsv_center": (142, 205, 175), "hsv_range": ((130, 40, 30), (154, 255, 245))},
    # ── 第三组 · 柔色系 ──
    "桃红": {"hsv_center": (174, 190, 255), "hsv_range": ((160, 60, 70), (10, 255, 255))},
    "夕橙": {"hsv_center": (14, 230, 225), "hsv_range": ((7, 55, 50), (23, 255, 255))},
    "桂黄": {"hsv_center": (30, 210, 215), "hsv_range": ((23, 50, 45), (38, 255, 255))},
    "茶绿": {"hsv_center": (70, 150, 180), "hsv_range": ((58, 35, 35), (82, 255, 255))},
    "湖青": {"hsv_center": (91, 160, 220), "hsv_range": ((80, 35, 40), (102, 255, 255))},
    "沧蓝": {"hsv_center": (111, 150, 175), "hsv_range": ((100, 30, 30), (125, 255, 255))},
    "黛紫": {"hsv_center": (151, 150, 145), "hsv_range": ((138, 30, 25), (165, 255, 235))},
}


# ── 七大基础色系 → 文化颜色 ──

BASE_FAMILIES: Dict[str, Dict[str, Any]] = {
    "红": {"hue_center": 0,   "colors": ["朱红", "枫红", "桃红"]},
    "橙": {"hue_center": 15,  "colors": ["灯橙", "暖橙", "夕橙"]},
    "黄": {"hue_center": 29,  "colors": ["梨黄", "藤黄", "桂黄"]},
    "绿": {"hue_center": 60,  "colors": ["叶绿", "玉绿", "茶绿"]},
    "青": {"hue_center": 87,  "colors": ["瓷青", "石青", "湖青"]},
    "蓝": {"hue_center": 112, "colors": ["海蓝", "澄蓝", "沧蓝"]},
    "紫": {"hue_center": 143, "colors": ["烟紫", "影紫", "黛紫"]},
}


# ── 结果数据结构 ──

@dataclass
class ColorDetectionResult:
    """颜色检测结果（兼容旧接口 + V2 新增调试字段）"""
    color_type: str                              # 颜色名称（兼容旧字段）
    color_name: str                              # 中文颜色名
    confidence: float                            # 匹配置信度 [0, 1]
    source: str                                  # 来源：object / clothing / fallback
    dominant_hsv: Tuple[int, int, int]           # 提取到的 HSV 主色
    dominant_lab: Optional[Tuple[float, float, float]] = None  # Lab 主色（V2 新增）
    base_family: Optional[str] = None            # 基础色系（V2 新增）
    valid_pixel_count: int = 0                   # 有效像素数（V2 新增）
    cluster_ratio: float = 0.0                   # K-Means 主簇占比（V2 新增）


# ── 物件颜色检测器 (V2) ──

class ObjectColorDetector:
    """物件颜色检测器 — 优化版"""

    MIN_SATURATION = 35
    MIN_VALUE = 30
    MAX_VALUE = 245
    MIN_COMPONENT_PIXELS = 450
    MIN_VALID_PIXELS = 500
    MIN_CLUSTER_RATIO = 0.18
    MIN_CONFIDENCE = 0.34
    MAX_KMEANS_SAMPLES = 12000
    DEFAULT_ROI = (0.30, 0.25, 0.70, 0.75)

    def __init__(self, calibration_path: Optional[str] = None):
        self.color_profiles = deepcopy(COLOR_HEX_VALUES)
        self.last_debug: Dict[str, Any] = {}
        cv2.setRNGSeed(42)
        self._prepare_profiles()
        if calibration_path is None:
            calibration_path = str(Path(__file__).with_name("color_calibration.json"))
        self._load_calibration(calibration_path)

    # ── 工具方法 ─────────────────────────────────────────────

    @staticmethod
    def circular_hue_distance(hue_a: float, hue_b: float) -> float:
        """圆形色相距离（0~180 环），正确处理 0°/179° 跨越"""
        difference = abs(float(hue_a) - float(hue_b))
        return min(difference, 180.0 - difference)

    @staticmethod
    def robust_hue_center(hues: np.ndarray) -> int:
        """鲁棒的色相中心：小跨度用中位数，大跨度用圆形均值"""
        hues = np.asarray(hues, dtype=np.float32)
        if hues.size == 0:
            return 0
        if float(np.max(hues) - np.min(hues)) < 90:
            return int(round(float(np.median(hues)))) % 180
        radians = hues * (2.0 * np.pi / 180.0)
        mean_sin = float(np.mean(np.sin(radians)))
        mean_cos = float(np.mean(np.cos(radians)))
        angle = float(np.arctan2(mean_sin, mean_cos))
        if angle < 0:
            angle += 2.0 * np.pi
        return int(round(angle * 180.0 / (2.0 * np.pi))) % 180

    @staticmethod
    def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
        """灰度世界白平衡，适应展厅复杂光照"""
        if image is None or image.size == 0:
            return image
        float_image = image.astype(np.float32)
        means = float_image.reshape(-1, 3).mean(axis=0)
        gray_mean = float(np.mean(means))
        gains = gray_mean / (means + 1e-6)
        gains = np.clip(gains, 0.90, 1.10)
        balanced = float_image * gains.reshape(1, 1, 3)
        return np.clip(balanced, 0, 255).astype(np.uint8)

    @staticmethod
    def hsv_to_lab(hsv_center: Tuple[int, int, int]) -> np.ndarray:
        """HSV 中心值 → Lab 参考向量"""
        hsv_pixel = np.uint8([[[int(hsv_center[0]), int(hsv_center[1]), int(hsv_center[2])]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
        return lab_pixel[0, 0].astype(np.float32)

    # ── 配置 ─────────────────────────────────────────────────

    def _prepare_profiles(self) -> None:
        """为每个颜色配置补充 Lab 参考向量"""
        for profile in self.color_profiles.values():
            hsv_center = profile.get("hsv_center")
            if hsv_center is None:
                continue
            profile["hsv_center"] = tuple(int(v) for v in hsv_center)
            if "lab_center" not in profile:
                profile["lab_center"] = self.hsv_to_lab(profile["hsv_center"])
            else:
                profile["lab_center"] = np.asarray(profile["lab_center"], dtype=np.float32)

    def _load_calibration(self, calibration_path: str) -> None:
        """加载标定文件，微调颜色配置"""
        path = Path(calibration_path)
        if not path.exists():
            return
        try:
            calibration_data = json.loads(path.read_text(encoding="utf-8"))
            colors = calibration_data.get("colors", calibration_data)
            for color_name, calibrated in colors.items():
                if color_name not in self.color_profiles:
                    continue
                profile = self.color_profiles[color_name]
                if "hsv_center" in calibrated:
                    profile["hsv_center"] = tuple(int(v) for v in calibrated["hsv_center"])
                if "hsv_range" in calibrated:
                    hsv_range = calibrated["hsv_range"]
                    profile["hsv_range"] = (
                        tuple(int(v) for v in hsv_range[0]),
                        tuple(int(v) for v in hsv_range[1]),
                    )
                if "lab_center" in calibrated:
                    profile["lab_center"] = np.asarray(calibrated["lab_center"], dtype=np.float32)
                else:
                    profile["lab_center"] = self.hsv_to_lab(profile["hsv_center"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("颜色标定文件读取失败，继续使用默认配置：%s", exc)

    # ── ROI ──────────────────────────────────────────────────

    def get_default_region(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """默认画面中央 40%×50% 区域"""
        height, width = frame.shape[:2]
        x1_ratio, y1_ratio, x2_ratio, y2_ratio = self.DEFAULT_ROI
        return (
            int(width * x1_ratio),
            int(height * y1_ratio),
            int(width * x2_ratio),
            int(height * y2_ratio),
        )

    @staticmethod
    def crop_region(frame: np.ndarray, region: Tuple[int, int, int, int]):
        """安全裁剪 ROI，返回 (roi, normalized_region)"""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in region]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(1, min(width, x2))
        y2 = max(1, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None, None
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    # ── 预处理 ───────────────────────────────────────────────

    def preprocess(self, roi: np.ndarray) -> np.ndarray:
        """白平衡 + 降噪"""
        balanced = self.gray_world_white_balance(roi)
        return cv2.GaussianBlur(balanced, (5, 5), 0)

    # ── 有效像素掩码 ─────────────────────────────────────────

    def build_valid_mask(self, hsv: np.ndarray) -> np.ndarray:
        """排除灰白背景、阴影和高光，只保留有意义的颜色区域"""
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        valid = (
            (saturation >= self.MIN_SATURATION)
            & (value >= self.MIN_VALUE)
            & (value <= self.MAX_VALUE)
        )
        mask = valid.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=2)
        return mask

    # ── 连通域选择 ───────────────────────────────────────────

    def select_object_component(self, mask: np.ndarray, hsv: np.ndarray):
        """
        在有效像素掩码中选择最可能是"中央物品"的连通域

        评分维度：面积、距画面中心距离、饱和度、是否贴边
        """
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        roi_height, roi_width = mask.shape[:2]
        roi_area = float(roi_height * roi_width)
        roi_center = np.asarray([roi_width / 2.0, roi_height / 2.0], dtype=np.float32)
        maximum_center_distance = float(np.linalg.norm(roi_center)) + 1e-6
        best_label = None
        best_bbox = None
        best_score = -1.0
        candidate_count = 0

        for label_id in range(1, component_count):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < self.MIN_COMPONENT_PIXELS:
                continue
            candidate_count += 1
            component_pixels = labels == label_id
            mean_saturation = float(np.mean(hsv[:, :, 1][component_pixels]))
            area_ratio = area / max(roi_area, 1.0)
            area_score = min(area_ratio / 0.35, 1.0)
            component_center = np.asarray(centroids[label_id], dtype=np.float32)
            center_distance = float(np.linalg.norm(component_center - roi_center))
            center_score = 1.0 - min(center_distance / maximum_center_distance, 1.0)
            saturation_score = min(mean_saturation / 180.0, 1.0)
            touches_edge = (
                x <= 2 or y <= 2
                or x + w >= roi_width - 2
                or y + h >= roi_height - 2
            )
            edge_penalty = 1.0 if touches_edge else 0.0
            score = (
                0.42 * area_score
                + 0.38 * center_score
                + 0.20 * saturation_score
                - 0.28 * edge_penalty
            )
            if score > best_score:
                best_score = score
                best_label = label_id
                best_bbox = (x, y, x + w, y + h)

        self.last_debug["component_count"] = candidate_count

        if best_label is None:
            return None, None, 0.0

        selected_mask = ((labels == best_label).astype(np.uint8) * 255)
        return selected_mask, best_bbox, float(np.clip(best_score, 0.0, 1.0))

    # ── K-Means 主色提取 ─────────────────────────────────────

    def extract_dominant_cluster(self, hsv: np.ndarray, lab: np.ndarray, object_mask: np.ndarray):
        """
        在 Lab 空间用 K-Means 聚类，提取物品的真实主色簇

        排除低饱和度/过暗/过亮的簇，返回最大有效簇的统计信息
        """
        mask_boolean = object_mask > 0
        hsv_pixels = hsv[mask_boolean]
        lab_pixels = lab[mask_boolean]
        valid_pixel_count = int(len(lab_pixels))

        if valid_pixel_count < self.MIN_VALID_PIXELS:
            return None

        if valid_pixel_count > self.MAX_KMEANS_SAMPLES:
            sample_indices = np.linspace(0, valid_pixel_count - 1, self.MAX_KMEANS_SAMPLES, dtype=np.int32)
            hsv_samples = hsv_pixels[sample_indices]
            lab_samples = lab_pixels[sample_indices]
        else:
            hsv_samples = hsv_pixels
            lab_samples = lab_pixels

        sample_count = len(lab_samples)
        cluster_count = 2 if sample_count < 1800 else 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.25)
        compactness, labels, _centers = cv2.kmeans(
            lab_samples.astype(np.float32), cluster_count, None, criteria, 5, cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.reshape(-1)
        cluster_sizes = np.bincount(labels, minlength=cluster_count)
        valid_clusters = []

        for cluster_id in range(cluster_count):
            in_cluster = labels == cluster_id
            if not np.any(in_cluster):
                continue
            cluster_hsv = hsv_samples[in_cluster]
            mean_saturation = float(np.mean(cluster_hsv[:, 1]))
            mean_value = float(np.mean(cluster_hsv[:, 2]))
            if mean_saturation < self.MIN_SATURATION:
                continue
            if not (self.MIN_VALUE <= mean_value <= self.MAX_VALUE):
                continue
            valid_clusters.append((cluster_id, int(cluster_sizes[cluster_id])))

        if not valid_clusters:
            return None

        dominant_cluster_id, dominant_cluster_size = max(valid_clusters, key=lambda item: item[1])
        dominant_mask = labels == dominant_cluster_id
        dominant_hsv_pixels = hsv_samples[dominant_mask]
        dominant_lab_pixels = lab_samples[dominant_mask]

        dominant_hue = self.robust_hue_center(dominant_hsv_pixels[:, 0])
        dominant_saturation = int(round(float(np.median(dominant_hsv_pixels[:, 1]))))
        dominant_value = int(round(float(np.median(dominant_hsv_pixels[:, 2]))))
        dominant_lab = np.median(dominant_lab_pixels, axis=0).astype(np.float32)
        cluster_ratio = dominant_cluster_size / max(sample_count, 1)

        self.last_debug["cluster_sizes"] = cluster_sizes.astype(int).tolist()
        self.last_debug["cluster_ratio"] = float(cluster_ratio)
        self.last_debug["kmeans_compactness"] = float(compactness)

        return {
            "dominant_hsv": (dominant_hue, dominant_saturation, dominant_value),
            "dominant_lab": dominant_lab,
            "cluster_ratio": float(cluster_ratio),
            "valid_pixel_count": valid_pixel_count,
        }

    # ── 两阶段颜色分类 ───────────────────────────────────────

    def classify_base_family(self, dominant_hsv: Tuple[int, int, int]):
        """第一阶段：判断 7 大基础色系"""
        hue, saturation, _value = dominant_hsv
        if saturation < self.MIN_SATURATION:
            return None, 0.0

        distances = []
        for family_name, family_data in BASE_FAMILIES.items():
            distance = self.circular_hue_distance(hue, family_data["hue_center"])
            distances.append((family_name, distance))

        distances.sort(key=lambda item: item[1])
        best_family, best_distance = distances[0]

        self.last_debug["family_distances"] = {
            family: round(float(d), 3) for family, d in distances
        }

        if best_distance > 27:
            return None, 0.0

        confidence = float(np.clip(1.0 - best_distance / 27.0, 0.0, 1.0))
        return best_family, confidence

    @staticmethod
    def is_in_hsv_range(hsv, hsv_range) -> bool:
        """检查 HSV 值是否在范围内（支持红色跨越 0/179）"""
        hue, saturation, value = hsv
        lower, upper = hsv_range
        lower_h, lower_s, lower_v = lower
        upper_h, upper_s, upper_v = upper

        if lower_h <= upper_h:
            hue_valid = lower_h <= hue <= upper_h
        else:
            hue_valid = hue >= lower_h or hue <= upper_h

        return (
            hue_valid
            and lower_s <= saturation <= upper_s
            and lower_v <= value <= upper_v
        )

    def match_cultural_color(self, base_family: str, dominant_hsv, dominant_lab):
        """第二阶段：在基础色系内匹配具体的文化颜色"""
        hue, saturation, value = dominant_hsv
        candidate_names = BASE_FAMILIES[base_family]["colors"]
        scored_candidates = []

        for color_name in candidate_names:
            profile = self.color_profiles.get(color_name)
            if not profile:
                continue

            center_h, center_s, center_v = profile["hsv_center"]
            lab_center = np.asarray(profile["lab_center"], dtype=np.float32)

            hue_distance = self.circular_hue_distance(hue, center_h)
            saturation_distance = abs(float(saturation) - float(center_s)) / 255.0
            value_distance = abs(float(value) - float(center_v)) / 255.0
            lab_distance = float(np.linalg.norm(dominant_lab - lab_center))

            normalized_lab_distance = min(lab_distance / 115.0, 2.0)
            normalized_hue_distance = min(hue_distance / 30.0, 2.0)

            total_distance = (
                0.52 * normalized_lab_distance
                + 0.16 * normalized_hue_distance
                + 0.18 * saturation_distance
                + 0.14 * value_distance
            )

            hsv_range = profile.get("hsv_range")
            if hsv_range is not None and not self.is_in_hsv_range(dominant_hsv, hsv_range):
                total_distance += 0.12

            scored_candidates.append({
                "name": color_name,
                "distance": float(total_distance),
                "lab_distance": lab_distance,
                "hue_distance": hue_distance,
                "saturation_distance": saturation_distance,
                "value_distance": value_distance,
            })

        if len(scored_candidates) < 2:
            return None

        scored_candidates.sort(key=lambda item: item["distance"])
        best = scored_candidates[0]
        second = scored_candidates[1]
        margin = second["distance"] - best["distance"]

        self.last_debug["candidates"] = scored_candidates
        self.last_debug["candidate_margin"] = float(margin)

        if best["distance"] > 1.08:
            return None
        if best["distance"] > 0.72 and margin < 0.025:
            return None

        return {
            "color_name": best["name"],
            "best_distance": float(best["distance"]),
            "margin": float(margin),
        }

    # ── 主接口 ───────────────────────────────────────────────

    def detect(self, frame: np.ndarray,
               region: Optional[Tuple[int, int, int, int]] = None) -> Optional[ColorDetectionResult]:
        """
        检测物件颜色（单帧）

        Args:
            frame: BGR 格式图像
            region: 可选，检测区域 (x1, y1, x2, y2)。为 None 时使用画面中央 ROI

        Returns:
            ColorDetectionResult 或 None（不确定时返回 None，由上层随机保底）
        """
        self.last_debug = {}

        if frame is None or frame.size == 0:
            self.last_debug["failure_reason"] = "empty_frame"
            return None

        if region is None:
            region = self.get_default_region(frame)

        roi, normalized_region = self.crop_region(frame, region)
        if roi is None or normalized_region is None:
            self.last_debug["failure_reason"] = "invalid_roi"
            return None

        self.last_debug["roi"] = normalized_region
        processed = self.preprocess(roi)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        valid_mask = self.build_valid_mask(hsv)

        valid_pixel_count = int(cv2.countNonZero(valid_mask))
        self.last_debug["valid_pixels_before_component"] = valid_pixel_count

        if valid_pixel_count < self.MIN_VALID_PIXELS:
            self.last_debug["failure_reason"] = "not_enough_valid_pixels"
            return None

        object_mask, object_bbox, region_score = self.select_object_component(valid_mask, hsv)
        if object_mask is None:
            self.last_debug["failure_reason"] = "no_valid_component"
            return None

        self.last_debug["object_bbox_in_roi"] = object_bbox
        self.last_debug["region_score"] = region_score

        cluster_result = self.extract_dominant_cluster(hsv, lab, object_mask)
        if cluster_result is None:
            self.last_debug["failure_reason"] = "no_dominant_cluster"
            return None

        dominant_hsv = cluster_result["dominant_hsv"]
        dominant_lab = cluster_result["dominant_lab"]
        cluster_ratio = cluster_result["cluster_ratio"]
        valid_pixel_count = cluster_result["valid_pixel_count"]

        self.last_debug["dominant_hsv"] = dominant_hsv
        self.last_debug["dominant_lab"] = dominant_lab.astype(float).tolist()

        if cluster_ratio < self.MIN_CLUSTER_RATIO:
            self.last_debug["failure_reason"] = "cluster_ratio_too_low"
            return None

        base_family, family_confidence = self.classify_base_family(dominant_hsv)
        if base_family is None:
            self.last_debug["failure_reason"] = "base_family_uncertain"
            return None

        self.last_debug["base_family"] = base_family

        cultural_match = self.match_cultural_color(base_family, dominant_hsv, dominant_lab)
        if cultural_match is None:
            self.last_debug["failure_reason"] = "cultural_color_uncertain"
            return None

        best_distance = cultural_match["best_distance"]
        margin = cultural_match["margin"]

        distance_score = float(np.clip(1.0 - best_distance / 1.08, 0.0, 1.0))
        margin_score = float(np.clip(margin / 0.18, 0.0, 1.0))
        pixel_score = float(np.clip(valid_pixel_count / 5000.0, 0.0, 1.0))

        confidence = (
            0.39 * distance_score
            + 0.22 * cluster_ratio
            + 0.14 * margin_score
            + 0.10 * pixel_score
            + 0.10 * region_score
            + 0.05 * family_confidence
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if confidence < self.MIN_CONFIDENCE:
            self.last_debug["failure_reason"] = "confidence_too_low"
            self.last_debug["confidence"] = confidence
            return None

        color_name = cultural_match["color_name"]

        self.last_debug["result"] = color_name
        self.last_debug["confidence"] = confidence

        return ColorDetectionResult(
            color_type=color_name,
            color_name=color_name,
            confidence=round(confidence, 3),
            source="object",
            dominant_hsv=dominant_hsv,
            dominant_lab=tuple(round(float(v), 3) for v in dominant_lab),
            base_family=base_family,
            valid_pixel_count=valid_pixel_count,
            cluster_ratio=round(cluster_ratio, 3),
        )

    def detect_dominant_region(self, frame: np.ndarray,
                                min_saturation: int = 30,
                                min_area: int = 1000) -> Optional[Tuple[int, int, int, int]]:
        """
        检测画面中央最显著的颜色区域

        Args:
            frame: BGR 格式图像
            min_saturation: 最小饱和度阈值
            min_area: 最小区域面积（像素）

        Returns:
            (x1, y1, x2, y2) 或 None
        """
        if frame is None or frame.size == 0:
            return None

        region = self.get_default_region(frame)
        roi, normalized_region = self.crop_region(frame, region)

        if roi is None or normalized_region is None:
            return None

        processed = self.preprocess(roi)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        original_min_saturation = self.MIN_SATURATION
        try:
            self.MIN_SATURATION = max(int(min_saturation), original_min_saturation)
            mask = self.build_valid_mask(hsv)
        finally:
            self.MIN_SATURATION = original_min_saturation

        object_mask, bbox, _region_score = self.select_object_component(mask, hsv)

        if object_mask is None or bbox is None:
            return None

        if cv2.countNonZero(object_mask) < min_area:
            return None

        roi_x1, roi_y1, _, _ = normalized_region
        x1, y1, x2, y2 = bbox

        return (
            roi_x1 + x1,
            roi_y1 + y1,
            roi_x1 + x2,
            roi_y1 + y2,
        )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("物件颜色检测器自测 (V2)")
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
            if result.base_family:
                cv2.putText(display, f"色系: {result.base_family}",
                           (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display, f"NO MATCH: {detector.last_debug.get('failure_reason', '?')}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Object Color Detection", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n自测结束")
