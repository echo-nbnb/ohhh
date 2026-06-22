"""
电脑端摄像头衣物颜色检测器 — 第一幕"择色"衣物兜底方案

当物件颜色匹配失败时，使用电脑端摄像头识别用户衣物颜色。

识别管线:
    电脑端摄像头帧 → MediaPipe人像分割 → 提取躯干区域
                                               ↓
                              HSV颜色统计 → 主色提取 → 六色匹配

使用方式:
    detector = WebcamColorDetector()
    color_type, confidence = detector.detect(frame)
"""

import os
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger("WebcamColorDetector")

# 模型路径（相对于项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_MODEL = os.path.join(PROJECT_ROOT, 'vision', 'pose_landmarker.task')
SELFIE_MODEL = os.path.join(PROJECT_ROOT, 'vision', 'selfie_segmenter.tflite')

# 21色HSV中心值（v2 三组）
COLOR_HEX_VALUES = {
    "朱红": {"hsv_center": (0, 254, 221), "hsv_range": ((0, 80, 40), (8, 255, 255))},
    "灯橙": {"hsv_center": (24, 255, 254), "hsv_range": ((14, 80, 40), (34, 255, 255))},
    "梨黄": {"hsv_center": (56, 187, 240), "hsv_range": ((42, 60, 40), (70, 255, 255))},
    "叶绿": {"hsv_center": (102, 210, 255), "hsv_range": ((85, 60, 40), (118, 255, 255))},
    "瓷青": {"hsv_center": (177, 218, 255), "hsv_range": ((162, 60, 40), (192, 255, 255))},
    "海蓝": {"hsv_center": (220, 224, 255), "hsv_range": ((205, 60, 40), (235, 255, 255))},
    "烟紫": {"hsv_center": (269, 255, 255), "hsv_range": ((255, 60, 40), (15, 255, 255))},
    "枫红": {"hsv_center": (0, 255, 196), "hsv_range": ((0, 80, 30), (8, 255, 255))},
    "暖橙": {"hsv_center": (38, 205, 255), "hsv_range": ((28, 60, 40), (48, 255, 255))},
    "藤黄": {"hsv_center": (53, 255, 246), "hsv_range": ((40, 60, 40), (66, 255, 255))},
    "玉绿": {"hsv_center": (123, 255, 235), "hsv_range": ((108, 60, 40), (138, 255, 255))},
    "石青": {"hsv_center": (160, 196, 253), "hsv_range": ((145, 50, 40), (175, 255, 255))},
    "澄蓝": {"hsv_center": (232, 219, 235), "hsv_range": ((218, 60, 40), (248, 255, 255))},
    "影紫": {"hsv_center": (266, 255, 187), "hsv_range": ((252, 60, 30), (280, 255, 255))},
    "桃红": {"hsv_center": (0, 190, 255), "hsv_range": ((0, 50, 50), (8, 255, 255))},
    "夕橙": {"hsv_center": (25, 251, 231), "hsv_range": ((15, 60, 40), (35, 255, 255))},
    "桂黄": {"hsv_center": (56, 255, 255), "hsv_range": ((42, 60, 40), (70, 255, 255))},
    "茶绿": {"hsv_center": (141, 199, 255), "hsv_range": ((126, 50, 40), (156, 255, 255))},
    "湖青": {"hsv_center": (182, 193, 255), "hsv_range": ((168, 50, 50), (196, 255, 255))},
    "沧蓝": {"hsv_center": (225, 187, 255), "hsv_range": ((210, 50, 50), (240, 255, 255))},
    "黛紫": {"hsv_center": (281, 255, 231), "hsv_range": ((268, 60, 40), (292, 255, 255))},
}


@dataclass
class ClothingColorResult:
    """衣物颜色检测结果"""
    color_type: str          # 颜色名称
    color_name: str         # 中文颜色名
    confidence: float        # 匹配置信度 [0, 1]
    dominant_hsv: Tuple[int, int, int]  # 提取到的HSV主色


class WebcamColorDetector:
    """
    电脑端摄像头衣物颜色检测器

    支持三种检测方法，通过 method 参数选择：
    - "simple": 基于肤色连通域分析（无MediaPipe依赖）
    - "pose":   基于MediaPipe Pose关节点定位躯干
    - "selfie": 基于MediaPipe SelfieSegmentation像素级分割

    用法:
        detector = WebcamColorDetector(method="pose")   # 推荐
        detector = WebcamColorDetector(method="selfie")  # 精度更高
        detector = WebcamColorDetector(method="simple")  # 不依赖MediaPipe
    """

    def __init__(self, device: int = 0, method: str = "simple"):
        """
        Args:
            device: 摄像头设备编号，默认0
            method: 检测方法，"simple" | "pose" | "selfie"
        """
        self.device = device
        self.method = method
        self._cap: Optional[cv2.VideoCapture] = None
        self._mediapipe_pose = None    # 延迟加载
        self._mediapipe_selfie = None  # 延迟加载

    def _ensure_pose(self):
        """延迟加载MediaPipe Pose (Tasks API)"""
        if self._mediapipe_pose is None:
            try:
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
                base_options = BaseOptions(model_asset_path=POSE_MODEL)
                options = PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=RunningMode.IMAGE
                )
                self._mediapipe_pose = PoseLandmarker.create_from_options(options)
                logger.info("MediaPipe Pose loaded")
            except Exception as e:
                logger.warning(f"MediaPipe Pose load failed: {e}")
                self._mediapipe_pose = False  # False = load failed

    def _ensure_selfie(self):
        """延迟加载MediaPipe SelfieSegmentation (Tasks API)"""
        if self._mediapipe_selfie is None:
            try:
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions
                base_options = BaseOptions(model_asset_path=SELFIE_MODEL)
                options = ImageSegmenterOptions(
                    base_options=base_options,
                    output_confidence_masks=True,
                    output_category_mask=False
                )
                self._mediapipe_selfie = ImageSegmenter.create_from_options(options)
                logger.info("MediaPipe SelfieSegmentation loaded")
            except Exception as e:
                logger.warning(f"MediaPipe SelfieSegmentation load failed: {e}")
                self._mediapipe_selfie = False

    def open(self) -> bool:
        """打开电脑端摄像头"""
        if self._cap is not None and self._cap.isOpened():
            return True

        self._cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
        if self._cap.isOpened():
            import time
            time.sleep(0.5)  # 等待摄像头初始化
            logger.info(f"电脑端摄像头已打开 (device={self.device})")
            return True
        else:
            logger.error(f"电脑端摄像头打开失败 (device={self.device})")
            return False

    def close(self):
        """关闭摄像头"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("电脑端摄像头已关闭")

    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧，带重试"""
        if self._cap is None or not self._cap.isOpened():
            return None
        # 重试3次应对瞬时错误
        for _ in range(3):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return frame
        return None

    def detect(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """
        从帧中检测衣物颜色

        Args:
            frame: BGR格式图像

        Returns:
            (颜色名称, 置信度) 或 (None, 0.0) 如果检测失败
        """
        if frame is None:
            return None, 0.0

        if self.method == "simple":
            return self._detect_simple(frame)
        elif self.method == "pose":
            self._ensure_pose()
            if self._mediapipe_pose is False:
                logger.warning("Pose unavailable, falling back to simple")
                return self._detect_simple(frame)
            try:
                return self._detect_with_pose(frame)
            except Exception as e:
                logger.warning(f"Pose detection error: {e}, falling back to simple")
                return self._detect_simple(frame)
        elif self.method == "selfie":
            self._ensure_selfie()
            if self._mediapipe_selfie is False:
                logger.warning("SelfieSegmentation unavailable, falling back to simple")
                return self._detect_simple(frame)
            try:
                return self._detect_with_selfie(frame)
            except Exception as e:
                logger.warning(f"Selfie detection error: {e}, falling back to simple")
                return self._detect_simple(frame)
        else:
            return self._detect_simple(frame)

    def _detect_with_pose(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """使用MediaPipe Pose关节点定位躯干，排除肤色后提取主色"""
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._mediapipe_pose.detect(mp_image)

        if not results.pose_landmarks:
            return None, 0.0

        h, w = frame.shape[:2]

        # MediaPipe关键点：11=左肩，12=右肩，23=左髋，24=右髋
        landmarks = results.pose_landmarks[0]

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        # 计算躯干区域边界（肩膀到臀部）
        x_min = int(min(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w)
        x_max = int(max(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w)
        y_min = int(min(left_shoulder.y, right_shoulder.y) * h)
        y_max = int(max(left_hip.y, right_hip.y) * h)

        # 添加边距
        margin = 10
        x_min = max(0, x_min - margin)
        x_max = min(w, x_max + margin)
        y_min = max(0, y_min - margin)
        y_max = min(h, y_max + margin)

        if x_max <= x_min or y_max <= y_min:
            return None, 0.0

        # 提取躯干区域
        torso_hsv = cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2HSV)

        # 排除肤色
        skin_mask = self._create_skin_mask(torso_hsv)
        non_skin_mask_u8 = (~skin_mask).astype(np.uint8)

        # 找连通域，取最大非肤色区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(non_skin_mask_u8, connectivity=8)
        if num_labels <= 1:
            return None, 0.0

        largest_label = 1
        largest_area = stats[1, cv2.CC_STAT_AREA]
        for i in range(2, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > largest_area:
                largest_area = stats[i, cv2.CC_STAT_AREA]
                largest_label = i

        mask_one = (labels == largest_label).astype(np.uint8) * 255
        component_hsv = cv2.bitwise_and(torso_hsv, torso_hsv, mask=mask_one)
        component_flat = component_hsv.reshape(-1, 3)
        valid = component_flat[:, 2] > 5
        if valid.sum() < 50:
            return None, 0.0

        valid_pixels = component_flat[valid]
        h_peak = int(np.median(valid_pixels[:, 0]))
        s_peak = int(np.median(valid_pixels[:, 1]))
        v_peak = int(np.median(valid_pixels[:, 2]))

        return self._match_to_color((h_peak, s_peak, v_peak))

    def _detect_with_selfie(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """使用MediaPipe SelfieSegmentation像素级分割，提取衣物主色"""
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._mediapipe_selfie.segment(mp_image)

        if results.confidence_masks is None:
            return None, 0.0

        # 取第一个置信度mask（单一前景mask）
        mask_img = results.confidence_masks[0]
        mask = (mask_img.numpy_view() > 0.5).astype(np.uint8) * 255
        h, w = frame.shape[:2]

        # 形态学清理
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 连通域分析：取最大人物区域（支持多人场景选最大）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return None, 0.0

        largest_label = 1
        largest_area = stats[1, cv2.CC_STAT_AREA]
        for i in range(2, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > largest_area:
                largest_area = stats[i, cv2.CC_STAT_AREA]
                largest_label = i

        person_mask = (labels == largest_label).astype(np.uint8) * 255

        # 取人物躯干区域（头部占上半部，取下半身的非肤色区域）
        torso_top = int(h * 0.3)
        torso_mask = person_mask[torso_top:, :]
        person_torso = frame[torso_top:, :]

        # 排除肤色
        torso_hsv = cv2.cvtColor(person_torso, cv2.COLOR_BGR2HSV)
        skin_mask_torso = self._create_skin_mask(torso_hsv)
        non_skin_u8 = (~skin_mask_torso).astype(np.uint8)
        non_skin_masked = cv2.bitwise_and(torso_hsv, torso_hsv, mask=non_skin_u8)

        # 连通域取最大非肤色区域
        num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(non_skin_masked, connectivity=8)
        if num2 > 1:
            largest_label2 = 1
            largest_area2 = stats2[1, cv2.CC_STAT_AREA]
            for i in range(2, num2):
                if stats2[i, cv2.CC_STAT_AREA] > largest_area2:
                    largest_area2 = stats2[i, cv2.CC_STAT_AREA]
                    largest_label2 = i
            cloth_mask = (labels2 == largest_label2).astype(np.uint8) * 255
        else:
            cloth_mask = non_skin_masked

        cloth_hsv = cv2.bitwise_and(torso_hsv, torso_hsv, mask=cloth_mask)
        cloth_flat = cloth_hsv.reshape(-1, 3)
        valid = cloth_flat[:, 2] > 5
        if valid.sum() < 50:
            return None, 0.0

        valid_pixels = cloth_flat[valid]
        h_peak = int(np.median(valid_pixels[:, 0]))
        s_peak = int(np.median(valid_pixels[:, 1]))
        v_peak = int(np.median(valid_pixels[:, 2]))

        return self._match_to_color((h_peak, s_peak, v_peak))

    def _detect_simple(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """
        简单方法：取画面下半部分躯干区域，排除肤色后提取主色
        不依赖MediaPipe，基于肤色分割和连通域分析
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 取画面下半部分作为躯干区域
        torso_top = int(h * 0.35)
        torso_bottom = int(h * 0.95)
        torso_hsv = hsv[torso_top:torso_bottom, :]

        # 排除肤色
        skin_mask = self._create_skin_mask(torso_hsv)
        non_skin_u8 = (~skin_mask).astype(np.uint8)

        # 连通域分析，取最大非肤色区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(non_skin_u8, connectivity=8)
        if num_labels <= 1:
            return None, 0.0

        largest_label = 1
        largest_area = stats[1, cv2.CC_STAT_AREA]
        for i in range(2, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > largest_area:
                largest_area = stats[i, cv2.CC_STAT_AREA]
                largest_label = i

        mask_one = (labels == largest_label).astype(np.uint8) * 255
        component_hsv = cv2.bitwise_and(torso_hsv, torso_hsv, mask=mask_one)
        component_flat = component_hsv.reshape(-1, 3)
        valid = component_flat[:, 2] > 5
        if valid.sum() < 50:
            return None, 0.0

        valid_pixels = component_flat[valid]
        h_peak = int(np.median(valid_pixels[:, 0]))
        s_peak = int(np.median(valid_pixels[:, 1]))
        v_peak = int(np.median(valid_pixels[:, 2]))

        return self._match_to_color((h_peak, s_peak, v_peak))

    def _create_skin_mask(self, hsv: np.ndarray) -> np.ndarray:
        """创建肤色掩码"""
        # 简单肤色检测：H在0-20或165-180，S在40-255，V在40-255
        lower_skin1 = np.array([0, 40, 40])
        upper_skin1 = np.array([20, 255, 255])
        lower_skin2 = np.array([165, 40, 40])
        upper_skin2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(mask1, mask2)

        # 膨胀核
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

        return skin_mask

    def _match_to_color(self, hsv: Tuple[int, int, int]) -> Tuple[Optional[str], float]:
        """
        将HSV颜色与六色匹配

        Args:
            hsv: (H, S, V) 元组

        Returns:
            (颜色名称, 置信度)
        """
        h, s, v = hsv

        best_color = None
        best_score = 0.0

        for color_name, color_info in COLOR_HEX_VALUES.items():
            hsv_center = color_info["hsv_center"]
            hsv_range = color_info["hsv_range"]

            # 计算距离
            h_dist = min(abs(h - hsv_center[0]), 180 - abs(h - hsv_center[0])) / 90.0
            s_dist = abs(s - hsv_center[1]) / 128.0
            v_dist = abs(v - hsv_center[2]) / 128.0

            # 综合距离（加权）
            total_dist = h_dist * 0.4 + s_dist * 0.3 + v_dist * 0.3

            # 转换为置信度（距离越小置信度越高）
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

        # 置信度阈值
        if best_score < 0.3:
            return None, 0.0

        return best_color, round(best_score, 3)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # 解析命令行参数
    method = "simple"
    if len(sys.argv) > 1:
        method = sys.argv[1].lower()
    valid_methods = ["simple", "pose", "selfie"]
    if method not in valid_methods:
        print(f"Usage: python webcam_color_detector.py [{'/'.join(valid_methods)}]")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print(f"Webcam Clothing Color Detector Self-Test")
    print(f"Method: {method}")
    print("=" * 50)

    detector = WebcamColorDetector(device=0, method=method)

    if not detector.open():
        print("Cannot open webcam, exit")
        sys.exit(1)

    print("\nWebcam test started (Q to quit)...")

    while True:
        frame = detector.read_frame()
        if frame is None:
            print("Cannot read frame")
            break

        color_name, confidence = detector.detect(frame)

        # Print to terminal
        if color_name:
            print(f"Detected: {color_name} ({confidence:.2f})")
        else:
            print("Analyzing...")

        # Visualization
        display = frame.copy()
        h, w = frame.shape[:2]

        if color_name:
            text = f"{color_name} ({confidence:.2f})"
            cv2.rectangle(display, (5, 5), (450, 55), (0, 0, 0), -1)
            cv2.putText(display, text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.rectangle(display, (5, 5), (450, 55), (0, 0, 0), -1)
            cv2.putText(display, "Analyzing...", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Clothing Color Detection", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cv2.destroyAllWindows()
    print("\nSelf-test ended")
