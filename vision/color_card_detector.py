"""
YOLO 颜色牌检测器 — 第一幕"择色"
展览暗光环境采用 颜色+纹路 双保险识别方案。

识别管线:
    摄像头帧 → YOLOv8n 定位卡片区域 → 裁剪卡片区域
                                           ↓
                    ┌──────────────────────┴──────────────────────┐
                    ↓                                             ↓
            颜色识别器                                    纹路识别器
            HSV 直方图匹配                              PiDiNet 边缘 + 模板匹配
                    ↓                                             ↓
                    └──────────────────────┬──────────────────────┘
                                           ↓
                                    决策融合
                              color × 0.4 + edge × 0.6
                                           ↓
                                   最终类别 + 置信度

使用方式:
    # 首次使用：采集模板
    collect_templates()  # 需在光线良好环境下对每种颜色牌采集一次

    # 正常检测
    detector = ColorCardDetector(
        yolo_path="yolo/color_card.pt",
        template_path="vision/color_card_templates.npz"
    )
    cards = detector.detect(frame)  # → List[ColorCardDetection]
"""

import os
import logging
import json
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

import torch
import cv2
import numpy as np

logger = logging.getLogger("ColorCardDetector")


# ---------------------------------------------------------------------------
# 颜色牌类型定义
# ---------------------------------------------------------------------------

class ColorCardType(Enum):
    """六种颜色牌"""
    YUELU_GREEN = "岳麓绿"
    ACADEMY_RED = "书院红"
    XIQIAN_YELLOW = "西迁黄"
    XIANGJIANG_BLUE = "湘江蓝"
    BADGE_GOLD = "校徽金"
    INK_BLACK = "墨色"

    @classmethod
    def from_class_id(cls, class_id: int) -> "ColorCardType":
        mapping = {
            0: cls.YUELU_GREEN,
            1: cls.ACADEMY_RED,
            2: cls.XIQIAN_YELLOW,
            3: cls.XIANGJIANG_BLUE,
            4: cls.BADGE_GOLD,
            5: cls.INK_BLACK,
        }
        return mapping.get(class_id, cls.YUELU_GREEN)

    @classmethod
    def to_class_id(cls, card_type: "ColorCardType") -> int:
        reverse = {v: k for k, v in {
            0: cls.YUELU_GREEN,
            1: cls.ACADEMY_RED,
            2: cls.XIQIAN_YELLOW,
            3: cls.XIANGJIANG_BLUE,
            4: cls.BADGE_GOLD,
            5: cls.INK_BLACK,
        }.items()}
        return reverse.get(card_type, 0)

    @classmethod
    def all_types(cls) -> List["ColorCardType"]:
        return list(cls)


# ---------------------------------------------------------------------------
# 检测结果数据类
# ---------------------------------------------------------------------------

@dataclass
class ColorCardDetection:
    """单张颜色牌检测结果"""
    card_type: ColorCardType
    confidence: float           # 融合置信度 [0, 1]
    color_confidence: float     # 颜色识别置信度
    edge_confidence: float     # 纹路识别置信度
    center: Tuple[int, int]   # 像素中心 (x, y)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    track_id: Optional[int] = None


# ---------------------------------------------------------------------------
# 模板管理器
# ---------------------------------------------------------------------------

class TemplateManager:
    """管理颜色牌的 HSV 直方图和边缘图模板"""

    def __init__(self, template_path: str):
        self.template_path = template_path
        self.templates: Dict[str, dict] = {}
        self._load_or_init()

    def _load_or_init(self):
        """加载已有模板或初始化空模板"""
        if os.path.exists(self.template_path):
            try:
                data = np.load(self.template_path, allow_pickle=True)
                self.templates = data.item()
                logger.info(f"模板已加载: {len(self.templates)} 种颜色牌")
            except Exception as e:
                logger.warning(f"模板加载失败: {e}，将创建新模板")
                self.templates = {}
        else:
            logger.info("未找到模板文件，将创建新模板")
            self.templates = {}

    def save(self):
        """保存模板到文件"""
        np.savez(self.template_path, **self.templates)
        logger.info(f"模板已保存: {self.template_path}")

    def add_template(self, card_type: ColorCardType, hsv_hist: np.ndarray,
                     edge_map: np.ndarray, color_hist_hsv: np.ndarray = None):
        """添加一个颜色牌的模板"""
        self.templates[card_type.value] = {
            "hsv_hist": hsv_hist,
            "edge_map": edge_map,
            "color_hist_hsv": color_hist_hsv or hsv_hist,
        }
        logger.info(f"已添加模板: {card_type.value}")

    def has_template(self, card_type: ColorCardType) -> bool:
        """检查是否有某颜色牌的模板"""
        return card_type.value in self.templates

    def get_template(self, card_type: ColorCardType) -> Optional[dict]:
        """获取某颜色牌的模板"""
        return self.templates.get(card_type.value)

    @property
    def is_ready(self) -> bool:
        """检查是否所有 6 种颜色牌都有模板"""
        return all(self.has_template(ct) for ct in ColorCardType.all_types())


# ---------------------------------------------------------------------------
# 颜色识别器
# ---------------------------------------------------------------------------

class ColorClassifier:
    """基于 HSV 颜色直方图的颜色识别器"""

    # 每种颜色牌的 HSV 主色调参考（展览暗光环境可能偏暗）
    HSV_RANGES = {
        ColorCardType.YUELU_GREEN:     ((35, 40, 20), (85, 255, 200)),
        ColorCardType.ACADEMY_RED:     ((0, 40, 20), (15, 255, 200)),
        ColorCardType.XIQIAN_YELLOW:   ((15, 40, 40), (40, 255, 255)),
        ColorCardType.XIANGJIANG_BLUE: ((95, 40, 20), (135, 255, 200)),
        ColorCardType.BADGE_GOLD:      ((10, 60, 60), (35, 200, 255)),
        ColorCardType.INK_BLACK:       ((0, 0, 0), (180, 50, 60)),
    }

    def __init__(self, template_manager: TemplateManager):
        self.tmpl_mgr = template_manager

    def classify(self, card_crop: np.ndarray) -> Tuple[ColorCardType, float]:
        """
        根据 HSV 直方图匹配识别颜色牌

        Args:
            card_crop: BGR 格式的卡片区域图像

        Returns:
            (颜色牌类型, 置信度 0-1)
        """
        # 转换为 HSV
        hsv = cv2.cvtColor(card_crop, cv2.COLOR_BGR2HSV)

        # 计算查询图像的 2D HSV 直方图 (H: 0-180, S: 0-256)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        best_type = None
        best_score = 0.0

        for card_type in ColorCardType.all_types():
            template = self.tmpl_mgr.get_template(card_type)
            if template is None:
                # 无模板时：用 HSV 范围检测
                score = self._score_by_hsv_range(hsv, card_type)
            else:
                # 有模板时：直方图匹配
                tmpl_hist = template["hsv_hist"]
                score = cv2.compareHist(hist, tmpl_hist, cv2.HISTCMP_CORREL)

            if score > best_score:
                best_score = score
                best_type = card_type

        # 归一化分数到 [0, 1]
        # 相关性系数范围通常是 [-1, 1]，映射到 [0, 1]
        normalized_score = (best_score + 1) / 2 if best_score >= 0 else 0

        return best_type, normalized_score

    def _score_by_hsv_range(self, hsv: np.ndarray,
                             card_type: ColorCardType) -> float:
        """无模板时，用 HSV 范围检测（备用方案）"""
        h_min, h_max = self.HSV_RANGES[card_type][0][0], self.HSV_RANGES[card_type][1][0]
        s_min, s_max = self.HSV_RANGES[card_type][0][1], self.HSV_RANGES[card_type][1][1]
        v_min, v_max = self.HSV_RANGES[card_type][0][2], self.HSV_RANGES[card_type][1][2]

        # 统计落在范围内的像素比例
        mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
        ratio = cv2.countNonZero(mask) / (hsv.shape[0] * hsv.shape[1])
        return ratio


# ---------------------------------------------------------------------------
# 纹路识别器
# ---------------------------------------------------------------------------

class EdgeClassifier:
    """基于 PiDiNet 边缘检测的纹路识别器"""

    def __init__(self, template_manager: TemplateManager,
                 pidinet_model=None, device: str = "cpu"):
        self.tmpl_mgr = template_manager
        self.pidinet = pidinet_model
        self.device = device
        self._edge_cache = {}

    def classify(self, card_crop: np.ndarray,
                 preprocess_func=None) -> Tuple[ColorCardType, float]:
        """
        根据边缘图纹路匹配识别颜色牌

        Args:
            card_crop: BGR 格式的卡片区域图像
            preprocess_func: 可选，自定义边缘检测预处理函数

        Returns:
            (颜色牌类型, 置信度 0-1)
        """
        # 提取边缘
        edge_map = self._extract_edge(card_crop, preprocess_func)

        best_type = None
        best_score = 0.0

        for card_type in ColorCardType.all_types():
            template = self.tmpl_mgr.get_template(card_type)
            if template is None:
                continue

            tmpl_edge = template["edge_map"]

            # 计算相似度：边缘重叠率 (IoU-style)
            score = self._edge_similarity(edge_map, tmpl_edge)

            if score > best_score:
                best_score = score
                best_type = card_type

        return best_type, best_score

    def _extract_edge(self, card_crop: np.ndarray,
                      preprocess_func=None) -> np.ndarray:
        """
        从卡片区域提取边缘图

        优先使用 PiDiNet，备选 Canny
        """
        if self.pidinet is not None and preprocess_func is not None:
            # 使用 PiDiNet
            try:
                h, w = card_crop.shape[:2]
                # 预处理：调整大小到 640x640（PiDiNet 输入尺寸）
                input_size = 640
                scale = input_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)

                resized = cv2.resize(card_crop, (new_w, new_h))
                square = np.zeros((input_size, input_size, 3), dtype=np.uint8)
                square[:new_h, :new_w] = resized

                # 转换为 tensor
                img_float = square.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_normalized = (img_float - mean) / std
                img_tensor = torch.from_numpy(
                    img_normalized.transpose(2, 0, 1)
                ).float().unsqueeze(0).to(self.device)

                # PiDiNet 推理
                with torch.no_grad():
                    outputs = self.pidinet(img_tensor)
                    edge = outputs[-1].squeeze().cpu().numpy()

                # 缩放回原始尺寸
                edge_resized = cv2.resize(edge, (w, h))
                return edge_resized

            except Exception as e:
                logger.warning(f"PiDiNet 推理失败: {e}，降级到 Canny")

        # 备选：Canny 边缘检测
        gray = cv2.cvtColor(card_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, 50, 150)
        return edges.astype(np.float32) / 255.0

    def _edge_similarity(self, edge1: np.ndarray, edge2: np.ndarray) -> float:
        """
        计算两张边缘图的相似度（IoU-style）

        Args:
            edge1, edge2: 归一化到 [0, 1] 的边缘图

        Returns:
            相似度分数 [0, 1]
        """
        # 确保尺寸一致
        if edge1.shape != edge2.shape:
            edge2 = cv2.resize(edge2, (edge1.shape[1], edge1.shape[0]))

        # 二值化
        threshold = 0.3
        e1_binary = (edge1 > threshold).astype(np.float32)
        e2_binary = (edge2 > threshold).astype(np.float32)

        # 计算 IoU
        intersection = np.logical_and(e1_binary, e2_binary).sum()
        union = np.logical_or(e1_binary, e2_binary).sum()

        if union == 0:
            return 0.0

        return intersection / union


# ---------------------------------------------------------------------------
# 双识别器融合
# ---------------------------------------------------------------------------

class DualColorCardDetector:
    """
    颜色+纹路双保险颜色牌检测器

    管线:
        YOLOv8n 定位 → 颜色识别 → 纹路识别 → 决策融合
        融合权重: color × 0.4 + edge × 0.6
    """

    # 融合权重
    COLOR_WEIGHT = 0.4
    EDGE_WEIGHT = 0.6

    def __init__(self,
                 yolo_path: Optional[str] = None,
                 template_path: str = "vision/color_card_templates.npz",
                 device: str = "cpu",
                 conf_threshold: float = 0.5):
        """
        Args:
            yolo_path: YOLO .pt 权重路径。None 则使用 mock 定位模式。
            template_path: 颜色/边缘模板文件路径
            device: 运行设备 "cpu" 或 "cuda"
            conf_threshold: 最终置信度阈值
        """
        self.yolo_path = yolo_path
        self.template_path = template_path
        self.device = device
        self.conf_threshold = conf_threshold

        self._yolo = None
        self._pidinet = None
        self._use_mock_yolo = yolo_path is None

        # 组件
        self.tmpl_mgr = TemplateManager(template_path)
        self.color_clf = ColorClassifier(self.tmpl_mgr)
        self.edge_clf = EdgeClassifier(self.tmpl_mgr, None, device)  # pidinet 延迟加载

        self._load_models()

    def _load_models(self):
        """加载 YOLO 和 PiDiNet 模型"""
        # YOLO
        if not self._use_mock_yolo and os.path.exists(self.yolo_path):
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(self.yolo_path)
                self._use_mock_yolo = False
                logger.info(f"YOLO 已加载: {self.yolo_path}")
            except Exception as e:
                logger.warning(f"YOLO 加载失败: {e}，使用 mock 模式")
                self._use_mock_yolo = True

        # PiDiNet（延迟加载，仅在首次需要纹路识别时加载）
        # 注意：PiDiNet 需要 torch，已在 edge_detection_ipcam.py 中实现

    def _ensure_pidinet(self):
        """延迟加载 PiDiNet"""
        if self._pidinet is None:
            try:
                from vision.edge_detection_ipcam import load_pidinet
                self._pidinet, _ = load_pidinet()
                self.edge_clf.pidinet = self._pidinet
                logger.info("PiDiNet 已加载")
            except Exception as e:
                logger.warning(f"PiDiNet 加载失败: {e}，使用 Canny 备选")
                self._pidinet = False  # 用 False 表示加载失败（不是 None）

    def detect(self, frame: np.ndarray,
               preprocess_func=None) -> List[ColorCardDetection]:
        """
        检测帧中的颜色牌

        Args:
            frame: BGR numpy 数组 (h, w, 3)
            preprocess_func: PiDiNet 预处理函数（可选）

        Returns:
            检测到的颜色牌列表
        """
        # 1. YOLO 定位（获取卡片区域）
        if self._use_mock_yolo:
            regions = self._mock_locate(frame)
        else:
            regions = self._yolo_locate(frame)

        if not regions:
            return []

        # 2. 对每个区域进行双识别
        detections = []
        for bbox in regions:
            x1, y1, x2, y2 = bbox
            card_crop = frame[y1:y2, x1:x2]

            if card_crop.size == 0:
                continue

            # 颜色识别
            color_type, color_conf = self.color_clf.classify(card_crop)

            # 纹路识别（确保 PiDiNet 已加载）
            self._ensure_pidinet()
            edge_type, edge_conf = self.edge_clf.classify(
                card_crop, preprocess_func
            )

            # 3. 决策融合
            final_type, final_conf = self._fuse(
                color_type, color_conf,
                edge_type, edge_conf
            )

            if final_conf < self.conf_threshold:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w, h = x2 - x1, y2 - y1

            detections.append(ColorCardDetection(
                card_type=final_type,
                confidence=final_conf,
                color_confidence=color_conf,
                edge_confidence=edge_conf,
                center=(cx, cy),
                bbox=(x1, y1, w, h),
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _fuse(self,
               color_type: ColorCardType, color_conf: float,
               edge_type: ColorCardType, edge_conf: float
               ) -> Tuple[ColorCardType, float]:
        """
        融合颜色和纹路的识别结果

        规则:
        1. 两者一致 → 高置信，取加权平均
        2. 两者不一致 → 优先纹路（权重更高），但降低置信度
        3. 纹路置信度 > 0.6 → 信任纹路结果
        """
        if color_type == edge_type:
            # 一致：加权平均
            fused_conf = (color_conf * self.COLOR_WEIGHT +
                         edge_conf * self.EDGE_WEIGHT)
            return color_type, fused_conf

        # 不一致
        if edge_conf > 0.6:
            # 纹路高置信，信任纹路
            return edge_type, edge_conf * 0.8
        elif color_conf > 0.7:
            # 颜色高置信（但纹路不够高），信任颜色
            return color_type, color_conf * 0.7
        else:
            # 两者都不够高，优先纹路（更高权重）
            return edge_type, edge_conf * 0.6

    def _yolo_locate(self, frame) -> List[Tuple[int, int, int, int]]:
        """YOLO 定位卡片区域"""
        results = self._yolo.track(frame, conf=0.5, persist=True, verbose=False)
        regions = []

        if len(results) == 0 or results[0].boxes is None:
            return regions

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            regions.append((int(x1), int(y1), int(x2), int(y2)))

        return regions

    def _mock_locate(self, frame) -> List[Tuple[int, int, int, int]]:
        """
        Mock 定位模式：返回整帧作为候选区域。

        调试用。实际部署时应使用 YOLO 定位。
        """
        h, w = frame.shape[:2]
        # 返回中心区域作为候选
        margin = 50
        return [(margin, margin, w - margin, h - margin)]


# ---------------------------------------------------------------------------
# 模板采集
# ---------------------------------------------------------------------------

def collect_templates(camera_func=None,
                      output_path: str = "vision/color_card_templates.npz",
                      use_pidinet: bool = True):
    """
    采集颜色牌模板（在光线良好环境下执行一次）

    对每种颜色牌拍摄 3-5 张不同角度/位置的照片，提取 HSV 直方图和边缘图作为模板。

    Args:
        camera_func: 返回下一帧的函数。如 None，则使用 cv2 摄像头。
        output_path: 模板保存路径
        use_pidinet: 是否使用 PiDiNet（True）或 Canny（False）

    使用方式:
        # 方式1：使用 IP 摄像头
        from vision.ipcamera import IPCamera
        cam = IPCamera("http://10.54.71.31:8080/video")
        cam.connect()
        collect_templates(lambda: cam.read_frame())

        # 方式2：使用本地摄像头
        import cv2
        cap = cv2.VideoCapture(0)
        collect_templates(lambda: cap.read()[1])

        # 方式3：已有图像文件
        import glob
        imgs = glob.glob("color_cards/*.jpg")
        # 需手动实现批量采集
    """
    import torch

    tmpl_mgr = TemplateManager(output_path)

    # 加载 PiDiNet
    pidinet_model = None
    if use_pidinet:
        try:
            from vision.edge_detection_ipcam import load_pidinet
            pidinet_model, _ = load_pidinet()
            pidinet_model.eval()
            logger.info("PiDiNet 已加载")
        except Exception as e:
            logger.warning(f"PiDiNet 加载失败: {e}")
            pidinet_model = None

    edge_clf = EdgeClassifier(tmpl_mgr, pidinet_model, "cpu")

    for card_type in ColorCardType.all_types():
        print(f"\n{'='*50}")
        print(f"请放置【{card_type.value}】颜色牌")
        print("按 SPACE 键拍照（建议 3-5 张不同角度）")
        print("按 NEXT 键跳到下一种颜色牌")
        print("按 Q 键退出")
        print(f"{'='*50}")

        hsv_hist_sum = None
        edge_sum = None
        sample_count = 0

        while True:
            if camera_func:
                frame = camera_func()
            else:
                import cv2
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    continue

            # 显示预览
            cv2.imshow("采集模板", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # 空格：采集
                # 手动框选或使用整帧
                h, w = frame.shape[:2]
                # 简单处理：中心区域
                margin = 100
                card_crop = frame[margin:h-margin, margin:w-margin]

                # 提取 HSV 直方图
                hsv = cv2.cvtColor(card_crop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [180, 256],
                                   [0, 180, 0, 256])
                cv2.normalize(hist, hist)

                # 提取边缘
                if pidinet_model is not None:
                    try:
                        # PiDiNet 预处理
                        input_size = 640
                        scale = input_size / max(card_crop.shape[:2])
                        new_h = int(card_crop.shape[0] * scale)
                        new_w = int(card_crop.shape[1] * scale)
                        resized = cv2.resize(card_crop, (new_w, new_h))
                        square = np.zeros((input_size, input_size, 3), dtype=np.uint8)
                        square[:new_h, :new_w] = resized

                        img_float = square.astype(np.float32) / 255.0
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        img_norm = (img_float - mean) / std
                        img_tensor = torch.from_numpy(
                            img_norm.transpose(2, 0, 1)
                        ).float().unsqueeze(0)

                        with torch.no_grad():
                            outputs = pidinet_model(img_tensor)
                            edge = outputs[-1].squeeze().numpy()

                        edge_resized = cv2.resize(
                            edge, (card_crop.shape[1], card_crop.shape[0])
                        )
                    except Exception as e:
                        logger.warning(f"PiDiNet 推理失败: {e}")
                        gray = cv2.cvtColor(card_crop, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
                        edge_resized = cv2.Canny(blurred, 50, 150).astype(
                            np.float32) / 255.0
                else:
                    gray = cv2.cvtColor(card_crop, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
                    edge_resized = cv2.Canny(blurred, 50, 150).astype(
                        np.float32) / 255.0

                # 累加
                if hsv_hist_sum is None:
                    hsv_hist_sum = hist
                    edge_sum = edge_resized
                else:
                    hsv_hist_sum += hist
                    edge_sum += edge_resized
                sample_count += 1

                print(f"  已采集 {sample_count} 张")

            elif key == ord('n') or key == ord('q'):  # n: 下一种 / q: 退出
                break

        if sample_count > 0:
            # 平均
            hsv_hist_avg = hsv_hist_sum / sample_count
            edge_avg = edge_sum / sample_count

            tmpl_mgr.add_template(card_type, hsv_hist_avg, edge_avg)

            if key == ord('q'):
                break

    tmpl_mgr.save()
    print(f"\n模板采集完成，已保存到: {output_path}")

    if tmpl_mgr.is_ready:
        print("所有 6 种颜色牌模板就绪！")
    else:
        missing = [ct.value for ct in ColorCardType.all_types()
                   if not tmpl_mgr.has_template(ct)]
        print(f"警告：以下颜色牌缺少模板: {missing}")

    cv2.destroyAllWindows()
    return tmpl_mgr


# ---------------------------------------------------------------------------
# 坐标映射
# ---------------------------------------------------------------------------

def map_to_canvas(detection: ColorCardDetection,
                  frame_size: Tuple[int, int],
                  canvas_size: Tuple[int, int] = (1920, 1080)
                  ) -> Tuple[int, int]:
    """将检测到的像素坐标映射到 Unity 画布坐标"""
    fx = canvas_size[0] / frame_size[0]
    fy = canvas_size[1] / frame_size[1]
    return (
        int(detection.center[0] * fx),
        int(detection.center[1] * fy),
    )


def to_unity_message(detections: List[ColorCardDetection],
                     frame_size: Tuple[int, int],
                     canvas_size: Tuple[int, int] = (1920, 1080)
                     ) -> dict:
    """将检测结果转换为 Unity 消息格式"""
    cards = []
    for det in detections:
        canvas_pos = map_to_canvas(det, frame_size, canvas_size)
        cards.append({
            "card_type": det.card_type.value,
            "class_id": ColorCardType.to_class_id(det.card_type),
            "confidence": round(det.confidence, 4),
            "color_confidence": round(det.color_confidence, 4),
            "edge_confidence": round(det.edge_confidence, 4),
            "pixel_center": list(det.center),
            "canvas_center": list(canvas_pos),
            "bbox": list(det.bbox),
            "track_id": det.track_id,
        })

    return {
        "type": "color_card_detections",
        "count": len(cards),
        "cards": cards,
    }


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_color_card_detector(
    yolo_path: Optional[str] = None,
    template_path: str = "vision/color_card_templates.npz",
    device: str = "cpu",
    conf_threshold: float = 0.5
) -> DualColorCardDetector:
    """工厂函数：创建双识别颜色牌检测器"""
    return DualColorCardDetector(
        yolo_path=yolo_path,
        template_path=template_path,
        device=device,
        conf_threshold=conf_threshold,
    )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import cv2

    print("=" * 50)
    print("颜色牌双识别器自测")
    print("=" * 50)

    # 检查模板状态
    tmpl_path = "vision/color_card_templates.npz"
    tmpl_mgr = TemplateManager(tmpl_path)

    print(f"\n模板状态: {'已就绪' if tmpl_mgr.is_ready else '未就绪'}")
    if not tmpl_mgr.is_ready:
        missing = [ct.value for ct in ColorCardType.all_types()
                   if not tmpl_mgr.has_template(ct)]
        print(f"缺少模板: {missing}")
        print("\n提示: 运行 collect_templates() 采集模板")
    else:
        print("所有 6 种颜色牌模板就绪")

    # 尝试加载 YOLO
    yolo_path = "yolo/color_card.pt"
    detector = create_color_card_detector(
        yolo_path=yolo_path if os.path.exists(yolo_path) else None,
        template_path=tmpl_path
    )

    print(f"\nYOLO 模式: {'真实' if detector._yolo else 'Mock'}")
    print(f"PiDiNet: {'已加载' if detector._pidinet else '待加载'}")

    print("\n开启摄像头测试（按 Q 退出）...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        # 可视化
        for det in detections:
            x1, y1, w, h = det.bbox
            x2, y2 = x1 + w, y1 + h

            # 框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 标签
            label = f"{det.card_type.value} {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 置信度详情
            detail = f"C:{det.color_confidence:.2f} E:{det.edge_confidence:.2f}"
            cv2.putText(frame, detail, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Color Card Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n自测结束")
