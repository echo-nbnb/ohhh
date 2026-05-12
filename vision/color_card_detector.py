"""
YOLO 颜色牌检测器 — 第一幕"择色"

摄像头俯拍桌面，实时检测用户放置的颜色牌（6 种），
输出每张牌的类型 + 像素位置，经坐标映射后发送 Unity。

当前状态: 预留 YOLO 接口，mock 模式可跑通全链路。
模型训练后只需放入权重文件并设置 model_path 即可切换真实检测。

使用方式:
    detector = ColorCardDetector()               # mock 模式
    detector = ColorCardDetector(model_path="yolo/color_card.pt")  # 真实模式
    cards = detector.detect(frame)               # → List[ColorCardDetection]
"""

import logging
import random
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("ColorCardDetector")


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
        """YOLO class_id → ColorCardType"""
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
        """ColorCardType → YOLO class_id"""
        reverse = {v: k for k, v in {
            0: cls.YUELU_GREEN,
            1: cls.ACADEMY_RED,
            2: cls.XIQIAN_YELLOW,
            3: cls.XIANGJIANG_BLUE,
            4: cls.BADGE_GOLD,
            5: cls.INK_BLACK,
        }.items()}
        return reverse.get(card_type, 0)


@dataclass
class ColorCardDetection:
    """单张颜色牌检测结果"""
    card_type: ColorCardType
    confidence: float             # 置信度 [0, 1]
    center: Tuple[int, int]       # 像素中心 (x, y)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    track_id: Optional[int] = None   # BoT-SORT 多目标跟踪 ID（跨帧一致）


# ---------------------------------------------------------------------------
# 颜色牌检测器
# ---------------------------------------------------------------------------

class ColorCardDetector:
    """
    颜色牌检测器

    两种模式:
    - mock 模式 (model_path=None): 返回模拟检测结果，用于开发调试
    - 真实模式 (model_path 指向 .pt 文件): YOLOv8 推理

    帧率目标: 摄像头 30fps，YOLOv8n 在 GPU 上 ~10ms/帧，CPU 上 ~50ms/帧
    """

    # 6 色 HSV 参考范围（暗光环境，需布展时微调）
    # (h_min, s_min, v_min), (h_max, s_max, v_max)
    HSV_RANGES = {
        ColorCardType.YUELU_GREEN:     ((35, 50, 30), (85, 255, 200)),
        ColorCardType.ACADEMY_RED:     ((0, 50, 30), (10, 255, 200)),
        ColorCardType.XIQIAN_YELLOW:   ((20, 50, 50), (35, 255, 255)),
        ColorCardType.XIANGJIANG_BLUE: ((100, 50, 30), (130, 255, 200)),
        ColorCardType.BADGE_GOLD:      ((15, 80, 80), (30, 200, 255)),
        ColorCardType.INK_BLACK:       ((0, 0, 0), (180, 255, 60)),
    }

    def __init__(self, model_path: Optional[str] = None,
                 conf_threshold: float = 0.5,
                 frame_size: Tuple[int, int] = (640, 480)):
        """
        Args:
            model_path: YOLO .pt 权重路径。None 则使用 mock 模式。
            conf_threshold: 置信度阈值（仅真实模式生效）
            frame_size: 摄像头帧尺寸 (w, h)，mock 模式用于生成合理位置
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.frame_size = frame_size
        self._model = None
        self._use_mock = model_path is None

        if not self._use_mock:
            self._load_model()

    def _load_model(self):
        """加载 YOLO 模型"""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logger.info(f"YOLO 模型已加载: {self.model_path}")
            self._use_mock = False
        except FileNotFoundError:
            logger.warning(f"模型文件不存在: {self.model_path}，降级到 mock 模式")
            self._use_mock = True
        except ImportError:
            logger.warning("ultralytics 未安装，降级到 mock 模式")
            self._use_mock = True
        except Exception as e:
            logger.error(f"YOLO 加载失败: {e}，降级到 mock 模式")
            self._use_mock = True

    # ------------------------------------------------------------------
    # 检测入口
    # ------------------------------------------------------------------

    def detect(self, frame) -> List[ColorCardDetection]:
        """
        检测帧中的颜色牌

        Args:
            frame: BGR numpy 数组 (h, w, 3)

        Returns:
            检测到的颜色牌列表，按置信度降序
        """
        if self._use_mock or self._model is None:
            return self._mock_detect(frame)
        return self._yolo_detect(frame)

    # ------------------------------------------------------------------
    # YOLO 真实检测
    # ------------------------------------------------------------------

    def _yolo_detect(self, frame) -> List[ColorCardDetection]:
        """YOLOv8 推理 → ColorCardDetection 列表"""
        results = self._model.track(
            frame, conf=self.conf_threshold, persist=True, verbose=False
        )
        detections = []

        if len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else None

            w, h_box = x2 - x1, y2 - y1
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            detections.append(ColorCardDetection(
                card_type=ColorCardType.from_class_id(cls_id),
                confidence=conf,
                center=(cx, cy),
                bbox=(int(x1), int(y1), int(w), int(h_box)),
                track_id=track_id,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------
    # Mock 检测（无模型时）
    # ------------------------------------------------------------------

    def _mock_detect(self, frame) -> List[ColorCardDetection]:
        """
        Mock 模式：返回空列表。

        开发阶段可在此手动注入假数据来验证 Unity 通信链路。
        示例:
            return [
                ColorCardDetection(
                    card_type=ColorCardType.YUELU_GREEN,
                    confidence=0.95,
                    center=(320, 240),
                    bbox=(280, 200, 80, 80),
                )
            ]
        """
        return []

    # ------------------------------------------------------------------
    # 坐标映射（摄像头 → Unity 画布）
    # ------------------------------------------------------------------

    def map_to_canvas(self, detection: ColorCardDetection,
                      canvas_size: Tuple[int, int] = (1920, 1080)) -> Tuple[int, int]:
        """
        将检测到的像素坐标映射到 Unity 画布坐标

        Args:
            detection: 检测结果
            canvas_size: Unity 画布尺寸

        Returns:
            (x, y) Unity 画布坐标
        """
        fx = canvas_size[0] / self.frame_size[0]
        fy = canvas_size[1] / self.frame_size[1]
        return (
            int(detection.center[0] * fx),
            int(detection.center[1] * fy),
        )

    def to_unity_message(self, detections: List[ColorCardDetection],
                         canvas_size: Tuple[int, int] = (1920, 1080)) -> dict:
        """
        将检测结果转换为 Unity 消息格式

        Args:
            detections: 检测结果列表
            canvas_size: Unity 画布尺寸

        Returns:
            {"type": "color_card_detections", "cards": [...]}
        """
        cards = []
        for det in detections:
            canvas_pos = self.map_to_canvas(det, canvas_size)
            cards.append({
                "card_type": det.card_type.value,
                "class_id": ColorCardType.to_class_id(det.card_type),
                "confidence": round(det.confidence, 4),
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

def create_color_card_detector(model_path: Optional[str] = None,
                               conf_threshold: float = 0.5,
                               frame_size: Tuple[int, int] = (640, 480)) -> ColorCardDetector:
    """工厂函数：创建颜色牌检测器"""
    return ColorCardDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        frame_size=frame_size,
    )


# ---------------------------------------------------------------------------
# 训练指引（模型就绪后删除此注释块）
# ---------------------------------------------------------------------------
#
# 1. 采集数据
#    python yolo/collect_data.py --output dataset/images/train --prefix color_card
#
# 2. 标注 (LabelImg / Label Studio)
#    - 对每张图标注每张颜色牌的 bbox + class_id
#    - 导出 YOLO txt 格式到 dataset/labels/train/
#
# 3. 训练
#    python yolo/train_yolo.py
#    (修改 train_yolo.py 中的 data= 指向 yolo/dataset.yaml)
#
# 4. 部署
#    detector = ColorCardDetector(model_path="yolo/color_card.pt")
# ---------------------------------------------------------------------------
