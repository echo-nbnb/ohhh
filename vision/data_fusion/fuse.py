#!/usr/bin/env python3
"""
数据融合模块
将YOLO视觉检测结果与ESP32模块数据进行融合

融合策略（YOLO为主，ESP32为备）：
1. YOLO检测到模块的坐标位置和类型（主数据源）
2. ESP32上报模块的ID和触摸状态（辅助确认）
3. 根据位置将YOLO检测结果与ESP32数据关联
4. 无YOLO检测时，使用ESP32数据（降级模式）
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ModuleType(Enum):
    """模块类型枚举"""
    COLOR = "color"
    OBJECT = "object"
    CHARACTER = "character"
    CONFIRM = "confirm"  # 确认章
    UNKNOWN = "unknown"


@dataclass
class YOLODetection:
    """YOLO检测结果"""
    class_id: int           # YOLO类别ID
    class_name: str         # 类别名称
    confidence: float        # 置信度
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]  # 中心点 (x, y)
    track_id: Optional[int] = None  # 多目标跟踪ID（跨帧一致）
    timestamp: float = 0    # 检测时间戳


@dataclass
class ESP32ModuleData:
    """ESP32模块数据"""
    module_id: str          # 模块唯一ID
    module_type: ModuleType # 模块类型
    touch_state: bool       # 是否被触摸
    online: bool = True     # 是否在线
    timestamp: float = 0    # 时间戳


@dataclass
class FusedModuleData:
    """融合后的模块数据"""
    module_id: str              # 模块ID
    module_type: ModuleType     # 模块类型
    position: Tuple[int, int]   # 位置 (x, y)
    touch_state: bool           # 触摸状态
    confidence: float = 1.0     # 置信度
    bbox: Optional[Tuple[int, int, int, int]] = None  # 边界框
    track_id: Optional[int] = None  # MOT跟踪ID（跨帧一致，用于多模块区分）
    timestamp: float = 0        # 时间戳


class DataFusion:
    """
    数据融合器
    将YOLO检测结果与ESP32模块数据进行融合
    """

    def __init__(self,
                 max_distance: int = 100,
                 id_mapping: Optional[Dict[int, ModuleType]] = None,
                 yolo_primary: bool = True):
        """
        Args:
            max_distance: 模块ID与检测位置的最大匹配距离（像素）
            id_mapping: YOLO class_id 到 ModuleType 的映射
            yolo_primary: True=YOLO为主（默认），False=ESP32为主
        """
        self.max_distance = max_distance
        self.id_mapping = id_mapping or self._default_id_mapping()
        self.yolo_primary = yolo_primary

        # 存储当前帧的检测结果和模块数据
        self.current_detections: List[YOLODetection] = []
        self.current_modules: Dict[str, ESP32ModuleData] = {}

        # 输出结果
        self.fused_modules: List[FusedModuleData] = []

        # 位置缓存：用于稳定匹配（避免单帧抖动）
        self.position_cache: Dict[str, Tuple[int, int]] = {}

    def _default_id_mapping(self) -> Dict[int, ModuleType]:
        """默认的ID映射（需根据实际训练结果调整）"""
        return {
            0: ModuleType.COLOR,      # 颜色块
            1: ModuleType.OBJECT,     # 物象块
            2: ModuleType.CHARACTER,  # 人物块
            3: ModuleType.CONFIRM,    # 确认章
        }

    def update_yolo_detections(self, detections: List[Dict[str, Any]]):
        """
        更新YOLO检测结果

        Args:
            detections: YOLO检测结果列表
                        每项包含: class_id, class_name, confidence, bbox, center
        """
        self.current_detections = []
        for det in detections:
            if det.get('confidence', 0) < 0.5:
                continue

            bbox = det['bbox']
            center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

            detection = YOLODetection(
                class_id=det['class_id'],
                class_name=det.get('class_name', 'unknown'),
                confidence=det.get('confidence', 0),
                bbox=bbox,
                center=center,
                track_id=det.get('track_id'),
                timestamp=det.get('timestamp', 0)
            )
            self.current_detections.append(detection)

    def update_esp32_modules(self, modules: List[Dict[str, Any]]):
        """
        更新ESP32模块数据

        Args:
            modules: ESP32模块数据列表
                     每项包含: module_id, module_type, touch_state, online
        """
        self.current_modules = {}
        for mod in modules:
            try:
                module_type = ModuleType(mod.get('module_type', 'unknown'))
            except ValueError:
                module_type = ModuleType.UNKNOWN

            module_data = ESP32ModuleData(
                module_id=mod['module_id'],
                module_type=module_type,
                touch_state=mod.get('touch_state', False),
                online=mod.get('online', True),
                timestamp=mod.get('timestamp', 0)
            )
            self.current_modules[module_data.module_id] = module_data

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """计算两点之间的欧氏距离"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _match_detections_to_modules(self) -> List[Tuple[YOLODetection, ESP32ModuleData, float]]:
        """
        将检测结果与模块数据进行匹配

        Returns:
            匹配列表: [(检测结果, 模块数据, 距离), ...]
        """
        matches = []

        # 对于每个检测结果，找到最近的模块
        for det in self.current_detections:
            best_match = None
            best_distance = float('inf')

            for module_id, module_data in self.current_modules.items():
                if not module_data.online:
                    continue

                # 使用缓存的位置（如果可用）
                if module_id in self.position_cache:
                    cached_pos = self.position_cache[module_id]
                    distance = self._calculate_distance(det.center, cached_pos)
                else:
                    # 首次匹配，使用检测中心
                    distance = self._calculate_distance(det.center, det.center)

                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_match = module_data

            if best_match is not None:
                matches.append((det, best_match, best_distance))

        return matches

    def fuse(self) -> List[FusedModuleData]:
        """
        执行数据融合

        YOLO为主模式：
        - 以YOLO检测为主数据源，直接使用YOLO的坐标和类型
        - ESP32数据用于辅助确认（如触摸状态）

        Returns:
            融合后的模块数据列表
        """
        self.fused_modules = []

        if self.yolo_primary:
            # YOLO为主模式
            self._fuse_yolo_primary()
        else:
            # ESP32为主模式（原逻辑）
            self._fuse_esp32_primary()

        return self.fused_modules

    def _fuse_yolo_primary(self):
        """
        YOLO为主的数据融合

        1. YOLO检测结果直接作为主数据
        2. 尝试匹配ESP32数据获取触摸状态
        3. 无ESP32匹配时，触摸状态默认False
        """
        matched_esp32_ids = set()

        for det in self.current_detections:
            # 从YOLO检测直接创建融合数据
            # 尝试从ESP32获取触摸状态
            touch_state = False
            # 优先用 track_id 生成模块ID（保证跨帧一致），无 tracker 时降级到坐标
            if det.track_id is not None:
                matched_module_id = f"yolo_track_{det.track_id}"
            else:
                matched_module_id = f"yolo_{det.class_id}_{det.center[0]}_{det.center[1]}"

            # 尝试匹配ESP32数据（根据位置匹配）
            for module_id, module_data in self.current_modules.items():
                if not module_data.online:
                    continue
                if module_id in self.position_cache:
                    cached_pos = self.position_cache[module_id]
                    distance = self._calculate_distance(det.center, cached_pos)
                    if distance < self.max_distance:
                        touch_state = module_data.touch_state
                        matched_module_id = module_id
                        matched_esp32_ids.add(module_id)
                        self.position_cache[module_id] = det.center
                        break

            # 获取模块类型（从YOLO class_id映射）
            module_type = self.id_mapping.get(det.class_id, ModuleType.UNKNOWN)

            fused = FusedModuleData(
                module_id=matched_module_id,
                module_type=module_type,
                position=det.center,
                touch_state=touch_state,
                confidence=det.confidence,
                bbox=det.bbox,
                track_id=det.track_id,
                timestamp=det.timestamp
            )
            self.fused_modules.append(fused)

        # 处理有ESP32数据但无YOLO匹配的（降级模式：仅ESP32）
        for module_id, module_data in self.current_modules.items():
            if module_id not in matched_esp32_ids and module_data.online:
                if module_id in self.position_cache:
                    position = self.position_cache[module_id]
                else:
                    position = (-1, -1)

                fused = FusedModuleData(
                    module_id=module_id,
                    module_type=module_data.module_type,
                    position=position,
                    touch_state=module_data.touch_state,
                    confidence=0.3,
                    bbox=None,
                    timestamp=module_data.timestamp
                )
                self.fused_modules.append(fused)

    def _fuse_esp32_primary(self):
        """
        ESP32为主的数据融合（原逻辑）
        """
        # 执行匹配
        matches = self._match_detections_to_modules()

        # 已匹配的模块ID集合
        matched_module_ids = set()

        for det, module_data, distance in matches:
            # 更新位置缓存
            self.position_cache[module_data.module_id] = det.center
            matched_module_ids.add(module_data.module_id)

            # 创建融合数据
            fused = FusedModuleData(
                module_id=module_data.module_id,
                module_type=module_data.module_type,
                position=det.center,
                touch_state=module_data.touch_state,
                confidence=det.confidence,
                bbox=det.bbox,
                timestamp=det.timestamp or module_data.timestamp
            )
            self.fused_modules.append(fused)

        # 处理未匹配的在线模块（只有ESP32数据，没有视觉检测）
        for module_id, module_data in self.current_modules.items():
            if module_id not in matched_module_ids and module_data.online:
                # 使用缓存的位置或默认值
                if module_id in self.position_cache:
                    position = self.position_cache[module_id]
                else:
                    position = (-1, -1)  # 未知位置

                fused = FusedModuleData(
                    module_id=module_id,
                    module_type=module_type,
                    position=position,
                    touch_state=module_data.touch_state,
                    confidence=0.3,  # 低置信度（推测）
                    bbox=None,
                    timestamp=module_data.timestamp
                )
                self.fused_modules.append(fused)

    def get_fused_data(self) -> List[FusedModuleData]:
        """获取融合结果"""
        return self.fused_modules

    def get_module_by_id(self, module_id: str) -> Optional[FusedModuleData]:
        """根据ID获取融合后的模块数据"""
        for module in self.fused_modules:
            if module.module_id == module_id:
                return module
        return None

    def get_modules_by_type(self, module_type: ModuleType) -> List[FusedModuleData]:
        """根据类型获取模块列表"""
        return [m for m in self.fused_modules if m.module_type == module_type]

    def clear(self):
        """清空当前数据"""
        self.current_detections.clear()
        self.current_modules.clear()
        self.fused_modules.clear()

    def clear_cache(self):
        """清空位置缓存"""
        self.position_cache.clear()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于输出到Unity）"""
        return {
            "fused_modules": [
                {
                    "module_id": m.module_id,
                    "module_type": m.module_type.value,
                    "position": {"x": m.position[0], "y": m.position[1]},
                    "touch_state": m.touch_state,
                    "confidence": m.confidence,
                    "track_id": m.track_id,
                    "bbox": {
                        "x": m.bbox[0] if m.bbox else 0,
                        "y": m.bbox[1] if m.bbox else 0,
                        "w": m.bbox[2] if m.bbox else 0,
                        "h": m.bbox[3] if m.bbox else 0
                    } if m.bbox else None
                }
                for m in self.fused_modules
            ],
            "timestamp": self.fused_modules[0].timestamp if self.fused_modules else 0
        }


class DataFusionPipeline:
    """
    数据融合管道
    将YOLO检测流程和ESP32数据接收流程整合
    """

    def __init__(self, max_distance: int = 100):
        self.fusion = DataFusion(max_distance=max_distance)
        self.running = False

    def process_yolo_results(self, results: List[Dict[str, Any]]):
        """处理YOLO检测结果"""
        self.fusion.update_yolo_detections(results)

    def process_esp32_data(self, modules: List[Dict[str, Any]]):
        """处理ESP32模块数据"""
        self.fusion.update_esp32_modules(modules)

    def fuse(self) -> List[FusedModuleData]:
        """执行融合"""
        return self.fusion.fuse()

    def get_output(self) -> Dict[str, Any]:
        """获取输出数据"""
        return self.fusion.to_dict()

    def reset(self):
        """重置管道"""
        self.fusion.clear()


def test_fusion():
    """测试数据融合"""
    fusion = DataFusion(max_distance=100)

    # 模拟YOLO检测结果
    yolo_detections = [
        {
            'class_id': 0,
            'class_name': 'color',
            'confidence': 0.95,
            'bbox': (100, 100, 50, 50),
            'center': (125, 125)
        },
        {
            'class_id': 1,
            'class_name': 'object',
            'confidence': 0.88,
            'bbox': (300, 200, 60, 60),
            'center': (330, 230)
        }
    ]

    # 模拟ESP32模块数据
    esp32_modules = [
        {
            'module_id': 'esp32_001',
            'module_type': 'color',
            'touch_state': False,
            'online': True
        },
        {
            'module_id': 'esp32_002',
            'module_type': 'object',
            'touch_state': True,
            'online': True
        }
    ]

    # 执行融合
    fusion.update_yolo_detections(yolo_detections)
    fusion.update_esp32_modules(esp32_modules)
    result = fusion.fuse()

    print("融合结果:")
    for r in result:
        print(f"  {r.module_id}: {r.module_type.value} at {r.position}, touch={r.touch_state}")

    print("\n输出字典:")
    print(fusion.to_dict())


if __name__ == "__main__":
    test_fusion()
