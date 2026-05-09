"""
手势连接模块
通过手势识别实现物块之间的连接

流程：手悬停物块上 → 开始握拳（建立连接）→ 保持握拳移动到另一物块 → 另一物块响应 → 手张开（完成连接）
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from enum import Enum

# 导入现有手部检测模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.hand_tracker import HandTracker
from vision.hand_detector import HandDetector, HandLandmarkIndex


class ConnectionState(Enum):
    """连接状态机"""
    IDLE = "idle"                    # 空闲，无连接
    HOVERING = "hovering"            # 手悬停在物块上（未握拳）
    CONNECTING = "connecting"        # 握拳保持连接中（已选定起始物块）
    COMPLETING = "completing"        # 手张开，准备完成连接


class Module:
    """物块定义（对应物理模块）"""
    def __init__(self, module_id: str, module_type: str, position: Tuple[int, int], size: Tuple[int, int] = (80, 80)):
        self.id = module_id
        self.type = module_type  # color / object / character
        self.position = position  # 中心坐标 (x, y)
        self.size = size          # (width, height)

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """判断点是否在物块范围内"""
        x, y = point
        cx, cy = self.position
        w, h = self.size
        return (cx - w//2 <= x <= cx + w//2) and (cy - h//2 <= y <= cy + h//2)


class GestureConnection:
    """
    手势连接管理器

    检测手部位置和握拳状态，实现物块间的连接
    """

    def __init__(self, canvas_size: Tuple[int, int] = (1920, 1080)):
        """
        初始化手势连接管理器

        Args:
            canvas_size: Unity画布尺寸，用于手部坐标映射
        """
        self.canvas_size = canvas_size

        # 初始化手部检测器
        self.hand_tracker = HandTracker()
        self.hand_detector = HandDetector()

        # 物块列表（由外部注入）
        self.modules: List[Module] = []

        # 连接状态
        self.state = ConnectionState.IDLE
        self.start_module: Optional[Module] = None
        self.current_module: Optional[Module] = None
        self.last_hover_module: Optional[Module] = None

        # 手势识别结果
        self.current_gesture = "✋ 检测中..."
        self.is_fist = False  # 当前是否握拳
        self.was_fist = False  # 上一帧是否握拳（用于检测握拳开始）

        # 连接完成回调
        self.on_connection_complete: Optional[callable] = None

    def set_modules(self, modules: List[Module]):
        """设置物块列表"""
        self.modules = modules

    def set_connection_callback(self, callback):
        """设置连接完成时的回调函数

        Args:
            callback: function(connection_type, module_a, module_b)
        """
        self.on_connection_complete = callback

    def _recognize_fist(self, landmarks) -> bool:
        """
        判断是否握拳

        检测方法：所有指尖都在对应的MCP下方（y值更大）
        """
        if landmarks is None:
            return False

        lm = landmarks

        # 判断各手指是否伸展（指尖y < MCP y 表示伸展）
        # 握拳 = 所有手指都未伸展（指尖y > MCP y）
        index_extended = lm[HandLandmarkIndex.INDEX_TIP].y < lm[HandLandmarkIndex.INDEX_MCP].y - 0.02
        middle_extended = lm[HandLandmarkIndex.MIDDLE_TIP].y < lm[HandLandmarkIndex.MIDDLE_MCP].y - 0.02
        ring_extended = lm[HandLandmarkIndex.RING_TIP].y < lm[HandLandmarkIndex.RING_MCP].y - 0.02
        pinky_extended = lm[HandLandmarkIndex.PINKY_TIP].y < lm[HandLandmarkIndex.PINKY_MCP].y - 0.02

        # 握拳 = 所有手指都未伸展
        is_fist = not (index_extended or middle_extended or ring_extended or pinky_extended)

        return is_fist

    def _find_module_at_position(self, position: Tuple[int, int]) -> Optional[Module]:
        """查找指定位置上的物块"""
        for module in self.modules:
            if module.contains_point(position):
                return module
        return None

    def _get_palm_center_in_canvas(self, hand_data: dict) -> Tuple[int, int]:
        """获取手掌中心在画布坐标中的位置"""
        palm = hand_data.get("palm_center", (0, 0))
        return palm

    def _get_connection_type(self, module_a: Module, module_b: Module) -> str:
        """根据两个物块的类型确定连接类型"""
        type_a = module_a.type
        type_b = module_b.type

        # 颜色 ↔ 颜色：精神共鸣
        if type_a == "color" and type_b == "color":
            return "spiritual_resonance"
        # 人物 ↔ 人物：精神对话
        if type_a == "character" and type_b == "character":
            return "spirit_dialogue"
        # 物象 ↔ 物象：意境叠加
        if type_a == "object" and type_b == "object":
            return "scene_combination"
        # 颜色 ↔ 物象：精神赋予
        if (type_a == "color" and type_b == "object") or (type_a == "object" and type_b == "color"):
            return "color_grant"
        # 人物 ↔ 颜色：人格染色
        if (type_a == "character" and type_b == "color") or (type_a == "color" and type_b == "character"):
            return "personality_dye"
        # 物象 ↔ 人物：场景进入
        if (type_a == "object" and type_b == "character") or (type_a == "character" and type_b == "object"):
            return "scene_entry"

        return "unknown"

    def process_frame(self, frame: np.ndarray, timestamp_ms: int = 0) -> Dict:
        """
        处理视频帧，检测手势连接

        Args:
            frame: 输入图像（BGR格式）
            timestamp_ms: 时间戳（毫秒）

        Returns:
            {
                "state": 当前状态,
                "gesture": 手势描述,
                "is_fist": 是否握拳,
                "hovering_module": 悬停的物块（如果有）,
                "connecting_module": 连接中的物块（如果有）,
                "start_module": 连接起点（如果有）,
                "hand_data": 手部数据（如果有）
            }
        """
        result = {
            "state": self.state.value,
            "gesture": self.current_gesture,
            "is_fist": self.is_fist,
            "hovering_module": None,
            "connecting_module": None,
            "start_module": self.start_module.id if self.start_module else None,
            "hand_data": None
        }

        # 获取手部数据和手势
        hand_data = self.hand_tracker.get_hand_position(frame, timestamp_ms)

        if hand_data is None:
            self.current_gesture = "✋ 未检测到手"
            self.is_fist = False
            self.was_fist = False
            # 重置状态
            if self.state != ConnectionState.IDLE:
                self._reset_state()
            return result

        result["hand_data"] = hand_data

        # 获取手掌位置（画布坐标）
        palm_pos = self._get_palm_center_in_canvas(hand_data)
        result["palm_position"] = palm_pos

        # 检测手势
        results, landmarks = self.hand_detector.process_frame(frame, timestamp_ms)
        if landmarks:
            self.current_gesture = self.hand_detector.recognize_gesture(landmarks)
            self.is_fist = self._recognize_fist(landmarks)
        else:
            self.current_gesture = "✋ 未检测到手"
            self.is_fist = False

        result["gesture"] = self.current_gesture
        result["is_fist"] = self.is_fist

        # 查找手掌下的物块
        self.current_module = self._find_module_at_position(palm_pos)
        result["hovering_module"] = self.current_module.id if self.current_module else None

        # 状态机处理
        self._update_connection_state()

        result["state"] = self.state.value
        result["connecting_module"] = self.current_module.id if self.current_module else None

        return result

    def _update_connection_state(self):
        """更新连接状态机"""
        if self.state == ConnectionState.IDLE:
            # 空闲状态
            if self.current_module and not self.is_fist:
                # 手悬停在物块上，但未握拳
                self.state = ConnectionState.HOVERING
                self.last_hover_module = self.current_module

            elif self.is_fist and self.current_module:
                # 开始握拳，选定起始物块
                self.state = ConnectionState.CONNECTING
                self.start_module = self.current_module

        elif self.state == ConnectionState.HOVERING:
            # 悬停状态
            if self.is_fist and self.current_module:
                # 开始握拳，建立连接
                self.state = ConnectionState.CONNECTING
                self.start_module = self.current_module
                self.last_hover_module = None

            elif not self.current_module or self.current_module != self.last_hover_module:
                # 手离开悬停物块
                self.last_hover_module = None
                self.state = ConnectionState.IDLE

        elif self.state == ConnectionState.CONNECTING:
            # 连接中状态（保持握拳移动）
            if not self.is_fist:
                # 手张开，判断是完成还是取消
                if self.current_module and self.current_module != self.start_module:
                    # 手张开时在目标物块上 → 进入完成状态
                    self.state = ConnectionState.COMPLETING
                else:
                    # 手张开时不在目标物块上 → 取消连接
                    self._reset_state()
            # else: 保持握拳，持续监测目标物块

        elif self.state == ConnectionState.COMPLETING:
            # 完成状态，等待手张开后重置
            if not self.is_fist:
                # 手张开，完成连接
                self._complete_connection()
                self._reset_state()

    def _complete_connection(self):
        """完成连接"""
        if self.start_module and self.current_module and self.start_module != self.current_module:
            connection_type = self._get_connection_type(self.start_module, self.current_module)

            print(f"[连接] {self.start_module.id} → {self.current_module.id} ({connection_type})")

            if self.on_connection_complete:
                self.on_connection_complete(connection_type, self.start_module, self.current_module)

    def _reset_state(self):
        """重置状态"""
        self.state = ConnectionState.IDLE
        self.start_module = None
        self.current_module = None
        self.last_hover_module = None

    def draw_debug_visualization(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        在画面上绘制调试信息

        Args:
            frame: 输入图像
            result: process_frame返回的结果

        Returns:
            绘制了调试信息的图像
        """
        display = frame.copy()

        # 绘制物块区域
        for module in self.modules:
            x, y = module.position
            w, h = module.size

            # 根据状态选择颜色
            if module == self.start_module:
                color = (0, 255, 255)  # 黄色：连接起点
            elif module == self.current_module and self.state == ConnectionState.CONNECTING:
                color = (0, 255, 0)    # 绿色：可连接目标
            elif module == self.current_module:
                color = (255, 255, 0)  # 浅蓝色：悬停中
            else:
                color = (100, 100, 100)  # 灰色：普通状态

            cv2.rectangle(display, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 2)
            cv2.putText(display, module.id, (x - w//2 + 5, y - h//2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 绘制手部位置
        if result.get("hand_data"):
            palm = result["hand_data"].get("palm_center", (0, 0))
            cv2.circle(display, palm, 15, (0, 255, 255), -1)
            cv2.putText(display, f"手掌: ({palm[0]}, {palm[1]})", (palm[0] + 20, palm[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 绘制状态信息
        state_text = f"状态: {self.state.value}"
        gesture_text = f"手势: {result.get('gesture', '未知')}"
        fist_text = f"握拳: {result.get('is_fist', False)}"

        cv2.putText(display, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, fist_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # 绘制连接线（如果正在连接）
        if self.start_module and self.current_module and self.state == ConnectionState.CONNECTING:
            start_pos = self.start_module.position
            end_pos = self.current_module.position
            cv2.line(display, start_pos, end_pos, (0, 255, 255), 3)

        return display

    def close(self):
        """关闭检测器"""
        if self.hand_tracker:
            self.hand_tracker.close()
        if self.hand_detector:
            self.hand_detector.close()


def create_gesture_connection(canvas_size: Tuple[int, int] = (1920, 1080)) -> GestureConnection:
    """工厂函数：创建手势连接管理器"""
    return GestureConnection(canvas_size)


# 测试代码
if __name__ == "__main__":
    import sys

    # 简单测试
    gc = GestureConnection()

    # 创建模拟物块
    modules = [
        Module("color_1", "color", (400, 300)),
        Module("object_1", "object", (600, 300)),
        Module("character_1", "character", (500, 450)),
    ]
    gc.set_modules(modules)

    def on_connection(connection_type, module_a, module_b):
        print(f"连接完成: {module_a.id} ↔ {module_b.id} (类型: {connection_type})")

    gc.set_connection_callback(on_connection)

    print("[测试] 手势连接模块已创建")
    print(f"[测试] 物块数量: {len(gc.modules)}")

    gc.close()