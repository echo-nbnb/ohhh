"""
手势状态机（多模式）
支持交互文档 interaction.md §2.2 定义的全体手势模式

模式层级:
    GLOBAL          ← 全局（握拳晕染、张手停止）
    DRAWING         ← 绘画（第二幕，食指伸出画画）
    CANDIDATE       ← 候选物象确认（第二幕，悬停+握拳确认）
    CHAR_RECOMMEND  ← 人物推荐确认（第三幕阶段一）
    CHAR_WHEEL      ← 人物轮盘浏览（第三幕阶段二）

用法:
    fsm = GestureStateMachine()
    fsm.on_mode_change = lambda mode, sub: sender.send_gesture_state(mode, sub, fsm.current_gesture)
    fsm.on_drawing_commit = lambda trajectory: bridge.commit()

    # 每帧调用
    fsm.process(landmarks, timestamp_ms)
"""

import time
import math
import logging
from typing import List, Optional, Tuple, Callable
from enum import Enum

logger = logging.getLogger("GestureFSM")


class GestureMode(Enum):
    GLOBAL = "GLOBAL"
    DRAWING = "DRAWING"
    CANDIDATE = "CANDIDATE"
    CHAR_RECOMMEND = "CHAR_RECOMMEND"
    CHAR_WHEEL = "CHAR_WHEEL"


class GestureType(Enum):
    NONE = "none"
    INDEX_POINTING = "index_pointing"
    FIST = "fist"
    OPEN_HAND = "open_hand"
    UNKNOWN = "unknown"


class DrawingSubState(Enum):
    IDLE = "idle"            # 未开始
    TRACKING = "tracking"    # 食指伸出，录制中
    COMPLETED = "completed"  # 握拳提交
    CANCELLED = "cancelled"  # 张手取消


class CandidateSubState(Enum):
    BROWSING = "browsing"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class CharRecommendSubState(Enum):
    BROWSING = "browsing"
    CONFIRMED = "confirmed"
    TO_WHEEL = "to_wheel"


class CharWheelSubState(Enum):
    SCROLLING = "scrolling"
    PREVIEWING = "previewing"
    CONFIRMED = "confirmed"
    TO_RECOMMEND = "to_recommend"


# ---------------------------------------------------------------------------
# 手势状态机
# ---------------------------------------------------------------------------

class GestureStateMachine:
    """
    多模式手势状态机

    每帧调用 process(landmarks, timestamp_ms)，
    内部自动检测手势、管理模式切换、触发回调。
    """

    def __init__(self):
        # 当前模式
        self.mode: GestureMode = GestureMode.GLOBAL
        self.drawing_sub: DrawingSubState = DrawingSubState.IDLE
        self.candidate_sub: CandidateSubState = CandidateSubState.BROWSING
        self.recommend_sub: CharRecommendSubState = CharRecommendSubState.BROWSING
        self.wheel_sub: CharWheelSubState = CharWheelSubState.SCROLLING

        # 当前手势
        self.current_gesture: GestureType = GestureType.NONE
        self.prev_gesture: GestureType = GestureType.NONE

        # 绘画轨迹
        self.trajectory: List[Tuple[float, float, int]] = []  # [(x, y, ts_ms), ...]

        # 手部状态
        self.palm_position: Tuple[float, float] = (0, 0)
        self.prev_palm_x: float = 0.0
        self.hand_detected: bool = False

        # 静止计时器（悬停预览）
        self._still_start: float = 0.0
        self._still_threshold: float = 1.0  # 秒
        self._still_move_threshold: float = 20.0  # 像素

        # 回调
        self.on_mode_change: Optional[Callable[[str, str, str], None]] = None
        self.on_drawing_start: Optional[Callable] = None
        self.on_drawing_commit: Optional[Callable[[List[Tuple[float, float, int]]], None]] = None
        self.on_drawing_cancel: Optional[Callable] = None
        self.on_object_confirmed: Optional[Callable] = None
        self.on_character_confirmed: Optional[Callable] = None
        self.on_reject_recommendations: Optional[Callable] = None
        self.on_hand_moved: Optional[Callable[[float], None]] = None  # delta_x

    # ------------------------------------------------------------------
    # 每帧处理
    # ------------------------------------------------------------------

    def process(self, landmarks: Optional[List], timestamp_ms: int = 0):
        """
        每帧调用

        Args:
            landmarks: MediaPipe hand landmarks (21 个 NormalizedLandmark) 或 None
            timestamp_ms: 毫秒时间戳
        """
        if landmarks is None or len(landmarks) < 21:
            self.hand_detected = False
            self.current_gesture = GestureType.NONE
            return

        self.hand_detected = True
        self.prev_gesture = self.current_gesture
        self.current_gesture = self._recognize_gesture(landmarks)

        # 更新手掌位置
        self.prev_palm_x = self.palm_position[0]
        palm = landmarks[0]  # wrist as rough palm center
        self.palm_position = (palm.x if hasattr(palm, 'x') else palm[0],
                              palm.y if hasattr(palm, 'y') else palm[1])

        # 检测手势变化
        gesture_changed = (self.current_gesture != self.prev_gesture)

        # 更新静止计时器
        dx = abs(self.palm_position[0] - self.prev_palm_x)
        if dx < self._still_move_threshold:
            if self._still_start == 0:
                self._still_start = time.time()
        else:
            self._still_start = 0.0

        # ── 模式切换 ──────────────────────────

        if self.mode == GestureMode.GLOBAL:
            self._process_global(gesture_changed, landmarks, timestamp_ms)

        elif self.mode == GestureMode.DRAWING:
            self._process_drawing(gesture_changed, landmarks, timestamp_ms)

        elif self.mode == GestureMode.CANDIDATE:
            self._process_candidate(gesture_changed)

        elif self.mode == GestureMode.CHAR_RECOMMEND:
            self._process_char_recommend(gesture_changed)

        elif self.mode == GestureMode.CHAR_WHEEL:
            self._process_char_wheel(gesture_changed, timestamp_ms)

    # ------------------------------------------------------------------
    # 各模式处理
    # ------------------------------------------------------------------

    def _process_global(self, gesture_changed: bool, landmarks: List, ts_ms: int):
        if gesture_changed and self.current_gesture == GestureType.INDEX_POINTING:
            # 食指伸出 → 进入绘画模式
            self._transition_to(GestureMode.DRAWING, "TRACKING")
            self.drawing_sub = DrawingSubState.TRACKING

    def _process_drawing(self, gesture_changed: bool, landmarks: List, ts_ms: int):
        if self.drawing_sub == DrawingSubState.TRACKING:
            # 录制食指指尖轨迹 (landmark 8)
            if self.current_gesture == GestureType.INDEX_POINTING:
                tip = landmarks[8]
                x = tip.x if hasattr(tip, 'x') else tip[0]
                y = tip.y if hasattr(tip, 'y') else tip[1]
                self.trajectory.append((float(x), float(y), ts_ms))

            if gesture_changed:
                if self.current_gesture == GestureType.FIST:
                    # 握拳 → 完成绘画
                    self.drawing_sub = DrawingSubState.COMPLETED
                    self._transition_to(GestureMode.CANDIDATE, "BROWSING")
                    if self.on_drawing_commit:
                        self.on_drawing_commit(self.trajectory)
                elif self.current_gesture == GestureType.OPEN_HAND:
                    # 张手 → 取消
                    self.drawing_sub = DrawingSubState.CANCELLED
                    self._transition_to(GestureMode.GLOBAL, "IDLE")
                    self.trajectory.clear()
                    if self.on_drawing_cancel:
                        self.on_drawing_cancel()

    def _process_candidate(self, gesture_changed: bool):
        if gesture_changed:
            if self.current_gesture == GestureType.FIST:
                self.candidate_sub = CandidateSubState.CONFIRMED
                self._transition_to(GestureMode.GLOBAL, "IDLE")
                self.trajectory.clear()
                if self.on_object_confirmed:
                    self.on_object_confirmed()
            elif self.current_gesture == GestureType.OPEN_HAND:
                self.candidate_sub = CandidateSubState.CANCELLED
                self._transition_to(GestureMode.DRAWING, "TRACKING")
                self.drawing_sub = DrawingSubState.TRACKING
                self.trajectory.clear()

    def _process_char_recommend(self, gesture_changed: bool):
        if gesture_changed:
            if self.current_gesture == GestureType.FIST:
                self.recommend_sub = CharRecommendSubState.CONFIRMED
                self._transition_to(GestureMode.GLOBAL, "IDLE")
                if self.on_character_confirmed:
                    self.on_character_confirmed()
            elif self.current_gesture == GestureType.OPEN_HAND:
                self.recommend_sub = CharRecommendSubState.TO_WHEEL
                self._transition_to(GestureMode.CHAR_WHEEL, "SCROLLING")
                if self.on_reject_recommendations:
                    self.on_reject_recommendations()

    def _process_char_wheel(self, gesture_changed: bool, ts_ms: int):
        # 水平位移 → 轮盘滚动
        dx = self.palm_position[0] - self.prev_palm_x
        if abs(dx) > 2 and self.on_hand_moved:
            self.wheel_sub = CharWheelSubState.SCROLLING
            self.on_hand_moved(dx)

        # 静止 > 1s → 预览
        if (self._still_start > 0 and
                time.time() - self._still_start > self._still_threshold):
            if self.wheel_sub != CharWheelSubState.PREVIEWING:
                self.wheel_sub = CharWheelSubState.PREVIEWING
                self._notify_mode_change("PREVIEWING")

        if gesture_changed:
            if self.current_gesture == GestureType.FIST:
                self.wheel_sub = CharWheelSubState.CONFIRMED
                self._transition_to(GestureMode.GLOBAL, "IDLE")
                if self.on_character_confirmed:
                    self.on_character_confirmed()
            elif self.current_gesture == GestureType.OPEN_HAND:
                self.wheel_sub = CharWheelSubState.TO_RECOMMEND
                self._transition_to(GestureMode.CHAR_RECOMMEND, "BROWSING")

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------

    def _transition_to(self, mode: GestureMode, sub_state: str):
        old_mode = self.mode.value
        self.mode = mode
        self._still_start = 0.0
        self._notify_mode_change(sub_state)
        logger.info(f"GestureFSM: {old_mode} → {mode.value} ({sub_state})")

    def _notify_mode_change(self, sub_state: str):
        if self.on_mode_change:
            gesture = self.current_gesture.value if self.current_gesture else "none"
            self.on_mode_change(self.mode.value, sub_state, gesture)

    # ------------------------------------------------------------------
    # 外部触发
    # ------------------------------------------------------------------

    def trigger_char_recommend(self):
        """外部触发：进入人物推荐模式（RAG 推荐完成后调用）"""
        self._transition_to(GestureMode.CHAR_RECOMMEND, "BROWSING")
        self.recommend_sub = CharRecommendSubState.BROWSING

    def trigger_object_candidates(self):
        """外部触发：显示物象候选（从 DRAWING.COMPLETED → CANDIDATE）"""
        self._transition_to(GestureMode.CANDIDATE, "BROWSING")
        self.candidate_sub = CandidateSubState.BROWSING

    def reset_to_global(self):
        """强制回到全局模式"""
        self._transition_to(GestureMode.GLOBAL, "IDLE")
        self.trajectory.clear()

    # ------------------------------------------------------------------
    # 手势识别
    # ------------------------------------------------------------------

    def _recognize_gesture(self, landmarks) -> GestureType:
        """
        从 21 个 MediaPipe landmarks 识别手势

        Returns:
            GestureType 枚举值
        """
        # 提取关键点 y 坐标
        lm = landmarks
        get_y = lambda idx: lm[idx].y if hasattr(lm[idx], 'y') else lm[idx][1]
        get_x = lambda idx: lm[idx].x if hasattr(lm[idx], 'x') else lm[idx][0]

        # 指尖和 MCP 的 y 坐标
        tips = {
            "index": (get_y(8), get_y(6)),   # tip, mcp
            "middle": (get_y(12), get_y(10)),
            "ring": (get_y(16), get_y(14)),
            "pinky": (get_y(20), get_y(18)),
        }

        # 手指伸展检测：指尖 y < MCP y（MediaPipe 坐标系 y 向下）
        threshold = 0.03
        extended = {
            name: tip_y < mcp_y - threshold
            for name, (tip_y, mcp_y) in tips.items()
        }

        all_extended = all(extended.values())
        none_extended = not any(extended.values())
        only_index = (extended["index"] and
                      not extended["middle"] and
                      not extended["ring"] and
                      not extended["pinky"])

        # 食指单独伸展 → 绘画模式
        if only_index:
            return GestureType.INDEX_POINTING

        # 全部伸展 → 张手
        if all_extended:
            return GestureType.OPEN_HAND

        # 全部握起 → 握拳
        if none_extended:
            return GestureType.FIST

        return GestureType.NONE

    # ------------------------------------------------------------------
    # 子状态查询
    # ------------------------------------------------------------------

    @property
    def sub_state(self) -> str:
        if self.mode == GestureMode.DRAWING:
            return self.drawing_sub.value
        elif self.mode == GestureMode.CANDIDATE:
            return self.candidate_sub.value
        elif self.mode == GestureMode.CHAR_RECOMMEND:
            return self.recommend_sub.value
        elif self.mode == GestureMode.CHAR_WHEEL:
            return self.wheel_sub.value
        return "IDLE"

    @property
    def is_drawing(self) -> bool:
        return self.mode == GestureMode.DRAWING and self.drawing_sub == DrawingSubState.TRACKING

    @property
    def is_fist(self) -> bool:
        return self.current_gesture == GestureType.FIST

    @property
    def is_open_hand(self) -> bool:
        return self.current_gesture == GestureType.OPEN_HAND

    @property
    def is_index_pointing(self) -> bool:
        return self.current_gesture == GestureType.INDEX_POINTING

    @property
    def trajectory_point_count(self) -> int:
        return len(self.trajectory)

    @property
    def is_still_previewing(self) -> bool:
        if self._still_start == 0:
            return False
        return time.time() - self._still_start > self._still_threshold


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_gesture_state_machine() -> GestureStateMachine:
    return GestureStateMachine()


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # 模拟 landmarks（用 namedtuple 或简单对象）
    class FakeLM:
        def __init__(self, x, y, z=0):
            self.x = x; self.y = y; self.z = z

    def make_fist():
        tips_up = [FakeLM(0.5, 0.9)] * 21  # 指尖低 → 握拳
        return tips_up

    def make_open():
        tips_up = [FakeLM(0.5, 0.2)] * 21  # 指尖高 → 张手
        return tips_up

    def make_index_point():
        lm = [FakeLM(0.5, 0.5)] * 21
        lm[8] = FakeLM(0.5, 0.1)   # 食指 tip 高（伸展）
        lm[6] = FakeLM(0.5, 0.4)   # 食指 MCP
        lm[12] = FakeLM(0.5, 0.8)  # 中指 tip 低（弯曲）
        lm[10] = FakeLM(0.5, 0.5)
        lm[16] = FakeLM(0.5, 0.8)
        lm[14] = FakeLM(0.5, 0.5)
        lm[20] = FakeLM(0.5, 0.8)
        lm[18] = FakeLM(0.5, 0.5)
        return lm

    fsm = create_gesture_state_machine()
    fsm.on_mode_change = lambda m, s, g: print(f"  → Unity: mode={m} sub={s} gesture={g}")
    fsm.on_drawing_commit = lambda traj: print(f"  → SketchBridge.commit() len={len(traj)}")

    print("=== 测试 1: GLOBAL + 食指伸出 → DRAWING ===")
    fsm.process(make_open(), 0)
    print(f"  mode={fsm.mode.value} sub={fsm.sub_state} gesture={fsm.current_gesture.value}")
    fsm.process(make_index_point(), 33)
    print(f"  mode={fsm.mode.value} sub={fsm.sub_state} gesture={fsm.current_gesture.value}")

    print("\n=== 测试 2: DRAWING + 录制轨迹 ===")
    for i in range(5):
        fsm.process(make_index_point(), 33 * (i + 2))
    print(f"  trajectory pts={fsm.trajectory_point_count}")

    print("\n=== 测试 3: 握拳 → CANDIDATE ===")
    fsm.process(make_fist(), 200)
    print(f"  mode={fsm.mode.value} sub={fsm.sub_state} gesture={fsm.current_gesture.value}")

    print("\n=== 测试 4: 张手 → 取消 → DRAWING ===")
    fsm.process(make_open(), 233)
    print(f"  mode={fsm.mode.value} sub={fsm.sub_state}")

    print("\n=== 测试 5: 外部触发人物推荐 ===")
    fsm.trigger_char_recommend()
    print(f"  mode={fsm.mode.value} sub={fsm.sub_state}")
    fsm.process(make_fist(), 300)
    print(f"  握拳确认 → mode={fsm.mode.value}")

    print("\n✅ 自测完成")
