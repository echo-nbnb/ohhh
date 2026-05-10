#!/usr/bin/env python3
"""
草图识别 → Unity 桥接模块
连接 SketchRecognizer + UnitySender，实现 Act 2 手势绘画→物象候选→用户确认 完整管线

用法:
    bridge = SketchBridge(sender, recognizer)
    bridge.set_color("岳麓绿")

    # 绘画中：累积指尖轨迹
    bridge.add_point(x, y, timestamp_ms)

    # 握拳确认：结束绘画并发送候选到 Unity
    candidates = bridge.commit()
    # → Unity 显示 3 个候选物象
    # → 用户选择后，Unity 发回 object_selected
    # → server.py 处理确认
"""

import time
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("SketchBridge")


@dataclass
class DrawingSession:
    """一次绘画会话"""
    points: List[Tuple[float, float, float]] = field(default_factory=list)
    # [(x, y, timestamp_ms), ...]
    color: str = ""
    started_at: float = 0.0

    def add(self, x: float, y: float, ts_ms: int):
        self.points.append((x, y, float(ts_ms)))

    def clear(self):
        self.points.clear()

    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        return [(x, y) for x, y, _ in self.points]

    @property
    def point_count(self) -> int:
        return len(self.points)


class SketchBridge:
    """
    草图识别 ↔ Unity 桥接

    用法:
        from unity_bridge.sender import UnitySender
        from vision.sketch_recognizer import create_sketch_recognizer

        sender = UnitySender(); sender.connect()
        recognizer = create_sketch_recognizer()
        bridge = SketchBridge(sender, recognizer)
        bridge.set_color("岳麓绿")

        # 绘画循环
        while drawing:
            bridge.add_point(x, y, ts)

        # 握拳 → 结束绘画
        candidates = bridge.commit()
        # → Unity 弹出 3 张候选卡片
    """

    def __init__(self, sender, recognizer):
        """
        Args:
            sender: UnitySender 实例
            recognizer: SketchRecognizer 实例
        """
        self.sender = sender
        self.recognizer = recognizer
        self.session = DrawingSession()
        self._selected_object: Optional[str] = None
        self._on_selected_callbacks: list = []

    # ── 颜色上下文 ───────────────────────────────────────

    def set_color(self, color_name: str):
        """设置第一幕颜色（影响物象加权排序）"""
        self.session.color = color_name

    # ── 轨迹录制 ──────────────────────────────────────────

    def add_point(self, x: float, y: float, timestamp_ms: int = 0):
        """添加指尖轨迹点"""
        if self.session.started_at == 0.0:
            self.session.started_at = time.time()
        self.session.add(x, y, timestamp_ms)

    def clear(self):
        """取消当前绘画（张手）"""
        self.session.clear()

    # ── 识别 + 发送 ──────────────────────────────────────

    def commit(self) -> List[Tuple[str, float, str]]:
        """
        握拳确认：结束绘画 → 识别 → 发送候选到 Unity

        Returns:
            [(name, score, qd_category), ...] Top-3 候选
        """
        if self.session.point_count < 5:
            logger.warning("SketchBridge.commit: 轨迹点数不足")
            return []

        # 识别
        results = self.recognizer.recognize_from_fingertip_history(
            self.session.points,
            color=self.session.color or None
        )

        if not results:
            logger.warning("SketchBridge.commit: 未识别到物象")
            return []

        # 转换为发送格式
        candidates = [(r.entity_name, r.score, r.qd_category) for r in results]

        # 发送到 Unity
        ok = self.sender.send_object_candidates(
            self.session.color or "未知",
            candidates
        )
        if ok:
            logger.info(f"SketchBridge: 已发送 {len(candidates)} 个候选 → Unity")
        else:
            logger.warning("SketchBridge: 发送失败（Unity 未连接）")

        return candidates

    # ── 候选预览（不发送） ────────────────────────────────

    def preview(self) -> List[Tuple[str, float, str]]:
        """仅识别不发送，用于调试"""
        if self.session.point_count < 5:
            return []
        results = self.recognizer.recognize_from_fingertip_history(
            self.session.points,
            color=self.session.color or None
        )
        return [(r.entity_name, r.score, r.qd_category) for r in results]

    # ── 选中回调 ──────────────────────────────────────────

    def on_selected(self, callback):
        """注册物象选中回调 callback(object_name)"""
        self._on_selected_callbacks.append(callback)

    def notify_selected(self, object_name: str):
        """由 server.py 调用，通知物象被选中"""
        self._selected_object = object_name
        for cb in self._on_selected_callbacks:
            try:
                cb(object_name)
            except Exception as e:
                logger.error(f"SketchBridge 回调异常: {e}")
