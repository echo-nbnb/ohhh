#!/usr/bin/env python3
"""
Unity 统一消息发送器
所有 Python→Unity 推送的单一入口，支持重连、双端口（主通道 :8888 + 手部通道 :8889）

用法:
    sender = UnitySender()
    sender.connect()
    sender.send_object_candidates("岳麓绿", [("古树", 0.89, "tree"), ("竹林", 0.72, "tree")])
"""

import json
import socket
import time
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("UnitySender")


class UnitySender:
    """Python → Unity 统一消息发送器"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8888,
                 hand_port: int = 8889):
        self.host = host
        self.port = port
        self.hand_port = hand_port

        self._main_socket: Optional[socket.socket] = None
        self._hand_socket: Optional[socket.socket] = None
        self._connected = False
        self._hand_connected = False

    # ── 连接管理 ──────────────────────────────────────────

    def connect(self) -> bool:
        """连接主通道"""
        if self._connected:
            return True
        try:
            self._main_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._main_socket.settimeout(5.0)
            self._main_socket.connect((self.host, self.port))
            self._main_socket.settimeout(None)
            self._connected = True
            logger.info(f"UnitySender 主通道已连接 {self.host}:{self.port}")
            return True
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            logger.warning(f"UnitySender 主通道连接失败 ({self.host}:{self.port}): {e}")
            self._main_socket = None
            self._connected = False
            return False

    def connect_hand(self) -> bool:
        """连接手部通道"""
        if self._hand_connected:
            return True
        try:
            self._hand_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._hand_socket.settimeout(5.0)
            self._hand_socket.connect((self.host, self.hand_port))
            self._hand_socket.settimeout(None)
            self._hand_connected = True
            logger.info(f"UnitySender 手部通道已连接 {self.host}:{self.hand_port}")
            return True
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            logger.warning(f"UnitySender 手部通道连接失败 ({self.host}:{self.hand_port}): {e}")
            self._hand_socket = None
            self._hand_connected = False
            return False

    def reconnect(self) -> bool:
        """重连所有通道"""
        self.close()
        time.sleep(0.5)
        ok_main = self.connect()
        ok_hand = self.connect_hand()
        return ok_main

    def close(self):
        """关闭所有连接"""
        for sock in [(self._main_socket, "主通道"),
                     (self._hand_socket, "手部通道")]:
            s, _ = sock
            if s:
                try:
                    s.close()
                except Exception:
                    pass
        self._main_socket = None
        self._hand_socket = None
        self._connected = False
        self._hand_connected = False
        logger.info("UnitySender 已断开所有连接")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_hand_connected(self) -> bool:
        return self._hand_connected

    # ── 底层发送 ──────────────────────────────────────────

    def send(self, data: dict) -> bool:
        """发送 JSON 消息到主通道，自动追加 \\n，返回是否成功"""
        if not self._connected or self._main_socket is None:
            logger.warning("UnitySender.send: 主通道未连接")
            return False
        try:
            msg = json.dumps(data, ensure_ascii=False) + "\n"
            self._main_socket.sendall(msg.encode("utf-8"))
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"UnitySender.send 失败: {e}")
            self._connected = False
            self._main_socket = None
            return False

    def send_raw(self, data: dict) -> bool:
        """发送到主通道，失败时自动重连并重试一次"""
        if self.send(data):
            return True
        if self.reconnect():
            return self.send(data)
        return False

    def send_hand(self, data: dict) -> bool:
        """发送到手部通道"""
        if not self._hand_connected or self._hand_socket is None:
            return False
        try:
            msg = json.dumps(data, ensure_ascii=False) + "\n"
            self._hand_socket.sendall(msg.encode("utf-8"))
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._hand_connected = False
            self._hand_socket = None
            return False

    # ── Act 2: 物象候选 ───────────────────────────────────

    def send_object_candidates(self, color: str,
                               candidates: List[Tuple[str, float, str]]) -> bool:
        """
        Args:
            color: 第一幕颜色名，如 "岳麓绿"
            candidates: [(名称, 得分, QuickDraw类别), ...] 按得分降序
        """
        return self.send({
            "type": "object_candidates",
            "color": color,
            "candidates": [
                {"name": name, "score": round(score, 4),
                 "qd_category": qd_cat}
                for name, score, qd_cat in candidates
            ]
        })

    # ── Act 3: 人物推荐 ───────────────────────────────────

    def send_character_candidates(self, candidates: List[Dict]) -> bool:
        """
        Args:
            candidates: [{"name": "张栻", "title": "思想家", "score": 0.85,
                          "reason": "经典搭配推荐"}, ...]
        """
        return self.send({
            "type": "character_candidates",
            "candidates": candidates
        })

    # ── Act 3: 人物轮盘 ───────────────────────────────────

    def send_wheel_state(self, groups: List[str], current_group: str,
                         characters: List[Dict],
                         highlighted_index: int = 0) -> bool:
        """
        Args:
            groups: 分组名列表
            current_group: 当前活跃分组名
            characters: 当前分组人物 [{"name": "周敦颐", "title": "理学开创者"}, ...]
            highlighted_index: 当前高亮卡片索引
        """
        return self.send({
            "type": "wheel_state",
            "groups": groups,
            "current_group": current_group,
            "characters": characters,
            "highlighted_index": highlighted_index
        })

    # ── 手势状态 ──────────────────────────────────────────

    def send_gesture_state(self, mode: str, sub_state: str,
                           gesture: str = "") -> bool:
        """
        Args:
            mode: GLOBAL | DRAWING | CANDIDATE | CHAR_RECOMMEND | CHAR_WHEEL
            sub_state: 各 mode 内的子状态
            gesture: 当前识别到的手势名称
        """
        return self.send({
            "type": "gesture_state",
            "mode": mode,
            "sub_state": sub_state,
            "gesture": gesture
        })

    # ── 手部数据 ──────────────────────────────────────────

    def send_hand_data(self, landmarks: List[Tuple[int, int]],
                       palm_center: Tuple[int, int],
                       wrist: Optional[Tuple[int, int]] = None,
                       fingertips: Optional[List[Tuple[int, int]]] = None,
                       gesture: str = "",
                       timestamp_ms: int = 0) -> bool:
        """发送手部跟踪数据到手部通道"""
        landmarks_flat = []
        for x, y in landmarks:
            landmarks_flat.extend([x, y])

        fingertips_flat = []
        if fingertips:
            for x, y in fingertips:
                fingertips_flat.extend([x, y])

        data = {
            "type": "hand_tracking",
            "palm_center": list(palm_center),
            "landmarks": landmarks_flat,
            "timestamp_ms": timestamp_ms,
            "gesture": gesture
        }
        if wrist:
            data["wrist"] = list(wrist)
        if fingertips_flat:
            data["fingertips"] = fingertips_flat

        return self.send_hand(data)
