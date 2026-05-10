#!/usr/bin/env python3
"""
人物推荐 → Unity 桥接模块
连接 CharacterRecommender + UnitySender，实现 Act 3 智能推荐管线

用法:
    bridge = CharacterBridge(sender, recommender)
    bridge.set_context(color="岳麓绿", objects=["古树"])
    candidates = bridge.recommend()
    # → Unity 显示 3 张人物推荐卡片
    # → 用户选择后，Unity 发回 character_selected
    # → server.py 处理确认
"""

import logging
from typing import List, Optional

logger = logging.getLogger("CharacterBridge")


class CharacterBridge:
    """
    人物推荐 ↔ Unity 桥接

    用法:
        from unity_bridge.sender import UnitySender
        from rag.character_recommend import create_character_recommender

        sender = UnitySender(); sender.connect()
        recommender = create_character_recommender()
        bridge = CharacterBridge(sender, recommender)
        bridge.set_context("岳麓绿", ["古树", "讲堂"])

        # 触发推荐
        candidates = bridge.recommend()
        # → Unity 弹出 3 张人物推荐卡片
    """

    def __init__(self, sender, recommender):
        """
        Args:
            sender: UnitySender 实例
            recommender: CharacterRecommender 实例
        """
        self.sender = sender
        self.recommender = recommender
        self.color: str = ""
        self.objects: List[str] = []
        self.selected_characters: List[str] = []
        self._selected_character: Optional[str] = None
        self._on_selected_callbacks: list = []

    # ── 上下文设置 ───────────────────────────────────────

    def set_context(self, color: str, objects: List[str],
                    selected_characters: Optional[List[str]] = None):
        """设置推荐上下文（颜色 + 已选物象 + 已选人物）"""
        self.color = color
        self.objects = list(objects)
        self.selected_characters = list(selected_characters or [])

    def add_object(self, object_name: str):
        """添加已选物象"""
        if object_name not in self.objects:
            self.objects.append(object_name)

    def add_selected_character(self, char_name: str):
        """添加已选人物（多人物场景）"""
        if char_name not in self.selected_characters:
            self.selected_characters.append(char_name)

    # ── 推荐 + 发送 ──────────────────────────────────────

    def recommend(self, use_llm: bool = True) -> List[dict]:
        """
        执行推荐 → 发送到 Unity

        Returns:
            [{"name": "张栻", "title": "思想家", "score": 0.85, "reason": "..."}, ...]
        """
        if not self.color:
            logger.warning("CharacterBridge: 未设置颜色上下文")
            return []

        results = self.recommender.recommend(
            color=self.color,
            objects=self.objects,
            selected_characters=self.selected_characters,
            use_llm=use_llm,
            top_k=3,
        )

        if not results:
            logger.warning("CharacterBridge: 无推荐结果")
            return []

        candidates = [
            {"name": r.name, "title": r.title, "score": round(r.score, 4),
             "reason": r.reason}
            for r in results
        ]

        ok = self.sender.send_character_candidates(candidates)
        if ok:
            logger.info(f"CharacterBridge: 已发送 {len(candidates)} 个人物推荐 → Unity")
        else:
            logger.warning("CharacterBridge: 发送失败（Unity 未连接）")

        return candidates

    # ── 选中回调 ──────────────────────────────────────────

    def on_selected(self, callback):
        """注册人物选中回调 callback(character_name)"""
        self._on_selected_callbacks.append(callback)

    def notify_selected(self, character_name: str):
        """由 server.py 调用，通知人物被选中"""
        self._selected_character = character_name
        self.selected_characters.append(character_name)
        for cb in self._on_selected_callbacks:
            try:
                cb(character_name)
            except Exception as e:
                logger.error(f"CharacterBridge 回调异常: {e}")
