"""
RAG 生成模块
负责调用本地小模型或云端API生成叙事内容
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """生成配置"""
    # 本地小模型（Ollama）
    local_model: str = "qwen2.5"  # 待定
    local_api_url: str = "http://localhost:11434/api/generate"

    # 云端API
    cloud_api_url: str = ""  # 待定
    cloud_api_key: str = ""  # 待定

    # 生成参数
    max_tokens: int = 500
    temperature: float = 0.7


class LocalGenerator:
    """
    本地小模型生成器
    负责将检索结果润色为流畅的描述
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        # TODO: 初始化Ollama连接

    def generate_description(self, entity_name: str, context: Dict) -> str:
        """
        为单个实体生成描述

        Args:
            entity_name: 实体名称
            context: 检索到的上下文

        Returns:
            润色后的描述
        """
        # TODO: 调用Ollama小模型
        # 暂时返回基础描述
        return context.get("description", f"{entity_name}。")

    def generate_connections_description(self, connections: List[Dict]) -> str:
        """
        生成连接关系描述

        Args:
            connections: 连接列表

        Returns:
            描述文本
        """
        # TODO: 调用Ollama小模型
        descriptions = []
        for conn in connections:
            if conn.get("meaning"):
                descriptions.append(conn["meaning"])
        return "；".join(descriptions) if descriptions else ""


class CloudGenerator:
    """
    云端大模型生成器
    负责生成完整的个性化叙事
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        # TODO: 初始化云端API连接

    def generate_narrative(self, context: Dict) -> Dict:
        """
        生成完整叙事

        Args:
            context: RAG检索到的上下文

        Returns:
            {
                "title": "你寻到的千年色",
                "paragraphs": ["第一段", "第二段", "第三段", "第四段"],
                "metadata": {...}
            }
        """
        # TODO: 调用云端API生成叙事
        # 暂时返回空结构
        return {
            "title": "你寻到的千年色",
            "paragraphs": [],
            "metadata": {
                "modules": context.get("modules", []),
                "connections": context.get("connections", [])
            }
        }

    def generate_image_prompt(self, context: Dict) -> str:
        """
        生成图像提示词

        Args:
            context: RAG检索到的上下文

        Returns:
            图像生成提示词
        """
        # TODO: 调用云端API生成图像提示词
        modules = context.get("modules", [])
        if not modules:
            return "一幅中国水墨画风格的湖湘文化场景"

        # 简单组合
        entities = [m.get("entity", "") for m in modules]
        return f"一幅中国水墨画风格的湖湘文化场景，包含{'、'.join(entities)}"


class NarrativeGenerator:
    """
    叙事生成器
    整合本地和云端生成器
    """

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.local_gen = LocalGenerator(self.config)
        self.cloud_gen = CloudGenerator(self.config)

    def generate_realtime_description(self, entity_name: str, context: Dict) -> str:
        """
        实时生成单个描述（本地小模型）

        Args:
            entity_name: 实体名称
            context: 检索上下文

        Returns:
            描述文本
        """
        return self.local_gen.generate_description(entity_name, context)

    def generate_for_cloud(self, context: Dict) -> Dict:
        """
        为云端生成准备数据

        Args:
            context: RAG检索上下文

        Returns:
            发送给云端API的完整数据
        """
        # 本地小模型润色模块描述
        for module in context.get("modules", []):
            if "description" not in module or not module["description"]:
                module["description"] = module.get("context", {}).get("description", "")

        # 生成本地描述（用于Unity即时显示）
        connection_descriptions = self.local_gen.generate_connections_description(
            context.get("connections", [])
        )

        return {
            "modules": context.get("modules", []),
            "connections": context.get("connections", []),
            "connection_descriptions": connection_descriptions,
            "cloud_prompt": self.cloud_gen.generate_image_prompt(context)
        }

    def generate_complete_narrative(self, context: Dict) -> Dict:
        """
        生成完整叙事（云端）

        Args:
            context: RAG检索上下文

        Returns:
            完整叙事数据
        """
        return self.cloud_gen.generate_narrative(context)


def create_generator(config: GenerationConfig = None) -> NarrativeGenerator:
    """
    创建叙事生成器

    Args:
        config: 生成配置

    Returns:
        NarrativeGenerator实例
    """
    return NarrativeGenerator(config)
