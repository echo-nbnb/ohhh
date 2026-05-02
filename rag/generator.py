"""
RAG 生成模块
负责调用本地小模型或云端API生成叙事内容
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass

# 阿里云百炼 API
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("[警告] dashscope未安装，将使用占位符实现")


@dataclass
class GenerationConfig:
    """生成配置"""
    # 阿里云百炼API配置
    api_key: str = ""  # 通过环境变量 DASHSCOPE_API_KEY 设置

    # 模型选择
    realtime_model: str = "qwen-turbo"      # 实时侧用小模型（快）
    narrative_model: str = "qwen-plus"      # 叙事卡用中等模型（质量）

    # 生成参数
    max_tokens: int = 500
    temperature: float = 0.7


class AliCloudGenerator:
    """
    阿里云百炼生成器
    通过dashscope API调用阿里云大模型
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        if DASHSCOPE_AVAILABLE and config.api_key:
            dashscope.api_key = config.api_key

    def _call_model(self, model: str, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        调用阿里云百炼模型

        Args:
            model: 模型名称
            prompt: 输入提示词
            max_tokens: 最大token数
            temperature: 温度参数

        Returns:
            模型生成的文本
        """
        if not DASHSCOPE_AVAILABLE:
            return f"[模拟输出] {prompt[:50]}..."

        try:
            response = Generation.call(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                result_format='message'
            )
            if response.status_code == 200:
                return response.output.text
            else:
                print(f"[错误] API调用失败: {response.message}")
                return ""
        except Exception as e:
            print(f"[错误] API调用异常: {e}")
            return ""

    def generate_realtime_description(self, entity_name: str, entity_context: Dict) -> str:
        """
        实时生成单个模块的描述（用于Unity即时显示）

        Args:
            entity_name: 实体名称
            entity_context: 实体的上下文信息

        Returns:
            一句润色后的描述
        """
        description = entity_context.get("description", "")
        symbolism = entity_context.get("symbolism", "")

        prompt = f"""你是一个湖湘文化解说员。请为以下文化元素生成一句简短（20字以内）的描述。

元素：{entity_name}
含义：{symbolism}
基础描述：{description}

要求：
- 语言优美，有文化韵味
- 直接点明精神内涵
- 格式：一句完整的话

输出："""

        return self._call_model(
            model=self.config.realtime_model,
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )

    def generate_connection_description(self, from_entity: str, to_entity: str,
                                       connection_type: str, meaning: str) -> str:
        """
        生成连接关系的描述

        Args:
            from_entity: 起始实体
            to_entity: 目标实体
            connection_type: 连接类型
            meaning: 已有含义描述

        Returns:
            润色后的连接描述
        """
        prompt = f"""你是一个湖湘文化解说员。请为以下连接关系生成一句描述。

连接：{from_entity} ↔ {to_entity}
类型：{connection_type}
含义：{meaning}

要求：
- 一句话，20字以内
- 体现两个元素之间的精神联结
- 语言优美

输出："""

        return self._call_model(
            model=self.config.realtime_model,
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )

    def generate_narrative(self, context: Dict) -> Dict:
        """
        生成完整叙事（用于叙事卡）

        Args:
            context: RAG检索到的完整上下文
            {
                "modules": [...],
                "connections": [...],
                "title": "你寻到的千年色"
            }

        Returns:
            {
                "title": "...",
                "paragraphs": ["第一段", "第二段", ...],
                "summary": "..."
            }
        """
        modules = context.get("modules", [])
        connections = context.get("connections", [])

        # 构建模块列表描述
        module_list = "\n".join([
            f"- {m.get('entity', '')}（{m.get('type', '')}）：{m.get('description', '')}"
            for m in modules
        ])

        # 构建连接描述
        conn_list = "\n".join([
            f"- {c.get('from', '')} ↔ {c.get('to', '')}：{c.get('meaning', '')}"
            for c in connections
        ]) if connections else "无"

        prompt = f"""你是一个湖湘文化叙事作家。请根据以下元素组合，生成一段个性化的湖大千年色叙事。

【用户选择的元素】
{module_list}

【元素之间的连接】
{conn_list}

【叙事主题】你寻到的千年色

要求：
- 4-5段文字，娓娓道来
- 从"我"的角度叙述，有代入感
- 融入湖湘文化和湖大历史
- 每段2-3句话
- 最后一段呼应主题，升华情感
- 语言优美，有文学性

输出格式（JSON）：
{{
    "title": "你寻到的千年色",
    "paragraphs": ["第一段内容", "第二段内容", ...],
    "summary": "一句话总结"
}}"""

        result = self._call_model(
            model=self.config.narrative_model,
            prompt=prompt,
            max_tokens=800,
            temperature=0.8
        )

        if not result:
            return {
                "title": "你寻到的千年色",
                "paragraphs": ["[生成失败，请稍后重试]"],
                "summary": ""
            }

        # 尝试解析JSON
        try:
            # 提取JSON部分（可能包含在markdown代码块中）
            if "```json" in result:
                start = result.find("```json") + 7
                end = result.find("```", start)
                result = result[start:end]
            elif "```" in result:
                start = result.find("```") + 3
                end = result.find("```", start)
                result = result[start:end]

            return json.loads(result)
        except json.JSONDecodeError:
            # 解析失败，返回原始文本
            return {
                "title": "你寻到的千年色",
                "paragraphs": [result],
                "summary": ""
            }

    def generate_image_prompt(self, context: Dict) -> str:
        """
        生成图像提示词（用于AI作画）

        Args:
            context: RAG检索上下文

        Returns:
            英文图像提示词
        """
        modules = context.get("modules", [])
        entities = [m.get("entity", "") for m in modules]

        prompt = f"""请为以下湖大千年色元素生成一幅Stable Diffusion图像提示词。

元素：{"、".join(entities)}

要求：
- 风格：中国水墨画 + 穿越感
- 包含元素：岳麓书院、湘江、人物（抽象人影/光柱）
- 色调：用户选择的颜色晕染成水墨背景
- 氛围：人文与山水交融，历史感与当代感并存
- 输出：英文提示词，200词以内
- 不要包含任何文字或字母

输出："""

        return self._call_model(
            model=self.config.realtime_model,
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )


class LocalGenerator:
    """
    本地小模型生成器（备用）
    当云端API不可用时使用本地模型
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    def generate_description(self, entity_name: str, context: Dict) -> str:
        """生成描述（本地备用）"""
        return context.get("description", f"{entity_name}。")

    def generate_connections_description(self, connections: List[Dict]) -> str:
        """生成连接描述（本地备用）"""
        descriptions = []
        for conn in connections:
            if conn.get("meaning"):
                descriptions.append(conn["meaning"])
        return "；".join(descriptions) if descriptions else ""


class NarrativeGenerator:
    """
    叙事生成器
    整合阿里云百炼和本地生成器
    """

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.ali_gen = AliCloudGenerator(self.config)
        self.local_gen = LocalGenerator(self.config)

    def generate_realtime_description(self, entity_name: str, entity_context: Dict) -> str:
        """
        实时生成单个描述（优先使用阿里云百炼）

        Args:
            entity_name: 实体名称
            entity_context: 检索上下文

        Returns:
            描述文本
        """
        if DASHSCOPE_AVAILABLE and self.config.api_key:
            return self.ali_gen.generate_realtime_description(entity_name, entity_context)
        else:
            return self.local_gen.generate_description(entity_name, entity_context)

    def generate_for_cloud(self, context: Dict) -> Dict:
        """
        为云端生成准备数据

        Args:
            context: RAG检索上下文

        Returns:
            发送给云端API的完整数据
        """
        modules = context.get("modules", [])
        connections = context.get("connections", [])

        # 生成本地连接描述（用于Unity即时显示）
        connection_descriptions = []
        for conn in connections:
            from_entity = conn.get("from", "")
            to_entity = conn.get("to", "")
            conn_type = conn.get("connection_type", "")
            meaning = conn.get("meaning", "")

            desc = self.ali_gen.generate_connection_description(
                from_entity, to_entity, conn_type, meaning
            ) if DASHSCOPE_AVAILABLE and self.config.api_key else meaning

            connection_descriptions.append(desc)

        # 生成图像提示词
        image_prompt = self.ali_gen.generate_image_prompt(context) if DASHSCOPE_AVAILABLE and self.config.api_key else ""

        return {
            "modules": modules,
            "connections": connections,
            "connection_descriptions": connection_descriptions,
            "cloud_prompt": image_prompt
        }

    def generate_complete_narrative(self, context: Dict) -> Dict:
        """
        生成完整叙事（阿里云百炼）

        Args:
            context: RAG检索上下文

        Returns:
            完整叙事数据
        """
        if DASHSCOPE_AVAILABLE and self.config.api_key:
            return self.ali_gen.generate_narrative(context)
        else:
            return {
                "title": "你寻到的千年色",
                "paragraphs": ["[请配置阿里云百炼API_KEY]"],
                "summary": ""
            }


def create_generator(config: GenerationConfig = None) -> NarrativeGenerator:
    """
    创建叙事生成器

    Args:
        config: 生成配置

    Returns:
        NarrativeGenerator实例
    """
    return NarrativeGenerator(config)


def create_config(api_key: str = None) -> GenerationConfig:
    """
    创建生成配置

    Args:
        api_key: 阿里云百炼API Key（优先从环境变量读取）

    Returns:
        GenerationConfig实例
    """
    import os
    return GenerationConfig(
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY", "")
    )
