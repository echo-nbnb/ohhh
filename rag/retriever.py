"""
RAG 检索模块
负责从知识库检索文化内容，为叙事生成提供上下文
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModuleInfo:
    """模块信息"""
    module_id: str
    module_type: str  # color / object / character
    entity: str       # 文化实体名称
    position: Optional[Tuple[int, int]] = None


@dataclass
class Connection:
    """模块连接"""
    from_module: str
    to_module: str
    connection_type: str  # spiritual_resonance / color_grant / personality_dye / scene_entry


@dataclass
class RetrievalResult:
    """检索结果"""
    module_id: str
    entity: str
    description: str           # 小模型润色后的描述
    context: Dict              # 原始检索上下文
    connections: List[Dict]     # 该模块的连接信息


class KnowledgeBase:
    """知识库"""

    def __init__(self, base_path: str = "rag/knowledge"):
        self.base_path = base_path
        self.entities = {}
        self.combinations = {}
        self.templates = {}
        self.module_map = {}  # 模块ID到文化实体的映射

    def load(self):
        """加载知识库"""
        # 加载实体
        for entity_type in ["colors", "objects", "characters"]:
            filepath = os.path.join(self.base_path, "entities", f"{entity_type}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.entities.update(data)

        # 加载组合
        combos_dir = os.path.join(self.base_path, "combinations")
        if os.path.exists(combos_dir):
            for filename in os.listdir(combos_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(combos_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        key = filename.replace('.json', '')
                        self.combinations[key] = data

        # 加载模板
        templates_dir = os.path.join(self.base_path, "templates")
        if os.path.exists(templates_dir):
            for filename in os.listdir(templates_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(templates_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        key = filename.replace('.json', '')
                        self.templates[key] = data

    def get_entity(self, entity_name: str) -> Optional[Dict]:
        """获取实体信息"""
        return self.entities.get(entity_name)

    def get_combination(self, key: str) -> Optional[Dict]:
        """获取组合解读"""
        return self.combinations.get(key)

    def get_template(self, template_name: str) -> Optional[Dict]:
        """获取模板"""
        return self.templates.get(template_name)

    def register_module(self, module_id: str, module_type: str, entity: str):
        """注册模块映射"""
        self.module_map[module_id] = {
            "type": module_type,
            "entity": entity
        }

    def get_module_entity(self, module_id: str) -> Optional[Dict]:
        """获取模块对应的文化实体"""
        return self.module_map.get(module_id)


class RAGRetriever:
    """RAG检索器"""

    # 连接类型定义
    CONNECTION_TYPES = {
        ("color", "color"): "spiritual_resonance",      # 颜色-颜色：精神共鸣
        ("color", "object"): "color_grant",              # 颜色-物象：精神赋予
        ("object", "color"): "color_grant",              # 物象-颜色：精神赋予
        ("character", "color"): "personality_dye",        # 人物-颜色：人格染色
        ("color", "character"): "personality_dye",        # 颜色-人物：人格染色
        ("character", "object"): "scene_entry",           # 人物-物象：人物进入场景
        ("object", "character"): "scene_entry",           # 物象-人物：人物进入场景
    }

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def get_connection_type(self, type1: str, type2: str) -> str:
        """获取连接类型"""
        key = (type1, type2)
        return self.CONNECTION_TYPES.get(key, "unknown")

    def get_connection_style(self, connection_type: str, from_entity: str = None) -> Dict:
        """获取连接的视觉样式"""
        styles = {
            "spiritual_resonance": {
                "line_color": "#FFD700",  # 金色
                "line_style": "dashed"
            },
            "color_grant": {
                "line_color": "#2E7D32",  # 由物象决定
                "line_style": "solid"
            },
            "personality_dye": {
                "line_color": "#800020",  # 由人物决定
                "line_style": "solid"
            },
            "scene_entry": {
                "line_color": "#FFFFFF",  # 白色
                "line_style": "solid"
            }
        }
        return styles.get(connection_type, {"line_color": "#FFFFFF", "line_style": "solid"})

    def retrieve_module_info(self, module_id: str, module_type: str, entity: str) -> RetrievalResult:
        """
        检索单个模块的信息

        Args:
            module_id: 模块ID
            module_type: 模块类型
            entity: 文化实体名称

        Returns:
            检索结果
        """
        # 获取实体信息
        entity_info = self.kb.get_entity(entity)

        if entity_info:
            # 基础描述
            description = entity_info.get("description", f"{entity}。")
            context = entity_info
        else:
            description = f"{entity}。"
            context = {}

        return RetrievalResult(
            module_id=module_id,
            entity=entity,
            description=description,
            context=context,
            connections=[]
        )

    def retrieve_connections(self, modules: List[ModuleInfo], connections: List[Connection]) -> List[Dict]:
        """
        检索连接信息

        Args:
            modules: 所有模块
            connections: 所有连接

        Returns:
            连接检索结果
        """
        results = []

        for conn in connections:
            # 获取两端模块的类型和实体
            from_info = self.kb.get_module_entity(conn.from_module)
            to_info = self.kb.get_module_entity(conn.to_module)

            if not from_info or not to_info:
                continue

            # 获取连接类型
            conn_type = self.get_connection_type(from_info["type"], to_info["type"])

            # 获取组合解读
            combo_key = f"{from_info['entity']}_{to_info['entity']}"
            combo_info = self.kb.get_combination(combo_key)

            # 获取样式
            style = self.get_connection_style(conn_type, from_info["entity"])

            results.append({
                "from": conn.from_module,
                "to": conn.to_module,
                "connection_type": conn_type,
                "meaning": combo_info.get("meaning", "") if combo_info else "",
                "unity_style": style
            })

        return results

    def build_context_for_generation(self, modules: List[ModuleInfo], connections: List[Connection]) -> Dict:
        """
        为云端生成构建完整上下文

        Args:
            modules: 所有模块
            connections: 所有连接

        Returns:
            完整的上下文数据
        """
        # 检索每个模块的信息
        module_infos = []
        for m in modules:
            info = self.kb.get_entity(m.entity)
            module_infos.append({
                "entity": m.entity,
                "type": m.module_type,
                "description": info.get("description", "") if info else "",
                "context": info if info else {}
            })

        # 检索连接信息
        conn_infos = self.retrieve_connections(modules, connections)

        return {
            "modules": module_infos,
            "connections": conn_infos
        }


def create_retriever(knowledge_path: str = "rag/knowledge") -> RAGRetriever:
    """
    创建RAG检索器

    Args:
        knowledge_path: 知识库路径

    Returns:
        RAGRetriever实例
    """
    kb = KnowledgeBase(knowledge_path)
    kb.load()
    return RAGRetriever(kb)
