"""
RAG 模块
寻麓千年色·AI个性化叙事生成

用法:
    from rag import RAGSystem

    rag = RAGSystem()
    rag.setup()

    # 注册模块
    rag.register_module("green_card", "color", "岳麓绿")
    rag.register_module("red_card", "color", "书院红")
    rag.register_module("tree", "object", "古树")

    # 实时检索（放置模块时）
    result = rag.retrieve_realtime("green_card")

    # 高质量生成（开始生成时）
    context = rag.build_generation_context()
    cloud_data = rag.prepare_for_cloud(context)
"""

from typing import Dict, List

from .retriever import (
    RAGRetriever,
    KnowledgeBase,
    ModuleInfo,
    Connection,
    RetrievalResult,
    create_retriever
)
from .generator import (
    NarrativeGenerator,
    GenerationConfig,
    LocalGenerator,
    AliCloudGenerator as CloudGenerator,
    create_generator
)

__all__ = [
    'RAGSystem',
    'RAGRetriever',
    'KnowledgeBase',
    'ModuleInfo',
    'Connection',
    'RetrievalResult',
    'create_retriever',
    'NarrativeGenerator',
    'GenerationConfig',
    'create_generator',
]


class RAGSystem:
    """
    RAG系统主入口
    整合检索和生成模块
    """

    def __init__(self, knowledge_path: str = "rag/knowledge"):
        self.knowledge_path = knowledge_path
        self.retriever = None
        self.generator = None
        self.modules = {}  # module_id -> ModuleInfo
        self.connections = []  # List[Connection]

    def setup(self):
        """初始化RAG系统"""
        self.retriever = create_retriever(self.knowledge_path)
        self.generator = create_generator()
        print("[RAG] 初始化完成")
        print(f"[RAG] 知识库: {self.knowledge_path}")
        print(f"[RAG] 实体数量: {len(self.retriever.kb.entities)}")
        print(f"[RAG] 组合数量: {len(self.retriever.kb.combinations)}")

    def register_module(self, module_id: str, module_type: str, entity: str):
        """
        注册模块

        Args:
            module_id: 模块唯一ID（如 "green_card", "tree_01"）
            module_type: 模块类型 (color/object/character)
            entity: 文化实体名称
        """
        self.modules[module_id] = ModuleInfo(
            module_id=module_id,
            module_type=module_type,
            entity=entity,
            position=None
        )
        self.retriever.kb.register_module(module_id, module_type, entity)
        print(f"[RAG] 注册模块: {module_id} -> {entity} ({module_type})")

    def add_connection(self, from_module: str, to_module: str):
        """
        添加连接

        Args:
            from_module: 起始模块ID
            to_module: 目标模块ID
        """
        if from_module not in self.modules or to_module not in self.modules:
            print(f"[RAG] 错误: 模块未注册")
            return

        conn = Connection(
            from_module=from_module,
            to_module=to_module,
            connection_type=""
        )
        self.connections.append(conn)

    def retrieve_realtime(self, module_id: str) -> Dict:
        """
        实时检索（放置模块时调用）

        Args:
            module_id: 模块ID

        Returns:
            {
                "module_id": "green_card",
                "entity": "岳麓绿",
                "description": "...",
                "connections": [...],
                "unity_data": {...}
            }
        """
        if module_id not in self.modules:
            return {}

        module = self.modules[module_id]

        # 检索模块信息
        result = self.retriever.retrieve_module_info(
            module_id, module.module_type, module.entity
        )

        # 获取该模块的连接
        module_connections = [
            c for c in self.connections
            if c.from_module == module_id or c.to_module == module_id
        ]

        # 填充连接类型
        for conn in module_connections:
            from_info = self.modules[conn.from_module]
            to_info = self.modules[conn.to_module]
            conn.connection_type = self.retriever.get_connection_type(
                from_info.module_type, to_info.module_type
            )

        # 检索连接信息
        conn_results = self.retriever.retrieve_connections(
            list(self.modules.values()),
            module_connections
        )

        # 小模型润色描述
        description = self.generator.generate_realtime_description(
            module.entity, result.context
        )

        # 构建Unity数据
        unity_data = {
            "node_color": result.context.get("color", "#FFFFFF"),
            "connections": conn_results
        }

        return {
            "module_id": module_id,
            "entity": module.entity,
            "description": description,
            "connections": conn_results,
            "unity_data": unity_data
        }

    def build_generation_context(self) -> Dict:
        """
        构建生成上下文

        Returns:
            完整的上下文数据
        """
        module_list = list(self.modules.values())
        return self.retriever.build_context_for_generation(
            module_list, self.connections
        )

    def prepare_for_cloud(self, context: Dict = None) -> Dict:
        """
        为云端生成准备数据

        Args:
            context: 可选，已构建的上下文

        Returns:
            发送给云端API的数据
        """
        if context is None:
            context = self.build_generation_context()

        return self.generator.generate_for_cloud(context)

    def generate_narrative(self, context: Dict = None) -> Dict:
        """
        生成完整叙事（云端）

        Args:
            context: 可选，已构建的上下文

        Returns:
            完整叙事
        """
        if context is None:
            context = self.build_generation_context()

        return self.generator.generate_complete_narrative(context)

    def clear(self):
        """清除所有模块和连接"""
        self.modules.clear()
        self.connections.clear()
        print("[RAG] 已清除")
