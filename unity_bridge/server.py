#!/usr/bin/env python3
"""
Unity Socket 服务器
用于与Unity端通信，接收事件并返回RAG处理结果

使用方法:
    python server.py
"""

import json
import socket
import threading
import time
from typing import Dict, Optional

import sys
sys.path.insert(0, '.')

from rag import RAGSystem


class UnityServer:
    """Unity通信服务器"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.is_running = False
        self.rag_system = None

    def start(self):
        """启动服务器"""
        # 初始化RAG系统
        print("[Server] 初始化RAG系统...")
        try:
            self.rag_system = RAGSystem()
            self.rag_system.setup()
        except Exception as e:
            print(f"[Server] RAG初始化失败: {e}")
            print("[Server] 继续启动，RAG功能将不可用")

        # 创建Socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        self.is_running = True
        print(f"[Server] 服务器启动成功 {self.host}:{self.port}")

        # 接受连接
        self.accept_loop()

    def accept_loop(self):
        """接受客户端连接"""
        while self.is_running:
            try:
                print("[Server] 等待Unity连接...")
                self.client_socket, addr = self.server_socket.accept()
                print(f"[Server] Unity已连接: {addr}")

                # 发送欢迎消息
                self.send({"type": "connected", "message": "Hello from Python"})

                # 处理消息
                self.handle_loop()

            except Exception as e:
                print(f"[Server] 连接错误: {e}")
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None

    def handle_loop(self):
        """处理消息循环"""
        buffer = ""

        while self.is_running:
            try:
                data = self.client_socket.recv(4096)
                if not data:
                    break

                buffer += data.decode('utf-8')

                # 处理完整的消息（以\n分隔）
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self.process_message(line)

            except Exception as e:
                print(f"[Server] 处理消息错误: {e}")
                break

        # 清理
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
        print("[Server] Unity已断开")

    def process_message(self, message: str):
        """
        处理收到的消息

        消息格式:
        {
            "event": "module_placed" | "module_connected" | "generation_start",
            "module_id": "...",
            "from_module": "...",
            "to_module": "...",
            ...
        }
        """
        try:
            data = json.loads(message)
            event = data.get("event", "")

            print(f"[Server] 收到事件: {event}")
            print(f"[Server] 数据: {data}")

            if event == "module_placed":
                self.handle_module_placed(data)
            elif event == "module_connected":
                self.handle_module_connected(data)
            elif event == "generation_start":
                self.handle_generation_start(data)
            else:
                print(f"[Server] 未知事件: {event}")

        except json.JSONDecodeError as e:
            print(f"[Server] JSON解析失败: {e}")
        except Exception as e:
            print(f"[Server] 处理失败: {e}")

    def handle_module_placed(self, data: Dict):
        """处理模块放置"""
        module_id = data.get("module_id", "")
        module_type = data.get("module_type", "")
        position = data.get("position", {})

        if not module_id or not module_type:
            return

        # 如果RAG系统未初始化，发送默认响应
        if self.rag_system is None:
            self.send({
                "type": "module_placed",
                "module_id": module_id,
                "entity": module_type,
                "description": f"模块 {module_id} 已放置",
                "unity_data": {
                    "node_color": "#FFFFFF"
                }
            })
            return

        # 注册模块（如果尚未注册）
        if module_id not in self.rag_system.modules:
            # 简化处理：使用模块类型作为实体名
            entity_map = {
                "color": f"Color_{module_id}",
                "object": f"Object_{module_id}",
                "character": f"Character_{module_id}"
            }
            entity = entity_map.get(module_type, module_type)
            self.rag_system.register_module(module_id, module_type, entity)

        # 检索实时描述
        result = self.rag_system.retrieve_realtime(module_id)

        # 发送到Unity
        self.send({
            "type": "module_placed",
            "module_id": module_id,
            "entity": result.get("entity", ""),
            "description": result.get("description", ""),
            "connections": result.get("connections", []),
            "unity_data": result.get("unity_data", {})
        })

    def handle_module_connected(self, data: Dict):
        """处理模块连接"""
        from_module = data.get("from_module", "")
        to_module = data.get("to_module", "")

        if not from_module or not to_module:
            return

        if self.rag_system:
            self.rag_system.add_connection(from_module, to_module)

        # 检索连接描述
        if self.rag_system and self.rag_system.retriever:
            module_from = self.rag_system.modules.get(from_module)
            module_to = self.rag_system.modules.get(to_module)

            if module_from and module_to:
                from_entity = module_from.entity
                to_entity = module_to.entity

                # 获取连接类型
                conn_type = self.rag_system.retriever.get_connection_type(
                    module_from.module_type, module_to.module_type
                )
                style = self.rag_system.retriever.get_connection_style(conn_type)

                self.send({
                    "type": "connection_created",
                    "from_module": from_module,
                    "to_module": to_module,
                    "connection_type": conn_type,
                    "line_color": style.get("line_color", "#FFFFFF"),
                    "is_dashed": style.get("line_style") == "dashed"
                })

    def handle_generation_start(self, data: Dict):
        """处理开始生成"""
        modules = data.get("modules", [])
        connections = data.get("connections", [])

        print(f"[Server] 开始生成叙事...")
        print(f"[Server] 模块数量: {len(modules)}")
        print(f"[Server] 连接数量: {len(connections)}")

        if self.rag_system:
            # 构建上下文
            context = self.rag_system.build_generation_context()

            # 准备云端数据
            cloud_data = self.rag_system.prepare_for_cloud(context)

            # 由于没有实际的云端API，生成模拟响应
            # 实际使用时，这里应该调用云端API

            # 模拟叙事生成
            narrative = self.generate_mock_narrative(context)

            self.send({
                "type": "generation_result",
                "title": "你寻到的千年色",
                "paragraphs": narrative["paragraphs"],
                "narrative": narrative["full_text"],
                "image_prompt": cloud_data.get("cloud_prompt", "一幅中国水墨画风格的湖大场景"),
                "metadata": {
                    "modules": [m.get("entity", "") for m in context.get("modules", [])],
                    "connections": [c.get("meaning", "") for c in context.get("connections", [])]
                }
            })
        else:
            # 无RAG系统，发送模拟数据
            self.send({
                "type": "generation_result",
                "title": "你寻到的千年色",
                "paragraphs": [
                    "你选择了某种颜色作为底色。",
                    "你放下了一些物象。",
                    "你请来了历史人物。",
                    "这就是你寻到的千年色。"
                ],
                "narrative": "这是你的个性化叙事。",
                "image_prompt": "一幅中国水墨画风格的湖大场景"
            })

    def generate_mock_narrative(self, context: Dict) -> Dict:
        """生成模拟叙事（实际应由云端API生成）"""
        modules = context.get("modules", [])
        connections = context.get("connections", [])

        entities = [m.get("entity", "") for m in modules]
        entity_str = "、".join(entities) if entities else "未知元素"

        paragraphs = [
            f"你选择了{entity_str}。",
            f"它们在一起，构成了一幅独特的画面。",
            "历史人物走入了这个世界，",
            "这就是你寻到的'千年色'。"
        ]

        return {
            "paragraphs": paragraphs,
            "full_text": "\n\n".join(paragraphs)
        }

    def send(self, data: Dict):
        """发送数据到Unity"""
        if self.client_socket is None:
            print("[Server] 错误: 没有已连接的客户端")
            return

        try:
            message = json.dumps(data, ensure_ascii=False) + "\n"
            self.client_socket.sendall(message.encode('utf-8'))
            print(f"[Server] 发送: {message.strip()}")
        except Exception as e:
            print(f"[Server] 发送失败: {e}")

    def stop(self):
        """停止服务器"""
        self.is_running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        print("[Server] 服务器已停止")


def main():
    print("=" * 50)
    print("寻麓千年色 - Unity Socket 服务器")
    print("=" * 50)

    server = UnityServer(host="127.0.0.1", port=8888)

    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[Server] 收到中断信号")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
