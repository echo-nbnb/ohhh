#!/usr/bin/env python3
"""
Unity Socket 服务器（简化测试版）
先测试通信，不依赖RAG

使用方法:
    python server_simple.py
"""

import json
import socket
import threading
import sys

print("=" * 50)
print("Unity Socket 服务器 (简化测试版)")
print("=" * 50)

class UnityServer:
    def __init__(self, host="127.0.0.1", port=8888):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.is_running = False

    def start(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.is_running = True
            print(f"[Server] 启动成功 {self.host}:{self.port}")
            print("[Server] 等待Unity连接...")
            self.accept_loop()
        except Exception as e:
            print(f"[Server] 启动失败: {e}")
            import traceback
            traceback.print_exc()

    def accept_loop(self):
        while self.is_running:
            try:
                self.client_socket, addr = self.server_socket.accept()
                print(f"[Server] Unity已连接: {addr}")
                self.send({"type": "connected", "message": "Hello from Python"})
                self.handle_loop()
            except Exception as e:
                print(f"[Server] accept错误: {e}")
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None

    def handle_loop(self):
        buffer = ""
        while self.is_running:
            try:
                data = self.client_socket.recv(4096)
                if not data:
                    break
                buffer += data.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self.process_message(line)
            except Exception as e:
                print(f"[Server] recv错误: {e}")
                break

        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
        print("[Server] Unity已断开")

    def process_message(self, message):
        print(f"[Server] 收到: {message}")
        try:
            data = json.loads(message)
            event = data.get("event", "")
            print(f"[Server] 事件: {event}")

            if event == "module_placed":
                self.send({
                    "type": "module_placed",
                    "module_id": data.get("module_id"),
                    "entity": "测试模块",
                    "description": "这是测试描述",
                    "unity_data": {"node_color": "#2E7D32"}
                })
            elif event == "generation_start":
                self.send({
                    "type": "generation_result",
                    "title": "测试叙事",
                    "paragraphs": ["第一段", "第二段", "第三段", "第四段"],
                    "narrative": "完整测试叙事"
                })
        except Exception as e:
            print(f"[Server] 处理失败: {e}")

    def send(self, data):
        if not self.client_socket:
            print("[Server] 无连接")
            return
        try:
            msg = json.dumps(data, ensure_ascii=False) + "\n"
            self.client_socket.sendall(msg.encode('utf-8'))
            print(f"[Server] 发送: {msg.strip()}")
        except Exception as e:
            print(f"[Server] 发送失败: {e}")

    def stop(self):
        self.is_running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        print("[Server] 已停止")

if __name__ == "__main__":
    server = UnityServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[Server] 收到中断")
        server.stop()
