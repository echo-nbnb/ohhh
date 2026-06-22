"""
WebSocket服务器 - 桥接HTML前端与Python TCP后端

浏览器WebSocket ← → ws_server ← → TCP backend
"""

import asyncio
import json
import socket
import threading
import argparse
from datetime import datetime

try:
    import websockets
except ImportError:
    print("[WS] 需要安装 websockets: pip install websockets")
    import sys
    sys.exit(1)

# 全局变量
HOST = "127.0.0.1"
BACKEND_MAIN_PORT = 8888
BACKEND_HAND_PORT = 8889
WS_PORT = 8080


class BackendConnection:
    """管理与Python TCP后端的连接"""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self._recv_buf = ""  # 接收缓冲区，防止多消息合并时丢包

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.connect((self.host, self.port))
            self.connected = True
            print(f"[WS→TCP] 已连接后端 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[WS→TCP] 连接失败: {e}")
            self.connected = False
            return False

    def send(self, data: dict):
        """发送消息到后端"""
        if not self.connected:
            return False
        try:
            msg = json.dumps(data, ensure_ascii=False) + "\n"
            self.sock.sendall(msg.encode('utf-8'))
            return True
        except Exception as e:
            print(f"[WS→TCP] 发送失败: {e}")
            self.connected = False
            return False

    def receive(self, timeout=1.0):
        """接收后端消息（非阻塞），带缓冲防止多消息合并丢包"""
        if not self.connected:
            return None
        try:
            # 先检查缓冲区是否有完整消息
            if '\n' in self._recv_buf:
                line, self._recv_buf = self._recv_buf.split('\n', 1)
                if line.strip():
                    return json.loads(line.strip())

            self.sock.settimeout(timeout)
            data = self.sock.recv(4096)
            if not data:
                self.connected = False
                return None

            self._recv_buf += data.decode('utf-8')

            # 从缓冲区取第一条完整消息
            if '\n' in self._recv_buf:
                line, self._recv_buf = self._recv_buf.split('\n', 1)
                if line.strip():
                    return json.loads(line.strip())
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[WS→TCP] 接收失败: {e}")
            self.connected = False
        return None

    def close(self):
        if self.sock:
            self.sock.close()
        self.connected = False


class WSClient:
    """单个WebSocket客户端连接"""

    def __init__(self, websocket, client_id):
        self.ws = websocket
        self.client_id = client_id
        self.backend_main = None
        self.backend_hand = None

    async def send(self, data: dict):
        """发送到浏览器"""
        try:
            msg_type = data.get("type", "?")
            print(f"[WS→Browser] #{self.client_id} {msg_type}")
            await self.ws.send(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            print(f"[WS→Browser] 发送失败: {e}")

    async def receive(self):
        """从浏览器接收，返回 None 表示超时，抛出异常表示连接关闭"""
        try:
            msg = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
            return json.loads(msg)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            # WebSocket 关闭，向上抛出让调用方处理
            raise ConnectionError(f"WebSocket closed: {e}")


class WebSocketServer:
    """WebSocket服务器"""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.clients: dict[int, WSClient] = {}
        self.client_id_counter = 0
        self.backend_main = BackendConnection(HOST, BACKEND_MAIN_PORT)
        self.backend_hand = BackendConnection(HOST, BACKEND_HAND_PORT)
        self._running = False

    async def handle_client(self, websocket, path=None):
        """处理单个客户端连接"""
        client_id = self.client_id_counter
        self.client_id_counter += 1
        client = WSClient(websocket, client_id)
        client.backend_main = self.backend_main
        client.backend_hand = self.backend_hand
        self.clients[client_id] = client

        print(f"[WS] 客户端 #{client_id} 已连接")
        await client.send({"type": "connected", "message": "ws_server_ready"})

        # 确保后端连接
        if not self.backend_main.connected:
            self.backend_main.connect()

        try:
            while self._running:
                # 接收浏览器消息
                try:
                    msg = await client.receive()
                except ConnectionError:
                    print(f"[WS] 客户端 #{client_id} 断开连接")
                    break
                if msg:
                    print(f"[Browser→WS] #{client_id}: {msg.get('type', '?')}")
                    # 转发到后端
                    self._route_message(msg, client)

                # 接收后端消息并转发
                await self._poll_backend(client)

                await asyncio.sleep(0.01)

        except Exception as e:
            print(f"[WS] 客户端 #{client_id} 异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del self.clients[client_id]
            print(f"[WS] 客户端 #{client_id} 已断开")

    def _route_message(self, msg: dict, client: WSClient):
        """路由消息到正确的后端端口"""
        msg_type = msg.get("type", "")

        # hand_tracking 走手部通道
        if msg_type == "hand_tracking":
            client.backend_hand.send(msg)
        # 其他消息走主通道
        else:
            client.backend_main.send(msg)

    async def _poll_backend(self, client: WSClient):
        """轮询后端消息，断线自动重连"""
        try:
            # 主通道 — 断线自动重连
            if not self.backend_main.connected:
                print("[WS→TCP] 主通道断开，尝试重连...")
                self.backend_main.connect()
            data = self.backend_main.receive(timeout=0.001)
            if data:
                print(f"[TCP→WS] #{client.client_id}: {data.get('type', '?')}")
                await client.send(data)

            # 手部通道
            data = self.backend_hand.receive(timeout=0.001)
            if data:
                print(f"[TCP→WS] #{client.client_id}: {data.get('type', '?')}")
                await client.send(data)
        except Exception as e:
            print(f"[WS] _poll_backend error: {e}")

    async def start(self):
        """启动服务器"""
        self._running = True
        self.backend_main.connect()

        async with websockets.serve(self.handle_client, self.host, self.port, max_size=10*1024*1024):
            print(f"[WS] WebSocket服务器启动 ws://{self.host}:{self.port}")
            print(f"[WS] 桥接到后端 TCP {HOST}:{BACKEND_MAIN_PORT} 和 {HOST}:{BACKEND_HAND_PORT}")
            await asyncio.Future()  # 永久运行

    def stop(self):
        self._running = False
        self.backend_main.close()
        self.backend_hand.close()


async def main():
    parser = argparse.ArgumentParser(description="WebSocket桥接服务器")
    parser.add_argument("--host", default=HOST, help="WebSocket监听地址")
    parser.add_argument("--port", type=int, default=WS_PORT, help="WebSocket监听端口")
    args = parser.parse_args()

    server = WebSocketServer(args.host, args.port)

    try:
        await server.start()
    except KeyboardInterrupt:
        print("\n[WS] 服务器关闭中...")
        server.stop()


if __name__ == "__main__":
    asyncio.run(main())
