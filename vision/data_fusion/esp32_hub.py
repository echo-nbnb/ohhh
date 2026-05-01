#!/usr/bin/env python3
"""
ESP32 Hub 接收模块
通过UDP接收ESP32模块上报的数据

协议格式：
- 每条消息以换行符结尾
- 消息为JSON格式

模块上报数据格式：
{
    "module_id": "esp32_001",
    "module_type": "color",  # color / object / character / confirm
    "touch_state": false,    # true = 被触摸, false = 未触摸
    "online": true
}

查询响应格式（查询时返回）：
{
    "modules": [
        {"module_id": "esp32_001", "module_type": "color", "touch_state": false, "online": true},
        ...
    ]
}
"""

import socket
import json
import threading
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ESP32Module:
    """ESP32模块数据"""
    module_id: str
    module_type: str  # color / object / character / confirm
    touch_state: bool
    online: bool
    last_update: float


class ESP32Hub:
    """
    ESP32 Hub 接收器
    接收并管理所有ESP32模块的数据
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8888):
        """
        Args:
            host: 监听地址
            port: 监听端口
        """
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.running = False

        # 模块数据存储
        self.modules: Dict[str, ESP32Module] = {}
        self.lock = threading.Lock()

        # 统计
        self.packet_count = 0
        self.last_packet_time = 0

    def start(self) -> bool:
        """启动Hub"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(0.1)  # 非阻塞超时
            self.running = True
            print(f"[ESP32Hub] 监听 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[ESP32Hub] 启动失败: {e}")
            return False

    def stop(self):
        """停止Hub"""
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None
        print("[ESP32Hub] 已停止")

    def _parse_message(self, data: bytes) -> Optional[Dict[str, Any]]:
        """解析接收到的消息"""
        try:
            # 尝试解码并解析JSON
            text = data.decode('utf-8').strip()
            return json.loads(text)
        except Exception as e:
            print(f"[ESP32Hub] 解析消息失败: {e}")
            return None

    def _update_module(self, msg: Dict[str, Any]):
        """更新模块数据"""
        module_id = msg.get('module_id')
        if not module_id:
            return

        module = ESP32Module(
            module_id=module_id,
            module_type=msg.get('module_type', 'unknown'),
            touch_state=msg.get('touch_state', False),
            online=msg.get('online', True),
            last_update=time.time()
        )

        with self.lock:
            self.modules[module_id] = module
            self.packet_count += 1
            self.last_packet_time = time.time()

    def receive_once(self) -> Optional[Dict[str, Any]]:
        """
        单次接收（不会阻塞）

        Returns:
            模块数据字典，或None（无数据）
        """
        try:
            data, addr = self.socket.recvfrom(1024)
            msg = self._parse_message(data)
            if msg:
                self._update_module(msg)
                return msg
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[ESP32Hub] 接收错误: {e}")
        return None

    def receive_all(self) -> List[Dict[str, Any]]:
        """
        接收所有可用数据

        Returns:
            模块数据列表
        """
        results = []
        while True:
            data = self.receive_once()
            if data is None:
                break
            results.append(data)
        return results

    def get_all_modules(self) -> List[Dict[str, Any]]:
        """
        获取所有模块的当前状态

        Returns:
            模块列表
        """
        with self.lock:
            return [
                {
                    "module_id": m.module_id,
                    "module_type": m.module_type,
                    "touch_state": m.touch_state,
                    "online": m.online,
                    "timestamp": m.last_update
                }
                for m in self.modules.values()
            ]

    def get_module(self, module_id: str) -> Optional[ESP32Module]:
        """获取指定模块的数据"""
        with self.lock:
            return self.modules.get(module_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                "module_count": len(self.modules),
                "packet_count": self.packet_count,
                "last_packet_time": self.last_packet_time
            }


def run_hub_test():
    """运行Hub测试"""
    hub = ESP32Hub(port=8888)
    if not hub.start():
        return

    print("[ESP32Hub] 开始测试，按Ctrl+C退出\n")

    try:
        while True:
            # 接收数据
            data = hub.receive_once()
            if data:
                print(f"[收到] {data}")

            # 定期打印状态
            stats = hub.get_stats()
            if stats['packet_count'] > 0 and int(time.time()) % 5 == 0:
                print(f"[状态] 模块数: {stats['module_count']}, 数据包: {stats['packet_count']}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[退出]")
    finally:
        hub.stop()


if __name__ == "__main__":
    run_hub_test()
