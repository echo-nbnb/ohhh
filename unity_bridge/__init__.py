"""
Unity 通信桥接模块

包含:
- server.py: Socket服务器，接收Unity事件，返回RAG处理结果
- sender.py: 统一消息发送器，所有 Python→Unity 推送的单一入口
- sketch_bridge.py: 草图识别→Unity 桥接，Act 2 完整管线
- character_bridge.py: 人物推荐→Unity 桥接，Act 3 完���管线
- hand_server.py: 手部跟踪服务器
"""

from .server import UnityServer
from .sender import UnitySender
from .sketch_bridge import SketchBridge
from .character_bridge import CharacterBridge

__all__ = ['UnityServer', 'UnitySender', 'SketchBridge', 'CharacterBridge']
