#!/usr/bin/env python3
"""WebSocket服务器诊断工具"""
import socket
import sys

def check_port(port, name):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(('127.0.0.1', port))
    if result == 0:
        print(f"[OK] Port {port} ({name}): IN USE")
        s.close()
        return True
    else:
        print(f"[--] Port {port} ({name}): FREE")
        return False

print("=== Port Status ===")
check_port(8888, "mock_backend主通道")
check_port(8889, "mock_backend手部通道")
check_port(8080, "ws_server")

print("\n=== WebSocket Test ===")
try:
    import websockets
    import asyncio

    async def test_ws():
        try:
            async with websockets.connect('ws://127.0.0.1:8080', timeout=3) as ws:
                print("[OK] Connected to ws://127.0.0.1:8080")
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                print(f"[OK] Received: {msg}")
                return True
        except Exception as e:
            print(f"[X] WebSocket error: {type(e).__name__}: {e}")
            return False

    result = asyncio.run(test_ws())
except ImportError:
    print("[X] websockets not installed: pip install websockets")

print("\n=== If ports are FREE, run these commands ===")
print("Terminal 1: python web/mock_backend.py")
print("Terminal 2: python web/ws_server.py")
print("Then refresh the browser")
