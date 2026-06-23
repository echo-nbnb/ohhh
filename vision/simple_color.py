"""
颜色识别 — 独立运行，3s稳定确认后发送到后端

用法:
    python vision/simple_color.py [--port 8888]

按 Q 退出
"""

import sys
import json
import socket
import time
import cv2
import numpy as np

HOST = "127.0.0.1"
PORT = 8888


def hsv_to_color(h, s, v):
    """7色：红橙黄绿青蓝紫 + 墨"""
    if s < 25 or v < 25: return "墨"
    if h <= 10 or h >= 170: return "红"
    if h <= 25: return "橙"
    if h <= 38: return "黄"
    if h <= 80: return "绿"
    if h <= 105: return "青"
    if h <= 135: return "蓝"
    return "紫"


def connect_backend():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.connect((HOST, PORT))
        print(f"[OK] 已连接到后端 {HOST}:{PORT}")
        return sock
    except Exception as e:
        print(f"[!] 无法连接后端: {e}")
        return None


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    global PORT
    PORT = port

    print("=" * 50)
    print("  颜色识别 — 独立发送模式")
    print("=" * 50)
    print(f"  后端: {HOST}:{PORT}")
    print("  Q=退出")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    sock = connect_backend()
    stable_name = None
    stable_since = 0
    last_sent = ""  # 避免重复发送

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 中心小方块取色
        s = 40
        roi = frame[h // 2 - s:h // 2 + s, w // 2 - s:w // 2 + s]
        avg = roi.mean(axis=0).mean(axis=0)
        hsv = cv2.cvtColor(np.uint8([[avg]]), cv2.COLOR_BGR2HSV)[0, 0]
        name = hsv_to_color(int(hsv[0]), int(hsv[1]), int(hsv[2]))

        # ── 3s 稳定确认 ──
        now = time.time()
        if stable_name != name:
            stable_name = name
            stable_since = now
        elif now - stable_since >= 2.0 and name != last_sent:
            # 发送给后端
            msg = json.dumps({"type": "object_color_detected", "color": name, "confidence": 0.7,
                              "source": "simple_color", "message": f"我捕捉到了一抹{name}。"},
                             ensure_ascii=False) + "\n"
            if sock:
                try:
                    sock.sendall(msg.encode("utf-8"))
                    print(f"[发送] {name}  HSV=({int(hsv[0])},{int(hsv[1])},{int(hsv[2])})")
                    print(f"  PALETTE: 红:0-8/172-180 橙:9-22 黄:23-32 绿:33-85 青:86-100 蓝:101-130 紫:131-171")
                except Exception:
                    print("[!] 发送失败，重连...")
                    sock = connect_backend()
                    if sock:
                        sock.sendall(msg.encode("utf-8"))
            else:
                print(f"[模拟] {name}")
            last_sent = name
            stable_name = None  # 重置，等下次

        # 显示
        cv2.rectangle(frame, (w // 2 - s, h // 2 - s), (w // 2 + s, h // 2 + s), (0, 255, 255), 2)
        cv2.rectangle(frame, (w - 120, 10), (w - 10, 80),
                      [int(c) for c in cv2.cvtColor(np.uint8([[[int(hsv[0]), 255, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]], -1)
        cv2.putText(frame, name, (w - 120, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        elapsed = now - stable_since
        status = f"STABLE {elapsed:.1f}s/2s" if elapsed < 2 else "SENT ✓"
        color = (0, 200, 255) if elapsed < 2 else (0, 255, 0)
        cv2.putText(frame, status, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"H:{int(hsv[0])} S:{int(hsv[1])} V:{int(hsv[2])}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Color → Backend", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'): break

    if sock: sock.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
