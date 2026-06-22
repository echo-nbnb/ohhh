#!/usr/bin/env python3
"""
Auto-advancing Mock Backend — 自动推进时间线，驱动 React 前端全流程

通过 ws_server.py 桥接到前端。发送完整的五幕所需消息序列。
无需外部输入，纯时间线驱动。
"""

import socket
import threading
import json
import time
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("AutoBackend")

MAIN_PORT = 8888
HAND_PORT = 8889


class AutoBackend:
    def __init__(self):
        self.main_client = None
        self.running = False

    # ── 启动 ────────────────────────────────────
    def start(self):
        self.running = True
        t1 = threading.Thread(target=self._run_main_server, daemon=False)
        t2 = threading.Thread(target=self._run_hand_server, daemon=False)
        t3 = threading.Thread(target=self._run_timeline, daemon=False)
        t1.start()
        t2.start()

        # 等两个 server 就绪
        time.sleep(0.5)
        t3.start()
        logger.info("Auto backend started on :8888 (main) + :8889 (hand)")
        return self

    # ── TCP 主通道 ──────────────────────────────
    def _run_main_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", MAIN_PORT))
        server.listen(1)
        logger.info(f"Main channel listening on :{MAIN_PORT}")

        while self.running:
            try:
                server.settimeout(1.0)
                client, addr = server.accept()
                logger.info(f"Main client connected: {addr}")
                self.main_client = client

                # 处理前端发来的消息
                buf = ""
                client.settimeout(0.5)
                while self.running:
                    try:
                        data = client.recv(4096)
                        if not data:
                            break
                        buf += data.decode("utf-8")
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            if line.strip():
                                try:
                                    msg = json.loads(line.strip())
                                    self._handle_frontend_message(msg)
                                except json.JSONDecodeError:
                                    pass
                    except socket.timeout:
                        continue
                    except Exception:
                        break

                client.close()
                self.main_client = None
                logger.info("Main client disconnected")
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Main server error: {e}")

        server.close()

    def _handle_frontend_message(self, msg):
        """处理前端发来的消息（如截图上传）"""
        msg_type = msg.get("type", "")
        if msg_type == "screenshot_upload":
            logger.info(f"← screenshot_upload received ({len(msg.get('image_base64', ''))} chars)")
            # 模拟生成 postcard 结果
            self._send({
                "type": "postcard_result",
                "image_url": "",
                "qr_base64": "",
                "unique_id": "auto-" + str(int(time.time()))
            })

    # ── TCP 手部通道 ────────────────────────────
    def _run_hand_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", HAND_PORT))
        server.listen(1)
        logger.info(f"Hand channel listening on :{HAND_PORT}")

        while self.running:
            try:
                server.settimeout(1.0)
                client, addr = server.accept()
                logger.info(f"Hand client connected: {addr}")
                client.settimeout(0.5)
                while self.running:
                    try:
                        data = client.recv(4096)
                        if not data:
                            break
                    except socket.timeout:
                        continue
                    except Exception:
                        break
                client.close()
                logger.info("Hand client disconnected")
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Hand server error: {e}")

        server.close()

    # ── 发送 ────────────────────────────────────
    def _send(self, msg):
        if self.main_client:
            try:
                data = json.dumps(msg, ensure_ascii=False) + "\n"
                self.main_client.send(data.encode("utf-8"))
                logger.info(f"→ {msg.get('type', '?')}: {str(msg.get('color', msg.get('name', msg.get('gesture', ''))))}")
            except Exception as e:
                logger.error(f"Send error: {e}")

    def _send_drawing_points(self, duration=4.0, interval=0.04):
        """
        发送模拟绘制点序列 — 画一个简单形状
        在 1280×720 坐标系中画弧线
        """
        cx, cy = 640, 400
        rx, ry = 180, 120
        steps = int(duration / interval)
        for i in range(steps):
            angle = (i / steps) * math.pi * 2.2
            x = int(cx + rx * math.cos(angle))
            y = int(cy + ry * math.sin(angle * 1.3))
            self._send({"type": "drawing_point", "x": x, "y": y})
            time.sleep(interval)

    # ── 自动时间线 ──────────────────────────────
    def _run_timeline(self):
        """
        完整五幕自动推进时间线。
        前端各 Act 有 stage-aware 消息处理，消息可提前发送，前端会缓存/忽略。
        """
        # 等 ws_server 连上来
        time.sleep(1.5)
        logger.info("=== 时间线开始 ===")

        # ═══════════════════════════════════════════
        # Act0 → Act1: 握拳触发入境
        # ═══════════════════════════════════════════
        time.sleep(1.0)
        self._send({"type": "gesture_state", "gesture": "fist", "mode": "GLOBAL"})
        logger.info(">>> 握拳 → 入境")

        # ═══════════════════════════════════════════
        # Act1 进行中… 提前发送两个颜色（前端会缓存在 INTRO/TRANSITION 阶段）
        # ═══════════════════════════════════════════
        time.sleep(3)
        self._send({
            "type": "object_color_detected",
            "color": "岳麓绿",
            "confidence": 0.92,
            "source": "object",
            "message": "我捕捉到了一抹近山之色。它属于岳麓绿。",
            "interpretation": "这是根脉的颜色。它来自树，也来自仍在生长的你。"
        })
        logger.info(">>> 颜色1: 岳麓绿")

        time.sleep(5)
        self._send({
            "type": "object_color_detected",
            "color": "书院红",
            "confidence": 0.88,
            "source": "object",
            "message": "我捕捉到了一抹近火之色。它属于书院红。",
            "interpretation": "这是发问的颜色。红墙记住了许多人年轻时的理想。"
        })
        logger.info(">>> 颜色2: 书院红")

        # ═══════════════════════════════════════════
        # 等 Act1 溶解 + Act2 展示（~30s 从 fist 算起）
        # ═══════════════════════════════════════════
        # Act1: 18s dissolve + 1.2s hold
        # Act2: step 2→3 (2.5s) → step 3→4 (2.5s) → complete (4s) ≈ 9s
        # 总需 ~30s，已经过了 ~9s
        time.sleep(22)

        # ═══════════════════════════════════════════
        # Act3: 发送模拟画点 → 物象识别 ×2
        # ═══════════════════════════════════════════
        logger.info(">>> Act3: 开始模拟绘画")

        # 第一轮绘画
        self._send_drawing_points(duration=3.0, interval=0.05)
        time.sleep(1.0)
        self._send({
            "type": "object_recognized",
            "color": "岳麓绿",
            "object": {"name": "古树", "score": 0.88, "qd_category": "tree"},
            "narration": "你画下了一棵古树。它扎根于此，千年不倒，见证了一代代学人来来去去。"
        })
        logger.info(">>> 物象1: 古树")

        # 第二轮绘画
        time.sleep(2)
        self._send_drawing_points(duration=2.5, interval=0.05)
        time.sleep(1.0)
        self._send({
            "type": "object_recognized",
            "color": "书院红",
            "object": {"name": "湘江", "score": 0.90, "qd_category": "river"},
            "narration": "你画下了湘江。它流过书院门前，带走时光，却带不走沉淀千年的文脉。"
        })
        logger.info(">>> 物象2: 湘江 → 4s 后自动进入 Act4")

        # ═══════════════════════════════════════════
        # Act4 准备: 在 Act4 mount 之前发送人物+叙事
        # （Act4 mount 比 object_recognized #2 晚 4s）
        # ═══════════════════════════════════════════
        self._send({
            "type": "character_revealed",
            "name": "张栻",
            "title": "岳麓书院早期讲学者之一",
            "era": "南宋",
            "message": "刚才与你说话的，是张栻。但他留下的不只是名字，更是一种敢于发问的底色。",
            "monologue": [
                "你选择了两种不肯沉默的颜色。",
                "又画下通往答案的形状。",
                "我知道那不是热闹的勇气。",
                "而是站在众人面前，仍愿意发问的执着。",
                "后来者，",
                "你是想寻找答案，",
                "还是想成为那个继续提问的人？"
            ],
            "spiritLine": "他留下的不只是名字，而是一种敢于发问的颜色。"
        })
        logger.info(">>> 人物: 张栻 + 独白")

        time.sleep(2)
        self._send({
            "type": "generation_result",
            "title": "岳麓绿映书院红，张栻问道",
            "narrative": "岳麓松风，千年一色。你的颜色与物象已经化入这片山水。",
            "paragraphs": [
                "刚才与你说话的，是我。",
                "我曾在岳麓书院的讲堂里讲学，",
                "也曾望着湘江水思考学问的方向。",
                "岳麓绿是根脉的颜色，书院红是发问的颜色。",
                "两种颜色相遇，便是一个人在天地间，",
                "既扎根，又向远方发问的姿态。",
                "愿你在岳麓绿的光里，",
                "保有书院红的心；",
                "敢问，敢辨，",
                "也敢向更远处走去。"
            ]
        })
        logger.info(">>> 叙事生成完成")

        # ═══════════════════════════════════════════
        # Act4 → Act5: 等 Act4 36s 时间线走完 → 握拳盖章
        # Act4 internal: FINAL_REVEAL @ 31.9s + completeDelay 5s = ~37s
        # ═══════════════════════════════════════════
        time.sleep(42)
        self._send({"type": "gesture_state", "gesture": "fist", "mode": "GLOBAL"})
        logger.info(">>> 握拳盖章 → Act5")

        # ═══════════════════════════════════════════
        # Act5: 明信片自动播放 → 截图 → backend 响应 postcard_result
        # （postcard_result 由 _handle_frontend_message 在收到 screenshot_upload 时发送）
        # ═══════════════════════════════════════════
        logger.info("=== 时间线完成（明信片截图后自动响应）===")

    def stop(self):
        self.running = False
        if self.main_client:
            try:
                self.main_client.close()
            except Exception:
                pass


if __name__ == "__main__":
    backend = AutoBackend().start()
    print("Auto backend running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        backend.stop()
