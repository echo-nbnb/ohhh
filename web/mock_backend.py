#!/usr/bin/env python3
"""
Mock Backend - 模拟后端 TCP 服务器
用于测试 HTML 前端与 ws_server 的连接
"""

import socket
import threading
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("MockBackend")

MAIN_PORT = 8888
HAND_PORT = 8889


class MockBackend:
    def __init__(self):
        self.main_socket = None
        self.hand_socket = None
        self.main_client = None
        self.hand_client = None
        self.running = False
        self.fsm_mode = "COLOR_EXTRACTION"
        self._color_sub = "AWAITING"  # AWAITING / ANALYZING / CONFIRMING / CONFIRMED
        self._color_source = "object"  # "object" | "clothing" | "ink"
        self._color_name = "岳麓绿"   # detected color name
        self._ready_count = 0
        self._ready_lock = threading.Lock()
        self.selected_objects = []     # 已确认的物象列表
        self._draw_count = 0           # 绘画次数，用于切换物象

    def _mark_ready(self):
        with self._ready_lock:
            self._ready_count += 1
            if self._ready_count >= 2:
                pass  # both servers ready

    def start(self):
        self.running = True
        t1 = threading.Thread(target=self._run_main)
        t2 = threading.Thread(target=self._run_hand)
        t1.start()
        t2.start()
        # 等待服务器真正开始监听（每个服务启动后调用_mark_ready）
        while self._ready_count < 2:
            time.sleep(0.1)
        logger.info(f"Mock backend started on :8888 (main) and :8889 (hand)")
        return self

    def _run_main(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', MAIN_PORT))
        server.listen(1)
        logger.info(f"Main channel listening on :{MAIN_PORT}")
        self._mark_ready()

        server.settimeout(1.0)
        while self.running:
            try:
                client, addr = server.accept()
                logger.info(f"Main channel client connected: {addr}")
                self.main_client = client
                self._handle_main(client)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Main channel error: {e}")

    def _run_hand(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', HAND_PORT))
        server.listen(1)
        logger.info(f"Hand channel listening on :{HAND_PORT}")
        self._mark_ready()

        server.settimeout(1.0)
        while self.running:
            try:
                client, addr = server.accept()
                logger.info(f"Hand channel client connected: {addr}")
                self.hand_client = client
                self._handle_hand(client)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Hand channel error: {e}")

    def _handle_main(self, client):
        """处理主通道消息"""
        client.settimeout(0.5)
        buffer = ""
        while self.running:
            try:
                data = client.recv(4096)
                if not data:
                    break
                buffer += data.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self._process_main_message(line.strip(), client)
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Main recv error: {e}")
                break

        logger.info("Main channel client disconnected")
        try:
            client.close()
        except:
            pass
        self.main_client = None

    def _handle_hand(self, client):
        """处理手部通道消息"""
        client.settimeout(0.5)
        buffer = ""
        while self.running:
            try:
                data = client.recv(4096)
                if not data:
                    break
                buffer += data.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self._process_hand_message(line.strip(), client)
            except socket.timeout:
                # 定期发送模拟手部数据
                if self.hand_client:
                    self._send_hand_data(self.hand_client)
                continue
            except Exception as e:
                logger.error(f"Hand recv error: {e}")
                break

        logger.info("Hand channel client disconnected")
        try:
            client.close()
        except:
            pass
        self.hand_client = None

    def _process_main_message(self, msg_str, client):
        """处理主通道消息"""
        try:
            msg = json.loads(msg_str)
            msg_type = msg.get("type", "unknown")
            logger.info(f"Main received: {msg_type}")

            if msg_type == "gesture_simulate":
                gesture = msg.get("gesture", "")
                self._handle_gesture(gesture)
                # 发送 FSM 状态更新
                sub = self._get_sub_state()
                response = {
                    "type": "gesture_state",
                    "mode": self.fsm_mode,
                    "sub_state": sub,
                    "gesture": gesture
                }
                self._send_main(response)

            elif msg_type == "generation_start":
                # 模拟生成结果
                response = {
                    "type": "generation_result",
                    "paragraphs": [
                        "一幅山水画卷缓缓展开，云雾缭绕间，隐约可见古人之身影。",
                        "色彩斑斓而不失雅致，笔触间流露出千年文化的厚重。",
                        "此画只应天上有，人间难得几回闻。"
                    ]
                }
                self._send_main(response)
                logger.info("Sent generation_result")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {msg_str[:100]}")

    def _get_sub_state(self):
        """获取当前子状态"""
        if self.fsm_mode == "COLOR_EXTRACTION":
            return self._color_sub
        elif self.fsm_mode == "DRAWING":
            return "TRACKING"
        elif self.fsm_mode == "CANDIDATE":
            return "BROWSING"
        return "IDLE"

    # ── 叙事模板 ────────────────────────────────────
    COLOR_TEXTS = {
        "岳麓绿": {
            "detected": "我捕捉到了一抹近山之色。它属于岳麓绿。",
            "interpretation": "这是根脉的颜色。它来自树，也来自仍在生长的你。",
            "confirmed": "我看到了……你的颜色是岳麓绿。它是生长、是根脉、是传承。"
        },
        "书院红": {
            "detected": "我捕捉到了一抹近火之色。它属于书院红。",
            "interpretation": "这是发问的颜色。红墙不只是建筑，它记住了许多人年轻时的理想。",
            "confirmed": "我看到了……你的颜色是书院红。它是发问、是责任、是理想。"
        },
        "西迁黄": {
            "detected": "我捕捉到了一抹近土之色。它属于西迁黄。",
            "interpretation": "这是路的颜色。它不明亮，却坚定。属于那些在风雨中仍选择前行的人。",
            "confirmed": "我看到了……你的颜色是西迁黄。它是坚韧、是跋涉、是坚守。"
        },
        "湘江蓝": {
            "detected": "我捕捉到了一抹近水之色。它属于湘江蓝。",
            "interpretation": "这是流动的颜色。它不停止，也不回头。把过去带向远方。",
            "confirmed": "我看到了……你的颜色是湘江蓝。它是流动、是包容、是追寻。"
        },
        "校徽金": {
            "detected": "我捕捉到了一抹近光之色。它属于校徽金。",
            "interpretation": "这是理想的颜色。它不是炫耀，而是人在某一刻相信自己可以抵达更远处。",
            "confirmed": "我看到了……你的颜色是校徽金。它是理想、是荣耀、是远方。"
        },
        "墨色": {
            "detected": "我捕捉到了一抹近墨之色。它属于墨色。",
            "interpretation": "这是求索的颜色。它很深，因为答案从来不浮在表面。",
            "confirmed": "我看到了……你的颜色是墨色。它是求索、是深沉、是真理。"
        },
    }

    CHARACTER_PERFORMANCE = {
        "王夫之": {
            "search": "你的颜色指向生长与传承，你画下的树指向根脉与守望。一位与「传承」有关的人，正向你走来。",
            "performance": [
                "你选择了绿。又画下了树。",
                "我知道那种绿。那不是寻常草木的绿，是一个人把一生种在书山里，仍相信春天会来的绿。",
                "后来者，你是想寻找归宿，还是想成为那个继续扎根的人？"
            ],
            "reveal": {"name": "王夫之", "title": "思想家", "era": "明末清初",
                       "summary": "湖湘学派集大成者，隐居船山著书立说"}
        },
        # 兜底模板
        "_default": {
            "search": "正在千年文脉中，寻找与你相遇的人……",
            "performance": [
                "你留下的颜色和意象，我已经看到了。",
                "在这座书院千年的回声里，有一个声音与你的选择相近。",
                "他不问你来处，只问你愿不愿意，继续走下去。"
            ],
            "reveal": {"name": "王夫之", "title": "思想家", "era": "明末清初",
                       "summary": "湖湘学派集大成者，隐居船山著书立说"}
        }
    }

    def _get_color_text(self, key):
        return self.COLOR_TEXTS.get("岳麓绿", self.COLOR_TEXTS["岳麓绿"])[key]

    def _get_char_data(self, name):
        return self.CHARACTER_PERFORMANCE.get(name, self.CHARACTER_PERFORMANCE["_default"])

    # ── 物象叙事模板 ────────────────────────────
    OBJECT_NARRATIONS = [
        {"name": "古树", "score": 0.88, "qd_category": "tree",
         "narration": "你画下了一棵古树。它扎根于此，千年不倒，见证了一代代学人来来去去。"},
        {"name": "讲堂", "score": 0.85, "qd_category": "house",
         "narration": "你画下了一座讲堂。这里曾回荡着朱熹与张栻的辩难之声，千年不绝。"},
        {"name": "石阶", "score": 0.82, "qd_category": "stairs",
         "narration": "你画下了石阶。每一级都通向高处，就像每一代学人都在前人肩上远望。"},
        {"name": "湘江", "score": 0.90, "qd_category": "river",
         "narration": "你画下了湘江。它流过书院门前，带走时光，却带不走沉淀千年的文脉。"},
        {"name": "桥梁", "score": 0.86, "qd_category": "bridge",
         "narration": "你画下了一座桥。它连接此岸与彼岸，也连接着出发与归来。"},
    ]

    def _get_object_data(self):
        """轮转返回不同物象"""
        obj = self.OBJECT_NARRATIONS[self._draw_count % len(self.OBJECT_NARRATIONS)]
        self._draw_count += 1
        return obj

    def _handle_gesture(self, gesture):
        """
        六段式叙事 FSM（v3 修订）:

        颜色兜底三路径:
          fist     → 物件检测成功 (object_color_detected → CONFIRMING → fist → GLOBAL)
          fist_obj_fail → 物件失败→衣物检测成功 (clothing_fallback → CLOTHING_CONFIRMING → fist → GLOBAL)
          fist_all_fail → 物件失败→衣物失败→墨色 (clothing_fallback → ink_default → GLOBAL)

        多物象:
          CANDIDATE + fist → object_confirmed → GLOBAL (继续画或完成)
          GLOBAL + index_pointing → DRAWING (画下一个)
          GLOBAL + fist (有物象时) → 触发人物演绎+生成 (完成筑景)
        """
        if gesture == "fist":
            self._on_fist()
        elif gesture == "fist_obj_fail":
            self._on_fist(fallback="clothing")
        elif gesture == "fist_all_fail":
            self._on_fist(fallback="ink")
        elif gesture == "open_hand":
            self._on_open_hand()
        elif gesture == "index_pointing":
            self._on_index_pointing()

    # ── 握拳 ────────────────────────────────────

    def _on_fist(self, fallback=None):
        # ---------- 颜色阶段 ----------
        if self.fsm_mode == "COLOR_EXTRACTION":
            if self._color_sub == "AWAITING":
                self._color_sub = "ANALYZING"
                self._color_source = fallback or "object"
                self._send_main({
                    "type": "color_extraction_start",
                    "message": "请将随身之物靠近光中。让它替你说话。"
                })

            elif self._color_sub == "CONFIRMING":
                self._color_sub = "CONFIRMED"
                self.fsm_mode = "GLOBAL"
                color = self._color_name
                texts = self.COLOR_TEXTS[color]
                self._send_main({
                    "type": "color_confirmed",
                    "color": color,
                    "source": self._color_source,
                    "message": texts["confirmed"]
                })

            elif self._color_sub == "CLOTHING_CONFIRMING":
                self._color_sub = "CONFIRMED"
                self.fsm_mode = "GLOBAL"
                color = self._color_name
                texts = self.COLOR_TEXTS[color]
                self._send_main({
                    "type": "color_confirmed",
                    "color": color,
                    "source": "clothing",
                    "message": texts["confirmed"]
                })

            # ANALYZING → 自动推进
            if self._color_sub == "ANALYZING":
                time.sleep(0.08)
                self._run_color_detection()

        # ---------- 全局态 ----------
        elif self.fsm_mode == "GLOBAL":
            if self.selected_objects:
                # 有物象 → 完成筑景 → 触发人物+生成
                self._trigger_character_and_generation()
            elif fallback:
                # 无物象 + 有兜底标记 → 重启颜色周期（测试用）
                self._restart_color_cycle(fallback)

        # ---------- 绘画提交 ----------
        elif self.fsm_mode == "DRAWING":
            self.fsm_mode = "CANDIDATE"
            obj = self._get_object_data()
            self._send_main({
                "type": "object_recognized",
                "color": self._color_name,
                "object": {"name": obj["name"], "score": obj["score"],
                           "qd_category": obj["qd_category"]},
                "narration": obj["narration"]
            })

        # ---------- 确认物象 → 全局态 ----------
        elif self.fsm_mode == "CANDIDATE":
            # 从上一个 object_recognized 中获取物象名（简化：用最近的）
            obj_name = self.OBJECT_NARRATIONS[(self._draw_count - 1) % len(self.OBJECT_NARRATIONS)]["name"]
            self.selected_objects.append(obj_name)
            self.fsm_mode = "GLOBAL"

            count = len(self.selected_objects)
            objs_str = "、".join(self.selected_objects)
            self._send_main({
                "type": "object_confirmed",
                "object": obj_name,
                "objects_so_far": list(self.selected_objects),
                "message": f"一个意象已经落下。" if count == 1 else
                           f"又一个意象落下。{objs_str}，它们在一起了。",
                "continue_prompt": "继续绘画：伸出食指。完成筑景：握拳。",
                "can_continue": True
            })

    def _restart_color_cycle(self, fallback):
        """从 GLOBAL 重启颜色周期（测试用）"""
        self.fsm_mode = "COLOR_EXTRACTION"
        self._color_sub = "ANALYZING"
        self._color_source = fallback
        self._send_main({
            "type": "color_extraction_start",
            "message": "请将随身之物靠近光中。让它替你说话。"
        })
        self._run_color_detection()

    # ── 颜色检测自动推进 ──────────────────────

    def _run_color_detection(self):
        """模拟物件/衣物颜色检测，根据 _color_source 走不同路径"""
        if self._color_source == "object":
            # 路径A: 物件匹配成功
            self._color_sub = "CONFIRMING"
            self._color_name = "岳麓绿"
            texts = self.COLOR_TEXTS[self._color_name]
            self._send_main({
                "type": "object_color_detected",
                "color": self._color_name,
                "confidence": 0.92,
                "source": "object",
                "message": texts["detected"],
                "interpretation": texts["interpretation"]
            })

        elif self._color_source == "clothing":
            # 路径B: 物件失败 → 衣物兜底成功
            self._color_sub = "CLOTHING_FALLBACK"
            self._send_main({
                "type": "object_color_failed",
                "message": "这件物品的颜色太安静了，它没有进入这座书院的色谱。"
            })
            time.sleep(0.08)
            self._color_sub = "CLOTHING_CONFIRMING"
            self._color_name = "书院红"
            texts = self.COLOR_TEXTS[self._color_name]
            self._send_main({
                "type": "clothing_color_detected",
                "color": self._color_name,
                "confidence": 0.85,
                "source": "clothing",
                "message": "那我看看今天的你。你今天穿着书院红。也许这就是你此刻的底色。",
                "interpretation": texts["interpretation"]
            })

        elif self._color_source == "ink":
            # 路径C: 物件失败 → 衣物也失败 → 墨色兜底
            self._color_sub = "CLOTHING_FALLBACK"
            self._send_main({
                "type": "object_color_failed",
                "message": "这件物品的颜色太安静了，它没有进入这座书院的色谱。"
            })
            time.sleep(0.08)
            self._send_main({
                "type": "clothing_color_failed",
                "message": "今天的你，颜色也不愿被命名。"
            })
            time.sleep(0.08)
            self._color_sub = "CONFIRMED"
            self.fsm_mode = "GLOBAL"
            self._color_name = "墨色"
            texts = self.COLOR_TEXTS["墨色"]
            self._send_main({
                "type": "color_confirmed",
                "color": "墨色",
                "source": "ink",
                "message": "有些颜色，不急着被命名。那就让墨色替你开始吧。"
            })

    # ── 人物+生成触发 ──────────────────────────

    def _trigger_character_and_generation(self):
        """完成筑景 → 搜索人物 → 演绎 → 揭示 → 自动生成"""
        self.fsm_mode = "GLOBAL"

        objs_str = "、".join(self.selected_objects)
        # 1. 物象总结
        self._send_main({
            "type": "objects_summary",
            "objects": list(self.selected_objects),
            "message": f"颜色已经展开。{objs_str}也已经落下。现在，我要在千年文脉里，寻找与你相遇的人。"
        })

        # 2. 搜索
        char_data = self._get_char_data("王夫之")
        time.sleep(0.1)
        self._send_main({
            "type": "character_search_start",
            "message": char_data["search"],
            "context": {"color": self._color_name, "objects": list(self.selected_objects)}
        })
        time.sleep(0.15)
        self._send_main({
            "type": "character_found",
            "message": "找到了。",
            "character_name_hidden": True
        })

        # 3. 第一人称演绎
        time.sleep(0.1)
        self._send_main({
            "type": "character_performance",
            "character": "????",
            "paragraphs": char_data["performance"]
        })

        # 4. 揭示
        time.sleep(0.1)
        reveal = char_data["reveal"]
        self._send_main({
            "type": "character_revealed",
            "name": reveal["name"],
            "title": reveal["title"],
            "era": reveal["era"],
            "summary": reveal["summary"],
            "message": f"刚才与你说话的，是{reveal['name']}。"
                       f"但他留下的，不只是名字，而是一种敢于扎根的颜色。"
        })

        # 5. 自动生成
        time.sleep(0.15)
        self._send_main({
            "type": "generation_result",
            "title": "岳麓松风",
            "paragraphs": [
                f"我站在岳麓书院的石阶前，"
                f"{self._color_name}在眼前铺展开来。",
                f"{objs_str}静静地立在那里，"
                f"仿佛已经等了千年。",
                "这不仅是颜色，也不仅是图像——"
                "这是一次穿越千年的相遇。",
                f"最终，我将这一抹{self._color_name}收入心底，"
                "作为湖大千年色最深的印记。"
            ],
            "narrative": f"岳麓松风，{self._color_name}染千年。",
            "image_prompt": "traditional Chinese ink wash painting...",
            "context": {
                "color": self._color_name,
                "objects": list(self.selected_objects),
                "character": reveal["name"]
            }
        })

        # 6. 重置状态（允许多次体验）
        self.selected_objects.clear()
        self._draw_count = 0
        self.fsm_mode = "COLOR_EXTRACTION"
        self._color_sub = "AWAITING"
        self._color_source = "object"
        self._color_name = "岳麓绿"

    # ── 张手 ────────────────────────────────────

    def _on_open_hand(self):
        if self.fsm_mode == "CANDIDATE":
            self.fsm_mode = "DRAWING"
        elif self.fsm_mode == "DRAWING":
            self.fsm_mode = "GLOBAL"

    # ── 食指伸出 ────────────────────────────────

    def _on_index_pointing(self):
        if self.fsm_mode == "GLOBAL":
            self.fsm_mode = "DRAWING"

    def _send_main(self, msg):
        """发送主通道消息"""
        if self.main_client:
            try:
                data = json.dumps(msg, ensure_ascii=False) + "\n"
                self.main_client.send(data.encode('utf-8'))
                logger.info(f"Main sent: {msg.get('type', '?')}")
            except Exception as e:
                logger.error(f"Main send error: {e}")

    def _send_hand_data(self, client):
        """发送模拟手部数据"""
        if client:
            try:
                import random
                data = {
                    "type": "hand_tracking",
                    "palm_center": [320 + random.randint(-10, 10), 240 + random.randint(-10, 10)],
                    "landmarks": [
                        {"x": 0.5 + random.uniform(-0.02, 0.02), "y": 0.5 + random.uniform(-0.02, 0.02)}
                        for _ in range(21)
                    ],
                    "gesture": "open_hand"
                }
                msg = json.dumps(data, ensure_ascii=False) + "\n"
                client.send(msg.encode('utf-8'))
            except:
                pass

    def stop(self):
        self.running = False
        if self.main_client:
            try:
                self.main_client.close()
            except:
                pass
        if self.hand_client:
            try:
                self.hand_client.close()
            except:
                pass


if __name__ == "__main__":
    backend = MockBackend().start()
    print("Mock backend running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping mock backend...")
        backend.stop()