#!/usr/bin/env python3
"""
完整集成测试 — 寻麓千年色

启动后自动:
  1. 打开 IP 摄像头
  2. 初始化手部跟踪 (MediaPipe)
  3. 运行手势状态机 (5 模式)
  4. 接驳草图识别 + 人物推荐桥接
  5. 双端口与 Unity 通信 (:8888 主通道, :8889 手部通道)

用法:
  python test_integrated.py [摄像头URL]

默认摄像头: config_ipcam.py 或 http://10.54.71.31:8080/video
无摄像头时自动使用假数据演示手势状态机流程。
"""

import cv2
import json
import socket
import sys
import time
import threading
import logging
from typing import Optional, Dict, List

sys.path.insert(0, ".")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Integrated")

# ── 模拟手部关键点（无摄像头时使用）───────────────────────

class FakeLandmark:
    """模拟 MediaPipe NormalizedLandmark"""
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z


def make_index_pointing_landmarks():
    """食指伸出"""
    lm = [FakeLandmark(0.5, 0.5)] * 21
    lm[8] = FakeLandmark(0.5, 0.15)   # 食指尖高
    lm[6] = FakeLandmark(0.5, 0.4)    # 食指 MCP
    lm[12] = FakeLandmark(0.5, 0.7)   # 中指尖低（弯曲）
    lm[10] = FakeLandmark(0.5, 0.5)
    lm[16] = FakeLandmark(0.5, 0.7)
    lm[14] = FakeLandmark(0.5, 0.5)
    lm[20] = FakeLandmark(0.5, 0.7)
    lm[18] = FakeLandmark(0.5, 0.5)
    return lm


def make_fist_landmarks():
    """握拳"""
    return [FakeLandmark(0.5, 0.8)] * 21


def make_open_hand_landmarks():
    """张手"""
    return [FakeLandmark(0.5, 0.2)] * 21


# ── 集成服务器 ────────────────────────────────────────────

class IntegratedServer:
    """集成测试服务器：双端口 + 手势状态机 + 桥接"""

    def __init__(self, camera_url: str = "", no_display: bool = False):
        self.camera_url = camera_url
        self.camera = None
        self.hand_tracker = None
        self.fsm = None
        self.sketch_bridge = None
        self.character_bridge = None

        # Socket
        self.main_socket: Optional[socket.socket] = None     # :8888 → Unity
        self.hand_socket: Optional[socket.socket] = None     # :8889 → Unity
        self.main_client: Optional[socket.socket] = None     # Unity 连接
        self.hand_client: Optional[socket.socket] = None
        self.main_server: Optional[socket.socket] = None
        self.hand_server: Optional[socket.socket] = None

        self.is_running = False
        self.use_fake_camera = False
        self.no_display = no_display
        self.current_color = "岳麓绿"  # 默认第一幕颜色
        self.selected_objects: List[str] = []

    # ── 启动 ───────────────────────────────────────────────

    def start(self):
        print("=" * 60)
        print("  寻麓千年色 — 集成测试服务器")
        print("=" * 60)

        # 1. 摄像头
        if not self._init_camera():
            print("\n[!] 摄像头不可用，使用模拟手势数据进行演示")
            print("    手势流程: 食指伸出→绘画→握拳确认→物象候选→人物推荐")
            self.use_fake_camera = True

        # 2. 手部跟踪
        if not self.use_fake_camera:
            self._init_hand_tracker()

        # 3. 手势状态机
        self._init_gesture_fsm()

        # 4. 桥接
        self._init_bridges()

        # 5. 启动 TCP 服务器
        self.is_running = True
        self._start_servers()

        # 6. 主循环
        print("\n[运行] 等待 Unity 连接...")
        if self.no_display:
            print("[模式] 无显示 — 延迟最低")
        else:
            print("[按键] ESC=退出 | 1-6=切换颜色")
        try:
            if self.use_fake_camera:
                self._run_demo_loop()
            else:
                self._run_camera_loop()
        except KeyboardInterrupt:
            print("\n[中断] Ctrl+C")
        finally:
            self._cleanup()

    # ── 摄像头 ─────────────────────────────────────────────

    def _init_camera(self) -> bool:
        url = self.camera_url
        if not url:
            try:
                from config_ipcam import CAMERA_URL
                url = CAMERA_URL
            except ImportError:
                url = "http://10.194.3.133:8080/video"

        print(f"\n[1] 连接摄像头: {url}")
        try:
            from vision.ipcamera import IPCamera
            self.camera = IPCamera(url)
            if self.camera.connect():
                print("[OK] 摄像头已连接")
                return True
        except Exception as e:
            print(f"     连接失败: {e}")
        return False

    def _init_hand_tracker(self):
        print("[2] 初始化手部跟踪 (MediaPipe)...")
        try:
            from vision.hand_tracker import HandTracker
            self.hand_tracker = HandTracker()
            print("[OK] 手部跟踪已就绪")
        except Exception as e:
            print(f"[错误] 手部跟踪初始化失败: {e}")
            self.use_fake_camera = True

    # ── 手势状态机 ─────────────────────────────────────────

    def _init_gesture_fsm(self):
        print("[3] 初始化手势状态机...")
        from vision.gesture_state_machine import create_gesture_state_machine
        self.fsm = create_gesture_state_machine()

        # 回调：模式切换 → 发送到 Unity
        self.fsm.on_mode_change = self._on_fsm_mode_change

        # 回调：绘画完成 → 识别物象
        self.fsm.on_drawing_commit = self._on_drawing_commit

        # 回调：绘画取消
        self.fsm.on_drawing_cancel = self._on_drawing_cancel

        # 回调：物象确认 → 触发人物推荐
        self.fsm.on_object_confirmed = self._on_object_confirmed

        # 回调：人物确认
        self.fsm.on_character_confirmed = self._on_character_confirmed

        # 回调：拒绝推荐 → 进入轮盘
        self.fsm.on_reject_recommendations = self._on_reject_recommendations

        print("[OK] 手势状态机已就绪 (初始: GLOBAL)")

    # ── 桥接 ───────────────────────────────────────────────

    def _init_bridges(self):
        print("[4] 初始化桥接模块...")

        # 草图识别器
        try:
            from vision.sketch_recognizer import create_sketch_recognizer
            recognizer = create_sketch_recognizer()
            self.sketch_bridge = _DirectSketchBridge(recognizer, self)
            print("[OK] SketchBridge 已就绪")
        except Exception as e:
            print(f"[!] SketchBridge 初始化失败: {e}")
            self.sketch_bridge = None

        # 人物推荐器
        try:
            from rag.character_recommend import CharacterRecommender
            recommender = CharacterRecommender()
            recommender._ensure_kb()
            self.character_bridge = _DirectCharacterBridge(recommender, self)
            print(f"[OK] CharacterBridge 已就绪 (人物库: {len(recommender._char_index)} 人)")
        except Exception as e:
            print(f"[!] CharacterBridge 初始化失败: {e}")
            self.character_bridge = None

    # ── TCP 服务器 ─────────────────────────────────────────

    def _start_servers(self):
        print("[5] 启动 TCP 服务器...")

        # :8888 主通道
        self.main_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.main_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.main_server.bind(("0.0.0.0", 8888))
        self.main_server.listen(1)
        self.main_server.settimeout(1.0)
        t1 = threading.Thread(target=self._accept_main, daemon=True, name="MainAccept")
        t1.start()
        print("[OK] 主通道 :8888 等待 Unity...")

        # :8889 手部通道
        self.hand_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.hand_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.hand_server.bind(("0.0.0.0", 8889))
        self.hand_server.listen(1)
        self.hand_server.settimeout(1.0)
        t2 = threading.Thread(target=self._accept_hand, daemon=True, name="HandAccept")
        t2.start()
        print("[OK] 手部通道 :8889 等待 Unity...")

    def _accept_main(self):
        while self.is_running:
            try:
                client, addr = self.main_server.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"\n[Unity] 主通道已连接: {addr}")
                self.main_client = client
                self._send_main({"type": "connected",
                                 "message": "integrated_server_ready"})
                # 发送初始手势状态
                self._send_gesture_state()
                self._handle_main_loop(client)
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"Main accept 错误: {e}")

    def _accept_hand(self):
        while self.is_running:
            try:
                client, addr = self.hand_server.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[Unity] 手部通道已连接: {addr}")
                self.hand_client = client
                self._send_hand({"type": "connected",
                                 "message": "hand_server_ready"})
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"Hand accept 错误: {e}")

    def _handle_main_loop(self, client: socket.socket):
        """处理来自 Unity 的消息（用 select 隔离收发，不影响 sendall）"""
        import select
        buf = ""
        while self.is_running:
            try:
                ready, _, _ = select.select([client], [], [], 0.5)
                if not ready:
                    continue
                data = client.recv(4096)
                if not data:
                    break
                buf += data.decode("utf-8")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if line.strip():
                        self._process_main_message(line.strip())
            except Exception:
                break
        print("[Unity] 主通道断开")
        self.main_client = None

    def _process_main_message(self, msg: str):
        try:
            data = json.loads(msg)
            msg_type = data.get("type", data.get("event", ""))
            print(f"[Unity→] {msg_type}: {json.dumps(data, ensure_ascii=False)[:120]}")

            if msg_type == "object_selected":
                obj_name = data.get("name", "")
                print(f"  → 物象选中: {obj_name}")
                self.selected_objects.append(obj_name)
                if self.character_bridge:
                    self.character_bridge.recommend(self.current_color,
                                                    self.selected_objects)
                # 通知 FSM
                if self.fsm:
                    self.fsm.trigger_object_candidates()

            elif msg_type == "character_selected":
                char_name = data.get("name", "")
                print(f"  → 人物选中: {char_name}")
                self._send_main({
                    "type": "character_confirmed",
                    "module_id": f"character_{char_name}",
                    "entity": char_name,
                })

            elif msg_type == "generation_start":
                print("  → 叙事生成请求")
                self._send_main({
                    "type": "generation_result",
                    "title": "你寻到的千年色",
                    "paragraphs": [
                        f"你选择了{self.current_color}作为底色。",
                        f"你放下了{'、'.join(self.selected_objects) if self.selected_objects else '一些物象'}。",
                        "历史人物走入了这个世界，",
                        "这就是你寻到的'千年色'。"
                    ],
                    "narrative": "叙事生成完成。",
                    "image_prompt": "一幅中国水墨画风格的湖大场景",
                })

            elif msg_type == "wheel_group_changed":
                pass  # 轮盘暂不实现
            elif msg_type == "wheel_character_selected":
                pass
            else:
                print(f"  → 未处理的消息类型: {msg_type}")

        except json.JSONDecodeError:
            print(f"[Unity→] JSON 解析失败: {msg[:100]}")

    # ── 发送方法 ───────────────────────────────────────────

    def _send_main(self, data: dict):
        msg_type = data.get("type", "?")
        if not self.main_client:
            print(f"  [SEND:DROP] {msg_type} — main_client 未连接")
            return
        try:
            msg = json.dumps(data, ensure_ascii=False) + "\n"
            self.main_client.sendall(msg.encode("utf-8"))
            if msg_type != "hand_tracking":
                print(f"  [SEND] {msg_type} ({len(msg)} bytes)")
        except Exception as e:
            print(f"  [SEND:ERR] {msg_type}: {e}")
            self.main_client = None

    def _send_hand(self, data: dict):
        if self.hand_client:
            try:
                msg = json.dumps(data, ensure_ascii=False) + "\n"
                self.hand_client.sendall(msg.encode("utf-8"))
            except Exception:
                self.hand_client = None

    def _send_gesture_state(self):
        if self.fsm:
            data = {
                "type": "gesture_state",
                "mode": self.fsm.mode.value,
                "sub_state": self.fsm.sub_state,
                "gesture": self.fsm.current_gesture.value if self.fsm.current_gesture else "none",
            }
            print(f"  [FSM→Unity] {data['mode']}/{data['sub_state']}/{data['gesture']}")
            self._send_main(data)

    # ── FSM 回调 ───────────────────────────────────────────

    def _on_fsm_mode_change(self, mode: str, sub_state: str, gesture: str):
        print(f"  [FSM] mode={mode} sub={sub_state} gesture={gesture}")
        self._send_gesture_state()

    def _on_drawing_commit(self, trajectory):
        print(f"  [FSM] 绘画提交! 轨迹点数={len(trajectory)}")
        if self.sketch_bridge:
            candidates = self.sketch_bridge.recognize(trajectory, self.current_color)
            if candidates:
                self._send_main({
                    "type": "object_candidates",
                    "color": self.current_color,
                    "candidates": candidates,
                })
                print(f"  → 已发送 {len(candidates)} 个物象候选到 Unity")
            else:
                print("  → 未识别到物象")

    def _on_drawing_cancel(self):
        print("  [FSM] 绘画取消")
        self._send_gesture_state()

    def _on_object_confirmed(self):
        print("  [FSM] 物象已确认 → 触发人物推荐")
        if self.character_bridge and self.selected_objects:
            self.character_bridge.recommend(self.current_color, self.selected_objects)

    def _on_character_confirmed(self):
        print("  [FSM] 人物已确认!")
        self._send_gesture_state()

    def _on_reject_recommendations(self):
        print("  [FSM] 拒绝推荐 → 进入轮盘")

    # ── 摄像头主循环 ───────────────────────────────────────

    def _run_camera_loop(self):
        from vision.hand_tracker import HAND_CONNECTIONS
        ts = 0
        while self.is_running:
            # ── 排空摄像头缓冲，只取最新帧 ──
            frame = None
            for _ in range(4):
                f = self.camera.read_frame()
                if f is not None:
                    frame = f

            if frame is None:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]

            # 手部检测
            results = self.hand_tracker._detect(frame, ts)

            if results.hand_landmarks:
                hand_lm = results.hand_landmarks[0]

                # → 手势状态机（使用归一化坐标）
                self.fsm.process(hand_lm, ts)

                # → Unity 手部数据（从已检测结果直接计算像素坐标）
                pixel_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
                palm_x = sum(p[0] for p in pixel_landmarks) // 21
                palm_y = sum(p[1] for p in pixel_landmarks) // 21
                wrist = pixel_landmarks[0]
                fingertips = [pixel_landmarks[i] for i in (4, 8, 12, 16, 20)]

                if self.hand_client:
                    lm_flat = []
                    for x, y in pixel_landmarks:
                        lm_flat.extend([x, y])
                    ft_flat = []
                    for x, y in fingertips:
                        ft_flat.extend([x, y])
                    self._send_hand({
                        "type": "hand_tracking",
                        "palm_center": [palm_x, palm_y],
                        "wrist": wrist,
                        "landmarks": lm_flat,
                        "fingertips": ft_flat,
                    })

                # ── 可视化（仅在非 --no-display 模式） ──
                if not self.no_display:
                    display = frame.copy()
                    for i, lm in enumerate(hand_lm):
                        px, py = int(lm.x * w), int(lm.y * h)
                        color = (0, 255, 0) if i % 4 == 0 else (0, 200, 0)
                        cv2.circle(display, (px, py), 4, color, -1)
                    for a, b in HAND_CONNECTIONS:
                        pt1 = (int(hand_lm[a].x * w), int(hand_lm[a].y * h))
                        pt2 = (int(hand_lm[b].x * w), int(hand_lm[b].y * h))
                        cv2.line(display, pt1, pt2, (0, 255, 0), 1)

                    cv2.putText(display, f"Mode: {self.fsm.mode.value}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display, f"Sub: {self.fsm.sub_state}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(display, f"Color: {self.current_color}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    u_status = "Unity: OK" if self.main_client else "Unity: --"
                    u_color = (0, 255, 0) if self.main_client else (0, 0, 255)
                    cv2.putText(display, u_status, (10, h - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, u_color, 2)
            else:
                self.fsm.process(None, ts)
                if not self.no_display:
                    display = frame.copy()

            if not self.no_display:
                cv2.imshow("Hand + Gesture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif ord('1') <= key <= ord('6'):
                colors = ["岳麓绿", "书院红", "西迁黄", "湘江蓝", "校徽金", "墨色"]
                self.current_color = colors[key - ord('1')]
                print(f"[颜色] 切换到: {self.current_color}")

            ts += 33

        cv2.destroyAllWindows()

    # ── 演示循环（无摄像头）────────────────────────────────

    def _run_demo_loop(self):
        """使用模拟手势数据演示状态机流转"""
        print("\n" + "=" * 60)
        print("  演示模式 — 自动模拟手势交互流程")
        print("  按键: 1=食指伸出 2=握拳 3=张手 ESC=退出")
        print("=" * 60)

        ts = 0
        current_gesture = make_open_hand_landmarks()
        gesture_name = "open_hand"

        while self.is_running:
            self.fsm.process(current_gesture, ts)

            mode = self.fsm.mode.value
            sub = self.fsm.sub_state
            gest = self.fsm.current_gesture.value
            pts = self.fsm.trajectory_point_count

            # 清除屏幕并打印状态
            print(f"\rMode={mode:16s} Sub={sub:12s} Gesture={gest:16s} Traj={pts:4d}  "
                  f"Color={self.current_color:6s}  "
                  f"Unity={'OK' if self.main_client else '--'}  "
                  f"[1=食指 2=握拳 3=张手 ESC=退出]",
                  end="", flush=True)

            # 在 DRAWING.TRACKING 时，累积一些轨迹后自动触发握拳
            if mode == "DRAWING" and sub == "TRACKING" and pts > 30:
                print(f"\n  → 自动握拳提交 (已录制 {pts} 点)")
                current_gesture = make_fist_landmarks()
                gesture_name = "fist"
                continue

            # 检查键盘
            key = cv2.waitKey(500) & 0xFF
            if key == 27:
                break
            elif key == ord('1'):
                current_gesture = make_index_pointing_landmarks()
                gesture_name = "index_pointing"
                print(f"\n  → 食指伸出")
            elif key == ord('2'):
                current_gesture = make_fist_landmarks()
                gesture_name = "fist"
                print(f"\n  → 握拳")
            elif key == ord('3'):
                current_gesture = make_open_hand_landmarks()
                gesture_name = "open_hand"
                print(f"\n  → 张手")
            elif ord('1') <= key <= ord('6'):
                colors = ["岳麓绿", "书院红", "西迁黄", "湘江蓝", "校徽金", "墨色"]
                self.current_color = colors[key - ord('1')]
                print(f"\n[颜色] → {self.current_color}")
            elif key == ord('r') and self.fsm.mode.value == "CANDIDATE":
                # 模拟在候选模式下确认物象
                self.selected_objects.append("古树")
                if self.character_bridge:
                    print("\n  → 手动触发人物推荐")
                    self.character_bridge.recommend(self.current_color, self.selected_objects)

            ts += 500

        cv2.destroyAllWindows()

    # ── 清理 ───────────────────────────────────────────────

    def _cleanup(self):
        self.is_running = False
        for sock in [self.main_client, self.hand_client,
                     self.main_server, self.hand_server]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        if self.camera:
            try:
                self.camera.release()
            except Exception:
                pass
        if self.hand_tracker:
            try:
                self.hand_tracker.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("\n[退出] 集成服务器已停止")


# ── 直接桥接（不经过 UnitySender，直接回调到 IntegratedServer）─

class _DirectSketchBridge:
    """草图识别桥接 — 直接发送到集成服务器"""

    def __init__(self, recognizer, server: IntegratedServer):
        self.recognizer = recognizer
        self.server = server

    def recognize(self, trajectory, color: str) -> List[Dict]:
        """识别轨迹并返回候选（trajectory 为 [(x,y,ts_ms), ...]）"""
        try:
            results = self.recognizer.recognize_from_fingertip_history(trajectory, color=color)
            if not results:
                return []
            return [
                {"name": r.entity_name, "score": round(r.score, 4),
                 "qd_category": r.qd_category}
                for r in results[:3]
            ]
        except Exception as e:
            logger.error(f"SketchBridge 识别失败: {e}")
            # 降级：返回假候选
            return [
                {"name": "古树", "score": 0.88, "qd_category": "tree"},
                {"name": "竹林", "score": 0.72, "qd_category": "tree"},
                {"name": "石阶", "score": 0.55, "qd_category": "stairs"},
            ]


class _DirectCharacterBridge:
    """人物推荐桥接 — 直接发送到集成服务器"""

    def __init__(self, recommender, server: IntegratedServer):
        self.recommender = recommender
        self.server = server

    def recommend(self, color: str, objects: List[str]) -> List[Dict]:
        try:
            results = self.recommender.recommend(
                color=color, objects=objects,
                selected_characters=[], use_llm=False, top_k=3
            )
            if not results:
                return []
            candidates = [
                {"name": r.name, "title": r.title,
                 "score": round(r.score, 4), "reason": r.reason}
                for r in results
            ]
            self.server._send_main({
                "type": "character_candidates",
                "candidates": candidates,
            })
            print(f"  → 已发送 {len(candidates)} 个人物推荐到 Unity")
            return candidates
        except Exception as e:
            logger.error(f"CharacterBridge 推荐失败: {e}")
            return []


# ── 入口 ──────────────────────────────────────────────────

def main():
    no_display = "--no-display" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    camera_url = args[0] if args else ""
    server = IntegratedServer(camera_url=camera_url, no_display=no_display)
    server.start()


if __name__ == "__main__":
    main()
