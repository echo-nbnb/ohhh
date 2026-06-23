#!/usr/bin/env python3
"""
完整集成测试 — 寻麓千年色

启动后自动:
  1. 打开 IP 摄像头
  2. 初始化手部跟踪 (MediaPipe)
  3. 运行手势状态机 (5 模式)
  4. 接驳草图识别 + 人物推荐桥接
  5. 双端口 TCP 通信 (:8888 主通道, :8889 手部通道)

用法:
  python test_integrated.py [摄像头URL]

默认摄像头: config_ipcam.py 或 http://10.54.71.31:8080/video
无摄像头时自动使用假数据演示手势状态机流程。
"""

import os
import cv2
import json
import socket
import sys
import time
import threading
import logging
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List

sys.path.insert(0, ".")

from vision.gesture_state_machine import GestureMode, DrawingSubState, ColorExtractionSubState  # noqa: E402
from vision.color_stability_detector import ColorStabilityDetector  # noqa: E402

# ── 物象旁白（筑景确认时发送）──
_OBJECT_NARRATION = {
    "东方红广场": "广场的石阶上，晨光正一寸一寸地醒来。你画下的，是无数人出发的地方。",
    "中国书院博物馆": "这座博物馆里，藏着千年书院的呼吸。你看见了它的轮廓。",
    "书卷": "纸页轻翻，墨香未散。一卷书，便是一座移动的书院。",
    "书架": "木纹里藏着年岁，一格一格，像是时间的抽屉。你画下了知识栖居的地方。",
    "书案": "一方书案，曾有人在此伏案终夜。你的指尖触碰了那个夜晚。",
    "匾额": "匾额高悬，字迹如铁。那是先贤留给这片土地的一句承诺。",
    "古树": "这棵树见过朱熹，见过张栻，见过无数来来往往的人。它的年轮里，是一部无声的校史。",
    "古籍": "纸页泛黄，字迹斑驳。翻开它，就翻开了一段被遗忘的对话。",
    "图书馆": "灯光照着无数书脊，像一座安静的城。你画下了那座城的大门。",
    "墨锭": "墨在水中化开，像夜在黎明散开。你画下了时间的颜色。",
    "学位帽": "穗子从右拨到左，一瞬之间，四年已过。你画下了一个人的抵达。",
    "实验室": "试管里的液体微微泛光，数据在屏幕上跳动。真理正在被测量。",
    "屋脊": "青瓦层叠，檐角微翘。屋脊之上，是千年的雨声和风声。",
    "山石": "岳麓山的石头，每一块都听过书声。你画下了一块沉默的见证者。",
    "岳麓书院": "惟楚有材，于斯为盛。你画下了一座千年学府的轮廓。",
    "岳麓山": "山不高，却厚重如史。岳麓山上，每一棵树都知道一些故事。",
    "操场": "跑道一圈一圈，汗水落在地上，长出了青春。你画下了一片奔跑的风景。",
    "教学楼": "窗子里透出灯光，黑板上的公式还没擦。你画下了求知的日常。",
    "显微镜": "镜下的世界，微小却宏大。你画下了发现的眼睛。",
    "林荫道": "梧桐叶落下的时候，整条路都是金色的。你画下了一条通往秋天的小径。",
    "校徽": "一个圆，一圈字，一座山，一条江。你画下的，是这里所有人的归属。",
    "校门": "没有门的校门，却比任何门都重。走进来的人，都成了它的一部分。",
    "楹联": "对联两侧，藏着对仗的智慧。你的笔触碰触了汉语的筋骨。",
    "毛笔": "一支笔，在纸上走了一千年。你画下了书写本身。",
    "湖南大学大礼堂": "穹顶之下，曾有无数人在这里聆听、思考、起立。你画下了声音的容器。",
    "湘江": "江水千年如一日地流，带走了时间，留下了名字。你画下了一条会说话的河。",
    "爱晚亭": "停车坐爱枫林晚。你画下了一座停驻在诗句里的亭子。",
    "牌楼路": "一条路，从牌楼延伸到山脚。你画下了起点和终点之间的风景。",
    "白鹤泉": "泉水清冽，传说有白鹤来饮。你画下了一汪有灵的水。",
    "石桥": "石头连着石头，就是路。你画下的桥，连接了此岸和彼岸。",
    "石阶": "一级一级，通向更高的地方。你画下了一段向上的旅程。",
    "砚台": "墨在砚中，水在墨中。一方砚，是书写之前的那一秒空白。",
    "碑刻": "石头上的文字，比纸上的活得更久。你画下了不朽。",
    "窗格": "一格一格，把光分成细碎的金子。你画下了光的形状。",
    "竹林": "风过竹林，沙沙作响。那是千年来同样的声音。",
    "竹简": "竹片上的文字，比纸更古老。你画下了文明的初页。",
    "笔记本": "一页一页，密密麻麻的字迹。你画下了一个人的努力。",
    "线装书": "棉线穿过书脊，纸页对折。你画下了一种古老的装订方式。",
    "经卷": "经文在手，墨色如新。一卷经，是信仰的重量。",
    "自卑亭": "行远自迩，登高自卑。你画下了一座谦逊的建筑。",
    "荣誉证书": "一张纸，盖着红章。那是努力被记得的方式。",
    "讲堂": "三尺讲台，一方黑板。你画下了知识传递的现场。",
    "设计院楼": "玻璃幕墙映着天光，建筑的图纸正在生成。你画下了创造的过程。",
    "赫曦台": "台名取自日出，站在这里，能看到第一缕光。你画下了黎明。",
    "长廊": "木柱排列，光影交错。长廊无尽，像时间本身。",
    "院墙": "青灰色的墙，隔开喧嚣与寂静。你画下了一道守护的边界。",
    "麓山南路": "从头走到尾，是一届又一届学生的四年。你画下了一条记忆的长河。",
    "黑板": "黑板上的粉笔痕，擦去了又来。你画下了一个轮回。",
}

# ── DeepSeek API 配置 ──
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Integrated")

# ── 摄像头调试日志（写入文件，方便实时查看）──
_debug_log = logging.getLogger("CameraDebug")
_debug_log.setLevel(logging.DEBUG)
_debug_fh = logging.FileHandler("camera_debug.log", encoding="utf-8", mode="w")
_debug_fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
_debug_log.addHandler(_debug_fh)
_debug_log.propagate = False  # 不输出到控制台，避免刷屏
_debug_log.info("=== Camera Debug Log Started ===")

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

    def __init__(self, camera_url: str = "", no_display: bool = False, no_camera: bool = False):
        self.camera_url = camera_url
        self.camera = None
        self.webcam = None  # 电脑端摄像头（衣物颜色）
        self.hand_tracker = None
        self.fsm = None
        self.sketch_bridge = None
        self.character_bridge = None
        self.object_color_detector = None  # 物件颜色检测
        self.webcam_color_detector = None  # 衣物颜色检测

        # Socket
        self.main_socket: Optional[socket.socket] = None     # :8888 → 前端
        self.hand_socket: Optional[socket.socket] = None     # :8889 → 前端
        self.main_client: Optional[socket.socket] = None     # 前端连接
        self.hand_client: Optional[socket.socket] = None
        self.main_server: Optional[socket.socket] = None
        self.hand_server: Optional[socket.socket] = None

        self.is_running = False
        self.use_fake_camera = False
        self.no_display = no_display
        self.no_camera = no_camera
        self.frame_count = 0
        self.current_color = "桂黄"  # 默认第一幕颜色
        self.selected_objects: List[str] = []
        self.sketch_trajectories: Dict[str, List] = {}  # {物象名: [(x,y,ts_ms), ...]}
        self._current_frame: Optional[np.ndarray] = None  # 最新帧，用于回调中检测

        # 手部出现/消失检测（用于 Cover Flow）
        self._prev_hand_detected = False
        self._hand_appeared_sent = False  # 防止重复发送
        self._pending_trajectory: List = []  # 临时存储待确认的轨迹
        self._waiting_for_screenshot = False  # 等前端截图再生成明信片
        self._pending_postcard_data = None    # 缓存明信片数据等截图
        self._hand_lost_frames = 0            # 手部丢失帧计数（自动提交用）
        self._color_done = False              # 颜色确认后才开始绘画
        self._collected_colors: List[str] = []  # 收集的两种颜色
        self._color_round = 1                  # 当前检测轮次：1=第一色，2=第二色
        self._send_lock = threading.Lock()    # 保护 _send_main 并发访问
        self._pipeline_running = False        # 防止重复启动人物管线
        self.color_detector = ColorStabilityDetector(box_size=120, confirm_seconds=3.0)

    # ── 启动 ───────────────────────────────────────────────

    def start(self):
        print("=" * 60)
        print("  寻麓千年色 — 集成测试服务器")
        print("=" * 60)

        # 1. 摄像头
        if self.no_camera:
            print("[!] --no-camera: 跳过摄像头，仅 TCP 手势模拟")
            self.use_fake_camera = True
        elif not self._init_camera():
            print("\n[!] 摄像头不可用，使用模拟手势数据进行演示")
            print("    手势流程: 食指伸出→绘画→握拳确认→物象候选→人物推荐")
            self.use_fake_camera = True

        # 2. 手部跟踪
        if not self.use_fake_camera:
            self._init_hand_tracker()

        # 3. 手势状态机
        self._init_gesture_fsm()

        # 4. 启动 TCP 服务器（先启动，确保不被桥接卡住）
        self.is_running = True
        self._start_servers()

        # 5. 桥接
        self._init_bridges()

        # 6. 主循环
        print("\n[运行] 等待 前端连接...")
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
        # 直接用电脑摄像头
        print(f"\n[1] 打开电脑摄像头...")
        ok = False
        try:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print("[OK] 电脑摄像头已就绪")
                ok = True
        except Exception as e:
            print(f"     电脑摄像头失败: {e}")

        if not ok:
            print("[!] 摄像头不可用")
            self.use_fake_camera = True

        return ok

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
        self.fsm = create_gesture_state_machine(debounce_frames=3)

        # 回调：模式切换 → 发送到前端
        self.fsm.on_mode_change = self._on_fsm_mode_change

        # 回调：颜色提取开始
        self.fsm.on_color_extraction_start = self._on_color_extraction_start

        # 回调：物件颜色检测完成
        self.fsm.on_object_color_detected = self._on_object_color_detected

        # 回调：衣物兜底触发
        self.fsm.on_clothing_fallback = self._on_clothing_fallback

        # 回调：颜色确认
        self.fsm.on_color_confirmed = self._on_color_confirmed

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

        # 启动时自动进入颜色提取状态
        self.fsm.trigger_color_extraction_start()

        print("[OK] 手势状态机已就绪 (初始: COLOR_EXTRACTION)")

    # ── 桥接 ───────────────────────────────────────────────

    def _init_bridges(self):
        print("[4] 初始化桥接模块...")

        # 物件颜色检测器
        try:
            from vision.color_detector import ObjectColorDetector
            self.object_color_detector = ObjectColorDetector()
            print("[OK] ObjectColorDetector 已就绪")
        except Exception as e:
            print(f"[!] ObjectColorDetector 初始化失败: {e}")

        # 衣物颜色检测器（simple 方法，不依赖 MediaPipe）
        try:
            from vision.webcam_color_detector import WebcamColorDetector
            self.webcam_color_detector = WebcamColorDetector(method="simple")
            print("[OK] WebcamColorDetector 已就绪 (simple)")
        except Exception as e:
            print(f"[!] WebcamColorDetector 初始化失败: {e}，将随机兜底")
            self.webcam_color_detector = None

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
        print("[OK] 主通道 :8888 等待前端...")

        # :8889 手部通道
        self.hand_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.hand_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.hand_server.bind(("0.0.0.0", 8889))
        self.hand_server.listen(1)
        self.hand_server.settimeout(1.0)
        t2 = threading.Thread(target=self._accept_hand, daemon=True, name="HandAccept")
        t2.start()
        print("[OK] 手部通道 :8889 等待前端...")

    def _accept_main(self):
        while self.is_running:
            try:
                client, addr = self.main_server.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"\n[前端] 主通道已连接: {addr}")

                # 断开旧客户端（如果有）
                if self.main_client:
                    try:
                        self.main_client.close()
                    except Exception:
                        pass

                self.main_client = client

                # 新连接：重置物件/轨迹，但不重置 _color_done
                # （避免 React StrictMode 双挂载把刚择好的色清零）
                self._pipeline_running = False
                self._hand_lost_frames = 0
                self._waiting_for_screenshot = False
                self._collected_colors.clear()
                self.selected_objects.clear()
                self.sketch_trajectories.clear()
                self._pending_trajectory = []
                self.fsm._recognized_object = None
                if self.fsm:
                    self.fsm.reset_to_global()
                    # 只在真正空闲时触发择色（_color_done=False 说明还没择过色）
                    if not self._color_done:
                        self.fsm.trigger_color_extraction_start()
                _debug_log.info("STATE_RESET | 新客户端连接，完整重置回 COLOR_EXTRACTION")

                self._send_main({"type": "connected",
                                 "message": "integrated_server_ready"})
                # 发送当前手势状态（同步给新前端）
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
                print(f"[前端] 手部通道已连接: {addr}")
                self.hand_client = client
                self._send_hand({"type": "connected",
                                 "message": "hand_server_ready"})
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"Hand accept 错误: {e}")

    def _handle_main_loop(self, client: socket.socket):
        """处理来自前端的消息（用 select 隔离收发，不影响 sendall）"""
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
        print("[前端] 主通道断开")
        self.main_client = None

    def _process_main_message(self, msg: str):
        try:
            data = json.loads(msg)
            msg_type = data.get("type", data.get("event", ""))
            print(f"[前端→] {msg_type}: {json.dumps(data, ensure_ascii=False)[:80]}")

            if msg_type == "trigger_color_detect":
                # 前端 R 键 → 启动框内颜色检测
                self._start_color_detect()
                return

            if msg_type == "trigger_character_pipeline":
                # 前端确认 2 个物象 → 启动人物推荐管线
                color = data.get("color", self.current_color)
                objects = data.get("objects", list(self.selected_objects))
                print(f"  → 前端触发人物管线: color={color} objects={objects}")
                if not self._pipeline_running:
                    self.selected_objects = list(objects)
                    self._pipeline_running = True
                    snapshot = {"color": color, "objects": objects}
                    threading.Thread(target=self._run_character_pipeline, args=(snapshot,), daemon=True, name="CharacterPipeline").start()
                return

            if msg_type == "start_color_extraction":
                # 前端进入 Act2 → 发送提示，等 R 键
                print("  → 前端进入 Act2，等待 R 键")
                return

            if msg_type == "gesture_simulate":
                # TCP 手势模拟（测试用）
                gesture = data.get("gesture", "")
                lm = self._make_fake_landmarks(gesture)
                if lm and self.fsm:
                    self.fsm.process(lm, int(time.time() * 1000))
                return

            elif msg_type == "object_selected":
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
            elif msg_type == "screenshot_upload":
                # 前端发来的明信片截图 → 上传OSS → 回传URL → 重置
                print(f"[Screenshot] RECEIVED! Size: {len(data.get('image_base64',''))}")
                import base64 as b64
                import io as _io
                from PIL import Image as _PILImage
                img_b64 = data.get("image_base64", "")
                if img_b64 and img_b64.startswith("data:image"):
                    try:
                        img_b64 = img_b64.split(",", 1)[1]
                        img_bytes = b64.b64decode(img_b64)
                        img = _PILImage.open(_io.BytesIO(img_bytes))
                        from rag.uploader import PostcardUploader
                        uploader = PostcardUploader()
                        result = uploader.upload(img)
                        self._send_main({
                            "type": "postcard_result",
                            "image_url": result["image_url"],
                            "qr_base64": result["qr_base64"],
                            "unique_id": result["unique_id"],
                            "message": "扫码带走你的千年色。"
                        })
                        print(f"  [Screenshot] 截图已上传: {result['image_url']}")
                        # 上传成功 → 重置状态准备下一轮
                        self._waiting_for_screenshot = False
                        self._pending_postcard_data = None
                        self.selected_objects.clear()
                        self.sketch_trajectories.clear()
                        self._pending_trajectory = []
                        self.fsm._recognized_object = None
                        self.fsm.reset_to_global()
                        self.fsm.trigger_color_extraction_start()
                        _debug_log.info("STATE_RESET | 截图上传完成，重置回 COLOR_EXTRACTION")
                    except Exception as e:
                        print(f"  [Screenshot] 上传失败: {e}")
            else:
                print(f"  → 未处理的消息类型: {msg_type}")

        except json.JSONDecodeError:
            print(f"[前端→] JSON 解析失败: {msg[:100]}")

    # ── 发送方法 ───────────────────────────────────────────

    def _send_main(self, data: dict):
        msg_type = data.get("type", "?")
        _debug_log.debug(f"SEND_MAIN | type={msg_type} | {json.dumps(data, ensure_ascii=False)[:200]}")
        with self._send_lock:
            if not self.main_client:
                if msg_type not in ("hand_tracking",):
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
            print(f"  [FSM→前端] {data['mode']}/{data['sub_state']}/{data['gesture']}")
            self._send_main(data)

    # ── FSM 回调 ───────────────────────────────────────────

    def _make_fake_landmarks(self, gesture: str):
        """TCP 手势模拟 → Fake Landmarks"""
        class FL:
            def __init__(self, x, y, z=0): self.x = x; self.y = y; self.z = z
        if gesture == "fist":
            lm = [FL(0.5, 0.5)] * 21
            for ti, mi in [(4,3),(8,6),(12,10),(16,14),(20,18)]:
                lm[ti] = FL(0.5, 0.85); lm[mi] = FL(0.5, 0.45)
            return lm
        elif gesture == "open_hand":
            lm = [FL(0.5, 0.5)] * 21
            for ti, mi in [(4,3),(8,6),(12,10),(16,14),(20,18)]:
                lm[ti] = FL(0.5, 0.15); lm[mi] = FL(0.5, 0.45)
            return lm
        elif gesture == "index_pointing":
            lm = [FL(0.5, 0.5)] * 21
            for ti, mi in [(4,3),(8,6),(12,10),(16,14),(20,18)]:
                lm[ti] = FL(0.5, 0.85); lm[mi] = FL(0.5, 0.45)
            lm[8] = FL(0.5, 0.1); lm[6] = FL(0.5, 0.4)
            return lm
        return None

    def _on_fsm_mode_change(self, mode: str, sub_state: str, gesture: str):
        _debug_log.info(f"FSM_MODE_CHANGE | mode={mode} sub={sub_state} gesture={gesture}")
        print(f"  [FSM] mode={mode} sub={sub_state} gesture={gesture}")
        if mode == "DRAWING" and sub_state == "TRACKING":
            self._send_main({
                "type": "drawing_start",
                "message": "伸出食指，开始作画。"
            })
        self._send_gesture_state()

    def _on_drawing_commit(self, trajectory):
        """第二幕：直接识别最优结果 → 自动确认 → 触发人物推荐"""
        print(f"  [FSM] 绘画提交! 轨迹点数={len(trajectory)}")
        # 保存轨迹，等待确认后存入 sketch_trajectories
        self._pending_trajectory = trajectory
        if self.sketch_bridge:
            results = self.sketch_bridge.recognize(trajectory, self.current_color)
            if results:
                # 去重：已有同名物象 → 取下一个候选
                top = results[0]
                name, score, qd_cat = top['name'], top['score'], top['qd_category']
                for r in results:
                    if r['name'] not in self.selected_objects:
                        name, score, qd_cat = r['name'], r['score'], r['qd_category']
                        break
                else:
                    # 所有候选都已存在 → 从 qd_map 找同类不同名物象
                    import random as _r
                    _near = [n for n in _OBJECT_NARRATION if n not in self.selected_objects]
                    if _near:
                        name = _r.choice(_near)
                        score, qd_cat = 0.30, "fallback"
                self.fsm._recognized_object = (name, score, qd_cat)
                _narration = _OBJECT_NARRATION.get(name, f"你画下了{name}。它的轮廓渐渐清晰。")
                self._send_main({
                    "type": "object_recognized",
                    "color": self.current_color,
                    "object": {
                        "name": name,
                        "score": round(score, 4),
                        "qd_category": qd_cat
                    },
                    "narration": _narration,
                })
                _debug_log.info(f"DRAWING_COMMIT | object={name} score={score:.2f} | AUTO_CONFIRM")
                print(f"  → 已识别物象: {name} ({score:.2f}) → {_narration[:30]}…")
                self._on_object_confirmed(name, score, qd_cat)
            else:
                # 识别结果为空 → 兜底随机物象（排除已有）
                import random as _r
                _all = list(_OBJECT_NARRATION.keys())
                _avail = [n for n in _all if n not in self.selected_objects]
                if not _avail:
                    _avail = _all
                name = _r.choice(_avail)
                _qd_fb = {"古树":"tree","书卷":"book","石阶":"stairs","岳麓书院":"house",
                          "湘江":"river","爱晚亭":"castle","石桥":"bridge","竹林":"bush",
                          "林荫道":"tree","讲堂":"church","图书馆":"house","实验室":"computer",
                          "岳麓山":"mountain","白鹤泉":"pond","校门":"door","院墙":"fence",
                          "长廊":"fence","屋脊":"umbrella","窗格":"hexagon","碑刻":"diamond",
                          "匾额":"face","校徽":"circle","东方红广场":"square","学位帽":"hat",
                          "设计院楼":"house","教学楼":"house","赫曦台":"castle",
                          "中国书院博物馆":"house","自卑亭":"house","操场":"baseball",
                          "山石":"mountain","墨锭":"coffee cup","砚台":"cup","毛笔":"pencil",
                          "笔记本":"pencil","书架":"backpack","书案":"basket","古籍":"book",
                          "线装书":"book","经卷":"book","竹简":"book","显微镜":"binoculars",
                          "楹联":"door","荣誉证书":"envelope","牌楼路":"stairs",
                          "麓山南路":"stairs","湖南大学大礼堂":"church","黑板":"television"}
                qd_cat = _qd_fb.get(name, "tree")
                score = round(_r.uniform(0.25, 0.35), 2)
                self.fsm._recognized_object = (name, score, qd_cat)
                self._send_main({"type": "object_recognized", "color": self.current_color,
                    "object": {"name": name, "score": round(score, 4), "qd_category": qd_cat}})
                _debug_log.info(f"DRAWING_COMMIT | object={name} score={score:.2f} | FALLBACK")
                print(f"  → 未识别到物象，兜底: {name}")
                self._on_object_confirmed(name, score, qd_cat)

    def _on_drawing_cancel(self):
        """绘画取消 → 回到全局等待重新画"""
        print("  [FSM] 绘画取消")
        self.fsm._recognized_object = None
        self._pending_trajectory = []
        self._send_main({
            "type": "drawing_cancelled",
            "message": ""
        })
        self._send_gesture_state()

    def _on_object_confirmed(self, name: str, score: float, qd_cat: str):
        """物象确认 → 第一轮回GLOBAL等第二轮，第二轮才触发人物推荐"""
        # 重名检测：如果和已有物象相同，随机换一个
        if name in self.selected_objects:
            import random as _r
            _all = ["东方红广场","中国书院博物馆","书卷","书架","书案","匾额",
                    "古树","古籍","图书馆","墨锭","学位帽","实验室","屋脊","山石",
                    "岳麓书院","岳麓山","操场","教学楼","显微镜","林荫道",
                    "校徽","校门","楹联","毛笔","湖南大学大礼堂","湘江","爱晚亭",
                    "牌楼路","白鹤泉","石桥","石阶","砚台","碑刻","窗格",
                    "竹林","竹简","笔记本","线装书","经卷","自卑亭",
                    "荣誉证书","讲堂","设计院楼","赫曦台","长廊","院墙",
                    "麓山南路","黑板"]
            _others = [n for n in _all if n not in self.selected_objects]
            new_name = _r.choice(_others) if _others else _r.choice(_all)
            print(f"  [FSM] 物象重名 {name} → 替换为 {new_name}")
            # 通知前端用新名字的图片
            self._send_main({
                "type": "object_recognized",
                "color": self.current_color,
                "object": {"name": new_name, "score": 0.35, "qd_category": "fallback"}
            })
            name = new_name
        print(f"  [FSM] 物象已确认: {name} ({score:.2f})")
        self.selected_objects.append(name)
        if self._pending_trajectory:
            self.sketch_trajectories[name] = self._pending_trajectory
            self._pending_trajectory = []

        objs = list(self.selected_objects)
        self._send_main({
            "type": "object_confirmed",
            "object": name,
            "objects_so_far": objs,
            "message": f"一个意象已经落下。" if len(objs) == 1
                       else f"又一个意象落下。{'、'.join(objs)}，它们在一起了。",
            "can_continue": len(objs) >= 2
        })

        # 只有确认了2个物象后才触发人物推荐管线
        if len(objs) < 2:
            print(f"  [FSM] 还需要第2个物象，回GLOBAL等待")
            self.fsm._recognized_object = None
            self._pending_trajectory = []
            self.fsm.reset_to_global()
            _debug_log.info("STATE_RESET_TO_GLOBAL | 等待第2个物象")
            return

        # ── 2个物象确认完毕，在后台线程中运行人物管线（避免阻塞摄像头循环）──
        if self._pipeline_running:
            print("  [FSM] 人物管线已在运行中，跳过重复触发")
            return
        self._pipeline_running = True
        # 快照当前状态，防止主线程修改导致不一致
        snapshot = {
            "color": self.current_color,
            "objects": list(self.selected_objects),
        }
        thread = threading.Thread(
            target=self._run_character_pipeline,
            args=(snapshot,),
            daemon=True,
            name="CharacterPipeline"
        )
        thread.start()

    def _run_character_pipeline(self, snapshot: dict):
        """后台线程：人物推荐 → LLM 独白/叙事 → 明信片生成 → 状态重置"""
        color = snapshot["color"]
        objs = snapshot["objects"]
        print(f"  [Pipeline] START color={color} objs={objs}")

        try:
            if not self.character_bridge:
                print(f"  [Pipeline] NO character_bridge, abort")
                self._pipeline_running = False
                return

            candidates = self.character_bridge.recommend(color, objs)
            if not candidates:
                self._pipeline_running = False
                return

            top = candidates[0]
            print(f"  [Pipeline] top character: {top.get('name','?')} ({top.get('title','')}) score={top.get('score',0):.3f}")
            # ── 搜索阶段（Act4 转场 pacing）──
            self._send_main({
                "type": "character_search_start",
                "message": f"你的{color}指向{'、'.join(objs)}。"
                           f"一位与「{top.get('reason','')}」有关的人，正向你走来。",
                "context": {"color": color, "objects": objs}
            })
            time.sleep(1.5)
            self._send_main({
                "type": "character_found",
                "message": "找到了。",
                "character_name_hidden": True
            })
            time.sleep(1.5)

            # ── LLM 独白 + 叙事（DeepSeek，失败 fallback 模板）──
            ch_name = top.get('name', '')
            ch_title = top.get('title', '')
            llm_performance = None
            llm_narrative = None
            try:
                import requests as _req
                def _ds(prompt, max_tok=400):
                    r = _req.post(DEEPSEEK_BASE_URL,
                        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": max_tok, "temperature": 0.8},
                        timeout=15)
                    if r.status_code == 200:
                        return r.json()["choices"][0]["message"]["content"].strip()
                    raise Exception(f"DeepSeek {r.status_code}")
                # 独白
                mp = f"写一段3-5句的第一人称独白，说话者是一位匿名的湖湘先贤（不要透露名字）。你对选择了「{color}」、画下「{'、'.join(objs)}」的后来说话。提及颜色和物象，有温度有文采。输出纯独白，不要引号不要角色名。"
                llm_raw = _ds(mp, 300)
                _all_names = ["朱熹","张栻","王夫之","周敦颐","胡宏","吕祖谦","陆九渊","王阳明","曾国藩","左宗棠","黄兴","蔡锷","宋教仁","陈天华","杨昌济","何叔衡","李达","程颢","程颐","胡安国","胡林翼","彭玉麟","谭嗣同","魏源","毛泽东","成仿吾","周谷城","何长工","熊十力","冯友兰","钱基博","金岳霖","梁漱溟","胡庶华","罗洪先"]
                for n in _all_names:
                    llm_raw = llm_raw.replace(n, "…")
                llm_performance = llm_raw.split("\n")
                # 叙事
                nprompt = f"为「寻麓千年色」写一段诗意叙事。颜色：{color}，物象：{'、'.join(objs)}，回应者：{ch_name}（{ch_title}）。要求：1)4-5句 2)语言优美有古韵 3)融入颜色和物象意境 4)体现湖湘千年文脉。只输出叙事。"
                llm_narrative = _ds(nprompt, 400).split("\n")
                print(f"  [Pipeline] DeepSeek OK monologue={len(llm_performance)}lines narrative={len(llm_narrative)}lines")
            except Exception as e:
                print(f"  [Pipeline] DeepSeek FAIL: {e}")

            # ── 角色演绎 ──
            perf = llm_performance if llm_performance else [
                f"你选择了{color}——那是千年书院里，午后的光影穿过窗格落下的颜色。",
                f"你画下{'、'.join(objs)}，笔触里藏着这片土地的记忆。",
                "我不知道你是谁，但我知道你在寻找。",
                "在岳麓山下的每一块石头里，在湘江的每一道波纹里，",
                "总有一个声音在等着后来的人。",
            ]
            print(f"  [Pipeline] send character_performance ({len(perf)} lines)")
            self._send_main({
                "type": "character_performance",
                "character": "????",
                "paragraphs": perf
            })
            time.sleep(1.5)

            # ── 人物揭示 ──
            portrait_map = {
                "胡宏":"huhong","李达":"lida","陆九渊":"lujiuyuan","王夫之":"wangfuzhi",
                "杨昌济":"yangchnagji","张栻":"zhangshi","周敦颐":"zhoudunyi","朱熹":"zhuxi",
                "黄兴":"huangxing","蔡锷":"caie","曾国藩":"zenguofan","左宗棠":"zuozongtang",
                "王阳明":"wangyangming","吕祖谦":"lvzuqian","宋教仁":"songjiaoreng",
                "陈天华":"chengtianhua","何叔衡":"heshuheng",
                "程颢":"chengjing","程颐":"chenyi","胡安国":"huanguo","罗洪先":"luohongxian",
            }
            pf = portrait_map.get(ch_name, "")
            print(f"  [Pipeline] send character_revealed name={ch_name} portrait={pf}")
            self._send_main({
                "type": "character_revealed",
                "name": ch_name,
                "title": ch_title,
                "portrait": pf,
                "monologue": perf,
                "message": f"刚才与你说话的，是{ch_name}。"
            })
            time.sleep(1.5)

            # ── 叙事生成 ──
            narrative_text = llm_narrative if llm_narrative else [
                f"{color}已经展开。",
                f"{'、'.join(objs)}也已经落下。",
                f"{ch_name}的声音回荡在千年书院中。",
                "这就是你寻到的千年色。"
            ]
            gen_result = {
                "type": "generation_result",
                "title": "你寻到的千年色",
                "paragraphs": narrative_text,
                "context": {"color": color, "objects": objs, "character": ch_name}
            }
            print(f"  [Pipeline] send generation_result title={gen_result['title']} paragraphs={len(narrative_text)}")
            self._send_main(gen_result)

            # ── 明信片生成 ──
            try:
                import random as _random
                from rag.uploader import PostcardUploader
                from PIL import Image, ImageDraw, ImageFont

                all_palette = {"岳麓绿":"#496b4a","书院红":"#8d3d36","湘江蓝":"#3f7082","西迁黄":"#a9823e","校徽金":"#c3a45e","墨色":"#333936","梨黄":"#F0E440","桂黄":"#F2E700","澄蓝":"#355BFF"}
                c1_hex = all_palette.get(color, "#496b4a")
                others = [h for n, h in all_palette.items() if n != color]
                c2_hex = others[_random.randint(0, len(others)-1)] if others else "#8d3d36"
                font_lg = font_md = font_sm = None
                for fp in [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc"]:
                    if os.path.exists(fp):
                        try:
                            font_lg = ImageFont.truetype(fp, 64)
                            font_md = ImageFont.truetype(fp, 38)
                            font_sm = ImageFont.truetype(fp, 24)
                            break
                        except Exception:
                            pass
                if font_lg is None:
                    font_lg = font_md = font_sm = ImageFont.load_default()
                W, H = 1200, 1600
                M = 60
                card = Image.new("RGB", (W, H), (252, 250, 246))
                draw = ImageDraw.Draw(card)
                bw = (W - 3 * M) // 2
                draw.rectangle([M, M, M + bw, M + 280], fill=c1_hex)
                draw.rectangle([M * 2 + bw, M, M * 2 + bw * 2, M + 280], fill=c2_hex)
                draw.text((M + 16, M + 296), color, font=font_sm, fill=(100, 90, 80))
                draw.text((M, M + 380), "你筑下的景", font=font_sm, fill=(140, 130, 120))
                draw.text((M, M + 420), "、".join(objs), font=font_lg, fill=(45, 38, 30))
                draw.text((M, M + 540), "回应你的人", font=font_sm, fill=(140, 130, 120))
                draw.text((M, M + 580), f"{top.get('name','')}　{top.get('title','')}", font=font_md, fill=(65, 55, 45))
                y = M + 680
                for p in gen_result["paragraphs"][:3]:
                    draw.text((M, y), p, font=font_sm, fill=(85, 75, 65))
                    y += 50
                draw.line([M, y + 20, W - M, y + 20], fill=(190, 182, 175), width=1)
                draw.text((M, y + 38), f"湖南大学 · 寻麓千年色 · {datetime.now().strftime('%Y.%m.%d %H:%M')}", font=font_sm, fill=(150, 145, 140))
                uploader = PostcardUploader()
                result = uploader.upload(card)
                self._send_main({"type": "postcard_result", "image_url": result["image_url"],
                                 "qr_base64": result["qr_base64"], "unique_id": result["unique_id"],
                                 "message": "扫码带走你的千年色。"})
                print(f"  [Postcard] 已上传: {result['image_url']}")
            except Exception as e:
                print(f"  [Postcard] 失败: {e}")

            # ── 等待 20s 后自动重置 ──
            time.sleep(10.0)

        except Exception as e:
            logger.error(f"Character pipeline 异常: {e}")
        finally:
            # ── 状态重置 ──
            self.selected_objects.clear()
            self.sketch_trajectories.clear()
            self._pending_trajectory = []
            self.fsm._recognized_object = None
            self.fsm.reset_to_global()
            self.fsm.trigger_color_extraction_start()
            self._pipeline_running = False
            _debug_log.info("STATE_RESET | 一轮完整流程结束，重置回 COLOR_EXTRACTION")

    def _on_character_confirmed(self):
        print("  [FSM] 人物已确认!")
        self._send_main({
            "type": "character_confirmed",
            "entity": self.selected_objects[-1] if self.selected_objects else ""
        })
        self._send_gesture_state()

    def _on_reject_recommendations(self):
        print("  [FSM] 拒绝推荐 → 跳过轮盘")

    # ── 颜色提取回调（两色：框内取色 + 3s 稳定确认）─────────

    def _on_color_extraction_start(self):
        """颜色提取开始 — 等待 R 键"""
        self._color_done = False
        self._collected_colors.clear()
        self._color_round = 1
        print("  [FSM] 颜色提取开始（等待 R 键，第 1 轮）")
        self._send_main({
            "type": "color_extraction_start",
            "message": "将物品放入框内，按 R 键开始。",
        })

    def _start_color_detect(self):
        """R 键 → 启动当前轮次的框内取色"""
        r = self._color_round
        print(f"  [Color] 启动第{r}轮颜色检测")
        self.color_detector.start()
        self._send_main({
            "type": "color_detection_active",
            "round": r,
            "message": f"正在观察方框中的颜色……（第{r}/2色）",
        })

    def _check_color_stability(self):
        """每帧检查：框内颜色稳定 3s → 确认当前轮次"""
        if not self.color_detector.active or self._current_frame is None:
            return

        # 每 10 帧推送检测进度
        if self.frame_count % 10 == 0:
            sc = self.color_detector.stable_color
            el = self.color_detector.elapsed
            if sc or self.color_detector._stable_result:
                self._send_main({
                    "type": "color_detect_progress",
                    "round": self._color_round,
                    "stable_color": sc,
                    "elapsed": round(el, 2),
                    "confirm_seconds": self.color_detector.confirm_seconds,
                })

        result = self.color_detector.update(self._current_frame)
        if result is None:
            return

        # ── 当前轮次确认 ──
        color_name = result.color_name
        confidence = result.confidence
        r = self._color_round

        self._collected_colors.append(color_name)
        self.current_color = color_name

        _debug_log.info(f"COLOR_CONFIRMED | round={r} | {color_name} conf={confidence:.3f} "
                        f"hsv={result.dominant_hsv} base={result.base_family}")
        print(f"  → 第{r}色确认: {color_name} ({confidence:.3f})")

        if r == 1:
            # 第一色确认 → 通知前端，等用户按 R 开始第二色
            self._send_main({
                "type": "object_color_detected",
                "color": color_name,
                "confidence": round(confidence, 3),
                "source": "object",
                "round": 1,
                "message": f"一缕{color_name}浮出光面，像旧纸上醒来的日色。",
            })
            self._color_round = 2
        else:
            # 第二色确认 → 两色齐备，进入绘画
            c0 = self._collected_colors[0]
            self._send_main({
                "type": "clothing_color_detected",
                "color": color_name,
                "confidence": round(confidence, 3),
                "source": "clothing",
                "round": 2,
                "message": f"{c0}与{color_name}相遇，像山门灯火照见夜色。",
            })
            self._on_color_confirmed()

    def _on_object_color_detected(self):
        """R 键触发 → 启动当前轮次检测"""
        self._start_color_detect()

    def _on_clothing_fallback(self):
        """兜底：全屏 V2 单帧检测，失败则随机"""
        print("  [FSM] 兜底取色（全屏）")

        if self._current_frame is None:
            self._send_main({"type": "clothing_color_failed", "message": "无法读取画面。"})
            self.fsm.trigger_clothing_color_detected(False)
            return

        try:
            from vision.color_detector import ObjectColorDetector
            v2 = ObjectColorDetector()
            v2.DEFAULT_ROI = (0.15, 0.15, 0.85, 0.85)
            v2.MIN_CLUSTER_RATIO = 0.08
            v2.MIN_CONFIDENCE = 0.22

            region = v2.detect_dominant_region(self._current_frame)
            if region is None:
                region = v2.get_default_region(self._current_frame)
            result = v2.detect(self._current_frame, region=region)

            if result is not None:
                color_name = result.color_name
                confidence = result.confidence
                print(f"  → 兜底(全屏): {color_name} ({confidence:.3f})")
            else:
                import random as _r
                _fb = ["朱红","灯橙","梨黄","叶绿","瓷青","海蓝","烟紫","枫红","暖橙","藤黄",
                       "玉绿","石青","澄蓝","影紫","桃红","夕橙","桂黄","茶绿","湖青","沧蓝","黛紫"]
                color_name = _r.choice(_fb)
                confidence = 0.3
                print(f"  → 兜底 V2 失败，随机: {color_name}")

            self._collected_colors.append(color_name)
            self.current_color = color_name
            self._send_main({
                "type": "object_color_detected",
                "color": color_name,
                "confidence": round(confidence, 3),
                "source": "clothing",
                "message": f"我捕捉到了一抹{color_name}。",
            })
            self._on_color_confirmed()
        except Exception as e:
            print(f"  → 兜底异常: {e}")
            self._send_main({"type": "clothing_color_failed", "message": "颜色检测失败。"})
            self.fsm.trigger_clothing_color_detected(False)

    def _on_color_confirmed(self):
        """颜色确认 → 允许绘画"""
        self._color_done = True
        color = self.current_color or "岳麓绿"
        _debug_log.info(f"COLOR_CONFIRMED | color={color}")
        print(f"  [FSM] 颜色已确认: {color}")
        self._send_main({
            "type": "color_confirmed",
            "color": color,
            "message": f"我看到了……你的颜色是{color}。",
        })
        if self.fsm and self.fsm.mode.value == "GLOBAL":
            return

    def _run_camera_loop(self):
        from vision.hand_tracker import HAND_CONNECTIONS
        ts = 0
        _last_log_frame = 0  # 帧日志限频

        # 鼠标回调: 标定点选取
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and self.hand_tracker:
                if self.hand_tracker.calibration_collecting:
                    done = self.hand_tracker.add_calibration_point(x, y)
                    if done:
                        print("[标定] 完成！4点已选取，透视变换已启用")
        if not self.no_display:
            cv2.namedWindow("Hand + Gesture")
            cv2.setMouseCallback("Hand + Gesture", on_mouse)

        while self.is_running:
            # ── 排空摄像头缓冲，只取最新帧 ──
            frame = None
            for _ in range(4):
                ret, f = self.camera.read()
                if ret and f is not None:
                    frame = f

            if frame is None:
                time.sleep(0.005)
                continue

            self.frame_count += 1
            h, w = frame.shape[:2]
            self._current_frame = frame
            self._check_color_stability()
            display = frame.copy()  # 始终初始化，后续标定/可视化复用

            if not self.no_display:
                self.color_detector.draw_roi(display)

            # 手部检测
            results = self.hand_tracker._detect(frame, ts)

            if results.hand_landmarks:
                hand_lm = results.hand_landmarks[0]
                self.fsm.process(hand_lm, ts)

                # 每15帧发送手势状态，确保前端同步
                if self.main_client and self.frame_count % 3 == 0:
                    self._send_gesture_state()

                # ── 帧级调试日志（每30帧写一次，避免刷盘）──
                if self.frame_count - _last_log_frame >= 30:
                    _last_log_frame = self.frame_count
                    tips = hand_lm
                    get_y = lambda idx: tips[idx].y if hasattr(tips[idx], 'y') else tips[idx][1]
                    _debug_log.info(
                        f"FRAME#{self.frame_count} | hand=YES | "
                        f"FSM={self.fsm.mode.value}/{self.fsm.sub_state} | "
                        f"gesture={self.fsm.current_gesture.value} | "
                        f"tips_y=[{get_y(8):.3f},{get_y(12):.3f},{get_y(16):.3f},{get_y(20):.3f}] | "
                        f"mcp_y=[{get_y(6):.3f},{get_y(10):.3f},{get_y(14):.3f},{get_y(18):.3f}] | "
                        f"tray={len(self.fsm.trajectory)} | "
                        f"color={self.current_color} | "
                        f"ws={'OK' if self.main_client else '--'}"
                    )

                # → 前端 手部数据（从已检测结果直接计算像素坐标）
                pixel_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]

                # ── 透视标定：将相机坐标映射到投影/屏幕坐标 ──
                if self.hand_tracker.is_calibrated and self.hand_tracker.transform_matrix is not None:
                    pts = np.array(pixel_landmarks, dtype=np.float32).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(pts, self.hand_tracker.transform_matrix)
                    calib_landmarks = [(int(p[0][0]), int(p[0][1])) for p in transformed]
                    calib_fingertips = [calib_landmarks[i] for i in (4, 8, 12, 16, 20)]
                else:
                    # 未标定：缩放到输出尺寸
                    calib_landmarks = [(int(x * self.hand_tracker.output_size[0] / w),
                                       int(y * self.hand_tracker.output_size[1] / h))
                                      for (x, y) in pixel_landmarks]
                    calib_fingertips = [calib_landmarks[i] for i in (4, 8, 12, 16, 20)]

                palm_x = sum(p[0] for p in calib_landmarks) // 21
                palm_y = sum(p[1] for p in calib_landmarks) // 21
                wrist = calib_landmarks[0]
                fingertips = calib_fingertips

                if self.hand_client:
                    ow, oh = self.hand_tracker.output_size
                    lm_flat = []
                    for x, y in calib_landmarks:
                        lm_flat.extend([x / max(ow, 1), y / max(oh, 1)])
                    ft_flat = []
                    for x, y in calib_fingertips:
                        ft_flat.extend([x / max(ow, 1), y / max(oh, 1)])
                    self._send_hand({
                        "type": "hand_tracking",
                        "palm_center": [palm_x / max(ow, 1), palm_y / max(oh, 1)],
                        "wrist": [wrist[0] / max(ow, 1), wrist[1] / max(oh, 1)],
                        "landmarks": lm_flat,
                        "fingertips": ft_flat,
                    })

                # ── Cover Flow: 手首次出现在识别区 ──
                if not self._prev_hand_detected and not self._hand_appeared_sent:
                    gesture_name = self.fsm.current_gesture.value if self.fsm.current_gesture else "open_hand"
                    ow, oh = self.hand_tracker.output_size
                    self._send_main({
                        "type": "hand_appeared",
                        "palm_center": [palm_x / max(ow, 1), palm_y / max(oh, 1)],
                        "gesture": gesture_name,
                    })
                    self._hand_appeared_sent = True
                    print(f"  [COVER_FLOW] hand_appeared sent! palm=({palm_x}, {palm_y})")

                self._prev_hand_detected = True

                # ── 手势绘画：FSM 检测手势 → 食指画线 / 握拳确认 ──
                if self.main_client and self._color_done:
                    if self.fsm.is_drawing:
                        # FSM 检测到食指伸出 → DRAWING.TRACKING → 发送画点
                        index_tip = fingertips[1]
                        ow, oh = self.hand_tracker.output_size
                        self._send_main({"type": "drawing_point",
                                         "x": 1.0 - index_tip[0] / max(ow, 1),
                                         "y": index_tip[1] / max(oh, 1)})
                self._hand_lost_frames = 0

                # ── 可视化 ──
                if not self.no_display:
                    for i, lm in enumerate(hand_lm):
                        px, py = int(lm.x * w), int(lm.y * h)
                        clr = (0, 255, 0) if i % 4 == 0 else (0, 200, 0)
                        cv2.circle(display, (px, py), 4, clr, -1)
                    for a, b in HAND_CONNECTIONS:
                        pt1 = (int(hand_lm[a].x * w), int(hand_lm[a].y * h))
                        pt2 = (int(hand_lm[b].x * w), int(hand_lm[b].y * h))
                        cv2.line(display, pt1, pt2, (0, 255, 0), 1)
                    # 绘制轨迹
                    if self.fsm.is_drawing and len(self.fsm.trajectory) >= 2:
                        pts = [(int(p[0] * w), int(p[1] * h)) for p in self.fsm.trajectory]
                        for i in range(len(pts)-1):
                            cv2.line(display, pts[i], pts[i+1], (0, 255, 255), 2)
                    # 状态文字
                    gesture = self.fsm.current_gesture.value if self.fsm.current_gesture else "?"
                    cv2.putText(display, f"Gesture: {gesture}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display, f"Mode: {self.fsm.mode.value} | {self.fsm.sub_state}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 100), 2)
                    pts_str = f"Traj: {len(self.fsm.trajectory)}"
                    cv2.putText(display, pts_str, (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                    status = "前端: OK" if self.main_client else "前端: --"
                    s_color = (0, 255, 0) if self.main_client else (0, 0, 255)
                    cv2.putText(display, status, (10, h - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 2)
            else:
                self.fsm.process(None, ts)
                if self.fsm.is_drawing:
                    self._hand_lost_frames += 1
                    if self._hand_lost_frames >= 30:
                        if len(self.fsm.trajectory) >= 10:
                            print(f"  [Auto-Commit] 手离开1s，轨迹={len(self.fsm.trajectory)}")
                            traj = list(self.fsm.trajectory)
                            self._on_drawing_commit(traj)
                            self.fsm.trajectory.clear()
                            self.fsm._transition_to(GestureMode.GLOBAL, "IDLE")
                        else:
                            print(f"  [Auto-Cancel] 轨迹不足 ({len(self.fsm.trajectory)})")
                            self.fsm.trajectory.clear()
                            self.fsm._transition_to(GestureMode.GLOBAL, "IDLE")
                            self._send_main({"type": "drawing_cancelled",
                                             "message": "没关系。有些图像，需要再画一次才会清晰。"})
                        self._hand_lost_frames = 0
                if self.frame_count - _last_log_frame >= 60:
                    _last_log_frame = self.frame_count
                if self._prev_hand_detected:
                    self._prev_hand_detected = False
                    self._hand_appeared_sent = False
                    print("  [COVER_FLOW] hand_disappeared")
                # ── Cover Flow: 手消失在识别区 ──
            if self.hand_tracker:
                display = self.hand_tracker.draw_calibration_overlay(display)

            if not self.no_display:
                cv2.imshow("Hand + Gesture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c') or key == ord('C'):
                # 开始/重置标定
                self.hand_tracker.start_calibration()
                print("[标定] 请依次点击投影画面的 左上→右上→右下→左下")
            elif key == ord('r') or key == ord('R'):
                # 手动触发颜色检测
                print(f"\n  [按键] R → 触发颜色检测 (已收集: {len(self._collected_colors)}/2)")
                if self.fsm and self.fsm.mode == GestureMode.COLOR_EXTRACTION:
                    self._on_object_color_detected()
            elif ord('1') <= key <= ord('6'):
                colors = ["朱红","灯橙","梨黄","叶绿","瓷青","海蓝","烟紫","枫红","暖橙","藤黄","玉绿","石青","澄蓝","影紫","桃红","夕橙","桂黄","茶绿","湖青","沧蓝","黛紫"]
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
                  f"前端={'OK' if self.main_client else '--'}  "
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
                colors = ["朱红","灯橙","梨黄","叶绿","瓷青","海蓝","烟紫","枫红","暖橙","藤黄","玉绿","石青","澄蓝","影紫","桃红","夕橙","桂黄","茶绿","湖青","沧蓝","黛紫"]
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
        if self.webcam:
            try:
                self.webcam.release()
            except Exception:
                pass
        if self.hand_tracker:
            try:
                self.hand_tracker.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("\n[退出] 集成服务器已停止")


# ── 直接桥接（不经过 前端Sender，直接回调到 IntegratedServer）─

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
                # 识别为空 → 从全部 48 物象中随机兜底
                return self._random_fallback()
            return [
                {"name": r.entity_name, "score": round(r.score, 4),
                 "qd_category": r.qd_category}
                for r in results[:3]
            ]
        except Exception as e:
            logger.error(f"SketchBridge 识别失败: {e}")
            return self._random_fallback()

    def _random_fallback(self) -> List[Dict]:
        """从 48 物象中随机选 3 个作为兜底"""
        import random as _r
        all_objects = [
            "东方红广场","中国书院博物馆","书卷","书架","书案","匾额",
            "古树","古籍","图书馆","墨锭","学位帽","实验室","屋脊","山石",
            "岳麓书院","岳麓山","操场","教学楼","显微镜","林荫道",
            "校徽","校门","楹联","毛笔","湖南大学大礼堂","湘江","爱晚亭",
            "牌楼路","白鹤泉","石桥","石阶","砚台","碑刻","窗格",
            "竹林","竹简","笔记本","线装书","经卷","自卑亭",
            "荣誉证书","讲堂","设计院楼","赫曦台","长廊","院墙",
            "麓山南路","黑板",
        ]
        picked = _r.sample(all_objects, min(3, len(all_objects)))
        qd_map = {"古树":"tree","书卷":"book","石阶":"stairs","岳麓书院":"house",
                  "湘江":"river","爱晚亭":"castle","石桥":"bridge","竹林":"bush",
                  "林荫道":"tree","讲堂":"church","图书馆":"house","实验室":"computer",
                  "岳麓山":"mountain","白鹤泉":"pond","校门":"door","院墙":"fence",
                  "长廊":"fence","屋脊":"umbrella","窗格":"hexagon","碑刻":"diamond",
                  "匾额":"face","校徽":"circle","东方红广场":"square","学位帽":"hat",
                  "设计院楼":"house","教学楼":"house","赫曦台":"castle",
                  "中国书院博物馆":"house","自卑亭":"house","操场":"baseball",
                  "山石":"mountain","墨锭":"coffee cup","砚台":"cup","毛笔":"pencil",
                  "笔记本":"pencil","书架":"backpack","书案":"basket","古籍":"book",
                  "线装书":"book","经卷":"book","竹简":"book","显微镜":"binoculars",
                  "楹联":"door","荣誉证书":"envelope","牌楼路":"stairs",
                  "麓山南路":"stairs","湖南大学大礼堂":"church","黑板":"television"}
        return [
            {"name": n, "score": round(0.7 - i * 0.2, 2),
             "qd_category": qd_map.get(n, "tree")}
            for i, n in enumerate(picked)
        ]


class _DirectCharacterBridge:
    """人物推荐桥接 — 直接发送到集成服务器"""

    def __init__(self, recommender, server: IntegratedServer):
        self.recommender = recommender
        self.server = server

    def recommend(self, color: str, objects: List[str]) -> List[Dict]:
        try:
            import random as _r
            # 只推荐有头像的人物
            _with_portrait = {"胡宏","李达","陆九渊","王夫之","杨昌济","张栻","周敦颐","朱熹",
                              "黄兴","蔡锷","曾国藩","左宗棠","王阳明","吕祖谦","宋教仁",
                              "陈天华","何叔衡","程颢","程颐","胡安国","罗洪先"}
            results = self.recommender.recommend(
                color=color, objects=objects,
                selected_characters=[], use_llm=False, top_k=8
            )
            # 过滤：只保留有头像的人物
            results = [r for r in results if r.name in _with_portrait]
            if not results:
                return []
            # 从前8中随机选3，增加多样性
            if len(results) > 3:
                results = _r.sample(results, 3)
                results.sort(key=lambda r: -r.score)
            candidates = [
                {"name": r.name, "title": r.title,
                 "score": round(r.score, 4), "reason": r.reason,
                 "monologue": r.monologue, "spiritLine": r.spiritLine}
                for r in results
            ]
            self.server._send_main({
                "type": "character_candidates",
                "candidates": candidates,
            })
            print(f"  → 人物推荐: {[(c['name'], round(c['score'],2)) for c in candidates]}")
            return candidates
        except Exception as e:
            logger.error(f"CharacterBridge 推荐失败: {e}")
            return []


# ── 入口 ──────────────────────────────────────────────────

def main():
    no_display = "--no-display" in sys.argv
    no_camera = "--no-camera" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    camera_url = args[0] if args else ""
    server = IntegratedServer(camera_url=camera_url, no_display=no_display, no_camera=no_camera)
    server.start()


if __name__ == "__main__":
    main()
