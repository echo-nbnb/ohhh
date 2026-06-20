#!/usr/bin/env python3
"""
Pure backend test — imports real components, simulates hand gestures,
captures all messages. No camera, no Unity, no TCP.

Usage:
    python test_backend_standalone.py
"""

import sys, io, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

from vision.gesture_state_machine import (
    GestureStateMachine, GestureMode, GestureType,
    DrawingSubState, CandidateSubState, CharRecommendSubState,
    ColorExtractionSubState
)
from vision.sketch_recognizer import create_sketch_recognizer
from rag.character_recommend import CharacterRecommender

G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; C = "\033[96m"; B = "\033[1m"; Z = "\033[0m"
PASS = "[PASS]"; FAIL = "[FAIL]"

# ============================================================
# Fake MediaPipe landmark helpers
# ============================================================
class FakeLM:
    def __init__(self, x, y, z=0):
        self.x = x; self.y = y; self.z = z

def make_open_hand():
    """五指张开：指尖 y < MCP y（MediaPipe y轴向下，指尖在上）"""
    lm = [FakeLM(0.5, 0.5)] * 21
    for tip_idx, mcp_idx in [(4,3),(8,6),(12,10),(16,14),(20,18)]:
        lm[tip_idx] = FakeLM(0.5, 0.15)  # 指尖高
        lm[mcp_idx] = FakeLM(0.5, 0.45)  # MCP低
    return lm

def make_fist():
    """握拳：指尖 y > MCP y（指尖在下方，弯曲）"""
    lm = [FakeLM(0.5, 0.5)] * 21
    for tip_idx, mcp_idx in [(4,3),(8,6),(12,10),(16,14),(20,18)]:
        lm[tip_idx] = FakeLM(0.5, 0.85)  # 指尖低
        lm[mcp_idx] = FakeLM(0.5, 0.45)  # MCP高
    return lm

def make_index_point():
    """仅食指伸出"""
    # 先全部握拳
    lm = make_fist()
    # 食指：tip在上，MCP在下
    lm[8] = FakeLM(0.5, 0.1)   # 食指尖高 (伸展)
    lm[6] = FakeLM(0.5, 0.4)   # 食指 MCP
    return lm


# ============================================================
# Message capturer
# ============================================================
class MessageCapture:
    """Capture all messages the backend would send to Unity."""
    def __init__(self):
        self.messages = []
        self.hand_messages = []

    def send(self, data: dict):
        self.messages.append(data)

    def send_hand(self, data: dict):
        self.hand_messages.append(data)

    def flush(self):
        msgs = list(self.messages)
        self.messages.clear()
        return msgs

    def flush_hand(self):
        msgs = list(self.hand_messages)
        self.hand_messages.clear()
        return msgs


# ============================================================
# Test runner
# ============================================================
class BackendTester:
    def __init__(self):
        self.fsm = None
        self.capture = MessageCapture()
        self.recognizer = None
        self.recommender = None
        self.current_color = ""
        self.selected_objects = []
        self.sketch_trajectories = {}
        self._pending_trajectory = []
        self.ok = 0
        self.ng = 0

    def setup(self):
        print(f"{C}{'='*60}{Z}")
        print(f"{C}  寻麓千年色 — 纯后端测试{Z}")
        print(f"{C}{'='*60}{Z}\n")

        # 1. Sketch recognizer
        print("[1] 初始化草图识别器...")
        self.recognizer = create_sketch_recognizer()
        print(f"{G}  [OK]{Z} SketchRecognizer (启发式模式)")

        # 2. Character recommender
        print("[2] 初始化人物推荐器...")
        self.recommender = CharacterRecommender()
        self.recommender._ensure_kb()
        print(f"{G}  [OK]{Z} CharacterRecommender ({len(self.recommender._char_index)} 人物)")

        # 3. Gesture FSM (测试用 debounce=1)
        print("[3] 初始化手势状态机...")
        self.fsm = GestureStateMachine(debounce_frames=1)
        self._bind_fsm_callbacks()
        # Start in color extraction mode
        self.fsm.trigger_color_extraction_start()
        print(f"{G}  [OK]{Z} GestureFSM ({self.fsm.mode.value}/{self.fsm.sub_state})\n")

    def _bind_fsm_callbacks(self):
        f = self.fsm
        f.on_mode_change = lambda m, s, g: self.capture.send({
            "type": "gesture_state", "mode": m, "sub_state": s, "gesture": g
        })
        f.on_color_extraction_start = lambda: self.capture.send({
            "type": "color_extraction_start",
            "message": "我来提取你的底色"
        })

        def on_object_detected():
            self.capture.send({
                "type": "object_color_detected",
                "color": "岳麓绿", "confidence": 0.92, "source": "object",
                "message": "我捕捉到了一抹岳麓绿。它象征生长、根脉与传承。",
                "interpretation": "这是根脉的颜色。它来自树，也来自仍在生长的你。"
            })
        f.on_object_color_detected = on_object_detected

        f.on_clothing_fallback = lambda: self.capture.send({
            "type": "clothing_fallback",
            "message": "那我看看今天的你。"
        })

        f.on_color_confirmed = lambda: self.capture.send({
            "type": "color_confirmed", "color": self.current_color or "岳麓绿",
            "message": f"我看到了……你的颜色是{self.current_color or '岳麓绿'}。"
        })

        f.on_drawing_start = lambda: self.capture.send({
            "type": "drawing_start", "message": "伸出食指，开始作画。"
        })

        f.on_drawing_commit = self._on_drawing_commit
        f.on_drawing_cancel = lambda: self.capture.send({
            "type": "drawing_cancelled", "message": "没关系。有些图像，需要再画一次才会清晰。"
        })

        def on_object_confirmed(name, score, qd_cat):
            self.selected_objects.append(name)
            self._pending_trajectory and self.sketch_trajectories.update(
                {name: self._pending_trajectory})
            self._pending_trajectory = []
            self.capture.send({
                "type": "object_confirmed", "object": name, "score": score,
                "objects_so_far": list(self.selected_objects),
                "message": f"一个意象已经落下。" if len(self.selected_objects) == 1
                           else f"又一个意象落下。{'、'.join(self.selected_objects)}，它们在一起了。",
                "can_continue": True
            })
            # Trigger character recommend
            self._do_character_recommend()
        f.on_object_confirmed = on_object_confirmed

        f.on_character_confirmed = lambda: self.capture.send({
            "type": "character_confirmed", "entity": "王夫之"
        })
        f.on_reject_recommendations = lambda: self.capture.send({
            "type": "character_rejected", "message": "进入轮盘浏览"
        })

    def _on_drawing_commit(self, trajectory):
        self._pending_trajectory = trajectory
        if self.recognizer and len(trajectory) >= 5:
            pts = [(x, y) for x, y, _ in trajectory]
            results = self.recognizer.recognize(pts, color=self.current_color or "岳麓绿")
            if results:
                top = results[0]
                self.fsm._recognized_object = (top.entity_name, top.score, top.qd_category)
                self.capture.send({
                    "type": "object_recognized",
                    "color": self.current_color or "岳麓绿",
                    "object": {"name": top.entity_name, "score": round(top.score, 4),
                               "qd_category": top.qd_category},
                    "narration": f"你画下了{top.entity_name}。"
                })
                return
        # No result
        self.fsm._recognized_object = None
        self._pending_trajectory = []
        self.capture.send({
            "type": "object_unrecognized",
            "message": "这道线还未凝成意象。试着画得更大一些。"
        })

    def _do_character_recommend(self):
        if not self.recommender:
            return
        results = self.recommender.recommend(
            color=self.current_color or "岳麓绿",
            objects=self.selected_objects,
            use_llm=False, top_k=3
        )
        if results:
            self.capture.send({
                "type": "character_candidates",
                "candidates": [
                    {"name": r.name, "title": r.title,
                     "score": round(r.score, 4), "reason": r.reason}
                    for r in results
                ]
            })
            # Auto-confirm top-1, trigger performance
            top = results[0]
            self.capture.send({
                "type": "character_search_start",
                "message": f"你的{self.current_color or '颜色'}指向{'、'.join(self.selected_objects)}。"
                           f"因此，一位与「{top.reason}」有关的人，正向你走来。",
                "context": {"color": self.current_color, "objects": list(self.selected_objects)}
            })
            self.capture.send({
                "type": "character_performance",
                "character": "????",
                "paragraphs": [
                    f"你选择了{self.current_color or '这些颜色'}。",
                    f"我知道你画下的{'、'.join(self.selected_objects)}。",
                    "后来者，千年文脉在此刻与你相遇。"
                ]
            })
            self.capture.send({
                "type": "character_revealed",
                "name": top.name, "title": top.title,
                "message": f"刚才与你说话的，是{top.name}。"
            })
            # Auto-trigger generation
            self.capture.send({
                "type": "generation_result",
                "title": "你寻到的千年色",
                "paragraphs": [
                    f"{self.current_color or '颜色'}已经展开。",
                    f"{'、'.join(self.selected_objects)}也已经落下。",
                    f"{top.name}的声音回荡在千年书院中。",
                    "这就是你寻到的千年色。"
                ],
                "context": {
                    "color": self.current_color,
                    "objects": list(self.selected_objects),
                    "character": top.name
                }
            })

    # ================================================================
    # Simulate gestures
    # ================================================================
    def gesture(self, landmarks, ts, wait=0.05):
        """Feed landmarks to FSM and return captured messages."""
        self.fsm.process(landmarks, ts)
        time.sleep(wait)
        return self.capture.flush()

    # ================================================================
    # Assertions
    # ================================================================
    def step(self, title):
        print(f"\n{B}{Y}-- {title} --{Z}")

    def chk(self, msgs, expected_type, desc=""):
        types = [m.get("type","") for m in msgs]
        if expected_type in types:
            self.ok += 1; print(f"  {G}{PASS}{Z} {desc}")
            return next(m for m in msgs if m.get("type")==expected_type)
        else:
            self.ng += 1; print(f"  {R}{FAIL}{Z} {desc}  expected={expected_type} got={types}")
            return None

    def chk_ok(self, c, desc):
        if c: self.ok += 1; print(f"  {G}{PASS}{Z} {desc}")
        else: self.ng += 1; print(f"  {R}{FAIL}{Z} {desc}")

    def show(self, msgs):
        for m in msgs:
            t = m.get('type','?')
            d = ''
            if t=='gesture_state': d = f"mode={m.get('mode','')} sub={m.get('sub_state','')}"
            elif t=='object_recognized':
                o = m.get('object',{}); d = f"name={o.get('name','')} score={o.get('score','')}"
            elif t=='object_confirmed':
                d = f"obj={m.get('object','')} total={m.get('objects_so_far','')}"
            elif t=='character_candidates':
                cs = m.get('candidates',[]); d = f"[{', '.join(c.get('name','') for c in cs)}]"
            elif t=='character_performance':
                ps = m.get('paragraphs',[]); d = f"by={m.get('character','')} {len(ps)} paras"
            elif t=='character_revealed': d = f"{m.get('name','')} ({m.get('title','')})"
            elif t=='generation_result':
                ps = m.get('paragraphs',[]); d = f"{len(ps)} paras"
            else: d = json.dumps(m, ensure_ascii=False)[:80]
            print(f"  -> {t}  {d}")

    def summary(self):
        t = self.ok + self.ng
        print(f"\n{C}{'='*60}{Z}")
        print(f"{B}Result: {self.ok}/{t} passed{Z}")
        if self.ng: print(f"{R}       {self.ng} failed{Z}")
        else: print(f"{G}       ALL PASSED{Z}")
        print(f"{C}{'='*60}{Z}")

    # ================================================================
    def run(self):
        self.setup()
        ts = 0

        # 手势切换：先喂 None（手移开），再喂目标手势，确保 gesture_changed=True
        def switch_to(target_lm):
            nonlocal ts
            # 先移开手 (None → 重置 prev_gesture)
            self.fsm.process(None, ts); ts+=100
            self.capture.flush()
            # 再喂目标手势
            self.fsm.process(target_lm(), ts); ts+=100
            time.sleep(0.02)
            return self.capture.flush()

        # ================================================================
        # STAGE 2: 寻色
        # ================================================================
        self.step("寻色-开始: 握拳 → 开始物件分析")
        msgs = switch_to(make_fist); self.show(msgs)
        # FSM should: AWAITING → OBJECT_ANALYZING, fire on_object_color_detected
        self.chk(msgs, "gesture_state", "FSM → OBJECT_ANALYZING")
        self.chk(msgs, "object_color_detected", "object_color_detected (auto callback)")
        print(f"  FSM: mode={self.fsm.mode.value} sub={self.fsm.sub_state}")

        self.step("寻色-检测完成: 自动推进到 CONFIRMING")
        self.fsm.trigger_object_color_detected(True)
        time.sleep(0.02)
        msgs = self.capture.flush(); self.show(msgs)
        self.chk(msgs, "gesture_state", "FSM → OBJECT_CONFIRMING")
        self.chk_ok(self.fsm.sub_state == "object_confirming",
                     f"sub=object_confirming (actual: {self.fsm.sub_state})")

        self.step("寻色-确认: 握拳确认颜色 → GLOBAL")
        self.current_color = "岳麓绿"
        msgs = switch_to(make_fist); self.show(msgs)
        self.chk(msgs, "color_confirmed", "color_confirmed")
        self.chk(msgs, "gesture_state", "FSM → GLOBAL")
        self.chk_ok(self.fsm.mode == GestureMode.GLOBAL,
                     f"mode=GLOBAL (actual: {self.fsm.mode.value})")

        # ================================================================
        # STAGE 3: 造象
        # ================================================================
        self.step("造象-绘画: 食指伸出 → DRAWING")
        msgs = switch_to(make_index_point); self.show(msgs)
        self.chk(msgs, "gesture_state", "FSM → DRAWING/TRACKING")
        self.chk_ok(self.fsm.mode == GestureMode.DRAWING and self.fsm.is_drawing,
                     f"mode=DRAWING is_drawing=True (actual: {self.fsm.mode.value})")

        # 录制绘画轨迹
        self.step("造象-绘画中: 录制指尖轨迹")
        import math
        for i in range(25):
            x = 0.3 + 0.4 * i / 25
            y = 0.15  # 指尖始终在 MCP(0.4) 上方，保持食指伸出
            lm = make_index_point()
            lm[8] = FakeLM(x, y)       # tip stays above MCP
            lm[6] = FakeLM(0.5, 0.4)   # MCP 保持不变
            self.fsm.process(lm, ts); ts += 33
        pts = self.fsm.trajectory_point_count
        print(f"  trajectory points = {pts}")
        self.chk_ok(pts >= 20, f"recorded {pts} trajectory points")

        self.step("造象-提交: 握拳 → 草图识别 → CANDIDATE")
        msgs = switch_to(make_fist); self.show(msgs)
        obj = self.chk(msgs, "object_recognized", "object_recognized (real SketchRecognizer)")
        if obj:
            o = obj.get("object", {})
            print(f"       -> 真实识别: {o.get('name','?')} score={o.get('score','?')} qd={o.get('qd_category','?')}")
        self.chk(msgs, "gesture_state", "FSM → CANDIDATE/BROWSING")
        self.chk_ok(self.fsm.mode == GestureMode.CANDIDATE,
                     f"mode=CANDIDATE (actual: {self.fsm.mode.value})")

        self.step("造象-确认: 握拳 → object_confirmed + 人物推荐 + 演绎 + 生成")
        msgs = switch_to(make_fist); self.show(msgs)
        self.chk(msgs, "object_confirmed", "object_confirmed (with objects_so_far)")
        self.chk(msgs, "character_candidates", "character_candidates (real CharacterRecommender)")
        cc = self.chk(msgs, "character_candidates", "")
        if cc:
            names = [c.get("name","?") for c in cc.get("candidates",[])]
            print(f"       -> 推荐人物: {names}")
        self.chk(msgs, "character_search_start", "character_search_start (因果解释)")
        self.chk(msgs, "character_performance", "character_performance (第一人称)")
        self.chk(msgs, "character_revealed", "character_revealed (揭示身份)")
        gen = self.chk(msgs, "generation_result", "generation_result (自动触发)")
        if gen:
            ctx = gen.get("context", {})
            self.chk_ok(len(ctx.get("objects",[])) >= 1,
                         f"context has {len(ctx.get('objects',[]))} objects")

        # ================================================================
        # Edge: 取消重画
        # ================================================================
        self.step("异常路径: 张手取消物象 → 回到 DRAWING")
        self.fsm.reset_to_global()
        self.capture.flush()
        switch_to(make_index_point)  # → DRAWING
        self.capture.flush()
        for i in range(20):
            lm = make_index_point(); lm[8] = FakeLM(0.3+0.4*i/20, 0.15)
            self.fsm.process(lm, ts); ts+=33
        switch_to(make_fist)          # → CANDIDATE
        self.capture.flush()
        msgs = switch_to(make_open_hand); self.show(msgs)
        self.chk(msgs, "gesture_state", "FSM after cancel")
        self.chk_ok(self.fsm.mode == GestureMode.DRAWING,
                     f"back to DRAWING (actual: {self.fsm.mode.value})")

        self.summary()


if __name__ == "__main__":
    BackendTester().run()
