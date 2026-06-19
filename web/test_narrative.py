#!/usr/bin/env python3
"""Full narrative flow test — v3 with color fallback + multi-object"""
import sys, io, socket, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MAIN, HAND = 8888, 8889
PASS, FAIL = "[PASS]", "[FAIL]"
G, Y, R, C, B = "\033[92m", "\033[93m", "\033[91m", "\033[96m", "\033[1m"
Z = "\033[0m"

class T:
    def __init__(self):
        self.m = None; self.h = None; self.ok = 0; self.ng = 0; self.n = 0

    def connect(self):
        self.m = socket.socket(); self.m.settimeout(5); self.m.connect(('127.0.0.1', MAIN))
        self.h = socket.socket(); self.h.settimeout(5); self.h.connect(('127.0.0.1', HAND))
        print(f"{G}[OK]{Z} Connected\n")

    def recv(self, wait=0.3):
        time.sleep(wait); self.m.settimeout(0.5); buf = b''
        try:
            while True:
                c = self.m.recv(8192)
                if not c: break
                buf += c
        except socket.timeout: pass
        except: pass
        msgs = []
        for l in buf.decode('utf-8').strip().split('\n'):
            if l.strip():
                try: msgs.append(json.loads(l.strip()))
                except: pass
        return msgs

    def send(self, d):
        self.m.sendall((json.dumps(d, ensure_ascii=False)+'\n').encode('utf-8'))

    def gesture(self, name):
        self.send({"type":"gesture_simulate","gesture":name})
        return self.recv()

    def step(self, title):
        self.n += 1
        print(f"\n{B}{Y}-- Step {self.n}: {title} --{Z}")

    def chk(self, msgs, t, desc=""):
        types = [m.get("type","") for m in msgs]
        if t in types:
            self.ok += 1; print(f"  {G}{PASS}{Z} {desc}")
            return next(m for m in msgs if m.get("type")==t)
        else:
            self.ng += 1; print(f"  {R}{FAIL}{Z} {desc}  expected={t} got={types}")
            return None

    def okk(self, c, desc):
        if c: self.ok += 1; print(f"  {G}{PASS}{Z} {desc}")
        else: self.ng += 1; print(f"  {R}{FAIL}{Z} {desc}")

    def show(self, msgs, maxlen=80):
        for m in msgs:
            t = m.get('type','?'); d=''
            if t=='gesture_state': d=f"mode={m.get('mode','')} sub={m.get('sub_state','')}"
            elif t=='color_extraction_start': d=f'"{m.get("message","")}"'
            elif t=='object_color_detected': d=f'color={m.get("color","")} src={m.get("source","")} conf={m.get("confidence","")}'
            elif t=='object_color_failed': d=f'"{m.get("message","")}"'
            elif t=='clothing_color_detected': d=f'color={m.get("color","")} src={m.get("source","")}'
            elif t=='clothing_color_failed': d=f'"{m.get("message","")}"'
            elif t=='color_confirmed': d=f'color={m.get("color","")} src={m.get("source","")}'
            elif t=='object_recognized':
                o=m.get('object',{}); d=f'name={o.get("name","")} score={o.get("score","")}'
            elif t=='object_confirmed':
                d=f'obj={m.get("object","")} total={m.get("objects_so_far","")} can_continue={m.get("can_continue","")}'
            elif t=='objects_summary': d=f'objs={m.get("objects","")}'
            elif t=='character_search_start': d=f'"{m.get("message","")[:50]}..."'
            elif t=='character_found': d=f'"{m.get("message","")}"'
            elif t=='character_performance':
                ps=m.get('paragraphs',[]); d=f'by={m.get("character","")} {len(ps)}paras'
            elif t=='character_revealed': d=f'{m.get("name","")} ({m.get("title","")})'
            elif t=='generation_result':
                ps=m.get('paragraphs',[]); d=f'title="{m.get("title","")}" {len(ps)}paras'
            else: d=json.dumps(m,ensure_ascii=False)[:60]
            print(f'  -> {t}  {d}')

    def close(self):
        for s in [self.m,self.h]:
            if s: s.close()

    def summary(self):
        t=self.ok+self.ng
        print(f"\n{C}{'='*50}{Z}")
        print(f"{B}Result: {self.ok}/{t} passed{Z}")
        if self.ng: print(f"{R}       {self.ng} failed{Z}")
        else: print(f"{G}       ALL PASSED{Z}")
        print(f"{C}{'='*50}{Z}")


def main():
    t = T()
    t.connect()

    # ================================================================
    print(f"\n{B}{C}{'='*60}{Z}")
    print(f"{B}{C}  TEST 1: Normal color path (物件检测成功){Z}")
    print(f"{B}{C}{'='*60}{Z}")

    t.step("Color - fist to start extraction")
    msgs = t.gesture("fist"); t.show(msgs)
    t.chk(msgs, "color_extraction_start", "start msg")
    t.chk(msgs, "object_color_detected", "object detected (source=object)")
    t.chk(msgs, "gesture_state", "FSM -> CONFIRMING")

    t.step("Color - fist to confirm")
    msgs = t.gesture("fist"); t.show(msgs)
    t.chk(msgs, "color_confirmed", "color confirmed")
    gs = t.chk(msgs, "gesture_state", "FSM -> GLOBAL")
    if gs: t.okk(gs.get("mode")=="GLOBAL", "mode=GLOBAL")

    t.step("Object - index -> DRAWING -> fist -> recognize")
    t.gesture("index_pointing"); t.recv(0.2)
    msgs = t.gesture("fist"); t.show(msgs)
    t.chk(msgs, "object_recognized", "object recognized")
    t.chk(msgs, "gesture_state", "FSM -> CANDIDATE")

    t.step("Object - fist to confirm -> GLOBAL (can continue)")
    msgs = t.gesture("fist"); t.show(msgs)
    oc = t.chk(msgs, "object_confirmed", "object confirmed with continue_prompt")
    if oc: t.okk(oc.get("can_continue")==True, "can_continue=True")

    # ================================================================
    print(f"\n{B}{C}{'='*60}{Z}")
    print(f"{B}{C}  TEST 2: Multi-object (多物象){Z}")
    print(f"{B}{C}{'='*60}{Z}")

    t.step("Draw 2nd object: index -> fist -> recognize")
    t.gesture("index_pointing"); t.recv(0.2)
    msgs = t.gesture("fist"); t.show(msgs)
    obj2 = t.chk(msgs, "object_recognized", "2nd object recognized")
    t.step("Confirm 2nd object -> GLOBAL")
    msgs = t.gesture("fist"); t.show(msgs)
    oc2 = t.chk(msgs, "object_confirmed", "2nd object confirmed, still can_continue")
    if oc2: t.okk(len(oc2.get("objects_so_far",[]))==2, "2 objects accumulated")

    t.step("Finish building (握拳完成筑景) -> character + generation")
    t.gesture("index_pointing"); t.recv(0.2)
    msgs = t.gesture("fist"); t.recv(0.3)  # recognize 3rd
    msgs = t.gesture("fist"); t.recv(0.3)  # confirm 3rd -> GLOBAL
    msgs = t.gesture("fist"); t.show(msgs)  # complete!
    t.chk(msgs, "objects_summary", "objects summary (筑景完成)")
    t.chk(msgs, "character_search_start", "search start with context")
    t.chk(msgs, "character_found", "character found")
    t.chk(msgs, "character_performance", "first-person performance")
    t.chk(msgs, "character_revealed", "character revealed")
    gen = t.chk(msgs, "generation_result", "generation auto-triggered")
    if gen:
        ctx = gen.get("context",{})
        t.okk(len(ctx.get("objects",[]))==3, f"generation context has 3 objects (got {len(ctx.get('objects',[]))})")

    # ================================================================
    print(f"\n{B}{C}{'='*60}{Z}")
    print(f"{B}{C}  TEST 3: Clothing fallback path (衣物兜底){Z}")
    print(f"{B}{C}{'='*60}{Z}")

    t.step("Color - fist_obj_fail (物件失败 → 衣物检测)")
    msgs = t.gesture("fist_obj_fail"); t.show(msgs)
    t.chk(msgs, "color_extraction_start", "start msg")
    t.chk(msgs, "object_color_failed", "object detection failed msg")
    t.chk(msgs, "clothing_color_detected", "clothing detected (source=clothing)")
    t.chk(msgs, "gesture_state", "FSM -> CLOTHING_CONFIRMING")

    t.step("Color - fist to confirm clothing color")
    msgs = t.gesture("fist"); t.show(msgs)
    c = t.chk(msgs, "color_confirmed", "clothing color confirmed")
    if c: t.okk(c.get("source")=="clothing", "source=clothing")

    # ================================================================
    print(f"\n{B}{C}{'='*60}{Z}")
    print(f"{B}{C}  TEST 4: Ink fallback path (墨色兜底){Z}")
    print(f"{B}{C}{'='*60}{Z}")

    t.step("Color - fist_all_fail (物件失败 → 衣物失败 → 墨色)")
    msgs = t.gesture("fist_all_fail"); t.show(msgs)
    t.chk(msgs, "color_extraction_start", "start msg")
    t.chk(msgs, "object_color_failed", "object failed")
    t.chk(msgs, "clothing_color_failed", "clothing failed")
    c2 = t.chk(msgs, "color_confirmed", "ink color confirmed (auto)")
    if c2: t.okk(c2.get("source")=="ink" and c2.get("color")=="墨色", "source=ink color=墨色")

    # ================================================================
    print(f"\n{B}{C}{'='*60}{Z}")
    print(f"{B}{C}  TEST 5: Cancel / retry{Z}")
    print(f"{B}{C}{'='*60}{Z}")

    # Walk to CANDIDATE
    t.gesture("fist"); t.recv(0.2)       # start color
    t.gesture("fist"); t.recv(0.2)       # confirm -> GLOBAL
    t.gesture("index_pointing"); t.recv(0.2)  # -> DRAWING
    msgs = t.gesture("fist"); t.recv(0.2)    # -> CANDIDATE

    t.step("Cancel object -> back to DRAWING")
    msgs = t.gesture("open_hand"); t.show(msgs)
    gs = t.chk(msgs, "gesture_state", "FSM after cancel")
    if gs: t.okk(gs.get("mode")=="DRAWING", "back to DRAWING")

    t.close()
    t.summary()

if __name__ == "__main__":
    main()
