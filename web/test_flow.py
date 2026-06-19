#!/usr/bin/env python3
"""Simple step-by-step test: connect to mock backend and walk through the flow."""
import sys, io, socket, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MAIN, HAND = 8890, 8891

# Start mock backend
import subprocess
proc = subprocess.Popen(
    [sys.executable, "web/mock_backend.py", "--ports", str(MAIN), str(HAND)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT
)
time.sleep(1.5)

# But mock_backend doesn't support --ports. Let's just connect to the real mock_backend.
# Actually, let's connect to the already running one on :8888/:8889
# First check if one is running

# Actually, let's just hardcode the ports like mock_backend.py uses
MAIN, HAND = 8888, 8889

print("Connecting...")
main = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
main.settimeout(5)
main.connect(('127.0.0.1', MAIN))
print(f"  Connected main :{MAIN}")

hand = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
hand.settimeout(5)
hand.connect(('127.0.0.1', HAND))
print(f"  Connected hand :{HAND}")

def recv(sock, wait=0.3):
    time.sleep(wait)
    sock.settimeout(0.5)
    msgs = []
    try:
        while True:
            data = sock.recv(8192)
            if not data: break
            for line in data.decode('utf-8').strip().split('\n'):
                if line.strip():
                    try: msgs.append(json.loads(line.strip()))
                    except: pass
    except socket.timeout:
        pass
    return msgs

def send(sock, data):
    msg = json.dumps(data, ensure_ascii=False) + "\n"
    sock.sendall(msg.encode('utf-8'))

def show(msgs):
    for m in msgs:
        t = m.get('type','?')
        detail = ''
        if t == 'color_extraction_start':
            detail = f' msg="{m.get("message","")}"'
        elif t == 'object_color_detected':
            detail = f' color={m.get("color","")} conf={m.get("confidence","")}'
        elif t == 'color_confirmed':
            detail = f' color={m.get("color","")} msg="{m.get("message","")[:30]}"'
        elif t == 'object_recognized':
            o = m.get('object',{})
            detail = f' name={o.get("name","")} score={o.get("score","")}'
        elif t == 'character_candidates':
            cs = m.get('candidates',[])
            detail = f' [{", ".join(c.get("name","") for c in cs)}]'
        elif t == 'character_confirmed':
            detail = f' entity={m.get("entity","")}'
        elif t == 'generation_result':
            ps = m.get('paragraphs',[])
            detail = f' paras={len(ps)} title="{m.get("title","")}"'
        elif t == 'gesture_state':
            detail = f' mode={m.get("mode","")} sub={m.get("sub_state","")} gesture={m.get("gesture","")}'
        elif t == 'hand_tracking':
            detail = f' palm={m.get("palm_center","")}'
        print(f'  -> {t}{detail}')

def step(n, title):
    print(f'\n=== Step {n}: {title} ===')

# ================================================================
# Walk through the narrative flow
# ================================================================

step(1, 'Stage 1 - Enter (入境): System ready')
# drain initial messages
initial = recv(main, 0.2)
show(initial)

step(2, 'Stage 2a - Color (寻色): User clenches fist to start extraction')
send(main, {"type": "gesture_simulate", "gesture": "fist"})
msgs = recv(main, 0.4)
show(msgs)

step(3, 'Stage 2b - Color (寻色): Confirm color, enter GLOBAL')
send(main, {"type": "gesture_simulate", "gesture": "fist"})
msgs = recv(main, 0.4)
show(msgs)

step(4, 'Stage 3a - Objects (造象): Index finger -> enter DRAWING')
send(main, {"type": "gesture_simulate", "gesture": "index_pointing"})
msgs = recv(main, 0.4)
show(msgs)

step(5, 'Stage 3b - Objects (造象): Fist to submit drawing -> object_recognized')
send(main, {"type": "gesture_simulate", "gesture": "fist"})
msgs = recv(main, 0.4)
show(msgs)

step(6, 'Stage 3c - Objects (造象): Fist to confirm object -> characters')
send(main, {"type": "gesture_simulate", "gesture": "fist"})
msgs = recv(main, 0.4)
show(msgs)

step(7, 'Stage 4 - Character (唤灵): Fist to confirm character -> generation')
send(main, {"type": "gesture_simulate", "gesture": "fist"})
msgs = recv(main, 0.5)
show(msgs)

step(8, 'Stage 5 - Card (成笺): Explicit generation_start')
send(main, {"type": "generation_start"})
msgs = recv(main, 0.4)
show(msgs)

step(9, 'Edge: Cancel (open_hand to retry)')
# Restart flow
send(main, {"type": "gesture_simulate", "gesture": "fist"})  # start color
recv(main, 0.3)
send(main, {"type": "gesture_simulate", "gesture": "fist"})  # confirm -> GLOBAL
recv(main, 0.3)
send(main, {"type": "gesture_simulate", "gesture": "index_pointing"})  # -> DRAWING
recv(main, 0.3)
send(main, {"type": "gesture_simulate", "gesture": "fist"})  # -> CANDIDATE
recv(main, 0.3)
# Now cancel
send(main, {"type": "gesture_simulate", "gesture": "open_hand"})
msgs = recv(main, 0.4)
show(msgs)

step(10, 'Hand channel data')
hand_data = recv(hand, 1.0)
show(hand_data[:3])

main.close()
hand.close()
proc.terminate()
print('\nDone.')
