/**
 * 寻麓千年色 - HTML前端主脚本
 * 完整交互流程：择色 → 筑景 → 唤灵 → 成色
 */

// 配置
const WS_URL = "ws://127.0.0.1:8080";
const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 480;

// 状态
const state = {
    ws: null,
    wsConnected: false,
    cameraReady: false,
    handTracker: null,
    currentAct: 0,       // 0-3 对应第一到第四幕
    fsmMode: "COLOR_EXTRACTION",
    gesture: "open_hand",
    palmPos: { x: 0, y: 0 },
    selectedColor: null,
    selectedObject: null,
    selectedCharacter: null,
    logCount: 0,
};

// DOM元素
const els = {
    wsStatus: document.getElementById("ws-status"),
    cameraFeed: document.getElementById("camera-feed"),
    handCanvas: document.getElementById("hand-canvas"),
    gestureDisplay: document.getElementById("gesture-display"),
    currentAct: document.getElementById("current-act"),
    fsmMode: document.getElementById("fsm-mode"),
    gestureType: document.getElementById("gesture-type"),
    palmPos: document.getElementById("palm-pos"),
    colorResult: document.getElementById("color-result"),
    objectResult: document.getElementById("object-result"),
    characterResult: document.getElementById("character-result"),
    generationResult: document.getElementById("generation-result"),
    logContent: document.getElementById("log-content"),
    msgCount: document.getElementById("msg-count"),
    // 按钮
    btnStart: document.getElementById("btn-start"),
    btnReset: document.getElementById("btn-reset"),
    btnFist: document.getElementById("btn-fist"),
    btnOpen: document.getElementById("btn-open"),
    btnGenerate: document.getElementById("btn-generate"),
    btnClearLog: document.getElementById("btn-clear-log"),
    // 结果显示
    detectedColor: document.getElementById("detected-color"),
    colorConfidence: document.getElementById("color-confidence"),
    colorSource: document.getElementById("color-source"),
    detectedObject: document.getElementById("detected-object"),
    objectConfidence: document.getElementById("object-confidence"),
    objectQdcat: document.getElementById("object-qdcat"),
    detectedCharacter: document.getElementById("detected-character"),
    characterReason: document.getElementById("character-reason"),
    resultImage: document.getElementById("result-image"),
    resultNarrative: document.getElementById("result-narrative"),
};

// 初始化
async function init() {
    setupEventListeners();
    updateUI();
    await connectWebSocket();
}

// 事件监听
function setupEventListeners() {
    els.btnStart.addEventListener("click", onStart);
    els.btnReset.addEventListener("click", onReset);
    els.btnFist.addEventListener("click", () => sendGesture("fist"));
    els.btnOpen.addEventListener("click", () => sendGesture("open_hand"));
    els.btnGenerate.addEventListener("click", onGenerate);
    els.btnClearLog.addEventListener("click", () => {
        els.logContent.innerHTML = "";
        state.logCount = 0;
        updateMsgCount();
    });
}

// WebSocket连接
async function connectWebSocket() {
    updateWSStatus("connecting");

    try {
        state.ws = new WebSocket(WS_URL);

        state.ws.onopen = () => {
            state.wsConnected = true;
            updateWSStatus("connected");
            log("info", "WebSocket已连接");
        };

        state.ws.onclose = () => {
            state.wsConnected = false;
            updateWSStatus("disconnected");
            log("error", "WebSocket断开，3秒后重连...");
            setTimeout(connectWebSocket, 3000);
        };

        state.ws.onerror = (e) => {
            log("error", "WebSocket错误");
            console.error(e);
        };

        state.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleMessage(data);
            } catch (e) {
                console.error("消息解析失败:", e);
            }
        };

    } catch (e) {
        log("error", `WebSocket连接失败: ${e.message}`);
        setTimeout(connectWebSocket, 3000);
    }
}

function send(data) {
    if (state.ws && state.wsConnected) {
        state.ws.send(JSON.stringify(data));
        log("send", data.type, data);
    }
}

function updateWSStatus(status) {
    els.wsStatus.className = `connection-status ${status}`;
    els.wsStatus.textContent = status === "connected" ? "已连接" : "连接中...";
}

// 消息处理
function handleMessage(data) {
    log("recv", data.type, data);

    switch (data.type) {
        case "connected":
            log("info", "后端已就绪");
            enableControls(true);
            break;

        case "color_extraction_start":
            setAct(0);
            setFSMMode("COLOR_EXTRACTION");
            log("info", "第一幕：开始择色");
            break;

        case "object_recognized":
            if (data.object) {
                state.selectedObject = data.object;
                showObjectResult(data.object);
                setAct(1);
                setFSMMode("CANDIDATE");
                log("info", `识别为: ${data.object.name} (${data.object.score})`);
            }
            break;

        case "object_confirmed":
            log("info", "物象已确认");
            break;

        case "character_candidates":
            if (data.candidates && data.candidates.length > 0) {
                const top = data.candidates[0];
                state.selectedCharacter = top;
                showCharacterResult(top);
                setAct(2);
                setFSMMode("CHAR_RECOMMEND");
                log("info", `推荐人物: ${top.name}`);
            }
            break;

        case "character_confirmed":
            log("info", "人物已确认");
            break;

        case "generation_result":
            showGenerationResult(data);
            setAct(3);
            setFSMMode("GLOBAL");
            log("info", "生成完成");
            break;

        case "gesture_state":
            if (data.mode) {
                setFSMMode(data.mode);
                updateActFromMode(data.mode);
            }
            break;

        case "hand_tracking":
            if (data.palm_center) {
                state.palmPos = { x: data.palm_center[0], y: data.palm_center[1] };
                els.palmPos.textContent = `${Math.round(data.palm_center[0])}, ${Math.round(data.palm_center[1])}`;
            }
            if (data.landmarks) {
                drawHandLandmarks(data.landmarks);
            }
            if (data.gesture) {
                updateGestureDisplay(data.gesture);
            }
            break;

        case "hand_appeared":
            if (data.gesture) {
                updateGestureDisplay(data.gesture);
            }
            break;

        default:
            console.log("未知消息类型:", data.type);
    }
}

// UI更新
function setAct(act) {
    state.currentAct = act;
    const actNames = ["第一幕择色", "第二幕筑景", "第三幕唤灵", "第四幕成色"];
    els.currentAct.textContent = actNames[act] || "--";
    els.currentAct.className = "status-value active";

    // 更新步骤指示器
    for (let i = 1; i <= 4; i++) {
        const step = document.getElementById(`step-${i}`);
        step.classList.remove("active", "completed");
        if (i - 1 < act) {
            step.classList.add("completed");
        } else if (i - 1 === act) {
            step.classList.add("active");
        }
    }
}

function setFSMMode(mode) {
    state.fsmMode = mode;
    els.fsmMode.textContent = mode || "--";
}

function updateActFromMode(mode) {
    if (mode.includes("COLOR")) setAct(0);
    else if (mode.includes("DRAWING") || mode.includes("CANDIDATE")) setAct(1);
    else if (mode.includes("CHAR")) setAct(2);
    else if (mode.includes("GLOBAL")) setAct(3);
}

function updateGestureDisplay(gesture) {
    state.gesture = gesture;
    els.gestureType.textContent = gesture || "--";

    els.gestureDisplay.className = "gesture-indicator";
    if (gesture === "fist") {
        els.gestureDisplay.classList.add("fist");
        els.gestureDisplay.textContent = "握拳";
    } else if (gesture === "open_hand") {
        els.gestureDisplay.classList.add("open");
        els.gestureDisplay.textContent = "张开";
    } else if (gesture === "index_pointing") {
        els.gestureDisplay.classList.add("index");
        els.gestureDisplay.textContent = "食指伸出";
    } else {
        els.gestureDisplay.textContent = gesture || "未知";
    }
}

function updateUI() {
    els.fsmMode.textContent = state.fsmMode;
    els.currentAct.textContent = state.currentAct ? ["第一幕择色", "第二幕筑景", "第三幕唤灵", "第四幕成色"][state.currentAct] : "--";
}

function enableControls(enabled) {
    els.btnStart.disabled = !enabled || state.cameraReady;
    els.btnReset.disabled = !enabled;
    els.btnFist.disabled = !enabled;
    els.btnOpen.disabled = !enabled;
    els.btnGenerate.disabled = !enabled;
}

// 结果显示
function showObjectResult(obj) {
    els.objectResult.style.display = "block";
    els.detectedObject.textContent = obj.name || "--";
    els.objectConfidence.textContent = obj.score ? obj.score.toFixed(2) : "--";
    els.objectQdcat.textContent = obj.qd_category || "--";
}

function showCharacterResult(char) {
    els.characterResult.style.display = "block";
    els.detectedCharacter.textContent = char.name || "--";
    els.characterReason.textContent = char.reason || "--";
}

function showGenerationResult(data) {
    els.generationResult.style.display = "block";

    // 显示叙事文本
    if (data.paragraphs) {
        els.resultNarrative.textContent = data.paragraphs.join("\n\n");
    } else if (data.narrative) {
        els.resultNarrative.textContent = data.narrative;
    }

    // 如果有图片数据
    if (data.image_base64) {
        els.resultImage.src = data.image_base64;
    } else {
        els.resultImage.alt = "暂无图片";
    }
}

// 日志
function log(type, msg, data) {
    state.logCount++;
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;

    const time = new Date().toLocaleTimeString();
    const dataStr = data ? ` ${JSON.stringify(data).substring(0, 100)}` : "";

    entry.innerHTML = `<span class="log-time">${time}</span><span class="log-type">[${type.toUpperCase()}]</span>${msg}${dataStr}`;
    els.logContent.insertBefore(entry, els.logContent.firstChild);
    updateMsgCount();
}

function updateMsgCount() {
    els.msgCount.textContent = `${state.logCount} 条`;
}

// 按钮事件
async function onStart() {
    log("info", "开始摄像头...");
    await startCamera();
    enableControls(true);
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: CAMERA_WIDTH,
                height: CAMERA_HEIGHT,
                facingMode: "user"
            }
        });

        els.cameraFeed.srcObject = stream;
        state.cameraReady = true;
        els.btnStart.disabled = true;

        // 设置画布
        els.handCanvas.width = CAMERA_WIDTH;
        els.handCanvas.height = CAMERA_HEIGHT;

        log("info", "摄像头已启动");
        updateUI();

    } catch (e) {
        log("error", `摄像头启动失败: ${e.message}`);
        alert("无法访问摄像头，请检查权限设置");
    }
}

function onReset() {
    state.selectedColor = null;
    state.selectedObject = null;
    state.selectedCharacter = null;
    state.currentAct = 0;
    state.fsmMode = "COLOR_EXTRACTION";

    els.colorResult.style.display = "none";
    els.objectResult.style.display = "none";
    els.characterResult.style.display = "none";
    els.generationResult.style.display = "none";

    els.handCanvas.getContext("2d").clearRect(0, 0, els.handCanvas.width, els.handCanvas.height);

    setAct(0);
    setFSMMode("COLOR_EXTRACTION");
    updateGestureDisplay("open_hand");

    log("info", "已重置");
}

function sendGesture(gesture) {
    send({
        type: "gesture_simulate",
        gesture: gesture,
        timestamp: Date.now()
    });
    log("send", `模拟手势: ${gesture}`);
}

function onGenerate() {
    send({
        type: "generation_start",
        timestamp: Date.now()
    });
    log("info", "触发生成");
}

// 手部关键点绘制
function drawHandLandmarks(landmarks) {
    const ctx = els.handCanvas.getContext("2d");
    ctx.clearRect(0, 0, els.handCanvas.width, els.handCanvas.height);

    if (!landmarks || landmarks.length === 0) return;

    const h = els.handCanvas.height;
    const w = els.handCanvas.width;

    // 绘制连接线
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],         // 拇指
        [0, 5], [5, 6], [6, 7], [7, 8],         // 食指
        [0, 9], [9, 10], [10, 11], [11, 12],    // 中指
        [0, 13], [13, 14], [14, 15], [15, 16],   // 无名指
        [0, 17], [17, 18], [18, 19], [19, 20],   // 小指
        [5, 9], [9, 13], [13, 17]                // 手掌
    ];

    ctx.strokeStyle = "#27ae60";
    ctx.lineWidth = 2;

    for (const [i, j] of connections) {
        if (landmarks[i] && landmarks[j]) {
            ctx.beginPath();
            ctx.moveTo(landmarks[i].x * w, landmarks[i].y * h);
            ctx.lineTo(landmarks[j].x * w, landmarks[j].y * h);
            ctx.stroke();
        }
    }

    // 绘制关节点
    for (const lm of landmarks) {
        ctx.beginPath();
        ctx.arc(lm.x * w, lm.y * h, 4, 0, 2 * Math.PI);
        ctx.fillStyle = "#e94560";
        ctx.fill();
    }
}

// 启动
init();
