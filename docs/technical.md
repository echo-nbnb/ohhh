# 技术架构文档

## 系统拓扑

```
摄像头 (USB/IP)
    │
    ▼
Python test_integrated.py
    ├── MediaPipe 手部追踪 (21 landmarks, ~30fps)
    ├── 手势 FSM (6 模式状态机, 1 帧防抖)
    ├── HSV 颜色检测 (六色匹配)
    ├── QuickDraw 草图识别 (启发式/ONNX)
    ├── RAG 人物推荐 (35 人物, 颜色+物象匹配)
    └── TCP Server
        ├── :8888 主通道 (叙事消息)
        └── :8889 手部通道 (手部数据流)
            │
            ▼
Python ws_server.py (WebSocket :8080)
    │ 桥接浏览器 ↔ TCP 后端
    │ 接收缓冲区防丢包, 断线自动重连
    ▼
React Frontend (Vite :5173)
    ├── Act0-5 组件
    ├── WebGL LiquidChrome (OGL)
    └── WebSocket 通信
```

## Python 后端

### 视觉模块 (`vision/`)

| 文件 | 功能 |
|------|------|
| `hand_tracker.py` | MediaPipe HandLandmarker 封装, 4点透视标定, 坐标平滑 |
| `gesture_state_machine.py` | 6 模式 FSM: COLOR_EXTRACTION → GLOBAL → DRAWING → CANDIDATE → CHAR_RECOMMEND → CHAR_WHEEL |
| `color_detector.py` | ObjectColorDetector: HSV 直方图峰值 → 六色匹配, 白平衡预处理 |
| `webcam_color_detector.py` | WebcamColorDetector: 衣物颜色兜底检测 |
| `sketch_recognizer.py` | QuickDraw 345 类 → 88 物象映射, 28×28 栅格化, 启发式/ONNX 推理, 颜色加权 |
| `ipcamera.py` | IP 摄像头 RTSP/HTTP 流封装 |

### RAG 模块 (`rag/`)

| 文件 | 功能 |
|------|------|
| `character_recommend.py` | 35 人物推荐: 参考表命中(0.50) + 同组加权(0.20) + 关键词匹配(0.25) + 基础分(0.05) |
| `generator.py` | 阿里云百炼 DashScope 调用: Qwen Turbo(实时) / Qwen3 Max(叙事) / Wan 2.7(图像) |
| `retriever.py` | 知识库检索: 208 实体 + 107 组合 + 100 模板 |
| `oss_config.py` | 阿里云 OSS 配置 (bucket: ohhhhhh) |

### 消息协议

TCP/WebSocket 使用 JSON + `\n` 分隔。主要消息类型：

| type | 方向 | 说明 |
|------|------|------|
| `hand_tracking` | backend → frontend | 手部 21 关键点 + 指尖坐标 |
| `gesture_state` | backend → frontend | FSM 当前模式/子状态/手势 |
| `color_extraction_start` | backend → frontend | 颜色提取开始 |
| `object_color_detected` | backend → frontend | 物件颜色匹配结果 |
| `clothing_color_detected` | backend → frontend | 衣物颜色匹配结果 |
| `color_confirmed` | backend → frontend | 颜色确认 |
| `drawing_point` | backend → frontend | 食指指尖坐标（绘画中） |
| `object_recognized` | backend → frontend | 物象识别结果 |
| `character_candidates` | backend → frontend | 人物推荐候选列表 |
| `character_performance` | backend → frontend | 人物第一人称台词 |
| `character_revealed` | backend → frontend | 人物身份揭示 |
| `generation_result` | backend → frontend | AI 叙事生成结果 |
| `postcard_result` | backend → frontend | 明信片 OSS URL + 二维码 |
| `gesture_simulate` | frontend → backend | TCP 手势模拟（测试用） |

### ws_server 关键修复

- **接收缓冲区**：`_recv_buf` 防止 TCP 粘包导致多消息合并丢包
- **断线自动重连**：`_poll_backend` 检测断线后自动 `connect()`

## React 前端

### 五幕组件

| 幕 | 组件 | 关键 Props |
|---|------|-----------|
| 0 | `Act0` | `onNext` |
| 1 | `Act1Entry` | `switchDelay`, `dissolveDelay` |
| 2 | `Act2ColorSeeking` | `step`, `recognizedColors`, `copyByStep` |
| 3 | `Act3FormingVision` | `primaryColor`, `secondaryColor`, `onRecognizeSketch` |
| 4 | `Act4SpiritCalling` | `primaryColor`, `secondaryColor`, `onFetchSpiritMatch` |
| 5 | `Act5Postcard` | `postcardData`, `debugStage` |

### 可移植模块

- **SpiritModule** (`src/SpiritModule/`)：1920×1080 舞台, ResizeObserver 等比缩放, `character` → 寻找→剪影→台词→显影→揭示
- **PostcardModule** (`D:/xwechat_files/.../PostcardModule/`)：已评估可接入, 待移植

### WebGL 液态效果

- **LiquidChrome** (`Act2ColorSeeking/LiquidChrome.jsx`)：单/双色 WebGL shader, OGL 渲染, 蓝黄交融+白高光+金属明暗
- **DualColorLiquidChrome** (`components/DualColorLiquidChrome/`)：Act5 专用版, 简化接口
- **LiquidChromeDual** (`components/LiquidChromeDual/`)：通用版, Act2/Act5 共用

### 状态管理

各幕通过 props 接收后端数据, 无全局状态库。`App.jsx` 使用 `useState` + `act` 状态切换五幕。

### WebSocket 通信

- `useWebSocket` hook：连接管理, 自动重连（3s 间隔）, 状态跟踪
- `backendAdapter.js`：后端消息 → 前端内部事件转换
- `messageTypes.js`：支持的消息类型集合

## 部署

### 开发环境

```bash
# Python
conda activate ohhh
python test_integrated.py    # 摄像头 + FSM
python web/ws_server.py      # WebSocket 桥接

# 前端
cd web/frontend
npm run dev                   # Vite dev server :5173
```

### 生产环境

```bash
cd web/frontend
npm run build                 # 产出 dist/
```

前端静态文件部署到 CDN/OSS, Python 服务部署到服务器。摄像头直连服务器 USB 端口。

## 性能参数

| 指标 | 值 |
|------|-----|
| 手部追踪 | MediaPipe GPU 推理, ~30fps |
| 手势防抖 | 1 帧 (~33ms) |
| WebSocket 延迟 | <10ms (localhost) |
| 颜色检测 | <5ms (HSV 直方图) |
| 草图识别 | <50ms (启发式) / <100ms (ONNX) |
| 人物推荐 | <10ms (启发式) / ~2s (LLM) |
| LiquidChrome | 60fps (WebGL GPU) |
