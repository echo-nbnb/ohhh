# 寻麓千年色

> 湖南大学设计艺术学院 · AI 驱动手势交互装置

用户通过**摄像头 + 手势**与投影画面交互——择色、筑景、唤灵、成笺，五幕沉浸式湖湘文化叙事体验。

---

## 技术栈

| 层 | 技术 |
|---|------|
| 前端 | React 18 + Vite + Tailwind CSS + WebGL (OGL) |
| 后端桥接 | Python WebSocket + TCP |
| 视觉 | MediaPipe 手部追踪 + HSV 颜色检测 + QuickDraw 草图识别 |
| AI | 阿里云百炼 DashScope (Qwen / Wan 2.7) |
| 存储 | 阿里云 OSS |

---

## 项目结构

```
ohhh/
├── vision/                          # 计算机视觉
│   ├── hand_tracker.py              # MediaPipe 手部 21 关键点追踪
│   ├── gesture_state_machine.py     # 6 模式手势 FSM
│   ├── color_detector.py            # HSV 物件颜色检测（六色匹配）
│   ├── webcam_color_detector.py     # 衣物颜色兜底检测
│   ├── sketch_recognizer.py         # QuickDraw 82 类草图识别
│   └── *.task / *.tflite           # MediaPipe 模型文件
│
├── rag/                             # RAG 检索引擎
│   ├── character_recommend.py       # 35 人物推荐（颜色+物象）
│   ├── generator.py                 # LLM 叙事生成
│   ├── retriever.py                 # 知识库检索
│   └── knowledge/                   # 实体 + 组合 + 模板
│
├── web/                             # Web 层
│   ├── ws_server.py                 # WebSocket 桥接 (:8080 → TCP :8888/:8889)
│   └── frontend/                    # React 前端
│       └── src/
│           ├── pages/               # 五幕 + 开场组件
│           │   ├── Act0/            # 第零幕 · 开场
│           │   ├── Act1Entry/       # 第一幕 · 入境
│           │   ├── Act2ColorSeeking/# 第二幕 · 寻色
│           │   ├── Act3FormingVision/# 第三幕 · 筑景
│           │   ├── Act4SpiritCalling/# 第四幕 · 唤灵（流程版）
│           │   └── Act5Postcard/    # 第五幕 · 成笺
│           ├── SpiritModule/        # 唤灵模块（可移植）
│           ├── components/          # 共享组件 (LiquidChrome)
│           └── assets/              # SVG 美术资源 (act0~act5)
│
├── docs/                            # 文档
│   ├── interaction.md               # 交互设计文档
│   └── technical.md                 # 技术架构文档
│
├── test_integrated.py               # 集成测试入口（摄像头 + FSM + TCP）
├── config_ipcam.py                  # 摄像头 URL 配置
└── requirements.txt                 # Python 依赖
```

---

## 快速开始

### 1. 环境

```bash
conda create -n ohhh python=3.12
conda activate ohhh
pip install -r requirements.txt
cd web/frontend && npm install
```

### 2. 启动（美术测试）

```bash
# 终端 1：Mock 后端
python web/mock_backend.py

# 终端 2：WebSocket 桥接
python web/ws_server.py

# 终端 3：前端
cd web/frontend && npm run dev
```

浏览器打开：

| 幕 | 地址 |
|---|------|
| 第零幕 | `http://127.0.0.1:5173/test_act0.html` |
| 第一幕 | `http://127.0.0.1:5173/test_act1.html` |
| 第二幕 | `http://127.0.0.1:5173/test_act2.html` |
| 第三幕 | `http://127.0.0.1:5173/test_act3.html` |
| 第四幕 | `http://127.0.0.1:5173/test_act4.html` |
| 第五幕 | `http://127.0.0.1:5173/test_act5.html` |

### 3. 启动（真实摄像头）

```bash
# 终端 1：集成后端（摄像头 + 手势 FSM）
python test_integrated.py

# 终端 2：WebSocket 桥接
python web/ws_server.py

# 终端 3：前端
cd web/frontend && npm run dev
```

浏览器打开 `http://127.0.0.1:5173`，切换 Live Mode。

---

## 五幕叙事

| 幕 | 名称 | 用户动作 | 视觉 |
|---|------|---------|------|
| 0 | 开场 | 观看 | 静止底图 + 标题 + 飘动 icon + 光效 |
| 1 | 入境 | 握拳 | 彩带呼吸 + 邀请文字 → 碎光散开白屏 |
| 2 | 寻色 | 握拳触发颜色检测 | 液态 Chrome 颜色盘 + 外周旋转 + 动态文字 |
| 3 | 筑景 | 食指绘画 + 握拳提交 | 颜色条 + 鼠标手绘 + 物象识别叠加 |
| 4 | 唤灵 | 自动播放 | 八阶段叙事 → 人物框 + 颜色块漂浮 + 身份揭示 |
| 5 | 成笺 | 自动播放 (27s) | 双色液态盘 + 人物盘 + AI 文案 + 二维码 |

---

## 手势交互

| 手势 | 作用 |
|------|------|
| ✊ 握拳 | 确认 / 提交 / 重启择色 |
| ☝️ 食指伸出 | 进入绘画 / 录制轨迹 |
| 🖐️ 张手 | 取消 / 返回 |

手势识别：MediaPipe 21 关键点 → FSM 6 模式状态机 → 1 帧防抖确认。

---

## 后端接入

各幕通过 props 接收后端数据，详见 `MIGRATION.md`。核心接口：

- **第二幕**：`step` + `recognizedColors` + `copyByStep`
- **第三幕**：`primaryColor` + `secondaryColor` + `onRecognizeSketch`
- **第四幕**：`primaryColor` + `secondaryColor` + `onFetchSpiritMatch`
- **第五幕**：`postcardData`（颜色、物象、人物、AI 文案、二维码）

---

## 备份

完整历史版本在 `backup/full-20260622` 分支。
