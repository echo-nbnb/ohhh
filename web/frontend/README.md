# 寻麓千年色 · Web 前端

《寻麓千年色》的 React 前端。Vite + React 18 + Tailwind CSS + WebGL (OGL)。

## 运行

```bash
cd web/frontend
npm install
npm run dev          # → http://127.0.0.1:5173
npm run build        # 生产构建 → dist/
```

## 美术测试（独立页面，不依赖后端）

| 幕 | 地址 |
|---|------|
| 第零幕 · 开场 | `http://127.0.0.1:5173/test_act0.html` |
| 第一幕 · 入境 | `http://127.0.0.1:5173/test_act1.html` |
| 第二幕 · 寻色 | `http://127.0.0.1:5173/test_act2.html` |
| 第三幕 · 筑景 | `http://127.0.0.1:5173/test_act3.html` |
| 第四幕 · 唤灵 | `http://127.0.0.1:5173/test_act4.html` |
| 第五幕 · 成笺 | `http://127.0.0.1:5173/test_act5.html` |

## 工程结构

```
src/
├── App.jsx                          # 主应用（五幕编排）
├── main.jsx                         # Vite 入口
├── index.css                        # 全局样式 + Tailwind
│
├── pages/                           # 五幕页面组件
│   ├── Act0/                        # 第零幕 · 开场
│   │   ├── Act0.jsx                 #   静止底图 + 标题 + 飘动 icon + 光效
│   │   └── Act0.css
│   ├── Act1Entry/                   # 第一幕 · 入境
│   │   ├── Act1Entry.jsx            #   彩带呼吸 + 邀请文字切换
│   │   ├── Act1Entry.css
│   │   ├── DissolveOverlay.jsx      #   碎光散开 → 白屏过渡
│   │   └── DissolveOverlay.css
│   ├── Act2ColorSeeking/            # 第二幕 · 寻色
│   │   ├── Act2ColorSeeking.jsx     #   4 步流程 + LiquidChrome 颜色盘
│   │   ├── Act2ColorSeeking.css
│   │   ├── LiquidChrome.jsx         #   WebGL 蓝黄双色液态效果
│   │   └── LiquidChrome.css
│   ├── Act3FormingVision/           # 第三幕 · 筑景
│   │   ├── Act3FormingVision.jsx    #   颜色条 + 鼠标手绘 + 物象识别叠加
│   │   └── Act3FormingVision.css
│   ├── Act4SpiritCalling/           # 第四幕 · 唤灵（流程版）
│   │   ├── Act4SpiritCalling.jsx    #   8 阶段自动播放 + 人物框 + 颜色块
│   │   └── Act4SpiritCalling.css
│   └── Act5Postcard/                # 第五幕 · 成笺
│       ├── Act5Postcard.jsx         #   11 帧叙事 + LiquidChrome + QR
│       └── Act5Postcard.css
│
├── components/                      # 共享组件
│   ├── DualColorLiquidChrome/       #   Act5 专用 WebGL 液态效果
│   └── LiquidChromeDual/            #   Act2/Act5 共用 WebGL 液态效果
│
├── SpiritModule/                    # 可移植唤灵模块
│   ├── SpiritPage.jsx               #   1920×1080 舞台, 等比缩放
│   ├── components/                  #   CharacterReveal, SpiritCanvas, etc.
│   ├── hooks/                       #   useSpiritSequence, useStageScale
│   └── config/                      #   布局坐标 + 时间配置
│
├── assets/                          # SVG 美术资源
│   ├── act0/                        #   背景 + 标题 + icon + 光效
│   ├── act1/                        #   背景 + 彩带 + 邀请文字 + 落款
│   ├── act2/                        #   背景 + 颜色盘 + 外周
│   ├── act3/                        #   背景 + 标题 + 5 条颜色条 + 桥
│   ├── act4/                        #   背景 + 标题 + 颜色条 + 人物框 + 颜色块
│   └── act5/                        #   背景 + 颜色圈 + 颜色盘 + 人物盘 + 图案 + 标题
│
├── data/                            # 前端数据
│   ├── colors.js                    #   六色定义 + findColor()
│   ├── objects.js                   #   8 物象 + recognizeObject()
│   └── characters.js                #   8 人物 + matchCharacter()
│
├── hooks/
│   └── useWebSocket.js              # WebSocket 连接管理
│
└── services/
    ├── backendAdapter.js            # 后端消息 → 前端事件转换
    ├── messageTypes.js              # 消息类型枚举
    └── mockEngine.js                # Demo Mode mock 数据
```

## 五幕叙事

| 幕 | 组件 | 后端对接 |
|---|------|---------|
| 0 | `Act0` | `onNext` 回调 |
| 1 | `Act1Entry` | `switchDelay`(5s) + `dissolveDelay`(13s) → 碎光到 Act2 |
| 2 | `Act2ColorSeeking` | `step` + `recognizedColors` + `copyByStep` |
| 3 | `Act3FormingVision` | `primaryColor` + `secondaryColor` + `onRecognizeSketch` |
| 4 | `Act4SpiritCalling` | `primaryColor` + `secondaryColor` + `onFetchSpiritMatch` |
| 5 | `Act5Postcard` | `postcardData`（颜色/物象/人物/AI文案/QR） |

## 技术要点

- **WebSocket**：`useWebSocket` hook, `backendAdapter.js` 消息归一化
- **WebGL**：`LiquidChrome` 双色液态效果 (OGL), 60fps GPU 渲染
- **颜色系统**：SVG mask-image 动态着色, `createMaskStyle()` 工具函数
- **飘动 icon**：CSS `@keyframes` + `translate3d`, 各幕统一复用
- **移植**：`SpiritModule` 和 `PostcardModule` 为独立可移植模块
- **测试页**：`test_act0~5.html` + `main-act0~5.jsx` 独立入口, 不影响主流程
