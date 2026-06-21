# 美术资源移植指南

## 文件结构

```
web/frontend/
├── src/
│   ├── assets/
│   │   ├── act0/                    # 第零幕 · 开场
│   │   │   ├── act0-bg.svg         ← 第0幕背景.svg
│   │   │   ├── act0-title.svg      ← 文字大标题.svg
│   │   │   ├── act0-icon.svg       ← 飘动icon.svg
│   │   │   └── act0-light.svg      ← 背景光效.svg
│   │   └── act1/                    # 第一幕 · 入境
│   │       ├── act1-bg.svg         ← 第1幕背景.svg
│   │       ├── ribbon.svg          ← 彩带.svg
│   │       ├── invite-1.svg        ← 邀请文字（Source Han Serif CN）.svg
│   │       ├── invite-2.svg        ← 邀请文字2（Source Han Serif CN）.svg
│   │       └── signature.svg       ← 落款（Source Han Serif CN）.svg
│   ├── pages/
│   │   ├── Act0/
│   │   │   ├── Act0.jsx            # 第零幕组件
│   │   │   └── Act0.css
│   │   └── Act1Entry/
│   │       ├── Act1Entry.jsx        # 第一幕组件
│   │       └── Act1Entry.css
│   ├── main-act0.jsx                # 第零幕独立测试入口
│   └── main-act1.jsx                # 第一幕独立测试入口
├── test_act0.html                   # 第零幕独立测试页
└── test_act1.html                   # 第一幕独立测试页
```

## 美术测试地址（Vite dev 已启动时）

| 幕 | 地址 |
|---|------|
| 第零幕 · 开场 | `http://127.0.0.1:5173/test_act0.html` |
| 第一幕 · 入境 | `http://127.0.0.1:5173/test_act1.html` |
| 第二幕 · 寻色 | `http://127.0.0.1:5173/test_act2.html` |
| 第三幕 · 筑景 | `http://127.0.0.1:5173/test_act3.html` |
| 第三幕 · 唤灵（SpiritModule） | 直接 `import SpiritPage from "./SpiritModule/SpiritPage"` |
| 第四幕 · 唤灵 | `http://127.0.0.1:5173/test_act4.html` |
| 第五幕 · 成笺 | `http://127.0.0.1:5173/test_act5.html` |

## 日后接入主流程的步骤

### 第零幕（Act0）→ 第一幕（Act1Entry）

1. 在 `App.jsx` 中引入：

```jsx
import Act0 from "./pages/Act0/Act0";
import Act1Entry from "./pages/Act1Entry/Act1Entry";
```

2. 状态管理，例如：

```jsx
const [act, setAct] = useState("act0");

if (act === "act0") return <Act0 onNext={() => setAct("act1")} />;
if (act === "act1") return <Act1Entry />;
```

3. 接入后端时：握拳确认 → `onNext` → 进入 Act1Entry。

### 第二幕（Act2ColorSeeking）接入后端

```jsx
import Act2ColorSeeking from "./pages/Act2ColorSeeking/Act2ColorSeeking";

// step: 1-4 由后端控制
//   step=1: 白盘 + "请将随身之物靠近光中"
//   step=2: 白盘呼吸 + "让我看看藏着怎样的颜色"
//   step=3: 单色 LiquidChrome + 文字变化（仅第一色）
//   step=4: 双色 LiquidChrome + 文字变化（两色交融）
//
// recognizedColors: 后端返回的 hex 颜色数组
//   ["#F2C94C"]          → 单色（step=3）
//   ["#F2C94C","#2F5E9E"] → 双色（step=4）
//
// copyByStep: 可选，后端返回的动态文案
<Act2ColorSeeking
  step={backendStep}
  recognizedColors={backendColors}
  copyByStep={{
    1: ["请将随身之物靠近光中。", "让它替你说话。"],
    2: ["让我看看……", "这件东西里", "藏着怎样的颜色。"],
    3: ["这是明黄。", "它像旧纸上的日光，温热而安静。"],
    4: ["这是青蓝。", "它像山影压进水色，带着一点夜的深意。"],
  }}
/>
```

颜色盘 LiquidChrome 参数可调：
```jsx
<LiquidChrome
  colorA={hexToNorm(colors[0])}       // 第一色，归一化 [r,g,b]
  colorB={hexToNorm(colors[1])}       // 第二色（单色时与第一色相同）
  speed={0.25}                        // 流动速度
  amplitude={0.40}                    // 波动幅度
  frequencyX={2.0}                    // X 轴波纹密度（越小越疏）
  frequencyY={1.6}                    // Y 轴波纹密度
  interactive={false}                 // 是否响应鼠标
/>
```

### 注意

- `test_act0.html` / `test_act1.html` / `test_act2.html` / `test_act3.html` 是纯美术测试页，不会影响 `index.html` 主流程
- 移植时只需把组件引入到 `App.jsx` 即可
- 原有 `App.jsx` / `main.jsx` / `index.html` 保持不动

### 第三幕（Act3FormingVision）接入后端

```jsx
import Act3FormingVision from "./pages/Act3FormingVision/Act3FormingVision";

<Act3FormingVision
  primaryColor="#F2E700"          // 后端传入的第一色
  secondaryColor="#2F59F5"       // 后端传入的第二色
  maxRounds={2}                  // 最大绘画轮次
  onRecognizeSketch={async (payload) => {
    // payload: { scene, round, colors, canvas, strokes, bbox, sketchImageDataUrl }
    const res = await fetch("/api/recognize-sketch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
    // 后端返回: { label, description: string[], stylizedImageUrl, overlay?: { scale, offsetX, offsetY } }
  }}
/>
```

### 第三幕唤灵（SpiritModule）

```jsx
import SpiritPage from "./SpiritModule/SpiritPage";

<SpiritPage
  character={{
    id: "zhang_shi",
    name: "张栻",
    title: "岳麓书院早期讲学者之一",
    image: "",
    monologue: ["你选择了红。", "又画下讲堂。"],
    spiritLine: ["刚才与你说话的，是他。", "但他留下的不只是名字，", "更是一种敢于发问的底色。"],
  }}
  autoPlay={true}
  onComplete={(character) => setAct("act4")}
/>
```

### 第四幕（Act4SpiritCalling）接入后端

```jsx
import Act4SpiritCalling from "./pages/Act4SpiritCalling/Act4SpiritCalling";

<Act4SpiritCalling
  primaryColor="#F2E700"          // 后端传入的第一色
  secondaryColor="#355BFF"       // 后端传入的第二色
  firstColorName="蓝"            // 第一色中文名
  secondColorName="黄"           // 第二色中文名
  firstImageryName="桥"          // 第一个物象名
  secondImageryName="树"         // 第二个物象名
  onFetchSpiritMatch={async (payload) => {
    // payload: { scene, colors: { primary, secondary, firstColorName, secondColorName }, imagery: { first, second } }
    const res = await fetch("/api/spirit-match", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    return res.json();
    // 后端返回: { person: { id, name, subtitle: string[], portraitUrl }, narrative: { centerStart, centerSeek, loading, found, rightInterim, leftBlue, leftYellow, rightFinal } }
  }}
/>
```

时间线（自动播放）：开场(0s) → 寻找(2.6s) → 加载(5.7s) → 找到(8.6s) → 空框(9.8s) → 蓝色台词(11.6s) → 黄色台词(17s) → 人物揭示(22.8s)
```

### 第五幕（Act5Postcard）接入后端 v2

```jsx
import Act5Postcard from "./pages/Act5Postcard/Act5Postcard";

<Act5Postcard
  postcardData={{
    colors: { primary: "#F2E700", secondary: "#355BFF", primaryName: "明黄", secondaryName: "青蓝" },
    selectedPlaces: ["岳麓山", "湘江水", "书院檐角"],
    topTitle: "千年色笺正在成形",
    mainTitleImageUrl: "/api/generated/postcard-title.svg",
    person: { name: "张栻", portraitUrl: "/api/assets/people/zhangshi.png" },
    imageryItems: [{ id: "bridge", name: "桥", imageUrl: "/api/generated/bridge-dots.svg", position: { left: 46, top: 53, width: 23 }, explanation: ["你画下了一座桥。", "桥连接两岸，也连接出发与归来。"] }],
    aiWriting: ["你的颜色从山水之间醒来。", "明黄像一束照进旧纸的光..."],
    downloadQrUrl: "/api/postcard/qr/session-001.png",
    createdAtText: "2026.06.21\n09:32",
  }}
  personPortraitUrl={character.portraitUrl}  // 也可从 postcardData.person.portraitUrl 传入
  autoPlay={true}
  onFetchPostcardData={async () => fetch("/api/postcard-data").then(r => r.json())}
/>
```

叙事时间线（自动播放 27.2s）：成笺(0s) → 地点+桥(2.6s) → 树(5.2s) → 颜色盘+人物盘(7.8s) → 标题背景(10.4s) → 上方标题(13s) → 物象解说(15.8s) → AI文案(18.6s) → 生成链路(21.4s) → 握拳盖章+日期(24.2s) → 最终标题+二维码(27.2s)

调试某帧：`<Act5Postcard debugStage={10} autoPlay={false} />`（0-10）

颜色盘使用 `DualColorLiquidChrome`（`src/components/DualColorLiquidChrome/`），Act5 专用 WebGL 液态效果。
```
