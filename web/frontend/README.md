# 寻麓千年色 · Web 正式前端框架

《寻麓千年色》是一个关于湖南大学与湖湘文化的 AI 交互体验项目。当前 `web/` 使用 Vite、React、JavaScript 与 Tailwind CSS，替代原 前端展示层，并为已有 Python 颜色识别、MediaPipe 手势、草图识别、人物匹配和 RAG 叙事模块预留接入结构。

## 当前状态

- Demo Mode 已使用 Mock 数据跑通完整五幕流程。
- Live Mode 已预留 WebSocket、统一消息枚举和后端适配层。
- 当前前端尚未与真实 Python 服务联调。
- 页面是舞台式单页应用，不使用 React Router，也没有手动跳步控制台。
- 人物始终由“颜色 + 物象”自动匹配 Top-1，不存在人物选择卡片。

## 运行方式

需要 Node.js 18 或更高版本。

```bash
cd web
npm install
npm run dev
```

访问：`http://127.0.0.1:5173`

生产构建：

```bash
npm run build
```

## 页面流程

```text
开场邀请 intro
→ 第一幕：择色 color
→ 第二幕：筑景 draw
→ 第三幕：唤灵 spirit
→ 第四幕：成色 postcard
```

1. 开场只有“开始寻色”按钮。
2. Demo 择色调用 `mockDetectColor()`，结果展示 1.5 秒后自动进入筑景。
3. Demo 绘画完成调用 `mockRecognizeObject()`，叙事化结果展示 1.5 秒后自动进入唤灵。
4. Demo 唤灵调用 `mockMatchCharacter()` 自动匹配 Top-1 人物，依次展示寻找、`????`、人物独白和姓名揭示。
5. 点击“进入成色”后调用 `mockGenerateNarrative()` 生成明信片内容。
6. 点击“重新开始”清空全部状态并回到开场。

## 工程结构

```text
src/
├─ App.jsx
├─ components/
│  ├─ Header.jsx
│  ├─ StageStepper.jsx
│  ├─ SceneNarration.jsx
│  ├─ DrawingCanvas.jsx
│  ├─ SpiritReveal.jsx
│  ├─ Postcard.jsx
│  ├─ SystemLog.jsx
│  ├─ StatusPanel.jsx
│  └─ stages/
│     ├─ IntroStage.jsx
│     ├─ ColorStage.jsx
│     ├─ DrawStage.jsx
│     ├─ SpiritStage.jsx
│     └─ PostcardStage.jsx
├─ data/
│  ├─ colors.js
│  ├─ objects.js
│  ├─ characters.js
│  └─ narrativeFlow.js
├─ hooks/
│  └─ useWebSocket.js
└─ services/
   ├─ backendAdapter.js
   ├─ mockEngine.js
   └─ messageTypes.js
```

`App.jsx` 只负责编排阶段、状态和自动推进。舞台组件负责展示；数据文件保存文化内容；services 层隔离 Demo 算法与后端字段变化。

## services 层

### `messageTypes.js`

集中定义所有 WebSocket 消息类型。业务代码必须引用 `MESSAGE_TYPES`，不要散落硬编码字符串。

### `backendAdapter.js`

所有后端消息必须先经过：

```js
normalizeBackendMessage(payload)
```

它再调用颜色、物象、人物或叙事专用归一化函数，把 Python 字段转换成前端统一结构。后端字段变化时优先只修改此文件。

### `mockEngine.js`

集中管理 Demo Mode：

- `mockDetectColor()`
- `mockRecognizeObject(points, color)`
- `mockMatchCharacter(color, objectResult)`
- `mockGenerateNarrative(context)`

组件和 `App.jsx` 不直接实现随机颜色、轨迹识别、人物评分或 Mock 叙事。

## 后续 Python 后端接入

1. Python 服务在 `ws://127.0.0.1:8000/ws` 提供 WebSocket。
2. 前端切换到 Live Mode，点击“连接后端”。
3. 后端发送 JSON 消息。
4. `useWebSocket.js` 负责连接和 JSON 解析。
5. `backendAdapter.js` 把原始消息归一化。
6. `App.jsx` 只处理统一后的业务对象。

如果 Python 输出字段与本文不同，不要在组件中增加兼容判断；应在 `backendAdapter.js` 中适配。

## WebSocket 消息格式

### `color_detected`

```json
{
  "type": "color_detected",
  "color": "岳麓绿",
  "source": "object",
  "confidence": 0.86
}
```

归一化后使用 `colorName`、`source`、`confidence`。

### `gesture_state`

```json
{
  "type": "gesture_state",
  "mode": "DRAWING",
  "gesture": "index_pointing"
}
```

### `drawing_point`

```json
{
  "type": "drawing_point",
  "x": 420,
  "y": 260
}
```

画布坐标系为 `800 × 460`。

### `object_recognized`

```json
{
  "type": "object_recognized",
  "name": "古树",
  "score": 0.82,
  "reason": "轨迹纵向生长，接近古树意象。"
}
```

### `character_matched`

```json
{
  "type": "character_matched",
  "character": {
    "name": "张栻",
    "title": "岳麓书院早期讲学者之一",
    "reason": "书院红与讲堂共同指向发问、讲学与书院精神。",
    "monologue": ["你选择了红。", "又画下讲堂。"],
    "spiritLine": "他留下的，不只是名字，而是一种敢于发问的颜色。"
  }
}
```

旧后端可发送 `characters_recommended`；适配层只取 `characters[0]`，页面不会显示选择卡片。

### `narrative_generated`

```json
{
  "type": "narrative_generated",
  "title": "岳麓绿 · 古树之境",
  "summary": "以岳麓绿为底，以古树入景。",
  "paragraphs": [
    "系统从用户的物件中提取出一抹岳麓绿。",
    "系统将绘画轨迹理解为古树。"
  ]
}
```

### `system_log`

```json
{
  "type": "system_log",
  "level": "info",
  "message": "视觉服务已就绪"
}
```

## 给美工对话的注意事项

- 可以优化 Tailwind 样式、排版、背景纹理、颜色晕染、人物剪影、明信片和轻量动效。
- 不要改变 `intro / color / draw / spirit / postcard` 阶段名和顺序。
- 不要修改 WebSocket 消息类型或 `backendAdapter.js` 的统一输出结构。
- 不要把 Mock 算法搬回组件或 `App.jsx`。
- 不要增加人物选择卡片。
- 保留 Demo Mode，它是无后端环境下的验收入口。
- 不要引入 shadcn/ui、Framer Motion 等大型依赖，除非项目组明确同意。
