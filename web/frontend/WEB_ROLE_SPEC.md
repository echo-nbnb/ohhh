# Web 对话职责边界

## 框架对话负责

- Vite、React、Tailwind CSS 工程结构。
- `App.jsx` 核心状态管理和五幕自动流程。
- Demo Mode 的完整可运行链路。
- Live Mode、WebSocket 连接与异常处理。
- `services/` 后端适配层和 Mock 引擎。
- 前后端消息契约、字段归一化和兼容策略。
- 构建、运行稳定性与工程文档。

核心状态包括：

```text
mode
currentStage
color
colorSource
gesture
points
objectResult
matchedCharacter
spiritStatus
narrative
logs
isAutoAdvancing
```

## 美工对话负责

- 页面视觉层级和 Tailwind 样式。
- 舞台背景、纹理、字体、色彩与响应式布局。
- 不改变业务状态的轻量动效。
- 颜色扫描与晕染视觉。
- 绘画画布、人物剪影和唤灵揭示视觉。
- 明信片、盖章和二维码占位区域设计。

## 美工对话不能改

- WebSocket 消息格式。
- `messageTypes.js` 的消息枚举。
- `backendAdapter.js` 的统一输出格式。
- `mockEngine.js` 的导出函数和数据结构。
- `App.jsx` 核心状态流和自动推进规则。
- `currentStage` 的阶段命名与顺序。
- Top-1 自动人物匹配规则。
- 不得增加 Top-3 或任何人物手动选择卡片。

## 固定页面流程

```text
intro → color → draw → spirit → postcard
```

- 择色结果展示后自动进入筑景。
- 物象结果展示后自动进入唤灵。
- 唤灵先显示寻找状态和 `????`，再播放人物独白，最后揭示姓名。
- 人物揭示后通过“进入成色”生成明信片。
- 重新开始必须清空全部流程状态。

## 前后端契约

所有消息类型统一定义在 `src/services/messageTypes.js`。

所有原始后端消息必须先经过：

```js
normalizeBackendMessage(payload)
```

视觉组件不得直接读取 Python 原始字段。若后端字段改变，应由框架对话修改 `backendAdapter.js`，并同步更新 README。

默认 WebSocket 地址：

```text
ws://127.0.0.1:8000/ws
```

当前支持：

- `color_detected`
- `gesture_state`
- `drawing_point`
- `object_recognized`
- `character_matched`
- `characters_recommended`
- `narrative_generated`
- `system_log`

## 仓库保护

前端工作只发生在 `web/` 中。不得删除或破坏仓库已有的 Python、Unity、RAG、vision、proposal 和课程资料文件。
