# 前端对接技术文档

## 架构

```
浏览器 React (Vite :5173)
    │  WebSocket ws://127.0.0.1:8080
    ▼
ws_server.py  (:8080)  ← WebSocket ↔ TCP 桥接
    │  TCP (:8888 主通道, :8889 手部通道)
    ▼
test_integrated.py  (:8888/:8889)  ← 摄像头 + 手势 FSM + 颜色/物象/人物识别
    │
    ├── vision/   MediaPipe 手部追踪 + 手势 FSM + HSV 颜色检测 + QuickDraw 草图识别
    ├── rag/      知识检索 + 人物推荐 (35人) + LLM 叙事生成 + OSS 明信片上传
    └── web/frontend/src/pages/  五幕 React 组件
```

## 启动

```bash
# 终端 1：后端
python test_integrated.py

# 终端 2：WebSocket 桥接
python web/ws_server.py

# 终端 3：前端
cd web/frontend && npm run dev
# → http://127.0.0.1:5173
```

美术测试（不需要后端）：

| 幕 | 地址 |
|---|------|
| 第零幕 | `http://127.0.0.1:5173/test_act0.html` |
| 第一幕 | `http://127.0.0.1:5173/test_act1.html` |
| 第二幕 | `http://127.0.0.1:5173/test_act2.html` |
| 第三幕 | `http://127.0.0.1:5173/test_act3.html` |
| 第四幕 | `http://127.0.0.1:5173/test_act4.html` |
| 第五幕 | `http://127.0.0.1:5173/test_act5.html` |

## WebSocket 消息协议

所有消息 JSON + `\n` 分隔，UTF-8 编码。

### 前端 → 后端

| type | 说明 |
|------|------|
| `gesture_simulate` | `{"gesture": "fist"\|"open_hand"\|"index_pointing"}` TCP 手势模拟 |

### 后端 → 前端

#### 连接与状态

| type | 关键字段 |
|------|---------|
| `connected` | `message: "integrated_server_ready"` |
| `gesture_state` | `mode` + `sub_state` + `gesture` |
| `hand_appeared` | `palm_center: [x, y]` |
| `hand_tracking` | `landmarks` + `fingertips` + `palm_center` |

#### 第一幕 · 择色

| type | 关键字段 |
|------|---------|
| `color_extraction_start` | `message` |
| `object_color_detected` | `color` + `confidence` + `source: "object"` |
| `object_color_failed` | `message` |
| `clothing_color_detected` | `color` + `confidence` + `source: "clothing"` |
| `clothing_color_failed` | `message` |
| `color_confirmed` | `color` + `message` |

#### 第二幕 · 筑景

| type | 关键字段 |
|------|---------|
| `drawing_start` | `message` |
| `drawing_point` | `x` + `y` （食指指尖） |
| `object_recognized` | `object.name` + `object.score` + `object.qd_category` |
| `drawing_cancelled` | `message` |
| `object_confirmed` | `object` + `objects_so_far` + `can_continue` |

#### 第三幕 · 唤灵

| type | 关键字段 |
|------|---------|
| `objects_summary` | `objects` + `message` |
| `character_search_start` | `message` + `context` |
| `character_found` | `message` |
| `character_candidates` | `candidates: [{name, title, score, reason, monologue, spiritLine}]` |
| `character_performance` | `paragraphs: string[]` |
| `character_revealed` | `name` + `title` + `message` |

#### 第四幕 · 成笺

| type | 关键字段 |
|------|---------|
| `generation_result` | `title` + `paragraphs` + `context` |
| `postcard_result` | `image_url` + `qr_base64` + `unique_id` |

## 手势 → FSM 状态映射

| FSM mode | sub_state | 手势 | 作用 |
|----------|-----------|------|------|
| `COLOR_EXTRACTION` | `AWAITING_OBJECT` | fist | 触发颜色分析 |
| `COLOR_EXTRACTION` | `OBJECT_CONFIRMING` | fist | 确认颜色（已自动确认） |
| `GLOBAL` | `IDLE` | index_pointing | 进入绘画 |
| `GLOBAL` | `IDLE` | fist | 重启择色 |
| `DRAWING` | `TRACKING` | index_pointing | 录制轨迹 |
| `DRAWING` | `TRACKING` | fist | 提交绘画→识物→自动确认 |
| `DRAWING` | `TRACKING` | open_hand | 取消绘画 |
| `CANDIDATE` | `BROWSING` | fist | 确认物象 |
| `CANDIDATE` | `BROWSING` | open_hand | 重画 |

## 前端适配器

`src/services/backendAdapter.js` 负责后端消息 → 前端内部事件转换：

| 后端消息 | 前端事件 |
|---------|---------|
| `object_color_detected` / `clothing_color_detected` / `color_confirmed` | `color_detected` |
| `object_recognized` | `object_recognized` |
| `character_candidates` / `character_revealed` | `character_matched` |
| `character_performance` → | `system_log` |
| `generation_result` | `narrative_generated` |
| `postcard_result` | `postcard_ready` |
| `hand_tracking` / `drawing_point` | `drawing_point` |

## 后端接入方式

各幕组件通过 props 接收后端数据，不需要全局状态库：

```jsx
// 第二幕
<Act2ColorSeeking step={step} recognizedColors={colors} copyByStep={copy} />

// 第三幕
<Act3FormingVision primaryColor={c1} secondaryColor={c2} onRecognizeSketch={fn} />

// 第四幕
<Act4SpiritCalling primaryColor={c1} secondaryColor={c2} onFetchSpiritMatch={fn} />

// 第五幕
<Act5Postcard postcardData={data} />
```

完整接入示例见 `MIGRATION.md`。
