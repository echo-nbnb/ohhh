# 前端对接技术文档

## 1. 架构

```
浏览器 (HTML/JS)
    │  WebSocket ws://127.0.0.1:8080
    ▼
ws_server.py  (:8080)
    │  TCP
    ▼
test_integrated.py  (:8888 主通道 / :8889 手部通道)
    │
    ├── vision/   MediaPipe 手部检测 + 手势状态机 + 颜色检测 + 草图识别
    ├── rag/      知识检索 + 人物推荐 + 叙事生成 + 明信片合成
    └── OSS       明信片上传 + 二维码生成
```

## 2. 启动顺序

```bash
# 终端 1: 后端（无摄像头模式用于开发测试）
python test_integrated.py --no-display --no-camera

# 终端 2: WebSocket 桥接
python web/ws_server.py

# 终端 3: 打开 web/index.html（直接双击或 Live Server）
```

## 3. WebSocket 连接

| 项 | 值 |
|----|-----|
| 地址 | `ws://127.0.0.1:8080` |
| 格式 | JSON 文本 |
| 编码 | UTF-8 |

---

## 4. 前端 → 后端（发送）

| type | 参数 | 说明 |
|------|------|------|
| `gesture_simulate` | `{"gesture": "fist"}` | 握拳 |
| `gesture_simulate` | `{"gesture": "open_hand"}` | 张手 |
| `gesture_simulate` | `{"gesture": "index_pointing"}` | 食指伸出 |
| `generation_start` | `{}` | 显式请求生成 |

## 5. 后端 → 前端（接收）

### 5.1 连接

| type | 字段 | 说明 |
|------|------|------|
| `connected` | `message: "integrated_server_ready"` | 后端就绪 |

### 5.2 状态同步

| type | 字段 | 说明 |
|------|------|------|
| `gesture_state` | `mode` | FSM 当前模式: `COLOR_EXTRACTION` / `GLOBAL` / `DRAWING` / `CANDIDATE` / `CHAR_RECOMMEND` |
| | `sub_state` | 子状态: `AWAITING_OBJECT` / `OBJECT_ANALYZING` / `OBJECT_CONFIRMING` / `TRACKING` / `BROWSING` / `IDLE` |
| | `gesture` | 当前检测到的手势: `fist` / `open_hand` / `index_pointing` |

### 5.3 第一段 · 寻色

| type | 字段 | 说明 |
|------|------|------|
| `color_extraction_start` | `message` | 引导文字 "请将随身之物靠近光中…" |
| `object_color_detected` | `color` | 颜色名（岳麓绿/书院红/西迁黄/湘江蓝/校徽金/墨色） |
| | `confidence` | 置信度 0~1 |
| | `source` | `"object"` |
| | `message` | 叙事文字 |
| `object_color_failed` | `message` | 物件未匹配提示 |
| `clothing_color_detected` | `color` | 衣物颜色名 |
| | `confidence` | 置信度 |
| | `source` | `"clothing"` |
| | `message` | 叙事文字 |
| `clothing_color_failed` | `message` | 衣物也未匹配提示 |
| `color_confirmed` | `color` | 最终确认的颜色 |
| | `source` | `"object"` / `"clothing"` / `"ink"` |
| | `message` | 确认叙事文字 |

### 5.4 第二段 · 造象

| type | 字段 | 说明 |
|------|------|------|
| `hand_appeared` | `palm_center: [x, y]` | 手首次进入画面 |
| `drawing_start` | `message` | "伸出食指，开始作画。" |
| `object_recognized` | `color` | 当前颜色 |
| | `object.name` | 识别物象名 |
| | `object.score` | 置信度 |
| | `object.qd_category` | QuickDraw 类别 |
| `drawing_cancelled` | `message` | 取消提示 |
| `object_confirmed` | `object` | 已确认的物象名 |
| | `objects_so_far: [...]` | 累积物象列表 |
| | `can_continue: true` | 是否可继续添加物象 |
| | `message` | 叙事文字 |

### 5.5 第三段 · 唤灵

| type | 字段 | 说明 |
|------|------|------|
| `character_search_start` | `message` | 因果解释文字 |
| | `context.color` | 用户颜色 |
| | `context.objects: [...]` | 用户物象列表 |
| `character_found` | `message` | "找到了。" |
| | `character_name_hidden: true` | 隐藏名字标志 |
| `character_performance` | `character: "????"` | 隐藏身份时显示 ???? |
| | `paragraphs: [...]` | 第一人称台词数组 |
| `character_revealed` | `name` | 人物名 |
| | `title` | 称号 |
| | `era` | 时代 |
| | `summary` | 简介 |
| | `message` | 揭示叙事文字 |

### 5.6 第四段 · 成笺

| type | 字段 | 说明 |
|------|------|------|
| `generation_result` | `title` | 叙事标题 |
| | `paragraphs: [...]` | 叙事段落数组 |
| | `context.color` | 用户颜色 |
| | `context.objects: [...]` | 用户物象 |
| | `context.character` | 选中人物 |
| `postcard_result` | `image_url` | **公网下载链接** |
| | `qr_base64` | **二维码 data URI** (`data:image/png;base64,...`) |
| | `unique_id` | 唯一编号 |
| | `message` | "扫码带走你的千年色。" |

---

## 6. 交互流程（完整手势序列）

```
用户进入    →  hand_appeared

握拳        →  color_extraction_start → object_color_detected
（自动）     →  gesture_state: OBJECT_CONFIRMING
握拳        →  color_confirmed → gesture_state: GLOBAL

食指伸出    →  drawing_start → gesture_state: DRAWING
（空中绘画）  →  指尖轨迹实时渲染
握拳        →  object_recognized → gesture_state: CANDIDATE
握拳        →  object_confirmed（可重复：食指再画 → 握拳提交 → 握拳确认）

握拳完成筑景 →  objects_summary → character_search_start
            →  character_found
            →  character_performance（人物第一人称演绎）
            →  character_revealed（揭示身份）
            →  generation_result（自动触发）
            →  postcard_result（OSS 上传 + 二维码）

张手        →  drawing_cancelled（回到绘画 / 取消）
```

---

## 7. 手势状态 → UI 阶段映射

| FSM mode | sub_state | 前端阶段 | 用户操作 |
|----------|-----------|---------|---------|
| `COLOR_EXTRACTION` | `AWAITING_OBJECT` | 寻色 · 等待 | 展示随身物品 |
| `COLOR_EXTRACTION` | `OBJECT_ANALYZING` | 寻色 · 分析中 | 等待 |
| `COLOR_EXTRACTION` | `OBJECT_CONFIRMING` | 寻色 · 确认 | 握拳确认 |
| `COLOR_EXTRACTION` | `CLOTHING_FALLBACK` | 寻色 · 衣物兜底 | 等待 |
| `GLOBAL` | `IDLE` | 自由态 | 食指绘画 / 握拳筑景 |
| `DRAWING` | `TRACKING` | 造象 · 绘画中 | 空中绘画 |
| `CANDIDATE` | `BROWSING` | 造象 · 确认物象 | 握拳确认 |
| `CHAR_RECOMMEND` | `BROWSING` | 唤灵 · 寻找中 | 等待 |
| `GLOBAL` | `IDLE` | 成笺 | 查看结果 / 扫码 |

---

## 8. 必需的前端组件

| 组件 | 触发消息 | 功能 |
|------|---------|------|
| 阶段指示器 | `gesture_state` | 显示当前在六段中哪一段 |
| 颜色展示 | `object_color_detected` + `color_confirmed` | 色雾聚成色名 + 解读文字 |
| 绘画画布 | `drawing_start` + `object_recognized` | 指尖轨迹实时渲染 |
| 物象列表 | `object_confirmed` | 已确认物象展示 |
| 人物剪影 | `character_found` | 模糊剪影 + 隐藏名字 |
| 人物台词 | `character_performance` | 逐段显示第一人称台词 |
| 人物揭示 | `character_revealed` | 姓名浮现动画 |
| 叙事卡片 | `generation_result` | 叙事文本 + 作品展示 |
| 二维码 | `postcard_result` | 二维码图片 + 下载引导 |
| 手势提示 | 主动显示 | 当前可用的手势和操作提示 |
