# 技术分析与实现

---

## 一、总体架构

```
识别区
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    ┌─────┐  ┌─────┐  ┌─────┐     ← 模块（ESP32+触摸）   │
│    │ESP32│  │ESP32│  │ESP32│                          │
│    └──┬──┘  └──┬──┘  └──┬──┘                          │
│       │        │        │                              │
│       └────────┼────────┘                              │
│                │ WiFi/蓝牙                             │
│                │                                       │
│         ┌──────▼──────┐                               │
│         │  ESP32 Hub  │ ← 接收模块ID+触摸状态          │
│         └──────┬──────┘                               │
└────────────────┼────────────────────────────────────────┘
                 │
          ┌──────┴──────┐
          │   摄像头    │ ← 视觉定位
          └──────┬──────┘
                 │
          ┌──────▼────────────────────────────────────────────┐
          │                   Python 本地端                   │
          │  ┌─────────────┐        ┌─────────────────────┐ │
          │  │  数据融合    │        │    RAG 检索生成      │ │
          │  │ 视觉位置     │   →    │  文化内容匹配        │ │
          │  │ + ESP32 ID  │        │  生成控制指令        │ │
          │  └─────────────┘        └─────────────────────┘ │
          └─────────────────────────┬─────────────────────┘
                                    │
                              ┌─────▼─────┐
                              │  Unity    │ ← 实时渲染输出
                              └───────────┘
                                    │
                              ┌─────▼─────┐
                              │  投影仪   │
                              └───────────┘
                                    │
                              云端大模型 → 多模态故事内容
```

---

## 二、RAG在技术路径中的位置与作用

### 2.1 RAG是什么

**RAG = Retrieval-Augmented Generation（检索增强生成）**

简单来说：**先检索，再生成**。不是让AI凭空编造，而是让它从知识库中找到相关信息，再基于这些信息生成回答。

### 2.2 RAG在整体架构中的位置

```
用户选择（颜色+物象+人物+连接）
           ↓
      ┌────▼────┐
      │ 数据融合 │ ← 模块一：视觉定位+ESP32识别
      └────┬────┘
           ↓
┌──────────▼──────────────────────────┐
│           RAG 模块（二）             │ ← 核心：知识检索 + 内容生成
│  ┌─────────┐    ┌──────────────┐    │
│  │ 知识库   │──▶│   检索引擎      │    │
│  │ 文化数据 │   │   精确+双向     │    │
│  └─────────┘    └──────┬───────┘    │
│                        ↓              │
│                 ┌──────▼───────┐     │
│                 │  阿里云百炼   │     │
│                 │  qwen-turbo  │     │
│                 └──────┬───────┘     │
│                        ↓              │
│                 ┌──────▼───────┐     │
│                 │  实时侧输出   │─────┼──→ Unity（实时渲染）
│                 │  高质量侧输出 │─────┼──→ 云端API（最终画作）
│                 └──────────────┘     │
└──────────────────────────────────────┘
```

### 2.3 RAG的两个输出层级

| 输出层级 | 触发时机 | 输出内容 | 去向 |
|---------|---------|---------|------|
| **实时侧** | 每放置一个模块 | 元素描述、连接关系、节点颜色/样式 | → Unity即时显示 |
| **高质量侧** | 点击确认后 | 完整文化上下文、多元素组合解读 | → 云端API生成最终画作 |

### 2.4 RAG的核心价值

| 传统生成 | RAG增强生成 |
|---------|-------------|
| AI凭空编造，可能胡编乱造 | 先检索知识库，确保内容准确 |
| 泛泛而谈 | 结合湖大文化知识，言之有据 |
| 统一叙事 | 根据用户选择组合，生成个性化内容 |

---

## 三、模块一：视觉定位（YOLO为主）+ ESP32（备选）

### 3.1 视觉定位（摄像头/YOLO + MOT）

```
实时视频帧流  →  YOLO 模型  →  BoT-SORT 跟踪  →  模块坐标 / 类型 / track_id / 数量
```

| 项目 | 内容 |
|------|------|
| **架构** | YOLO v8n / YOLO v5s + Ultralytics BoT-SORT |
| **参数量** | 1.5M ～ 3.5M |
| **运行位置** | 本地（电脑端） |
| **核心框架** | PyTorch + Ultralytics |
| **跟踪器** | BoT-SORT（内置），`persist=True` 跨帧 ID 一致 |
| **输出结果** | 模块坐标、类型、track_id、数量、置信度 |
| **实际状态** | ✅ 已实现 (`yolo/predict_yolo.py`, `vision/data_fusion/main.py`) |

#### 3.1.1 多目标跟踪（MOT）

采用 Ultralytics 内置的 **BoT-SORT** 跟踪器，`model.track(persist=True)` 一行接入：

- **跨帧 ID 一致性**：同一物理模块在连续帧中保持相同 `track_id`
- **放置/移除检测**：新 track_id 出现 = 模块放置事件；track_id 消失 = 模块移除事件
- **多模块区分**：无需 ESP32 ID 即可区分不同模块，融合器用 `yolo_track_{id}` 生成稳定模块 ID
- **降级**：`--no-track` 回退到逐帧 `model.predict()`，不生成 tracking ID

> ESP32 备选方案仍可补充触摸状态，track_id 解决了"哪个模块是哪个"的核心身份问题。

### 3.2 ESP32 模块识别（备选）

```
模块内置ESP32 + 触摸传感器  →  WiFi/蓝牙  →  ESP32 Hub  →  唯一ID + 触摸状态
```

| 项目 | 内容 |
|------|------|
| **通信方式** | WiFi UDP / 蓝牙 BLE |
| **每个模块** | ESP32 + 触摸传感器 + 唯一ID |
| **作用** | 备选方案：用于辅助身份确认或触摸检测 |
| **输出结果** | 模块唯一ID、触摸状态、在线状态 |
| **实际状态** | ⚠️ 部分实现 (`test_socket_server.py`) |

### 3.3 数据融合（YOLO为主）

```
YOLO检测+track（主） + ESP32数据（备）  →  融合数据
                                              │
                      ┌───────────────────────┘
                      ↓
              每个模块的：track_id + 位置 + 类型 + 置信度 + 触摸状态
```

| 项目 | 内容 |
|------|------|
| **融合逻辑** | YOLO+track为主（track_id 保证跨帧身份一致），ESP32数据辅助触摸状态 |
| **实际状态** | ✅ 已实现 (`vision/data_fusion/`) |
| **实现文件** | `fuse.py` - DataFusion核心融合器（支持 track_id） |
| | `esp32_hub.py` - ESP32数据接收器（备选） |
| | `main.py` - 数据融合主程序（集成 MOT 跟踪） |

---

## 四、模块二：AI 个性化叙事生成（RAG）

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG 模块                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐   │
│  │  知识库     │────▶│  检索引擎      │────▶│  阿里云百炼     │   │
│  │  (文化数据) │     │  (精确+双向)   │     │  (qwen-turbo)  │   │
│  └─────────────┘     └─────────────┘     └────────────────┘    │
│                                                   │             │
│  ┌───────────────────────────────────────────────┴─────────┐   │
│  │                      输出分层                          │   │
│  ├──────────────────────┬───────────────────────────────┤   │
│  │     实时侧           │        高质量侧               │   │
│  │  (每放置一个模块)    │       (开始生成时)            │   │
│  ├──────────────────────┼───────────────────────────────┤   │
│  │  • 元素一句话描述     │  • 完整文化上下文              │   │
│  │  • 连接关系类型      │  • 多元素组合解读              │   │
│  │  • 节点颜色/样式    │  • 打包发送给云端API           │   │
│  │  → Unity即时显示     │  → 云端生成图文叙事            │   │
│  └──────────────────────┴───────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 输入输出定义

**输入（来自Unity或ESP32 Hub）：**

```python
# 实时侧输入（每放置/连接一个模块）
{
    "event": "module_placed",      # module_placed / module_connected / generation_start
    "module_id": "esp32_001",
    "module_type": "color",        # color / object / character
    "position": (320, 240),
    "connections": [               # 当前模块的所有连接
        {"to": "esp32_003", "type": "color_grant"},
    ],
    "all_modules": [...]           # 当前所有已放置模块
}

# 高质量侧输入（开始生成时）
{
    "event": "generation_start",
    "all_modules": [...],
    "all_connections": [...]
}
```

**实时侧输出（Python → Unity）：**

```python
{
    "module_id": "esp32_001",
    "entity": "岳麓绿",
    "description": "这是一种生命的颜色，代表着千年不息的传承。",
    "connections": [
        {"to": "esp32_003", "type": "color_grant", "style": "绿色光线"}
    ],
    "unity_data": {
        "node_color": "#2E7D32",
        "line_color": "#FFD700",
        "line_style": "dashed"
    }
}
```

**高质量侧输出（Python → 云端API）：**

```python
{
    "api_payload": {
        "modules": [
            {"entity": "岳麓绿", "description": "...", "context": "..."},
            {"entity": "书院", "description": "...", "context": "..."},
            {"entity": "张栻", "description": "...", "context": "..."},
        ],
        "combinations": [...],
        "character_spirit": "传道济民",
        "prompt_template": "根据以下元素生成个性化叙事..."
    }
}
```

### 4.3 知识库结构

```
rag/knowledge/
├── 颜色实体定义.txt             # 原始数据（23条）
├── 物象典故.txt                 # 原始数据（88条）
├── 人物故事.txt                 # 原始数据（160条）
├── 组合解读.txt                 # 原始数据（100组）
├── 叙事模板.txt                 # 原始数据（100句）
├── entities/                    # 实体 JSON
│   ├── colors.json            # 23 个颜色实体
│   ├── objects.json           # 88 个物象实体
│   └── characters.json        # 97 个人物实体
├── combinations/               # 组合解读 JSON（107条）
│   ├── 岳麓绿_岳麓书院.json    # 文件名即查询 key
│   ├── 湖湘红_湘江.json
│   └── ...
└── templates/                 # 叙事模板 JSON
    └── narrative.json         # 开头/场景/人物/结尾 4 类共 100 句
```

| 项目 | 状态 |
|------|------|
| 知识库目录结构 | ✅ 已建立 |
| 颜色实体定义 | ✅ 23 条（2026-05-06） |
| 物象典故 | ✅ 88 条（2026-05-06） |
| 人物故事 | ✅ 97 条（2026-05-06） |
| 组合解读 | ✅ 107 条（2026-05-06） |
| 叙事模板 | ✅ 100 句（2026-05-06） |
| 构建工具 | ✅ `rag/build_knowledge.py`（txt→json 自动化） |
| **总实体数** | **208**（23+88+97） |
| RAG 加载验证 | ✅ `kb.get_entity('岳麓绿')` → "岳麓书院千年古木" |

**数据格式**：
- **颜色**: `type, description, color(hex), symbolism, related_entities, mood, era, theme`
- **物象**: `type, description, symbolism, related_entities, historical_context`
- **人物**: `type, name, years, title, description, spirit, related_entities, quotes, stories`
- **组合**: `entity1, entity2, meaning, interpretation`（文件名 = `{e1}_{e2}.json`，检索直接命中）
- **模板**: 4 类场景句子数组（开头 30 + 场景 30 + 人物 20 + 结尾 20）

---

## 五、模块三：多模态内容生成与呈现

```
RAG叙事文本  →  阿里云百炼  →  多模态故事包  →  Unity 呈现
```

| 项目 | 内容 | 状态 |
|------|------|------|
| **文本生成** | 阿里云百炼（RAG检索增强） | ✅ 已实现 |
| **实时文本** | qwen-turbo（快速响应） | ✅ 已实现 |
| **叙事卡文本** | qwen-plus（高质量生成） | ✅ 已实现 |
| **视觉生成** | wanx-v1（万象，云端生成） | ✅ 已实现 |
| **输出** | 叙事卡、明信片图片 | ✅ 已实现（postcard.py） |

**已接入功能**：

| 函数 | 说明 |
|------|------|
| `generate_realtime_description()` | 实时生成单模块描述（qwen-turbo） |
| `generate_connection_description()` | 生成连接关系描述（qwen-turbo） |
| `generate_narrative()` | 生成完整叙事卡（qwen-plus） |
| `generate_image_prompt()` | 生成画作提示词（qwen-turbo / 降级模板） |
| `generate_image()` | 调用万象 wanx-v1 生成图像 |
| `create_postcard()` | 生成明信片/叙事卡图片（动态布局） |
| `to_json()` | 输出完整叙事卡数据（含Base64图像） |

**使用方式**：

```python
from rag.generator import create_generator, create_config

# 配置API Key（通过环境变量 DASHSCOPE_API_KEY）
config = create_config(api_key="your-key")
generator = create_generator(config)

# 实时文本生成
desc = generator.generate_realtime_description("岳麓绿", context)

# 叙事卡生成
narrative = generator.generate_complete_narrative(context)
```

---

## 六、模块四：Unity 实时渲染与交互呈现

| 输入数据 | 输出效果 |
|---------|---------|
| 模块位置+ID（实时） | 场景中模块的虚拟对应物 |
| 模块触摸状态（实时） | 即时光影反馈、元素生长动画 |
| 个性化叙事文本 | 四幕叙事旁白、最终精神画像 |

| 项目 | 状态 |
|------|------|
| Python→Unity通信 | ⚠️ 部分实现 (`unity_bridge/sender.py`) |
| Unity→Python通信 | ⚠️ 部分实现 (`unity_bridge/server.py`) |
| 手势数据发送 | ⚠️ 部分实现 (`unity_bridge/hand_server.py`) |
| Unity工程 | ❌ 未实现（单独管理） |

---

## 七、生图提示词转换逻辑

### 7.1 方案三：LLM归因（已实现）

```
用户组合（颜色+物象+人物+连接）
       ↓
   LLM 阅读理解（qwen-turbo）
       ↓
   生成完整生图提示词
       ↓
   发送到云端 API（阿里云百炼 wanx-v1 万象）
```

### 7.2 实现代码

```python
def generate_prompt(user_input):
    prompt = f"""根据以下湖大千年色组合，生成一幅 Stable Diffusion 画作描述：

颜色：{user_input['colors']}
物象：{user_input['objects']}
人物：{user_input['characters']}
连接：{user_input['connections']}

要求：
- 风格：穿越感，水墨画 + 老照片实景混合
- 人物：保留两个人影，远距离呈现
- 底图：颜色晕染成水墨背景
- 叙事元素融入画面

输出：一段英文提示词，200词以内，用于 text-to-image 生成"""

    response = ollama.chat(model="llama3", messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']
```

### 7.3 失败处理

- LLM 输出不稳定 → 降级到模板填充（方案一）
- 云端 API 超时 → 返回"画作生成中，请稍候"

| 项目 | 状态 |
|------|------|
| LLM归因逻辑 | ✅ 已实现 |
| 失败降级处理 | ✅ 已实现 |

---

## 八、技术实现状态总览

| 模块 | 子模块 | 状态 |
|------|--------|------|
| **模块一** | YOLO视觉定位 | ✅ 已实现 |
| | MOT多目标跟踪 | ✅ 已实现（BoT-SORT） |
| | 摄像头连接 | ✅ 已实现 |
| | 手部检测 | ✅ 已实现 |
| | 区域标定 | ✅ 已实现 |
| | ESP32通信 | ⚠️ 部分实现 |
| | **数据融合** | ✅ 已实现 |
| **模块二** | 检索引擎(精确+双向) | ✅ 已实现 |
| | 生成模块(阿里云百炼) | ✅ 已接入 |
| | 知识库内容 | ✅ 已构建（208实体+107组合+100模板） |
| | 输入输出定义 | ✅ 已实现（RAGSystem 全部接口） |
| | 连接图分析+模板融合 | ✅ 已实现（`_analyze_connections` + `_pick_templates`） |
| | 人物推荐引擎 | ✅ 已实现（54核心人物，四维打分） |
| **模块三** | 文本生成(实时) | ✅ 已实现 |
| | 叙事卡生成 | ✅ 已实现 |
| | 视觉生成 | ✅ 已实现（wanx-v1 万象） |
| | 云端API | ✅ 已接入 |
| | **明信片合成** | ✅ 已实现（postcard.py 重写布局引擎） |
| **模块四** | Unity通信 | ⚠️ 部分实现 |
| | 手势数据发送 | ⚠️ 部分实现 |
| | Unity工程 | ❌ 未实现 |
| **生图转换** | 生成图像提示词 | ✅ 已实现 |

---

## 九、完成度评估

### 按子模块统计

| 模块 | 总项 | ✅ | ⚠️ | ❌ | 完成度 |
|------|------|----|----|----|--------|
| 模块一：视觉定位与融合 | 8 | 7 | 1 | 0 | **88%** |
| 模块二：RAG 检索与生成 | 6 | 6 | 0 | 0 | **100%** |
| 模块三：多模态内容生成 | 5 | 5 | 0 | 0 | **100%** |
| 模块四：Unity 渲染交互 | 3 | 0 | 2 | 1 | **17%** |
| 生图提示词转换 | 1 | 1 | 0 | 0 | **100%** |
| **总计** | **23** | **19** | **3** | **1** | **~87%** |

> 模块二说明：检索采用精确匹配+双向查找（BM25 模糊检索为 P2 搁置，非必需）。输入输出接口全部实现（`RAGSystem` 类）。连接图分析（`_analyze_connections`）、模板+LLM融合叙事、明信片端到端管道均已可用。模块二已基本完成。

### 按可运行能力评估

| 能力 | 状态 |
|------|------|
| 摄像头 → YOLO 检测+MOT跟踪 → 数据融合 | ✅ 端到端可跑 |
| 融合数据 → RAG 知识检索 → 实时描述 | ✅ 端到端可跑 |
| 检索上下文 → 叙事生成 → 图像生成 → 明信片合成 | ✅ 端到端可跑 |
| Python 端 → Unity 实时渲染 | ❌ 未打通 |
| ESP32 硬件模块通信 | ⚠️ 备选方案，非主线 |
| QuickDraw 模型训练 | ✅ 82 类训练完成，val acc 83.8%，ONNX 已导出 |

**核心 AI 管线 ~97%，Unity 交互层 ~17%，整体 ~87%。**

---

## 十、明信片图文生成 —— 问题跟踪

### ✅ 已解决

#### 问题 1：损坏的导入语句（已修复 2026-05-06）

**修复**：`rag/__init__.py` 中 `CloudGenerator` 改为 `AliCloudGenerator`。

#### 问题 2：明信片与 RAG 系统集成管道（已实现 2026-05-06）

`RAGSystem.generate_postcard()` 已串联文本叙事→图像生成→明信片合成三步。

#### 问题 3：字体自动适配（已实现 2026-05-06）

`FontManager` 已改为自动扫描：项目 `assets/fonts/` → `C:\Windows\Fonts` → 兜底。当前运行字体：`simhei.ttf`。

#### 问题 4：知识库内容（已构建 2026-05-06）

208 实体 + 107 组合 + 100 模板，详见 §4.3。

---

### 🐛 集成测试期间发现的 Bug（2026-05-06）

#### Bug 1：文本生成输出为空

**原因**：`result_format='message'` 时内容在 `choices[0].message.content`，代码读的是 `output.text`。

**修复** (`rag/generator.py:85`)：优先取 `choices[0].message.content`。

#### Bug 2：图像模型名错误

**原因**：`qwen-image-2.0-pro` 不存在。

**修复** (`rag/generator.py:34`)：改为 `wanx-v1`，支持 `prompt_extend` 和 `negative_prompt`。

#### Bug 3：明信片排版严重错乱

| 子问题 | 根因 | 修复 |
|--------|------|------|
| 图片不可见 | 文字占满画布，图片仅剩 106px 被跳过 | 图片放标题下方，500px 优先区 |
| 文字不换行 | PIL `draw.text` 不处理 `\n` | `textbbox` 按像素宽度逐字换行 |
| 画布不足 | 固定 1440px | 预计算动态高度 |
| 行间距失效 | 一段多行只加一个 line_spacing | 每行独立绘制 |

**修复** (`rag/postcard.py`)：完全重写布局引擎，布局顺序为 标题 → AI 水墨画（装裱）→ 正文（自动换行）→ 落款印章。

---

#### Bug 4：API Key 未传递到生成器（已修复 2026-05-06）

**原因**：`create_generator()` 无参调用时创建空 `GenerationConfig()`，不读环境变量。`generate_image()` 也直读 `self.config.api_key` 不降级到 env。

**修复**：
- `create_generator()`：无 config 时调用 `create_config()` 自动读 env
- `generate_image()`：`api_key = self.config.api_key or os.environ.get("DASHSCOPE_API_KEY", "")`

#### ✅ 端到端测试通过（2026-05-06）

`RAGSystem.generate_postcard()` 全流程跑通：

| 阶段 | 结果 |
|------|------|
| 知识库加载 | 208实体 + 107组合 + 100模板 |
| 模块注册 | m1=岳麓绿(color), m2=岳麓书院(object), m3=朱熹(character) |
| 连接检索 | m1↔m2: color_grant "学术传承", m2↔m3: scene_entry |
| 叙事生成 | 5段完整叙事，模板+LLM融合 |
| 图像生成 | wanx-v1 生成成功 |
| 明信片合成 | 动态布局 → `output/postcard.png` (~1MB) |

---

### ⚠️ 待处理

| 优先级 | 问题 | 说明 |
|--------|------|------|
| P1 | 字体 zip 解压 | `assets/fonts/SourceHanSansSC.zip` 解压后优先使用思源字体 |
| P2 | 多场景覆盖测试 | 测试不同模块组合的生成效果 |
| P2 | BM25/模糊检索 | 当前精确匹配，缺模糊检索能力 |

---

## 十一、交互变更带来的新开发任务（2026-05-09）

交互文档 `proposal/interaction.md` 进行了以下核心变更：
- **第二幕**：物理物象块 → 手势绘画 + 草图识别
- **第三幕**：物理人物块 → 智能推荐（方案A）+ 轮盘自选（方案B）

以下是由此产生的新开发模块和需扩展的现有模块。

---

### 11.1 新增模块总览

| 模块 | 文件（建议） | 类型 | 说明 |
|------|-------------|------|------|
| **草图识别** | `vision/sketch_recognizer.py` | 新增 | 指尖轨迹栅格化 → CNN 分类 → 物象映射 |
| **人物推荐引擎** | `rag/character_recommend.py` | ✅ 已实现 | 四维打分（参考表+同组+关键词+基础）→ Top-3 推荐 |
| **人物轮盘** | `vision/character_wheel.py` | 新增 | 5组人物横向滚动 + 手势控制 |
| **手势状态机** | `vision/gesture_state_machine.py` | 扩展 `gesture_connection.py` | 多模式状态管理（绘画/候选选择/人物推荐/人物轮盘） |
| **QuickDraw 模型训练** | `vision/quickdraw/` | ✅ 已就绪 | 下载93类.npy数据 → 训练CNN → 导出ONNX |

---

### 11.2 模块一：草图识别（`vision/sketch_recognizer.py`）

**输入**：MediaPipe 食指指尖轨迹（`List[(x, y, t)]`）
**输出**：Top-3 物象候选（`List[(entity_name, confidence)]`）

**管线**：

```
指尖轨迹（landmark 8 点序列）
       ↓
轨迹归一化：平移到原点，缩放到 28×28 画布
       ↓
轨迹栅格化：28×28 单通道灰度图，笔画宽度 2px，抗锯齿
       ↓
CNN 推理：自训练 QuickDraw CNN（93 类）
  • 模型加载：ONNX（vision/models/quickdraw_mobilenet.onnx）
  • 降级方案：HeuristicPredictor（几何特征分类，模型不可用时自动切换）
       ↓
QuickDraw → 88 物象映射表（一级映射）
  • 例：{"river": ["湘江", "流水", "湖面"], "tree": ["古树", "竹林", "林荫道"], ...}
  • 一个 QuickDraw 类可映射多个物象
       ↓
颜色上下文加权（二级排序）
  • 输入：第一幕选择的颜色
  • 加权规则：见 interaction.md §2.2.3 的加权表
       ↓
Top-3 候选物象 [(name, score), ...]
```

**QuickDraw 模型训练模块**（`vision/quickdraw/`）：

| 文件 | 说明 |
|------|------|
| `config.py` | 82 个 QuickDraw 类别列表（经官方 categories.txt 校准）、超参数、路径配置 |
| `download_data.py` | 并发下载（默认 8 线程），从 GCS 下载 `.npy` 位图文件，支持 `--retry` 重试失败 |
| `model.py` | 2层 CNN：Conv→BN→ReLU→MaxPool ×2 → FC→Dropout ×2 → logits |
| `dataset.py` | PyTorch Dataset，自动划分 train/val（85/15），每类取前 15K 样本 |
| `train.py` | 训练入口：AdamW + CosineAnnealing → 导出 ONNX，支持 `--resume` 断点续训 |

**断点续训**：
```bash
python -m vision.quickdraw.train --resume    # 从上次中断点继续
python -m vision.quickdraw.train --export-only  # 仅从 checkpoint 导出 ONNX
```
每个 epoch 保存最佳模型到 `checkpoint.pth`，包含 optimizer state、epoch、best_acc。

训练流程：
```bash
python -m vision.quickdraw.download_data   # 1. 下载数据（82类，~8.5GB, 10-20min）
python -m vision.quickdraw.train           # 2. 训练 + 导出 ONNX（GPU 1-2h / CPU 4-6h）
```
输出文件：
- `vision/models/quickdraw_mobilenet.onnx` — 供 `sketch_recognizer.py` 加载
- `vision/models/quickdraw_classes.json` — 类别索引映射
- `vision/quickdraw/checkpoint.pth` — 断点文件，用于恢复训练

**模型规格**：

| 属性 | 值 |
|------|-----|
| 架构 | 2层CNN（Conv 5×5, 32→64 通道） + 2层FC（512→128） + 输出层 |
| 输入 | `(B, 1, 28, 28)` float32 灰度图 |
| 输出 | `(B, 82)` logits |
| 参数量 | ~635K |
| 训练数据 | Google QuickDraw .npy 位图，82 类 × 15K 样本/类 ≈ 1.2M |
| 数据增强 | 无（.npy 数据已含天然笔触变异，足够覆盖手势绘画差异） |

**类别校准**：原始 93 类中有 16 个不在 QuickDraw 官方 345 类别中，已移除并将对应物象重新分配到最近的有效类别。最终 82 类全覆盖 88 物象。

**需要开发的子任务**：

| 任务 | 说明 | 优先级 | 状态 |
|------|------|--------|------|
| 轨迹归一化 + 栅格化 | 点序列 → 28×28 灰度图，含抗锯齿笔画渲染 | P0 | ✅ |
| QuickDraw 数据下载 | 82 类 .npy 位图，~8.5GB | P0 | ✅ |
| QuickDraw 模型训练 | 自训练 CNN → ONNX 本地推理，含断点续训 | P0 | ✅ |
| 类别映射表 | QuickDraw 82 类英文名 → 88 物象中文名映射 | P0 | ✅ |
| 颜色加权算法 | 根据第一幕颜色对映射后候选排序 | P1 | ✅ |
| 几何特征降级 | HeuristicPredictor：9 种几何分支，模型不可用时自动降级 | P0 | ✅ |

**训练结果**（2026-05-09）：

| 指标 | 值 |
|------|-----|
| 最终验证准确率 | **83.8%** |
| 训练时间 | ~10 分钟（RTX 4060 Laptop GPU, 15 epochs） |
| ONNX 模型大小 | 2.5MB |
| 训练集 | 1,045,500 样本（82 类 × ~12.7K） |
| 验证集 | 184,500 样本 |
| 参数量 | 653,234 |

收敛过程：
```
Epoch  1: val acc 77.8%
Epoch  5: val acc 82.1%
Epoch 10: val acc 83.5%
Epoch 15: val acc 83.8%  ← 最佳
```

**QuickDraw 数据来源**：
- [Google Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)（5000 万张草图，345 类）
- 28×28 灰度位图 `.npy` 格式，HTTP 直接下载
- 本项目选取其中 82 个与 88 物象有映射关系的类别（经官方 categories.txt 校验）
- 不支持类别（16个）：waterfall, tower, window, road, bookshelf, pen, paper, newspaper, blackboard, plate, boat, flag, trophy, medal, magnifying glass, baseball cap

---

### 11.3 模块二：人物推荐引擎（`rag/character_recommend.py`）✅ 已实现

**输入**：第一幕颜色 + 第二幕所有已选物象 + 已选人物列表
**输出**：Top-3 推荐人物 `[RecommendResult(name, title, score, reason), ...]`

**管线**：

```
颜色 + 物象 + 已选人物
       ↓
四维打分（纯本地，无依赖）
  (1) 内置参考表命中  (0~0.50) : 已知颜色+物象→人物映射（与 interaction.md 推荐示例表一致）
  (2) 已选人物同组加权 (0~0.20) : 同理学脉/湘军/维新/现代学人 group 加权
  (3) 关键词文本匹配  (0~0.25) : 颜色关键词 + 物象名在人物 title/spirit/description 中的命中
  (4) 实体基础分      (0~0.05) : description 长度、title 存在性
       ↓
降序排序 → Top-15 候选
       ↓
[可选] LLM 精选（qwen-turbo）
  • prompt：给定颜色 X、物象 Y、已选人物 Z，从15候选精选3位
  • 降级：LLM 不可用时直接返回启发式 Top-3
       ↓
Top-3 推荐人物 + 推荐理由
```

**数据来源**：
- 54 个核心人物嵌入在 `CORE_CHARACTERS` 列表中（6 组：理学脉络12/湘军将帅4/维新革命6/现代学人12/校园角色13/抽象意象9）
- 原因：`build_knowledge.py` 的 TXT 解析器存在格式缺陷，历史人物名大量丢失，改为手动维护精确列表
- 颜色→物象→人物参考表与 `interaction.md` §第三幕推荐示例完全一致

**核心类/接口**：

| 类/函数 | 说明 |
|---------|------|
| `CharacterRecommender` | 推荐引擎主类 |
| `recommend(color, objects, selected, use_llm)` | 主接口，返回 `List[RecommendResult]` |
| `_score_one(name, data, color, objects, selected)` | 单人物四维打分 |
| `_llm_rerank(candidates, ...)` | LLM 二次精选（可选，需 dashscope API） |
| `create_character_recommender(knowledge_path, api_key)` | 工厂函数 |
| `CHARACTER_GROUPS` | 6 组人物分组字典 |
| `COLOR_TO_SPIRIT_KEYWORDS` | 6 色 → 精神关键词表 |

**实现状态**：

| 子任务 | 状态 | 说明 |
|--------|------|------|
| 内置参考表 | ✅ 已实现 | 6色 × 多种物象 → 推荐人物，与交互文档一致 |
| 已选人物同组加权 | ✅ 已实现 | CHARACTER_GROUPS 6组覆盖全部核心人物 |
| 关键词文本匹配 | ✅ 已实现 | 颜色关键词 + 物象名双通道命中 |
| LLM 排序 prompt | ✅ 已实现 | 可选启用，失败自动降级到启发式 |
| 已选人物排除 | ✅ 已实现 | 推荐结果自动过滤已选人物 |

---

### 11.4 模块三：人物轮盘（`vision/character_wheel.py`）

**输入**：160 人物数据 + 手势输入（滑动/悬停/握拳/张手）
**输出**：选中的人物 entity_name

**数据结构**：

```
5 组人物：
  ├── 古代先贤（~30人，墨金色 #C5A35A）
  │     朱熹、张栻、王夫之、周敦颐、程颢、程颐……
  ├── 近代湖湘（~25人，赭红色 #9B3A3A）
  │     曾国藩、左宗棠、胡林翼、谭嗣同、魏源、黄兴……
  ├── 现代学人（~30人，藏蓝色 #2C3E6B）
  │     毛泽东、杨昌济、李达、成仿吾、熊十力、冯友兰……
  ├── 校园角色（~55人，青绿色 #3A7D5A）
  │     学子、教师、研究者、志愿者、讲解员、图书管理员……
  └── 抽象意象（~20人，月白色 #D5D5C8）
        理学之魂、书院守望者、文化摆渡人、知识火种者……
```

**轮盘交互逻辑**：

```
手势水平位移 Δx → 滚动速度 = f(Δx)
       ↓
当前组内卡片横向滚动
       ↓
手静止 > 1s 且手掌在某卡片区域 → 卡片放大 + 显示简介
       ↓
握拳 → 选中当前高亮卡片
       ↓
张手 → 退出轮盘，回到推荐阶段
```

**需要开发的子任务**：

| 任务 | 说明 | 优先级 |
|------|------|--------|
| 人物分组数据 | 160 人物分 5 组，每组含 name + title + description + color JSON | P0 |
| 横向滚动逻辑 | 卡片池渲染、惯性滚动、边界回弹 | P0 |
| 滑动速度映射 | 手势水平位移增量 → 轮盘滚动速度的非线性映射 | P0 |
| 悬停检测 + 计时 | 手静止区域判断 + 1s 计时器 → 卡片放大 | P1 |
| 分组切换 | 手势上/下滑切换人物分组 | P1 |

> **注意**：轮盘的**渲染**由 Unity 前端负责，本模块只输出轮盘**逻辑状态**（当前组、当前高亮卡片、滚动偏移、选中事件），通过 `unity_bridge` 发送给 Unity。

---

### 11.5 模块四：手势状态机扩展（扩展现有 `vision/gesture_connection.py`）

当前 `GestureConnection` 只有一套状态机（IDLE / HOVERING / CONNECTING / COMPLETING），需要扩展为**多模式状态机**：

```
顶层模式（Mode）：
  ├── GLOBAL        ← 全局模式（握拳晕染、张手停止，已有）
  ├── DRAWING       ← 绘画模式（第二幕，新增）
  ├── CANDIDATE     ← 候选物象确认（第二幕，新增）
  ├── CHARACTER_RECOMMEND  ← 人物推荐确认（第三幕阶段一，新增）
  └── CHARACTER_WHEEL      ← 人物轮盘浏览（第三幕阶段二，新增）
```

**各模式内的子状态**：

| 模式 | 子状态 | 触发手势 | 行为 |
|------|--------|---------|------|
| DRAWING | TRACKING | 食指伸出 | 追踪指尖轨迹 |
| DRAWING | COMPLETED | 握拳 | 轨迹定型 → 送入草图识别 |
| DRAWING | CANCELLED | 张手 | 清空轨迹 |
| CANDIDATE | BROWSING | 手悬停候选卡片上 | 高亮对应卡片 |
| CANDIDATE | CONFIRMED | 握拳 | 确认选中物象 |
| CANDIDATE | CANCELLED | 张手 | 回到 DRAWING |
| CHAR_RECOMMEND | BROWSING | 手悬停推荐卡片上 | 高亮 + 简介 |
| CHAR_RECOMMEND | CONFIRMED | 握拳 | 确认选中人物 |
| CHAR_RECOMMEND | TO_WHEEL | 张手 | 切换到 CHAR_WHEEL |
| CHAR_WHEEL | SCROLLING | 手水平移动 | 轮盘滚动 |
| CHAR_WHEEL | PREVIEWING | 悬停 > 1s | 卡片放大 + 简介 |
| CHAR_WHEEL | CONFIRMED | 握拳 | 确认选中人物 |
| CHAR_WHEEL | TO_RECOMMEND | 张手 | 回到 CHAR_RECOMMEND |

**需要开发的子任务**：

| 任务 | 说明 | 优先级 |
|------|------|--------|
| 多模式状态机基类 | Mode + SubState 二级状态管理 | P0 |
| 食指伸出检测 | `_recognize_index_pointing()`：仅食指伸展（landmark 8 tip < mcp），其余握起 | P0 |
| 绘画模式 | 轨迹录制 `List[(x,y,t)]`、握拳结束、张手清空 | P0 |
| 水平位移检测 | 连续帧手掌 x 坐标差值 → 轮盘滚动速度映射 | P1 |
| 静止计时器 | 手掌在卡片区域内静止 > 1s 触发悬停预览 | P1 |

---

### 11.6 已有模块需适配的变更

| 模块 | 文件 | 变更内容 |
|------|------|---------|
| **RAG 检索** | `rag/knowledge_base.py` | 新增 `search_characters_by_color_object(color, objects)` 人物推荐检索接口 |
| **数据融合** | `vision/data_fusion/fuse.py` | 新增 `event_type`：`sketch_completed`、`object_candidate_selected`、`character_recommended`、`character_wheel_selected` |
| **Unity Bridge** | `unity_bridge/` | 新增消息类型：候选物象卡片数据、人物推荐卡片数据、轮盘状态数据 |
| **手势连接** | `vision/gesture_connection.py` | 状态机扩展为多模式（见 §11.5），食指伸出检测 |

---

### 11.7 Unity Bridge 打通方案

**Unity 项目路径**：`D:\unity projects\ohhh-display`

#### 11.7.1 当前通信架构

```
Python 端                              Unity 端
─────────                              ────────
unity_bridge/server.py  :8888  ←→  TCP Client (C#)
  ├── module_placed  → Unity          接收描述、颜色、连线样式
  ├── connection_created → Unity
  └── generation_result → Unity

unity_bridge/hand_server.py :8889  →  TCP Client (C#)
  └── hand_tracking (每帧)

unity_bridge/sender.py  空文件 ← 待实现
```

已实现的消息类型见 §11.6 上方表格。

#### 11.7.2 实现顺序

| 阶段 | 内容 | 说明 |
|------|------|------|
| **一** | `sender.py` 统一发送器 | 封装 TCP 发送逻辑，支持重连、多端口。所有 Python→Unity 推送统一入口 |
| **二** | Act 2 物象候选 | 新增 `object_candidates` (Py→Unity) + `object_selected` (Unity→Py) |
| **三** | Act 3 人物推荐 | 新增 `character_candidates` + `character_selected` 消息 |
| **四** | Act 3 人物轮盘 | 新增 `wheel_state` + `wheel_selected` 消息 |
| **五** | 手势状态机接驳 | `gesture_connection.py` 多模式状态机回调 → `UnitySender` |

#### 11.7.3 新增消息类型规格

**Phase 1 — sender.py 接口**

```python
class UnitySender:
    """Python → Unity 统一消息发送器"""
    def __init__(self, host="127.0.0.1", port=8888, hand_port=8889):
        ...
    def send(self, data: dict) -> bool:
        """发送 JSON 消息，自动追加 \n，返回是否成功"""
    def send_object_candidates(self, color, candidates): ...
    def send_character_candidates(self, candidates): ...
    def send_wheel_state(self, groups, current_group, characters, highlighted): ...
    def send_hand_data(self, landmarks, palm_center): ...  # 迁移已有逻辑
    def close(self): ...
```

**Phase 2 — Act 2 物象候选**

```
Python → Unity: object_candidates
{
  "type": "object_candidates",
  "color": "岳麓绿",
  "candidates": [
    {"name": "古树", "score": 0.89, "qd_category": "tree"},
    {"name": "竹林", "score": 0.72, "qd_category": "tree"},
    {"name": "林荫道", "score": 0.65, "qd_category": "tree"}
  ]
}

Unity → Python: object_selected
{
  "type": "object_selected",
  "name": "古树"
}
```

**Phase 3 — Act 3 人物推荐**

```
Python → Unity: character_candidates
{
  "type": "character_candidates",
  "candidates": [
    {"name": "王夫之", "title": "思想家", "score": 0.85, "reason": "经典搭配推荐"},
    {"name": "胡安国", "title": "经学家", "score": 0.72, "reason": "关键词匹配"},
    {"name": "胡宏", "title": "思想家", "score": 0.68, "reason": "与已选人物同脉"}
  ]
}

Unity → Python: character_selected
{
  "type": "character_selected",
  "name": "王夫之"
}
```

**Phase 4 — Act 3 人物轮盘**

```
Python → Unity: wheel_state
{
  "type": "wheel_state",
  "groups": ["理学脉络", "湘军将帅", "维新革命", "现代学人", "校园角色"],
  "current_group": "理学脉络",
  "characters": [
    {"name": "周敦颐", "title": "理学开创者"},
    {"name": "程颢", "title": "理学宗师"},
    ...
  ],
  "highlighted_index": 0
}

Unity → Python: wheel_group_changed
{
  "type": "wheel_group_changed",
  "group": "湘军将帅"
}

Unity → Python: wheel_character_selected
{
  "type": "wheel_character_selected",
  "name": "曾国藩"
}
```

**Phase 5 — 手势状态机→Unity**

```
Python → Unity: gesture_state
{
  "type": "gesture_state",
  "mode": "DRAWING",           # GLOBAL | DRAWING | CANDIDATE | CHAR_RECOMMEND | CHAR_WHEEL
  "sub_state": "TRACKING",     # 各 mode 内的子状态
  "gesture": "index_pointing"  # 当前识别到的手势
}
```

#### 11.7.4 Unity C# 端接入要点

**TCP 连接**：
- 主通道：`127.0.0.1:8888` — 接收所有业务消息（candidates, wheel_state 等），发送用户选择
- 手部通道：`127.0.0.1:8889` — 实时手部 landmark 数据（每帧）

**消息解析**：
- 格式：`JSON + \n` 分隔
- 每条消息含 `type` 字段用于路由分发
- 使用 `JsonUtility.FromJson<T>()` 反序列化（需扁平化数组，见 hand_server.py 的 `_flatten` 处理）

**消息路由**：
```csharp
void OnMessage(string json) {
    var msg = JsonUtility.FromJson<BaseMessage>(json);
    switch (msg.type) {
        case "object_candidates":     HandleObjectCandidates(json); break;
        case "character_candidates":  HandleCharacterCandidates(json); break;
        case "wheel_state":           HandleWheelState(json); break;
        case "hand_tracking":         HandleHandTracking(json); break;
        case "gesture_state":         HandleGestureState(json); break;
    }
}
```

**UI 组件需求**：
| 消息 | Unity UI 表现 |
|------|-------------|
| `object_candidates` | 3 张横向候选卡片，显示名称 + 置信度条 + 颜色标识 |
| `character_candidates` | 3 张人物卡片，显示名称 + 称号 + 推荐理由 |
| `wheel_state` | 当前组横向滚动卡片列表，高亮卡片放大 |
| `gesture_state` | 状态指示器（绘画中/识别中/选择中） |

---

### 11.8 开发优先级

```
P0（核心通路必须跑通）：
  1. sender.py 统一发送器
  2. object_candidates / object_selected 消息（Act 2 通路）
  3. character_candidates / character_selected 消息（Act 3 推荐通路）
  4. wheel_state / wheel_character_selected 消息（Act 3 轮盘通路）
  5. 手势状态机多模式框架
  6. 食指伸出检测 + 轨迹录制
  7. QuickDraw ONNX 模型推理（✅ 已完成）
  8. 人物推荐引擎接口（✅ 已完成）

P1（完整体验）：
  9. gesture_state 消息（状态机状态同步到 Unity）
  10. 颜色上下文加权排序（✅ 已完成）
  11. 轮盘分组切换（Unity→Python→更新 wheel_state）
  12. Unity C# 候选卡片 UI + 轮盘 UI

P2（优化打磨）：
  13. sender.py 断线重连逻辑
  14. 轮盘悬停计时预览
```
