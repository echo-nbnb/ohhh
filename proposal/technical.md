# 技术分析与实现

---

## 一、总体架构

```
识别区（桌面俯拍）
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    ┌─────┐  ┌─────┐  ┌─────┐     ← 物理颜色牌（6 种）  │
│    │ 绿  │  │ 红  │  │ 黄  │                          │
│    └──┬──┘  └──┬──┘  └──┬──┘                          │
│       │        │        │                              │
│       └────────┼────────┘                              │
│                │                                       │
│         ┌──────▼──────┐                               │
│         │   摄像头    │ ← 俯拍检测类型+位置 + 手部追踪 │
│         └──────┬──────┘                               │
└────────────────┼────────────────────────────────────────┘
                 │
          ┌──────▼────────────────────────────────────────────┐
          │                   Python 本地端                   │
          │  ┌─────────────┐  ┌──────────┐  ┌────────────┐ │
          │  │  YOLO+双识别 │  │ MediaPipe │  │  手势状态机 │ │
          │  │  检测 (6类)  │  │ 手部追踪  │  │  (5 模式)  │ │
          │  └──────┬──────┘  └────┬─────┘  └─────┬──────┘ │
          │         │              │              │         │
          │  ┌──────┴──────────────┴──────────────┴──────┐  │
          │  │             Bridge 层                      │  │
          │  │  color_card / sketch / character          │  │
          │  └──────────────────┬────────────────────────┘  │
          │                     │                            │
          │  ┌──────────────────┴────────────────────────┐  │
          │  │         RAG 检索生成                       │  │
          │  │   知识库检索 → LLM 归因 → 叙事 + 图生图    │  │
          │  │   人物风格映射 → 颜色晕染底图 + 风格prompt │  │
          │  │   → Wan 2.7 图生图融合                   │  │
          │  └──────────────────┬────────────────────────┘  │
          └─────────────────────┼───────────────────────────┘
                                │
                          ┌─────▼─────┐
                          │  Unity    │ ← 实时渲染输出
                          └─────┬─────┘
                                │
                          ┌─────▼─────┐
                          │  投影仪   │
                          └───────────┘
```

**数据流**：摄像头帧 → 颜色牌检测（YOLO+双识别）+ 手部追踪 → 手势状态机 → 草图识别/人物推荐 → RAG 叙事生成 → 双端口 TCP → Unity 渲染

---

## ★ PPT 章节一：技术路径总览

### 1.1 技术架构图（PPT 第 X 页）

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户交互层                                    │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│   │ 颜色牌  │  │ 手势    │  │ 手势    │  │ 手势    │               │
│   │ 放置    │  │ 绘画    │  │ 悬停    │  │ 握拳    │               │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘               │
└────────┼────────────┼────────────┼────────────┼───────────────────────┘
         │            │            │            │
         ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         视觉输入层                                    │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│   │ YOLOv8n+双识别   │  │ MediaPipe        │  │ QuickDraw CNN    │ │
│   │ 6类 / 30fps     │  │ 21关键点 / 30fps │  │ 82类 / ONNX     │ │
│   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└────────────┼───────────────────┼───────────────────┼─────────────┘
             │                   │                   │
             ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         智能处理层                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Bridge 层                                │   │
│   │   ColorCardBridge  │  SketchBridge  │  CharacterBridge      │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                 │                                   │
│   ┌─────────────────────────────┴─────────────────────────────┐   │
│   │                 RAG 检索生成引擎                             │   │
│   │   知识库(208实体+107组合) → 阿里云百炼 → 叙事+图像           │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         渲染输出层                                    │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│   │ Unity 实时渲染   │  │ 双端口 TCP       │  │ 明信片合成       │ │
│   │ 手势轨迹/候选   │  │ :8888/:8889     │  │ PIL 动态布局    │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 技术路径文字（PPT 要点）

**标题**：技术路径——从用户选择到个性画作的四步流水线

**要点 1：视觉感知**
- 摄像头俯拍桌面，实时捕获颜色牌位置与手部动作
- YOLOv8n 检测 6 类颜色牌（岳麓绿/书院红/西迁黄/湘江蓝/校徽金/墨色）
- MediaPipe 追踪 21 个手部关键点，识别握拳/张手/食指伸出

**要点 2：意图理解**
- 食指指尖轨迹 → 28×28 灰度图 → QuickDraw MobileNet CNN 分类
- 颜色上下文加权：将第一幕颜色作为先验知识影响物象排序
- 人物推荐：四维打分（内置表+同组加权+关键词+实体分）

**要点 3：叙事生成**
- RAG 检索：知识库（208 实体+107 组合）精确匹配
- 阿里云百炼：qwen-plus 生成 4-5 段个性化叙事
- 人物第一人称：选中的历史人物成为叙事主语

**要点 4：视觉呈现**
- Wan 2.7 图生图：颜色晕染底图 + 人物风格 prompt → 融合图像
- 明信片合成：动态布局（画布高度随文字自适应）
- 双端口 TCP：:8888 主通道（事件）+ :8889 手部通道（~30fps）

---

## ★ PPT 章节二：数据路径详解

### 2.1 端到端数据流图（PPT 第 X 页）

```
用户操作                      Python 后端                     Unity 前端
────────                     ──────────                     ──────────

[放置颜色牌] ──────────────→ YOLOv8n 检测
                                    │
                                    │ {color_type, position}
                                    ▼
                              ColorCardBridge ───────────→ [光圈绽放动画]
                                    │
                                    ▼
                              Bridge Layer ─────────────────→ [颜色基调建立]


[食指绘画] ────────────────→ MediaPipe 追踪指尖
                                    │
                                    │ {trajectory: [(x,y,ts),...]}
                                    ▼
                              SketchRecognizer
                              (QuickDraw CNN + 颜色加权)
                                    │
                                    │ {top3_objects: [{name, score},...]}
                                    ▼
                              SketchBridge ───────────────→ [Top-3 物象浮现]


[悬停+握拳] ───────────────→ FSM 状态机
                                    │
                                    │ {object_selected: "岳麓书院"}
                                    ▼
                              RAG Retriever
                              (知识库检索)
                                    │
                                    ▼
                              CharacterBridge ──────────────→ [物象发光生长]


[人物推荐] ────────────────→ CharacterRecommend
                              (四维打分排序)
                                    │
                                    │ {top3_chars: [{name, title, score},...]}
                                    ▼
                              CharacterBridge ──────────────→ [人物卡片升起]


[触发生成] ────────────────→ generate_narrative()
                              (阿里云百炼 qwen-plus)
                                    │
                                 SSE 流式输出
                                    │
                                    ▼
                              [逐字书写动画] ← ────────────── Unity
                                    │
                              generate_image_with_base()
                              (Wan 2.7 图生图)
                                    │
                              [晕染扩散动画] ← ────────────── Unity
                                    │
                              create_postcard()
                              (PIL 明信片合成)
                                    │
                              {image_base64, title, paragraphs}
                                    │
                                    ▼
                              UnitySender ────────────────→ [完整明信片呈现]
```

### 2.2 关键数据格式（PPT 表格）

**摄像头帧 → 手部追踪数据**

| 字段 | 类型 | 示例 | 说明 |
|------|------|------|------|
| `landmarks` | `List[21, (x,y,z)]` | `[[0.52, 0.31, 0.0], ...]` | 21 个关键点归一化坐标 |
| `gesture` | `str` | `"index_pointing"` | 当前手势：握拳/张手/食指伸出 |
| `mode` | `str` | `"DRAWING"` | FSM 模式：GLOBAL/DRAWING/CANDIDATE |
| `sub_state` | `str` | `"TRACKING"` | 子状态：TRACKING/COMPLETED/CANCELLED |

**草图识别结果**

| 字段 | 类型 | 示例 | 说明 |
|------|------|------|------|
| `entity_name` | `str` | `"岳麓书院"` | 物象中文名 |
| `score` | `float` | `0.847` | 综合置信度 (0~1) |
| `qd_category` | `str` | `"house"` | 来源 QuickDraw 类别 |
| `raw_confidence` | `float` | `0.723` | CNN 原始概率 |

**叙事生成结果**

| 字段 | 类型 | 示例 | 说明 |
|------|------|------|------|
| `title` | `str` | `"岳麓松风"` | 生成的标题 |
| `paragraphs` | `List[str]` | `["第一段...", "第二段..."]` | 4-5 段叙事正文 |
| `summary` | `str` | `"绿染千年，理学文脉生生不息"` | 一句话总结 |
| `image_base64` | `str` | `"data:image/png;base64,..."` | 明信片完整图像 |

### 2.3 数据路径文字（PPT 要点）

**标题**：数据路径——信息如何在各模块间流动

**阶段 1：选择流动**
- 颜色牌：物理位置 → YOLOv8n bbox → 坐标映射 → Unity 颜色基调
- 物象：指尖轨迹 → CNN 分类 → 颜色加权 → Top-3 候选

**阶段 2：确认流动**
- 物象确认：FSM 状态变化 → RAG 检索 → 实时描述 → Unity 显示
- 人物推荐：四维打分 → Top-3 卡片 → 用户悬停/握拳确认

**阶段 3：生成流动**
- 叙事流：RAG 上下文 → LLM 流式生成 → SSE 推送 → Unity 逐字显示
- 图像流：底图 Base64 + 风格 prompt → Wan 2.7 → 融合图像 → 明信片

**关键设计：双端口 TCP**
- :8888 主通道，事件驱动，JSON + `\n` 分隔
- :8889 手部通道，~30fps，21 关键点实时推送
- TCP_NODELAY 禁用 Nagle 算法，消除 ~200ms 缓冲延迟

---

## ★ PPT 章节三：各模块实现

### 3.1 视觉输入模块（PPT 第 X 页）

#### 3.1.1 颜色牌检测技术

**架构图**：
```
摄像头帧 (640×480)
      │
      ▼
YOLOv8n 推理 (~3M 参数)
      │
      ├── class_id: 0-5 (6 种颜色)
      ├── bbox: [x1, y1, x2, y2]
      ├── confidence: 0.0-1.0
      └── track_id: BoT-SORT 跟踪
      │
      ▼
ColorCardBridge
      │
      ▼
{color_type, position} → Unity
```

**PPT 要点**：
- **模型**：YOLOv8n + Ultralytics BoT-SORT 跟踪
- **检测范围**：6 类（岳麓绿/书院红/西迁黄/湘江蓝/校徽金/墨色）
- **输出**：class_id、像素坐标、bbox、track_id、置信度
- **坐标映射**：摄像头 640×480 → Unity 1920×1080 线性映射

#### 3.1.2 手势识别技术

**FSM 状态机图**：
```
                    ┌─────────────┐
                    │   GLOBAL    │ ← 全局空闲（握拳晕染/张手停止）
                    └──────┬──────┘
                           │
              食指伸出 ─────┼───── 握拳/张手
                           ▼
                    ┌─────────────┐
                    │  DRAWING    │ ← 追踪指尖轨迹
                    │  TRACKING   │
                    └──────┬──────┘
                           │
              握拳提交 ─────┼───── 张手取消
                           ▼              ▼
                    ┌─────────────┐   ┌─────────────┐
                    │  CANDIDATE  │   │   GLOBAL    │
                    │  BROWSING   │   │   IDLE      │
                    └──────┬──────┘   └─────────────┘
                           │
              握拳确认 ─────┼───── 张手取消
                           ▼              ▼
                    ┌─────────────┐   ┌─────────────┐
                    │   GLOBAL    │   │  DRAWING    │
                    │   IDLE      │   │  TRACKING   │
                    └─────────────┘   └─────────────┘
```

**PPT 要点**：
- **MediaPipe Hand Landmarker**：21 个关键点，30fps
- **5 种手势识别**：握拳/张手/食指伸出/悬停/未知
- **5 模式 FSM**：GLOBAL / DRAWING / CANDIDATE / CHAR_RECOMMEND / CHAR_WHEEL

#### 3.1.3 草图识别技术

**识别管线图**：
```
食指指尖轨迹 (landmark 8 点序列)
           │
           ▼
轨迹归一化 ──── 平移到原点，缩放到 28×28
           │
           ▼
栅格化 ─────── 28×28 灰度图（笔画宽度 2px，抗锯齿）
           │
           ▼
CNN 推理 ───── QuickDraw MobileNet (82 类 ONNX)
           │  ~635K 参数，验证准确率 83.8%
           ▼
QuickDraw 类别 → 88 物象映射表
           │
           ▼
颜色上下文加权（第一幕颜色作为先验）
           │
           ▼
Top-3 候选物象 → Unity
```

**PPT 要点**：
- **输入**：轨迹点序列 → 28×28 灰度图
- **模型**：QuickDraw MobileNet，82 类，ONNX 2.5MB
- **准确率**：83.8%（RTX 4060 训练 ~10 分钟）
- **降级**：模型不可用时自动切换到 HeuristicPredictor（几何特征）

---

### 3.2 RAG 生成模块（PPT 第 X 页）

#### 3.2.1 知识库架构

**知识库结构图**：
```
rag/knowledge/
├── entities/
│   ├── colors.json      ─── 23 条（岳麓绿/书院红/...）
│   ├── objects.json     ─── 88 条（岳麓书院/湘江/古树/...）
│   └── characters.json  ─── 97 条（朱熹/张栻/曾国藩/...）
├── combinations/        ─── 107 条（颜色×物象/物象×人物/...）
└── templates/           ─── 100 句（开头30+场景30+人物20+结尾20）
```

**PPT 要点**：
- **实体总数**：208（23+88+97）
- **组合解读**：107 条预定义组合含义
- **叙事模板**：100 句模板句式
- **检索方式**：精确匹配 + 双向查找

#### 3.2.2 人物推荐引擎

**四维打分流程**：
```
颜色 + 物象 + 已选人物
         │
         ├── 内置参考表命中 ──────────── 0~0.50 分
         │
         ├── 已选人物同组加权 ────────── 0~0.20 分
         │
         ├── 关键词文本匹配 ──────────── 0~0.25 分
         │
         └── 实体基础分 ─────────────── 0~0.05 分
         │
         ▼
降序排序 → Top-15 → [可选 LLM 精选] → Top-3 推荐
```

**PPT 要点**：
- **核心人物**：54 人，6 分组
- **分组**：理学脉络(12)/湘军将帅(4)/维新革命(6)/现代学人(12)/校园角色(13)/抽象意象(9)
- **评分权重**：内置表为主(0.50)，文本匹配为辅(0.25)

#### 3.2.3 人物视觉风格映射

**风格决策表**：

| 分组 | 时代 | 风格 | 英文风格关键词 | 代表人物 |
|------|------|------|---------------|----------|
| 理学脉络 | 古代 | **古风** | Song dynasty ink wash, Zen aesthetics | 朱熹、张栻 |
| 湘军将帅 | 近代 | **写实** | Qing dynasty photography, sepia | 曾国藩、左宗棠 |
| 维新革命 | 近代 | **写实** | Revolutionary woodcut, bold contrasts | 谭嗣同、黄兴 |
| 现代学人 | 现代 | **写实** | Academic realism, natural lighting | 毛泽东、杨昌济 |
| 校园角色 | 现代 | **写实** | Campus slice of life, warm sunlight | 学子、教师 |
| 抽象意象 | 跨时代 | **穿越风** | Ink dissolving into digital, dreamlike | 理学之魂 |

#### 3.2.4 图生图管线

**融合流程图**：
```
Unity 前端                        Python 后端                      阿里云 API
─────────                      ──────────                      ─────────
用户选色 → 颜色晕染渲染
         │
         ├── Base64 底图 ────────→ generate_image_with_base()
         │                              │
用户选人物 ──→ build_character_style_prompt()
         │                              │
         │    古风/写实/穿越风 prompt ──→ Wan 2.7 图生图
         │                              │
         │                         ← 融合图像
         │                              │
         └──────────→ create_postcard() ←
                           │
                     明信片合成 → Unity 展示
```

**PPT 要点**：
- **底图来源**：Unity 前端实时渲染颜色晕染水墨纹理（Base64）
- **风格 prompt**：人物分组 → 时代 → 视觉风格英文描述
- **融合模型**：wan2.7-image-pro（异步，最长等待 2 分钟）
- **输出**：融合图像 URL + 本地下载 + 明信片 base64

---

### 3.3 通信与渲染模块（PPT 第 X 页）

#### 3.3.1 双端口 TCP 架构

**消息流图**：
```
Python (:8888 主通道)                    Python (:8889 手部通道)
        │                                       │
   事件驱动                                     │
   候选/确认/生成结果                    ~30fps 持续推送
        │                                       │
        ▼                                       ▼
┌───────────────────────────────────────────────────────────────┐
│                      Unity TCP Client                          │
│  ┌─────────────────┐        ┌─────────────────────────────┐  │
│  │ PythonConnection │        │ HandTrackingConnection      │  │
│  │ (主通道 :8888)  │        │ (手部通道 :8889)             │  │
│  └────────┬────────┘        └─────────────┬───────────────┘  │
│           │                                 │                  │
│           └─────────────┬───────────────────┘                  │
│                         ▼                                      │
│              ┌──────────────────┐                              │
│              │  消息路由分发器   │                              │
│              └────────┬─────────┘                              │
│                       │                                        │
│         ┌─────────────┼─────────────┐                          │
│         ▼             ▼             ▼                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                  │
│  │ 物象/人物  │ │ 手势状态    │ │ 手部关键点 │                  │
│  │ 候选UI     │ │ 指示器      │ │ 骨架可视化 │                  │
│  └────────────┘ └────────────┘ └────────────┘                  │
└───────────────────────────────────────────────────────────────┘
```

#### 3.3.2 Bridge 层职责

| Bridge | 输入 | 输出 | 文件 |
|--------|------|------|------|
| ColorCardBridge | YOLO 检测结果 | 颜色类型+坐标 → Unity | 待模型训练后新建 |
| SketchBridge | CNN 分类结果 | Top-3 物象候选 → Unity | `unity_bridge/sketch_bridge.py` |
| CharacterBridge | RAG 推荐结果 | Top-3 人物推荐 → Unity | `unity_bridge/character_bridge.py` |

---

### 3.4 生成过程表演化（PPT 第 X 页）

#### 3.4.1 阶段化生成时间线

**时序图**：
```
时间轴  0-3s     3-8s      8-20s     20-40s    40-50s    50s+
       │         │         │         │         │         │
       ▼         ▼         ▼         ▼         ▼         ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│  理解   │ │  构思   │ │  书写   │ │  晕染   │ │  落定   │ │  完成   │
│         │ │         │ │         │ │         │ │         │ │         │
│墨韵涟漪 │ │古籍书页 │ │毛笔笔触 │ │光点扩散 │ │人物剪影 │ │完整明信 │
│逐字浮现 │ │从四周飘 │ │逐字浮现 │ │颜色晕染 │ │精神光柱 │ │片呈现   │
│         │ │入拼合   │ │         │ │         │ │         │ │         │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │           │           │           │           │
     ▼           ▼           ▼           ▼           ▼           ▼
  RAG检索    LLM流式输出   叙事分批    Wan2.7生成   图像下载     全部完成
  开始       SSE推送       到达Unity   进度推送     明信片合成
```

#### 3.4.2 前端状态机

**状态转换表**：

| 阶段 | 状态名 | 前端动画 | 后端动作 | 预计时长 |
|------|--------|---------|---------|---------|
| 1 | `loading_narrative` | 墨韵涟漪 + "让我想想……" | RAG 检索 | 0-3s |
| 2 | `writing_narrative` | 古籍书页飘入拼合 | LLM 生成中 | 3-8s |
| 3 | `narrative_streaming` | 毛笔逐字书写 | SSE 流式输出 | 8-20s |
| 4 | `generating_image` | 光点向外晕染扩散 | Wan 2.7 图像生成 | 20-40s |
| 5 | `composing_postcard` | 人物剪影浮现+光柱 | 明信片 PIL 合成 | 40-50s |
| 6 | `completed` | 完整明信片 + 落款 | 交付 base64 | 50s+ |

#### 3.4.3 SSE 流式输出接口

**API 设计**：
```
GET /api/generate/stream
  Query: context={base64_encoded_context}
  Response: text/event-stream

  data: {"stage": "retrieving", "progress": 0}
  data: {"stage": "writing", "text": "第一段内容...", "progress": 30}
  data: {"stage": "writing", "text": "第二段内容...", "progress": 60}
  data: {"stage": "done", "progress": 100}
```

**WebSocket 图像进度**：
```
WS /ws/image-progress

{"stage": "init", "percent": 0}
{"stage": "encoding", "percent": 20}
{"stage": "diffusion", "percent": 50}
{"stage": "decoding", "percent": 80}
{"stage": "done", "percent": 100, "image_base64": "..."}
```

---

### 3.5 Unity 组件架构（PPT 第 X 页）

**组件关系图**：
```
Scene Setup (一键创建)
      │
      ├── PythonConnection (:8888 主通道)
      │         │
      │         └── ObjectCandidateUI (Top-3 物象候选)
      │         └── CharacterCandidateUI (Top-3 人物推荐)
      │         └── GestureStateUI (右上角 FSM 指示)
      │
      ├── HandTrackingConnection (:8889 手部通道)
      │         │
      │         └── HandTrackingVisualizer (21 关键点骨架)
      │
      └── GenerationUI (生成过程动画)
                │
                ├── InkRippleEffect (墨韵涟漪)
                ├── BrushStrokeText (毛笔书写)
                └── InkSpreadEffect (晕染扩散)
```

---

## ★ PPT 章节四：性能与可靠性

### 4.1 延迟优化措施（PPT 表格）

| 措施 | 效果 | 实现位置 |
|------|------|---------|
| TCP_NODELAY | 消除 ~200ms Nagle 缓冲延迟 | `unity_bridge/server.py` |
| 摄像头缓冲排空 | 消除 2-3 帧旧数据堆积 | `vision/hand_tracker.py` |
| no-display 模式 | 省去 frame.copy() + cv2.imshow | 摄像头读取循环 |
| Canvas UI 替代 Sprite | 直接屏幕像素映射，无纹理上传 | Unity HandTrackingVisualizer |
| 预创建对象池 | 消除每帧 GC 内存抖动 | Unity HandTrackingVisualizer |

### 4.2 失败处理策略（PPT 表格）

| 故障场景 | 检测方式 | 降级策略 |
|---------|---------|---------|
| 云端 LLM 不可用 | API 返回错误/超时 | 模板填充 + 预设文案 |
| 图像生成超时 | Wan 2.7 任务超时 2 分钟 | 显示"画作生成中，请稍候" |
| ONNX 模型缺失 | 文件不存在/加载异常 | 自动切换 HeuristicPredictor |
| Unity TCP 断开 | select 超时/is_running=False | 等待重连，不阻塞主循环 |
| 知识库实体缺失 | get_entity() 返回 None | 使用实体名作为描述 |

---

## ★ PPT 章节五：关键文件索引

### 5.1 文件树（PPT 图表）

```
D:\projects\ohhh\
├── vision/                          # 视觉输入
│   ├── color_card_detector.py        # YOLO 颜色牌检测（接口就绪）
│   ├── hand_detector.py             # MediaPipe 手部检测
│   ├── hand_tracker.py              # 手部追踪封装
│   ├── gesture_state_machine.py     # 5 模式手势 FSM
│   ├── sketch_recognizer.py        # QuickDraw CNN 草图识别
│   └── quickdraw/                   # CNN 训练/数据/模型
│       ├── model.py                 # MobileNet 模型定义
│       ├── train.py                 # 训练脚本
│       └── quickdraw_mobilenet.onnx # 82 类 ONNX 模型 (2.5MB)
│
├── rag/                             # RAG 检索生成
│   ├── retriever.py                 # 知识库检索引擎
│   ├── generator.py                  # LLM 生成（阿里云百炼）
│   ├── character_recommend.py       # 人物推荐（四维打分）
│   ├── postcard.py                  # 明信片合成（PIL）
│   └── knowledge/                   # 知识库 JSON
│       ├── entities/                # 208 实体
│       ├── combinations/            # 107 组合解读
│       └── templates/               # 100 叙事模板
│
├── unity_bridge/                    # Unity 通信
│   ├── server.py                    # :8888/:8889 双端口 TCP
│   ├── sender.py                    # UnitySender 统一发送
│   ├── sketch_bridge.py             # 草图→物象候选
│   └── character_bridge.py          # RAG→人物推荐
│
├── test_integrated.py               # 端到端集成测试
└── config_ipcam.py                  # IP 摄像头配置
```

### 5.2 技术栈总结（PPT 表格）

| 层级 | 技术选型 | 版本/规格 |
|------|---------|----------|
| 视觉检测 | YOLOv8n + MediaPipe | YOLOv8n (~3M) / MediaPipe (Google) |
| 草图识别 | QuickDraw MobileNet ONNX | 82 类 / ~635K 参数 / 83.8% acc |
| 手势追踪 | MediaPipe Hand Landmarker | 21 关键点 / 30fps |
| LLM 生成 | 阿里云百炼 qwen-plus | qwen-turbo(实时) / qwen-plus(叙事) |
| 图像生成 | Wan 2.7 image-pro | wan2.7-image-pro (异步) |
| 知识库 | 本地 JSON | 208 实体 + 107 组合 |
| 通信协议 | TCP 双端口 | :8888(主通道) / :8889(手部) |
| 前端渲染 | Unity 2022+ | Canvas UI + TCP Client |
| 图像处理 | Pillow | 明信片动态布局 |

---

## 二、模块一：视觉输入

### 2.1 颜色牌检测（YOLOv8n + PiDiNet 双识别）

第一幕用户将物理颜色牌放置于桌面，摄像头俯拍检测类型和位置。

展览环境光照暗，**仅靠颜色识别不可靠**，采用**颜色+纹路双保险**方案。

**识别管线图**：
```
摄像头帧 (640×480)
      │
      ▼
┌─────────────────┐
│  YOLOv8n 检测     │  ← 定位颜色牌区域（bbox）
│  (颜色牌定位)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   裁剪卡片区域    │
└────────┬────────┘
         │
         ├────────────────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│   颜色识别器      │   │   纹路识别器      │
│  HSV直方图匹配   │   │ PiDiNet+模板匹配 │
└────────┬────────┘   └────────┬────────┘
         │                     │
         │ confidence_A         │ confidence_B
         └──────────┬────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   决策融合     │
            │ color × 0.4   │
            │  + edge × 0.6 │
            │  = 最终结果    │
            └───────────────┘
```

| 项目 | 内容 |
|------|------|
| **检测目标** | 6 类：岳麓绿 / 书院红 / 西迁黄 / 湘江蓝 / 校徽金 / 墨色 |
| **定位模型** | YOLOv8n (~3M 参数) + BoT-SORT 跟踪 |
| **颜色识别** | HSV 颜色直方图匹配（受光照影响大，作为辅助） |
| **纹路识别** | PiDiNet 边缘检测 + 边缘图模板匹配（核心识别，不受光照影响） |
| **融合权重** | 颜色 0.4 + 纹路 0.6（纹路更可靠） |
| **输出** | class_id (0-5)、像素中心坐标、bbox、track_id、置信度 |
| **文件** | `vision/color_card_detector.py`（颜色+纹路双识别） |

**各颜色牌预期纹路特征**：

| 颜色牌 | 纹路特征 |
|--------|---------|
| 岳麓绿 | 竖条纹/松针纹理 |
| 书院红 | 横向砖纹/瓦片 |
| 西迁黄 | 折线条纹/地图纹理 |
| 湘江蓝 | 波浪纹理/水纹 |
| 校徽金 | 光泽渐变/徽章边缘 |
| 墨色 | 泼墨/浓淡渐变 |

### 2.2 手势识别（MediaPipe）

全程手势驱动，MediaPipe 逐帧追踪手部 21 个关键点。

```
摄像头帧  →  MediaPipe Hand Landmarker  →  21 关键点 (x,y,z)
                                          →  手势分类：食指伸出 / 握拳 / 张手
                                          →  手势状态机 (5 模式)
```

| 项目 | 内容 |
|------|------|
| **模型** | MediaPipe Hand Landmarker (Google) |
| **帧率** | 30fps（摄像头）/ 无上限（no-display 模式） |
| **文件** | `vision/hand_detector.py`, `vision/hand_tracker.py`, `vision/gesture_state_machine.py` |

### 2.3 手势状态机（5 模式 FSM）

| 模式 | 子状态 | 触发手势 | 行为 |
|------|--------|---------|------|
| **GLOBAL** | IDLE | — | 全局空闲，检测握拳晕染/张手停止 |
| **DRAWING** | TRACKING → COMPLETED / CANCELLED | 食指伸出→握拳提交/张手取消 | 追踪指尖轨迹 → 送入草图识别 |
| **CANDIDATE** | BROWSING → CONFIRMED / CANCELLED | 悬停+握拳确认/张手取消 | Top-3 物象选择 |
| **CHAR_RECOMMEND** | BROWSING → CONFIRMED / TO_WHEEL | 握拳确认/张手拒绝 | 人物推荐确认或进入轮盘 |
| **CHAR_WHEEL** | SCROLLING → PREVIEWING → CONFIRMED / TO_RECOMMEND | 水平滑动+悬停+握拳 | 轮盘浏览选择人物 |

### 2.4 草图识别（QuickDraw CNN）

第二幕用户食指绘画，轨迹栅格化后经 CNN 分类为物象。

```
食指指尖轨迹 (landmark 8 点序列)
       ↓
轨迹归一化 → 栅格化 28×28 灰度图（笔画宽度 2px，抗锯齿）
       ↓
CNN 推理：QuickDraw MobileNet（82 类，ONNX 本地推理）
  · 降级方案：HeuristicPredictor（9 种几何特征，模型不可用时自动切换）
       ↓
QuickDraw 类别 → 88 物象映射表
       ↓
颜色上下文加权（第一幕颜色对候选物象加权/降权）
       ↓
Top-3 候选物象 → Unity
```

| 属性 | 值 |
|------|-----|
| 架构 | 2层CNN（Conv 5×5, 32→64）+ 2层FC（512→128）+ 输出层 |
| 输入/输出 | `(1,28,28)` → `(82,)` logits |
| 参数量 | ~635K，ONNX 2.5MB |
| 验证准确率 | **83.8%**（82 类，RTX 4060 训练 ~10 分钟） |
| 文件 | `vision/sketch_recognizer.py`, `vision/quickdraw/` |

---

## 三、模块二：RAG 检索增强生成

**RAG = Retrieval-Augmented Generation**：先从知识库检索相关文化内容，再交由 LLM 基于检索结果生成，确保准确性和个性化。

```
用户选择（颜色 + 物象 + 人物 + 连接）
       ↓
┌──────────────────────────────────────────────┐
│  知识库（208 实体 + 107 组合 + 100 模板）      │
│       ↓                                       │
│  检索引擎（精确匹配 + 双向查找）                │
│       ↓                                       │
│  阿里云百炼（qwen-turbo / qwen-plus）          │
│       ↓                                       │
│  输出分层：                                    │
│    实时侧 → Unity 即时显示（元素描述、连接关系） │
│    高质量侧 → 云端 API 生成最终画作             │
└──────────────────────────────────────────────┘
```

### 3.1 知识库

| 类别 | 数量 | 格式 |
|------|------|------|
| 颜色实体 | 23 条 | `type, description, color(hex), symbolism, related_entities, mood, era, theme` |
| 物象典故 | 88 条 | `type, description, symbolism, related_entities, historical_context` |
| 人物故事 | 97 条 | `type, name, years, title, description, spirit, related_entities, quotes, stories` |
| 组合解读 | 107 条 | `entity1, entity2, meaning, interpretation`（文件名即查询 key） |
| 叙事模板 | 100 句 | 开头30 + 场景30 + 人物20 + 结尾20 |

> 构建工具：`rag/build_knowledge.py`（txt → json 自动化）

### 3.2 检索引擎

精确匹配 + 双向查找，从知识库检索颜色、物象、人物、组合解读、叙事模板。BM25 模糊检索标记为 P2（非必需）。

### 3.3 LLM 生成

| 函数 | 模型 | 说明 |
|------|------|------|
| `generate_realtime_description()` | qwen-turbo | 实时单模块描述 |
| `generate_connection_description()` | qwen-turbo | 连接关系描述 |
| `generate_narrative()` | qwen-plus | 完整叙事卡（模板+LLM融合） |
| `generate_image_prompt()` | qwen-turbo | 画作英文提示词 |
| `generate_image()` | wan2.7-image-pro | 文生图（Wan 2.7 异步） |
| `generate_image_with_base()` | wan2.7-image-pro | 图生图：颜色晕染底图 + 人物风格 prompt |
| `build_character_style_prompt()` | — | 人物→视觉风格 prompt 构建 |
| `create_postcard()` | — | 明信片合成（PIL 动态布局） |

**失败处理**：LLM 不稳定 → 降级模板填充；云端 API 超时 → "画作生成中，请稍候"

### 3.4 人物推荐引擎

```
颜色 + 物象 + 已选人物
       ↓
四维打分（纯本地，无依赖）：
  (1) 内置参考表命中 (0~0.50)
  (2) 已选人物同组加权 (0~0.20)
  (3) 关键词文本匹配 (0~0.25)
  (4) 实体基础分 (0~0.05)
       ↓
降序 → Top-15 → [可选] LLM 精选 → Top-3 推荐
```

| 项目 | 内容 |
|------|------|
| 核心人物 | 54 人，6 组（理学脉络12/湘军将帅4/维新革命6/现代学人12/校园角色13/抽象意象9） |
| 文件 | `rag/character_recommend.py` |

### 3.5 人物视觉风格映射

基于人物分组自动确定视觉风格，用于图生图的风格 prompt 构建。

```
人物名 → 分组查询 → 时代判定 → 视觉风格 prompt
```

| 分组 | 时代 | 风格 | 英文风格关键词 | 示例人物 |
|------|------|------|---------------|----------|
| 理学脉络 | 古代 | **古风** | Song dynasty ink wash, sparse brushwork, rice paper, Zen | 朱熹、张栻、王夫之、王阳明 |
| 湘军将帅 | 近代 | **写实** | Qing dynasty photography, sepia, documentary gravitas | 曾国藩、左宗棠 |
| 维新革命 | 近代 | **写实** | Revolutionary woodcut print, bold contrasts, heroic | 谭嗣同、魏源、黄兴 |
| 现代学人 | 现代 | **写实** | Academic realism, natural lighting, intellectual warmth | 毛泽东、杨昌济 |
| 校园角色 | 现代 | **写实** | Campus slice of life, warm sunlight, nostalgic | 学子、教师、留学生 |
| 抽象意象 | 跨时代 | **穿越风** | Ink dissolving into digital, double exposure, dreamlike | 理学之魂、书院守望者 |

> 映射表：`rag/generator.py` → `CHARACTER_GROUP_STYLE`，查询函数 `get_character_style()`

### 3.6 图生图管线

颜色晕染底图由 Unity 前端渲染提供（用户选色后实时生成水墨渐变纹理），Python 端负责风格融合。

```
Unity 前端                        Python 后端                      阿里云 API
─────────                      ──────────                      ─────────
用户选色 → 颜色晕染渲染
         │
         ├── Base64 底图 ────────→ generate_image_with_base()
         │                              │
用户选人物 ──→ build_character_style_prompt()
         │                              │
         │    古风/写实/穿越风 prompt ──→ Wan 2.7 图生图
         │                              │
         │                         ← 融合图像
         │                              │
         └──────────→ create_postcard() ←
                           │
                     明信片合成 → Unity 展示
```

| 步骤 | 输入 | 模型/方式 | 输出 |
|------|------|-----------|------|
| 颜色晕染 | 颜色名 + hex | Unity 前端实时渲染 | 水墨渐变纹理 (Base64) |
| 风格 prompt | 人物名 → 分组 → 风格 | `get_character_style()` | 英文视觉风格 prompt |
| 图生图融合 | 底图 + 风格 prompt | `wan2.7-image-pro`（异步） | 融合图像 URL + 本地下载 |
| 明信片合成 | 融合图像 + 叙事文本 | PIL 动态布局 | 完整叙事卡 |

> 文件：`rag/generator.py` → `generate_image_with_base()`, `build_character_style_prompt()`

### 3.7 生成质量改进路线图

基于《智能设计方法3-智能生成》课程内容，在 Prompt Engineering、RAG、Role-Playing、Agent、Hallucination 五个维度规划改进。

| # | 改进项 | 优先级 | 原理 | 改动位置 | 状态 |
|---|--------|--------|------|----------|------|
| 1 | **Few-shot ICL** | P0 | Prompt 中加入 1-2 个"输入→高质量叙事"范例，利用 In-Context Learning 让模型理解期望输出风格和质量标准 | `generator.py:generate_narrative()` | ✅ 已实现 |
| 2 | **Hallucination 缓解** | P0 | Prompt 中明确列出知识库关键事实并约束"必须据此创作，不得编造"；生成后提取人名/年代/地点与知识库做字符串匹配，标记潜在幻觉 | `generator.py:generate_narrative()` + `_verify_hallucination()` | ✅ 已实现 |
| 3 | **Self-Reflection 迭代** | P1 | Agent 循环：生成初稿 → LLM 切换为"评审"角色从文学性/事实准确性/情感共鸣打分 → 低于阈值则带评审意见重新生成 | `generator.py` 新增 `_self_review()` + `_revise()` | 🔸 待实现 |
| 4 | **Persona Card 深化** | P1 | 构建完整角色卡（身份、语言风格、知识范围、情感倾向、禁忌），替代当前一句话角色提示，提升叙事一致性和角色代入感 | `generator.py` 新增 `PERSONA_CARD` 常量 | 🔸 待实现 |
| 5 | **RAG 检索增强** | P2 | 查询扩展（颜色关联概念扩展检索）、Cross-encoder 重排序（精排检索结果），提升检索召回率和精度 | `retriever.py` 新增 `expand_query()` + `rerank()` | 🔸 待实现 |
| 6 | **Temperature 分阶段调参** | P2 | 事实提取 0.1 / 风格 prompt 0.6 / 叙事创作 0.8-0.9 / 自评审 0.2，替代当前全局固定值 | `GenerationConfig` 新增分阶段字段 | 🔸 待实现 |
| 7 | **JSON 修复层** | P2 | 生成 JSON 解析失败时，用轻量模型（qwen-turbo）对原始文本做二次格式化，替代当前直接丢弃 | `generator.py:_call_model()` 返回后处理 | 🔸 待实现 |
| 8 | **多模态闭环校验** | P3 | Wan 2.7 生成图像后，用 Qwen-VL 描述图像内容，对比叙事主题一致性，偏差大则调整 prompt 重新生成 | `generator.py` 新增 `_verify_image_narrative_consistency()` | 🔸 待实现 |

**各改进项关联课程知识点**：

| # | 对应课程章节 |
|---|-------------|
| 1 | In-Context Learning（few-shot 示例优于纯指令） |
| 2 | Hallucination 四类成因及缓解策略；RAG 作为主要缓解手段 |
| 3 | Agent 架构：Perception → Planning → Tool → Reflection 循环 |
| 4 | Role-Playing（Persona Card、CharacterEval 评估框架） |
| 5 | RAG 进阶技术：查询扩展、重排序、混合检索 |
| 6 | Temperature 参数原理（课程代码实践部分） |
| 7 | Instruction Tuning + 结构化输出约束 |
| 8 | Describe Anything（ICCV 2025）+ 文→图→文一致性校验 |

> 来源：《智能设计方法3-智能生成》课程 + 当前项目 `rag/` 模块代码审查

---

## 四、模块三：Unity 通信与渲染

### 4.1 通信架构

双端口 TCP，JSON + `\n` 分隔：

| 端口 | 通道 | 频率 | 内容 |
|------|------|------|------|
| **:8888** | 主通道 | 事件驱动 | 候选物象、推荐人物、手势状态、确认/拒绝 |
| **:8889** | 手部通道 | ~30fps | 21 关键点 + 5 指尖 + 掌心坐标 |

Python 端入口：`test_integrated.py`（串联摄像头→手部→FSM→Bridge→TCP 全链路）

### 4.2 消息类型

**Python → Unity**：

| 消息 | 内容 |
|------|------|
| `object_candidates` | Top-3 物象候选（名称、分数、QuickDraw 类别） |
| `character_candidates` | Top-3 人物推荐（名称、称号、分数、推荐理由） |
| `gesture_state` | 当前 FSM 模式 + 子状态 + 手势类型 |
| `hand_tracking` | 21 个关键点像素坐标 + 指尖 + 掌心（每帧） |

**Unity → Python**：

| 消息 | 内容 |
|------|------|
| `object_selected` | 用户确认选中的物象名 |
| `character_selected` | 用户确认选中的人物名 |
| `wheel_group_changed` | 轮盘分组切换 |
| `wheel_character_selected` | 轮盘选中人物 |
| `generation_start` | 触发最终叙事生成 |

### 4.3 Bridge 层

| Bridge | 文件 | 职责 |
|--------|------|------|
| ColorCardBridge | （待模型训练后新建） | YOLO 检测 → Unity 颜色牌消息 |
| SketchBridge | `unity_bridge/sketch_bridge.py` | CNN 分类 → Top-3 物象候选 |
| CharacterBridge | `unity_bridge/character_bridge.py` | RAG 推荐 → Top-3 人物推荐 |

### 4.4 Unity 端组件

| 组件 | 功能 |
|------|------|
| `PythonConnection` | :8888 主通道 TCP 客户端，消息路由分发 |
| `HandTrackingConnection` | :8889 手部通道 TCP 客户端 |
| `HandTrackingVisualizer` | Canvas UI 实时渲染 21 关键点 + 骨架 + 指尖 + 掌心 |
| `ObjectCandidateUI` | Top-3 物象候选卡片（悬停高亮 + 点击确认） |
| `CharacterCandidateUI` | Top-3 人物推荐卡片（名称+称号+理由+得分条） |
| `GestureStateUI` | 右上角 FSM 模式指示器 |
| `SceneSetup` | 一键自动创建所有通信和 UI 组件 |

### 4.5 延迟优化

| 措施 | 效果 |
|------|------|
| TCP_NODELAY（禁用 Nagle） | 消除 ~200ms 缓冲延迟 |
| 摄像头缓冲排空（每帧 4 次 read） | 消除 2-3 帧旧数据堆积 |
| no-display 模式 | 省去 frame.copy() + cv2.imshow |
| Canvas UI（替代 Sprite） | 直接屏幕像素映射 |
| 预创建对象池 | 消除每帧 GC 内存抖动 |

---

## 五、实现状态

### 5.1 模块状态

| 模块 | 子模块 | 状态 |
|------|--------|------|
| **视觉输入** | YOLO 颜色牌检测 | 🔸 接口就绪，待训练 |
| | MediaPipe 手部追踪 | ✅ |
| | 手势状态机（5 模式 FSM） | ✅ |
| | QuickDraw CNN 草图识别 | ✅ val acc 83.8% |
| | IP 摄像头连接 | ✅ |
| **RAG 内容生成** | 知识库（208 实体 + 107 组合 + 100 模板） | ✅ |
| | 检索引擎（精确+双向） | ✅ |
| | LLM 生成（qwen-turbo / qwen-plus） | ✅ |
| | 云端生图（wan2.7-image-pro，文生图+图生图） | ✅ |
| | 人物视觉风格映射（6 组 → 3 风格） | ✅ |
| | 图生图管线（颜色晕染底图 + 人物风格融合） | ✅ |
| | 人物推荐引擎（54 人，四维打分） | ✅ |
| | 明信片合成（动态布局） | ✅ |
| **Unity 通信** | 双端口 TCP 服务器 | ✅ |
| | UnitySender 统一发送器 | ✅ |
| | Bridge 层（sketch / character） | ✅ |
| | Unity C# 消息路由 + UI 组件 | ✅ |
| | 轮盘浏览 | ⏸️ 暂缓 |

### 5.2 完成度

| 模块 | 总项 | ✅ | 🔸 | ⏸️ | 完成度 |
|------|------|----|----|----|--------|
| 视觉输入 | 5 | 4 | 1 | 0 | **80%** |
| RAG 内容生成 | 6 | 6 | 0 | 0 | **100%** |
| Unity 通信与渲染 | 5 | 4 | 0 | 1 | **80%** |
| **总计** | **16** | **14** | **1** | **1** | **~88%** |

> 颜色牌 YOLO 模型训练后（1-2 天工作量：采集+标注+训练），视觉输入模块可达 100%，整体完成度 ~94%。
> 轮盘为 P2 暂缓项，不影响四幕叙事主流程。

### 5.3 端到端可运行能力

| 能力 | 状态 |
|------|------|
| 摄像头 → MediaPipe → FSM → SketchBridge → Unity 物象候选 | ✅ |
| 摄像头 → MediaPipe → FSM → CharacterBridge → Unity 人物推荐 | ✅ |
| 颜色牌检测 → Unity 消息推送 | 🔸 mock 模式可跑 |
| RAG 检索 → LLM 叙事 → Wan 2.7 文生图/图生图 → 明信片合成 | ✅ |
| 人物推荐 → 风格映射 → 颜色晕染底图 → 图生图融合 | ✅ |
| 手部 21 关键点 → Unity 实时可视化 | ✅ |

---

## 六、附录

### 6.1 已解决的 Bug

| # | 问题 | 修复 |
|---|------|------|
| 1 | 文本生成输出为空（读错字段路径） | `rag/generator.py`: 优先取 `choices[0].message.content` |
| 2 | 图像模型名错误（`qwen-image-2.0-pro` 不存在） | 改为 `wanx-v1`，支持 `prompt_extend` |
| 3 | 明信片排版严重错乱（图片被挤压/文字不换行/画布不足） | `rag/postcard.py` 完全重写布局引擎 |
| 4 | API Key 未传递到生成器 | `create_generator()` 自动读环境变量 |
| 5 | 集成测试时间戳非单调递增 | 每帧只调一次 `_detect()`，复用结果 |
| 6 | Unity 始终连不上（`is_running` 时序错误） | 移动 `is_running = True` 到 `_start_servers()` 之前 |
| 7 | gesture_state 消息静默丢失（超时断开） | `select.select()` 隔离收发超时 |
| 8 | Unity 手部数据解析失败（字段名不匹配） | 重写 `HandTrackingData` 对齐 Python 格式 |
| 9 | 延迟高 | 双方 TCP_NODELAY + 缓冲排空 + no-display 模式 |
| 10 | 图像生成需支持图生图（颜色晕染 + 人物风格） | 切换 `wan2.7-image-pro`，新增人物风格映射（6组→3风格），新增 `generate_image_with_base()` 图生图接口 |

### 6.2 待处理

| 优先级 | 事项 | 说明 |
|--------|------|------|
| **P0** | YOLO 颜色牌数据采集+训练 | 6 类颜色牌，~100-200 张/类，标注后训练 |
| P1 | 字体 zip 解压 | `assets/fonts/SourceHanSansSC.zip` → 优先思源字体 |
| P2 | 多场景覆盖测试 | 不同颜色+物象+人物组合的生成效果 |
| P2 | BM25 模糊检索 | 当前精确匹配，缺模糊检索能力 |
| P2 | 人物轮盘 | Phase 4，非主线暂缓 |

### 6.3 关键文件索引

```
vision/
├── color_card_detector.py      # YOLO 颜色牌检测（接口就绪）
├── hand_detector.py            # MediaPipe 手部检测
├── hand_tracker.py             # 手部追踪封装
├── gesture_state_machine.py    # 5 模式手势 FSM
├── sketch_recognizer.py        # QuickDraw CNN 草图识别
├── quickdraw/                  # CNN 训练/数据下载/模型定义
│   ├── train.py, model.py, dataset.py, config.py
│   └── download_data.py
└── models/
    └── quickdraw_mobilenet.onnx # 82 类 ONNX 模型 (2.5MB)

rag/
├── retriever.py                # BM25 + 精确匹配检索引擎
├── generator.py                # LLM 生成（阿里云百炼）
├── character_recommend.py      # 人物推荐引擎（四维打分）
├── build_knowledge.py          # txt → json 知识库构建
├── postcard.py                 # 明信片合成（PIL 动态布局）
└── knowledge/                  # 知识库 JSON
    ├── entities/               # colors(23) + objects(88) + characters(97)
    ├── combinations/           # 107 条组合解读
    └── templates/              # 100 句叙事模板

unity_bridge/
├── server.py                   # :8888/:8889 双端口 TCP 服务器
├── sender.py                   # UnitySender 统一消息发送器
├── sketch_bridge.py            # 草图 → 物象候选桥接
└── character_bridge.py         # RAG → 人物推荐桥接

test_integrated.py              # 端到端集成测试入口
```

---

> 湖南大学设计艺术学院 · 智能设计方法 · 2026
