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
          │  │  YOLO 颜色牌 │  │ MediaPipe │  │  手势状态机 │ │
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
          │  │   知识库检索 → LLM 归因 → 叙事 + 生图      │  │
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

**数据流**：摄像头帧 → 颜色牌检测 + 手部追踪 → 手势状态机 → 草图识别/人物推荐 → RAG 叙事生成 → 双端口 TCP → Unity 渲染

---

## 二、模块一：视觉输入

### 2.1 颜色牌检测（YOLOv8n）

第一幕用户将物理颜色牌放置于桌面，摄像头俯拍检测类型和位置。

```
摄像头帧  →  YOLOv8n 推理  →  BoT-SORT 跟踪  →  类型 + 坐标 + track_id
```

| 项目 | 内容 |
|------|------|
| **检测目标** | 6 类：岳麓绿 / 书院红 / 西迁黄 / 湘江蓝 / 校徽金 / 墨色 |
| **模型** | YOLOv8n (~3M 参数) + Ultralytics BoT-SORT |
| **输出** | class_id (0-5)、像素中心坐标、bbox、track_id、置信度 |
| **坐标映射** | 摄像头 640×480 → 线性映射 → Unity 1920×1080 |
| **文件** | `vision/color_card_detector.py`（接口就绪，mock 模式可跑通，待采集数据训练） |

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
| `generate_image_prompt()` | qwen-turbo | 画作提示词（失败降级模板） |
| `generate_image()` | wanx-v1 | 云端生成水墨画 |
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
| | 云端生图（wanx-v1） | ✅ |
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
| RAG 检索 → LLM 叙事 → wanx-v1 生图 → 明信片合成 | ✅ |
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
