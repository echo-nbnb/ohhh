# RAG 模块开发指南

## 概述

RAG（检索增强生成）模块负责根据模块组合生成湖大千年色叙事体验。

## 架构

```
Unity触摸事件 → Python RAG系统 → Unity即时反馈
                                    ↓
                             云端API生成叙事

模块放置 → 本地RAG检索 → 小模型润色 → Unity节点显示
                                                    ↓
模块组合完成 → 上下文构建 → 云端大模型 → 完整图文叙事
```

## 目录结构

```
rag/
├── __init__.py           # RAG系统主入口
├── retriever.py          # 检索模块
├── generator.py          # 生成模块
├── knowledge/            # 知识库
│   ├── entities/         # 实体定义
│   │   ├── colors.json    # 颜色实体
│   │   ├── objects.json   # 物象实体
│   │   └── characters.json # 人物实体
│   ├── combinations/    # 组合解读
│   │   └── green_academy.json  # 示例：绿+书院
│   └── templates/        # 叙事模板
│       └── final_portrait.json
└── RAG_README.md       # 本文档
```

## 快速开始

```python
from rag import RAGSystem

# 初始化
rag = RAGSystem()
rag.setup()

# 注册模块（需要根据实际ESP32 ID填充）
rag.register_module("esp32_001", "color", "岳麓绿")
rag.register_module("esp32_002", "color", "书院红")
rag.register_module("esp32_003", "object", "书院")
rag.register_module("esp32_004", "character", "张栻")

# 放置模块时检索（实时侧）
result = rag.retrieve_realtime("esp32_001")
# 返回: {entity, description, connections, unity_data}

# 添加连接
rag.add_connection("esp32_001", "esp32_003")

# 开始生成时构建上下文（高质量侧）
context = rag.build_generation_context()
cloud_data = rag.prepare_for_cloud(context)
# 返回: {modules, connections, cloud_prompt}
```

## 知识库填充

### 1. 实体定义

编辑 `knowledge/entities/` 下的JSON文件：

**colors.json** 示例：
```json
{
  "岳麓绿": {
    "type": "color",
    "description": "绿色，生命的颜色",
    "color": "#2E7D32",
    "symbolism": "生长、传承、根脉",
    "related_entities": ["书院", "古树"],
    "historical_context": "岳麓书院千年古木，象征学问传承"
  }
}
```

**objects.json** 示例：
```json
{
  "书院": {
    "type": "object",
    "description": "千年学府，读书讲学之地",
    "symbolism": "求知、传承、讲学",
    "related_entities": ["岳麓绿", "张栻"],
    "historical_context": "岳麓书院，创建于976年"
  }
}
```

**characters.json** 示例：
```json
{
  "张栻": {
    "type": "character",
    "name": "张栻",
    "years": "1133-1180",
    "title": "岳麓书院山长",
    "spirit": "传道济民",
    "quotes": ["传道济民"],
    "stories": ["1167年朱熹来访会讲"]
  }
}
```

### 2. 组合解读

创建 `knowledge/combinations/green_academy.json`：
```json
{
  "from": "岳麓绿",
  "to": "书院",
  "meaning": "绿荫掩映，书声琅琅，真理在其中流转",
  "scene_description": "古建筑的绿色光影，树影婆娑"
}
```

### 3. 连接类型与样式

| 连接类型 | 模块组合 | 样式 |
|---------|---------|------|
| `spiritual_resonance` | 颜色↔颜色 | 金色虚线 |
| `color_grant` | 颜色↔物象 | 物象对应颜色 |
| `personality_dye` | 人物↔颜色 | 人物对应颜色 |
| `scene_entry` | 人物↔物象 | 白色实线 |

## 待填充内容

| 项目 | 状态 | 说明 |
|------|------|------|
| 模块映射表 | ❌ | 12个ESP32 ID → 文化实体 |
| 颜色知识库 | ❌ | 6种颜色的详细描述 |
| 物象知识库 | ❌ | 物象典故 |
| 人物知识库 | ❌ | 人物故事 |
| 组合解读 | ❌ | 模块组合的文化含义 |
| 本地小模型 | ❌ | Ollama部署 |
| 云端API | ❌ | 待定 |

## 开发日志

- 2026-04-26: 框架搭建完成，保留接口待填充
