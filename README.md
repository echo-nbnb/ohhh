# 寻麓千年色 · 智能交互装置

> 湖南大学设计艺术学院 · 智能设计方法课程 · 大二下学期

---

## 项目概述

**寻麓千年色** 是一个以湖湘文化为底蕴的 **AI 驱动手势交互装置**。

用户全程通过**手部动作**与投影画面交互——摄像头实时捕捉手部 21 个关键点，AI 识别手势意图（绘画、选择、确认），驱动四幕叙事体验。从选择颜色、手势绘画物象，到 AI 推荐历史人物、生成个性化精神画像，形成完整的沉浸式文化叙事体验。

**"寻麓千年色"** —— 不是在找颜色，而是在寻找湖大千年精神中属于"我"的那一抹色彩。

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           摄像头（IP Camera / USB）                       │
│                         俯拍桌面：颜色牌 + 手部                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  YOLOv8n + 双识别 │  │   MediaPipe     │  │   QuickDraw     │
│  颜色牌检测      │  │   手部追踪      │  │   CNN 草图识别   │
│  (颜色+纹路)    │  │   21关键点/30fps │  │   (82类/ONNX)  │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Bridge 层         │
                    │  color_card / sketch │
                    │  / character         │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  RAG 检索生成引擎     │
                    │  知识库 + LLM 生成   │
                    │  Wan 2.7 图生图     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  双端口 TCP 服务器    │
                    │  :8888 主通道        │
                    │  :8889 手部通道      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │      Unity 渲染端     │
                    │  实时渲染 + 投影     │
                    └─────────────────────┘
```

---

## 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [四幕叙事流程](#四幕叙事流程)
- [手势交互](#手势交互)
- [技术模块详解](#技术模块详解)
- [训练教程](#训练教程)
- [常见问题](#常见问题)

---

## 快速开始

### 环境配置

```bash
# 1. 创建并激活环境
conda create -n ohhh python=3.12
conda activate ohhh

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 PyTorch（用于 PiDiNet 边缘检测）
pip install torch torchvision
```

### 运行测试

```bash
# 端到端集成测试（需要 IP 摄像头）
python test_integrated.py

# 快速测试（无摄像头，mock 模式）
python test_integrated_fast.py

# 测试图像生成（无需摄像头）
python test_generation_full.py
```

### 配置摄像头

编辑 `config_ipcam.py`：

```python
CAMERA_URL = "http://你的摄像头IP:8080/video"
```

---

## 项目结构

```
ohhh/
├── vision/                          # 视觉模块
│   ├── color_card_detector.py       # 颜色牌检测（颜色+纹路双识别）
│   ├── hand_detector.py             # MediaPipe 手部检测
│   ├── hand_tracker.py              # 手部追踪封装
│   ├── gesture_state_machine.py     # 手势状态机（5模式）
│   ├── sketch_recognizer.py        # QuickDraw 草图识别
│   ├── edge_detection_ipcam.py     # PiDiNet 边缘检测
│   ├── ipcamera.py                 # IP 摄像头连接
│   └── quickdraw/                   # QuickDraw CNN 模型
│
├── rag/                             # RAG 检索生成
│   ├── retriever.py                # 知识库检索引擎
│   ├── generator.py                 # LLM 叙事生成（阿里云百炼）
│   ├── character_recommend.py       # 人物推荐引擎
│   ├── postcard.py                  # 明信片合成
│   └── knowledge/                   # 知识库
│       ├── entities/                # 208 实体（颜色23/物象88/人物97）
│       ├── combinations/            # 107 组合解读
│       └── templates/               # 100 叙事模板
│
├── unity_bridge/                    # Unity 通信
│   ├── server.py                    # 双端口 TCP 服务器
│   ├── sender.py                    # 数据发送器
│   ├── sketch_bridge.py            # 草图→物象桥接
│   └── character_bridge.py          # 人物推荐桥接
│
├── yolo/                           # YOLO 颜色牌检测
│   ├── dataset.yaml                # 数据集配置
│   └── train_yolo.py               # 训练脚本
│
├── proposal/                        # 设计文档
│   ├── interaction.md              # 交互设计详述
│   └── technical.md                # 技术分析与实现
│
├── test_integrated.py              # 端到端集成测试
├── test_generation_full.py         # 图像生成测试
└── config_ipcam.py                 # 摄像头配置
```

---

## 四幕叙事流程

```
开场 → 第一幕:择色 → 第二幕:筑景 → 第三幕:唤灵 → 第四幕:成色
```

### 开场引入
全黑缓缓亮起，湖大校门水墨剪影浮现。泛黄邀请函飘落："岳麓千年，色隐其中。后来者，汝心何色？" AI 旁白欢迎玩家。

### 第一幕：择色
用户将物理颜色牌放置桌面。YOLO+双识别检测类型和位置，Unity 渲染光圈绽放。

| 颜色牌 | 颜色名 | 预期纹路 |
|--------|--------|---------|
| 绿色牌 | 岳麓绿 | 竖条纹/松针纹理 |
| 红色牌 | 书院红 | 横向砖纹/瓦片 |
| 黄色牌 | 西迁黄 | 折线条纹/地图纹理 |
| 蓝色牌 | 湘江蓝 | 波浪纹理/水纹 |
| 金色牌 | 校徽金 | 光泽渐变/徽章边缘 |
| 黑色牌 | 墨色 | 泼墨/浓淡渐变 |

### 第二幕：筑景
食指伸出绘画，指尖光点轨迹实时渲染。握拳提交后 CNN 识别为物象，颜色上下文加权后 Top-3 候选浮现。悬停+握拳确认。

### 第三幕：唤灵
RAG 检索推荐 Top-3 历史人物。悬停预览简介，握拳确认选中。AI 以人物第一人称独白。

### 第四幕：成色
自由调整后确认生成。RAG 检索 + LLM 叙事 + Wan 2.7 图生图 → 个性化明信片。

---

## 手势交互

### 全局手势（任意阶段）

| 手势 | 动作 | 效果 |
|------|------|------|
| 握拳挥动 | 五指握拳 + 手部移动 | 颜色晕染扩散 |
| 五指张开 | 五指伸展 | 停止晕染/取消 |

### 绘画手势（第二幕）

| 手势 | 动作 | 效果 |
|------|------|------|
| 食指伸出 | 仅食指伸展 | 进入绘画，指尖为笔 |
| 握拳 | 五指握起 | 结束绘画/确认物象 |
| 五指张开 | 五指伸展 | 取消重画 |

### 选择手势（第二/三幕）

| 手势 | 动作 | 效果 |
|------|------|------|
| 手悬停 | 掌心在目标区域 | 高亮放大 |
| 握拳 | 悬停+握拳 | 确认选中 |
| 五指张开 | 选中后伸展 | 取消返回 |

### 轮盘手势（第三幕轮盘模式）

| 手势 | 动作 | 效果 |
|------|------|------|
| 手左右滑动 | 手掌水平移动 | 轮盘滚动 |
| 手悬停 > 1秒 | 静止 | 放大预览 |
| 握拳 | 悬停+握拳 | 确认 |
| 五指张开 | 伸展 | 退出轮盘 |

---

## 技术模块详解

### 颜色牌检测（颜色+纹路双保险）

展览光照暗，**仅靠颜色不可靠**，采用双识别方案：

```
摄像头帧 → YOLO 定位 → 裁剪卡片区域
                              ↓
           ┌──────────────────┼──────────────────┐
           ↓                                     ↓
    颜色识别器                              纹路识别器
    HSV 直方图匹配                          PiDiNet 边缘 + 模板匹配
    (辅助 0.4)                             (核心 0.6)
           └──────────────────┬──────────────────┘
                              ↓
                       决策融合
                  color × 0.4 + edge × 0.6
```

**文件**：`vision/color_card_detector.py`

### 手势状态机（5 模式 FSM）

| 模式 | 状态 | 说明 |
|------|------|------|
| GLOBAL | IDLE | 全局空闲 |
| DRAWING | TRACKING → COMPLETED/CANCELLED | 绘画模式 |
| CANDIDATE | BROWSING → CONFIRMED/CANCELLED | 物象选择 |
| CHAR_RECOMMEND | BROWSING → CONFIRMED/TO_WHEEL | 人物推荐 |
| CHAR_WHEEL | SCROLLING → PREVIEWING → CONFIRMED | 人物轮盘 |

**文件**：`vision/gesture_state_machine.py`

### 草图识别（QuickDraw CNN）

```
食指指尖轨迹 → 28×28 灰度图 → CNN 推理 → 物象映射 → 颜色加权 → Top-3
```

- 模型：QuickDraw MobileNet，82 类，ONNX 2.5MB
- 准确率：83.8%
- 88 物象知识库映射

**文件**：`vision/sketch_recognizer.py`

### RAG 检索生成

```
用户选择 → 知识库检索 → LLM 生成 → 个性化叙事 + 图生图
```

- 知识库：208 实体 + 107 组合 + 100 模板
- LLM：阿里云百炼（qwen-turbo 实时/qwen-plus 叙事）
- 生图：Wan 2.7 image-pro（图生图融合）

**文件**：`rag/generator.py`

### 人物推荐

四维打分排序：

| 维度 | 权重 | 说明 |
|------|------|------|
| 内置参考表 | 0.50 | 颜色+物象→人物内置映射 |
| 同组加权 | 0.20 | 已选人物的同组人物 |
| 关键词匹配 | 0.25 | 文本相似度 |
| 实体基础分 | 0.05 | 实体 popularity |

**文件**：`rag/character_recommend.py`

### Unity 通信

- **:8888** 主通道：事件驱动（候选/确认/生成结果）
- **:8889** 手部通道：~30fps 逐帧手部数据

**文件**：`unity_bridge/server.py`

---

## 训练教程

### 阶段 1：采集模板（必做）

模板用于纹路识别，每种颜色牌采集一次即可。

```bash
# 启动采集（需连接摄像头）
python vision/color_card_detector.py
```

**操作**：
1. 按 `SPACE` 拍照（每种 3-5 张不同角度）
2. 按 `N` 跳到下一种颜色牌
3. 按 `Q` 退出

**采集顺序**：岳麓绿 → 书院红 → 西迁黄 → 湘江蓝 → 校徽金 → 墨色

**输出**：`vision/color_card_templates.npz`

### 阶段 2：标注数据（YOLO 定位用）

```bash
# 安装标注工具
pip install labelImg

# 启动标注（指定图片目录）
labelImg dataset/images/train
```

**设置**：
1. 点击 `Change Save Dir` → `dataset/labels/train`
2. 格式选择 `YOLO`（右下角）
3. 勾选 `Auto Save Mode`

**标签**：

| ID | 标签名 | 颜色牌 |
|----|--------|--------|
| 0 | yuelu_green | 岳麓绿 |
| 1 | academy_red | 书院红 |
| 2 | xiqian_yellow | 西迁黄 |
| 3 | xiangjiang_blue | 湘江蓝 |
| 4 | badge_gold | 校徽金 |
| 5 | ink_black | 墨色 |

### 阶段 3：训练 YOLO

```bash
# 开始训练
python yolo/train_yolo.py
```

**训练参数**（可修改 `yolo/train_yolo.py`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 100 | 训练轮数 |
| imgsz | 640 | 输入尺寸 |
| batch | 8 | 批大小 |

**输出**：`runs/detect/color_card_detector/weights/best.pt`

### 阶段 4：部署

```bash
# 复制模型
copy runs\detect\color_card_detector\weights\best.pt yolo\color_card.pt

# 验证
python vision/color_card_detector.py
```

---

## 常见问题

### Q1: 颜色牌检测不准确

1. **模板不足**：在展览现场同款灯光下重新采集模板
2. **YOLO 未训练**：按照训练教程标注并训练 YOLO
3. **光照差异**：暗光环境下，纹路识别权重更高（0.6）

### Q2: 草图识别准确率低

- 当前模型验证准确率 83.8%，属于正常范围
- 可采集更多 QuickDraw 数据重新训练

### Q3: Unity 连接失败

1. 检查 `config_ipcam.py` 摄像头配置
2. 检查防火墙是否允许 Python 端口
3. 查看 `test_integrated.py` 输出日志

### Q4: 图像生成失败

1. 检查 `DASHSCOPE_API_KEY` 环境变量
2. 查看 `rag/generator.py` 日志错误信息
3. 网络问题可使用 mock 模式测试

---

## 开发状态

| 模块 | 状态 | 说明 |
|------|------|------|
| 手势识别（MediaPipe） | ✅ 完成 | 21 关键点追踪 |
| 手势状态机（FSM） | ✅ 完成 | 5 模式切换 |
| 颜色牌检测（双识别） | 🔸 待训练 | 需采集模板+训练 YOLO |
| 草图识别（CNN） | ✅ 完成 | 准确率 83.8% |
| 人物推荐（RAG） | ✅ 完成 | 四维打分 |
| Python-Unity 通信 | ✅ 完成 | 双端口 TCP |
| Unity 渲染 | 🔸 进行中 | Unity 端开发 |
| 知识库 | ✅ 完成 | 208 实体 |
| 图像生成 | ✅ 完成 | Wan 2.7 图生图 |

---

## 贡献指南

### Git 提交规范

```
[模块名] 具体做了什么

例如：
[vision] 添加 YOLO 识别模块
[rag] 完成 BM25 检索基础实现
```

### 分支管理

| 分支 | 用途 |
|------|------|
| main | 稳定版本 |
| dev | 开发分支 |
| feature/名字 | 个人开发 |

---

## 致谢

- **课程**：智能设计方法（Prompt Engineering / Role-Playing / Agent / Hallucination）
- **数据集**：Google Quick, Draw!
- **模型**：MediaPipe / YOLOv8 / PiDiNet / 阿里云百炼

---

**湖南大学设计艺术学院 · 智能设计方法 · 哦齁齁齁组 · 2026**
