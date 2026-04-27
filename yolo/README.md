# YOLO 模块识别模型训练

## 环境说明

- **GPU**: NVIDIA GeForce RTX 4060 Laptop (8GB)
- **Python**: 3.12
- **CUDA**: 12.1
- **PyTorch**: 2.4.1+cu121
- **关键依赖**:
  - `torch==2.4.1+cu121`
  - `torchvision==0.19.1+cu121`
  - `ultralytics==8.4.41`

## 训练状态

✅ **训练路径已跑通** — 完整实现路径如下：

```
数据集准备 → 自定义训练脚本 → 模型推理验证
```

| 训练批次 | 时间 | 状态 |
|---------|------|------|
| `train_20260427_213530` | 2026/04/27 21:35 | ✅ 已完成 |
| `train_20260427_214155` | 2026/04/27 21:41 | ✅ 已完成 |

> ⚠️ **注意**: 原始的 `model.train()` API 在当前环境下存在 segfault 问题（CUDA/ultralytics 兼容性问题），因此使用自定义训练脚本 `train_yolo_v2.py` 绕过训练引擎。

## 安装依赖

```bash
# 确认当前 conda 环境
conda activate ohhh

# 如果需要重新安装匹配版本
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.4.41 opencv-python
```

## 数据集准备

数据集结构：
```
dataset/
├── images/
│   ├── train/          # 训练图片 (.jpg)
│   └── val/            # 验证图片 (.jpg)
└── labels/
    ├── train/          # 训练标签 (.txt, YOLO格式)
    └── val/            # 验证标签 (.txt, YOLO格式)
```

标签格式（YOLO txt）：
```
class_id x_center y_center width height
# 全部是归一化值（0-1）
# 例如单类检测:
0 0.512 0.483 0.128 0.096
```

`dataset.yaml` 配置：
```yaml
path: ../dataset
train: images/train
val: images/val

nc: 1
names:
  0: m
```

## 训练（✅ 已验证可运行）

### 自定义训练脚本（推荐）

```bash
cd D:\projects\ohhh\yolo

python train_yolo_v2.py
```

- 训练 100 轮，batch=4，图像尺寸=640
- 输出目录：`yolo/runs/detect/train_YYYYMMDD_HHMMSS/`
- 保存文件：`best.pt`、`epoch_10.pt`、`epoch_20.pt`...、`last.pt`

### 原始训练脚本（备选）

```bash
python train_yolo.py
```

> 注意：可能存在 segfault 问题，建议使用 `train_yolo_v2.py`

## 训练参数调整

编辑 `train_yolo_v2.py` 中的配置区域：

```python
IMG_SIZE = 640    # 输入图片尺寸（640/320/160）
BATCH_SIZE = 4    # 批次大小（RTX 4060 8G 建议 2-8）
EPOCHS = 100      # 训练轮数
LR = 0.01         # 学习率
```

## 使用训练好的模型

```python
from ultralytics import YOLO

# 加载最新训练结果
model = YOLO('runs/detect/train_20260427_214155/best.pt')

# 预测
results = model.predict('your_image.jpg', imgsz=640, conf=0.5)

# 绘制结果
results[0].plot()
```

或使用 predict_yolo.py 脚本：
```bash
python predict_yolo.py
```

## 问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `train_yolo.py` segfault (0xC0000005) | ultralytics 训练引擎与 CUDA 不兼容 | ✅ 已解决：使用 `train_yolo_v2.py` |
| `train_yolo_v2.py` 崩溃 | 可能是 workers > 0 | 确保 `num_workers=0` |
| loss 为 0 或不下降 | 数据路径问题或标签格式错误 | 检查 dataset.yaml 的 path 是否正确 |
| 显存不足 | batch 太大 | 减小 `BATCH_SIZE` 到 2 或 1 |

## 文件说明

- `train_yolo.py` - 原始训练脚本（ultralytics 官方用法）
- `train_yolo_v2.py` - 自定义训练脚本（✅ 已验证可运行）
- `predict_yolo.py` - 预测脚本
- `dataset.yaml` - 数据集配置文件
- `yolov8n.pt` - 预训练权重
- `runs/detect/train_YYYYMMDD_HHMMSS/` - 训练输出目录

## 训练输出

最新训练结果位于：
```
yolo/runs/detect/train_20260427_214155/
├── best.pt          # 最佳模型
├── epoch_10.pt      # 第10轮
├── epoch_20.pt      # 第20轮
├── ...
└── last.pt          # 最终模型
```