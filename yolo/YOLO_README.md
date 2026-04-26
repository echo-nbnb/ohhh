# YOLO 训练指南

本目录包含 YOLO 模块识别模型的训练和预测脚本。

---

## 目录结构

```
yolo/
├── dataset.yaml         # 数据集配置
├── train_yolo.py       # 训练脚本
├── predict_yolo.py     # 预测脚本
├── collect_data.py     # 数据采集脚本
└── YOLO_README.md      # 本文档

../dataset/             # 数据集目录
├── images/
│   ├── train/          # 训练图片
│   └── val/            # 验证图片
└── labels/
    ├── train/          # 训练标签
    └── val/            # 验证标签

../runs/detect/         # 训练输出目录（训练后生成）
```

---

## 完整训练流程

### 第一步：采集数据

```bash
cd yolo
python collect_data.py
```

**操作说明：**
- `SPACE` - 拍摄当前帧
- `S` - 开始/停止自动拍摄（每1秒自动拍一张）
- `R` - 查看已采集数量
- `ESC/q` - 退出

**采集建议：**
- 每个模块 30-50 张图片
- 覆盖不同角度、光照、位置
- 变换模块在画面中的位置

**采集到验证集：**
```bash
python collect_data.py --val
```

---

### 第二步：标注数据

安装标注工具：
```bash
pip install labelImg
```

启动标注：
```bash
labelImg ../dataset/images/train
```

**操作说明：**
1. 点击 "Open Dir" 选择 `../dataset/images/train`
2. 点击 "Change Save Dir" 选择 `../dataset/labels/train`
3. 按 `w` 画框
4. 输入类别名 `module`
5. 按 `d` 下一张
6. 按 `Ctrl+S` 保存

**标注格式**：YOLO（.txt）
**类别**：统一标注为 `module`

---

### 第三步：训练模型

```bash
python train_yolo.py
```

**训练参数说明：**
- `epochs=100` - 训练100轮，可根据效果增减
- `batch=16` - 批次大小（4060建议16-32）
- `imgsz=640` - 输入图片尺寸

**训练时间**：约 30-60 分钟（取决于数据量和batch大小）

**训练输出**：
- 模型保存在 `../runs/detect/module_detector/weights/best.pt`

---

### 第四步：测试模型

```bash
# 使用摄像头实时检测
python predict_yolo.py --source 0

# 检测单张图片
python predict_yolo.py --source ../test.jpg

# 指定模型
python predict_yolo.py --model ../my_model.pt
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `train_yolo.py` | 训练脚本 |
| `predict_yolo.py` | 预测脚本 |
| `collect_data.py` | 数据采集脚本 |
| `dataset.yaml` | 数据集配置文件 |

---

## 常见问题

**Q: 训练时显存不足？**
```python
# 减小 batch 大小
batch=8  # 或更小
```

**Q: 模型效果不好？**
- 增加训练数据量
- 增加训练轮数
- 调整图片尺寸 `imgsz=1280`

**Q: 如何添加新模块？**
1. 采集新模块图片
2. 标注为对应类别（如 module_b）
3. 修改 `dataset.yaml` 中的 nc 和 names
4. 重新训练

---

## 下一步

如果当前统一识别方案效果不好，可以：
1. 重新标注为12个独立类别
2. 使用更大/更深的模型（yolov8s, yolov8m）
3. 收集更多样本
