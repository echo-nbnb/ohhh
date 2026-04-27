# YOLO 模块检测 + Unity 可视化（测试版）

用 YOLOv8 检测两个相同物块，通过 TCP 传输到 Unity 端以边框方式可视化。

## 目录结构

```
yolo_v1/
├── server.py                  # Python 入口：YOLO检测 + TCP服务器
├── vision/
│   ├── __init__.py
│   └── ipcamera.py           # IP摄像头读取
├── unity/
│   ├── Scripts/
│   │   ├── YoloConnection.cs    # TCP客户端
│   │   ├── YoloVisualizer.cs    # Unity可视化（边框）
│   │   └── YoloData.cs          # 数据结构
│   └── Scenes/
│       └── YoloTest.unity       # 测试场景
└── README.md
```

## 快速启动

### 第一步：采集训练数据

准备两个相同物块，摆在摄像头前：

```bash
cd D:/projects/ohhh/yolo_v1
python -c "import sys; sys.path.insert(0,'.'); from vision.ipcamera import IPCamera; cam = IPCamera('http://10.54.71.31:8080/video'); cam.connect()"
```

采集图片（放在 `../dataset/images/train`）：

```bash
cd D:/projects/ohhh/yolo
python collect_data.py --output ../dataset/images/train --prefix module
```

**采集建议：**
- 每个物块 30-50 张，覆盖不同角度和光照
- 变换物块在画面中的位置

### 第二步：标注数据

```bash
pip install labelImg
labelImg ../dataset/images/train
```

1. `w` 画框 → 输入类别名 `module`
2. `d` 下一张 → `Ctrl+s` 保存
3. 格式选 **YOLO**

### 第三步：训练

> **重要**：由于 `ultralytics` 训练引擎在当前 CUDA 环境下存在 segfault 问题，建议使用自定义训练脚本 `train_yolo_v2.py`。

```bash
cd D:/projects/ohhh/yolo_v1/train

# 推荐方式：自定义训练脚本（已验证可运行）
python train_yolo_v2.py

# 或者：原始训练脚本（可能 segfault）
python train_yolo.py
```

模型输出到 `train/runs/detect/train_YYYYMMDD_HHMMSS/last.pt`

**训练参数说明：**
- `IMG_SIZE = 640` - 输入图片尺寸
- `BATCH_SIZE = 4` - 批次大小（RTX 4060 8G 建议 2-8）
- `EPOCHS = 100` - 训练轮数
- `LR = 0.01` - 学习率

**调试参数修改 `train_yolo_v2.py` 中的配置区域。**

### 第四步：启动服务器

```bash
cd D:/projects/ohhh/yolo_v1
python server.py
```

服务器监听 **8890** 端口，YOLO 模型自动从 `yolov8n.pt` 下载。

### 第五步：Unity 端

1. 用 Unity 打开 `D:\unity projects\ohhh-display`
2. 打开场景 `YoloTest.unity`
3. Play

## 数据流

```
IPCam → YOLOv8 检测 → TCP (JSON) → Unity → 边框可视化
```

## 通信协议

```json
{
    "type": "yolo_detection",
    "count": 2,
    "detections": [
        {"x1": 100, "y1": 200, "x2": 300, "y2": 400, "conf": 0.95},
        {"x1": 500, "y1": 300, "x2": 700, "y2": 500, "conf": 0.87}
    ]
}
```

## 主要文件说明

| 文件 | 说明 |
|------|------|
| `server.py` | TCP服务器，双线程（检测+读取分离），约30fps发送 |
| `ipcamera.py` | IPCamera 读取，640x480分辨率 |
| `YoloConnection.cs` | Unity TCP客户端，线程安全队列 |
| `YoloVisualizer.cs` | Unity可视化，画边框（LineRenderer） |

## 已知局限

- **延迟**：约 50-100ms
- **无坐标标定**：像素坐标直接映射，未做透视变换
- **仅两类物块**：训练的是统一 "module" 类别，不区分具体物块
