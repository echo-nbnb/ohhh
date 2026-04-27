# 手部跟踪 + Unity 可视化（第一版）

实时从 IP 摄像头检测手部位置，通过 TCP socket 传输到 Unity 端，以点线方式可视化。

## 目录结构

```
hand_tracking_v1/
├── test_socket_server.py       # Python 入口
├── config_ipcam.py              # IP摄像头地址配置
├── hand_landmarker.task         # MediaPipe 模型文件
├── OPTIMIZATION.md              # 后续优化方案
├── vision/                      # 视觉模块
│   ├── __init__.py
│   ├── ipcamera.py             # IP摄像头读取
│   └── hand_tracker.py         # MediaPipe 手部检测
├── unity/                      # Unity 工程文件
│   ├── Scripts/
│   │   ├── HandTrackingConnection.cs    # TCP 客户端
│   │   ├── HandTrackingVisualizer.cs    # 可视化（点+线+框）
│   │   ├── HandTrackingData.cs          # 数据结构
│   │   └── UnityMainThreadDispatcher.cs # 跨线程调度
│   └── Scenes/
│       └── Test.unity          # 测试场景
└── README.md
```

## 快速启动

### Python 端

```bash
cd D:/projects/ohhh/hand_tracking_v1
python test_socket_server.py
```

### Unity 端

1. 用 Unity 打开 `D:\unity projects\ohhh-display` 项目
2. 打开场景 `Test.unity`
3. Play

## 数据流

```
IPCam → Python (MediaPipe 检测) → TCP (JSON) → Unity → 可视化
```

## 通信协议

```json
{
    "type": "hand_tracking",
    "palm_center": [x, y],
    "wrist": [x, y],
    "contour": [x0,y0, x1,y1, ..., x6,y6],
    "bounding_box": [x_min, y_min, x_max, y_max]
}
```

## 主要文件说明

| 文件 | 说明 |
|------|------|
| `test_socket_server.py` | TCP 服务器，双线程架构（检测+读取分离） |
| `config_ipcam.py` | IP摄像头地址，需修改为实际 IP |
| `hand_tracker.py` | MediaPipe Hands 检测，输出21点+指尖+手腕 |
| `HandTrackingConnection.cs` | Unity TCP 客户端，线程安全队列 |
| `HandTrackingVisualizer.cs` | Unity 可视化，contour线+指尖点+手掌点+包围框 |
| `Test.unity` | 场景，已配置好所有组件 |

## 已知局限

- **延迟较高**：约 70-150ms，详见 `OPTIMIZATION.md`
- **无坐标标定**：像素坐标直接映射，未做四点透视变换
- **仅单手检测**：MediaPipe 配置为 `num_hands=1`
