# 手部跟踪延迟优化方案

## 已完成
- [x] TCP_NODELAY（禁用 Nagle 算法，数据立即发送）

## 待实施（按延迟收益排序）

### 1. MediaPipe 输入分辨率缩放
- **改动位置**：`vision/hand_tracker.py` 的 `get_data_for_unity` 方法
- **改动内容**：将输入帧 resize 到 960x540 再送入检测
- **预期延迟改善**：20-50ms
- **副作用**：检测精度略微下降（但位置精度影响不大）

### 2. struct 二进制协议
- **改动位置**：`test_socket_server.py` + Unity 端需新增 `BinaryReader`
- **改动内容**：用 `struct.pack` 替代 `json.dumps`，直接发二进制数据
- **预期延迟改善**：5-10ms
- **注意**：Unity 端需用 `BinaryReader` 解析

### 3. Unity 端 JSON 库换用Utf8Json/Newtonsoft
- **改动位置**：Unity 项目
- **预期延迟改善**：5-10ms
- **注意**：需安装 NuGet 包

### 4. MediaPipe LIVE_STREAM 模式
- **改动位置**：`vision/hand_tracker.py`
- **改动内容**：改用 `RunningMode.LIVE_STREAM`，只检测单手
- **预期延迟改善**：5-10ms
- **注意**：API 变化较大，需重构

### 5. dataclass 替代 dict
- **改动位置**：`vision/hand_tracker.py`
- **改动内容**：hand_data 用 dataclass/SimpleNamespace
- **预期延迟改善**：1-3ms

---

## 架构优化（根本解决）

### 双线程架构（已完成）
- 检测线程：跑 MediaPipe
- 主线程：读摄像头 + 发 Unity
- 共享状态：latest_frame + latest_hand_data

---

## 延迟预算参考

| 环节 | 耗时 |
|------|------|
| IPCam → Python（摄像头读取） | 约 10-30ms |
| MediaPipe 检测（1920x1080） | 约 50-100ms |
| MediaPipe 检测（960x540） | 约 15-30ms |
| JSON 序列化 | 约 2-5ms |
| TCP 传输 | 约 1-5ms |
| Unity JSON 解析 | 约 5-15ms（用 JsonUtility） |
| **总计（优化前）** | **约 70-150ms** |
| **总计（优化后预计）** | **约 30-60ms** |
