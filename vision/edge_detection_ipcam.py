"""
IP摄像头实时边缘检测
基于 PiDiNet (Pixel Difference Network) 模型

参考: https://github.com/yunfan1202/intelligent_design/tree/main/Perception

使用方法:
    python edge_detection_ipcam.py

需要先安装依赖:
    pip install opencv-python numpy torch torchvision
"""

import sys
import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 导入 IPCamera
from ipcamera import IPCamera


def load_pidinet(model_path='pidinet/BSDS_raw.pth', config='carv4', sa=True, dil=True):
    """
    加载 PiDiNet 模型

    Args:
        model_path: 模型权重路径
        config: 模型配置 (carv4)
        sa: 使用 CSAM (Compact Spatial Attention Module)
        dil: 使用 CDCM (Compact Dilation Convolution based Module)

    Returns:
        model: 加载好的 PiDiNet 模型
        args: 配置对象
    """
    # 添加当前目录到 path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from pidinet.models import pidinet

    class Args:
        def __init__(self):
            self.config = config
            self.sa = sa
            self.dil = dil

    args = Args()

    # 创建模型
    model = pidinet(args)
    model.eval()

    # 加载权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint['state_dict']

        # 处理权重键名 (移除 'module.' 前缀)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict, strict=False)
        print(f"[PiDiNet] 模型加载成功: {model_path}")
    else:
        print(f"[PiDiNet] 警告: 模型文件不存在 {model_path}")

    return model, args


def preprocess_image(img, target_size=640):
    """
    预处理图像用于 PiDiNet

    Args:
        img: BGR 格式的 numpy 数组
        target_size: 目标尺寸

    Returns:
        tensor: 预处理后的 tensor
    """
    # BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 调整大小
    h, w = img_rgb.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (new_w, new_h))

    # 填充到正方形
    img_square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    img_square[:new_h, :new_w] = img_resized

    # 归一化并转换为 tensor
    img_float = img_square.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std

    # HWC to CHW
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()

    return img_tensor, scale, new_h, new_w


def detect_edge(model, img_tensor, use_cuda=True):
    """
    使用 PiDiNet 检测边缘

    Args:
        model: PiDiNet 模型
        img_tensor: 预处理后的图像 tensor [1, 3, H, W]
        use_cuda: 是否使用 CUDA

    Returns:
        edge: 边缘图 (H, W), 0-1 范围
    """
    with torch.no_grad():
        if use_cuda and torch.cuda.is_available():
            model = model.cuda()
            img_tensor = img_tensor.cuda()

        outputs = model(img_tensor)

        # 使用最后一个输出 (融合结果)
        edge = outputs[-1]  # [1, 1, H, W]
        edge = edge.squeeze().cpu().numpy()  # [H, W]

    return edge


def postprocess_edge(edge, original_shape, scale, new_h, new_w):
    """
    后处理边缘图

    Args:
        edge: 模型输出的边缘图
        original_shape: 原始图像形状 (H, W)
        scale: 缩放比例
        new_h, new_w: 填充后的尺寸

    Returns:
        edge_original: 恢复到原始尺寸的边缘图
    """
    h, w = original_shape

    # 先裁剪到 new_h, new_w
    edge_crop = edge[:new_h, :new_w]

    # 缩放到原始尺寸
    edge_resized = cv2.resize(edge_crop, (w, h), interpolation=cv2.INTER_LINEAR)

    return edge_resized


def create_binary_edge_image(edge):
    """
    将边缘图转换为黑白图像：边缘为白色，其他为黑色

    Args:
        edge: 边缘图 (0-1)

    Returns:
        binary_img: 黑白图像 (uint8), 边缘为255, 非边缘为0
    """
    edge_uint8 = (edge * 255).astype(np.uint8)
    return edge_uint8


def overlay_edge(original_img, edge, color=(0, 255, 0), alpha=0.5):
    """
    将边缘叠加到原图上

    Args:
        original_img: 原图 (BGR)
        edge: 边缘图 (0-1)
        color: 边缘颜色 (BGR)
        alpha: 透明度

    Returns:
        result: 叠加后的图像
    """
    # 确保原图是 BGR 3通道
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    h, w = original_img.shape[:2]

    # 边缘图转换为 0-255 的 uint8
    edge_uint8 = (edge * 255).astype(np.uint8)

    # 创建一个彩色的边缘图像
    colored_edge = np.zeros_like(original_img)
    for c in range(3):
        colored_edge[:, :, c] = edge_uint8

    # 用颜色值调制
    for c in range(3):
        colored_edge[:, :, c] = np.where(edge_uint8 > 30, color[c], 0)

    # 叠加
    result = cv2.addWeighted(original_img, 1, colored_edge, alpha, 0)

    return result


def main():
    parser = argparse.ArgumentParser(description='IP摄像头实时边缘检测')
    parser.add_argument('--url', type=str, default='http://10.54.71.31:8080/video',
                        help='IP摄像头URL')
    parser.add_argument('--model', type=str, default='pidinet/BSDS_raw.pth',
                        help='模型权重路径')
    parser.add_argument('--width', type=int, default=640,
                        help='摄像头帧宽度')
    parser.add_argument('--height', type=int, default=480,
                        help='摄像头帧高度')
    parser.add_argument('--target-size', type=int, default=640,
                        help='模型输入尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.5,
                        help='边缘检测阈值')
    parser.add_argument('--no-cuda', action='store_true',
                        help='禁用 CUDA')
    args = parser.parse_args()

    # 加载模型
    print("=" * 50)
    print("加载 PiDiNet 模型...")
    model, _ = load_pidinet(args.model)
    print("=" * 50)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f"使用 CUDA: {use_cuda}")

    # 连接摄像头
    print(f"\n连接摄像头: {args.url}")
    camera = IPCamera(args.url, target_width=args.width, target_height=args.height)

    if not camera.connect():
        print("[错误] 无法连接到摄像头!")
        print("提示: 请确保:")
        print("  1. 手机IP摄像头APP已启动")
        print("  2. URL地址正确 (如 http://192.168.1.101:8080/video)")
        print("  3. 手机和电脑在同一局域网下")
        return

    print("[OK] 摄像头连接成功!")

    # 创建窗口
    window_name = 'Edge Detection - PiDiNet'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 480)

    print("\n开始边缘检测 (按 'q' 退出)...")
    print("-" * 50)

    frame_count = 0
    fps = 0

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("[警告] 无法读取帧, 尝试重新连接...")
                camera.release()
                camera.connect()
                continue

            frame_count += 1

            # 预处理
            img_tensor, scale, new_h, new_w = preprocess_image(frame, args.target_size)
            img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度

            # 边缘检测
            edge = detect_edge(model, img_tensor, use_cuda)

            # 后处理
            edge = postprocess_edge(edge, frame.shape[:2], scale, new_h, new_w)

            # 应用阈值生成二值边缘图（白色边缘，黑色背景）
            edge_binary = (edge > args.conf_thres).astype(np.float32)
            binary_img = create_binary_edge_image(edge_binary)

            # 显示信息
            info_text = f"Frame: {frame_count} | Threshold: {args.conf_thres:.2f} | CUDA: {use_cuda}"
            cv2.putText(binary_img, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

            # 并排显示：原图 + 边缘检测结果（转为3通道以便拼接）
            combined = np.hstack([
                cv2.resize(frame, (640, 360)),
                cv2.resize(cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR), (640, 360))
            ])

            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                args.conf_thres = min(0.9, args.conf_thres + 0.05)
                print(f"阈值: {args.conf_thres:.2f}")
            elif key == ord('-'):
                args.conf_thres = max(0.1, args.conf_thres - 0.05)
                print(f"阈值: {args.conf_thres:.2f}")

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\n程序退出")


if __name__ == '__main__':
    main()
