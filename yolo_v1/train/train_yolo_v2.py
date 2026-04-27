#!/usr/bin/env python3
"""
YOLO 训练 - 完整 detection loss 实现 v2
针对 YOLOv8 架构正确处理 DFL 输出
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import yaml
from ultralytics import YOLO
from datetime import datetime


class ModuleDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  Dataset: {len(self.files)} images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

        label_path = os.path.join(self.label_dir, os.path.splitext(fname)[0] + '.txt')
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        boxes.append([cls, x, y, w, h])

        label_tensor = torch.zeros(30, 5)
        if len(boxes) > 0:
            label_tensor[:len(boxes)] = torch.tensor(boxes, dtype=torch.float32)

        return img_tensor, label_tensor, fname


def dfl_decode(box_tensor, reg_max=16):
    """
    解码 DFL (Distribution Focal Loss) 格式的 box 预测
    box_tensor: [B, 4*reg_max, N] raw DFL output
    返回: [B, 4, N] decoded xywh
    """
    B, C, N = box_tensor.shape
    reg_max = C // 4

    box_tensor = box_tensor.view(B, 4, reg_max, N)
    box_dist = F.softmax(box_tensor, dim=2)

    img_size = 640
    i = torch.arange(reg_max, device=box_tensor.device, dtype=torch.float32)
    step = img_size / reg_max
    i = i * step

    box_decoded = (box_dist * i.view(1, 1, reg_max, 1)).sum(dim=2)
    return box_decoded


def compute_iou(box1, box2):
    """box1, box2: [N, 4] xywh normalized"""
    def to_xyxy(b):
        x, y, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    b1 = to_xyxy(box1)
    b2 = to_xyxy(box2)

    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union_area = area1 + area2 - inter_area + 1e-7

    return inter_area / union_area


def yolo_loss(pred_boxes, pred_scores, targets):
    """计算 YOLO 风格 detection loss"""
    device = pred_boxes.device
    B = pred_boxes.shape[0]
    N = pred_boxes.shape[2]
    reg_max = 16

    decoded_boxes = dfl_decode(pred_boxes, reg_max)

    obj_conf = torch.sigmoid(pred_scores[:, 4:5, :])  # [B, 1, N]

    total_box_loss = torch.tensor(0.0, device=device)
    total_obj_loss = torch.tensor(0.0, device=device)
    box_count = 0

    for b in range(B):
        gt = targets[b]
        valid = gt[:, 0] >= 0
        gt_boxes = gt[valid]

        if len(gt_boxes) == 0:
            obj_pred = obj_conf[b].t().squeeze(-1)
            obj_target = torch.zeros(N, device=device)
            total_obj_loss += F.binary_cross_entropy_with_logits(obj_pred, obj_target, reduction='sum') / B
            continue

        obj_target = torch.zeros(N, device=device)
        decoded = decoded_boxes[b].t()  # [N, 4]

        dx_l = decoded[:, 0]
        dy_t = decoded[:, 1]
        dx_r = decoded[:, 2]
        dy_b = decoded[:, 3]

        pred_w = (dx_l + dx_r) / 640
        pred_h = (dy_t + dy_b) / 640

        matched_mask = torch.zeros(N, dtype=torch.bool, device=device)

        for i, gt_box in enumerate(gt_boxes):
            gt_w = gt_box[3]
            gt_h = gt_box[4]

            gt_area = (gt_w * gt_h).clamp(min=1e-6)
            pred_area = (pred_w * pred_h).clamp(min=1e-6)
            gt_ratio = gt_w / gt_h.clamp(min=1e-6)
            pred_ratio = pred_w / pred_h.clamp(min=1e-6)

            area_ratio = (pred_area / gt_area).clamp(0.01, 100)
            ratio_diff = (gt_ratio - pred_ratio).abs()
            match_cost = (area_ratio.log()**2 + ratio_diff**2).sqrt()
            best_idx = match_cost.argmin()

            matched_mask[best_idx] = True

        if matched_mask.sum() > 0:
            pred_cx = (dx_r - dx_l) / 2 / 640
            pred_cy = (dy_b - dy_t) / 2 / 640
            pred_w_box = pred_w
            pred_h_box = pred_h

            matched_pred = torch.stack([pred_cx[matched_mask], pred_cy[matched_mask],
                                        pred_w_box[matched_mask], pred_h_box[matched_mask]], dim=1)
            matched_gt = gt_boxes[:, 1:5]

            # matched_idxs are the original indices in [0, N-1]
            matched_idxs = matched_mask.nonzero(as_tuple=True)[0]

            for k, idx in enumerate(matched_idxs):
                dist = ((matched_pred[k, 0] - matched_gt[:, 0])**2 +
                        (matched_pred[k, 1] - matched_gt[:, 1])**2 +
                        (matched_pred[k, 2] - matched_gt[:, 2])**2 +
                        (matched_pred[k, 3] - matched_gt[:, 3])**2)
                best_gt_idx = dist.argmin()
                box_loss = F.l1_loss(matched_pred[k], matched_gt[best_gt_idx])
                total_box_loss += box_loss / len(matched_idxs)

            obj_pred = obj_conf[b].t().squeeze(-1)
            obj_target_matched = torch.zeros(N, device=device)
            obj_target_matched[matched_idxs] = 1.0
            total_obj_loss += F.binary_cross_entropy_with_logits(obj_pred, obj_target_matched, reduction='sum') / B

        box_count += 1

    if box_count > 0:
        total_box_loss = total_box_loss / box_count
        total_obj_loss = total_obj_loss / B

    return {
        'box': total_box_loss * 7.5,
        'obj': total_obj_loss * 1.0,
        'total': torch.zeros(1, device=device)
    }


def train():
    print("=" * 50)
    print("YOLO 训练 v2 - 完整 Detection Loss")
    print("=" * 50)

    IMG_SIZE = 640
    BATCH_SIZE = 4
    EPOCHS = 100
    LR = 0.01
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"设备: {DEVICE}")
    print(f"批次: {BATCH_SIZE}, 图像: {IMG_SIZE}, 轮次: {EPOCHS}")

    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

    print("\n加载数据集...")
    with open('dataset.yaml') as f:
        data = yaml.safe_load(f)

    base_path = data['path']
    train_img = os.path.join(base_path, data['train'])
    train_lbl = os.path.join(base_path, 'labels', 'train')
    val_img = os.path.join(base_path, data['val'])
    val_lbl = os.path.join(base_path, 'labels', 'val')

    print(f"训练: {train_img}")
    train_ds = ModuleDataset(train_img, train_lbl, IMG_SIZE)
    val_ds = ModuleDataset(val_img, val_lbl, IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\n加载 YOLOv8n 模型...")
    model = YOLO('yolov8n.pt')
    model.to(DEVICE)
    model.model.train()

    print(f"模型参数量: {sum(p.numel() for p in model.model.parameters()):,}")

    for name, param in model.model.named_parameters():
        param.requires_grad = 'model.22' in name

    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable:,}")

    optimizer = torch.optim.SGD(
        [p for p in model.model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n开始训练...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"runs/detect/train_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.model.train()
        epoch_box = 0
        epoch_obj = 0
        batch_count = 0

        for batch_idx, (images, targets, filenames) in enumerate(train_loader):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model.model(images)
            pred_boxes = outputs['boxes']
            pred_scores = outputs['scores']

            losses = yolo_loss(pred_boxes, pred_scores, targets)
            loss = losses['box'] + losses['obj']

            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 10.0)
                optimizer.step()

            epoch_box += losses['box'].item()
            epoch_obj += losses['obj'].item()
            batch_count += 1

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} batch {batch_idx}/{len(train_loader)} "
                      f"box={losses['box'].item():.4f} obj={losses['obj'].item():.4f}")

        scheduler.step()

        avg_box = epoch_box / max(batch_count, 1)
        avg_obj = epoch_obj / max(batch_count, 1)
        avg_total = avg_box + avg_obj

        print(f"Epoch {epoch}/{EPOCHS} - box:{avg_box:.4f} obj:{avg_obj:.4f} total:{avg_total:.4f}")

        if avg_total < best_loss and avg_total > 0:
            best_loss = avg_total
            ckpt = os.path.join(save_dir, 'best.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, ckpt)
            print(f"  -> 保存最佳模型: {ckpt}")

        if epoch % 10 == 0:
            ckpt = os.path.join(save_dir, f'epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, ckpt)

    print(f"\n训练完成! 最佳 loss: {best_loss:.4f}")

    final_path = os.path.join(save_dir, 'last.pt')
    torch.save(model.model.state_dict(), final_path)
    print(f"最终模型: {final_path}")

    return save_dir


if __name__ == "__main__":
    train()