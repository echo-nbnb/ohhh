#!/usr/bin/env python3
"""点状风格化生成 — 物象图片 → 透明底点状 PNG"""

import sys, os, math, random, cv2
import numpy as np
from pathlib import Path

INPUT_DIR = Path("rag/knowledge/物象图片")
OUTPUT_DIR = Path("web/frontend/src/assets/act3/objects")
MAX_SIZE = 800

def hex_to_rgb(hx):
    hx = hx.lstrip("#")
    return int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)

def lerp(a, b, t): return a + (b - a) * t
def clamp(v, lo, hi): return max(lo, min(hi, v))

def mix_color(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))

def generate_pointillism(img_bgr, primary_hex="#596b8a", secondary_hex="#b38a5f",
                        base_density=0.025, detail_density=0.045,
                        base_size=2.0, detail_size=2.8, edge_boost=1.8):
    """输入 BGR 图像，返回透明底点状化 RGBA 图像"""
    h, w = img_bgr.shape[:2]
    scale = min(1, MAX_SIZE / max(w, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = cv2.resize(img_bgr, (nw, nh))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # 透明主体：抠白底
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    whiteness = (r + g + b) / 3
    max_ch = np.max(rgb, axis=2)
    min_ch = np.min(rgb, axis=2)
    low_sat = (max_ch - min_ch) < 18
    very_white = (r >= 230) & (g >= 230) & (b >= 230)
    near_white = (whiteness >= 235) & low_sat
    is_bg = very_white | near_white
    alpha = np.where(is_bg, 0, np.clip((255 - whiteness) * 12, 0, 255))
    alpha[alpha < 20] = 0
    alpha_map = alpha.astype(np.float32) / 255.0

    # 灰度图
    gray = np.where(alpha_map < 0.02, 1.0,
                    (r * 0.299 + g * 0.587 + b * 0.114) / 255.0)

    # 边缘图（Sobel）
    gray_u8 = (np.clip(gray * 255, 0, 255)).astype(np.uint8)
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(np.sqrt(gx ** 2 + gy ** 2), 0, 1)
    edge[alpha_map < 0.02] = 0

    # 生成点状画
    primary = hex_to_rgb(primary_hex)
    secondary = hex_to_rgb(secondary_hex)
    base_count = int(nw * nh * base_density)
    detail_count = int(nw * nh * detail_density)
    outer_count = int(nw * nh * 0.0025)

    canvas = np.zeros((nh, nw, 4), dtype=np.uint8)

    def draw_dot(cx, cy, radius, color, opacity):
        x0 = max(0, int(cx - radius - 1))
        x1 = min(nw - 1, int(cx + radius + 1))
        y0 = max(0, int(cy - radius - 1))
        y1 = min(nh - 1, int(cy + radius + 1))
        rr = int(radius + 0.5)
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                if (px - cx) ** 2 + (py - cy) ** 2 <= rr ** 2:
                    a = int(opacity * 255)
                    # 用 int 避免 uint8 溢出
                    r0, g0, b0, a0 = int(canvas[py, px, 0]), int(canvas[py, px, 1]), int(canvas[py, px, 2]), int(canvas[py, px, 3])
                    canvas[py, px, 0] = min(255, (r0 * (255 - a) + color[0] * a) // 255)
                    canvas[py, px, 1] = min(255, (g0 * (255 - a) + color[1] * a) // 255)
                    canvas[py, px, 2] = min(255, (b0 * (255 - a) + color[2] * a) // 255)
                    canvas[py, px, 3] = min(255, a0 + a)

    # 基础层
    for _ in range(base_count):
        x, y = random.random() * nw, random.random() * nh
        ix, iy = int(x), int(y)
        a = alpha_map[iy, ix]
        if a < 0.06: continue
        darkness = 1 - gray[iy, ix]
        accept = 0.12 + darkness * 0.72 + a * 0.10
        if random.random() > accept: continue
        t = clamp(0.20 + darkness * 0.62 + random.uniform(-0.12, 0.12), 0, 1)
        color = mix_color(primary, secondary, t)
        radius = random.uniform(0.35, base_size)
        opacity = clamp(0.06 + darkness * 0.24 + a * 0.06 + random.random() * 0.06, 0.03, 0.42)
        draw_dot(x, y, radius, color, opacity)

    # 细节层
    for _ in range(detail_count):
        x, y = random.random() * nw, random.random() * nh
        ix, iy = int(x), int(y)
        a = alpha_map[iy, ix]
        if a < 0.06: continue
        e = clamp(edge[iy, ix] * edge_boost, 0, 1)
        darkness = 1 - gray[iy, ix]
        accept = clamp(e * 1.15 + darkness * 0.26 + a * 0.06, 0, 1)
        if random.random() > accept: continue
        t = clamp(0.35 + e * 0.45 + random.uniform(-0.16, 0.16), 0, 1)
        color = mix_color(primary, secondary, t)
        radius = random.uniform(0.18, detail_size)
        opacity = clamp(0.13 + e * 0.42 + random.random() * 0.08, 0.08, 0.66)
        draw_dot(x, y, radius, color, opacity)

    # 外围散点
    for _ in range(outer_count):
        ix = random.randint(0, nw - 1)
        iy = random.randint(0, nh - 1)
        if alpha_map[iy, ix] > 0.05: continue
        # 检查附近是否有主体
        near = False
        for oy in range(-5, 6):
            for ox in range(-5, 6):
                nx, ny = clamp(ix + ox, 0, nw - 1), clamp(iy + oy, 0, nh - 1)
                if alpha_map[ny, nx] > 0.05 and edge[ny, nx] > 0.06:
                    near = True
                    break
            if near: break
        if not near: continue
        color = mix_color(primary, secondary, random.random())
        draw_dot(ix + random.random(), iy + random.random(),
                 random.uniform(0.18, 0.95), color, random.uniform(0.025, 0.095))

    return canvas


def process_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(INPUT_DIR.glob("*"))
    if not images:
        print(f"未找到图片: {INPUT_DIR}")
        return

    for path in images:
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        name = path.stem
        print(f"处理: {name}")
        # cv2.imread 不支持 Unicode 路径，用 numpy 读取
        with open(str(path.resolve()), "rb") as f:
            data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  读取失败: {path}")
            continue

        result = generate_pointillism(img)
        out = OUTPUT_DIR / f"{name}.png"
        out_str = str(out.resolve())
        succ = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))[1].tofile(out_str)
        print(f"  → {out}")

    print(f"\n完成，输出到 {OUTPUT_DIR}")


if __name__ == "__main__":
    process_all()
