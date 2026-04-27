#!/usr/bin/env python3
"""
将 labelme 的 JSON 标注转换为 YOLO txt 格式
"""

import json
import os
from pathlib import Path

def convert_labelme_to_yolo(json_path: str, output_dir: str, class_name: str = "module"):
    """转换单个 labelme JSON 文件为 YOLO 格式"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    yolo_lines = []
    for shape in data['shapes']:
        label = shape['label'].lower().strip()
        if label != class_name.lower():
            print(f"  WARNING: 跳过未知类别 '{shape['label']}'，应为 '{class_name}'")
            continue

        points = shape['points']  # [[x1,y1], [x2,y2]]
        x1, y1 = points[0]
        x2, y2 = points[1]

        # 转为中心点 + 宽高（归一化到 0-1）
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        width = abs(x2 - x1) / image_width
        height = abs(y2 - y1) / image_height

        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 写入 txt
    txt_name = Path(json_path).stem + '.txt'
    txt_path = os.path.join(output_dir, txt_name)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    return len(yolo_lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', default='../dataset/labels/train')
    parser.add_argument('--output_dir', default='../dataset/labels/train')
    parser.add_argument('--class_name', default='module')
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(json_dir.glob('*.json'))
    print(f"找到 {len(json_files)} 个 labelme JSON 文件")

    total = 0
    for json_file in json_files:
        n = convert_labelme_to_yolo(str(json_file), str(output_dir), args.class_name)
        total += n
        print(f"  {json_file.name} -> {json_file.stem}.txt ({n} 目标)")

    print(f"\n转换完成，共 {total} 个目标")


if __name__ == '__main__':
    main()
