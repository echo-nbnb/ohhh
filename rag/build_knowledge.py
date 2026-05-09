"""
知识库构建脚本 — 从 txt 文件生成 JSON
"""
import json
import os
import re

BASE = "rag/knowledge"

# ============================================================
# 1. 颜色实体 → entities/colors.json
# ============================================================
COLOR_HEX_MAP = {
    "岳麓绿": "#2E7D32", "湖湘红": "#B71C1C", "湘水蓝": "#1565C0",
    "书卷白": "#F5F0E8", "荣光金": "#FFD700", "历史黑": "#212121",
    "晨曦橙": "#FF9800", "夜读紫": "#4A148C", "烟雨灰": "#9E9E9E",
    "墨韵蓝": "#1A237E", "青石灰": "#607D8B", "霞光粉": "#F8BBD0",
    "深林绿": "#1B5E20", "铁锈红": "#8B4513", "云雾白": "#ECEFF1",
    "湖面蓝": "#42A5F5", "光影金": "#FFC107", "暮色橙": "#FF5722",
    "石墨黑": "#37474F", "松柏绿": "#2E7D32", "烈日黄": "#FDD835",
    "冷光蓝": "#0277BD", "灰白调": "#BDBDBD",
}

def parse_colors():
    path = os.path.join(BASE, "颜色实体定义.txt")
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 7:
                continue
            name, meaning, desc, related, mood, era, theme = parts[:7]
            related_list = [r.strip() for r in related.replace("、", ",").split(",") if r.strip()]
            result[name] = {
                "type": "color",
                "description": desc,
                "color": COLOR_HEX_MAP.get(name, "#888888"),
                "symbolism": meaning,
                "related_entities": related_list,
                "historical_context": f"{name}，{meaning}，{desc}",
                "mood": mood,
                "era": era,
                "theme": theme,
            }
    return result

# ============================================================
# 2. 物象实体 → entities/objects.json
# ============================================================
def parse_objects():
    path = os.path.join(BASE, "物象典故.txt")
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            name, meaning, desc, related = parts[:4]
            related_list = [r.strip() for r in related.replace("、", ",").split(",") if r.strip()]
            result[name] = {
                "type": "object",
                "description": desc,
                "symbolism": meaning,
                "related_entities": related_list,
                "historical_context": f"{name}，{meaning}，{desc}",
            }
    return result

# ============================================================
# 3. 人物实体 → entities/characters.json
# ============================================================
def parse_characters():
    path = os.path.join(BASE, "人物故事.txt")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 清理标题和标记
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'人物故事.*\n', '', text)

    # 核心思路：每个条目格式为 "名称 | 角色 | 描述 | 特质"
    # 将所有行连起来，再按 | 分割，每4段构成一个条目
    segments = [s.strip() for s in text.split("|")]
    result = {}
    i = 0
    while i + 3 < len(segments):
        name = segments[i]
        role = segments[i + 1]
        desc = segments[i + 2]
        spirit = segments[i + 3]

        # 验证 name 看起来像个人名（2-6个中文字符，不含空格和数字）
        if (2 <= len(name) <= 15 and not re.search(r'[\s\d]', name)
                and not name.startswith("（") and not name.startswith("原")):
            result[name] = {
                "type": "character",
                "name": name,
                "years": "",
                "title": role,
                "description": desc,
                "spirit": spirit,
                "related_entities": [],
                "quotes": [],
                "stories": [f"{name}，{role}，{desc}"],
            }
            i += 4
        else:
            i += 1
    return result

# ============================================================
# 4. 组合解读 → combinations/*.json
# ============================================================
def parse_combinations():
    path = os.path.join(BASE, "组合解读.txt")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 合并所有行为连续文本
    text = re.sub(r'\n+', ' ', text)

    # 每条格式: Entity1 + Entity2 | Meaning | Interpretation
    # 用正则精确匹配，避免按 | 分割的偏移问题
    pattern = r'(\S+)\s*\+\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)'
    matches = re.findall(pattern, text)

    all_combos = []
    seen = set()
    for e1, e2, meaning, interp in matches:
        # 过滤非实体名（含数字、过长等）
        if re.search(r'\d', e1 + e2):
            continue
        key = f"{e1}_{e2}"
        if key in seen:
            continue
        seen.add(key)
        all_combos.append({
            "key": key, "entity1": e1, "entity2": e2,
            "meaning": meaning, "interpretation": interp,
        })
    return all_combos

# ============================================================
# 5. 叙事模板 → templates/*.json
# ============================================================
def parse_templates():
    path = os.path.join(BASE, "叙事模板.txt")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = {}
    # 解析 【section名（数量）】
    pattern = r'【(.+?)（(\d+)条）】\s*'
    splits = re.split(pattern, text)
    # splits[0] = before first match, splits[1]=name, splits[2]=count, splits[3]=content...
    i = 1
    while i + 2 < len(splits):
        name = splits[i].strip()
        count = splits[i + 1].strip()
        content = splits[i + 2].strip()
        # 按中文逗号/顿号分割句子
        sentences = [s.strip() for s in re.split(r'[，,、\n]', content) if s.strip() and len(s.strip()) > 1]
        sections[name] = sentences
        i += 3

    return sections

# ============================================================
# 写入 JSON
# ============================================================
def write_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [OK] {filepath} ({len(data)} entries)")

def main():
    print("构建知识库...")

    # 1. 颜色
    colors = parse_colors()
    write_json(os.path.join(BASE, "entities", "colors.json"), colors)

    # 2. 物象
    objects = parse_objects()
    write_json(os.path.join(BASE, "entities", "objects.json"), objects)

    # 3. 人物
    characters = parse_characters()
    write_json(os.path.join(BASE, "entities", "characters.json"), characters)

    # 4. 组合解读 → combinations/{entity1}_{entity2}.json（文件名即查询key）
    combos = parse_combinations()
    combo_dir = os.path.join(BASE, "combinations")
    # 清理旧文件
    if os.path.isdir(combo_dir):
        for old in os.listdir(combo_dir):
            if old.endswith('.json'):
                os.remove(os.path.join(combo_dir, old))
    for c in combos:
        safe_key = c["key"].replace("/", "_").replace("\\", "_")
        filepath = os.path.join(combo_dir, f"{safe_key}.json")
        write_json(filepath, {
            "entity1": c["entity1"],
            "entity2": c["entity2"],
            "meaning": c["meaning"],
            "interpretation": c["interpretation"],
        })
    print(f"  [OK] 组合解读: {len(combos)} 条")

    # 5. 叙事模板
    templates = parse_templates()
    write_json(os.path.join(BASE, "templates", "narrative.json"), templates)

    # 汇总
    print(f"\n=== 知识库构建完成 ===")
    print(f"  颜色实体: {len(colors)}")
    print(f"  物象实体: {len(objects)}")
    print(f"  人物实体: {len(characters)}")
    print(f"  组合解读: {len(combos)}")
    print(f"  叙事模板: {sum(len(v) for v in templates.values())} 条")


if __name__ == "__main__":
    main()
