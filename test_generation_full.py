"""
完整生成测试：颜色 + 物象 + 人物 → 叙事 + 图生图 → 明信片
用户选择黄色（西迁黄），底部图 test.png 来自 Unity 颜色晕染
"""
import sys, os, json, base64, time
from datetime import datetime

# 确保项目根目录在 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = r"C:\Users\lenovo\Desktop"
BASE_IMAGE = os.path.join(OUTPUT_DIR, "test.png")

# ── 测试配置（用户选黄色，物象随机但复杂） ──
COLOR = "西迁黄"
OBJECTS = ["道路", "石阶", "行李箱", "背包"]  # 西迁主题物象


def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def main():
    print("=" * 60)
    print("  寻麓千年色 · 完整生成测试")
    print(f"  颜色: {COLOR}  物象: {', '.join(OBJECTS)}")
    print(f"  底图: {BASE_IMAGE}")
    print("=" * 60)

    # ── 1. 初始化 RAG 系统 ──
    print("\n[1/6] 初始化 RAG 系统...")
    from rag import RAGSystem
    rag = RAGSystem()
    rag.setup()

    # ── 2. 人物推荐 ──
    print("\n[2/6] 人物推荐...")
    from rag.character_recommend import CharacterRecommender
    recommender = CharacterRecommender()
    results = recommender.recommend(COLOR, OBJECTS, use_llm=False, top_k=3)

    if not results:
        print("[错误] 人物推荐为空")
        return

    print(f"  Top-3 人物:")
    for i, r in enumerate(results):
        print(f"    {i+1}. {r.name}（{r.title}） score={r.score:.3f}  {r.reason}")

    # 取 top-1
    character = results[0]
    print(f"\n  选中人物: {character.name}（{character.title}）")

    # ── 3. 生成叙事文本 ──
    print("\n[3/6] 生成叙事文本...")
    # 构建上下文
    context = {
        "modules": [
            {"entity": COLOR, "type": "color", "description": "西迁之路的颜色"},
        ] + [
            {"entity": obj, "type": "object", "description": obj}
            for obj in OBJECTS
        ],
        "connections": [],
    }

    from rag.generator import create_config, create_generator
    config = create_config()
    gen = create_generator(config)

    narrative = gen.generate_complete_narrative(context)
    title = narrative.get("title", "你寻到的千年色")
    paragraphs = narrative.get("paragraphs", [])
    summary = narrative.get("summary", "")

    # 如果 LLM 生成失败（额度用完等），使用模板降级
    if not paragraphs or paragraphs[0].startswith("["):
        print("  [降级] 文本 API 不可用，使用模板叙事")
        paragraphs = [
            f"我选择了{COLOR}，那是西迁路上最温暖的记忆。",
            f"沿着{OBJECTS[0]}，踏过{OBJECTS[1]}，",
            f"背着{OBJECTS[3]}，提着{OBJECTS[2]}，",
            f"一代代湖大人走过了最艰难的岁月。",
        ]
        summary = f"{COLOR}如灯，照亮历史的道路。"
        narrative = {
            "title": "你寻到的千年色",
            "paragraphs": paragraphs,
            "summary": summary,
        }
        title = narrative["title"]

    print(f"  标题: {title}")
    for i, p in enumerate(paragraphs):
        print(f"  段{i+1}: {p[:50]}...")
    if summary:
        print(f"  总结: {summary}")

    # ── 4. 图像生成 ──
    print("\n[4/6] 图像生成...")

    # 4a. 生成场景 prompt（文本API可能不可用）
    print("  生成英文场景 prompt...")
    image_prompt = gen.ali_gen.generate_image_prompt(context)
    if not image_prompt or image_prompt.startswith("["):
        print("  [降级] 使用模板 prompt")
        image_prompt = (
            f"ancient road winding through mountains, students carrying luggage, "
            f"warm yellow sunlight, Chinese wartime university migration, "
            f"historical documentary style, stone steps, backpacks, suitcases"
        )
    print(f"  场景 prompt: {image_prompt[:120]}...")

    # 4b. 构建人物风格 prompt
    print("  构建人物风格 prompt...")
    character_style = gen.build_character_style_prompt(
        character_name=character.name,
        color_name=COLOR,
        base_prompt=image_prompt,
    )
    print(f"  风格 prompt: {character_style[:150]}...")
    print(f"  风格来源: {character.name}")

    # 4c. 图生图（底图 + 风格 prompt）
    if os.path.exists(BASE_IMAGE):
        print(f"  [图生图模式] 加载底图 {BASE_IMAGE}...")
        base64_img = encode_image_base64(BASE_IMAGE)
        print(f"  底图 Base64 长度: {len(base64_img)}")

        print("  提交 Wan 2.7 图生图任务（预计 30-60 秒）...")
        t0 = time.time()
        img_result = gen.generate_image_with_base(character_style, base64_img)
        elapsed = time.time() - t0
    else:
        print(f"  [文生图模式] 底图不存在，使用纯文本生成...")
        t0 = time.time()
        img_result = gen.generate_image(character_style)
        elapsed = time.time() - t0

    print(f"  耗时: {elapsed:.1f}s")

    if img_result["status"] == "success":
        images = img_result.get("images", [])
        print(f"  生成 {len(images)} 张图像")
        for i, img in enumerate(images):
            print(f"    图{i+1}: URL={img.get('image_url', 'N/A')[:60]}...")
            local = img.get("local_path")
            if local:
                print(f"           本地={local}")
                print(f"           Base64长度={len(img.get('base64', '') or '')}")
            else:
                print(f"           本地=下载失败")
        generated_img = images[0] if images else None
    else:
        print(f"  [失败] {img_result.get('error', 'unknown')}")
        generated_img = None

    # ── 5. 明信片合成 ──
    print("\n[5/6] 明信片合成...")
    from rag.postcard import PostcardGenerator

    postcard_gen = PostcardGenerator()

    # 用生成的图像作为底图
    image_source = None
    if generated_img:
        image_source = generated_img.get("base64") or generated_img.get("image_url")

    if image_source:
        postcard = postcard_gen.create_postcard(
            narrative_result=narrative,
            image_source=image_source,
        )
        if postcard:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            postcard_path = os.path.join(OUTPUT_DIR, f"postcard_{timestamp}.png")
            postcard_gen.save(postcard, postcard_path)
            print(f"  明信片已保存: {postcard_path}")
        else:
            print("  明信片合成失败（可能缺字体）")
            # 降级：直接保存生成的原图
            if generated_img and generated_img.get("local_path"):
                postcard_path = generated_img["local_path"]
                print(f"  降级使用原图: {postcard_path}")
    else:
        # 无图像，只生成文字版明信片
        print("  无图像，生成纯文字明信片...")
        postcard = postcard_gen.create_postcard(narrative_result=narrative)
        if postcard:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            postcard_path = os.path.join(OUTPUT_DIR, f"postcard_text_{timestamp}.png")
            postcard_gen.save(postcard, postcard_path)
            print(f"  明信片已保存: {postcard_path}")

    # ── 6. 保存完整 JSON ──
    print("\n[6/6] 保存完整结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"generation_{timestamp}.json")

    result_data = {
        "color": COLOR,
        "objects": OBJECTS,
        "character": {
            "name": character.name,
            "title": character.title,
            "score": character.score,
            "reason": character.reason,
        },
        "narrative": narrative,
        "image_prompt": image_prompt,
        "character_style_prompt": character_style,
        "image_result_status": img_result.get("status", "error"),
        "generated_at": datetime.now().isoformat(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"  JSON 已保存: {json_path}")

    print("\n" + "=" * 60)
    print("  测试完成！输出文件：")
    print(f"    桌面目录: {OUTPUT_DIR}")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fpath) and timestamp in f:
            size_kb = os.path.getsize(fpath) / 1024
            print(f"      {f}  ({size_kb:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
