"""
RAG 生成模块
负责调用阿里云百炼API生成叙事内容和图像
"""

import json
import os
import base64
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath

logger = logging.getLogger("NarrativeGenerator")

# 阿里云百炼 API
try:
    import dashscope
    from dashscope import Generation, ImageSynthesis
    from dashscope.aigc.image_generation import ImageGeneration
    from dashscope.api_entities.dashscope_response import Message
    from http import HTTPStatus
    import requests
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    ImageGeneration = None  # type: ignore
    Message = None  # type: ignore
    print("[警告] dashscope未安装，将使用占位符实现")


@dataclass
class GenerationConfig:
    """生成配置"""
    # 阿里云百炼API配置
    api_key: str = ""  # 通过环境变量 DASHSCOPE_API_KEY 设置

    # 模型选择
    realtime_model: str = "qwen-turbo-latest"   # 实时侧用 Turbo（快、便宜）
    narrative_model: str = "qwen3-max"          # 叙事卡用 Qwen3 Max（最高质量）
    image_model: str = "wan2.7-image-pro"       # 图像生成模型（Wan 2.7 支持文生图 + 图生图）

    # 生成参数
    max_tokens: int = 500
    temperature: float = 0.7

    # 图像参数
    image_size: str = "1024*1024"  # 默认分辨率
    image_n: int = 1               # 生成数量


# ---------------------------------------------------------------------------
# 人物 → 时代 → 视觉风格 映射表
# 基于 character_recommend.py 的人物分组
# 古人 → 古风 / 近现代 → 写实 / 跨时代 → 穿越风
# ---------------------------------------------------------------------------
CHARACTER_GROUP_STYLE = {
    # 古代 → 古风 (传统水墨)
    "理学脉络": {
        "era": "古代",
        "style": "古风",
        "visual_prompt": (
            "traditional Chinese ink wash painting, Song dynasty literati style, "
            "elegant sparse brushwork, rice paper texture, calligraphic sensibility, "
            "Zen aesthetics, misty mountains, subtle ink gradations, negative space composition"
        ),
    },
    # 近代 → 写实 (历史纪实)
    "湘军将帅": {
        "era": "近代",
        "style": "写实",
        "visual_prompt": (
            "historical realism, late Qing dynasty photography aesthetic, "
            "sepia and earth tones, documentary gravitas, dramatic chiaroscuro lighting, "
            "weathered textures, monumental composition"
        ),
    },
    "维新革命": {
        "era": "近代",
        "style": "写实",
        "visual_prompt": (
            "revolutionary realism, early 20th century Chinese woodcut print style, "
            "bold black and red contrasts, heroic composition, dynamic tension, "
            "propaganda poster sensibility, stark shadows"
        ),
    },
    # 现代 → 写实 (当代纪实)
    "现代学人": {
        "era": "现代",
        "style": "写实",
        "visual_prompt": (
            "modern Chinese academic realism, soft natural lighting, "
            "intellectual atmosphere, scholarly setting, warm muted tones, "
            "contemporary documentary photography, thoughtful composition"
        ),
    },
}

# 人物名 → 分组 的懒加载索引
_character_group_index: dict = {}


def _build_character_group_index():
    """构建 人物名→分组 索引"""
    global _character_group_index
    if _character_group_index:
        return
    for group_name, members in {
        "理学脉络": ["周敦颐", "程颢", "程颐", "胡安国", "胡宏", "朱熹", "张栻", "吕祖谦",
                     "陆九渊", "王阳明", "王夫之", "罗洪先"],
        "湘军将帅": ["曾国藩", "左宗棠", "胡林翼", "彭玉麟"],
        "维新革命": ["谭嗣同", "魏源", "黄兴", "蔡锷", "宋教仁", "陈天华"],
        "现代学人": ["毛泽东", "杨昌济", "何叔衡", "李达", "成仿吾", "周谷城", "何长工",
                     "熊十力", "冯友兰", "钱基博", "金岳霖", "梁漱溟", "胡庶华"],
    }.items():
        for name in members:
            _character_group_index[name] = group_name


def get_character_style(character_name: str) -> dict:
    """
    根据人物名获取视觉风格

    Returns:
        {"era": "古代", "style": "古风", "visual_prompt": "..."}
        未匹配时返回默认现代写实风格
    """
    _build_character_group_index()
    group = _character_group_index.get(character_name, "现代学人")
    return CHARACTER_GROUP_STYLE.get(group, CHARACTER_GROUP_STYLE["现代学人"])


class AliCloudGenerator:
    """
    阿里云百炼生成器
    通过dashscope API调用阿里云大模型
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        if DASHSCOPE_AVAILABLE and config.api_key:
            dashscope.api_key = config.api_key
        # 设置API地址（北京地域）
        if DASHSCOPE_AVAILABLE:
            dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    def _call_model(self, model: str, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        调用阿里云百炼文本模型

        Args:
            model: 模型名称
            prompt: 输入提示词
            max_tokens: 最大token数
            temperature: 温度参数

        Returns:
            模型生成的文本
        """
        if not DASHSCOPE_AVAILABLE:
            return f"[模拟输出] {prompt[:50]}..."

        try:
            response = Generation.call(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                result_format='message'
            )
            if response.status_code == 200:
                # output.text 可能为 None，优先从 choices[0].message.content 读取
                if (hasattr(response.output, 'choices')
                        and response.output.choices
                        and hasattr(response.output.choices[0], 'message')
                        and hasattr(response.output.choices[0].message, 'content')):
                    return response.output.choices[0].message.content
                return response.output.text or ""
            else:
                print(f"[错误] API调用失败: {response.message}")
                return ""
        except Exception as e:
            print(f"[错误] API调用异常: {e}")
            return ""

    def _download_image(self, image_url: str, save_path: str = None) -> Tuple[Optional[str], Optional[str]]:
        """
        下载图像到本地

        Args:
            image_url: 图像URL
            save_path: 保存路径（可选）

        Returns:
            (local_path, base64_string) 或 (None, None)
        """
        if not save_path:
            # 从URL提取文件名
            parsed = urlparse(image_url)
            filename = PurePosixPath(unquote(parsed.path)).parts[-1]
            save_path = f"./temp_{filename}"

        try:
            response = requests.get(image_url, stream=True, timeout=30)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return save_path, None
        except Exception as e:
            print(f"[错误] 图像下载失败: {e}")
            return None, None

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        将图像编码为Base64

        Args:
            image_path: 图像路径

        Returns:
            Base64字符串，格式: data:image/png;base64,xxx
        """
        try:
            with open(image_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
        except Exception as e:
            print(f"[错误] Base64编码失败: {e}")
            return None

    def generate_realtime_description(self, entity_name: str, entity_context: Dict) -> str:
        """
        实时生成单个模块的描述（用于前端即时显示）

        Args:
            entity_name: 实体名称
            entity_context: 实体的上下文信息

        Returns:
            一句润色后的描述
        """
        description = entity_context.get("description", "")
        symbolism = entity_context.get("symbolism", "")

        prompt = f"""你是一个湖湘文化解说员。请为以下文化元素生成一句简短（20字以内）的描述。

元素：{entity_name}
含义：{symbolism}
基础描述：{description}

要求：
- 语言优美，有文化韵味
- 直接点明精神内涵
- 格式：一句完整的话

输出："""

        return self._call_model(
            model=self.config.realtime_model,
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )

    def generate_connection_description(self, from_entity: str, to_entity: str,
                                       connection_type: str, meaning: str) -> str:
        """
        生成连接关系的描述

        Args:
            from_entity: 起始实体
            to_entity: 目标实体
            connection_type: 连接类型
            meaning: 已有含义描述

        Returns:
            润色后的连接描述
        """
        prompt = f"""你是一个湖湘文化解说员。请为以下连接关系生成一句描述。

连接：{from_entity} ↔ {to_entity}
类型：{connection_type}
含义：{meaning}

要求：
- 一句话，20字以内
- 体现两个元素之间的精神联结
- 语言优美

输出："""

        return self._call_model(
            model=self.config.realtime_model,
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )

    def _build_narrative_facts(self, modules: List[Dict], connections: List[Dict]) -> str:
        """
        从 modules 和 connections 中提取知识库关键事实，
        供 Few-shot ICL 示例和 Hallucination 约束使用。
        """
        lines = []
        for m in modules:
            entity = m.get("entity", "")
            kind = m.get("type", "")
            desc = m.get("description", "")
            symbol = m.get("symbolism", "")
            hist = m.get("historical_context", "")
            era = m.get("era", "")
            mood = m.get("mood", "")

            fact_parts = [f"{entity}（{kind}）"]
            if desc:
                fact_parts.append(f"描述：{desc}")
            if symbol:
                fact_parts.append(f"象征意义：{symbol}")
            if hist:
                fact_parts.append(f"历史背景：{hist}")
            if era:
                fact_parts.append(f"所属时代：{era}")
            if mood:
                fact_parts.append(f"情感基调：{mood}")
            lines.append(" | ".join(fact_parts))

        for c in connections:
            from_e = c.get("from", "")
            to_e = c.get("to", "")
            meaning = c.get("meaning", "")
            if from_e and to_e and meaning:
                lines.append(f"连接「{from_e}」与「{to_e}」：{meaning}")

        return "\n".join(lines) if lines else "（知识库暂无详细记录）"

    def _verify_hallucination(self, narrative_result: Dict,
                              modules: List[Dict],
                              connections: List[Dict]) -> List[str]:
        """
        校验叙事内容是否存在幻觉（幻觉的内容来自用户输入，不在知识库中）。

        检查策略：从叙事中提取人名/年代/地点，与知识库做字符串匹配。
        匹配失败的条目记录为"潜在幻觉"。

        Returns:
            潜在幻觉描述列表（空列表表示通过校验）
        """
        warnings = []

        # 提取知识库中的实体名称集
        known_names = set()
        for m in modules:
            for key in ["entity", "name"]:
                val = m.get(key, "").strip()
                if val:
                    known_names.add(val)

        # 合并 connections 中的实体
        for c in connections:
            for key in ["from", "to"]:
                val = c.get(key, "").strip()
                if val:
                    known_names.add(val)

        # 从段落文本中简单抽取"疑似人名"（2-4字且非颜色/物象名的词）
        text_blob = "".join(narrative_result.get("paragraphs", []))
        # 粗筛：长度2-4的连续汉字串
        import re
        suspected_names = re.findall(r"[一-鿿]{2,4}", text_blob)
        for name in suspected_names:
            if name not in known_names and name not in ["湖大", "岳麓", "湘江", "书院"]:
                warnings.append(f"未识别实体「{name}」，请确认是否捏造")

        return warnings

    def generate_narrative(self, context: Dict) -> Dict:
        """
        生成完整叙事（用于叙事卡）

        Args:
            context: RAG检索到的完整上下文
            {
                "modules": [...],
                "connections": [...],
                "title": "你寻到的千年色"
            }

        Returns:
            {
                "title": "...",
                "paragraphs": ["第一段", "第二段", ...],
                "summary": "...",
                "hallucination_warnings": [...]  # 潜在幻觉提示（供调试/人工审核）
            }
        """
        modules = context.get("modules", [])
        connections = context.get("connections", [])

        # 构建知识库事实（用于 ICL 示例 + Hallucination 约束）
        facts_text = self._build_narrative_facts(modules, connections)

        # 构建模块列表描述
        module_list = "\n".join([
            f"- {m.get('entity', '')}（{m.get('type', '')}）：{m.get('description', '')}"
            for m in modules
        ])

        # 构建连接描述
        conn_list = "\n".join([
            f"- {c.get('from', '')} ↔ {c.get('to', '')}：{c.get('meaning', '')}"
            for c in connections
        ]) if connections else "无"

        # Few-shot ICL 示例（1 对）
        few_shot_example = """示例：
输入：
- 岳麓绿（颜色）：岳麓书院的苍松翠柏，象征理学文脉的生生不息
- 书院（物象）：千年学府，湖湘文化的精神源地
- 朱熹（人物）：南宋理学家，曾主讲岳麓书院，推动湘学发展
连接「岳麓绿」与「书院」：绿意映衬下的书院，理学精神薪火相传

输出格式（JSON）：
{
    "title": "岳麓松风",
    "paragraphs": [
        "我站在岳麓书院的石阶前，眼前是一片苍翠欲滴的绿。这绿不同于别处，它承载着千年理学的文脉，</n>仿佛朱熹当年讲学时的那片松林依然在这里守望。",
        "书院的青瓦白墙在绿荫掩映下愈显古朴。我仿佛听见远山的钟声穿过松涛，</n>那是跨越千年的呼唤——从周敦颐的太极到朱熹的注解，理学在此一脉相承。",
        "我伸手触碰石柱上的苔痕，凉意沁入掌心。这绿不仅是颜色，更是湖湘士人的精神底色——</n>「诚朴」二字，在这一片苍翠中悄然浮现。",
        "当湘江的水汽随风送来，我感到自己已与这座书院融为一体。</n>我是过客，亦是归人；我寻色，亦在寻根。",
        "最终，我将这一抹岳麓绿收入心底，作为湖大千年色最深的印记：</n>它不仅是自然之色，更是理学之魂在此地的永驻。"
    ],
    "summary": "岳麓松风，绿染千年，理学文脉在此生生不息。"
}"""

        prompt = f"""你是一个湖湘文化叙事作家。请根据以下元素组合，生成一段个性化的湖大千年色叙事。

【知识库事实】（你必须严格基于以下事实创作，不得捏造任何不在此列的人物、年代、事件）
{facts_text}

【用户选择的元素】
{module_list}

【元素之间的连接】
{conn_list}

【叙事主题】你寻到的千年色

{few_shot_example}

要求：
- 4-5段文字，娓娓道来
- 从"我"的角度叙述，有代入感
- 必须严格基于【知识库事实】中的内容创作，**不得捏造**未列出的人名、历史事件或象征意义
- 融入湖湘文化和湖大历史
- 每段2-3句话
- 最后一段呼应主题，升华情感
- 语言优美，有文学性

输出格式（JSON）：
{{
    "title": "你寻到的千年色",
    "paragraphs": ["第一段内容", "第二段内容", ...],
    "summary": "一句话总结"
}}"""

        result = self._call_model(
            model=self.config.narrative_model,
            prompt=prompt,
            max_tokens=800,
            temperature=0.8
        )

        if not result:
            return {
                "title": "你寻到的千年色",
                "paragraphs": ["[生成失败，请稍后重试]"],
                "summary": "",
                "hallucination_warnings": []
            }

        # 尝试解析JSON
        try:
            # 提取JSON部分（可能包含在markdown代码块中）
            if "```json" in result:
                start = result.find("```json") + 7
                end = result.find("```", start)
                result = result[start:end]
            elif "```" in result:
                start = result.find("```") + 3
                end = result.find("```", start)
                result = result[start:end]

            parsed = json.loads(result)
            # Hallucination 校验
            parsed["hallucination_warnings"] = self._verify_hallucination(
                parsed, modules, connections
            )
            return parsed
        except json.JSONDecodeError:
            # 解析失败，返回原始文本
            return {
                "title": "你寻到的千年色",
                "paragraphs": [result],
                "summary": "",
                "hallucination_warnings": []
            }

    def generate_image_prompt(self, context: Dict) -> str:
        """
        生成图像提示词（用于AI作画）

        Args:
            context: RAG检索上下文

        Returns:
            英文图像提示词
        """
        modules = context.get("modules", [])
        entities = [m.get("entity", "") for m in modules]

        prompt = f"""请为以下湖大千年色元素生成一幅Stable Diffusion图像提示词。

元素：{"、".join(entities)}

要求：
- 风格：中国水墨画 + 穿越感
- 包含元素：岳麓书院、湘江、人物（抽象人影/光柱）
- 色调：用户选择的颜色晕染成水墨背景
- 氛围：人文与山水交融，历史感与当代感并存
- 输出：英文提示词，200词以内
- 不要包含任何文字或字母

输出："""

        return self._call_model(
            model=self.config.realtime_model,
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )

    # ------------------------------------------------------------------
    # Wan 2.7 图像生成（异步：提交任务 → 轮询等待 → 下载结果）
    # ------------------------------------------------------------------

    def _generate_image_wan(self, prompt: str, size: str = None, n: int = 1,
                            base_image: str = None) -> Dict:
        """
        Wan 2.7 异步图像生成（文生图 / 图生图）

        Args:
            prompt: 图像提示词（英文）
            size: 分辨率，如 "2048*2048"
            n: 生成数量
            base_image: 可选，底图 Base64 或 URL。传入则走图生图模式

        Returns:
            {"status": "success"/"error", "images": [...], "error": "..."}
        """
        if not DASHSCOPE_AVAILABLE or ImageGeneration is None:
            return {"status": "error", "error": "dashscope 未安装"}

        if not self.config.api_key:
            return {"status": "error", "error": "未配置 API_KEY"}

        size = size or self.config.image_size
        n = n or self.config.image_n

        try:
            # 构建消息内容
            content = []
            if base_image:
                content.append({"image": base_image})
            content.append({"text": prompt})

            message = Message(role="user", content=content)

            # 提交异步任务
            logger.info(f"[Wan] 提交{'图生图' if base_image else '文生图'}任务...")
            response = ImageGeneration.async_call(
                model=self.config.image_model,
                api_key=self.config.api_key,
                messages=[message],
                n=n,
                size=size,
            )

            if response.status_code != 200:
                return {
                    "status": "error",
                    "error": f"任务提交失败: {response.code} - {response.message}"
                }

            task_id = response.output.task_id
            logger.info(f"[Wan] 任务已提交, task_id={task_id}")

            # 轮询等待任务完成（最长等待 2 分钟）
            result = ImageGeneration.wait(task=response, api_key=self.config.api_key)
            logger.info(f"[Wan] 任务完成, status={result.output.task_status}")

            if result.output.task_status == "SUCCEEDED":
                logger.info(f"[Wan] choices count: {len(result.output.choices)}")
                images = []
                for choice in result.output.choices:
                    for content_item in choice.message.content:
                        # content_item may be dict or object
                        if isinstance(content_item, dict):
                            image_url = content_item.get("image", "")
                        elif hasattr(content_item, 'image'):
                            image_url = content_item.image or ""
                        else:
                            logger.warning(f"[Wan] unknown content_item type: {type(content_item)}")
                            continue

                        if image_url:
                            local_path, _ = self._download_image(image_url)
                            base64_str = None
                            if local_path:
                                base64_str = self._encode_image_to_base64(local_path)
                            images.append({
                                "image_url": image_url,
                                "local_path": local_path,
                                "base64": base64_str,
                            })

                logger.info(f"[Wan] 生成完成, 共 {len(images)} 张")
                return {"status": "success", "images": images}
            else:
                # 尝试从 result 中获取更多错误信息
                err_msg = str(result.output.task_status)
                if hasattr(result, 'code') and result.code:
                    err_msg += f" | code={result.code}"
                if hasattr(result, 'message') and result.message:
                    err_msg += f" | message={result.message}"
                return {
                    "status": "error",
                    "error": f"任务失败: {err_msg}"
                }

        except Exception as e:
            logger.error(f"[Wan] 图像生成异常: {e}")
            return {"status": "error", "error": f"图像生成异常: {str(e)}"}

    def generate_image(self, prompt: str, size: str = None, n: int = 1) -> Dict:
        """
        图像生成（Wan 2.7 异步，文生图）

        Args:
            prompt: 图像提示词（英文）
            size: 分辨率，默认 "2048*2048"
            n: 生成数量

        Returns:
            {"status": "success"/"error", "images": [...], "error": "..."}
        """
        return self._generate_image_wan(prompt, size, n)

    def generate_image_with_base(self, prompt: str, base_image: str,
                                  size: str = None) -> Dict:
        """
        图生图：基于底图 + 风格 prompt 生成新图像

        颜色晕染底图（前端提供）+ 人物风格 prompt → 融合图像

        Args:
            prompt: 风格提示词（英文，由人物视觉风格决定）
            base_image: 底图 Base64 字符串（data:image/png;base64,...）或 URL
            size: 分辨率，默认 "2048*2048"

        Returns:
            同 generate_image
        """
        return self._generate_image_wan(prompt, size, n=1, base_image=base_image)

    # ------------------------------------------------------------------
    # 人物风格 prompt 生成
    # ------------------------------------------------------------------

    def build_character_style_prompt(self, character_name: str,
                                      color_name: str = "",
                                      base_prompt: str = "") -> str:
        """
        根据人物 + 颜色构建图生图的风格 prompt

        Args:
            character_name: 人物名（如 "朱熹", "毛泽东"）
            color_name: 颜色名（如 "岳麓绿", "书院红"）
            base_prompt: 基础场景 prompt（可选，来自 generate_image_prompt）

        Returns:
            英文风格 prompt
        """
        style = get_character_style(character_name)
        visual = style["visual_prompt"]

        parts = [visual]

        if color_name:
            parts.append(f"dominant color atmosphere inspired by {color_name}")

        if base_prompt:
            parts.append(f"scene: {base_prompt[:200]}")

        parts.append("high quality, artistic, museum exhibition grade")

        return ", ".join(parts)

    def generate_image_sync(self, prompt: str, size: str = None) -> Dict:
        """同步生成（封装异步调用，阻塞等待）"""
        return self.generate_image(prompt, size, n=1)


class LocalGenerator:
    """
    本地小模型生成器（备用）
    当云端API不可用时使用本地模型
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    def generate_description(self, entity_name: str, context: Dict) -> str:
        """生成描述（本地备用）"""
        return context.get("description", f"{entity_name}。")

    def generate_connections_description(self, connections: List[Dict]) -> str:
        """生成连接描述（本地备用）"""
        descriptions = []
        for conn in connections:
            if conn.get("meaning"):
                descriptions.append(conn["meaning"])
        return "；".join(descriptions) if descriptions else ""


class NarrativeGenerator:
    """
    叙事生成器
    整合阿里云百炼和本地生成器
    """

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.ali_gen = AliCloudGenerator(self.config)
        self.local_gen = LocalGenerator(self.config)

    def generate_realtime_description(self, entity_name: str, entity_context: Dict) -> str:
        """
        实时生成单个描述（优先使用阿里云百炼）

        Args:
            entity_name: 实体名称
            entity_context: 检索上下文

        Returns:
            描述文本
        """
        if DASHSCOPE_AVAILABLE and self.config.api_key:
            return self.ali_gen.generate_realtime_description(entity_name, entity_context)
        else:
            return self.local_gen.generate_description(entity_name, entity_context)

    def generate_for_cloud(self, context: Dict) -> Dict:
        """
        为云端生成准备数据

        Args:
            context: RAG检索上下文

        Returns:
            发送给云端API的完整数据
        """
        modules = context.get("modules", [])
        connections = context.get("connections", [])

        # 生成本地连接描述（用于前端即时显示）
        connection_descriptions = []
        for conn in connections:
            from_entity = conn.get("from", "")
            to_entity = conn.get("to", "")
            conn_type = conn.get("connection_type", "")
            meaning = conn.get("meaning", "")

            desc = self.ali_gen.generate_connection_description(
                from_entity, to_entity, conn_type, meaning
            ) if DASHSCOPE_AVAILABLE and self.config.api_key else meaning

            connection_descriptions.append(desc)

        # 生成图像提示词
        image_prompt = self.ali_gen.generate_image_prompt(context) if DASHSCOPE_AVAILABLE and self.config.api_key else ""

        return {
            "modules": modules,
            "connections": connections,
            "connection_descriptions": connection_descriptions,
            "cloud_prompt": image_prompt
        }

    def generate_complete_narrative(self, context: Dict) -> Dict:
        """
        生成完整叙事（阿里云百炼）

        Args:
            context: RAG检索上下文

        Returns:
            完整叙事数据
        """
        if DASHSCOPE_AVAILABLE and self.config.api_key:
            return self.ali_gen.generate_narrative(context)
        else:
            return {
                "title": "你寻到的千年色",
                "paragraphs": ["[请配置阿里云百炼API_KEY]"],
                "summary": ""
            }

    def generate_image(self, prompt: str, size: str = None) -> Dict:
        """
        文生图（Wan 2.7 异步）

        Args:
            prompt: 图像提示词（英文）
            size: 分辨率

        Returns:
            图像结果字典
        """
        return self.ali_gen.generate_image(prompt, size)

    def generate_image_with_base(self, prompt: str, base_image: str,
                                  size: str = None) -> Dict:
        """
        图生图：颜色晕染底图 + 人物风格 prompt → 融合图像

        Args:
            prompt: 风格提示词（英文）
            base_image: 底图 Base64 或 URL（前端提供颜色晕染图）
            size: 分辨率

        Returns:
            图像结果字典
        """
        return self.ali_gen.generate_image_with_base(prompt, base_image, size)

    def build_character_style_prompt(self, character_name: str,
                                      color_name: str = "",
                                      base_prompt: str = "") -> str:
        """
        根据人物 + 颜色构建图生图风格 prompt

        Args:
            character_name: 人物名
            color_name: 颜色名
            base_prompt: 基础场景 prompt

        Returns:
            英文风格 prompt
        """
        return self.ali_gen.build_character_style_prompt(
            character_name, color_name, base_prompt
        )


def create_generator(config: GenerationConfig = None) -> NarrativeGenerator:
    """
    创建叙事生成器

    Args:
        config: 生成配置

    Returns:
        NarrativeGenerator实例
    """
    return NarrativeGenerator(config)


def create_config(api_key: str = None) -> GenerationConfig:
    """
    创建生成配置

    Args:
        api_key: 阿里云百炼API Key（优先从环境变量读取）

    Returns:
        GenerationConfig实例
    """
    return GenerationConfig(
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY", "")
    )


# 测试函数
def test_image_generation():
    """测试图像生成"""
    config = create_config()
    generator = NarrativeGenerator(config)

    # 测试prompt生成
    test_context = {
        "modules": [
            {"entity": "岳麓绿", "type": "color", "description": "生命的颜色"},
            {"entity": "书院", "type": "object", "description": "千年学府"}
        ],
        "connections": []
    }

    prompt = generator.ali_gen.generate_image_prompt(test_context)
    print(f"生成的提示词: {prompt}")

    # 测试图像生成
    print("\n开始生成图像...")
    result = generator.generate_image(prompt)

    if result["status"] == "success":
        for i, img in enumerate(result["images"]):
            print(f"\n图像{i+1}:")
            print(f"  URL: {img['image_url']}")
            print(f"  本地路径: {img['local_path']}")
            print(f"  Base64长度: {len(img['base64']) if img['base64'] else 0}")
    else:
        print(f"生成失败: {result['error']}")


if __name__ == "__main__":
    test_image_generation()
