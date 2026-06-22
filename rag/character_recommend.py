"""
人物推荐引擎
颜色 + 物象 → RAG 检索人物知识库 → 启发式排序 → LLM 精选 Top-3

用法:
    from rag.character_recommend import CharacterRecommender
    recommender = CharacterRecommender()
    results = recommender.recommend(color="岳麓绿", objects=["古树", "讲堂"])
    # → [RecommendResult(name="王夫之", score=0.85, reason="..."), ...]
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class RecommendResult:
    """单条推荐结果"""
    name: str               # 人物名
    title: str              # 称号
    score: float            # 综合得分 (0~1)
    reason: str             # 推荐理由
    monologue: list = field(default_factory=list)   # 第一人称台词
    spiritLine: str = ""    # 精神金句


# ---------------------------------------------------------------------------
# 颜色 → 人物 spirit 关联词表
# 当知识库实体的 related_entities / spirit 字段不足时，作为补充检索词
# ---------------------------------------------------------------------------
COLOR_TO_SPIRIT_KEYWORDS: Dict[str, List[str]] = {
    # 第一组 · 亮色系
    "朱红": ["热烈", "赤诚", "初心", "讲学", "思想", "革命", "热血", "精神", "历史"],
    "灯橙": ["温暖", "灯火", "守望", "关怀", "教育", "传承", "师承", "坚守"],
    "梨黄": ["明净", "收获", "沉淀", "学术", "经典", "书院", "文化", "传统"],
    "叶绿": ["生长", "生机", "蔓延", "根脉", "理学", "自然", "教育", "学术"],
    "瓷青": ["清透", "澄澈", "明镜", "哲学", "思考", "开放", "探索", "包容"],
    "海蓝": ["深邃", "探索", "求知", "远方", "真理", "学问", "研究", "理论"],
    "烟紫": ["神秘", "思辨", "超越", "智慧", "创新", "引领", "发展", "未来"],
    # 第二组 · 浓色系
    "枫红": ["深沉", "厚重", "积淀", "历史", "精神", "坚守", "责任", "担当"],
    "暖橙": ["醇厚", "温情", "守候", "教育", "师承", "办学", "文脉", "传统"],
    "藤黄": ["古雅", "文脉", "传承", "经典", "书院", "学术", "文化", "奋斗"],
    "玉绿": ["温润", "清雅", "含蓄", "理学", "自然", "生长", "教育", "思想"],
    "石青": ["沉稳", "坚定", "磐石", "坚守", "哲学", "信仰", "探索", "学术"],
    "澄蓝": ["纯粹", "追问", "求真", "真理", "思考", "研究", "严谨", "知识"],
    "影紫": ["幽深", "内省", "潜思", "智慧", "理论", "学术", "创新", "超越"],
    # 第三组 · 柔色系
    "桃红": ["柔和", "青春", "绽放", "教育", "传承", "希望", "理想", "未来"],
    "夕橙": ["黄昏", "余韵", "回望", "历史", "记忆", "坚守", "师承", "传统"],
    "桂黄": ["金桂", "荣光", "远方", "书院", "文化", "引领", "发展", "学术"],
    "茶绿": ["淡雅", "从容", "澄怀", "自然", "生长", "教育", "理学", "思想"],
    "湖青": ["涟漪", "流动", "包容", "开放", "探索", "哲学", "远方", "流水"],
    "沧蓝": ["苍茫", "广阔", "志向", "真理", "学术", "研究", "理论", "知识"],
    "黛紫": ["典雅", "深邃", "余韵", "智慧", "思辨", "创新", "超越", "历史"],
}


# ---------------------------------------------------------------------------
# 人物分组（用于已选人物的上下文加权）
# ---------------------------------------------------------------------------
CHARACTER_GROUPS: Dict[str, List[str]] = {
    "理学脉络": ["周敦颐", "程颢", "程颐", "胡安国", "胡宏", "朱熹", "张栻", "吕祖谦",
                 "陆九渊", "王阳明", "王夫之", "罗洪先"],
    "湘军将帅": ["曾国藩", "左宗棠", "胡林翼", "彭玉麟"],
    "维新革命": ["谭嗣同", "魏源", "黄兴", "蔡锷", "宋教仁", "陈天华"],
    "现代学人": ["毛泽东", "杨昌济", "何叔衡", "李达", "成仿吾", "周谷城", "何长工",
                 "熊十力", "冯友兰", "钱基博", "金岳霖", "梁漱溟"],
}


class CharacterRecommender:
    """
    人物推荐引擎

    pipeline:
        颜色语义 + 物象关联 → 97 人物粗筛 → LLM 精选 Top-3

    得分组成（共 ~1.0）:
      - 内置参考表命中 (0~0.50) : 已知的颜色+物象→人物映射
      - 已选人物同组加权 (0~0.20) : 同理学脉/湘军/维新等
      - 关键词文本匹配 (0~0.25) : 颜色关键词/物象名在人物文本中的命中
      - 实体基础分 (0~0.05) : description 长度等
    """

    # ---- 内置参考表（颜色+物象 → 推荐人物）----
    REFERENCE_TABLE: Dict[str, Dict[str, List[str]]] = {
        # 第一组 · 亮色系
        "朱红": {"岳麓书院": ["张栻", "朱熹", "王夫之"], "红墙": ["张栻", "王夫之"], "讲堂": ["朱熹", "张栻"]},
        "灯橙": {"岳麓书院": ["胡宏", "张栻", "周敦颐"], "钟楼": ["朱熹", "张栻"]},
        "梨黄": {"岳麓书院": ["钱基博", "冯友兰", "李达"], "书卷": ["李达", "杨昌济"]},
        "叶绿": {"岳麓书院": ["胡安国", "胡宏", "周敦颐"], "古树": ["王夫之", "胡宏"], "竹林": ["周敦颐", "程颢"]},
        "瓷青": {"岳麓书院": ["魏源", "周敦颐", "王夫之"], "湖面": ["王夫之", "周敦颐"]},
        "海蓝": {"岳麓书院": ["朱熹", "王夫之", "魏源"], "湘江": ["曾国藩", "毛泽东"], "流水": ["魏源", "谭嗣同"]},
        "烟紫": {"岳麓书院": ["王阳明", "陆九渊", "熊十力"], "碑刻": ["王夫之", "曾国藩"]},
        # 第二组 · 浓色系
        "枫红": {"岳麓书院": ["谭嗣同", "黄兴", "蔡锷"], "红墙": ["张栻", "朱熹"]},
        "暖橙": {"岳麓书院": ["杨昌济", "何叔衡", "毛泽东"], "林荫道": ["朱熹", "张栻"]},
        "藤黄": {"岳麓书院": ["胡庶华", "成仿吾", "何长工"], "道路": ["胡庶华", "成仿吾"]},
        "玉绿": {"岳麓书院": ["程颢", "程颐", "胡安国"], "古树": ["王夫之", "胡安国"]},
        "石青": {"岳麓书院": ["吕祖谦", "罗洪先", "周敦颐"], "石阶": ["何叔衡", "杨昌济"]},
        "澄蓝": {"岳麓书院": ["曾国藩", "左宗棠", "胡林翼"], "桥梁": ["曾国藩", "左宗棠"], "湘江": ["左宗棠", "毛泽东"]},
        "影紫": {"岳麓书院": ["金岳霖", "梁漱溟", "冯友兰"], "书卷": ["李达", "魏源"]},
        # 第三组 · 柔色系
        "桃红": {"岳麓书院": ["宋教仁", "陈天华", "谭嗣同"], "讲堂": ["朱熹", "周敦颐"]},
        "夕橙": {"岳麓书院": ["彭玉麟", "胡林翼", "曾国藩"], "钟楼": ["张栻", "朱熹"]},
        "桂黄": {"岳麓书院": ["毛泽东", "杨昌济", "李达"], "图书馆": ["钱基博", "冯友兰"]},
        "茶绿": {"岳麓书院": ["周敦颐", "胡宏", "程颢"], "竹林": ["程颐", "周敦颐"]},
        "湖青": {"岳麓书院": ["魏源", "谭嗣同", "王夫之"], "流水": ["魏源", "谭嗣同"], "湖面": ["周敦颐", "王夫之"]},
        "沧蓝": {"岳麓书院": ["王夫之", "朱熹", "曾国藩"], "碑刻": ["曾国藩", "王夫之"]},
        "黛紫": {"岳麓书院": ["熊十力", "金岳霖", "梁漱溟"], "书卷": ["杨昌济", "李达"]},
    }

    # ---- 核心人物精确数据（TXT 解析器存在格式缺陷，此处手动维护）----
    CORE_CHARACTERS: List[Dict] = [
        # 古代先贤
        {"name":"朱熹","title":"理学大师","description":"在岳麓书院讲学","spirit":"思想传播","group":"理学脉络"},
        {"name":"张栻","title":"山长","description":"主持书院发展","spirit":"教育传承","group":"理学脉络"},
        {"name":"王夫之","title":"思想家","description":"湖湘学派集大成者","spirit":"理论建构","group":"理学脉络"},
        {"name":"周敦颐","title":"理学开创者","description":"奠定学脉，《爱莲说》作者","spirit":"学术源头","group":"理学脉络"},
        {"name":"程颢","title":"理学宗师","description":"洛学开创，思想传播","spirit":"温润教化","group":"理学脉络"},
        {"name":"程颐","title":"理学宗师","description":"体系构建，严谨治学","spirit":"格物致知","group":"理学脉络"},
        {"name":"胡安国","title":"经学家","description":"湖湘学派奠基","spirit":"学术开拓","group":"理学脉络"},
        {"name":"胡宏","title":"思想家","description":"理学深化发展","spirit":"思想深化","group":"理学脉络"},
        {"name":"吕祖谦","title":"学者","description":"讲学交流，理学传播","spirit":"学术交流","group":"理学脉络"},
        {"name":"陆九渊","title":"心学家","description":"思想争鸣，心即理","spirit":"思辨创新","group":"理学脉络"},
        {"name":"王阳明","title":"心学大师","description":"知行合一","spirit":"实践哲学","group":"理学脉络"},
        {"name":"罗洪先","title":"学者","description":"理学传播与坚守","spirit":"坚守传承","group":"理学脉络"},
        # 近代湖湘
        {"name":"曾国藩","title":"政治家","description":"湖湘文化代表，湘军领袖","spirit":"经世致用","group":"湘军将帅"},
        {"name":"左宗棠","title":"军政家","description":"湖湘代表，收复新疆","spirit":"实干担当","group":"湘军将帅"},
        {"name":"胡林翼","title":"政治家","description":"湘军核心将领","spirit":"治理才能","group":"湘军将帅"},
        {"name":"彭玉麟","title":"将领","description":"水师建设，忠诚坚毅","spirit":"忠诚报国","group":"湘军将帅"},
        {"name":"谭嗣同","title":"维新者","description":"思想激进，戊戌六君子","spirit":"牺牲精神","group":"维新革命"},
        {"name":"魏源","title":"思想家","description":"开眼看世界，师夷长技","spirit":"进取开放","group":"维新革命"},
        {"name":"黄兴","title":"革命家","description":"辛亥力量，革命实践","spirit":"革命奋斗","group":"维新革命"},
        {"name":"蔡锷","title":"将领","description":"护国运动领袖","spirit":"正义护国","group":"维新革命"},
        {"name":"宋教仁","title":"政治家","description":"宪政探索先驱","spirit":"理想宪政","group":"维新革命"},
        {"name":"陈天华","title":"思想者","description":"民族觉醒先驱","spirit":"激昂觉醒","group":"维新革命"},
        # 现代学人
        {"name":"毛泽东","title":"革命者","description":"青年在长沙求学活动","spirit":"改变历史","group":"现代学人"},
        {"name":"杨昌济","title":"教育家","description":"启迪青年，新民学会","spirit":"深远启迪","group":"现代学人"},
        {"name":"何叔衡","title":"教育者","description":"革命实践先驱","spirit":"坚定实践","group":"现代学人"},
        {"name":"李达","title":"哲学家","description":"理论建设，中共一大代表","spirit":"深刻理论","group":"现代学人"},
        {"name":"成仿吾","title":"校长","description":"高校发展建设者","spirit":"教育建设","group":"现代学人"},
        {"name":"周谷城","title":"历史学家","description":"学术研究严谨","spirit":"严谨学术","group":"现代学人"},
        {"name":"何长工","title":"教育推动者","description":"教育实践，务实办学","spirit":"务实教育","group":"现代学人"},
        {"name":"熊十力","title":"哲学家","description":"新儒学建构","spirit":"深邃哲学","group":"现代学人"},
        {"name":"冯友兰","title":"哲学家","description":"中国哲学史体系","spirit":"现代体系","group":"现代学人"},
        {"name":"钱基博","title":"国学家","description":"传统文化深厚","spirit":"深厚国学","group":"现代学人"},
        {"name":"金岳霖","title":"逻辑学家","description":"理性体系建构","spirit":"逻辑精密","group":"现代学人"},
        {"name":"梁漱溟","title":"思想家","description":"文化反思，乡村建设","spirit":"深刻反思","group":"现代学人"},
        {"name":"胡庶华","title":"校长","description":"西迁办学坚守者","spirit":"坚守文脉","group":"现代学人"},
    ]

    def __init__(self, knowledge_base=None, generator=None):
        """
        Args:
            knowledge_base: KnowledgeBase 实例（可选，不传则自动加载）
            generator: NarrativeGenerator 实例（可选，用于 LLM 精选）
        """
        self._kb = knowledge_base
        self._generator = generator
        self._kb_loaded = False

    def _ensure_kb(self):
        if self._kb is not None and self._kb_loaded:
            return
        # 核心人物列表作为主数据，兼容从知识库补充
        self._char_index: Dict[str, Dict] = {}
        # Only include characters with portrait files
        _has_portrait = {"朱熹","张栻","王夫之","周敦颐","胡宏","吕祖谦","陆九渊","王阳明",
                         "曾国藩","左宗棠","黄兴","蔡锷","宋教仁","陈天华",
                         "杨昌济","何叔衡","李达"}
        for c in self.CORE_CHARACTERS:
            if c["name"] not in _has_portrait:
                continue
            self._char_index[c["name"]] = {
                "type": "character",
                "name": c["name"],
                "years": "",
                "title": c["title"],
                "description": c["description"],
                "spirit": c["spirit"],
                "related_entities": [],
            }
        self._kb_loaded = True

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def recommend(
        self,
        color: str,
        objects: List[str],
        selected_characters: Optional[List[str]] = None,
        use_llm: bool = True,
        top_k: int = 3,
    ) -> List[RecommendResult]:
        """
        根据颜色和物象推荐人物

        Args:
            color: 第一幕选择的颜色名称
            objects: 第二幕确认的物象名称列表
            selected_characters: 已选人物列表（用于上下文加权）
            use_llm: 是否用 LLM 精选（需要 dashscope API）
            top_k: 返回数量

        Returns:
            推荐人物列表，按得分降序
        """
        self._ensure_kb()

        # 1. 粗筛: 所有人物打分
        scored = self._score_all_characters(color, objects, selected_characters or [])

        # 2. 精选
        if use_llm and self._generator is not None:
            try:
                return self._llm_rerank(scored[:15], color, objects, top_k)
            except Exception:
                pass  # 降级到启发式

        # 3. 降级: 直接返回启发式 top-k
        # 当所有候选分数都很低（无参考表命中）时，从 top-15 中随机选取避免总是同一人
        if scored and scored[0][1] < 0.15:
            import random
            pool = scored[:min(15, len(scored))]
            # 按分数加权随机选 top_k 个不重复的
            weights = [s[1] for s in pool]
            total_w = sum(weights)
            chosen = []
            remaining = list(pool)
            for _ in range(min(top_k, len(remaining))):
                if not remaining:
                    break
                w = [s[1] for s in remaining]
                tw = sum(w)
                if tw <= 0:
                    pick = random.choice(remaining)
                else:
                    # 加权随机
                    r = random.random() * tw
                    cumulative = 0
                    pick = remaining[-1]
                    for item in remaining:
                        cumulative += item[1]
                        if r <= cumulative:
                            pick = item
                            break
                chosen.append(pick)
                remaining.remove(pick)
            return self._build_results(chosen)

        return self._build_results(scored[:top_k])

    # ------------------------------------------------------------------
    # 粗筛打分
    # ------------------------------------------------------------------

    def _score_all_characters(
        self,
        color: str,
        objects: List[str],
        selected: List[str],
    ) -> List[Tuple[str, float, str]]:
        """对所有人物打分，返回 [(name, score, reason), ...] 降序"""
        selected_set = set(selected)
        scored = []
        for name, data in self._char_index.items():
            if name in selected_set:
                continue  # 排除已选人物
            score, reason = self._score_one(name, data, color, objects, selected)
            if score > 0:
                scored.append((name, score, reason))
        scored.sort(key=lambda x: -x[1])
        return scored

    def _score_one(
        self,
        name: str,
        data: Dict,
        color: str,
        objects: List[str],
        selected: List[str],
    ) -> Tuple[float, str]:
        """对单个人物打分"""
        reasons = []
        total = 0.0

        spirit = data.get("spirit", "")
        description = data.get("description", "")
        title = data.get("title", "")

        # 清洗 spirit 字段：去除末尾可能混入的下一实体名
        spirit_clean = self._clean_spirit(spirit)

        # 搜索文本
        text = f"{title} {spirit_clean} {description}"

        # (1) 内置参考表命中 (0~0.50)
        ref_score = 0.0
        ref_table = self.REFERENCE_TABLE.get(color, {})
        for obj in objects:
            if obj in ref_table and name in ref_table[obj]:
                idx = ref_table[obj].index(name)
                ref_score = max(ref_score, 0.50 - idx * 0.12)
        total += ref_score
        if ref_score > 0.2:
            reasons.append("经典搭配推荐")

        # (2) 已选人物同组加权 (0~0.20)
        peer_score = 0.0
        if selected:
            for sel_name in selected:
                for group_members in CHARACTER_GROUPS.values():
                    if sel_name in group_members and name in group_members and name != sel_name:
                        peer_score = max(peer_score, 0.20)
                if sel_name in text and name != sel_name:
                    peer_score = max(peer_score, 0.10)
        total += peer_score
        if peer_score > 0.08:
            reasons.append("与已选人物同脉")

        # (3) 关键词文本匹配 (0~0.25)
        kw_score = 0.0
        keywords = COLOR_TO_SPIRIT_KEYWORDS.get(color, [])
        for kw in keywords:
            if kw in text:
                kw_score += 0.04
        kw_score = min(kw_score, 0.15)

        for obj in objects:
            if obj in text:
                kw_score += 0.05
            elif len(obj) >= 2 and (obj[:2] in text or obj[-2:] in text):
                kw_score += 0.02
        kw_score = min(kw_score, 0.25)
        total += kw_score
        if kw_score > 0.06:
            reasons.append("关键词匹配")

        # (4) 基础分 (0~0.05)
        base = 0.02
        if len(description) > 10:
            base += 0.02
        if title and title not in ("", " "):
            base += 0.01
        total += base

        # 归一化
        total = min(total, 1.0)
        if not reasons:
            reasons.append("综合关联")

        return total, "；".join(reasons)

    @staticmethod
    def _clean_spirit(spirit: str) -> str:
        """清洗 spirit 字段，移除混入的下一实体名"""
        # 规则：如果 spirit 包含" 张栻"" 李达"等明显的下一实体名，截断
        # 常见污染模式：spirit = "思想传播 张栻" 其中 张栻 是下一个实体名
        # 快速判断：空格后如果是2-3个汉字的人名，去掉
        import re
        if not spirit:
            return spirit
        # 匹配 "xxx 2-3字人名" 模式，去掉末尾人名
        cleaned = re.sub(r'\s+\S{2,3}$', '', spirit)
        return cleaned if cleaned else spirit

    # ------------------------------------------------------------------
    # LLM 精选
    # ------------------------------------------------------------------

    def _llm_rerank(
        self,
        candidates: List[Tuple[str, float, str]],
        color: str,
        objects: List[str],
        top_k: int,
    ) -> List[RecommendResult]:
        """用 LLM 从候选人物中精选 top-k"""

        # 构建候选列表文本
        candidate_lines = []
        for i, (name, score, reason) in enumerate(candidates):
            data = self._char_index.get(name, {})
            title = data.get("title", "")
            desc = data.get("description", "")
            candidate_lines.append(f"{i+1}. {name}（{title}，{desc}）")

        prompt = f"""你是一个湖湘文化策展人。用户在第一幕选择了"{color}"，第二幕画出了{"、".join(objects)}。

请从以下候选人中，选出最合适的 {top_k} 位人物，让他们的声音回应这个场景。考虑：
1. 人物的时代背景和思想路线是否与颜色+物象的氛围契合
2. 人物之间是否有对话可能（如果已选多个）

候选人：
{chr(10).join(candidate_lines)}

请返回 JSON 格式，不要包含其他文字：
{{"recommendations": [
  {{"name": "人物名", "reason": "一句话推荐理由（20字内）"}},
  ...
]}}"""

        try:
            from rag.generator import create_generator
            gen = self._generator or create_generator()
            # 直接调底层 API
            if hasattr(gen, '_call_model'):
                resp = gen._call_model("qwen-turbo", prompt, max_tokens=300)
            elif hasattr(gen, '_ali_gen') and hasattr(gen._ali_gen, '_call_model'):
                resp = gen._ali_gen._call_model("qwen-turbo", prompt, max_tokens=300)
            else:
                raise RuntimeError("no LLM available")

            # 解析 JSON
            import json
            resp = resp.strip()
            if resp.startswith("```"):
                resp = resp.split("\n", 1)[1]
                resp = resp.rsplit("```", 1)[0]
            data = json.loads(resp)
            recs = data.get("recommendations", [])

            results = []
            for i, rec in enumerate(recs):
                name = rec.get("name", "")
                # 找原始分数
                orig_score = next((s for n, s, r in candidates if n == name), 0.5)
                score = orig_score * 0.7 + (1.0 - i * 0.15) * 0.3
                data = self._char_index.get(name, {})
                results.append(RecommendResult(
                    name=name,
                    title=data.get("title", ""),
                    score=round(min(score, 1.0), 3),
                    reason=rec.get("reason", ""),
                ))
            return results[:top_k]

        except Exception:
            raise  # 让调用方降级

    # ------------------------------------------------------------------
    # 结果构建
    # ------------------------------------------------------------------

    def _build_results(self, scored: List[Tuple[str, float, str]]) -> List[RecommendResult]:
        """从 (name, score, reason) 列表构建结果"""
        results = []
        for name, score, reason in scored:
            data = self._char_index.get(name, {})
            desc = data.get("description", "")
            spirit = data.get("spirit", "")
            title = data.get("title", "")
            results.append(RecommendResult(
                name=name,
                title=title,
                score=round(score, 3),
                reason=reason,
                monologue=[
                    f"后来者，我是{name}。",
                    f"我曾在岳麓山下，{desc}。",
                    f"你的选择，让我想起了当年的自己。"
                ],
                spiritLine=f"{spirit}——{name}，{title}",
            ))
        return results

    # ------------------------------------------------------------------
    # 辅助: 人物分组查询
    # ------------------------------------------------------------------

    def get_group(self, name: str) -> Optional[str]:
        """查询人物所属分组"""
        for group_name, members in CHARACTER_GROUPS.items():
            if name in members:
                return group_name
        return None

    def get_group_members(self, group_name: str) -> List[str]:
        """查询分组内所有成员"""
        return CHARACTER_GROUPS.get(group_name, [])


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_character_recommender(
    knowledge_path: str = "rag/knowledge",
    api_key: str = "",
) -> CharacterRecommender:
    """工厂函数：创建人物推荐器"""
    from rag.retriever import KnowledgeBase
    kb = KnowledgeBase(knowledge_path)
    kb.load()

    generator = None
    if api_key:
        try:
            from rag.generator import create_config, create_generator
            config = create_config(api_key=api_key)
            generator = create_generator(config)
        except Exception:
            pass

    return CharacterRecommender(knowledge_base=kb, generator=generator)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8')

    recommender = CharacterRecommender()
    recommender._ensure_kb()
    print(f"人物总数: {len(recommender._char_index)}")

    test_cases = [
        ("书院红", ["讲堂"], []),
        ("湘江蓝", ["桥梁"], []),
        ("岳麓绿", ["古树"], []),
        ("西迁黄", ["道路"], []),
        ("墨色", ["书卷"], []),
    ]

    for color, objects, selected in test_cases:
        print(f"\n{'='*50}")
        print(f"颜色={color}  物象={objects}")
        results = recommender.recommend(color, objects, selected, use_llm=False)
        for i, r in enumerate(results):
            print(f"  {i+1}. {r.name}（{r.title}） score={r.score:.3f}  {r.reason}")

    # 测试已选人物上下文加权
    print(f"\n{'='*50}")
    print("颜色=书院红  物象=['讲堂']  已选=['张栻']")
    results = recommender.recommend("书院红", ["讲堂"], ["张栻"], use_llm=False)
    for i, r in enumerate(results):
        print(f"  {i+1}. {r.name}（{r.title}） score={r.score:.3f}  {r.reason}")
