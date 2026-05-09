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


# ---------------------------------------------------------------------------
# 颜色 → 人物 spirit 关联词表
# 当知识库实体的 related_entities / spirit 字段不足时，作为补充检索词
# ---------------------------------------------------------------------------
COLOR_TO_SPIRIT_KEYWORDS: Dict[str, List[str]] = {
    "岳麓绿": ["传承", "生长", "根脉", "学术", "教育", "书院", "理学", "文化", "传统", "自然"],
    "书院红": ["责任", "热血", "担当", "讲学", "思想", "师承", "坚守", "精神", "历史", "革命"],
    "西迁黄": ["坚韧", "坚守", "办学", "文脉", "西迁", "奋斗", "道路", "实践", "建设", "教育"],
    "湘江蓝": ["包容", "追寻", "流水", "远方", "探索", "真理", "思考", "哲学", "开放", "学术"],
    "校徽金": ["荣耀", "理想", "星光", "希望", "成就", "引领", "创新", "发展", "荣誉", "未来"],
    "墨色": ["求索", "真理", "书卷", "思考", "学问", "研究", "严谨", "理论", "学术", "知识"],
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
    "校园角色": ["学子", "教师", "研究者", "新生", "毕业生", "辅导员", "志愿者",
                 "讲解员", "图书管理员", "古籍修复师", "食堂师傅", "保安", "留学生"],
    "抽象意象": ["理学之魂", "书院守望者", "湘江行者", "知识火种者", "文化摆渡人",
                 "思想叩问者", "未来开拓者", "时光记录人", "历史回响者"],
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
        "书院红": {
            "讲堂": ["张栻", "朱熹", "周敦颐"],
            "红墙": ["张栻", "王夫之", "朱熹"],
            "钟楼": ["朱熹", "张栻"],
            "碑刻": ["王夫之", "曾国藩"],
        },
        "湘江蓝": {
            "桥梁": ["曾国藩", "左宗棠", "胡林翼"],
            "湖面": ["王夫之", "周敦颐"],
            "流水": ["魏源", "谭嗣同"],
            "湘江": ["曾国藩", "左宗棠", "毛泽东"],
        },
        "岳麓绿": {
            "古树": ["王夫之", "胡安国", "胡宏"],
            "竹林": ["周敦颐", "程颢", "程颐"],
            "林荫道": ["朱熹", "张栻"],
            "草地": ["学子", "新生"],
        },
        "西迁黄": {
            "道路": ["胡庶华", "何长工", "成仿吾"],
            "石阶": ["何叔衡", "杨昌济"],
            "行李箱": ["留学生", "交换生"],
            "背包": ["学子", "新生"],
        },
        "墨色": {
            "书卷": ["李达", "杨昌济", "魏源"],
            "书架": ["图书管理员", "古籍修复师"],
            "黑板": ["教师", "哲学讲师"],
            "笔记本": ["学子", "博士生"],
            "论文": ["青年学者", "李达"],
        },
        "校徽金": {
            "图书馆": ["钱基博", "冯友兰"],
            "奖杯": ["竞赛指导老师", "运动员"],
            "荣誉墙": ["知名教授", "校庆组织者"],
            "学位帽": ["毕业生", "创业者"],
        },
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
        # 校园角色
        {"name":"学子","title":"求知者","description":"学习成长，专注求索","spirit":"求知","group":"校园角色"},
        {"name":"教师","title":"引导者","description":"传道授业解惑","spirit":"教育","group":"校园角色"},
        {"name":"研究者","title":"探索者","description":"科研专注","spirit":"探索","group":"校园角色"},
        {"name":"新生","title":"初学者","description":"初入校园，好奇憧憬","spirit":"好奇","group":"校园角色"},
        {"name":"毕业生","title":"传承者","description":"离校发展，感恩母校","spirit":"感恩","group":"校园角色"},
        {"name":"图书管理员","title":"守护者","description":"管理典籍","spirit":"细致","group":"校园角色"},
        {"name":"古籍修复师","title":"技艺者","description":"修复文献","spirit":"专注","group":"校园角色"},
        {"name":"志愿者","title":"服务者","description":"校园服务","spirit":"热心","group":"校园角色"},
        {"name":"讲解员","title":"传播者","description":"讲述历史","spirit":"清晰","group":"校园角色"},
        {"name":"辅导员","title":"指导者","description":"学生成长引导","spirit":"细心","group":"校园角色"},
        {"name":"留学生","title":"交流者","description":"跨文化学习","spirit":"开放","group":"校园角色"},
        {"name":"食堂师傅","title":"烹饪者","description":"温暖日常","spirit":"温暖","group":"校园角色"},
        # 抽象意象
        {"name":"理学之魂","title":"象征者","description":"思想凝聚","spirit":"抽象凝聚","group":"抽象意象"},
        {"name":"书院守望者","title":"守护者","description":"文脉延续","spirit":"坚定守望","group":"抽象意象"},
        {"name":"湘江行者","title":"见证者","description":"历史流动","spirit":"沉静见证","group":"抽象意象"},
        {"name":"知识火种者","title":"传播者","description":"点燃思想","spirit":"延续传播","group":"抽象意象"},
        {"name":"文化摆渡人","title":"连接者","description":"古今贯通","spirit":"传承连接","group":"抽象意象"},
        {"name":"思想叩问者","title":"提问者","description":"推动进步","spirit":"锐利叩问","group":"抽象意象"},
        {"name":"未来开拓者","title":"创新者","description":"面向未来","spirit":"希望开拓","group":"抽象意象"},
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
        for c in self.CORE_CHARACTERS:
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
            results.append(RecommendResult(
                name=name,
                title=data.get("title", ""),
                score=round(score, 3),
                reason=reason,
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
