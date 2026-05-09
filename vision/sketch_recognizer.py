"""
草图识别模块
指尖轨迹 → 栅格化 → CNN(QuickDraw MobileNet) → 物象映射 → 颜色加权排序

模型来源: Google Quick, Draw! 预训练 MobileNet
  下载: https://github.com/googlecreativelab/quickdraw-dataset
  推荐使用社区转换的 ONNX 权重，放在 vision/models/quickdraw_mobilenet.onnx
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# QuickDraw 345 类 → 88 物象 映射表
# ---------------------------------------------------------------------------
# 每个 QuickDraw 类别映射到 1~N 个物象知识库条目
QUICKDRAW_TO_OBJECT: Dict[str, List[str]] = {
    # 自然景观
    "river":        ["湘江", "流水", "湖面"],
    "mountain":     ["岳麓山", "山峰", "山门"],
    "cloud":        ["云层", "雾气", "天空", "云平台", "无线网络"],
    "rain":         ["雨滴", "风声"],
    "moon":         ["夜空", "星空", "湖面"],
    "star":         ["星光", "灯光", "灯塔"],
    "sun":          ["阳光", "灯光", "草地"],
    "ocean":        ["湘江", "流水", "湖面"],
    "pond":         ["湖面", "草地"],
    "waterfall":    ["流水", "湘江"],

    # 建筑/结构
    "house":        ["岳麓书院", "书院", "教学楼", "宿舍", "图书馆", "会议室", "书店", "打印店", "公司", "实习单位"],
    "castle":       ["红墙", "展览馆", "纪念碑", "爱晚亭"],
    "church":       ["钟楼", "讲堂", "爱晚亭"],
    "tower":        ["钟楼", "灯塔", "纪念碑"],
    "door":         ["校门", "山门", "入口"],
    "window":       ["窗格", "教室"],
    "fence":        ["栏杆", "长廊"],
    "bridge":       ["桥梁", "石桥"],
    "stairs":       ["石阶", "楼梯", "看台"],
    "road":         ["道路", "跑道", "长廊", "林荫道", "城市街道"],
    "streetlight":  ["灯光", "路灯"],
    "lighthouse":   ["灯塔", "纪念碑", "钟楼"],
    "square":       [],  # 见下方符号段统一定义

    # 植物
    "tree":         ["古树", "竹林", "林荫道"],
    "flower":       ["草地", "花园"],
    "leaf":         ["竹林", "古树", "林荫道"],
    "bush":         ["草地", "竹林"],
    "grass":        ["草地", "操场"],
    "cactus":       ["古树", "实验台"],
    "palm_tree":    ["古树", "林荫道"],

    # 室内/学习
    "book":         ["书卷", "书架", "笔记本", "论文", "期刊"],
    "bookshelf":    ["书架", "图书馆"],
    "pencil":       ["粉笔", "钢笔", "笔记本"],
    "pen":          ["粉笔", "钢笔"],
    "paper":        ["试卷", "论文", "笔记本"],
    "newspaper":    ["报刊栏", "公告栏", "期刊"],
    "blackboard":   ["黑板", "电子屏"],
    "clock":        ["钟楼"],
    "calendar":     ["时间轴", "公告栏"],
    "envelope":     ["信件", "毕业证", "证书"],

    # 物品
    "computer":     ["电脑", "数据屏", "服务器", "代码", "算法", "模型", "实验室"],
    "cell phone":   ["手机", "电子屏"],
    "headphones":   ["耳机"],
    "television":   ["电子屏", "投影仪"],
    "microphone":   ["讲座", "论坛", "讲坛", "讲席"],
    "camera":       ["镜头", "记录"],
    "binoculars":   ["显微镜", "观察", "望远镜"],
    "key":          ["棋", "入口"],
    "hammer":       ["实验工具"],
    "screwdriver":  ["实验工具"],

    # 器具
    "backpack":     ["背包", "行李箱"],
    "suitcase":     ["行李箱", "背包"],
    "cup":          ["水杯", "咖啡厅"],
    "coffee_cup":   ["咖啡厅", "水杯"],
    "wine_bottle":  ["水杯"],
    "knife":        ["实验工具"],
    "fork":         ["食堂"],
    "spoon":        ["食堂"],
    "plate":        ["食堂"],
    "basket":       ["书架", "背包"],

    # 交通
    "bicycle":      ["共享单车", "校道"],
    "car":          ["公交站", "校道", "道路"],
    "bus":          ["公交站", "道路"],
    "train":        ["火车站", "地铁站", "道路"],
    "airplane":     ["机场"],
    "boat":         ["湘江", "湖面"],
    "sailboat":     ["湘江", "湖面"],

    # 人物/身体（映射到抽象或校园角色）
    "face":         ["镜子", "观察"],
    "eye":          ["观察", "镜子"],
    "hand":         ["手势", "连接"],
    "foot":         ["脚印", "跑道"],

    # 运动
    "baseball":     ["操场", "运动场"],
    "basketball":   ["篮球场"],
    "tennis_racquet": ["运动场", "操场"],
    "soccer_ball":  ["操场", "运动场"],

    # 符号/标志
    "flag":         ["旗杆", "旗帜"],
    "trophy":       ["奖杯", "荣誉墙", "竞赛"],
    "medal":        ["荣誉墙", "证书"],
    "light_bulb":   ["灯光", "创意"],
    "magnifying_glass": ["显微镜", "探索"],
    "compass":      ["指南针", "探索"],
    "map":          ["地图", "导航"],
    "hat":          ["学位帽"],
    "baseball_cap": ["学位帽"],

    # 乐器
    "guitar":       ["乐器", "舞台"],
    "piano":        ["乐器", "讲堂"],
    "violin":       ["乐器", "舞台"],
    "trumpet":      ["号角", "钟楼"],

    # 其他
    "umbrella":     ["雨滴", "屋檐"],
    "wheel":        ["齿轮", "共享单车"],
    "zigzag":       ["闪电", "石阶", "电"],
    "triangle":     ["山门", "屋顶", "旗帜"],
    "circle":       ["湖面", "钟楼", "镜子", "奖杯"],
    "square":       ["广场", "黑板", "电子屏", "书架", "校训墙", "会议室", "影子"],
    "line":         ["道路", "跑道", "湘江", "长廊"],
    "diamond":      ["碑刻", "证书"],
    "hexagon":      ["实验台", "蜂巢"],
    "octagon":      ["亭子", "钟楼"],
}

# 验证：所有 88 物象至少被一个 QuickDraw 类别覆盖
_ALL_OBJECTS = {
    "岳麓书院", "湘江", "爱晚亭", "讲堂", "书卷", "古树", "碑刻", "校门",
    "图书馆", "操场", "实验室", "桥梁", "石阶", "红墙", "讲席", "钟楼",
    "林荫道", "教学楼", "宿舍", "食堂", "会议室", "展览馆", "纪念碑",
    "草地", "广场", "山门", "湖面", "竹林", "石桥", "长廊", "屋檐",
    "窗格", "雨滴", "风声", "灯光", "黑板", "粉笔", "电脑", "投影仪",
    "试卷", "笔记本", "书架", "报刊栏", "实验台", "显微镜", "数据屏",
    "运动场", "跑道", "篮球场", "看台", "旗杆", "校训墙", "荣誉墙",
    "毕业证", "学位帽", "背包", "水杯", "共享单车", "公交站", "咖啡厅",
    "书店", "打印店", "公告栏", "电子屏", "无线网络", "服务器", "云平台",
    "代码", "算法", "模型", "论文", "期刊", "讲座", "论坛", "竞赛",
    "奖杯", "证书", "实习单位", "公司", "城市街道", "地铁站", "机场",
    "火车站", "行李箱", "手机", "耳机", "镜子", "影子",
}


@dataclass
class SketchResult:
    """单次识别结果"""
    entity_name: str        # 物象中文名
    score: float            # 综合置信度 (0~1)
    qd_category: str        # 来源 QuickDraw 类别
    raw_confidence: float   # CNN 原始置信度


@dataclass
class RecognizerConfig:
    """识别器配置"""
    raster_size: int = 28              # 栅格化尺寸
    stroke_width: float = 2.0          # 笔画宽度（像素）
    min_trajectory_points: int = 5     # 最少轨迹点数
    normalize_padding: float = 0.1     # 归一化边距比例
    top_k: int = 3                     # 返回候选数
    model_path: str = ""               # ONNX 模型路径（空则用启发式降级）


class SketchRasterizer:
    """
    指尖轨迹 → 28×28 灰度图

    步骤：
    1. 归一化：平移到原点，缩放到 raster_size × raster_size（保留 10% 边距）
    2. 栅格化：在灰度图上逐段绘制抗锯齿线条
    """

    def __init__(self, size: int = 28, stroke_width: float = 2.0, padding: float = 0.1):
        self.size = size
        self.stroke_width = stroke_width
        self.padding = padding

    def normalize(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        归一化轨迹点到 [0, size] 范围

        Returns:
            np.ndarray of shape (N, 2), float
        """
        pts = np.array(points, dtype=np.float32)
        if len(pts) < 2:
            return pts

        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        span = max_xy - min_xy

        # 防止零除（单个点）
        if span[0] < 1e-6:
            span[0] = 1.0
        if span[1] < 1e-6:
            span[1] = 1.0

        # 等比缩放（保持宽高比，取较长的边）
        scale = (self.size * (1 - 2 * self.padding)) / max(span[0], span[1])
        center = (min_xy + max_xy) / 2.0
        canvas_center = self.size / 2.0

        normalized = (pts - center) * scale + canvas_center
        return np.clip(normalized, 0, self.size - 1)

    def rasterize(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        将归一化后的轨迹渲染为灰度图

        Args:
            points: 归一化后的轨迹点 [(x, y), ...]

        Returns:
            np.ndarray shape (size, size), dtype float32, range [0, 1]
        """
        canvas = np.zeros((self.size, self.size), dtype=np.float32)
        pts = np.array(points, dtype=np.float32)

        if len(pts) < 2:
            if len(pts) == 1:
                x, y = int(pts[0][0]), int(pts[0][1])
                if 0 <= x < self.size and 0 <= y < self.size:
                    canvas[y, x] = 1.0
            return canvas

        # 逐段绘制
        half_w = self.stroke_width / 2.0
        for i in range(len(pts) - 1):
            self._draw_line_aa(canvas, pts[i], pts[i + 1], half_w)

        return np.clip(canvas, 0.0, 1.0)

    def _draw_line_aa(self, canvas: np.ndarray, p0: np.ndarray, p1: np.ndarray, half_w: float):
        """抗锯齿线段绘制"""
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)

        if length < 1e-6:
            self._paint_point(canvas, int(x0), int(y0), half_w)
            return

        steps = max(1, int(length * 2))  # 2× 过采样
        for t in np.linspace(0, 1, steps):
            x = x0 + t * dx
            y = y0 + t * dy
            self._paint_point(canvas, x, y, half_w)

    def _paint_point(self, canvas: np.ndarray, cx: float, cy: float, half_w: float):
        """在画布上绘制一个抗锯齿圆点"""
        x_min = max(0, int(np.floor(cx - half_w - 1)))
        x_max = min(self.size - 1, int(np.ceil(cx + half_w + 1)))
        y_min = max(0, int(np.floor(cy - half_w - 1)))
        y_max = min(self.size - 1, int(np.ceil(cy + half_w + 1)))

        for py in range(y_min, y_max + 1):
            for px in range(x_min, x_max + 1):
                dist = math.hypot(px - cx, py - cy)
                if dist <= half_w + 0.5:
                    # 抗锯齿：边缘渐变
                    alpha = 1.0 - max(0.0, (dist - half_w + 0.5))
                    alpha = np.clip(alpha, 0.0, 1.0)
                    canvas[py, px] = max(canvas[py, px], alpha)

    def process(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """完整流程：归一化 + 栅格化 → 28×28 灰度图"""
        normalized = self.normalize(points)
        return self.rasterize([(p[0], p[1]) for p in normalized])


class QuickDrawMapper:
    """QuickDraw 345 类 → 88 物象 映射"""

    def __init__(self):
        # 构建反向索引：物象名 → [(qd_category, weight), ...]
        self.object_to_qd: Dict[str, List[Tuple[str, float]]] = {}
        for qd_cat, objects in QUICKDRAW_TO_OBJECT.items():
            n = len(objects)
            for i, obj in enumerate(objects):
                # 列表越前权重越高
                weight = 1.0 - (i / (n + 1)) * 0.5
                self.object_to_qd.setdefault(obj, []).append((qd_cat, weight))

    def map_predictions(self, qd_probs: Dict[str, float]) -> List[SketchResult]:
        """
        将 QuickDraw 分类结果映射到物象

        Args:
            qd_probs: {qd_category: confidence, ...}

        Returns:
            按置信度降序排列的物象列表
        """
        obj_scores: Dict[str, float] = {}

        for qd_cat, conf in qd_probs.items():
            objects = QUICKDRAW_TO_OBJECT.get(qd_cat, [])
            for rank, obj_name in enumerate(objects):
                # 列表位置越靠前，权重越高；置信度是 CNN 输出的概率
                position_weight = 1.0 - rank * 0.15
                score = conf * position_weight
                if obj_name not in obj_scores or score > obj_scores[obj_name]:
                    obj_scores[obj_name] = score

        results = []
        for obj_name, score in sorted(obj_scores.items(), key=lambda x: -x[1]):
            # 找到贡献最大的 qd 类别
            best_qd = "unknown"
            best_qd_conf = 0.0
            for qd_cat, objects in QUICKDRAW_TO_OBJECT.items():
                if obj_name in objects and qd_probs.get(qd_cat, 0) > best_qd_conf:
                    best_qd = qd_cat
                    best_qd_conf = qd_probs[qd_cat]

            results.append(SketchResult(
                entity_name=obj_name,
                score=score,
                qd_category=best_qd,
                raw_confidence=best_qd_conf,
            ))

        return results


class ColorWeighter:
    """颜色上下文加权器"""

    # 颜色 → 加权物象组
    COLOR_WEIGHTS: Dict[str, Dict[str, List[str]]] = {
        "岳麓绿": {
            "boost": ["古树", "竹林", "草地", "林荫道", "山门", "岳麓山", "屋檐", "窗格"],
            "penalize": ["服务器", "电脑", "数据屏", "代码", "算法", "机场", "火车站"],
        },
        "书院红": {
            "boost": ["红墙", "讲堂", "钟楼", "碑刻", "校训墙", "荣誉墙", "展览馆"],
            "penalize": ["流水", "湖面", "雨滴", "竹林", "草地"],
        },
        "西迁黄": {
            "boost": ["道路", "石阶", "行李箱", "背包", "火车站", "机场", "跑道", "公交站"],
            "penalize": ["红墙", "钟楼", "湖面", "湘江"],
        },
        "湘江蓝": {
            "boost": ["桥梁", "石桥", "湖面", "流水", "湘江", "长廊", "雨滴"],
            "penalize": ["红墙", "沙漠", "火焰"],
        },
        "校徽金": {
            "boost": ["图书馆", "奖杯", "荣誉墙", "学位帽", "证书", "毕业证"],
            "penalize": ["垃圾桶", "枯树"],
        },
        "墨色": {
            "boost": ["书卷", "书架", "笔记本", "论文", "黑板", "粉笔", "碑刻"],
            "penalize": ["操场", "运动场", "篮球场", "阳光", "灯光"],
        },
    }

    BOOST_FACTOR = 1.4
    PENALIZE_FACTOR = 0.6

    def apply(self, results: List[SketchResult], color_name: Optional[str]) -> List[SketchResult]:
        """对识别结果应用颜色加权"""
        if not color_name or color_name not in self.COLOR_WEIGHTS:
            return results

        rules = self.COLOR_WEIGHTS[color_name]
        boost_set = set(rules.get("boost", []))
        penalize_set = set(rules.get("penalize", []))

        weighted = []
        for r in results:
            new_score = r.score
            if r.entity_name in boost_set:
                new_score = min(1.0, r.score * self.BOOST_FACTOR)
            elif r.entity_name in penalize_set:
                new_score = r.score * self.PENALIZE_FACTOR
            weighted.append(SketchResult(
                entity_name=r.entity_name,
                score=new_score,
                qd_category=r.qd_category,
                raw_confidence=r.raw_confidence,
            ))

        weighted.sort(key=lambda x: -x.score)
        return weighted


class HeuristicPredictor:
    """
    启发式预测器（模型不可用时的降级方案）

    使用轨迹几何特征做粗分类，作为占位实现。
    实际部署时替换为 ONNX 模型推理。
    """

    # 特征 → QuickDraw 类别映射
    FEATURE_CATEGORIES = {
        "vertical_line":   ["tree", "tower", "pencil", "line"],
        "horizontal_line": ["river", "road", "bridge", "line"],
        "circle":          ["sun", "clock", "face", "circle", "pond"],
        "triangle":        ["mountain", "triangle", "house"],
        "rectangle":       ["house", "book", "square", "door"],
        "curve":           ["river", "cloud", "mountain", "snake"],
        "zigzag":          ["stairs", "zigzag", "lightning"],
        "complex":         ["house", "castle", "tree", "flower"],
    }

    def predict(self, points: List[Tuple[float, float]]) -> Dict[str, float]:
        """基于轨迹几何特征返回 QuickDraw 类别概率（模拟）"""
        if len(points) < 3:
            return {"circle": 0.5, "line": 0.3, "square": 0.2}

        pts = np.array(points, dtype=np.float32)

        # 特征提取
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        span = max_xy - min_xy
        w, h = max(span[0], 1e-6), max(span[1], 1e-6)
        aspect_ratio = w / h

        # 闭合性：起止点距离 / 总路径长度
        start_end_dist = np.linalg.norm(pts[-1] - pts[0])
        total_length = sum(np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts)-1))
        closedness = 1.0 - min(1.0, start_end_dist / max(total_length * 0.3, 1e-6))

        # 方向变化次数
        if len(pts) >= 3:
            dirs = np.diff(pts, axis=0)
            angles = np.arctan2(dirs[:, 1], dirs[:, 0])
            angle_diffs = np.abs(np.diff(angles))
            direction_changes = np.sum(angle_diffs > math.radians(30))
        else:
            direction_changes = 0

        # 到中心点的平均距离（判断是否为圆）
        center = (min_xy + max_xy) / 2.0
        distances = np.linalg.norm(pts - center, axis=1)
        dist_std = np.std(distances) / max(np.mean(distances), 1e-6)

        probs = {}

        # 闭合 + 低距离方差 → 圆形
        if closedness > 0.6 and dist_std < 0.3:
            probs["circle"] = 0.6
            probs["face"] = 0.2
            probs["sun"] = 0.15
        # 低闭合 + 宽>高 → 横线/河流/桥梁
        elif aspect_ratio > 1.8 and direction_changes < 5:
            probs["river"] = 0.4
            probs["line"] = 0.3
            probs["bridge"] = 0.2
            probs["road"] = 0.1
        # 低闭合 + 高>宽 → 竖线/树/塔
        elif aspect_ratio < 0.5 and direction_changes < 5:
            probs["tree"] = 0.35
            probs["tower"] = 0.25
            probs["line"] = 0.2
            probs["pencil"] = 0.15
        # 中等方向变化 + 非极端宽高比 → 房屋/建筑/书（放曲线分支前）
        elif 3 <= direction_changes <= 8 and 0.5 <= aspect_ratio <= 2.0:
            probs["house"] = 0.40
            probs["square"] = 0.25
            probs["book"] = 0.20
            probs["castle"] = 0.15
        # 多方向变化 + 低闭合 → 锯齿/楼梯/山
        elif direction_changes >= 5 and closedness < 0.3:
            probs["stairs"] = 0.35
            probs["zigzag"] = 0.25
            probs["mountain"] = 0.2
            probs["triangle"] = 0.15
        # 多方向变化 + 有闭合 → 房屋/复杂
        elif direction_changes >= 5 and closedness > 0.3:
            probs["house"] = 0.35
            probs["castle"] = 0.2
            probs["book"] = 0.2
            probs["square"] = 0.15
        # 低方向变化 + 曲线 → 河流/云/山
        elif direction_changes >= 3:
            probs["river"] = 0.3
            probs["cloud"] = 0.25
            probs["mountain"] = 0.25
            probs["snake"] = 0.15
        # 默认
        else:
            probs["house"] = 0.25
            probs["tree"] = 0.2
            probs["book"] = 0.2
            probs["line"] = 0.15
            probs["circle"] = 0.1
            probs["river"] = 0.1

        # 归一化
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}


class SketchRecognizer:
    """
    草图识别器

    用法:
        recognizer = SketchRecognizer()
        recognizer.load_model("vision/models/quickdraw_mobilenet.onnx")  # 可选

        # 传入 MediaPipe 追踪的食指指尖轨迹
        results = recognizer.recognize(trajectory_points, color="岳麓绿")
        # results: [SketchResult, ...] 长度 = top_k
    """

    def __init__(self, config: Optional[RecognizerConfig] = None):
        self.config = config or RecognizerConfig()
        self.rasterizer = SketchRasterizer(
            size=self.config.raster_size,
            stroke_width=self.config.stroke_width,
            padding=self.config.normalize_padding,
        )
        self.mapper = QuickDrawMapper()
        self.weighter = ColorWeighter()
        self.heuristic = HeuristicPredictor()

        self._model = None          # ONNX InferenceSession
        self._model_loaded = False
        self._qd_class_names: List[str] = []  # QuickDraw 345 类名列表

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load_model(self, model_path: str) -> bool:
        """加载 ONNX 模型"""
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(model_path)
            self._model_loaded = True
            # 从模型元数据读类别名（或使用默认 345 类列表）
            self._qd_class_names = self._get_quickdraw_classes()
            return True
        except ImportError:
            print("[SketchRecognizer] onnxruntime 未安装，使用启发式降级")
            return False
        except Exception as e:
            print(f"[SketchRecognizer] 模型加载失败: {e}，使用启发式降级")
            return False

    def _get_quickdraw_classes(self) -> List[str]:
        """获取 QuickDraw 345 类名列表"""
        # 使用映射表中的类别作为最小集
        classes = sorted(QUICKDRAW_TO_OBJECT.keys())
        # 补充常见 QuickDraw 类别（共 345 个，这里列关键的部分）
        return classes

    # ------------------------------------------------------------------
    # 主要接口
    # ------------------------------------------------------------------

    def recognize(
        self,
        trajectory: List[Tuple[float, float]],
        color: Optional[str] = None,
    ) -> List[SketchResult]:
        """
        识别指尖轨迹对应的物象

        Args:
            trajectory: 食指指尖轨迹点序列 [(x, y), ...]，像素坐标
            color: 第一幕选择的颜色名称，用于上下文加权（可选）

        Returns:
            Top-K 候选物象列表，按置信度降序
        """
        if len(trajectory) < self.config.min_trajectory_points:
            return []

        # 1. 轨迹 → 28×28 灰度图
        raster = self.rasterizer.process(trajectory)

        # 2. 模型推理（或降级）
        if self._model_loaded:
            qd_probs = self._predict_onnx(raster)
        else:
            qd_probs = self.heuristic.predict(trajectory)

        # 3. QuickDraw → 物象映射
        results = self.mapper.map_predictions(qd_probs)

        # 4. 颜色上下文加权
        results = self.weighter.apply(results, color)

        return results[:self.config.top_k]

    def _predict_onnx(self, raster: np.ndarray) -> Dict[str, float]:
        """ONNX 模型推理"""
        # 添加 batch 和 channel 维度: (28,28) → (1,1,28,28)
        input_data = raster.reshape(1, 1, self.config.raster_size, self.config.raster_size).astype(np.float32)
        input_name = self._model.get_inputs()[0].name
        output = self._model.run(None, {input_name: input_data})[0][0]

        # softmax
        output = output - np.max(output)
        exp = np.exp(output)
        probs = exp / exp.sum()

        return {self._qd_class_names[i]: float(probs[i])
                for i in range(len(probs))
                if float(probs[i]) > 0.001}

    # ------------------------------------------------------------------
    # 绘制工具
    # ------------------------------------------------------------------

    def get_raster_preview(self, trajectory: List[Tuple[float, float]]) -> np.ndarray:
        """获取栅格化预览图（用于调试显示）"""
        raster = self.rasterizer.process(trajectory)
        return (raster * 255).astype(np.uint8)

    def recognize_from_fingertip_history(
        self,
        fingertip_history: List[Tuple[float, float, float]],
        color: Optional[str] = None,
    ) -> List[SketchResult]:
        """
        从带时间戳的指尖历史记录中识别

        Args:
            fingertip_history: [(x, y, timestamp_ms), ...]
            color: 颜色上下文

        Returns:
            Top-K 候选物象
        """
        trajectory = [(x, y) for x, y, _ in fingertip_history]
        return self.recognize(trajectory, color)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_sketch_recognizer(
    model_path: str = "",
    raster_size: int = 28,
    top_k: int = 3,
) -> SketchRecognizer:
    """工厂函数"""
    config = RecognizerConfig(
        raster_size=raster_size,
        top_k=top_k,
        model_path=model_path,
    )
    recognizer = SketchRecognizer(config)
    if model_path:
        recognizer.load_model(model_path)
    return recognizer


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # 模拟几种简笔画轨迹
    import math

    def make_circle(cx=200, cy=200, r=80, n=40):
        pts = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        return pts

    def make_line(x0=50, y0=200, x1=350, y1=200, n=20):
        return [(x0 + (x1-x0)*i/n, y0 + (y1-y0)*i/n) for i in range(n+1)]

    def make_house(cx=200, cy=200, s=80):
        pts = []
        pts.append((cx - s, cy + s))
        pts.append((cx - s, cy))
        pts.append((cx, cy - s))
        pts.append((cx + s, cy))
        pts.append((cx + s, cy + s))
        return pts

    def make_zigzag(x0=50, y0=200, step=40, n=8):
        pts = [(x0, y0)]
        for i in range(1, n+1):
            pts.append((x0 + i*step, y0 if i%2==0 else y0 - step))
        return pts

    recognizer = create_sketch_recognizer()

    test_cases = [
        ("圆形 → 湖面/钟楼", make_circle(), "湘江蓝"),
        ("横线 → 湘江/道路", make_line(), "湘江蓝"),
        ("房屋 → 书院/教学楼", make_house(), "书院红"),
        ("锯齿 → 石阶", make_zigzag(), "岳麓绿"),
    ]

    for name, trajectory, color in test_cases:
        results = recognizer.recognize(trajectory, color=color)
        print(f"\n{name} (颜色={color}):")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r.entity_name}  score={r.score:.3f}  "
                  f"(qd={r.qd_category}, raw={r.raw_confidence:.3f})")
