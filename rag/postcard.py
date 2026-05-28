"""
明信片/叙事卡生成模块
生成包含底图+叙事文字的完整纪念图片
"""

import os
import json
import base64
import numpy as np
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 图像处理
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[警告] Pillow未安装，明信片生成不可用")


@dataclass
class PostcardConfig:
    """明信片配置"""
    # 尺寸 (竖版3:4)
    width: int = 1080
    height: int = 1440

    # 颜色 (宣纸色)
    bg_color: Tuple[int, int, int] = (245, 240, 230)
    title_color: Tuple[int, int, int] = (50, 40, 30)
    text_color: Tuple[int, int, int] = (60, 50, 40)
    signature_color: Tuple[int, int, int] = (120, 100, 80)
    seal_color: Tuple[int, int, int] = (180, 50, 50)

    # 字体配置
    font_dir: str = "assets/fonts"
    system_font_dir: str = r"C:\Windows\Fonts"

    # 布局比例
    image_ratio: float = 0.55    # 底图占默认画面55%


class FontManager:
    """字体管理器"""

    def __init__(self, config: PostcardConfig):
        self.config = config
        self._title_font = None
        self._text_font = None
        self._signature_font = None
        self._seal_font = None

    def _search_font_paths(self, font_name: str) -> List[str]:
        """搜索字体文件可能的路径"""
        paths = [
            # 项目目录
            os.path.join(self.config.font_dir, font_name),
            os.path.join(self.config.font_dir, font_name + ".ttf"),
            os.path.join(self.config.font_dir, font_name + ".otf"),
            # Windows系统字体
            os.path.join(self.config.system_font_dir, font_name),
            os.path.join(self.config.system_font_dir, font_name + ".ttf"),
            os.path.join(self.config.system_font_dir, font_name + ".ttc"),
        ]
        return paths

    def _load_font(self, font_name: str, fallback_size: int = 36) -> ImageFont.FreeTypeFont:
        """加载字体文件"""
        for path in self._search_font_paths(font_name):
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, fallback_size)
                except Exception:
                    continue

        # 如果都找不到，返回默认字体
        print(f"[警告] 字体 {font_name} 未找到，使用默认字体")
        return ImageFont.load_default()

    def get_title_font(self, size: int = 60) -> ImageFont.FreeTypeFont:
        """获取标题字体"""
        if self._title_font is None:
            self._title_font = self._load_font("simhei.ttf", size)
        return self._title_font

    def get_text_font(self, size: int = 32) -> ImageFont.FreeTypeFont:
        """获取正文字体"""
        if self._text_font is None:
            self._text_font = self._load_font("simhei.ttf", size)
        return self._text_font

    def get_signature_font(self, size: int = 28) -> ImageFont.FreeTypeFont:
        """获取落款字体"""
        if self._signature_font is None:
            self._signature_font = self._load_font("simhei.ttf", size)
        return self._signature_font

    def get_seal_font(self, size: int = 24) -> ImageFont.FreeTypeFont:
        """获取印章字体"""
        if self._seal_font is None:
            self._seal_font = self._load_font("simhei.ttf", size)
        return self._seal_font


class PostcardGenerator:
    """
    明信片生成器
    生成包含底图+叙事文字的完整纪念图片
    """

    def __init__(self, config: PostcardConfig = None):
        self.config = config or PostcardConfig()
        self.font_manager = FontManager(self.config)

    def _load_base64_image(self, base64_str: str) -> Optional[Image.Image]:
        """加载Base64图像"""
        if not PIL_AVAILABLE:
            return None

        try:
            # 移除data URI前缀
            if "base64," in base64_str:
                base64_str = base64_str.split("base64,")[1]

            image_data = base64.b64decode(base64_str)
            return Image.open(BytesIO(image_data))
        except Exception as e:
            print(f"[错误] 图像加载失败: {e}")
            return None

    def _load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """从URL下载并加载图像"""
        if not PIL_AVAILABLE:
            return None

        try:
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"[错误] 图像下载失败: {e}")
            return None

    # 布局常量
    MARGIN_X = 60               # 正文左右边距
    CHARS_PER_LINE = 22          # 每行约字数（32px 字体）
    LINE_HEIGHT = 50             # 行高
    PARA_SPACING = 30            # 段间距
    FOOTER_HEIGHT = 140          # 落款 + 印章区高度
    PAINTING_GAP = 50            # 底图与正文间距

    def _wrap_para(self, draw: ImageDraw.Image, text: str,
                   font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """按像素宽度逐字换行，返回行列表"""
        lines = []
        current = ""
        for ch in text:
            test = current + ch
            if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
                current = test
            else:
                lines.append(current)
                current = ch
        if current:
            lines.append(current)
        return lines

    def _calc_text_height(self, draw: ImageDraw.Image, paragraphs: List[str],
                          font: ImageFont.FreeTypeFont, max_width: int) -> int:
        """计算叙事文字所需总高度"""
        total = 0
        for para in paragraphs:
            lines = self._wrap_para(draw, para, font, max_width)
            total += len(lines) * self.LINE_HEIGHT + self.PARA_SPACING
        return total

    # ── 分层合成新增方法 ─────────────────────────────────────

    def _rasterize_trajectory(self, trajectory: List[Tuple[float, float, float]],
                               target_size: Tuple[int, int]) -> Image.Image:
        """
        将用户轨迹转换为带笔画宽度的线稿图像

        Args:
            trajectory: [(x, y, ts_ms), ...] 归一化坐标 (0-1)
            target_size: 目标尺寸 (width, height)

        Returns:
            RGBA PIL Image，黑线白底
        """
        w, h = target_size
        # 创建 RGBA 画布（透明底）
        canvas = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        if len(trajectory) < 2:
            return canvas

        # 转换坐标并绘制线段
        stroke_width = 3
        prev_x, prev_y = None, None

        for x_norm, y_norm, _ in trajectory:
            x = int(x_norm * w)
            y = int(y_norm * h)
            if prev_x is not None:
                # 抗锯齿线段
                draw.line([(prev_x, prev_y), (x, y)],
                         fill=(0, 0, 0, 255), width=stroke_width)
            prev_x, prev_y = x, y

        return canvas

    def _find_empty_region(self, img: Image.Image,
                           text_size: Tuple[int, int],
                           margin: int = 40) -> Tuple[int, int]:
        """
        寻找图像中颜色最均匀（最空）的区域放置文字

        Args:
            img: PIL Image
            text_size: 文字区域尺寸 (width, height)
            margin: 边距

        Returns:
            (x, y) 左上角坐标
        """
        w, h = img.size
        tw, th = text_size

        # 缩小图像加速计算
        scale = 8
        small_w, small_h = w // scale, h // scale
        small_img = img.resize((small_w, small_h), Image.LANCZOS)
        small_gray = small_img.convert('L')

        tw_s, th_s = tw // scale, th // scale

        # 九宫格检测：计算每个格子的方差
        grid_scores = []
        grid_positions = []

        for gy in range(3):
            for gx in range(3):
                x_start = gx * small_w // 3 + margin // scale
                y_start = gy * small_h // 3 + margin // scale
                x_end = (gx + 1) * small_w // 3 - margin // scale - tw_s
                y_end = (gy + 1) * small_h // 3 - margin // scale - th_s

                if x_end <= x_start or y_end <= y_start:
                    continue

                # 计算这个格子的颜色方差
                region = small_gray.crop((x_start, y_start, x_end, y_end))
                var = region.var()

                grid_scores.append(var)
                grid_positions.append((gx, gy, x_start * scale, y_start * scale))

        # 选择方差最小（最空）的格子
        best_idx = grid_scores.index(min(grid_scores)) if grid_scores else 0
        _, _, best_x, best_y = grid_positions[best_idx] if grid_positions else (0, 0, margin, margin)

        # 如果最空区域方差仍然太高，放右下角
        if grid_scores and max(grid_scores) > 1000:
            best_x = w - tw - margin
            best_y = h - th - margin

        return (max(margin, min(best_x, w - tw - margin)),
                max(margin, min(best_y, h - th - margin)))

    def _get_paper_texture(self, style: str) -> Image.Image:
        """
        获取时代风格的纸张纹理

        Args:
            style: 风格类型
                - "classical": 宣纸纹理（理学脉络）
                - "modern": 米白纸张（现代学人）
                - "kraft": 牛皮纸（校园角色）
                - "vintage": 仿古纸（近代）

        Returns:
            PIL Image
        """
        w, h = self.config.width, self.config.height
        colors = {
            "classical": (245, 240, 230),   # 宣纸色
            "modern": (250, 248, 245),     # 米白
            "kraft": (210, 180, 140),      # 牛皮纸
            "vintage": (235, 225, 210),   # 仿旧纸
        }
        bg_color = colors.get(style, colors["classical"])
        paper = Image.new('RGB', (w, h), bg_color)

        # 添加噪声纹理模拟纸张质感
        np.random.seed(42)
        noise = np.random.randint(-5, 5, (h, w, 3), dtype=np.int16)
        paper_arr = np.array(paper, dtype=np.int16)
        paper_arr = np.clip(paper_arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(paper_arr)

    def _draw_period_border(self, canvas: Image.Image, style: str):
        """
        绘制时代风格边框

        Args:
            canvas: 画布
            style: 风格类型
                - "classical": 水墨边栏 + 简约竖线
                - "military": 印章感 + 朱红边线
                - "revolution": 木刻感 + 粗黑边框
                - "modern": 素描纸 + 铅笔线条
                - "campus": 牛皮纸 + 手账风
                - "abstract": 水墨飞白
        """
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size
        margin = 30

        if style == "classical":
            # 水墨边栏：深色边框 + 简约竖线
            draw.rectangle([(margin, margin), (w - margin, h - margin)],
                          outline=(60, 50, 40), width=3)
            # 顶部和底部装饰线
            draw.line([(margin + 20, margin + 15), (w - margin - 20, margin + 15)],
                     fill=(60, 50, 40), width=1)
            draw.line([(margin + 20, h - margin - 15), (w - margin - 20, h - margin - 15)],
                     fill=(60, 50, 40), width=1)

        elif style == "military":
            # 印章感：朱红双边线
            draw.rectangle([(margin, margin), (w - margin, h - margin)],
                          outline=(180, 50, 50), width=2)
            draw.rectangle([(margin + 8, margin + 8), (w - margin - 8, h - margin - 8)],
                          outline=(150, 40, 40), width=1)
            # 角落印章符号
            for cx, cy in [(margin + 20, margin + 20), (w - margin - 20, h - margin - 20)]:
                draw.ellipse([(cx - 8, cy - 8), (cx + 8, cy + 8)],
                           outline=(180, 50, 50), width=2)

        elif style == "revolution":
            # 木刻感：粗黑边框
            draw.rectangle([(margin, margin), (w - margin, h - margin)],
                          outline=(30, 20, 20), width=5)
            draw.rectangle([(margin + 10, margin + 10), (w - margin - 10, h - margin - 10)],
                          outline=(30, 20, 20), width=1)

        elif style == "modern":
            # 素描纸：浅灰边框
            draw.rectangle([(margin, margin), (w - margin, h - margin)],
                          outline=(180, 175, 165), width=2)

        elif style == "campus":
            # 手账风：浅色边框
            draw.rectangle([(margin, margin), (w - margin, h - margin)],
                          outline=(160, 140, 110), width=2)
            # 点状装饰
            for i in range(margin + 15, w - margin - 15, 20):
                draw.ellipse([(i - 2, margin + 12), (i + 2, margin + 16)],
                           fill=(160, 140, 110))

        else:  # abstract
            # 水墨飞白：渐隐边框
            draw.rectangle([(margin, margin), (w - margin, h - margin)],
                          outline=(80, 70, 60), width=2)

    def _draw_vertical_calligraphy(self, canvas: Image.Image, text: str,
                                    position: Tuple[int, int],
                                    font_size: int = 28,
                                    color: Tuple[int, int, int] = (60, 50, 40)):
        """
        绘制竖排书法文字（落款用）

        Args:
            canvas: 画布
            text: 文字内容
            position: (x, y) 左上角
            font_size: 字体大小
            color: 文字颜色
        """
        draw = ImageDraw.Draw(canvas)
        font = self.font_manager._load_font("simhei.ttf", font_size)

        x, y = position
        char_height = font_size + 8

        for char in text:
            draw.text((x, y), char, font=font, fill=color)
            y += char_height

    def create_layered_postcard(self,
                                 narrative_result: Dict,
                                 wan_image: Image.Image = None,
                                 color_wash_base64: str = None,
                                 sketch_trajectories: Dict[str, List] = None,
                                 character_group: str = "classical",
                                 signature_text: str = None,
                                 unique_id: str = None) -> Optional[Image.Image]:
        """
        生成分层明信片

        Layer 0: 纸质纹理底（宣纸/牛皮纸/水墨纸，由人物时代决定）
        Layer 1: 颜色晕染层（来自第一幕，半透明）
        Layer 2: AI生成主画面（Wan输出）
        Layer 3: 用户原始线稿（低透明度叠加，15%）
        Layer 4: (跳过，不使用)
        Layer 5: 人物光柱/剪影（预留）
        Layer 6: 个性化文字（叙事文本节选）
        Layer 7: 时间戳 + 唯一编号 + 落款印章
        Layer 8: 装饰边框（按人物时代风格）

        Args:
            narrative_result: 叙事内容 {"title": ..., "paragraphs": ..., "summary": ...}
            wan_image: Wan生成的AI主画面
            color_wash_base64: 颜色晕染底图（base64）
            sketch_trajectories: {物象名: [(x,y,ts_ms), ...]} 用户绘画轨迹
            character_group: 人物分组，影响纸张/边框/风格
            signature_text: 落款文字（默认"湖南大学·寻麓千年色"）
            unique_id: 唯一编号

        Returns:
            PIL Image
        """
        if not PIL_AVAILABLE:
            return None

        w, h = self.config.width, self.config.height
        paper_styles = {
            "classical": "classical",     # 理学脉络
            "military": "vintage",         # 湘军将帅
            "revolution": "vintage",       # 维新革命
            "modern": "modern",           # 现代学人
            "campus": "kraft",            # 校园角色
            "abstract": "classical",       # 抽象意象
        }
        border_styles = {
            "classical": "classical",
            "military": "military",
            "revolution": "revolution",
            "modern": "modern",
            "campus": "campus",
            "abstract": "abstract",
        }

        paper_style = paper_styles.get(character_group, "classical")
        border_style = border_styles.get(character_group, "classical")

        # Layer 0: 纸质纹理
        canvas = self._get_paper_texture(paper_style)

        # Layer 1: 颜色晕染（半透明叠加在纸张上）
        if color_wash_base64:
            color_wash = self._load_base64_image(color_wash_base64)
            if color_wash:
                # 缩放到画布尺寸，半透明叠加
                color_wash = color_wash.resize((w, h), Image.LANCZOS)
                # 创建半透明版本
                color_arr = np.array(color_wash)
                alpha_wash = Image.fromarray(color_arr).convert('RGBA')
                # 降低透明度 30%
                rgba = np.array(alpha_wash)
                rgba[:, :, 3] = (rgba[:, :, 3] * 0.3).astype(np.uint8)
                alpha_wash = Image.fromarray(rgba)
                canvas_rgba = canvas.convert('RGBA')
                canvas_rgba = Image.alpha_composite(canvas_rgba, alpha_wash)
                canvas = canvas_rgba.convert('RGB')

        # Layer 2: AI主画面（缩放居中）
        painting_area_top = int(h * 0.15)
        painting_area_bottom = int(h * 0.65)
        painting_area_h = painting_area_bottom - painting_area_top

        if wan_image:
            aspect = wan_image.width / wan_image.height
            target_w = int(painting_area_h * aspect)
            target_w = min(target_w, w - 60)
            target_h = int(target_w / aspect)
            wan_resized = wan_image.resize((target_w, target_h), Image.LANCZOS)
            x_offset = (w - target_w) // 2
            canvas.paste(wan_resized, (x_offset, painting_area_top))

        # Layer 3: 用户线稿叠加（15%透明度）
        if sketch_trajectories:
            for obj_name, trajectory in sketch_trajectories.items():
                if len(trajectory) < 5:
                    continue
                # 转换轨迹坐标到画布尺寸
                sketch_img = self._rasterize_trajectory(trajectory, (w, painting_area_h))

                # 调整位置：在画面下半部居中
                sketch_x = (w - sketch_img.width) // 2
                sketch_y = painting_area_top + int(painting_area_h * 0.3)

                # 15%透明度叠加
                sketch_rgba = np.array(sketch_img)
                sketch_rgba[:, :, 3] = (sketch_rgba[:, :, 3] * 0.15).astype(np.uint8)
                sketch_alpha = Image.fromarray(sketch_rgba).convert('RGBA')

                canvas_rgba = canvas.convert('RGBA')
                canvas_rgba.paste(sketch_alpha, (sketch_x, sketch_y), sketch_alpha)
                canvas = canvas_rgba.convert('RGB')

        # Layer 6: 个性化文字（找空白区域嵌入竖排落款）
        draw = ImageDraw.Draw(canvas)
        # 取叙事摘要作为落款文字（取第一段或摘要）
        calligraphic_text = narrative_result.get("summary", "")
        if not calligraphic_text and narrative_result.get("paragraphs"):
            calligraphic_text = narrative_result["paragraphs"][0][:20]

        if calligraphic_text:
            # 估算文字尺寸（竖排）
            test_img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            self._draw_vertical_calligraphy(test_img, calligraphic_text[:10],
                                           (w // 2, h // 2), 32)
            # 找一个合适的位置
            text_w, text_h = 60, len(calligraphic_text[:10]) * 40
            text_x, text_y = self._find_empty_region(canvas, (text_w, text_h))
            # 实际绘制（深灰色墨色）
            self._draw_vertical_calligraphy(canvas, calligraphic_text[:15],
                                           (w - 120, h - 300), 28,
                                           (60, 50, 40))

        # Layer 7: 时间戳 + 编号 + 印章
        date_str = datetime.now().strftime("%Y.%m.%d")
        id_str = unique_id or f"ML{int(datetime.now().timestamp())}"

        font_small = self.font_manager._load_font("simhei.ttf", 20)
        draw.text((40, h - 60), f"湖南大学·寻麓千年色", font=font_small,
                 fill=(120, 100, 80))
        draw.text((40, h - 35), f"{date_str}  #{id_str}", font=font_small,
                 fill=(120, 100, 80))

        # 印章
        seal_text = "湖大"
        font_seal = self.font_manager._load_font("simhei.ttf", 22)
        seal_size = 65
        seal_x, seal_y = w - 100, h - 90
        draw.rectangle([(seal_x, seal_y), (seal_x + seal_size, seal_y + seal_size)],
                      outline=self.config.seal_color, width=2)
        bbox = draw.textbbox((0, 0), seal_text, font=font_seal)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((seal_x + (seal_size - text_w) // 2,
                   seal_y + (seal_size - text_h) // 2),
                  seal_text, font=font_seal, fill=self.config.seal_color)

        # Layer 8: 时代风格边框
        self._draw_period_border(canvas, border_style)

        return canvas

    def _create_canvas(self, height: int = None) -> Image.Image:
        """创建画布（宣纸底），高度可选动态"""
        h = height or self.config.height
        return Image.new('RGB', (self.config.width, h), self.config.bg_color)

    def _paste_painting(self, canvas: Image.Image, painting: Image.Image,
                        content_start_y: int):
        """粘贴底图，content_start_y 为正文起始位置"""
        target_height = int(self.config.height * self.config.image_ratio)
        aspect_ratio = painting.width / painting.height
        target_width = int(target_height * aspect_ratio)

        if target_width > self.config.width:
            target_width = self.config.width
            target_height = int(target_width / aspect_ratio)

        painting = painting.resize((target_width, target_height), Image.LANCZOS)
        y_offset = content_start_y - target_height - self.PAINTING_GAP
        x_offset = (self.config.width - target_width) // 2
        canvas.paste(painting, (x_offset, max(y_offset, 0)))

    def _draw_title(self, canvas: Image.Image, title: str):
        """绘制标题"""
        draw = ImageDraw.Draw(canvas)
        font = self.font_manager.get_title_font(60)
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        x = (self.config.width - text_width) // 2
        y = 100
        draw.text((x + 2, y + 2), title, font=font, fill=(180, 170, 160))
        draw.text((x, y), title, font=font, fill=self.config.title_color)

    def _draw_narrative(self, canvas: Image.Image, paragraphs: List[str],
                        start_y: int) -> int:
        """绘制叙事正文，返回结束 Y 坐标"""
        draw = ImageDraw.Draw(canvas)
        font = self.font_manager.get_text_font(32)
        max_width = self.config.width - self.MARGIN_X * 2
        y = start_y

        for para in paragraphs:
            lines = self._wrap_para(draw, para, font, max_width)
            for line in lines:
                draw.text((self.MARGIN_X, y), line, font=font,
                         fill=self.config.text_color)
                y += self.LINE_HEIGHT
            y += self.PARA_SPACING

        return y

    def _draw_divider(self, canvas: Image.Image, content_start_y: int):
        """绘制分隔线"""
        draw = ImageDraw.Draw(canvas)
        y = content_start_y - 30
        draw.line([(100, y), (self.config.width - 100, y)],
                 fill=(180, 170, 160), width=1)

    def _draw_signature(self, canvas: Image.Image, text: str,
                        date_str: str = None, top_y: int = 0):
        """绘制落款，top_y 为正文结束后的起始位置"""
        draw = ImageDraw.Draw(canvas)
        font = self.font_manager.get_signature_font(28)
        if date_str is None:
            date_str = datetime.now().strftime("%Y.%m.%d")
        x = self.config.width - 280
        y = top_y + 20
        draw.text((x, y), text, font=font, fill=self.config.signature_color)
        draw.text((x, y + 40), date_str, font=font, fill=self.config.signature_color)

    def _draw_seal(self, canvas: Image.Image, text: str = "湖大",
                   top_y: int = 0):
        """绘制印章，top_y 为正文结束后的起始位置"""
        draw = ImageDraw.Draw(canvas)
        font = self.font_manager.get_seal_font(28)
        seal_size = 70
        seal_x = self.config.width - 100 - seal_size
        seal_y = top_y + 10
        draw.rectangle([seal_x, seal_y, seal_x + seal_size, seal_y + seal_size],
                      outline=self.config.seal_color, width=2)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = seal_x + (seal_size - text_w) // 2
        text_y = seal_y + (seal_size - text_h) // 2
        draw.text((text_x, text_y), text, font=font, fill=self.config.seal_color)

    def create_postcard(self,
                       narrative_result: Dict,
                       image_source: str = None) -> Optional[Image.Image]:
        """
        生成明信片（画布高度随文字内容动态调整）

        Args:
            narrative_result: {
                "title": "你寻到的千年色",
                "paragraphs": ["第一段", "第二段", ...],
                "summary": "一句话总结"
            }
            image_source: 图像来源（Base64字符串 或 URL）

        Returns:
            PIL Image对象，失败返回None
        """
        if not PIL_AVAILABLE:
            print("[错误] Pillow未安装")
            return None

        # 计算底图区域
        painting_area_height = int(self.config.height * self.config.image_ratio)
        content_start_y = painting_area_height + self.PAINTING_GAP

        # 计算叙事文字所需高度
        paragraphs = narrative_result.get("paragraphs", [])
        if paragraphs:
            temp_img = Image.new('RGB', (self.config.width, 100))
            temp_draw = ImageDraw.Draw(temp_img)
            font = self.font_manager.get_text_font(32)
            max_width = self.config.width - self.MARGIN_X * 2
            text_height = self._calc_text_height(temp_draw, paragraphs, font, max_width)
        else:
            text_height = 0

        # 动态画布高度
        total_height = content_start_y + text_height + self.FOOTER_HEIGHT
        canvas_height = max(self.config.height, total_height)

        # 1. 创建画布
        canvas = self._create_canvas(canvas_height)

        # 2. 加载并粘贴底图
        painting = None
        if image_source:
            if image_source.startswith("data:"):
                painting = self._load_base64_image(image_source)
            elif image_source.startswith("http"):
                painting = self._load_image_from_url(image_source)

            if painting:
                self._paste_painting(canvas, painting, content_start_y)

        # 3. 绘制标题
        title = narrative_result.get("title", "你寻到的千年色")
        self._draw_title(canvas, title)

        # 4. 绘制分隔线
        self._draw_divider(canvas, content_start_y)

        # 5. 绘制叙事正文
        end_y = content_start_y
        if paragraphs:
            end_y = self._draw_narrative(canvas, paragraphs, content_start_y)

        # 6. 绘制落款
        date_str = datetime.now().strftime("%Y.%m.%d")
        self._draw_signature(canvas, "湖南大学 · 寻麓千年色", date_str, top_y=end_y)

        # 7. 绘制印章
        self._draw_seal(canvas, "湖大", top_y=end_y)

        return canvas

    def save(self, canvas: Image.Image, filepath: str, format: str = "PNG"):
        """
        保存明信片

        Args:
            canvas: PIL Image对象
            filepath: 保存路径
            format: 保存格式 (PNG/JPEG/PDF)
        """
        canvas.save(filepath, format=format)
        print(f"[OK] 明信片已保存: {filepath}")

    def to_base64(self, canvas: Image.Image, format: str = "PNG") -> str:
        """
        将明信片转为Base64

        Args:
            canvas: PIL Image对象
            format: 图像格式

        Returns:
            Base64字符串 (带data URI前缀)
        """
        buffer = BytesIO()
        canvas.save(buffer, format=format)
        image_bytes = buffer.getvalue()
        encoded = base64.b64encode(image_bytes).decode('utf-8')

        mime_type = "image/png" if format.upper() == "PNG" else "image/jpeg"
        return f"data:{mime_type};base64,{encoded}"

    def to_json(self,
                narrative_result: Dict,
                image_source: str = None,
                include_base64: bool = True) -> Dict:
        """
        生成完整的叙事卡JSON数据

        Args:
            narrative_result: 叙事内容
            image_source: 底图来源
            include_base64: 是否包含Base64图像

        Returns:
            {
                "title": "...",
                "paragraphs": [...],
                "summary": "...",
                "image_base64": "...",
                "created_at": "...",
                "metadata": {...}
            }
        """
        result = {
            "title": narrative_result.get("title", "你寻到的千年色"),
            "paragraphs": narrative_result.get("paragraphs", []),
            "summary": narrative_result.get("summary", ""),
            "created_at": datetime.now().isoformat(),
        }

        # 生成明信片图像
        canvas = self.create_postcard(narrative_result, image_source)
        if canvas:
            if include_base64:
                result["image_base64"] = self.to_base64(canvas)
            result["image_width"] = canvas.width
            result["image_height"] = canvas.height

        return result


def create_generator(config: PostcardConfig = None) -> PostcardGenerator:
    """创建明信片生成器"""
    return PostcardGenerator(config)


# 测试函数
def test_postcard():
    """测试明信片生成"""
    if not PIL_AVAILABLE:
        print("[跳过] Pillow未安装")
        return

    generator = PostcardGenerator()

    # 测试数据
    test_narrative = {
        "title": "你寻到的千年色",
        "paragraphs": [
            "在岳麓山的绿荫下，我找到了属于自己的颜色。",
            "那是千年传承的力量，是实事求是的光芒。",
            "湖大的色彩，不在过去，在此刻，在我的心中。",
        ],
        "summary": "寻色之旅，是一场与历史的对话。"
    }

    print("[测试] 生成明信片...")

    # 生成（不加载底图）
    canvas = generator.create_postcard(test_narrative)

    if canvas:
        # 保存测试
        output_path = "./test_postcard.png"
        generator.save(canvas, output_path)
        print(f"[OK] 测试明信片已保存: {output_path}")

        # 生成JSON
        json_data = generator.to_json(test_narrative)
        print(f"[OK] JSON数据: {len(json_data.get('image_base64', ''))} bytes")
    else:
        print("[错误] 明信片生成失败")


if __name__ == "__main__":
    test_postcard()
