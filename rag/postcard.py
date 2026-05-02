"""
明信片/叙事卡生成模块
生成包含底图+叙事文字的完整纪念图片
"""

import os
import json
import base64
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
    image_ratio: float = 0.55    # 底图占画面55%
    content_start_y: int = 850   # 文字内容起始Y


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

    def _create_canvas(self) -> Image.Image:
        """创建画布（宣纸底）"""
        canvas = Image.new('RGB',
                          (self.config.width, self.config.height),
                          self.config.bg_color)
        return canvas

    def _paste_painting(self, canvas: Image.Image, painting: Image.Image):
        """粘贴底图"""
        # 调整底图尺寸
        target_height = int(self.config.height * self.config.image_ratio)
        aspect_ratio = painting.width / painting.height
        target_width = int(target_height * aspect_ratio)

        # 如果宽度超出，缩放
        if target_width > self.config.width:
            target_width = self.config.width
            target_height = int(target_width / aspect_ratio)

        painting = painting.resize((target_width, target_height), Image.LANCZOS)

        # 粘贴到底部（留出上方空间给文字）
        y_offset = self.config.content_start_y - target_height - 50
        x_offset = (self.config.width - target_width) // 2

        canvas.paste(painting, (x_offset, y_offset))

    def _draw_title(self, canvas: Image.Image, title: str):
        """绘制标题"""
        draw = ImageDraw.Draw(canvas)
        font = self.font_manager.get_title_font(60)

        # 计算居中位置
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        x = (self.config.width - text_width) // 2
        y = 100

        # 绘制阴影效果（模拟书法墨迹）
        draw.text((x + 2, y + 2), title, font=font, fill=(180, 170, 160))
        draw.text((x, y), title, font=font, fill=self.config.title_color)

    def _draw_narrative(self, canvas: Image.Image, paragraphs: List[str]):
        """绘制叙事正文"""
        draw = ImageDraw.Draw(canvas)
        font = self.font_manager.get_text_font(32)

        y = self.config.content_start_y
        line_spacing = 70  # 行间距

        for para in paragraphs:
            # 自动换行（简单实现）
            words = para
            if len(words) > 20:
                # 每行约20字
                lines = [words[i:i+20] for i in range(0, len(words), 20)]
                words = "\n".join(lines)

            draw.text((60, y), words, font=font, fill=self.config.text_color)
            y += line_spacing

    def _draw_divider(self, canvas: Image.Image):
        """绘制分隔线（简约水墨风格）"""
        draw = ImageDraw.Draw(canvas)
        y = self.config.content_start_y - 30

        # 画一条淡淡的水平线
        draw.line([(100, y), (self.config.width - 100, y)],
                 fill=(180, 170, 160), width=1)

    def _draw_signature(self, canvas: Image.Image, text: str, date_str: str = None):
        """绘制落款"""
        draw = ImageDraw.Draw(canvas)
        font = self.font_manager.get_signature_font(28)

        if date_str is None:
            date_str = datetime.now().strftime("%Y.%m.%d")

        # 底部右侧
        x = self.config.width - 280
        y = self.config.height - 100

        draw.text((x, y), text, font=font, fill=self.config.signature_color)
        draw.text((x, y + 40), date_str, font=font, fill=self.config.signature_color)

    def _draw_seal(self, canvas: Image.Image, text: str = "湖大"):
        """绘制印章（右下角）"""
        draw = ImageDraw.Draw(canvas)
        font = self.font_manager.get_seal_font(28)

        # 印章位置
        seal_size = 70
        seal_x = self.config.width - 100 - seal_size
        seal_y = self.config.height - 100 - seal_size

        # 印章框
        draw.rectangle([seal_x, seal_y, seal_x + seal_size, seal_y + seal_size],
                      outline=self.config.seal_color, width=2)

        # 印章文字（居中）
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
        生成明信片

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

        # 1. 创建画布
        canvas = self._create_canvas()

        # 2. 加载并粘贴底图
        painting = None
        if image_source:
            if image_source.startswith("data:"):
                painting = self._load_base64_image(image_source)
            elif image_source.startswith("http"):
                painting = self._load_image_from_url(image_source)

            if painting:
                self._paste_painting(canvas, painting)

        # 3. 绘制标题
        title = narrative_result.get("title", "你寻到的千年色")
        self._draw_title(canvas, title)

        # 4. 绘制分隔线
        self._draw_divider(canvas)

        # 5. 绘制叙事正文
        paragraphs = narrative_result.get("paragraphs", [])
        if paragraphs:
            self._draw_narrative(canvas, paragraphs)

        # 6. 绘制落款
        date_str = datetime.now().strftime("%Y.%m.%d")
        self._draw_signature(canvas, "湖南大学 · 寻麓千年色", date_str)

        # 7. 绘制印章
        self._draw_seal(canvas, "湖大")

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
