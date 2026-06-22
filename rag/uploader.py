"""
明信片 → OSS 上传 + 二维码生成
"""

import os
import io
import uuid
import base64
import logging
from datetime import datetime

import oss2
import qrcode
from PIL import Image

from .oss_config import ACCESS_KEY_ID, ACCESS_KEY_SECRET, ENDPOINT, BUCKET_NAME, PUBLIC_DOMAIN

logger = logging.getLogger("Uploader")

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class PostcardUploader:
    """明信片上传器"""

    def __init__(self):
        self.auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
        self.bucket = oss2.Bucket(self.auth, ENDPOINT, BUCKET_NAME)

    def upload(self, image: Image.Image, unique_id: str = None) -> dict:
        """
        上传明信片到 OSS，生成二维码

        Args:
            image: PIL Image 对象
            unique_id: 唯一 ID，不传则自动生成

        Returns:
            {
                "image_url": "https://...",     # 公网下载链接
                "qr_base64": "data:image/png;base64,...",  # 二维码 Base64
                "local_path": "./output/xxx.png",          # 本地路径
                "unique_id": "MLxxxxx"
            }
        """
        unique_id = unique_id or self._gen_id()

        # 1. 本地保存
        local_path = os.path.join(OUTPUT_DIR, f"postcard_{unique_id}.png")
        image.save(local_path, "PNG")
        logger.info(f"Saved: {local_path}")

        # 2. 上传 OSS
        oss_key = f"postcards/{unique_id}.png"
        try:
            self.bucket.put_object_from_file(oss_key, local_path)
            image_url = f"{PUBLIC_DOMAIN}/{oss_key}"
            logger.info(f"Uploaded: {image_url}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            # 兜底：用本地路径
            image_url = f"file://{os.path.abspath(local_path)}"

        # 3. 生成二维码
        qr_img = self._make_qr(image_url)
        qr_path = os.path.join(OUTPUT_DIR, f"qr_{unique_id}.png")
        qr_img.save(qr_path, "PNG")
        qr_base64 = self._img_to_base64(qr_img)

        return {
            "image_url": image_url,
            "qr_base64": qr_base64,
            "qr_local_path": qr_path,
            "local_path": local_path,
            "unique_id": unique_id
        }

    def _gen_id(self):
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        short = str(uuid.uuid4())[:6]
        return f"ML{ts}_{short}"

    def _make_qr(self, url: str, size: int = 3200) -> Image.Image:
        """生成二维码"""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        return qr.make_image(fill_color="black", back_color="white").resize((size, size))

    def _img_to_base64(self, img: Image.Image) -> str:
        """PIL Image → data URI"""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"


# 测试
if __name__ == "__main__":
    print("Testing OSS upload...")
    uploader = PostcardUploader()

    # 创建测试图片
    test_img = Image.new("RGB", (200, 300), (245, 240, 230))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_img)
    draw.text((10, 140), "Test Postcard", fill=(50, 40, 30))

    result = uploader.upload(test_img)
    print(f"URL: {result['image_url']}")
    print(f"QR: {result['qr_base64'][:60]}...")
    print("Done!")
