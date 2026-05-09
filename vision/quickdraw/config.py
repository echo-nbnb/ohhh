"""
QuickDraw 训练配置
"""

import os

# ---- 路径 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "quickdraw_mobilenet.onnx")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pth")

# ---- 数据集 ----
# 每类下载样本数上限（.npy 文件通常有 100K-200K 样本，取前 N 个加速训练）
SAMPLES_PER_CLASS = 15000

# 训练/验证 分割
TRAIN_RATIO = 0.85
BATCH_SIZE = 256
NUM_WORKERS = 2

# ---- 模型 ----
# 输入: 1 × 28 × 28 灰度图
# 输出: NUM_CLASSES 类 QuickDraw 类别
INPUT_CHANNELS = 1
IMAGE_SIZE = 28

# ---- 训练 ----
EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# ---- QuickDraw 类别列表 ----
# 从 sketch_recognizer.py 的 QUICKDRAW_TO_OBJECT 映射表提取
# 文件名 = 类别名.replace(' ', '%20') 用于下载
CATEGORIES = [
    # 自然景观（10）
    "river", "mountain", "cloud", "rain", "moon", "star", "sun",
    "ocean", "pond",
    # 建筑/结构（10）
    "house", "castle", "church", "door", "fence", "bridge", "stairs",
    "streetlight", "lighthouse", "square",
    # 植物（7）
    "tree", "flower", "leaf", "bush", "grass", "cactus", "palm tree",
    # 室内/学习（7）
    "book", "pencil", "clock", "calendar", "envelope",
    "computer", "television",
    # 物品（10）
    "cell phone", "headphones", "microphone", "camera", "binoculars",
    "key", "hammer", "screwdriver", "compass", "map",
    # 器具（9）
    "backpack", "suitcase", "cup", "coffee cup", "wine bottle",
    "knife", "fork", "spoon", "basket",
    # 交通（7）
    "bicycle", "car", "bus", "train", "airplane", "sailboat",
    "ambulance",
    # 身体（4）
    "face", "eye", "hand", "foot",
    # 运动（4）
    "baseball", "basketball", "tennis racquet", "soccer ball",
    # 符号（4）
    "light bulb", "hat", "guitar", "piano",
    # 乐器（2）
    "violin", "trumpet",
    # 几何/其他（9）
    "umbrella", "wheel", "zigzag", "triangle", "circle",
    "line", "diamond", "hexagon", "octagon",
]

NUM_CLASSES = len(CATEGORIES)

# 类别名 → 索引
CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}
IDX_TO_CATEGORY = {i: cat for i, cat in enumerate(CATEGORIES)}

# QuickDraw 官方类别名 → 数据文件名（URL 编码）
# 空格在 URL 中需要编码为 %20
def category_filename(cat: str) -> str:
    return cat.replace(" ", "%20") + ".npy"

def category_url(cat: str) -> str:
    return "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/" + category_filename(cat)
