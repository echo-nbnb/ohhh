"""
手势检测模块
使用MediaPipe Hands检测手部范围（凸包）
参考: D:\projects\gesturesRecognition
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 手部关键点索引
class HandLandmarkIndex:
    """MediaPipe Hand 21个关键点索引"""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class HandAreaDrawer:
    """手部范围绘制器（凸包）"""

    def __init__(self):
        # 绘制颜色 (BGR)
        self.hull_color = (0, 200, 0)      # 绿色 - 凸包轮廓
        self.fill_color = (0, 100, 0, 50)  # 半透明填充
        self.outline_color = (0, 255, 0)   # 深绿色 - 外轮廓
        self.point_color = (0, 255, 255)  # 黄色 - 端点

    def get_pixel_points(self, landmarks, w: int, h: int) -> List[Tuple[int, int]]:
        """获取像素坐标的关键点"""
        points = []
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
        return points

    def draw_hand_area(self,
                       image: np.ndarray,
                       landmarks,
                       fill_alpha: float = 0.3,
                       line_thickness: int = 2) -> np.ndarray:
        """
        绘制手部覆盖范围（凸包）

        Args:
            image: 输入图像 (BGR格式)
            landmarks: MediaPipe hand landmarks
            fill_alpha: 填充透明度 (0-1)
            line_thickness: 轮廓线粗细

        Returns:
            绘制了手部范围的图像
        """
        if landmarks is None:
            return image

        h, w = image.shape[:2]
        points = self.get_pixel_points(landmarks, w, h)

        # 凸包需要至少3个点
        if len(points) < 3:
            return image

        # 转换为numpy数组
        points_array = np.array(points, dtype=np.int32)

        # 计算凸包
        hull = cv2.convexHull(points_array)

        # 绘制半透明填充
        hull_fill = hull.copy()
        overlay = image.copy()
        cv2.fillPoly(overlay, [hull_fill], self.hull_color)
        cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0, image)

        # 绘制凸包轮廓
        cv2.polylines(image, [hull], True, self.outline_color, line_thickness)

        return image

    def draw_outer_points(self,
                          image: np.ndarray,
                          landmarks,
                          point_radius: int = 5) -> np.ndarray:
        """
        绘制最外侧点（指尖和手腕）

        Args:
            image: 输入图像 (BGR格式)
            landmarks: MediaPipe hand landmarks
            point_radius: 端点半径

        Returns:
            绘制了端点的图像
        """
        if landmarks is None:
            return image

        h, w = image.shape[:2]
        points = self.get_pixel_points(landmarks, w, h)

        # 只绘制最外侧的点：手腕(0)和5个指尖(4,8,12,16,20)
        outer_indices = [0, 4, 8, 12, 16, 20]
        for idx in outer_indices:
            if idx < len(points):
                cv2.circle(image, points[idx], point_radius, self.point_color, -1)
                cv2.circle(image, points[idx], point_radius // 2, (255, 255, 255), -1)

        return image

    def draw_fingertips_connections(self,
                                    image: np.ndarray,
                                    landmarks,
                                    line_thickness: int = 2) -> np.ndarray:
        """
        绘制指尖连线形成手部轮廓

        Args:
            image: 输入图像 (BGR格式)
            landmarks: MediaPipe hand landmarks
            line_thickness: 连线粗细

        Returns:
            绘制了指尖连线的图像
        """
        if landmarks is None:
            return image

        h, w = image.shape[:2]
        points = self.get_pixel_points(landmarks, w, h)

        # 指尖和手腕索引
        fingertip_indices = [20, 4, 8, 12, 16, 0]  # 小指->拇指->食指->中指->无名指->手腕

        # 依次连接
        for i in range(len(fingertip_indices) - 1):
            pt1 = points[fingertip_indices[i]]
            pt2 = points[fingertip_indices[i + 1]]
            cv2.line(image, pt1, pt2, self.outline_color, line_thickness)

        # 绘制端点
        self.draw_outer_points(image, landmarks)

        return image


class HandDetector:
    """手部关键点检测器"""

    def __init__(self,
                 num_hands: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        初始化手部检测器

        Args:
            num_hands: 最大检测手数
            min_detection_confidence: 检测置信度阈值
            min_tracking_confidence: 追踪置信度阈值
        """
        self.num_hands = num_hands

        # 创建HandLandmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.area_drawer = HandAreaDrawer()

    def process_frame(self, image: np.ndarray, timestamp_ms: int = 0):
        """
        处理视频帧

        Args:
            image: 输入图像 (BGR格式)
            timestamp_ms: 时间戳（毫秒）

        Returns:
            (检测结果, 第一个手的landmarks)
        """
        # BGR转RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为MediaPipe图像
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # 检测
        results = self.detector.detect_for_video(mp_image, timestamp_ms)

        hand_landmarks = None
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand_landmarks = results.hand_landmarks[0]

        return results, hand_landmarks

    def detect_and_draw(self,
                        image: np.ndarray,
                        timestamp_ms: int = 0,
                        mode: str = "area") -> Tuple[np.ndarray, any]:
        """
        检测并绘制手部范围

        Args:
            image: 输入图像 (BGR格式)
            timestamp_ms: 时间戳
            mode: 绘制模式 - "area"(凸包) / "outline"(指尖连线) / "points"(端点)

        Returns:
            (绘制了手部范围的图像, 检测结果)
        """
        results, landmarks = self.process_frame(image, timestamp_ms)

        if landmarks:
            if mode == "area":
                image = self.area_drawer.draw_hand_area(image, landmarks)
            elif mode == "outline":
                image = self.area_drawer.draw_fingertips_connections(image, landmarks)
            elif mode == "points":
                image = self.area_drawer.draw_outer_points(image, landmarks)
            else:
                # 默认绘制凸包
                image = self.area_drawer.draw_hand_area(image, landmarks)

        return image, results

    def get_landmarks(self, results: any) -> List[List[Tuple[float, float, float]]]:
        """
        从检测结果提取关键点坐标

        Args:
            results: HandLandmarkerResult

        Returns:
            每只手的21个关键点列表，每个关键点为(x, y, z)元组
        """
        if not results or not results.hand_landmarks:
            return []

        landmarks_list = []
        for hand_landmarks in results.hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            landmarks_list.append(landmarks)

        return landmarks_list

    def recognize_gesture(self, landmarks) -> str:
        """识别手势"""
        if landmarks is None:
            return "✋ 未检测到手"

        try:
            lm = landmarks

            # 各指尖坐标
            thumb_tip = (lm[4].x, lm[4].y)
            index_tip = (lm[8].x, lm[8].y)
            middle_tip = (lm[12].x, lm[12].y)
            ring_tip = (lm[16].x, lm[16].y)
            pinky_tip = (lm[20].x, lm[20].y)

            # MCP坐标
            thumb_mcp = (lm[2].x, lm[2].y)
            index_mcp_y = lm[5].y
            middle_mcp_y = lm[9].y
            ring_mcp_y = lm[13].y
            pinky_mcp_y = lm[17].y

            # 判断手指伸展
            index_extended = index_tip[1] < index_mcp_y - 0.02
            middle_extended = middle_tip[1] < middle_mcp_y - 0.02
            ring_extended = ring_tip[1] < ring_mcp_y - 0.02
            pinky_extended = pinky_tip[1] < pinky_mcp_y - 0.02
            thumb_extended = thumb_tip[0] < thumb_mcp[0] - 0.02

            fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
            extended_count = sum(fingers)

            # 计算拇指和食指距离
            thumb_index_dist = np.sqrt(
                (thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2
            )

            # 手势识别
            if not any(fingers):
                return "✊ 拳头"
            if thumb_extended and not any(fingers[1:]):
                return "👍 点赞"
            if extended_count == 4 and not thumb_extended:
                return "🖐️ 手掌"
            if not thumb_extended and extended_count == 1:
                return "☝️ 1"
            if not thumb_extended and extended_count == 2:
                return "✌️ 2"
            if not thumb_extended and extended_count == 3:
                return "🤟 3"
            if not thumb_extended and extended_count == 4:
                return "🖖 4"
            if thumb_extended and index_extended and not middle_extended:
                return "🤘 摇滚"
            if thumb_index_dist < 0.05:
                return "👌 OK"

            return f"🤔 手势({extended_count}指)"

        except:
            return "✋ 检测中..."

    def close(self):
        """关闭检测器"""
        self.detector.close()


def create_hand_detector(num_hands: int = 1) -> HandDetector:
    """工厂函数：创建手部检测器"""
    return HandDetector(num_hands=num_hands)
