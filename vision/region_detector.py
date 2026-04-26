"""
区域标定与检测模块
通过物理标定物（彩色矩形）在图像中圈定识别区域
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class RegionCalibrator:
    """
    区域标定器
    通过检测物理标定物（彩色矩形框）来确定识别区域
    """

    def __init__(self,
                 calibrate_color: str = "red",
                 min_area: int = 1000):
        """
        Args:
            calibrate_color: 标定物颜色 ("red", "blue", "yellow", "green")
            min_area: 最小区域面积（过滤噪声）
        """
        self.calibrate_color = calibrate_color
        self.min_area = min_area
        self.calibrated_region = None  # 标定后的区域 (x, y, w, h)
        self.calibration_complete = False

        # 颜色HSV范围
        self.color_ranges = {
            "red": {
                "lower": np.array([0, 100, 100]),
                "upper": np.array([10, 255, 255]),
                "lower2": np.array([160, 100, 100]),
                "upper2": np.array([180, 255, 255]),
            },
            "blue": {
                "lower": np.array([100, 150, 50]),
                "upper": np.array([130, 255, 255]),
            },
            "yellow": {
                "lower": np.array([20, 100, 100]),
                "upper": np.array([30, 255, 255]),
            },
            "green": {
                "lower": np.array([40, 80, 80]),
                "upper": np.array([80, 255, 255]),
            },
        }

    def _get_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """根据颜色获取二值化掩码"""
        if self.calibrate_color == "red":
            # 红色有两个范围（因为在HSV中是循环的）
            mask1 = cv2.inRange(hsv_image,
                               self.color_ranges["red"]["lower"],
                               self.color_ranges["red"]["upper"])
            mask2 = cv2.inRange(hsv_image,
                               self.color_ranges["red"]["lower2"],
                               self.color_ranges["red"]["upper2"])
            return cv2.bitwise_or(mask1, mask2)
        else:
            return cv2.inRange(hsv_image,
                              self.color_ranges[self.calibrate_color]["lower"],
                              self.color_ranges[self.calibrate_color]["upper"])

    def _find_rectangle_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """在掩码中找到矩形轮廓"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect = None
        best_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            # 近似轮廓为矩形
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            # 只考虑4边形的较大轮廓
            if len(approx) == 4 and area > best_area:
                best_area = area
                best_rect = approx

        return best_rect

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """将4个点按左上、右上、右下、左下排序"""
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)

        # 求和最小的为左上角
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]

        # 求和最大的为右下角
        rect[2] = pts[np.argmax(s)]

        # 差值最小的为右上角
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]

        # 差值最大的为左下角
        rect[3] = pts[np.argmax(diff)]

        return rect.astype(np.int32)

    def calibrate(self, frame: np.ndarray) -> bool:
        """
        标定：从帧中检测标定物并设置识别区域

        Args:
            frame: 输入图像 (BGR格式)

        Returns:
            标定是否成功
        """
        # 转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 获取颜色掩码
        mask = self._get_mask(hsv)

        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 找到矩形轮廓
        contour = self._find_rectangle_contour(mask)

        if contour is not None:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            self.calibrated_region = (x, y, w, h)
            self.calibration_complete = True
            return True

        return False

    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        """获取已标定的区域 (x, y, w, h)"""
        return self.calibrated_region

    def is_calibrated(self) -> bool:
        """是否已标定"""
        return self.calibration_complete

    def reset(self):
        """重置标定"""
        self.calibrated_region = None
        self.calibration_complete = False

    def draw_calibration_visualization(self,
                                       frame: np.ndarray,
                                       show_mask: bool = False) -> np.ndarray:
        """
        绘制标定可视化

        Args:
            frame: 输入图像
            show_mask: 是否显示检测掩码

        Returns:
            可视化后的图像
        """
        result = frame.copy()

        # 转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 获取掩码
        mask = self._get_mask(hsv)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if show_mask:
            cv2.imshow("Calibration Mask", mask)

        # 找到矩形
        contour = self._find_rectangle_contour(mask)

        if contour is not None:
            # 绘制检测到的矩形
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)

            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 绘制边界框和标签
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, "Calibration Region", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示成功信息
            cv2.putText(result, "[Calibrated]", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(result, "[Not Calibrated - Show Color Marker]",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return result


class RegionDetector:
    """
    区域检测器
    在指定区域内检测手部等目标
    """

    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None):
        """
        Args:
            region: 检测区域 (x, y, w, h)，如果为None则检测整个画面
        """
        self.region = region

    def set_region(self, region: Tuple[int, int, int, int]):
        """设置检测区域"""
        self.region = region

    def clear_region(self):
        """清除区域，检测整个画面"""
        self.region = None

    def is_point_in_region(self, x: int, y: int) -> bool:
        """判断点是否在区域内"""
        if self.region is None:
            return True
        rx, ry, rw, rh = self.region
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def crop_region(self, frame: np.ndarray) -> np.ndarray:
        """裁剪出检测区域"""
        if self.region is None:
            return frame
        x, y, w, h = self.region
        return frame[y:y+h, x:x+w]

    def get_region_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        获取区域图像及其在原图中的偏移量

        Returns:
            (区域图像, (offset_x, offset_y))
        """
        if self.region is None:
            return frame, (0, 0)

        x, y, w, h = self.region
        return frame[y:y+h, x:x+w], (x, y)

    def draw_region(self, frame: np.ndarray,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
        """在图像上绘制检测区域"""
        result = frame.copy()

        if self.region is not None:
            x, y, w, h = self.region
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

            label = f"Region: {w}x{h}"
            cv2.putText(result, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return result


class ManualRegionSelector:
    """
    手动区域选择器 - 四点透视方式
    通过鼠标点击4个点确定任意四边形区域，适应透视变换
    """

    def __init__(self):
        self.points = []  # 存储点击的4个点
        self.max_points = 4
        self.preview_point = None  # 鼠标当前位置预览

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.preview_point = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < self.max_points:
                self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键删除最后一个点
            if self.points:
                self.points.pop()

    def get_points(self) -> List[Tuple[int, int]]:
        """获取当前所有点"""
        return self.points.copy()

    def get_quad(self) -> Optional[np.ndarray]:
        """
        获取四边形区域的4个点（按顺序排列）

        Returns:
            4个点的numpy数组，按左上、右上、右下、左下顺序排列
        """
        if len(self.points) != 4:
            return None

        pts = np.array(self.points, dtype=np.float32)

        # 计算每个点的权重（用于排序）
        # 左上: x+y最小  右下: x+y最大
        # 右上: x-y最小  左下: x-y最大
        sums = pts.sum(axis=1)
        diffs = pts[:, 0] - pts[:, 1]

        # 按和排序获取左上、右下
        tl_idx = np.argmin(sums)  # 左上
        br_idx = np.argmax(sums)  # 右下

        # 按差排序获取右上、左下
        tr_idx = np.argmin(diffs)  # 右上
        bl_idx = np.argmax(diffs)  # 左下

        quad = np.array([
            pts[tl_idx],   # 左上
            pts[tr_idx],   # 右上
            pts[br_idx],   # 右下
            pts[bl_idx],   # 左下
        ], dtype=np.float32)

        return quad

    def get_warped_region(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        获取透视变换后的矩形区域

        Args:
            frame: 输入图像

        Returns:
            (透视变换后的图像, 变换矩阵) 或 None
        """
        quad = self.get_quad()
        if quad is None:
            return None

        # 计算矩形边界
        x, y, w, h = cv2.boundingRect(quad.astype(np.int32))

        # 目标矩形顶点
        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(quad, dst)

        # 应用透视变换
        warped = cv2.warpPerspective(frame, M, (w, h))

        return warped, M

    def is_complete(self) -> bool:
        """是否已选择4个点"""
        return len(self.points) == self.max_points

    def reset(self):
        """重置选择"""
        self.points = []
        self.preview_point = None

    def draw_selection(self, frame: np.ndarray) -> np.ndarray:
        """绘制选择框和预览"""
        result = frame.copy()

        # 绘制已选择的点
        for i, pt in enumerate(self.points):
            color = (0, 255, 0)
            cv2.circle(result, pt, 8, color, -1)
            cv2.circle(result, pt, 8, (255, 255, 255), 2)
            cv2.putText(result, str(i + 1), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制点之间的连线
        if len(self.points) >= 2:
            pts_array = np.array(self.points, dtype=np.int32)
            cv2.polylines(result, [pts_array], False, (0, 255, 0), 2)

        # 如果有4个点，绘制完整四边形
        if len(self.points) == 4:
            cv2.polylines(result, [pts_array], True, (0, 255, 0), 3)
            quad = self.get_quad()
            if quad is not None:
                x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
                cv2.putText(result, f"Region: {w}x{h}", (x, y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 预览下一个点的位置
        if self.preview_point and len(self.points) < self.max_points:
            remaining = self.max_points - len(self.points)
            cv2.putText(result, f"Click {remaining} more point(s)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        return result


def create_calibrator(color: str = "red") -> RegionCalibrator:
    """工厂函数：创建标定器"""
    return RegionCalibrator(calibrate_color=color)


def create_region_detector(region: Optional[Tuple[int, int, int, int]] = None) -> RegionDetector:
    """工厂函数：创建区域检测器"""
    return RegionDetector(region=region)


def create_manual_selector() -> ManualRegionSelector:
    """工厂函数：创建手动选择器（四点透视方式）"""
    return ManualRegionSelector()
