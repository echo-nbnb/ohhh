"""
视觉识别模块
"""

from .ipcamera import IPCamera, IPCameraManager, create_camera_from_url
from .detect import VisionDetector, FrameProcessor, StreamViewer, test_camera_connection
from .hand_detector import (
    HandDetector, HandAreaDrawer, HandLandmarkIndex,
    create_hand_detector
)
from .region_detector import (
    RegionCalibrator, RegionDetector, ManualRegionSelector,
    create_calibrator, create_region_detector, create_manual_selector
)

__all__ = [
    'IPCamera',
    'IPCameraManager',
    'create_camera_from_url',
    'VisionDetector',
    'FrameProcessor',
    'StreamViewer',
    'test_camera_connection',
    'HandDetector',
    'HandAreaDrawer',
    'HandLandmarkIndex',
    'create_hand_detector',
    'RegionCalibrator',
    'RegionDetector',
    'ManualRegionSelector',
    'create_calibrator',
    'create_region_detector',
    'create_manual_selector',
]
