"""图像预处理模块.

实现 PDF 第 4 步"图像预处理"中描述的色阶、曲线、亮度、对比度、
灰度、饱和度、滤波、锐化、平滑、查找边缘、底片等功能.
主要目的是加大孔洞区域与周边区域的色彩对比.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class PreprocessParams:
    """预处理参数."""
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    blur_kernel: int = 0
    sharpen: bool = False
    invert: bool = False
    to_gray: bool = False
    edge_detect: bool = False
    auto_levels: bool = True


def auto_levels(image: np.ndarray) -> np.ndarray:
    """自动色阶拉伸.

    将图像的灰度直方图拉伸到 [0, 255] 范围,
    增强暗色孔洞与亮色岩石的对比度.
    """
    if image.ndim == 2:
        out = image.copy()
        for c in range(1):
            channel = out
            lo = float(np.percentile(channel, 2))
            hi = float(np.percentile(channel, 98))
            if hi - lo < 1e-6:
                continue
            channel = (channel - lo) * (255.0 / (hi - lo))
            channel = np.clip(channel, 0, 255).astype(np.uint8)
            out = channel
        return out

    out = image.copy()
    if out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lo = float(np.percentile(out, 2))
    hi = float(np.percentile(out, 98))
    if hi - lo > 1e-6:
        out = np.clip((out - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    if image.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return out


def adjust_brightness(image: np.ndarray, delta: float) -> np.ndarray:
    """调整亮度(范围 -100~100)."""
    delta = float(np.clip(delta, -100, 100))
    out = cv2.convertScaleAbs(image, alpha=1.0, beta=delta)
    return out


def adjust_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    """调整对比度(alpha 范围 0~3)."""
    alpha = float(np.clip(alpha, 0.0, 3.0))
    out = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return out


def adjust_saturation(image: np.ndarray, scale: float) -> np.ndarray:
    """调整饱和度(对彩色图像,转换为 HSV 调整 S 通道)."""
    if image.ndim == 2 or image.shape[2] == 1:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma 校正.
    标准公式: out = in^gamma × 255
    - gamma > 1: 中间调变暗
    - gamma < 1: 中间调变亮
    """
    if gamma <= 0:
        return image
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """转灰度图."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur_image(image: np.ndarray, kernel: int) -> np.ndarray:
    """高斯/中值平滑."""
    if kernel <= 0:
        return image
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    if image.ndim == 2:
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def median_filter(image: np.ndarray, kernel: int = 3) -> np.ndarray:
    """中值滤波(对椒盐噪声效果好)."""
    if kernel <= 0:
        return image
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.medianBlur(image, kernel)


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """锐化(USM 风格)."""
    if image.ndim == 2:
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def invert_image(image: np.ndarray) -> np.ndarray:
    """底片效果(像素取反)."""
    return cv2.bitwise_not(image)


def detect_edges(image: np.ndarray) -> np.ndarray:
    """Canny 边缘检测."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)


def curve_adjust(image: np.ndarray, low: float = 0, mid: float = 128, high: float = 255) -> np.ndarray:
    """三段式曲线调节(类似 Photoshop 曲线)."""
    low = max(0, min(low, 254))
    mid = max(low + 1, min(mid, high - 1))
    high = max(mid + 1, min(high, 255))
    if image.ndim == 2:
        src = image
    else:
        src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i <= low:
            lut[i] = 0
        elif i <= mid:
            t = (i - low) / max(1, mid - low)
            lut[i] = int(t * 128)
        elif i <= high:
            t = (i - mid) / max(1, high - mid)
            lut[i] = int(128 + t * 127)
        else:
            lut[i] = 255
    return cv2.LUT(src, lut)


def preprocess(image: np.ndarray, params: PreprocessParams | None = None) -> np.ndarray:
    """一键预处理流水线.
    顺序: 底片 → 亮度/对比度 → Gamma → 色阶 → 滤波 → 锐化 → 转灰度 → 边缘.
    """
    if params is None:
        params = PreprocessParams()
    out = image.copy()
    if params.invert:
        out = invert_image(out)
    if params.brightness != 0 or params.contrast != 1.0:
        out = cv2.convertScaleAbs(out, alpha=params.contrast, beta=params.brightness)
    if params.gamma != 1.0:
        out = adjust_gamma(out, params.gamma)
    if params.saturation != 1.0:
        out = adjust_saturation(out, params.saturation)
    if params.auto_levels:
        out = auto_levels(out)
    if params.blur_kernel > 0:
        out = blur_image(out, params.blur_kernel)
    if params.sharpen:
        out = sharpen_image(out)
    if params.to_gray:
        out = to_grayscale(out)
    if params.edge_detect:
        out = detect_edges(out)
    return out


def apply_pore_enhancement(image: np.ndarray) -> np.ndarray:
    """面向孔洞识别的增强策略.
    1. 转到 LAB 颜色空间,仅对 L 通道做 CLAHE
    2. 自动色阶拉伸
    """
    if image.ndim == 2:
        return auto_levels(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return auto_levels(bgr)


__all__ = [
    "PreprocessParams",
    "auto_levels", "adjust_brightness", "adjust_contrast",
    "adjust_saturation", "adjust_gamma",
    "to_grayscale", "blur_image", "median_filter",
    "sharpen_image", "invert_image", "detect_edges",
    "curve_adjust", "preprocess", "apply_pore_enhancement",
]
