"""区域分割与孔洞提取模块.

实现 PDF 第 5 步"孔洞提取"中描述的区域分割、颜色匹配度、
连续区域、反选/撤销/还原等核心功能.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .preprocessing import to_grayscale


@dataclass
class SegmentationParams:
    """区域分割参数.
    Attributes:
        color: BGR 颜色三元组(种子点颜色)
        tolerance: 颜色匹配度,0~255,越小越严格
        connectivity: 4 或 8
        continuous: True 表示只对当前种子点邻域做连续区域分割,
                    False 表示全局匹配相同颜色的所有区域
        min_area_px: 最小保留区域(像素)
    """
    color: Tuple[int, int, int] = (0, 0, 0)
    tolerance: int = 30
    connectivity: int = 8
    continuous: bool = True
    min_area_px: int = 0


def _build_color_lut(target_bgr: Tuple[int, int, int], tolerance: int) -> np.ndarray:
    """构建 256×256×256 颜色匹配 LUT.
    由于直接构建三维 LUT 内存过大(64MB),改为按通道判断.
    """
    b, g, r = target_bgr
    tol = int(np.clip(tolerance, 0, 255))
    b_lo, b_hi = max(0, b - tol), min(255, b + tol)
    g_lo, g_hi = max(0, g - tol), min(255, g + tol)
    r_lo, r_hi = max(0, r - tol), min(255, r + tol)
    return b_lo, b_hi, g_lo, g_hi, r_lo, r_hi


def segment_by_color(image: np.ndarray, params: SegmentationParams) -> np.ndarray:
    """根据颜色匹配度提取区域.
    返回单通道 uint8 掩码,匹配区域为 255.
    """
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
    b_lo, b_hi, g_lo, g_hi, r_lo, r_hi = _build_color_lut(params.color, params.tolerance)
    b, g, r = cv2.split(bgr)
    mask = (
        (b >= b_lo) & (b <= b_hi) &
        (g >= g_lo) & (g <= g_hi) &
        (r >= r_lo) & (r <= r_hi)
    )
    return mask.astype(np.uint8) * 255


def _bgr_dist(image_bgr: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """BGR 欧氏距离图(用于连续区域分割)."""
    target = np.array(color, dtype=np.float32)
    diff = image_bgr.astype(np.float32) - target
    return np.sqrt(np.sum(diff * diff, axis=2))


def flood_fill_region(
    image: np.ndarray,
    seed: Tuple[int, int],
    params: SegmentationParams,
) -> np.ndarray:
    """连续区域分割(类似洪水填充).
    从 seed 点出发,只沿邻域扩展到与 seed 颜色相近的像素.
    """
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
    h, w = bgr.shape[:2]
    sy, sx = seed
    if not (0 <= sy < h and 0 <= sx < w):
        return np.zeros((h, w), dtype=np.uint8)
    tol = int(np.clip(params.tolerance, 0, 255))
    connectivity = params.connectivity
    seed_color = tuple(int(c) for c in bgr[sy, sx])
    b_lo, b_hi, g_lo, g_hi, r_lo, r_hi = _build_color_lut(seed_color, tol)
    b, g, r = cv2.split(bgr)
    match_b = (b >= b_lo) & (b <= b_hi)
    match_g = (g >= g_lo) & (g <= g_hi)
    match_r = (r >= r_lo) & (r <= r_hi)
    matches = match_b & match_g & match_r
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = connectivity | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(bgr, mask, (sx, sy), 255, loDiff=(0, 0, 0), upDiff=(0, 0, 0), flags=flags)
    full_mask = mask[1:-1, 1:-1]
    full_mask = (full_mask > 0) & matches
    return full_mask.astype(np.uint8) * 255


def extract_pores(
    image: np.ndarray,
    method: str = "auto",
    params: Optional[SegmentationParams] = None,
) -> np.ndarray:
    """孔洞自动/半自动提取.
    method: "auto" | "otsu" | "adaptive" | "color" | "manual"
    params: 当 method="color" 或 "manual" 时使用
    """
    if method == "color":
        if params is None:
            raise ValueError("color 方法需要 SegmentationParams")
        return segment_by_color(image, params)
    if method == "manual":
        if params is None or not hasattr(params, "_seed"):
            raise ValueError("manual 方法需要带 _seed 的 SegmentationParams")
        return flood_fill_region(image, params._seed, params)

    gray = to_grayscale(image)
    if method == "otsu" or method == "auto":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5,
        )
    raise ValueError(f"未知 method: {method}")


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """反选掩码."""
    return cv2.bitwise_not(mask)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """去除面积小于 min_area 的连通域(去噪)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def keep_largest_components(mask: np.ndarray, top_k: int = 1) -> np.ndarray:
    """仅保留面积最大的 top_k 个连通域."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    indices = np.argsort(areas)[::-1] + 1
    keep = set(int(i) for i in indices[:top_k])
    out = np.zeros_like(mask)
    for i in keep:
        out[labels == i] = 255
    return out


__all__ = [
    "SegmentationParams",
    "segment_by_color", "flood_fill_region", "extract_pores",
    "invert_mask", "remove_small_components", "keep_largest_components",
]
