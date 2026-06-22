"""数学形态学与二次编辑模块.

实现 PDF 第 6-7 步中描述的区域去噪、孔洞填充、
区域膨胀/腐蚀(数学形态)等核心功能.
"""

from __future__ import annotations

import cv2
import numpy as np


def dilate_region(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """区域膨胀."""
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=iterations)


def erode_region(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """区域腐蚀."""
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel, iterations=iterations)


def morphological_open(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """开运算(先腐蚀后膨胀) - 去除小噪点."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def morphological_close(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """闭运算(先膨胀后腐蚀) - 填补小孔."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def remove_noise(
    mask: np.ndarray,
    min_area: int = 10,
    max_area: int = 0,
) -> np.ndarray:
    """区域去噪.
    min_area: 小于该面积的区域被去除
    max_area: 大于该面积(>0)的区域被去除
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    out = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        out[labels == i] = 255
    return out


def fill_holes(mask: np.ndarray, min_hole_area: int = 0) -> np.ndarray:
    """填充掩码内的孔洞.
    min_hole_area: 小于该面积的孔洞不填充(避免误填小特征)
    """
    h, w = mask.shape[:2]
    inv = cv2.bitwise_not(mask)
    flood = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    if min_hole_area > 0:
        holes_clean = np.zeros_like(holes)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_hole_area:
                holes_clean[labels == i] = 255
        holes = holes_clean
    return cv2.bitwise_or(mask, holes)


def watershed_split(mask: np.ndarray, min_distance: int = 5) -> np.ndarray:
    """分水岭分割(分离粘连孔洞).
    适用于多个孔洞粘连成一个连通域的情况.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return mask
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[mask == 0] = 0
    image_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(image_bgr, markers)
    out = np.zeros_like(mask)
    for label_id in range(2, n + 1):
        out[markers == label_id] = 255
    return out


__all__ = [
    "dilate_region", "erode_region",
    "morphological_open", "morphological_close",
    "remove_noise", "fill_holes", "watershed_split",
]
