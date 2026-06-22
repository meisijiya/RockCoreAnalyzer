"""合成裂缝测试图生成器.

生成已知位置/方向/长度的合成裂缝图像 + ground truth(骨架图),
用于客观测试裂缝识别准确率.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class SyntheticFracture:
    """合成裂缝定义."""
    x1: int          # 起点 x
    y1: int          # 起点 y
    x2: int          # 终点 x
    y2: int          # 终点 y
    width_px: int    # 裂缝宽度(像素)
    curvature: float = 0.0  # 弯曲程度(0=直线,越大越弯)
    n_segments: int = 1     # 折线分段数(用于构造折线/分支)


def _draw_fracture_on_mask(
    mask: np.ndarray,
    fx1: int, fy1: int, fx2: int, fy2: int,
    width_px: int,
    curvature: float = 0.0,
    rng: np.random.Generator = None,
) -> None:
    """在 mask 上画一条裂缝."""
    if rng is None:
        rng = np.random.default_rng()
    if curvature <= 0 or width_px <= 1:
        cv2.line(mask, (fx1, fy1), (fx2, fy2), 255, width_px, cv2.LINE_AA)
        return
    # 生成带弯曲的折线
    n_pts = max(5, int(curvature * 20))
    xs = np.linspace(fx1, fx2, n_pts)
    ys = np.linspace(fy1, fy2, n_pts)
    # 垂直方向扰动
    dx = fx2 - fx1
    dy = fy2 - fy1
    length = math.sqrt(dx * dx + dy * dy) + 1e-6
    nx = -dy / length
    ny = dx / length
    perturb = rng.normal(0, curvature * length * 0.05, n_pts)
    # 累积扰动(更平滑)
    perturb = np.cumsum(perturb) / np.maximum(1, np.arange(1, n_pts + 1).astype(float))
    pts = np.column_stack([
        (xs + nx * perturb).astype(np.int32),
        (ys + ny * perturb).astype(np.int32),
    ])
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], False, 255, width_px, cv2.LINE_AA)


def generate_synthetic_fracture_rock(
    width: int = 800,
    height: int = 600,
    fractures: List[SyntheticFracture] = None,
    background_color: Tuple[int, int, int] = (180, 160, 130),
    fracture_color: Tuple[int, int, int] = (15, 12, 10),
    noise_level: int = 3,
    output_path: str = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成裂缝图 + ground truth mask.

    Returns:
        (image, gt_mask)
        image: BGR 图像
        gt_mask: 二值裂缝 mask(0/255)
    """
    rng = np.random.default_rng(seed)
    # 背景(模拟砂岩/灰岩的浅色基质)— 用更强的低频纹理 + 较轻高频噪声
    image = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = np.array(background_color, dtype=np.float32)
    for c in range(3):
        plane = base_color[c] + rng.normal(0, 12, (height, width)).astype(np.float32)
        plane = cv2.GaussianBlur(plane, (31, 31), 0)  # 更强的低频平滑
        image[..., c] = np.clip(plane, 0, 255).astype(np.uint8)
    # 模拟岩心纹理(降低噪声水平,避免假阳性)
    texture = rng.normal(0, noise_level, (height, width, 3)).astype(np.float32)
    image = np.clip(image.astype(np.float32) + texture, 0, 255).astype(np.uint8)
    # 一些随机划痕(数量减半,颜色更亮使其接近背景,降低假阳性)
    for _ in range(15):
        x1, y1 = rng.integers(0, width), rng.integers(0, height)
        x2, y2 = rng.integers(0, width), rng.integers(0, height)
        cv2.line(image, (x1, y1), (x2, y2), (140, 130, 110), 1, cv2.LINE_AA)
    # 绘制裂缝
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    if fractures is None:
        fractures = []
    for fr in fractures:
        _draw_fracture_on_mask(
            gt_mask, fr.x1, fr.y1, fr.x2, fr.y2,
            width_px=fr.width_px, curvature=fr.curvature, rng=rng,
        )
        # 在原图上画同样的裂缝
        fx1, fy1 = fr.x1, fr.y1
        fx2, fy2 = fr.x2, fr.y2
        w = fr.width_px
        if fr.curvature <= 0:
            cv2.line(image, (fx1, fy1), (fx2, fy2), fracture_color, w, cv2.LINE_AA)
        else:
            n_pts = max(5, int(fr.curvature * 20))
            xs = np.linspace(fx1, fx2, n_pts)
            ys = np.linspace(fy1, fy2, n_pts)
            dx = fx2 - fx1
            dy = fy2 - fy1
            length = math.sqrt(dx * dx + dy * dy) + 1e-6
            nx = -dy / length
            ny = dx / length
            perturb = rng.normal(0, fr.curvature * length * 0.05, n_pts)
            perturb = np.cumsum(perturb) / np.maximum(1, np.arange(1, n_pts + 1).astype(float))
            pts = np.column_stack([
                (xs + nx * perturb).astype(np.int32),
                (ys + ny * perturb).astype(np.int32),
            ]).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], False, fracture_color, w, cv2.LINE_AA)
    # 边缘模糊(更真实)
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, image)
    return image, gt_mask


def make_default_synthetic_fracture() -> Tuple[List[SyntheticFracture], np.ndarray, np.ndarray]:
    """生成默认合成测试集:8条已知裂缝(不同长度/宽度/方向/弯曲,空间分布避免合并)."""
    fractures = [
        # 大缝 (宽 5~8px @96dpi ~ 1.3~2.1mm) — 分布在图像四个象限
        SyntheticFracture(x1=80, y1=80, x2=380, y2=110, width_px=6),    # 左上水平
        SyntheticFracture(x1=450, y1=180, x2=750, y2=320, width_px=5),  # 右上斜向
        SyntheticFracture(x1=180, y1=350, x2=240, y2=560, width_px=4),  # 左下近垂直
        # 中缝 (宽 2~3px ~ 0.5~0.8mm)
        SyntheticFracture(x1=520, y1=380, x2=750, y2=420, width_px=3, curvature=1.5),  # 右下微弯
        SyntheticFracture(x1=300, y1=60, x2=320, y2=250, width_px=2),   # 中上近垂直
        SyntheticFracture(x1=600, y1=70, x2=620, y2=240, width_px=2),   # 右侧近垂直
        # 小缝 (宽 1px ~ 0.26mm,接近报告阈值)
        SyntheticFracture(x1=60, y1=520, x2=160, y2=530, width_px=1),
        SyntheticFracture(x1=400, y1=510, x2=500, y2=520, width_px=1),
    ]
    image, gt_mask = generate_synthetic_fracture_rock(
        width=800, height=600, fractures=fractures, seed=42,
    )
    return fractures, image, gt_mask


if __name__ == "__main__":
    fractures, image, gt_mask = make_default_synthetic_fracture()
    print(f"合成裂缝数: {len(fractures)}")
    print(f"图像大小: {image.shape}")
    print(f"GT mask 像素数: {int((gt_mask > 0).sum())}")
    cv2.imwrite("/tmp/synthetic_fracture.png", image)
    cv2.imwrite("/tmp/synthetic_fracture_gt.png", gt_mask)
    print("Saved to /tmp/synthetic_fracture.png & /tmp/synthetic_fracture_gt.png")
