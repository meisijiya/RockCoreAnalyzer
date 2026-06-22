"""合成测试图生成器.

为评估算法准确率,生成已知孔洞位置和大小的合成岩心图像,
作为 ground truth 用于客观测试识别准确率(目标≥80%).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class SyntheticPore:
    """合成孔洞定义."""
    center: Tuple[int, int]  # (x, y)
    radius: int  # 像素半径
    shape: str = "circle"  # circle/ellipse/irregular
    irregular_size: int = 0  # 不规则扰动幅度(0=规则)


def generate_synthetic_rock(
    width: int = 600,
    height: int = 400,
    pores: List[SyntheticPore] = None,
    background_color: Tuple[int, int, int] = (180, 160, 130),
    pore_color: Tuple[int, int, int] = (15, 12, 10),
    noise_level: int = 8,
    output_path: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成岩心图 + 对应 ground truth 掩码."""
    rng = np.random.default_rng(42)
    # 背景(模拟砂岩/灰岩的浅色基质)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = np.array(background_color, dtype=np.float32)
    for c in range(3):
        plane = base_color[c] + rng.normal(0, 8, (height, width)).astype(np.float32)
        plane = cv2.GaussianBlur(plane, (15, 15), 0)
        image[..., c] = np.clip(plane, 0, 255).astype(np.uint8)
    # 模拟岩心纹理(细噪声)
    texture = rng.normal(0, noise_level, (height, width, 3)).astype(np.float32)
    image = np.clip(image.astype(np.float32) + texture, 0, 255).astype(np.uint8)
    # 添加少量纹理(小划痕)
    for _ in range(20):
        x1, y1 = rng.integers(0, width), rng.integers(0, height)
        x2, y2 = rng.integers(0, width), rng.integers(0, height)
        cv2.line(image, (x1, y1), (x2, y2), (90, 80, 70), 1, cv2.LINE_AA)
    # 绘制孔洞
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    if pores is None:
        pores = []
    for p in pores:
        cx, cy = p.center
        if p.shape == "circle":
            cv2.circle(image, (cx, cy), p.radius, pore_color, -1, cv2.LINE_AA)
            cv2.circle(gt_mask, (cx, cy), p.radius, 255, -1)
        elif p.shape == "ellipse":
            axes = (p.radius, max(2, p.radius // 2))
            cv2.ellipse(image, (cx, cy), axes, 0, 0, 360, pore_color, -1, cv2.LINE_AA)
            cv2.ellipse(gt_mask, (cx, cy), axes, 0, 0, 360, 255, -1)
        elif p.shape == "irregular":
            # 多边形不规则孔洞
            n_points = rng.integers(6, 12)
            angles = np.linspace(0, 2 * np.pi, n_points)
            radii = p.radius + rng.integers(-p.irregular_size, p.irregular_size + 1, n_points)
            pts = np.stack([
                cx + radii * np.cos(angles),
                cy + radii * np.sin(angles),
            ], axis=1).astype(np.int32)
            cv2.fillPoly(image, [pts], pore_color, lineType=cv2.LINE_AA)
            cv2.fillPoly(gt_mask, [pts], 255)
    # 高斯模糊孔洞边缘(更真实)
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, image)
    return image, gt_mask


def make_default_synthetic() -> Tuple[List[SyntheticPore], np.ndarray, np.ndarray]:
    """生成默认合成测试集:15个已知孔洞,涵盖大中小不同尺寸."""
    pores = [
        # 大洞 (>10mm @96dpi ~ 38px)
        SyntheticPore(center=(80, 80), radius=45, shape="circle"),
        SyntheticPore(center=(250, 120), radius=50, shape="ellipse"),
        SyntheticPore(center=(500, 70), radius=42, shape="irregular", irregular_size=8),
        # 中洞 (5-10mm ~ 19-38px)
        SyntheticPore(center=(150, 250), radius=25, shape="circle"),
        SyntheticPore(center=(350, 200), radius=22, shape="ellipse"),
        SyntheticPore(center=(450, 320), radius=28, shape="irregular", irregular_size=4),
        SyntheticPore(center=(550, 220), radius=20, shape="circle"),
        # 小洞 (1-5mm ~ 4-19px)
        SyntheticPore(center=(100, 350), radius=8, shape="circle"),
        SyntheticPore(center=(180, 380), radius=10, shape="circle"),
        SyntheticPore(center=(280, 350), radius=6, shape="ellipse"),
        SyntheticPore(center=(400, 380), radius=9, shape="irregular", irregular_size=2),
        SyntheticPore(center=(520, 360), radius=7, shape="circle"),
        SyntheticPore(center=(50, 200), radius=12, shape="ellipse"),
        # 针孔 (<1mm ~ <4px,部分故意小到检测不到)
        SyntheticPore(center=(220, 80), radius=2, shape="circle"),
        SyntheticPore(center=(420, 130), radius=3, shape="circle"),
    ]
    img, gt = generate_synthetic_rock(
        width=600, height=400, pores=pores,
        output_path="/home/ljh2923/opencode-project/new-koushui/data/ground_truth/synthetic_01.png",
    )
    return pores, img, gt


if __name__ == "__main__":
    pores, img, gt = make_default_synthetic()
    cv2.imwrite("/home/ljh2923/opencode-project/new-koushui/data/ground_truth/synthetic_01.png", img)
    cv2.imwrite("/home/ljh2923/opencode-project/new-koushui/data/ground_truth/synthetic_01_gt.png", gt)
    print(f"生成 {len(pores)} 个合成孔洞")
    print(f"图片: {img.shape}, GT 孔洞像素: {int((gt == 255).sum())}")
