"""合成粒度测试图生成器.

生成已知数量/位置/大小/形状的颗粒图像 + ground truth(实例 mask),
用于客观测试粒度识别准确率 (composite = 0.3*IoU + 0.7*F1).

颗粒形状模拟真实岩石:多边形(默认) 或 椭圆
颗粒之间允许接触(粘连),模拟花岗岩/砂岩的颗粒支撑结构.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class SyntheticGrain:
    """合成颗粒定义.

    Attributes:
        cx: 中心 x
        cy: 中心 y
        radius_px: 平均半径(像素)
        shape: "polygon"(默认)/"ellipse"/"round_rect"
        sides: 多边形边数 (5~12)
        rotation_deg: 旋转角度
        brightness: 颗粒亮度 (0~255)
    """
    cx: int
    cy: int
    radius_px: int
    shape: str = "polygon"
    sides: int = 6
    rotation_deg: float = 0.0
    brightness: int = 200


def _draw_grain_shape(
    image: np.ndarray,
    grain: SyntheticGrain,
    color: Tuple[int, int, int],
) -> None:
    """在图像上画一个颗粒."""
    cx, cy = grain.cx, grain.cy
    r = grain.radius_px
    if grain.shape == "ellipse":
        # 椭圆:在多边形基础上做轻微变形
        axes = (int(r * 1.0), int(r * 0.8))
        cv2.ellipse(image, (cx, cy), axes, grain.rotation_deg, 0, 360, color, -1, cv2.LINE_AA)
        return
    if grain.shape == "round_rect":
        # 圆角矩形
        rect = (cx - r, cy - r, 2 * r, 2 * r)
        box = cv2.RoundedRect
        # 用 ellipse 近似
        axes = (int(r * 0.95), int(r * 0.95))
        cv2.ellipse(image, (cx, cy), axes, grain.rotation_deg, 0, 360, color, -1, cv2.LINE_AA)
        return
    # polygon: 生成多边形顶点
    n_pts = grain.sides
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + np.deg2rad(grain.rotation_deg)
    # 每个顶点添加随机扰动 (让形状不规则)
    rng = np.random.default_rng(hash((cx, cy)) & 0xFFFF)
    pts = []
    for ang in angles:
        rj = r * (0.75 + 0.5 * rng.random())
        pts.append((cx + rj * np.cos(ang), cy + rj * np.sin(ang)))
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(image, [pts], color, lineType=cv2.LINE_AA)


def generate_synthetic_grain_rock(
    width: int = 800,
    height: int = 600,
    grains: List[SyntheticGrain] = None,
    background_color: Tuple[int, int, int] = (60, 55, 50),
    grain_color_range: Tuple[int, int] = (180, 230),
    noise_level: int = 8,
    output_path: str = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成颗粒图 + ground truth 实例 mask.

    Returns:
        image: BGR 图像
        instance_mask: 实例分割 mask (0=背景, 1, 2, 3... = 颗粒 ID)
    """
    rng = np.random.default_rng(seed)
    # 背景:深色基质 (模拟黑云母/绿泥石等)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = np.array(background_color, dtype=np.float32)
    for c in range(3):
        plane = base_color[c] + rng.normal(0, 15, (height, width)).astype(np.float32)
        plane = cv2.GaussianBlur(plane, (15, 15), 0)
        image[..., c] = np.clip(plane, 0, 255).astype(np.uint8)
    # 噪声
    noise = rng.normal(0, noise_level, (height, width, 3)).astype(np.float32)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    instance_mask = np.zeros((height, width), dtype=np.int32)
    if grains is None:
        grains = []
    # 绘制颗粒
    for idx, g in enumerate(grains):
        # 随机亮度(让颗粒有差异)
        b = int(rng.integers(grain_color_range[0], grain_color_range[1] + 1))
        # 三通道相近 (浅色矿物)
        b = max(grain_color_range[0], min(grain_color_range[1], g.brightness if g.brightness else b))
        color = (b, b, b)
        # 在图像上画颗粒
        _draw_grain_shape(image, g, color)
        # 在 mask 上画对应 ID (用 fillPoly)
        if g.shape == "polygon":
            n_pts = g.sides
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + np.deg2rad(g.rotation_deg)
            rng_local = np.random.default_rng(hash((g.cx, g.cy)) & 0xFFFF)
            pts = []
            for ang in angles:
                rj = g.radius_px * (0.75 + 0.5 * rng_local.random())
                pts.append((g.cx + rj * np.cos(ang), g.cy + rj * np.sin(ang)))
            pts_arr = np.array(pts, dtype=np.int32)
        elif g.shape == "ellipse":
            axes = (int(g.radius_px * 1.0), int(g.radius_px * 0.8))
            cv2.ellipse(instance_mask, (g.cx, g.cy), axes, g.rotation_deg, 0, 360, idx + 1, -1, cv2.LINE_AA)
            pts_arr = None
        else:
            axes = (int(g.radius_px * 0.95), int(g.radius_px * 0.95))
            cv2.ellipse(instance_mask, (g.cx, g.cy), axes, g.rotation_deg, 0, 360, idx + 1, -1, cv2.LINE_AA)
            pts_arr = None
        if pts_arr is not None:
            cv2.fillPoly(instance_mask, [pts_arr], idx + 1, lineType=cv2.LINE_AA)
    # 边缘轻微模糊
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, image)
    return image, instance_mask


def make_default_synthetic_grain(
    n_grains: int = 30,
    width: int = 1000,
    height: int = 800,
    seed: int = 42,
    min_distance_ratio: float = 1.4,
) -> Tuple[List[SyntheticGrain], np.ndarray, np.ndarray]:
    """生成默认合成测试集:N 个已知颗粒 (互不重叠).

    颗粒分布:在 (60, 60) 到 (width-60, height-60) 范围内泊松盘采样,
    保证相邻颗粒中心距离 ≥ min_distance_ratio * max(半径1, 半径2).

    Args:
        n_grains: 颗粒数量
        width, height: 画布大小
        seed: 随机种子
        min_distance_ratio: 颗粒最小间距系数 (1.0=刚好不重叠, 越大越稀疏)

    Returns:
        grains, image, instance_mask
    """
    rng = np.random.default_rng(seed)
    grains: List[SyntheticGrain] = []
    placed: List[Tuple[int, int, int]] = []  # (cx, cy, r)
    max_attempts = 200
    while len(grains) < n_grains:
        r = int(rng.integers(25, 50))
        placed_ok = False
        for _ in range(max_attempts):
            cx = int(rng.integers(80, width - 80))
            cy = int(rng.integers(80, height - 80))
            min_d = min_distance_ratio * r
            ok = True
            for (px, py, pr) in placed:
                if np.hypot(cx - px, cy - py) < min_d + pr * 0.7:
                    ok = False
                    break
            if ok:
                sides = int(rng.integers(5, 10))
                rot = float(rng.uniform(0, 360))
                b = int(rng.integers(180, 230))
                grains.append(SyntheticGrain(
                    cx=cx, cy=cy, radius_px=r,
                    shape="polygon", sides=sides,
                    rotation_deg=rot, brightness=b,
                ))
                placed.append((cx, cy, r))
                placed_ok = True
                break
        if not placed_ok:
            # 找不到位置,放弃
            break
    image, instance_mask = generate_synthetic_grain_rock(
        width=width, height=height, grains=grains, seed=seed,
    )
    return grains, image, instance_mask


def make_granite_synthetic_grain(
    n_grains: int = 25,
    width: int = 858,
    height: int = 570,
    seed: int = 42,
) -> Tuple[List[SyntheticGrain], np.ndarray, np.ndarray]:
    """生成花岗岩风格合成图 (用于真实图测试).

    花岗岩特征:
    - 深色基质背景 (黑云母、角闪石等)
    - 多个浅色矿物晶体 (石英、长石) 大小不一
    - 颗粒边界有深色矿物勾勒
    - 颗粒分布不均匀(聚集)

    Returns:
        grains: 颗粒定义
        image: BGR 合成图
        instance_mask: 实例 mask
    """
    rng = np.random.default_rng(seed)
    grains: List[SyntheticGrain] = []
    placed: List[Tuple[int, int, int]] = []
    # 浅色矿物: 大小不一, 部分聚集
    n_big = n_grains // 3  # 大颗粒 (10-15mm)
    n_mid = n_grains // 3  # 中颗粒 (5-10mm)
    n_small = n_grains - n_big - n_mid  # 小颗粒 (2-5mm)
    sizes = (
        [(rng.integers(60, 100)) for _ in range(n_big)] +
        [(rng.integers(35, 60)) for _ in range(n_mid)] +
        [(rng.integers(15, 35)) for _ in range(n_small)]
    )
    rng.shuffle(sizes)
    for r in sizes:
        # 聚集:在已有颗粒附近
        if placed and rng.random() < 0.4:
            base = placed[rng.integers(0, len(placed))]
            cx = int(np.clip(base[0] + rng.normal(0, 80), 80, width - 80))
            cy = int(np.clip(base[1] + rng.normal(0, 80), 80, height - 80))
        else:
            cx = int(rng.integers(80, width - 80))
            cy = int(rng.integers(80, height - 80))
        # 允许部分重叠 (花岗岩典型)
        placed_ok = False
        for _ in range(100):
            ok = True
            for (px, py, pr) in placed:
                dist = np.hypot(cx - px, cy - py)
                if dist < (r + pr) * 0.85:  # 15% 重叠
                    ok = False
                    break
            if ok or len(placed) == 0:
                placed_ok = True
                break
            cx = int(rng.integers(80, width - 80))
            cy = int(rng.integers(80, height - 80))
        if not placed_ok:
            continue
        sides = int(rng.integers(5, 9))
        rot = float(rng.uniform(0, 360))
        b = int(rng.integers(190, 230))
        grains.append(SyntheticGrain(
            cx=cx, cy=cy, radius_px=int(r),
            shape="polygon", sides=sides,
            rotation_deg=rot, brightness=b,
        ))
        placed.append((cx, cy, int(r)))
    image, instance_mask = generate_synthetic_grain_rock(
        width=width, height=height, grains=grains, seed=seed,
    )
    return grains, image, instance_mask


def make_overlapping_synthetic_grain(
    n_pairs: int = 8,
    width: int = 800,
    height: int = 600,
    seed: int = 42,
    contact_ratio: float = 0.9,
) -> Tuple[List[SyntheticGrain], np.ndarray, np.ndarray]:
    """生成粘连颗粒测试集:成对颗粒部分接触.

    用于测试分水岭分割粘连颗粒的能力.
    contact_ratio: 颗粒中心距离 / 半径的比值 (1.0=刚好接触,<1=重叠)

    Args:
        n_pairs: 颗粒对数
        width, height: 画布
        seed: 随机种子
        contact_ratio: 颗粒间距比 (默认 0.9 = 略微重叠,真实岩石典型)
    """
    rng = np.random.default_rng(seed)
    grains: List[SyntheticGrain] = []
    for _ in range(n_pairs):
        cx = int(rng.integers(120, width - 120))
        cy = int(rng.integers(120, height - 120))
        r = int(rng.integers(30, 50))
        offset = int(r * contact_ratio)
        # 两个略重叠的颗粒 (左侧)
        grains.append(SyntheticGrain(
            cx=cx - offset, cy=cy,
            radius_px=r, shape="ellipse", brightness=210,
        ))
        grains.append(SyntheticGrain(
            cx=cx + offset, cy=cy,
            radius_px=r, shape="ellipse", brightness=210,
        ))
    image, instance_mask = generate_synthetic_grain_rock(
        width=width, height=height, grains=grains, seed=seed,
    )
    return grains, image, instance_mask


if __name__ == "__main__":
    # 测试默认合成集
    grains, image, instance_mask = make_default_synthetic_grain()
    print(f"合成颗粒数: {len(grains)}")
    print(f"图像大小: {image.shape}")
    print(f"实例 mask 类别: {len(np.unique(instance_mask)) - 1} (不含背景)")
    cv2.imwrite("/tmp/synthetic_grain.png", image)
    cv2.imwrite("/tmp/synthetic_grain_gt.png", (instance_mask * 10).astype(np.uint8))
    # 测试粘连颗粒集
    grains2, image2, instance_mask2 = make_overlapping_synthetic_grain()
    print(f"\n粘连颗粒数: {len(grains2)}")
    cv2.imwrite("/tmp/synthetic_grain_overlap.png", image2)
    cv2.imwrite("/tmp/synthetic_grain_overlap_gt.png", (instance_mask2 * 10).astype(np.uint8))
    print("Saved to /tmp/synthetic_grain*.png")