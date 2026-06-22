"""生成多张不同难度的合成测试图,验证算法稳健性."""

import sys
sys.path.insert(0, '/home/ljh2923/opencode-project/new-koushui')

import cv2
import numpy as np

from rockpore.core import (
    SyntheticPore, generate_synthetic_rock, scale_from_dpi, evaluate_accuracy,
)


def make_easy_set():
    """简单集:浅色岩石+大孔洞+低噪声."""
    pores = [
        SyntheticPore(center=(100, 100), radius=40, shape="circle"),
        SyntheticPore(center=(300, 200), radius=35, shape="ellipse"),
        SyntheticPore(center=(500, 100), radius=45, shape="irregular", irregular_size=5),
        SyntheticPore(center=(150, 300), radius=20, shape="circle"),
        SyntheticPore(center=(400, 350), radius=25, shape="ellipse"),
        SyntheticPore(center=(550, 300), radius=18, shape="circle"),
    ]
    img, gt = generate_synthetic_rock(600, 400, pores, background_color=(200, 190, 170), noise_level=4)
    cv2.imwrite("/home/ljh2923/opencode-project/new-koushui/data/ground_truth/easy.png", img)
    cv2.imwrite("/home/ljh2923/opencode-project/new-koushui/data/ground_truth/easy_gt.png", gt)
    return img, gt


def make_hard_set():
    """困难集:暗色岩石+高对比纹理+大噪声+密集小孔洞."""
    pores = []
    rng = np.random.default_rng(7)
    # 10 个中等孔洞
    for _ in range(10):
        cx = int(rng.integers(50, 550))
        cy = int(rng.integers(50, 350))
        r = int(rng.integers(8, 30))
        pores.append(SyntheticPore(center=(cx, cy), radius=r, shape="circle"))
    # 15 个小孔洞
    for _ in range(15):
        cx = int(rng.integers(20, 580))
        cy = int(rng.integers(20, 380))
        r = int(rng.integers(3, 8))
        pores.append(SyntheticPore(center=(cx, cy), radius=r, shape="circle"))
    img, gt = generate_synthetic_rock(600, 400, pores, background_color=(120, 110, 95), noise_level=20)
    cv2.imwrite("/home/ljh2923/opencode-project/new-koushui/data/ground_truth/hard.png", img)
    cv2.imwrite("/home/ljh2923/opencode-project/new-koushui/data/ground_truth/hard_gt.png", gt)
    return img, gt


def make_dense_set():
    """密集集:大量重叠/相邻孔洞."""
    pores = [
        SyntheticPore(center=(100, 100), radius=20, shape="circle"),
        SyntheticPore(center=(140, 100), radius=18, shape="circle"),  # 相邻
        SyntheticPore(center=(180, 100), radius=22, shape="circle"),  # 相邻
        SyntheticPore(center=(300, 200), radius=30, shape="ellipse"),
        SyntheticPore(center=(320, 230), radius=25, shape="circle"),  # 部分重叠
        SyntheticPore(center=(500, 300), radius=15, shape="irregular", irregular_size=3),
        SyntheticPore(center=(540, 320), radius=18, shape="irregular", irregular_size=4),
        SyntheticPore(center=(200, 300), radius=12, shape="circle"),
        SyntheticPore(center=(400, 100), radius=14, shape="circle"),
    ]
    img, gt = generate_synthetic_rock(600, 400, pores, background_color=(170, 155, 135), noise_level=6)
    cv2.imwrite("/home/ljh2923/opencode-project/new-koushui/data/ground_truth/dense.png", img)
    cv2.imwrite("/home/ljh2923/opencode-project/new-koushui/data/ground_truth/dense_gt.png", gt)
    return img, gt


def main():
    sets = {
        "easy": make_easy_set(),
        "hard": make_hard_set(),
        "dense": make_dense_set(),
    }
    scale = scale_from_dpi(96)
    for name, (img, gt) in sets.items():
        report = evaluate_accuracy(img, gt, scale)
        print(f"\n=== {name.upper()} ===")
        print(f" Pixel IoU: {report.pixel_iou}")
        print(f" Detection F1: {report.detection_f1} (P={report.detection_precision}, R={report.detection_recall})")
        print(f" Diameter MAE: {report.diameter_mae} mm")
        print(f" GT {report.pore_count_gt} vs Pred {report.pore_count_pred} (匹配 {report.matched_pores})")
        print(f" Composite: {report.composite_score}  → {'✓ PASS' if report.passes_target(0.80) else '✗ FAIL'}")


if __name__ == "__main__":
    main()
