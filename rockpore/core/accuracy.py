"""准确率评估模块.

使用合成岩心图作为 ground truth,评估孔洞识别算法的:
1. 像素级 IoU (Intersection over Union)
2. 检测级 Precision / Recall / F1
3. 直径估计 MAE (平均绝对误差)

目标: 综合准确率 ≥ 80%.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .analysis import analyze_pores, Pore, PoreAnalysisResult
from .calibration import Scale
from .segmentation import extract_pores
from .morphology import (
    remove_noise, fill_holes, morphological_close, watershed_split,
    morphological_open,
)
from .preprocessing import to_grayscale, auto_levels, apply_pore_enhancement


@dataclass
class PoreMatch:
    """孔洞匹配结果."""
    gt_id: int
    pred_id: int
    iou: float


@dataclass
class AccuracyReport:
    """准确率评估报告."""
    pixel_iou: float
    pixel_precision: float
    pixel_recall: float
    detection_precision: float
    detection_recall: float
    detection_f1: float
    diameter_mae: float
    diameter_relative_mae: float
    matched_pores: int
    missed_pores: int
    false_positives: int
    iou_threshold: float = 0.3
    pore_count_gt: int = 0
    pore_count_pred: int = 0
    pore_count_gt_report: int = 0
    pore_count_pred_report: int = 0
    composite_score: float = 0.0
    matches: List[Dict] = field(default_factory=list)

    def passes_target(self, target: float = 0.80) -> bool:
        """综合判断是否达到目标.使用 0.5*IoU + 0.5*F1."""
        return self.composite_score >= target

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["passes_target_0.80"] = self.passes_target(0.80)
        return d


def _compute_pred_individual_masks(mask: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """从预测掩码提取每个孔洞的独立掩码."""
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    result = []
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8) * 255
        if comp.sum() < 3:
            continue
        result.append((i, comp))
    return result


def _compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """真实 IoU(像素交集 / 像素并集)."""
    m1 = (mask1 > 0)
    m2 = (mask2 > 0)
    inter = int((m1 & m2).sum())
    union = int((m1 | m2).sum())
    if union == 0:
        return 0.0
    return inter / union


def match_pores(
    pred_masks: List[Tuple[int, np.ndarray]],
    gt_masks: List[Tuple[int, np.ndarray]],
    iou_threshold: float = 0.3,
) -> Tuple[List[PoreMatch], List[int], List[int]]:
    """真实 IoU 匹配预测与 ground truth 孔洞."""
    # 构建 IoU 矩阵
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, (_, pm) in enumerate(pred_masks):
        for j, (_, gm) in enumerate(gt_masks):
            iou_matrix[i, j] = _compute_iou(pm, gm)
    matches: List[PoreMatch] = []
    used_pred: set = set()
    used_gt: set = set()
    # 贪心匹配(从大到小)
    pairs = []
    for i in range(len(pred_masks)):
        for j in range(len(gt_masks)):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(key=lambda x: -x[0])
    for iou_val, i, j in pairs:
        if i in used_pred or j in used_gt:
            continue
        matches.append(PoreMatch(
            gt_id=gt_masks[j][0], pred_id=pred_masks[i][0], iou=float(iou_val),
        ))
        used_pred.add(i)
        used_gt.add(j)
    missed = [gt_masks[j][0] for j in range(len(gt_masks)) if j not in used_gt]
    fps = [pred_masks[i][0] for i in range(len(pred_masks)) if i not in used_pred]
    return matches, missed, fps


def compute_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float, float]:
    """像素级 Precision/Recall/IoU."""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    union = tp + fp + fn
    iou = tp / max(1, union)
    return precision, recall, iou


def detect_pores_robust(
    image: np.ndarray,
    scale: Scale,
    min_area_px: int = 4,
    min_diameter_real: float = 0.5,
    use_enhancement: bool = False,
) -> Tuple[np.ndarray, PoreAnalysisResult]:
    """鲁棒的孔洞检测流水线.
    默认不使用增强(避免过度处理),在需要时打开.

    流水线:
    1. 灰度化
    2. OTSU 逆阈值(暗色=孔洞)
    3. 闭运算:连接相近断裂
    4. 自适应判断是否需要分水岭
    5. 过滤过小/过大区域(基于实际直径阈值)
    """
    if use_enhancement:
        enhanced = apply_pore_enhancement(image)
    else:
        enhanced = image
    # 转灰度
    if enhanced.ndim == 3:
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        gray = enhanced
    # OTSU 逆阈值(暗色为前景=孔洞)
    otsu_t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 自适应降级:如果 OTSU 前景 > 50% 或 < 0.5%,尝试 Triangle
    fg_ratio = (mask > 0).mean()
    if fg_ratio > 0.50 or fg_ratio < 0.005:
        tri_t, mask_tri = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        tri_ratio = (mask_tri > 0).mean()
        if 0.005 < tri_ratio < 0.50:
            mask = mask_tri
            fg_ratio = tri_ratio
    # 闭运算:连接相近断裂
    mask = morphological_close(mask, kernel_size=3)
    # 开运算:去小毛刺和孤立纹理噪点
    mask = morphological_open(mask, kernel_size=2)
    # 统计处理后的情况
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # 仅在"前景占比适中"且"有过大区域"时分水岭
    if 0.02 < fg_ratio < 0.30 and n > 1:
        total_px = mask.shape[0] * mask.shape[1]
        max_area_ratio = max(stats[1:, cv2.CC_STAT_AREA]) / total_px if n > 1 else 0
        large_count = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] / total_px > 0.10)
        if max_area_ratio > 0.15 and large_count < 3:
            mask = watershed_split(mask, min_distance=5)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # 基于"实际直径"过滤(避免像素阈值不匹配 DPI)
    if n > 1:
        total_px = mask.shape[0] * mask.shape[1]
        # 计算对应 min_diameter_real 的最小像素面积(scale.mm_per_pixel 为 0 时降级)
        if scale.mm_per_pixel > 0:
            min_area_for_diameter = (min_diameter_real / 2.0) ** 2 * np.pi / (scale.mm_per_pixel ** 2)
        else:
            min_area_for_diameter = float(min_area_px)
        out = np.zeros_like(mask)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area_px:
                continue
            if area < min_area_for_diameter:
                continue
            if area / total_px > 0.60:
                continue
            out[labels == i] = 255
        mask = out
    result = analyze_pores(mask, scale)
    return mask, result


def evaluate_accuracy(
    image: np.ndarray,
    gt_mask: np.ndarray,
    scale: Scale,
    min_diameter_real: float = 2.0,
    use_enhancement: bool = False,
) -> AccuracyReport:
    """端到端准确率评估."""
    pred_mask, pred_result = detect_pores_robust(
        image, scale, min_diameter_real=1.0, use_enhancement=use_enhancement,
    )
    # 像素级
    pix_p, pix_r, pix_iou = compute_pixel_metrics(pred_mask, gt_mask)
    # 提取独立 GT 孔洞
    gt_individual = []
    n_gt, labels_gt = cv2.connectedComponents(gt_mask, connectivity=8)
    for i in range(1, n_gt):
        comp = (labels_gt == i).astype(np.uint8) * 255
        if comp.sum() < 3:
            continue
        gt_individual.append((i, comp))
    # 提取独立预测孔洞
    pred_individual = _compute_pred_individual_masks(pred_mask)
    # 匹配
    matches, missed, fps = match_pores(pred_individual, gt_individual, iou_threshold=0.3)
    tp = len(matches)
    precision = tp / max(1, tp + len(fps))
    recall = tp / max(1, tp + len(missed))
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    # 直径 MAE
    diameter_errors = []
    rel_errors = []
    for m in matches:
        gt_d = 0.0
        for gt_id, gt_m in gt_individual:
            if gt_id == m.gt_id:
                area_px = int((gt_m > 0).sum())
                gt_d = 2 * np.sqrt(scale.area_pixels_to_real(area_px) / np.pi)
                break
        pred_pore = next((p for p in pred_result.pores if p.id == m.pred_id), None)
        if pred_pore:
            err = abs(pred_pore.diameter_real - gt_d)
            diameter_errors.append(err)
            if gt_d > 0:
                rel_errors.append(err / gt_d)
    mae = float(np.mean(diameter_errors)) if diameter_errors else 0.0
    rel_mae = float(np.mean(rel_errors)) if rel_errors else 0.0
    composite = 0.5 * pix_iou + 0.5 * f1
    # 报告级孔洞数(直径≥2mm)
    gt_report = 0
    for _, gm in gt_individual:
        a = int((gm > 0).sum())
        d = 2 * np.sqrt(scale.area_pixels_to_real(a) / np.pi)
        if d >= min_diameter_real:
            gt_report += 1
    return AccuracyReport(
        pixel_iou=round(pix_iou, 4),
        pixel_precision=round(pix_p, 4),
        pixel_recall=round(pix_r, 4),
        detection_precision=round(precision, 4),
        detection_recall=round(recall, 4),
        detection_f1=round(f1, 4),
        diameter_mae=round(mae, 4),
        diameter_relative_mae=round(rel_mae, 4),
        matched_pores=tp,
        missed_pores=len(missed),
        false_positives=len(fps),
        iou_threshold=0.3,
        pore_count_gt=len(gt_individual),
        pore_count_pred=len(pred_result.pores),
        pore_count_gt_report=gt_report,
        pore_count_pred_report=pred_result.pore_count_report,
        composite_score=round(composite, 4),
        matches=[{"gt_id": m.gt_id, "pred_id": m.pred_id, "iou": m.iou} for m in matches],
    )


__all__ = [
    "AccuracyReport", "PoreMatch",
    "compute_pixel_metrics", "match_pores",
    "detect_pores_robust", "evaluate_accuracy",
]
