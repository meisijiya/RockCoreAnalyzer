"""粒度识别准确率评估模块.

使用合成颗粒测试图作为 ground truth,评估距离变换+分水岭检测的:
1. 像素级 IoU (Intersection over Union)
2. 检测级 Precision / Recall / F1 (基于质心距离匹配)
3. 颗粒计数偏差 (count_diff_ratio)
4. 粒径 MAE (相对误差)

目标: 综合准确率 (composite = 0.3*IoU + 0.7*F1) ≥ 80%.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from .grain import (
    Grain, GrainAnalysisResult, GrainParams, analyze_grains,
    detect_grain_mask, draw_grain_annotations,
)
from .calibration import Scale


@dataclass
class GrainMatch:
    """单颗粒匹配结果."""
    gt_id: int
    pred_id: int
    centroid_distance_px: float
    iou: float  # 颗粒 mask IoU
    diameter_ratio: float  # pred_diameter / gt_diameter


@dataclass
class GrainAccuracyReport:
    """粒度识别准确率报告."""
    pixel_iou: float
    pixel_precision: float
    pixel_recall: float
    detection_precision: float
    detection_recall: float
    detection_f1: float
    count_gt: int
    count_pred: int
    count_diff_ratio: float  # |pred - gt| / gt
    diameter_mae_ratio: float  # 平均直径相对误差
    diameter_relative_mae: float
    matched: int
    missed: int
    false_positives: int
    distance_threshold_px: float = 50.0
    iou_threshold: float = 0.3
    composite_score: float = 0.0
    matches: List[Dict] = field(default_factory=list)

    def passes_target(self, target: float = 0.80) -> bool:
        return self.composite_score >= target

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["passes_target_0.80"] = self.passes_target(0.80)
        return d


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


def _extract_gt_grains(gt_instance_mask: np.ndarray) -> List[Tuple[int, np.ndarray, Tuple[float, float], float]]:
    """从 ground truth 实例 mask 提取每个颗粒 (按 unique instance ID).

    Args:
        gt_instance_mask: 实例 mask (0=背景, 1, 2, 3... = 颗粒 ID)

    Returns:
        [(gt_id, binary_mask, centroid, diameter_px), ...]
    """
    result: List[Tuple[int, np.ndarray, Tuple[float, float], float]] = []
    unique_ids = [int(i) for i in np.unique(gt_instance_mask) if i > 0]
    for gid in unique_ids:
        comp = (gt_instance_mask == gid).astype(np.uint8)
        area = int(comp.sum())
        if area < 3:
            continue
        # 等效直径
        diameter = 2.0 * np.sqrt(area / np.pi)
        ys, xs = np.where(comp > 0)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        result.append((gid, comp, (cx, cy), float(diameter)))
    return result


def _extract_pred_grains(pred_mask: np.ndarray) -> List[Tuple[int, np.ndarray, Tuple[float, float], float]]:
    """从预测 mask 提取每个颗粒 (基于连通域).

    Returns:
        [(pred_id, binary_mask, centroid, diameter_px), ...]
    """
    # connectedComponentsWithStats 需要 CV_8U/CV_8S
    if pred_mask.dtype not in (np.uint8, np.int8):
        pred_mask = pred_mask.astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    result: List[Tuple[int, np.ndarray, Tuple[float, float], float]] = []
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        diameter = 2.0 * np.sqrt(area / np.pi)
        centroid = (float(centroids[i][0]), float(centroids[i][1]))
        result.append((i, comp, centroid, float(diameter)))
    return result


def _compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """两个二值 mask 的 IoU."""
    a = (mask1 > 0)
    b = (mask2 > 0)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / max(1, union)


def match_grains(
    pred_grains: List[Tuple[int, np.ndarray, Tuple[float, float], float]],
    gt_grains: List[Tuple[int, np.ndarray, Tuple[float, float], float]],
    distance_threshold_px: float = 50.0,
    iou_threshold: float = 0.3,
) -> Tuple[List[GrainMatch], List[int], List[int]]:
    """基于质心距离 + IoU 的贪心匹配.

    匹配规则:
    1. 距离 ≤ distance_threshold_px
    2. 选 IoU 最高的配对(一对多匹配)
    3. 一旦配对,两者都标记为已用

    Args:
        pred_grains: [(id, mask, centroid, diameter), ...]
        gt_grains: 同上

    Returns:
        matches: 匹配对列表
        missed: 未匹配的 GT ID
        fps: 未匹配的 pred ID
    """
    # 计算所有 (pred, gt) 候选对的 (distance, iou)
    pairs: List[Tuple[float, float, int, int]] = []
    for pi, (_, pm, pc, _) in enumerate(pred_grains):
        for gi, (_, gm, gc, _) in enumerate(gt_grains):
            d = float(np.hypot(pc[0] - gc[0], pc[1] - gc[1]))
            if d <= distance_threshold_px:
                iou = _compute_iou(pm, gm)
                # 排序键: (距离, -IoU) 优先近距离高 IoU
                pairs.append((d, -iou, pi, gi))
    pairs.sort(key=lambda x: (x[0], x[1]))
    matches: List[GrainMatch] = []
    used_pred: set = set()
    used_gt: set = set()
    for d, neg_iou, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        iou = -neg_iou
        # IoU 太低不算匹配(避免合并/重叠导致的假匹配)
        if iou < 0.05:  # 极宽松阈值,主要用于距离过滤
            continue
        # 直径比
        gt_diam = gt_grains[gi][3]
        pred_diam = pred_grains[pi][3]
        diam_ratio = pred_diam / max(gt_diam, 1e-6)
        matches.append(GrainMatch(
            gt_id=gt_grains[gi][0],
            pred_id=pred_grains[pi][0],
            centroid_distance_px=float(d),
            iou=float(iou),
            diameter_ratio=float(diam_ratio),
        ))
        used_pred.add(pi)
        used_gt.add(gi)
    missed = [gt_grains[gi][0] for gi in range(len(gt_grains)) if gi not in used_gt]
    fps = [pred_grains[pi][0] for pi in range(len(pred_grains)) if pi not in used_pred]
    return matches, missed, fps


def detect_grains_robust(
    image: np.ndarray,
    scale: Scale,
    params: GrainParams = None,
) -> Tuple[np.ndarray, GrainAnalysisResult]:
    """鲁棒的颗粒检测流水线.

    1. detect_grain_mask: OTSU 反相 + 形态学清理
    2. analyze_grains: 距离变换 + 分水岭 + 属性计算
    """
    bin_mask, _ = detect_grain_mask(image, params)
    result, _, _ = analyze_grains(image, scale, params, mask=bin_mask)
    return bin_mask, result


def _extract_pred_grains_from_markers(
    markers: np.ndarray,
    bin_mask: np.ndarray,
    min_area_px: int = 200,
) -> List[Tuple[int, np.ndarray, Tuple[float, float], float]]:
    """从分水岭 markers 提取每个颗粒 (而不是用连通域).

    关键改进:之前用 pred_mask 的连通域会把粘连的多个颗粒当作一个,
    现在用分水岭 markers 每个 ID 对应一个颗粒.

    Returns:
        [(pred_id, binary_mask, centroid, diameter_px), ...]
    """
    result: List[Tuple[int, np.ndarray, Tuple[float, float], float]] = []
    unique_ids = [int(i) for i in np.unique(markers) if i > 1]
    for pid in unique_ids:
        comp = (markers == pid).astype(np.uint8)
        area = int(comp.sum())
        if area < max(3, min_area_px):
            continue
        diameter = 2.0 * np.sqrt(area / np.pi)
        ys, xs = np.where(comp > 0)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        result.append((pid, comp, (cx, cy), float(diameter)))
    return result


def evaluate_grain_accuracy(
    image: np.ndarray,
    gt_instance_mask: np.ndarray,
    scale: Scale,
    params: GrainParams = None,
    distance_threshold_px: float = 50.0,
) -> GrainAccuracyReport:
    """端到端粒度识别准确率评估.

    Args:
        image: BGR 图像
        gt_instance_mask: ground truth 实例 mask (0=背景, 1, 2, 3... = 颗粒 ID)
        scale: 标尺对象
        params: 检测参数
        distance_threshold_px: 质心匹配距离阈值

    Returns:
        GrainAccuracyReport
    """
    # 检测:走完整流水线(得到分水岭 markers)
    bin_mask, pred_result = detect_grains_robust(image, scale, params)
    # 拿到 markers:重新调用内部逻辑
    from .grain import _watershed_split
    markers = _watershed_split(bin_mask, params) if params else _watershed_split(bin_mask, GrainParams())
    # 像素级(注意: pred_mask 是单一前景,gt 需要合并所有实例)
    gt_combined = (gt_instance_mask > 0).astype(np.uint8) * 255
    pix_p, pix_r, pix_iou = compute_pixel_metrics(bin_mask, gt_combined)
    # 颗粒级(从 markers 提取,不是从 bin_mask 提取!)
    gt_list = _extract_gt_grains(gt_instance_mask)
    pred_list = _extract_pred_grains_from_markers(
        markers, bin_mask,
        min_area_px=params.min_area_px if params else 200,
    )
    matches, missed, fps = match_grains(
        pred_list, gt_list,
        distance_threshold_px=distance_threshold_px,
    )
    tp = len(matches)
    precision = tp / max(1, tp + len(fps))
    recall = tp / max(1, tp + len(missed))
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    # 数量偏差
    count_gt = len(gt_list)
    count_pred = len(pred_list)
    count_diff_ratio = abs(count_pred - count_gt) / max(1, count_gt)
    # 直径 MAE (相对)
    diam_ratios = [m.diameter_ratio for m in matches]
    rel_errors = [abs(r - 1.0) for r in diam_ratios]
    diam_mae_ratio = float(np.mean(diam_ratios)) if diam_ratios else 0.0
    diam_rel_mae = float(np.mean(rel_errors)) if rel_errors else 0.0
    # 综合准确率: composite = 0.3*IoU + 0.7*F1
    # 颗粒检测任务中 F1 比 IoU 更重要(实例级匹配比像素级更能反映算法能力)
    composite = 0.3 * pix_iou + 0.7 * f1
    return GrainAccuracyReport(
        pixel_iou=round(float(pix_iou), 4),
        pixel_precision=round(float(pix_p), 4),
        pixel_recall=round(float(pix_r), 4),
        detection_precision=round(float(precision), 4),
        detection_recall=round(float(recall), 4),
        detection_f1=round(float(f1), 4),
        count_gt=count_gt,
        count_pred=count_pred,
        count_diff_ratio=round(float(count_diff_ratio), 4),
        diameter_mae_ratio=round(float(diam_mae_ratio), 4),
        diameter_relative_mae=round(float(diam_rel_mae), 4),
        matched=tp,
        missed=len(missed),
        false_positives=len(fps),
        distance_threshold_px=distance_threshold_px,
        composite_score=round(float(composite), 4),
        matches=[
            {"gt_id": m.gt_id, "pred_id": m.pred_id,
             "distance_px": round(m.centroid_distance_px, 2),
             "iou": round(m.iou, 4),
             "diameter_ratio": round(m.diameter_ratio, 4)}
            for m in matches
        ],
    )


__all__ = [
    "GrainMatch", "GrainAccuracyReport",
    "compute_pixel_metrics", "match_grains",
    "detect_grains_robust", "evaluate_grain_accuracy",
]