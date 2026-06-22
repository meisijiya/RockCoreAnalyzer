"""裂缝识别准确率评估模块.

使用合成裂缝测试图作为 ground truth,评估 HoughLinesP 检测的:
1. 像素级 IoU (Intersection over Union)
2. 检测级 Precision / Recall / F1 (线段距离 ≤ 阈值 视为匹配)
3. 长度估计 MAE

目标: 综合准确率 ≥ 80%.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .fracture import (
    FractureAnalysisResult, FractureParams, analyze_fractures,
    detect_fracture_mask, draw_fracture_annotations,
)
from .calibration import Scale


@dataclass
class FractureMatch:
    """单条裂缝匹配结果."""
    gt_id: int
    pred_id: int
    distance_px: float   # 中心点距离
    length_ratio: float  # 长度比 (pred/gt)


@dataclass
class FractureAccuracyReport:
    """裂缝识别准确率报告."""
    pixel_iou: float
    pixel_precision: float
    pixel_recall: float
    detection_precision: float
    detection_recall: float
    detection_f1: float
    length_mae: float
    length_relative_mae: float
    matched: int
    missed: int
    false_positives: int
    distance_threshold_px: float = 30.0
    fracture_count_gt: int = 0
    fracture_count_pred: int = 0
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


def _extract_gt_fractures(gt_mask: np.ndarray) -> List[Tuple[int, np.ndarray, Tuple[float, float]]]:
    """从 ground truth mask 提取每条裂缝的区域 + 中心点."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
    result = []
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8) * 255
        if int((comp > 0).sum()) < 3:
            continue
        centroid = (float(centroids[i][0]), float(centroids[i][1]))
        result.append((i, comp, centroid))
    return result


def match_fractures(
    pred_centroids: List[Tuple[int, Tuple[float, float]]],
    gt_centroids: List[Tuple[int, Tuple[float, float]]],
    distance_threshold_px: float = 30.0,
) -> Tuple[List[FractureMatch], List[int], List[int]]:
    """基于中心点距离的贪心匹配."""
    pairs = []
    for i, (_, pc) in enumerate(pred_centroids):
        for j, (_, gc) in enumerate(gt_centroids):
            d = float(np.hypot(pc[0] - gc[0], pc[1] - gc[1]))
            if d <= distance_threshold_px:
                pairs.append((d, i, j))
    pairs.sort(key=lambda x: x[0])
    matches: List[FractureMatch] = []
    used_pred: set = set()
    used_gt: set = set()
    for d, i, j in pairs:
        if i in used_pred or j in used_gt:
            continue
        matches.append(FractureMatch(
            gt_id=gt_centroids[j][0],
            pred_id=pred_centroids[i][0],
            distance_px=d,
            length_ratio=0.0,  # 由上层填充
        ))
        used_pred.add(i)
        used_gt.add(j)
    missed = [gt_centroids[j][0] for j in range(len(gt_centroids)) if j not in used_gt]
    fps = [pred_centroids[i][0] for i in range(len(pred_centroids)) if i not in used_pred]
    return matches, missed, fps


def detect_fractures_robust(
    image: np.ndarray,
    scale: Scale,
    params: FractureParams = None,
) -> Tuple[np.ndarray, np.ndarray, FractureAnalysisResult]:
    """鲁棒的裂缝检测流水线.

    1. CLAHE 增强 (由 detect_fracture_mask 内部完成)
    2. Canny 边缘
    3. HoughLinesP 线段检测
    4. 线段 → 区域(dilation)
    5. 形态学闭运算连接相近线段
    6. 开运算去除小毛刺(提升 IoU)
    7. 去除过短区域
    """
    mask, edges, _ = detect_fracture_mask(image, params)
    # 形态学闭运算连接断裂
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # 开运算去小毛刺(提升 IoU,减少假阳性像素)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    # 去除过小区域
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        cleaned = np.zeros_like(mask)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            length = float(np.sqrt(w * w + h * h))
            if length >= 15 and area >= 5:
                cleaned[labels == i] = 255
        mask = cleaned
    result = analyze_fractures(mask, scale)
    return mask, edges, result


def evaluate_fracture_accuracy(
    image: np.ndarray,
    gt_mask: np.ndarray,
    scale: Scale,
    params: FractureParams = None,
    distance_threshold_px: float = 30.0,
) -> FractureAccuracyReport:
    """端到端裂缝识别准确率评估."""
    pred_mask, _, pred_result = detect_fractures_robust(image, scale, params)
    # 像素级
    pix_p, pix_r, pix_iou = compute_pixel_metrics(pred_mask, gt_mask)
    # GT 裂缝
    gt_list = _extract_gt_fractures(gt_mask)
    # 预测裂缝(从 result 中提取)
    pred_list = [(f.id, f.centroid) for f in pred_result.fractures]
    # 匹配(基于中心点距离)
    matches, missed, fps = match_fractures(
        [(pid, c) for pid, c in pred_list],
        [(gid, c) for gid, _, c in gt_list],
        distance_threshold_px=distance_threshold_px,
    )
    tp = len(matches)
    precision = tp / max(1, tp + len(fps))
    recall = tp / max(1, tp + len(missed))
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    # 长度 MAE
    gt_by_id = {gid: comp for gid, comp, _ in gt_list}
    gt_length_px = {}
    for gid, comp in gt_by_id.items():
        ys, xs = np.where(comp > 0)
        if len(xs) > 0:
            gt_length_px[gid] = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    length_errors: List[float] = []
    rel_errors: List[float] = []
    for m in matches:
        gt_len = gt_length_px.get(m.gt_id, 0)
        pred_pore = next((f for f in pred_result.fractures if f.id == m.pred_id), None)
        if pred_pore and gt_len > 0:
            err = abs(pred_pore.length_px - gt_len)
            length_errors.append(err)
            rel_errors.append(err / gt_len)
            m.length_ratio = pred_pore.length_px / gt_len
    mae = float(np.mean(length_errors)) if length_errors else 0.0
    rel_mae = float(np.mean(rel_errors)) if rel_errors else 0.0
    # 裂缝检测 composite_score: 0.3*IoU + 0.7*F1
    # 线段检测任务中 F1 比 IoU 更能反映算法本质(Pixel IoU 受线宽/膨胀影响大)
    composite = 0.3 * pix_iou + 0.7 * f1
    return FractureAccuracyReport(
        pixel_iou=round(pix_iou, 4),
        pixel_precision=round(pix_p, 4),
        pixel_recall=round(pix_r, 4),
        detection_precision=round(precision, 4),
        detection_recall=round(recall, 4),
        detection_f1=round(f1, 4),
        length_mae=round(mae, 4),
        length_relative_mae=round(rel_mae, 4),
        matched=tp,
        missed=len(missed),
        false_positives=len(fps),
        distance_threshold_px=distance_threshold_px,
        fracture_count_gt=len(gt_list),
        fracture_count_pred=len(pred_list),
        composite_score=round(composite, 4),
        matches=[
            {"gt_id": m.gt_id, "pred_id": m.pred_id,
             "distance_px": round(m.distance_px, 2),
             "length_ratio": round(m.length_ratio, 4)}
            for m in matches
        ],
    )


__all__ = [
    "FractureAccuracyReport", "FractureMatch",
    "compute_pixel_metrics", "match_fractures",
    "detect_fractures_robust", "evaluate_fracture_accuracy",
]
