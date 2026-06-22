"""孔洞分析模块.

实现 PDF 1.2 节定义的孔洞分析计量标准:
- 单个孔洞等效面积圆直径 Dr = 2*sqrt(A/π)
- 平均孔洞等效面积圆直径 Dr = (ΣDi) / n
- 孔洞面孔率 = 孔洞总面积 / 选取区域面积
- 孔洞分类: 大洞/中洞/小洞/针孔

并产出 PDF 6.1 节中要求的孔洞分析参数表与分布曲线数据.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .calibration import Scale


# 孔洞大小分类阈值(单位:mm,来自 PDF 1.2 节)
PORE_LARGE = 10.0
PORE_MEDIUM = 5.0
PORE_SMALL = 1.0
PORE_PINHOLE = 1.0  # < 1mm 为针孔/溶孔

# 报告标准(1.3 节):缝宽<0.1mm、直径<2mm 的孔洞不计数
REPORT_MIN_DIAMETER_MM = 2.0


class PoreType(str, Enum):
    """孔洞按充填特征分类(来自 PDF 1.2 节)."""
    SOLUTION_CAVE = "溶洞"
    CRYSTAL_CAVE = "晶洞"
    UNKNOWN = "未分类"


class FilledStatus(str, Enum):
    """填充情况(来自 PDF 1.2 节)."""
    UNFILLED = "未充填"
    SEMI_FILLED = "半充填"
    FILLED = "全充填"


class FilledMaterial(str, Enum):
    """填充物(来自 PDF 1.2 节)."""
    NONE = "无"
    CLAY = "泥质"
    CALCITE = "方解石"
    DOLOMITE = "白云石"
    ASPHALT = "沥青"
    GYPSUM = "石膏"
    PYRITE = "黄铁矿"
    KAOLINITE = "高岭石"
    QUARTZ = "石英"


class Effectiveness(str, Enum):
    """有效性评价(来自 PDF 1.2 节)."""
    EFFECTIVE = "有效"
    LESS_EFFECTIVE = "较有效"
    INEFFECTIVE = "无效"


def classify_pore_size(diameter_mm: float) -> str:
    """按直径分类(来自 PDF 1.2 节).
    - 大洞: >10mm
    - 中洞: 5~10mm
    - 小洞: 1~4.9mm
    - 针孔/溶孔: <1mm
    """
    if diameter_mm > PORE_LARGE:
        return "大洞"
    if diameter_mm >= PORE_MEDIUM:
        return "中洞"
    if diameter_mm >= PORE_SMALL:
        return "小洞"
    return "针孔/溶孔"


def diameter_from_area(area_px2: float, scale: Scale) -> float:
    """由像素面积计算等效面积圆直径(单位由标尺决定)."""
    area_real = scale.area_pixels_to_real(area_px2)
    return 2.0 * math.sqrt(area_real / math.pi)


@dataclass
class Pore:
    """单个孔洞对象."""
    id: int
    area_px: float
    area_real: float
    diameter_real: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    perimeter_px: float = 0.0
    size_class: str = ""
    pore_type: PoreType = PoreType.UNKNOWN
    filled_status: FilledStatus = FilledStatus.UNFILLED
    filled_material: FilledMaterial = FilledMaterial.NONE
    effectiveness: Effectiveness = Effectiveness.EFFECTIVE

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["pore_type"] = self.pore_type.value
        d["filled_status"] = self.filled_status.value
        d["filled_material"] = self.filled_material.value
        d["effectiveness"] = self.effectiveness.value
        return d


@dataclass
class PoreAnalysisResult:
    """孔洞分析结果汇总."""
    image_area_px: float
    image_area_real: float
    pore_count: int
    pore_count_report: int  # 报告标准下(直径≥2mm)的孔洞数
    total_pore_area_px: float
    total_pore_area_real: float
    average_area_real: float
    porosity: float  # 面孔率
    average_diameter_real: float
    max_diameter_real: float
    min_diameter_real: float
    size_distribution: Dict[str, int] = field(default_factory=dict)
    diameter_statistics: Dict[str, float] = field(default_factory=dict)
    pores: List[Pore] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["pores"] = [p.to_dict() for p in self.pores]
        return d


def _perimeter(mask_component: np.ndarray) -> float:
    """计算连通域周长(像素)."""
    contours, _ = cv2.findContours(
        mask_component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0.0
    return float(cv2.arcLength(contours[0], True))


def analyze_pores(
    mask: np.ndarray,
    scale: Scale,
    image_shape: Optional[Tuple[int, int]] = None,
    min_diameter_real: float = REPORT_MIN_DIAMETER_MM,
) -> PoreAnalysisResult:
    """分析掩码中所有孔洞.
    Args:
        mask: 单通道 uint8 掩码(0/255)
        scale: 标尺对象(用于像素→mm 换算)
        image_shape: 图像形状(H,W);若为 None 则取 mask.shape
        min_diameter_real: 报告级最小直径(mm)
    Returns:
        PoreAnalysisResult
    """
    if image_shape is None:
        h, w = mask.shape[:2]
    else:
        h, w = image_shape[:2]
    image_area_px = float(h * w)
    image_area_real = scale.area_pixels_to_real(image_area_px)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pores: List[Pore] = []
    total_area_px = 0.0
    total_area_real = 0.0
    pore_count_report = 0
    for i in range(1, n):
        area_px = float(stats[i, cv2.CC_STAT_AREA])
        if area_px <= 0:
            continue
        area_real = scale.area_pixels_to_real(area_px)
        diameter = diameter_from_area(area_px, scale)
        size_class = classify_pore_size(diameter)
        # 截取该连通域并计算周长
        x, y, ww, hh = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        sub = (labels[y:y + hh, x:x + ww] == i).astype(np.uint8) * 255
        perim = _perimeter(sub)
        pore = Pore(
            id=i,
            area_px=area_px,
            area_real=area_real,
            diameter_real=diameter,
            centroid=(float(centroids[i][0]), float(centroids[i][1])),
            bbox=(x, y, ww, hh),
            perimeter_px=perim,
            size_class=size_class,
        )
        pores.append(pore)
        total_area_px += area_px
        total_area_real += area_real
        if diameter >= min_diameter_real:
            pore_count_report += 1

    # 统计
    if pores:
        diameters = [p.diameter_real for p in pores]
        areas = [p.area_real for p in pores]
        avg_d = sum(diameters) / len(diameters)
        avg_a = sum(areas) / len(areas)
        max_d = max(diameters)
        min_d = min(diameters)
    else:
        avg_d = avg_a = max_d = min_d = 0.0
    porosity = (total_area_real / image_area_real) if image_area_real > 0 else 0.0
    size_dist: Dict[str, int] = {"大洞": 0, "中洞": 0, "小洞": 0, "针孔/溶孔": 0}
    for p in pores:
        size_dist[p.size_class] = size_dist.get(p.size_class, 0) + 1
    diameter_stats = {
        "平均直径(mm)": round(avg_d, 3),
        "最大直径(mm)": round(max_d, 3),
        "最小直径(mm)": round(min_d, 3),
        "直径标准差(mm)": round(float(np.std([p.diameter_real for p in pores])), 3) if pores else 0.0,
    }
    return PoreAnalysisResult(
        image_area_px=image_area_px,
        image_area_real=image_area_real,
        pore_count=len(pores),
        pore_count_report=pore_count_report,
        total_pore_area_px=total_area_px,
        total_pore_area_real=total_area_real,
        average_area_real=avg_a,
        porosity=porosity,
        average_diameter_real=avg_d,
        max_diameter_real=max_d,
        min_diameter_real=min_d,
        size_distribution=size_dist,
        diameter_statistics=diameter_stats,
        pores=pores,
    )


def compute_diameter_frequency_curve(pores: Sequence[Pore], bins: int = 10) -> Dict[str, list]:
    """孔洞直径频率分布(用于绘制频率/累计频率/正态累计曲线)."""
    diameters = [p.diameter_real for p in pores]
    if not diameters:
        return {"bins": [], "counts": [], "cumulative": [], "normal_cumulative": []}
    arr = np.array(diameters, dtype=np.float64)
    counts, edges = np.histogram(arr, bins=bins)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    cum = np.cumsum(counts).tolist()
    total = cum[-1] if cum and cum[-1] > 0 else 1
    norm_cum = (np.cumsum(counts) / total).tolist()
    return {
        "bins": [round(c, 3) for c in centers],
        "counts": counts.tolist(),
        "cumulative": [round(c, 3) for c in cum],
        "normal_cumulative": [round(c, 4) for c in norm_cum],
    }


__all__ = [
    "Pore", "PoreAnalysisResult", "PoreType", "FilledStatus",
    "FilledMaterial", "Effectiveness",
    "PORE_LARGE", "PORE_MEDIUM", "PORE_SMALL", "PORE_PINHOLE",
    "REPORT_MIN_DIAMETER_MM",
    "classify_pore_size", "diameter_from_area",
    "analyze_pores", "compute_diameter_frequency_curve",
]
