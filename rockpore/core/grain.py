"""粒度分析模块.

实现基于距离变换 + 分水岭的颗粒识别与定量分析:
- 灰度 + CLAHE → OTSU 反相 → 形态学清理 → 距离变换 → 分水岭分割
- 单颗粒属性: 面积/周长/长轴/短轴/长宽比/圆度
- 整体统计: 颗粒个数、粒级分布、平均粒径

参考 PDF 第 3 章粒度分析 + Wentworth (1958) 粒级分类 + Folk & Ward (1957) 粒度参数:

粒级分类 (Wentworth 1958, 单位 mm):
- 巨砾   >256
- 粗砾   64~256
- 中砾   16~64
- 细砾    2~16
- 极粗砂  1~2
- 粗砂   0.5~1
- 中砂   0.25~0.5
- 细砂   0.125~0.25
- 极细砂 0.0625~0.125
- 粉砂   0.0039~0.0625
- 黏土   <0.0039

粒度参数 (Folk & Ward 1957, 基于粒径 φ 值, φ = -log2(d_mm)):
- 平均粒径 Mz = (φ16 + φ50 + φ84) / 3
- 分选系数 σ1 = (φ84 - φ16) / 4 + (φ95 - φ5) / 6.6
- 偏度   Sk1 = (φ16 + φ84 - 2·φ50) / (2·(φ84-φ16)) + (φ5 + φ95 - 2·φ50) / (2·(φ95-φ5))
- 峰度   KG  = (φ95 - φ5) / (2.44·(φ75 - φ25))
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibration import Scale


# Wentworth 粒级分类阈值 (单位 mm)
SIZE_BOUNDARY_BOLLDER_LARGE = 256.0
SIZE_BOUNDARY_BOLLDER_MID = 64.0
SIZE_BOUNDARY_BOLLDER_SMALL = 16.0
SIZE_BOUNDARY_GRANULE = 2.0
SIZE_BOUNDARY_VC_SAND = 1.0
SIZE_BOUNDARY_C_SAND = 0.5
SIZE_BOUNDARY_M_SAND = 0.25
SIZE_BOUNDARY_F_SAND = 0.125
SIZE_BOUNDARY_VF_SAND = 0.0625
SIZE_BOUNDARY_SILT = 0.0039


def classify_grain_size(diameter_mm: float) -> str:
    """按粒径分类 (Wentworth 1958).

    Args:
        diameter_mm: 颗粒粒径 (mm)

    Returns:
        粒级名称: 巨砾/粗砾/中砾/细砾/极粗砂/粗砂/中砂/细砂/极细砂/粉砂/黏土
    """
    if diameter_mm >= SIZE_BOUNDARY_BOLLDER_LARGE:
        return "巨砾"
    if diameter_mm >= SIZE_BOUNDARY_BOLLDER_MID:
        return "粗砾"
    if diameter_mm >= SIZE_BOUNDARY_BOLLDER_SMALL:
        return "中砾"
    if diameter_mm >= SIZE_BOUNDARY_GRANULE:
        return "细砾"
    if diameter_mm >= SIZE_BOUNDARY_VC_SAND:
        return "极粗砂"
    if diameter_mm >= SIZE_BOUNDARY_C_SAND:
        return "粗砂"
    if diameter_mm >= SIZE_BOUNDARY_M_SAND:
        return "中砂"
    if diameter_mm >= SIZE_BOUNDARY_F_SAND:
        return "细砂"
    if diameter_mm >= SIZE_BOUNDARY_VF_SAND:
        return "极细砂"
    if diameter_mm >= SIZE_BOUNDARY_SILT:
        return "粉砂"
    return "黏土"


# 中文粒级名 → 标准阈值 (mm)
SIZE_CLASSES_ORDERED: List[Tuple[str, float]] = [
    ("巨砾", SIZE_BOUNDARY_BOLLDER_LARGE),
    ("粗砾", SIZE_BOUNDARY_BOLLDER_MID),
    ("中砾", SIZE_BOUNDARY_BOLLDER_SMALL),
    ("细砾", SIZE_BOUNDARY_GRANULE),
    ("极粗砂", SIZE_BOUNDARY_VC_SAND),
    ("粗砂", SIZE_BOUNDARY_C_SAND),
    ("中砂", SIZE_BOUNDARY_M_SAND),
    ("细砂", SIZE_BOUNDARY_F_SAND),
    ("极细砂", SIZE_BOUNDARY_VF_SAND),
    ("粉砂", SIZE_BOUNDARY_SILT),
    ("黏土", 0.0),
]


@dataclass
class GrainParams:
    """粒度检测参数.

    Attributes:
        # 二值化参数
        blur_kernel: 高斯模糊核大小 (奇数, 0=不模糊)
        use_otsu: True=OTSU 反相 (推荐,深色矿物勾勒的边界)
        otsu_offset: OTSU 阈值偏移 (负值=更暗)
        # 形态学
        morph_close: 闭运算核 (连接断裂颗粒边界)
        morph_open: 开运算核 (去噪)
        # 颗粒筛选
        min_area_px: 候选最小面积 (像素)
        max_area_ratio: 单颗粒最大占比 (排除超大假阳性)
        # 距离变换 + 分水岭
        distance_threshold_ratio: 距离变换峰值筛选比例
        h_distance: 距离变换中 hillock 的高度阈值比例
        # 标注
        min_solidity: 最小密实度 (面积/凸包面积),过滤不规则形状
        circularity_min: 最小圆度,过滤异常形状
    """
    # 二值化
    blur_kernel: int = 7
    use_otsu: bool = True
    otsu_offset: int = -10
    # 形态学
    morph_close: int = 7
    morph_open: int = 3
    # 颗粒筛选
    min_area_px: int = 200
    max_area_ratio: float = 0.20
    # 距离变换 + 分水岭
    distance_threshold_ratio: float = 0.30
    h_distance: float = 0.4
    # 标注
    min_solidity: float = 0.4
    circularity_min: float = 0.2

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Grain:
    """单颗粒对象.

    Attributes:
        id: 颗粒编号
        area_px: 像素面积
        perimeter_px: 像素周长
        major_axis_px: 长轴 (像素,基于拟合椭圆)
        minor_axis_px: 短轴 (像素)
        diameter_mm: 等效直径 (mm) (基于面积,圆面积公式)
        diameter_major_mm: 长轴直径 (mm)
        centroid: 质心 (x, y) 像素坐标
        bbox: 包围盒 (x, y, w, h)
        orientation_deg: 长轴倾角 (-90~90)
        circularity: 圆度 4πA/P²
        solidity: 密实度 A/凸包A
        aspect_ratio: 长宽比 major/minor
        size_class: 粒级名称 (Wentworth)
        contour: 轮廓点 (N×2) (可选,用于重绘)
    """
    id: int
    area_px: float
    perimeter_px: float
    major_axis_px: float
    minor_axis_px: float
    diameter_mm: float
    diameter_major_mm: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    orientation_deg: float
    circularity: float
    solidity: float
    aspect_ratio: float
    size_class: str = ""
    contour: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("contour", None)
        d["centroid"] = list(self.centroid)
        d["bbox"] = list(self.bbox)
        return d


@dataclass
class GrainAnalysisResult:
    """粒度分析结果汇总.

    Attributes:
        image_area_px: 图像总面积 (像素)
        image_area_real: 图像实际面积 (mm²)
        grain_count: 颗粒总数 (所有检测到的)
        grain_count_filtered: 过滤后的颗粒数
        total_area_px: 颗粒总面积 (像素)
        total_area_real: 颗粒总面积 (mm²)
        average_diameter_mm: 平均粒径 (mm,等效直径)
        median_diameter_mm: 中位粒径 (mm)
        max_diameter_mm: 最大粒径 (mm)
        min_diameter_mm: 最小粒径 (mm)
        average_circularity: 平均圆度
        size_distribution: 粒级分布 {粒级名: 颗粒数}
        grains: 颗粒列表
    """
    image_area_px: float
    image_area_real: float
    grain_count: int
    grain_count_filtered: int
    total_area_px: float
    total_area_real: float
    average_diameter_mm: float
    median_diameter_mm: float
    max_diameter_mm: float
    min_diameter_mm: float
    average_circularity: float
    size_distribution: Dict[str, int] = field(default_factory=dict)
    grains: List[Grain] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["grains"] = [g.to_dict() for g in self.grains]
        return d


# ============================================================
# 算法实现
# ============================================================


def _preprocess(image: np.ndarray, params: GrainParams) -> np.ndarray:
    """灰度化 + CLAHE + 可选模糊."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # CLAHE 局部直方图均衡化 (增强对比度)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if params.blur_kernel > 0:
        ksize = max(3, params.blur_kernel | 1)
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return gray


def detect_grain_mask(image: np.ndarray, params: Optional[GrainParams] = None) -> Tuple[np.ndarray, np.ndarray]:
    """检测颗粒前景 mask.

    颗粒前景:浅色矿物(石英/长石),深色基质(黑云母)被识别为背景.

    Returns:
        bin_mask: 二值前景 (颗粒=255)
        edges: 边缘图 (供可视化)
    """
    if params is None:
        params = GrainParams()
    gray = _preprocess(image, params)
    h, w = gray.shape
    # 1. 阈值: OTSU 正向 (浅色矿物 → 白)
    # 之前 v1.1.3 误用 OTSU_INV 导致反相,合成图大量误识别 → 现改为正向
    if params.use_otsu:
        otsu_t, bin_mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        if params.otsu_offset != 0:
            new_t = max(0, min(255, otsu_t + params.otsu_offset))
            _, bin_mask = cv2.threshold(gray, new_t, 255, cv2.THRESH_BINARY)
    else:
        # 自适应阈值 (正向:浅色颗粒为前景)
        block = 21
        bin_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, 5,
        )
    # 2. 闭运算:连接断裂颗粒边界
    if params.morph_close > 0:
        k = max(1, params.morph_close | 1)
        bin_mask = cv2.morphologyEx(
            bin_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        )
    # 3. 开运算:去噪
    if params.morph_open > 0:
        k = max(1, params.morph_open | 1)
        bin_mask = cv2.morphologyEx(
            bin_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        )
    edges = bin_mask.copy()
    return bin_mask, edges


def _find_seed_points(
    bin_mask: np.ndarray,
    params: GrainParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """距离变换找种子点 (分水岭起点).

    核心策略:在距离变换图中找 local maxima (山峰)作为分水岭种子.
    local maxima 比 connectedComponents 更准确:每个颗粒中心对应一个山峰,
    不会因为颗粒粘连而漏种子.

    Returns:
        markers: 标记图像 (每个前景区域一个标记 ID)
        sure_fg: 确定前景 mask
        dist_transform: 距离变换结果
    """
    # 1. 确定前景:形态学腐蚀 (去掉边缘像素)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_fg = cv2.erode(bin_mask, kernel, iterations=2)
    # 2. 距离变换
    dist_transform = cv2.distanceTransform(sure_fg, cv2.DIST_L2, 5)
    dist_max = dist_transform.max()
    if dist_max < 1.0:
        # 太小 (没有明显颗粒),返回空
        return np.zeros_like(bin_mask, dtype=np.int32), sure_fg, dist_transform
    # 3. 找 local maxima (山峰)
    # 使用较大的 kernel 保证是真正的山峰,不是小波动
    peak_kernel_size = max(5, int(dist_max * 0.15) | 1)
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_kernel_size, peak_kernel_size))
    dilated = cv2.dilate(dist_transform, peak_kernel)
    # 山峰 = dist == dilated AND dist > threshold
    thresh = dist_max * params.distance_threshold_ratio
    peak_mask = ((dist_transform == dilated) & (dist_transform > thresh)).astype(np.uint8) * 255
    # 4. 连通域标记
    n_peaks, peak_labels = cv2.connectedComponents(peak_mask)
    if n_peaks <= 1:
        # 没有山峰,fallback 到阈值法
        _, sure_thresh = cv2.threshold(dist_transform, thresh, 255, cv2.THRESH_BINARY)
        sure_thresh = np.uint8(sure_thresh)
        n_peaks, peak_labels = cv2.connectedComponents(sure_thresh)
    # 5. 构建 markers (OpenCV 分水岭要求: background=1, 前景=2,3,...)
    markers = np.zeros_like(bin_mask, dtype=np.int32)
    markers[bin_mask == 0] = 1  # background
    # 每个山峰一个 ID
    for i in range(1, n_peaks):
        markers[peak_labels == i] = i + 1
    return markers, sure_fg, dist_transform


def _watershed_split(bin_mask: np.ndarray, params: GrainParams) -> np.ndarray:
    """距离变换 + 分水岭分割粘连颗粒.

    Returns:
        markers: 分割后的标记图 (每个连通区域一个 ID,0=边界,1=background)
    """
    markers, sure_fg, _ = _find_seed_points(bin_mask, params)
    # 分水岭 (需要 3 通道图像)
    if bin_mask.ndim == 2:
        bin_mask_3ch = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)
    else:
        bin_mask_3ch = bin_mask.copy()
    markers_out = cv2.watershed(bin_mask_3ch, markers.copy())
    return markers_out


def _extract_grain_contours(bin_mask: np.ndarray, markers: np.ndarray, params: GrainParams) -> List[np.ndarray]:
    """从分水岭结果提取每个颗粒的精确轮廓.

    策略:对每个标记区域,在原 mask 上找精确轮廓 (sub-pixel 精度)。
    """
    grains_contours: List[np.ndarray] = []
    h, w = bin_mask.shape
    image_area = w * h
    unique_markers = np.unique(markers)
    for marker_id in unique_markers:
        if marker_id <= 1:  # background or unknown
            continue
        region_mask = (markers == marker_id).astype(np.uint8) * 255
        area_px = int((region_mask > 0).sum())
        if area_px < params.min_area_px:
            continue
        # 排除超大区域
        if area_px / image_area > params.max_area_ratio:
            continue
        # 找精确轮廓
        contours, _ = cv2.findContours(
            region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            continue
        # 取最大轮廓
        cnt = max(contours, key=cv2.contourArea)
        grains_contours.append(cnt)
    return grains_contours


def _compute_grain_properties(
    contours: List[np.ndarray],
    scale: Scale,
    image_area_px: int,
    next_id: int = 1,
) -> List[Grain]:
    """从轮廓列表计算颗粒属性."""
    grains: List[Grain] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3:  # 太小跳过
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        # 拟合椭圆 (至少 5 个点)
        if len(cnt) >= 5:
            (cx, cy), (major_px, minor_px), angle = cv2.fitEllipse(cnt)
            # fitEllipse 返回 (MAJOR, MINOR),major >= minor
            major_px = max(major_px, 1.0)
            minor_px = max(minor_px, 1.0)
            orientation = float(angle)
        else:
            # 用 bbox 替代
            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
            cx = x_b + w_b / 2
            cy = y_b + h_b / 2
            major_px = max(w_b, h_b)
            minor_px = min(w_b, h_b)
            orientation = 0.0
        # 等效直径 (基于面积):A = π(d/2)² → d = 2√(A/π)
        diameter_px = 2.0 * math.sqrt(area / math.pi)
        # 转 mm
        if scale.pixels_per_unit > 0:
            diameter_mm = scale.pixels_to_real(diameter_px)
            diameter_major_mm = scale.pixels_to_real(major_px)
        else:
            diameter_mm = diameter_px
            diameter_major_mm = major_px
        # 圆度 4πA/P² (1=完美圆)
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        # 凸包面积 → 密实度
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        # 长宽比
        aspect = major_px / max(minor_px, 1.0)
        # bbox
        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
        # 粒级分类 (基于长轴 mm)
        grain = Grain(
            id=next_id,
            area_px=float(area),
            perimeter_px=float(perimeter),
            major_axis_px=float(major_px),
            minor_axis_px=float(minor_px),
            diameter_mm=float(diameter_mm),
            diameter_major_mm=float(diameter_major_mm),
            centroid=(float(cx), float(cy)),
            bbox=(int(x_b), int(y_b), int(w_b), int(h_b)),
            orientation_deg=orientation,
            circularity=float(circularity),
            solidity=float(solidity),
            aspect_ratio=float(aspect),
            size_class=classify_grain_size(diameter_major_mm) if scale.pixels_per_unit > 0 else "未标定",
            contour=cnt,
        )
        grains.append(grain)
        next_id += 1
    return grains


def _filter_grains(
    grains: List[Grain],
    min_solidity: float = 0.4,
    circularity_min: float = 0.2,
) -> List[Grain]:
    """过滤掉形状异常的颗粒 (如过长、过扁、过碎片化)."""
    out: List[Grain] = []
    for g in grains:
        if g.solidity < min_solidity:
            continue
        if g.circularity < circularity_min:
            continue
        if g.aspect_ratio > 5.0:  # 过长也过滤
            continue
        out.append(g)
    return out


def analyze_grains(
    image: np.ndarray,
    scale: Scale,
    params: Optional[GrainParams] = None,
    mask: Optional[np.ndarray] = None,
) -> Tuple[GrainAnalysisResult, np.ndarray, np.ndarray]:
    """粒度分析主函数.

    Args:
        image: BGR 或灰度图像
        scale: 标尺对象 (用于像素→mm 转换)
        params: 检测参数 (None=默认)
        mask: 可选 mask (限制分析区域,255=有效)

    Returns:
        result: GrainAnalysisResult
        bin_mask: 二值前景
        markers: 分水岭分割结果 (0=边界)
    """
    if params is None:
        params = GrainParams()

    h, w = image.shape[:2]
    image_area_px = float(w * h)
    if scale.pixels_per_unit > 0:
        image_area_real = scale.pixels_to_real(w) * scale.pixels_to_real(h)
    else:
        image_area_real = image_area_px

    # 1. 检测二值前景
    bin_mask, _ = detect_grain_mask(image, params)
    if mask is not None and mask.shape == bin_mask.shape:
        bin_mask = cv2.bitwise_and(bin_mask, mask)

    # 2. 分水岭分割
    markers = _watershed_split(bin_mask, params)

    # 3. 提取颗粒轮廓
    contours = _extract_grain_contours(bin_mask, markers, params)

    # 4. 计算颗粒属性
    grains = _compute_grain_properties(contours, scale, int(image_area_px))

    # 5. 形状过滤
    grains_filtered = _filter_grains(
        grains,
        min_solidity=params.min_solidity,
        circularity_min=params.circularity_min,
    )

    # 6. 统计
    if grains_filtered:
        diameters = [g.diameter_mm for g in grains_filtered]
        avg_diam = float(np.mean(diameters))
        med_diam = float(np.median(diameters))
        max_diam = float(np.max(diameters))
        min_diam = float(np.min(diameters))
        avg_circ = float(np.mean([g.circularity for g in grains_filtered]))
        total_area_px = float(sum(g.area_px for g in grains_filtered))
    else:
        avg_diam = med_diam = max_diam = min_diam = 0.0
        avg_circ = 0.0
        total_area_px = 0.0

    # 粒级分布
    size_dist: Dict[str, int] = {name: 0 for name, _ in SIZE_CLASSES_ORDERED}
    for g in grains_filtered:
        size_dist[g.size_class] = size_dist.get(g.size_class, 0) + 1

    total_area_real = total_area_px * (image_area_real / image_area_px) if image_area_px > 0 else 0.0

    result = GrainAnalysisResult(
        image_area_px=image_area_px,
        image_area_real=float(image_area_real),
        grain_count=len(grains),
        grain_count_filtered=len(grains_filtered),
        total_area_px=total_area_px,
        total_area_real=float(total_area_real),
        average_diameter_mm=avg_diam,
        median_diameter_mm=med_diam,
        max_diameter_mm=max_diam,
        min_diameter_mm=min_diam,
        average_circularity=avg_circ,
        size_distribution=size_dist,
        grains=grains_filtered,
    )
    return result, bin_mask, markers


def draw_grain_annotations(
    image: np.ndarray,
    grains: List[Grain],
    color: Tuple[int, int, int] = (255, 128, 0),
    thickness: int = 2,
    show_id: bool = True,
    show_class: bool = False,
) -> np.ndarray:
    """在图像上绘制颗粒标注.

    Returns:
        标注后的图像 (副本)
    """
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for g in grains:
        # 绘制轮廓
        if g.contour is not None:
            cv2.drawContours(out, [g.contour], -1, color, thickness)
        # 质心
        cx, cy = int(g.centroid[0]), int(g.centroid[1])
        cv2.circle(out, (cx, cy), 4, color, -1)
        cv2.circle(out, (cx, cy), 6, color, 1)
        # 编号 / 类别
        if show_id or show_class:
            text = f"#{g.id}"
            if show_class:
                text += f" {g.size_class}"
            cv2.putText(out, text, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def generate_mask_from_markers(markers: np.ndarray, grain_ids: List[int]) -> np.ndarray:
    """从分水岭标记图生成 mask (可视化用,展示颗粒分布).

    Args:
        markers: 分水岭标记图
        grain_ids: 选中的颗粒 ID (marker_id)

    Returns:
        mask: 选中颗粒的 mask (255=选中)
    """
    mask = np.zeros_like(markers, dtype=np.uint8)
    for mid in grain_ids:
        mask[markers == mid] = 255
    return mask


def compute_size_distribution_text(result: GrainAnalysisResult) -> str:
    """格式化粒级分布为可读文本."""
    lines = ["粒级分布:"]
    lines.append("-" * 40)
    for name, _ in SIZE_CLASSES_ORDERED:
        count = result.size_distribution.get(name, 0)
        if count > 0:
            lines.append(f"  {name:8s}: {count:4d} 颗")
    lines.append("-" * 40)
    lines.append(f"  合计: {result.grain_count_filtered} 颗")
    return "\n".join(lines)


def detect_grain_mask_with_params(image: np.ndarray, params_dict: Dict) -> np.ndarray:
    """便捷函数: 用 dict 参数检测 mask (供 GUI 使用)."""
    params = GrainParams(**{k: v for k, v in params_dict.items() if k in GrainParams.__dataclass_fields__})
    bin_mask, _ = detect_grain_mask(image, params)
    return bin_mask