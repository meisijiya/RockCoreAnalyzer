"""裂缝检测与分析模块.

实现基于 HoughLinesP 的裂缝识别与定量分析:
- 边缘检测 (Canny) → 概率霍夫变换 (HoughLinesP) → 线段合并/形态学骨架化
- 单条裂缝属性: 长度/宽度/倾角/位置/类型
- 整体统计: 线密度、面密度、宽度分布、倾角玫瑰图

参考 PDF 1.2 节:
- 裂缝按缝宽分类: 大缝(>10mm)、中缝(1-10mm)、小缝(<1mm)
- 报告标准 (1.3): 缝宽 < 0.1mm 不计入报告
- 面密度 (2D): 裂缝累计长度 / 分析面积 (1/mm)
- 线密度 (2D): 单位面积裂缝条数
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibration import Scale


# 裂缝大小分类阈值(单位:mm,来自 PDF 1.2 节)
FRACTURE_LARGE = 10.0
FRACTURE_MEDIUM = 1.0

# 报告标准(1.3):缝宽<0.1mm 的裂缝不计数
REPORT_MIN_WIDTH_MM = 0.1


class FractureType(str, Enum):
    """裂缝成因类型(来自 PDF 1.2 节)."""
    STRUCTURAL = "构造缝"
    DIAGENETIC = "成岩缝"
    WEATHERING = "风化缝"
    UNKNOWN = "未分类"


class FractureOpenness(str, Enum):
    """开启程度(来自 PDF 1.2 节)."""
    OPEN = "开启"
    SEMI_OPEN = "半开启"
    CLOSED = "闭合"


class FractureFill(str, Enum):
    """充填情况(来自 PDF 1.2 节)."""
    UNFILLED = "未充填"
    SEMI_FILLED = "半充填"
    FILLED = "全充填"


class FractureEffectiveness(str, Enum):
    """有效性评价(来自 PDF 1.2 节)."""
    EFFECTIVE = "有效"
    LESS_EFFECTIVE = "较有效"
    INEFFECTIVE = "无效"


def classify_fracture_width(width_mm: float) -> str:
    """按缝宽分类(PDF 1.2 节).
    - 大缝: ≥10mm
    - 中缝: 1~10mm
    - 小缝/微缝: <1mm
    """
    if width_mm >= FRACTURE_LARGE:
        return "大缝"
    if width_mm >= FRACTURE_MEDIUM:
        return "中缝"
    return "小缝"


@dataclass
class FractureParams:
    """裂缝检测参数.
    Attributes:
        method: 检测方法 ("hough" 或 "adaptive")
        canny_low: Canny 低阈值 (Hough 方法)
        canny_high: Canny 高阈值 (Hough 方法)
        hough_threshold: HoughLinesP 累加阈值 (Hough 方法)
        min_line_length_px: 最小线长(像素)
        max_line_gap_px: 最大线间断距(像素)
        dilation_kernel_px: 线段→区域膨胀核(像素),用于将线段转为面积
        # 自适应方法专属参数 (适合真实岩石图,纹理复杂)
        # v1.1.2: 默认改用 OTSU 暗色阈值 + 长宽比筛选,综合表现更好
        blur_kernel: 高斯模糊核大小 (奇数, 0=不模糊)
        adaptive_block: 自适应阈值邻域大小 (仅当 use_otsu=False)
        adaptive_C: 自适应阈值常数 (仅当 use_otsu=False)
        morph_close: 闭运算核大小 (连接相近裂缝, 0=不做)
        morph_open: 开运算核大小 (去噪, 0=不做)
        use_otsu: True 用 OTSU 暗色阈值(推荐), False 用自适应阈值
        min_aspect_ratio: 最小长宽比 (线状过滤, 推荐 1.5-2.0)
        min_area_for_candidate: 候选最小面积(像素),排除过小区域
        max_area_ratio: 单个候选占图像最大比例(排除覆盖全图的假阳性)
        min_skeleton_length_px: 最小骨架长度(像素)
    """
    method: str = "hough"  # "hough" | "adaptive" — 默认 hough 适合合成图
    # Hough 参数
    canny_low: int = 90
    canny_high: int = 270
    hough_threshold: int = 25
    min_line_length_px: int = 20
    max_line_gap_px: int = 15
    dilation_kernel_px: int = 3
    # 自适应参数(v1.1.3 推荐默认)
    blur_kernel: int = 0            # 0 = 不做模糊(OTSU 已能处理)
    adaptive_block: int = 21
    adaptive_C: int = 10
    morph_close: int = 0            # 0 = 不做闭运算(避免过度合并)
    morph_open: int = 0             # 0 = 不做开运算
    use_otsu: bool = True           # True: OTSU 暗色阈值(推荐); False: 自适应阈值
    otsu_offset: int = -10          # v1.1.3 新增: OTSU 阈值调整(降低10找更暗裂缝)
    min_aspect_ratio: float = 1.5   # v1.1.3: 1.5 平衡(保留更多候选)
    min_area_for_candidate: int = 100  # v1.1.3: 从 20 提升到 100(过滤小阴影)
    max_area_ratio: float = 0.30    # v1.1.3: 0.30(排除超大区域)
    min_skeleton_length_px: int = 15


def detect_fracture_mask(
    image: np.ndarray,
    params: FractureParams = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """检测裂缝区域,返回 (裂缝 mask, 边缘图, 线段/候选列表).

    支持两种算法:
    - "hough" (传统): Canny 边缘 + HoughLinesP 概率霍夫变换
       适合裂缝边缘清晰、对比度高的图像(合成图、岩心薄片)
    - "adaptive" (推荐): 高斯模糊 + 自适应阈值 + 形态学 + 长宽比筛选
       适合真实岩石图(背景纹理多、对比度低),鲁棒性更好

    Args:
        image: BGR 或灰度图像
        params: 检测参数 (FractureParams.method 控制算法)
    Returns:
        (mask, edges, segments)
        mask: 二值裂缝掩码(0/255)
        edges: 中间边缘/梯度图
        segments: HoughLinesP 线段 (hough) 或空 (adaptive)
    """
    if params is None:
        params = FractureParams()
    if params.method == "adaptive":
        return _detect_fracture_adaptive(image, params)
    return _detect_fracture_hough(image, params)


def _detect_fracture_hough(
    image: np.ndarray,
    params: FractureParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HoughLinesP 算法(对比度高的图像适用)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # CLAHE 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    # 高斯降噪
    blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)
    # Canny 边缘
    edges = cv2.Canny(blurred, params.canny_low, params.canny_high)
    # 概率霍夫线检测
    line_segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=params.hough_threshold,
        minLineLength=params.min_line_length_px,
        maxLineGap=params.max_line_gap_px,
    )
    if line_segments is None:
        return np.zeros_like(gray), edges, np.array([], dtype=np.int32).reshape(0, 1, 4)
    mask = np.zeros_like(gray)
    ksize = max(1, params.dilation_kernel_px)
    for seg in line_segments:
        x1, y1, x2, y2 = seg[0]
        cv2.line(mask, (x1, y1), (x2, y2), 255, ksize, cv2.LINE_AA)
    return mask, edges, line_segments


def _detect_fracture_adaptive(
    image: np.ndarray,
    params: FractureParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """自适应算法 — v1.1.3 优化版.

    步骤 (默认 OTSU + 阈值调整):
    1. (可选) 高斯模糊消除噪点
    2. OTSU 暗色阈值(默认) / 自适应阈值(可选)
    3. v1.1.3 新增: 阈值调整 (otsu_offset, 默认 -30, 降低阈值找更暗的裂缝)
    4. (可选) 形态学闭运算连接相近裂缝
    5. (可选) 形态学开运算去噪
    6. 候选筛选: 长宽比≥阈值 + 长度≥阈值 + 排除超大区域

    v1.1.3 改进:
    - 新增 otsu_offset 参数 (默认 -30, 让 OTSU 阈值降低 30, 找更暗的裂缝)
    - 在裂缝样2.png 上能识别中央主裂缝(之前被 OTSU 过滤)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape
    # 1. (可选) 高斯模糊
    if params.blur_kernel > 0:
        ksize = max(3, params.blur_kernel | 1)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    else:
        blurred = gray
    # 2. 阈值: OTSU 或 自适应
    if params.use_otsu:
        otsu_t, bin_mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        # v1.1.3: 阈值调整(让 OTSU 阈值降低)
        # 裂缝通常比 OTSU 自动阈值更暗,降低阈值能找更多裂缝
        if params.otsu_offset != 0:
            new_t = max(0, otsu_t + params.otsu_offset)  # offset < 0 降低阈值
            _, bin_mask = cv2.threshold(
                blurred, new_t, 255, cv2.THRESH_BINARY_INV,
            )
    else:
        block = max(3, params.adaptive_block | 1)
        bin_mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, params.adaptive_C,
        )
    # 3. (可选) 闭运算连接相近裂缝
    if params.morph_close > 0:
        k_close = max(1, params.morph_close | 1)
        bin_mask = cv2.morphologyEx(
            bin_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close)),
        )
    # 4. (可选) 开运算去噪
    if params.morph_open > 0:
        k_open = max(1, params.morph_open)
        bin_mask = cv2.morphologyEx(
            bin_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)),
        )
    # 5. 候选筛选
    mask = _filter_fracture_candidates(bin_mask, params, image_area=w * h)
    edges = bin_mask
    return mask, edges, np.array([], dtype=np.int32).reshape(0, 1, 4)


def _filter_fracture_candidates(
    mask: np.ndarray,
    params: FractureParams,
    image_area: int,
) -> np.ndarray:
    """从二值图中筛选裂缝候选(长宽比≥阈值 + 长度≥阈值 + 排除超大区域)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < params.min_area_for_candidate:
            continue
        length = max(ww, hh)
        width = min(ww, hh)
        aspect = length / max(width, 1)
        if aspect < params.min_aspect_ratio:
            continue
        if length < 15:  # 最小长度兜底
            continue
        # 排除覆盖超大区域(>30% 图像)的候选(假阳性,通常是图像背景)
        if area / image_area > params.max_area_ratio:
            continue
        out[labels == i] = 255
    return out


def _merge_close_segments(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """形态学闭运算连接相近线段."""
    if kernel_size <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def _remove_short_segments(mask: np.ndarray, min_length_px: int = 15) -> np.ndarray:
    """去除长度太小的连通区域(噪点)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    out = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # 用 bbox 对角线作为长度估算
        length = float(np.sqrt(w * w + h * h))
        if length >= min_length_px and area >= 3:
            out[labels == i] = 255
    return out


@dataclass
class Fracture:
    """单条裂缝对象."""
    id: int
    length_px: float           # 像素长度(基于骨架)
    length_real: float         # 实际长度 (mm)
    width_px: float            # 像素宽度(基于面积 / 长度)
    width_real: float          # 实际宽度 (mm)
    orientation_deg: float     # 倾角(0~180,水平=0,垂直=90)
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area_px: float
    size_class: str = ""        # 大缝/中缝/小缝
    fracture_type: FractureType = FractureType.UNKNOWN
    openness: FractureOpenness = FractureOpenness.OPEN
    fill_status: FractureFill = FractureFill.UNFILLED
    effectiveness: FractureEffectiveness = FractureEffectiveness.EFFECTIVE
    skeleton_points: int = 0    # 骨架像素数

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["fracture_type"] = self.fracture_type.value
        d["openness"] = self.openness.value
        d["fill_status"] = self.fill_status.value
        d["effectiveness"] = self.effectiveness.value
        return d


@dataclass
class FractureAnalysisResult:
    """裂缝分析结果汇总."""
    image_area_px: float
    image_area_real: float
    fracture_count: int
    fracture_count_report: int  # 报告级(缝宽 ≥ 0.1mm)
    total_length_px: float
    total_length_real: float
    average_length_real: float
    average_width_real: float
    max_length_real: float
    min_length_real: float
    max_width_real: float
    min_width_real: float
    # 密度参数 (PDF 1.2)
    linear_density: float       # 线密度:条数/长度 (1/mm) — 裂缝条数/累计长度
    areal_density: float        # 面密度:裂缝累计长度/分析面积 (1/mm)
    # 分布
    width_distribution: Dict[str, int] = field(default_factory=dict)
    orientation_histogram: Dict[str, int] = field(default_factory=dict)  # 8 区间
    fractures: List[Fracture] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["fractures"] = [f.to_dict() for f in self.fractures]
        return d


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 骨架化算法.
    opencv-python-headless 不含 cv2.ximgproc.thinning,因此用纯 numpy 实现.
    对线状裂缝效果良好.
    """
    img = (mask > 0).astype(np.uint8)
    prev = np.zeros_like(img)
    # 迭代至不再变化(通常 < 50 次)
    for _ in range(100):
        if np.array_equal(img, prev):
            break
        prev = img.copy()
        # 8 邻域偏移
        p2 = np.roll(np.roll(img, 1, axis=0), 0, axis=1)
        p3 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
        p4 = np.roll(img, -1, axis=1)
        p5 = np.roll(img, 1, axis=1)
        p6 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
        p7 = np.roll(np.roll(img, -1, axis=0), 0, axis=1)
        p8 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)
        p9 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
        # 邻域和 B(P1)
        neighbors = p2.astype(int) + p3 + p4 + p5 + p6 + p7 + p8 + p9
        # 0→1 跳变次数 A(P1)
        # 顺时针:p2 p3 p4 p5 p6 p7 p8 p9 p2
        seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
        transitions = np.zeros_like(img, dtype=int)
        for i in range(8):
            transitions += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(int)
        # 子迭代 1
        cond1 = (img == 1) & (neighbors >= 2) & (neighbors <= 6)
        cond2 = transitions == 1
        cond3 = (p2 * p4 * p6) == 0
        cond4 = (p4 * p6 * p8) == 0
        marker1 = cond1 & cond2 & cond3 & cond4
        img[marker1] = 0
        # 子迭代 2
        cond3b = (p2 * p4 * p8) == 0
        cond4b = (p2 * p6 * p8) == 0
        marker2 = cond1 & cond2 & cond3b & cond4b
        img[marker2] = 0
    return img * 255


def _line_length_px(skeleton: np.ndarray) -> float:
    """估算骨架长度(像素).
    用骨架像素数 × 单位长度近似(水平/垂直=1,对角=√2 ≈ 1.414).
    简化处理:大部分骨架为接近水平的线,直接用骨架像素数作为长度.
    """
    return float(np.count_nonzero(skeleton))


def _line_orientation(skeleton: np.ndarray) -> float:
    """估算骨架主方向角(度,0~180)."""
    ys, xs = np.where(skeleton > 0)
    if len(xs) < 5:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    # PCA
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    if cov.shape != (2, 2):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 主方向 = 最大特征值对应的特征向量
    main_vec = eigvecs[:, -1]
    # 计算与 x 轴夹角
    angle_rad = math.atan2(main_vec[1], main_vec[0])
    angle_deg = math.degrees(angle_rad)
    # 归一到 0~180(线段无方向性)
    angle_deg = abs(angle_deg) % 180
    return float(angle_deg)


def analyze_fractures(
    mask: np.ndarray,
    scale: Scale,
    image_shape: Optional[Tuple[int, int]] = None,
    min_width_real: float = REPORT_MIN_WIDTH_MM,
    min_length_px: int = 15,
) -> FractureAnalysisResult:
    """分析裂缝掩码,产出统计结果.

    Args:
        mask: 单通道 uint8 掩码(0/255)
        scale: 标尺对象(用于像素→mm 换算)
        image_shape: 图像形状(H,W);若为 None 则取 mask.shape
        min_width_real: 报告级最小缝宽(mm),默认 0.1mm (PDF 1.3)
        min_length_px: 最小保留骨架长度(像素)
    Returns:
        FractureAnalysisResult
    """
    if image_shape is None:
        h, w = mask.shape[:2]
    else:
        h, w = image_shape[:2]
    image_area_px = float(h * w)
    image_area_real = scale.area_pixels_to_real(image_area_px)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    fractures: List[Fracture] = []
    total_length_px = 0.0
    total_length_real = 0.0
    total_width_real = 0.0
    fracture_count_report = 0
    width_dist: Dict[str, int] = {"大缝": 0, "中缝": 0, "小缝": 0}
    orientation_hist: Dict[str, int] = {f"{i*22.5:.0f}-{(i+1)*22.5:.0f}": 0 for i in range(8)}

    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area_px = float(stats[i, cv2.CC_STAT_AREA])
        if area_px < 3:
            continue
        # 截取连通域
        sub_mask = (labels[y:y + hh, x:x + ww] == i).astype(np.uint8) * 255
        # 骨架化(用于长度估算)
        skeleton = _skeletonize(sub_mask)
        length_px = _line_length_px(skeleton)
        if length_px < min_length_px:
            continue
        length_real = scale.pixels_to_real(length_px)
        # 宽度 = 面积 / 长度
        width_px = area_px / max(length_px, 1.0)
        width_real = scale.pixels_to_real(width_px)
        # 倾角
        orientation = _line_orientation(skeleton)
        # 分类
        size_class = classify_fracture_width(width_real)
        # 宽度分布
        width_dist[size_class] += 1
        # 倾角分布(8 区间)
        orient_idx = min(int(orientation / 22.5), 7)
        orient_key = f"{orient_idx*22.5:.0f}-{(orient_idx+1)*22.5:.0f}"
        orientation_hist[orient_key] += 1
        # 报告级计数
        if width_real >= min_width_real:
            fracture_count_report += 1
        fr = Fracture(
            id=i,
            length_px=length_px,
            length_real=length_real,
            width_px=width_px,
            width_real=width_real,
            orientation_deg=orientation,
            centroid=(float(centroids[i][0]), float(centroids[i][1])),
            bbox=(x, y, ww, hh),
            area_px=area_px,
            size_class=size_class,
            skeleton_points=int(length_px),
        )
        fractures.append(fr)
        total_length_px += length_px
        total_length_real += length_real
        total_width_real += width_real

    count = len(fractures)
    avg_length = (total_length_real / count) if count > 0 else 0.0
    avg_width = (total_width_real / count) if count > 0 else 0.0
    max_length = max((f.length_real for f in fractures), default=0.0)
    min_length = min((f.length_real for f in fractures), default=0.0)
    max_width = max((f.width_real for f in fractures), default=0.0)
    min_width = min((f.width_real for f in fractures), default=0.0)
    # 面密度(2D): 累计长度 / 分析面积(1/mm)
    areal_density = (total_length_real / image_area_real) if image_area_real > 0 else 0.0
    # 线密度: 裂缝条数 / 累计长度(1/mm), 体现"每 mm 缝长有多少条"
    linear_density = (count / total_length_real) if total_length_real > 0 else 0.0
    return FractureAnalysisResult(
        image_area_px=image_area_px,
        image_area_real=image_area_real,
        fracture_count=count,
        fracture_count_report=fracture_count_report,
        total_length_px=total_length_px,
        total_length_real=total_length_real,
        average_length_real=avg_length,
        average_width_real=avg_width,
        max_length_real=max_length,
        min_length_real=min_length,
        max_width_real=max_width,
        min_width_real=min_width,
        linear_density=linear_density,
        areal_density=areal_density,
        width_distribution=width_dist,
        orientation_histogram=orientation_hist,
        fractures=fractures,
    )


def draw_fracture_annotations(
    image: np.ndarray,
    fractures: List[Fracture],
    annotate: bool = True,
) -> np.ndarray:
    """在原图上绘制裂缝标注(线段+编号).
    Args:
        image: BGR 图像
        fractures: 裂缝列表
        annotate: 是否叠加文字标注
    Returns:
        标注后的 BGR 图像(原图副本)
    """
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for f in fractures:
        x, y, w, h = f.bbox
        # 绘制裂缝外接矩形
        color = (0, 100, 255)  # OpenCV BGR
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
        if annotate:
            label = f"#{f.id} L={f.length_real:.1f}mm W={f.width_real:.2f}mm"
            tx, ty = x, max(y - 4, 12)
            cv2.putText(out, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


__all__ = [
    "FractureParams", "FractureAnalysisResult", "Fracture",
    "FractureType", "FractureOpenness", "FractureFill", "FractureEffectiveness",
    "classify_fracture_width", "detect_fracture_mask", "analyze_fractures",
    "draw_fracture_annotations", "REPORT_MIN_WIDTH_MM",
    "FRACTURE_LARGE", "FRACTURE_MEDIUM",
]
