"""核心算法模块.

包含标尺换算、图像预处理、孔洞分割、数学形态学、孔洞分析与报告生成.
"""

from .calibration import (
    Scale, Calibration, ScaleUnit,
    pixel_to_mm, mm_to_pixel, scale_from_dpi, get_image_dpi,
)
from .preprocessing import preprocess, auto_levels, adjust_brightness, adjust_contrast
from .segmentation import segment_by_color, extract_pores, flood_fill_region
from .morphology import dilate_region, erode_region, remove_noise, fill_holes
from .analysis import (
    analyze_pores, Pore, PoreAnalysisResult, classify_pore_size,
    PORE_LARGE, PORE_MEDIUM, PORE_SMALL, PORE_PINHOLE
)
from .report import generate_report, ReportData
from .accuracy import (
    AccuracyReport, evaluate_accuracy, detect_pores_robust,
)
from .synthetic import (
    generate_synthetic_rock, SyntheticPore, make_default_synthetic,
)
from .io_utils import imread_unicode, imwrite_unicode, find_sample_image

__all__ = [
    "Scale", "Calibration", "pixel_to_mm", "mm_to_pixel",
    "preprocess", "auto_levels", "adjust_brightness", "adjust_contrast",
    "segment_by_color", "extract_pores", "flood_fill_region",
    "dilate_region", "erode_region", "remove_noise", "fill_holes",
    "analyze_pores", "Pore", "PoreAnalysisResult", "classify_pore_size",
    "PORE_LARGE", "PORE_MEDIUM", "PORE_SMALL", "PORE_PINHOLE",
    "generate_report", "ReportData",
    "AccuracyReport", "evaluate_accuracy", "detect_pores_robust",
    "generate_synthetic_rock", "SyntheticPore", "make_default_synthetic",
    "imread_unicode", "imwrite_unicode", "find_sample_image",
]
