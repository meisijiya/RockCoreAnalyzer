"""核心算法模块.

包含标尺换算、图像预处理、孔洞分割、数学形态学、孔洞分析与报告生成,
以及裂缝检测/分析/合成/准确率评估.
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
from .fracture import (
    FractureParams, FractureAnalysisResult, Fracture,
    FractureType, FractureOpenness, FractureFill, FractureEffectiveness,
    classify_fracture_width, detect_fracture_mask, analyze_fractures,
    draw_fracture_annotations, REPORT_MIN_WIDTH_MM,
    FRACTURE_LARGE, FRACTURE_MEDIUM,
)
from .synthetic_fracture import (
    SyntheticFracture, generate_synthetic_fracture_rock,
    make_default_synthetic_fracture,
)
from .fracture_accuracy import (
    FractureAccuracyReport, evaluate_fracture_accuracy,
    detect_fractures_robust, compute_pixel_metrics, match_fractures,
)
from .grain import (
    GrainParams, GrainAnalysisResult, Grain,
    classify_grain_size, detect_grain_mask, analyze_grains,
    draw_grain_annotations, compute_size_distribution_text,
    SIZE_CLASSES_ORDERED,
)
from .synthetic_grain import (
    SyntheticGrain, generate_synthetic_grain_rock,
    make_default_synthetic_grain, make_overlapping_synthetic_grain,
    make_granite_synthetic_grain,
)
from .grain_accuracy import (
    GrainAccuracyReport, evaluate_grain_accuracy,
    detect_grains_robust, compute_pixel_metrics, match_grains,
)
from .report_exporter import ReportExporter, SUPPORTED_FORMATS

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
    # 裂缝分析
    "FractureParams", "FractureAnalysisResult", "Fracture",
    "FractureType", "FractureOpenness", "FractureFill", "FractureEffectiveness",
    "classify_fracture_width", "detect_fracture_mask", "analyze_fractures",
    "draw_fracture_annotations", "REPORT_MIN_WIDTH_MM",
    "FRACTURE_LARGE", "FRACTURE_MEDIUM",
    "SyntheticFracture", "generate_synthetic_fracture_rock",
    "make_default_synthetic_fracture",
    "FractureAccuracyReport", "evaluate_fracture_accuracy",
    "detect_fractures_robust", "compute_pixel_metrics", "match_fractures",
    # 粒度分析
    "GrainParams", "GrainAnalysisResult", "Grain",
    "classify_grain_size", "detect_grain_mask", "analyze_grains",
    "draw_grain_annotations", "compute_size_distribution_text",
    "SIZE_CLASSES_ORDERED",
    "SyntheticGrain", "generate_synthetic_grain_rock",
    "make_default_synthetic_grain", "make_overlapping_synthetic_grain",
    "make_granite_synthetic_grain",
    "GrainAccuracyReport", "evaluate_grain_accuracy",
    "detect_grains_robust", "match_grains",
    "ReportExporter", "SUPPORTED_FORMATS",
]
