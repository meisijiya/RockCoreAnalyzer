"""裂缝检测与分析单元测试."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from rockpore.core.calibration import Scale, ScaleUnit, scale_from_dpi
from rockpore.core.fracture import (
    FRACTURE_LARGE, FRACTURE_MEDIUM, REPORT_MIN_WIDTH_MM,
    Fracture, FractureAnalysisResult, FractureParams,
    analyze_fractures, classify_fracture_width, detect_fracture_mask,
    draw_fracture_annotations,
)
from rockpore.core.fracture_accuracy import (
    compute_pixel_metrics, evaluate_fracture_accuracy,
)
from rockpore.core.synthetic_fracture import (
    SyntheticFracture, generate_synthetic_fracture_rock,
    make_default_synthetic_fracture,
)


# ============= FractureParams =============
class TestFractureParams:
    def test_default_values(self):
        p = FractureParams()
        assert p.canny_low > 0
        assert p.canny_high > p.canny_low
        assert p.hough_threshold > 0
        assert p.min_line_length_px > 0

    def test_custom_values(self):
        p = FractureParams(canny_low=50, canny_high=150, hough_threshold=40)
        assert p.canny_low == 50
        assert p.canny_high == 150
        assert p.hough_threshold == 40


# ============= 裂缝宽度分类 =============
class TestFractureWidthClass:
    def test_large_fracture(self):
        assert classify_fracture_width(FRACTURE_LARGE) == "大缝"
        assert classify_fracture_width(15.0) == "大缝"

    def test_medium_fracture(self):
        assert classify_fracture_width(FRACTURE_MEDIUM) == "中缝"
        assert classify_fracture_width(3.0) == "中缝"

    def test_small_fracture(self):
        assert classify_fracture_width(0.5) == "小缝"
        assert classify_fracture_width(0.001) == "小缝"

    def test_boundary(self):
        # 边界 1mm 应为中缝
        assert classify_fracture_width(1.0) == "中缝"
        # 边界 10mm 应为大缝
        assert classify_fracture_width(10.0) == "大缝"


# ============= 合成图生成 =============
class TestSyntheticFracture:
    def test_default_synthetic_shape(self):
        fractures, image, gt_mask = make_default_synthetic_fracture()
        assert len(fractures) >= 5
        assert image.ndim == 3
        assert image.shape[2] == 3
        assert gt_mask.shape == image.shape[:2]
        assert gt_mask.dtype == np.uint8

    def test_default_synthetic_mask_density(self):
        fractures, image, gt_mask = make_default_synthetic_fracture()
        # GT mask 应有非空像素
        assert int((gt_mask > 0).sum()) > 100
        # 不应超过图像的 10%（裂缝不会占满整个图像）
        assert int((gt_mask > 0).sum()) < image.shape[0] * image.shape[1] * 0.1

    def test_single_fracture(self):
        fr = SyntheticFracture(x1=100, y1=100, x2=300, y2=100, width_px=4)
        image, gt_mask = generate_synthetic_fracture_rock(
            width=500, height=300, fractures=[fr], seed=1,
        )
        assert gt_mask.shape == (300, 500)
        assert int((gt_mask > 0).sum()) > 100
        # GT 应集中在水平线 y=100 附近
        ys, xs = np.where(gt_mask > 0)
        assert 90 < ys.mean() < 110

    def test_curved_fracture(self):
        fr = SyntheticFracture(x1=100, y1=100, x2=300, y2=100, width_px=3, curvature=2.0)
        image, gt_mask = generate_synthetic_fracture_rock(
            width=500, height=300, fractures=[fr], seed=2,
        )
        assert int((gt_mask > 0).sum()) > 100

    def test_reproducibility(self):
        fr = SyntheticFracture(x1=100, y1=100, x2=300, y2=200, width_px=3)
        img1, m1 = generate_synthetic_fracture_rock(400, 300, [fr], seed=42)
        img2, m2 = generate_synthetic_fracture_rock(400, 300, [fr], seed=42)
        assert np.array_equal(img1, img2)
        assert np.array_equal(m1, m2)


# ============= 裂缝检测 =============
class TestDetectFractureMask:
    def test_detect_from_synthetic(self):
        fractures, image, gt_mask = make_default_synthetic_fracture()
        pred_mask, edges, lines = detect_fracture_mask(image)
        assert pred_mask.shape == image.shape[:2]
        assert pred_mask.dtype == np.uint8
        assert edges.shape == image.shape[:2]
        assert int((pred_mask > 0).sum()) > 0
        assert isinstance(lines, np.ndarray)
        assert lines.shape[0] > 0  # 合成图应检出大量线段

    def test_detect_grayscale_input(self):
        fractures, image, gt_mask = make_default_synthetic_fracture()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pred_mask, _, _ = detect_fracture_mask(gray)
        assert pred_mask.shape == gray.shape
        assert int((pred_mask > 0).sum()) > 0

    def test_detect_blank_image(self):
        blank = np.full((300, 400, 3), 200, dtype=np.uint8)
        pred_mask, _, lines = detect_fracture_mask(blank)
        # 空白图应检出 0 条线段(无边缘)
        assert isinstance(lines, np.ndarray)
        assert lines.shape[0] == 0
        assert int((pred_mask > 0).sum()) == 0


# ============= 裂缝分析 =============
class TestAnalyzeFractures:
    def _make_test_mask(self):
        """构造测试用裂缝 mask(2条水平裂缝 + 1 条垂直裂缝)."""
        mask = np.zeros((200, 300), dtype=np.uint8)
        cv2.line(mask, (30, 50), (150, 50), 255, 3)
        cv2.line(mask, (170, 80), (270, 80), 255, 3)
        cv2.line(mask, (100, 120), (100, 180), 255, 3)
        return mask

    def test_analyze_count(self):
        mask = self._make_test_mask()
        scale = scale_from_dpi(96)
        result = analyze_fractures(mask, scale)
        assert result.fracture_count == 3

    def test_analyze_dimensions(self):
        mask = self._make_test_mask()
        scale = scale_from_dpi(96)
        result = analyze_fractures(mask, scale)
        # 第一条裂缝长度应约 120 px = 120/96 * 25.4 ≈ 31.75 mm
        fr1 = result.fractures[0]
        assert 28 < fr1.length_real < 33  # mm (允许骨架化 5-10% 误差)
        # 宽度 ≈ 3-5 px = 0.8-1.4 mm (含抗锯齿)
        assert 0.6 < fr1.width_real < 1.6

    def test_analyze_density(self):
        mask = self._make_test_mask()
        scale = scale_from_dpi(96)
        result = analyze_fractures(mask, scale)
        assert result.areal_density > 0  # 面密度
        assert result.linear_density >= 0

    def test_analyze_empty_mask(self):
        empty = np.zeros((100, 100), dtype=np.uint8)
        scale = scale_from_dpi(96)
        result = analyze_fractures(empty, scale)
        assert result.fracture_count == 0
        assert result.total_length_real == 0.0
        assert result.areal_density == 0.0

    def test_analyze_too_short_filtered(self):
        """过短的裂缝应被过滤."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(mask, (10, 10), (13, 10), 255, 1)  # 长度 3px < min
        scale = scale_from_dpi(96)
        result = analyze_fractures(mask, scale, min_length_px=15)
        assert result.fracture_count == 0

    def test_analyze_orientation(self):
        mask = self._make_test_mask()
        scale = scale_from_dpi(96)
        result = analyze_fractures(mask, scale)
        # 第 3 条裂缝(垂直)倾角应接近 90°;水平接近 0° 或 180°
        vertical = [f for f in result.fractures if f.bbox[2] < f.bbox[3]]
        assert len(vertical) >= 1
        v = vertical[0]
        # 垂直线的 PCA 主方向应接近 90°(±20°)
        assert 70 < v.orientation_deg < 110 or (v.orientation_deg > 160 and v.orientation_deg < 200)


# ============= 像素级准确率 =============
class TestPixelMetrics:
    def test_perfect_overlap(self):
        m = np.zeros((100, 100), dtype=np.uint8)
        m[20:60, 30:70] = 255
        p, r, iou = compute_pixel_metrics(m, m)
        assert p == 1.0
        assert r == 1.0
        assert iou == 1.0

    def test_no_overlap(self):
        a = np.zeros((100, 100), dtype=np.uint8)
        b = np.zeros((100, 100), dtype=np.uint8)
        a[20:40, 20:40] = 255
        b[60:80, 60:80] = 255
        p, r, iou = compute_pixel_metrics(a, b)
        assert p == 0.0
        assert r == 0.0
        assert iou == 0.0

    def test_partial_overlap(self):
        a = np.zeros((100, 100), dtype=np.uint8)
        a[20:60, 20:60] = 255
        b = np.zeros((100, 100), dtype=np.uint8)
        b[40:80, 40:80] = 255
        p, r, iou = compute_pixel_metrics(a, b)
        assert 0 < p < 1
        assert 0 < r < 1
        assert 0 < iou < 1


# ============= 端到端准确率评估 =============
class TestEvaluateFractureAccuracy:
    def test_default_synthetic_passes_target(self):
        fractures, image, gt_mask = make_default_synthetic_fracture()
        scale = scale_from_dpi(96)
        report = evaluate_fracture_accuracy(image, gt_mask, scale, distance_threshold_px=50)
        # 调优后默认参数应达到 ≥ 80% 综合准确率
        assert report.composite_score >= 0.80, \
            f"composite={report.composite_score} 未达到 80% 目标"
        assert report.detection_f1 >= 0.85, \
            f"F1={report.detection_f1} 过低"
        assert report.matched >= 5  # 至少匹配 5 条

    def test_evaluation_report_fields(self):
        fractures, image, gt_mask = make_default_synthetic_fracture()
        scale = scale_from_dpi(96)
        report = evaluate_fracture_accuracy(image, gt_mask, scale)
        assert hasattr(report, "pixel_iou")
        assert hasattr(report, "detection_f1")
        assert hasattr(report, "matched")
        assert hasattr(report, "fracture_count_gt")
        assert report.fracture_count_gt > 0

    def test_to_dict_serialization(self):
        fractures, image, gt_mask = make_default_synthetic_fracture()
        scale = scale_from_dpi(96)
        report = evaluate_fracture_accuracy(image, gt_mask, scale)
        d = report.to_dict()
        assert "pixel_iou" in d
        assert "composite_score" in d
        assert "passes_target_0.80" in d
        assert isinstance(d["matches"], list)


# ============= 注释绘制 =============
class TestDrawFractureAnnotations:
    def test_annotations_creates_image(self):
        fractures, image, gt_mask = make_default_synthetic_fracture()
        scale = scale_from_dpi(96)
        from rockpore.core.fracture import analyze_fractures
        mask, _, _ = detect_fracture_mask(image)
        result = analyze_fractures(mask, scale)
        annotated = draw_fracture_annotations(image, result.fractures)
        assert annotated.shape == image.shape
        assert not np.array_equal(annotated, image)  # 应有标注差异

    def test_annotations_grayscale_input(self):
        fractures, image, gt_mask = make_default_synthetic_fracture()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        annotated = draw_fracture_annotations(gray, [])
        assert annotated.ndim == 3
        assert annotated.shape == image.shape


# ============= 集成测试:与孔洞分析协同 =============
class TestFractureReportFields:
    def test_report_includes_density(self):
        fractures, image, gt_mask = make_default_synthetic_fracture()
        scale = scale_from_dpi(96)
        mask, _, _ = detect_fracture_mask(image)
        result = analyze_fractures(mask, scale)
        # 验证报告完整字段
        assert result.image_area_real > 0
        assert result.total_length_real > 0
        assert result.width_distribution  # 大缝/中缝/小缝
        assert result.orientation_histogram  # 倾角 8 区间

    def test_size_class_distribution(self):
        fractures, image, gt_mask = make_default_synthetic_fracture()
        scale = scale_from_dpi(96)
        mask, _, _ = detect_fracture_mask(image)
        result = analyze_fractures(mask, scale)
        # 大缝、中缝、小缝之和应等于裂缝总数
        total = sum(result.width_distribution.values())
        assert total == result.fracture_count


# ============= 真实岩石图测试 =============
class TestRealRockImage:
    """测试真实岩石图(用户提供的 裂缝样.jpg).

    HoughLinesP 在纹理复杂图像上失效,adaptive 算法应能识别主裂缝.
    """

    @pytest.fixture
    def real_image_path(self):
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        path = project_root / "裂缝样.jpg"
        if not path.exists():
            # 跳过真实图测试(开发环境可能没有)
            pytest.skip(f"真实裂绊图不存在: {path}")
        return str(path)

    def test_hough_fails_on_real_image(self, real_image_path):
        """Hough 算法在真实裂绊图上会产生大量假阳性(预期失败)."""
        from rockpore.core.io_utils import imread_unicode
        img = imread_unicode(real_image_path)
        params = FractureParams(method="hough")
        mask, _, lines = detect_fracture_mask(img, params)
        # Hough 在真实图上覆盖 >50%(假阳性)
        coverage = (mask > 0).mean()
        assert coverage > 0.50, "Hough 在真实图上理应失败"

    def test_adaptive_works_on_real_image(self, real_image_path):
        """Adaptive 算法应能识别真实裂绊图的主裂绊."""
        from rockpore.core.io_utils import imread_unicode
        img = imread_unicode(real_image_path)
        params = FractureParams(method="adaptive")
        mask, _, _ = detect_fracture_mask(img, params)
        result = analyze_fractures(mask, scale_from_dpi(96))
        # Adaptive 应识别 ≥ 2 条裂缝(包括主裂缝)
        assert result.fracture_count >= 2, \
            f"adaptive 只识别 {result.fracture_count} 条裂缝, 应至少 2 条"
        # 覆盖比例应 < 50%(避免覆盖全图)
        coverage = (mask > 0).mean()
        assert coverage < 0.50, f"adaptive 覆盖率 {coverage*100:.1f}% 过高"

    def test_adaptive_finds_main_fracture(self, real_image_path):
        """Adaptive 算法应识别中央垂直裂绊(L≥10mm, 倾角接近垂直)."""
        from rockpore.core.io_utils import imread_unicode
        img = imread_unicode(real_image_path)
        params = FractureParams(method="adaptive")
        mask, _, _ = detect_fracture_mask(img, params)
        result = analyze_fractures(mask, scale_from_dpi(96))
        # 找主裂绊(长度最大的)
        main = max(result.fractures, key=lambda f: f.length_real)
        # 主裂绊应 ≥ 10mm 长
        assert main.length_real >= 10.0, \
            f"主裂绊长度 {main.length_real:.1f}mm 过短"


# ============= 算法对比测试 =============
class TestAlgorithmComparison:
    """对比 Hough 与 Adaptive 算法的差异."""

    def test_both_methods_return_same_interface(self):
        """两种方法返回相同的 (mask, edges, segments) 三元组."""
        fractures, image, gt_mask = make_default_synthetic_fracture()
        hough_mask, hough_edges, hough_segs = detect_fracture_mask(image, FractureParams(method="hough"))
        adapt_mask, adapt_edges, adapt_segs = detect_fracture_mask(image, FractureParams(method="adaptive"))
        # 类型相同
        assert isinstance(hough_mask, np.ndarray) and isinstance(adapt_mask, np.ndarray)
        assert isinstance(hough_edges, np.ndarray) and isinstance(adapt_edges, np.ndarray)
        # shape 相同
        assert hough_mask.shape == adapt_mask.shape
        assert hough_edges.shape == adapt_edges.shape

    def test_adaptive_candidates_have_aspect_ratio(self):
        """Adaptive 算法候选应满足长宽比约束."""
        fractures, image, gt_mask = make_default_synthetic_fracture()
        params = FractureParams(method="adaptive", min_aspect_ratio=2.0)
        mask, _, _ = detect_fracture_mask(image, params)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            pytest.skip("无候选")
        for i in range(1, n):
            ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(ww, hh) / max(min(ww, hh), 1)
            assert aspect >= 2.0, f"候选 #{i} 长宽比 {aspect:.1f} < 2.0"
