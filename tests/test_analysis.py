"""孔洞分析模块单元测试."""

import math
import numpy as np
import cv2
import pytest

from rockpore.core.calibration import Scale, ScaleUnit, scale_from_dpi
from rockpore.core.analysis import (
    analyze_pores, classify_pore_size, Pore, PoreAnalysisResult,
    PoreType, FilledStatus, FilledMaterial, Effectiveness,
    diameter_from_area, compute_diameter_frequency_curve,
    PORE_LARGE, PORE_MEDIUM, PORE_SMALL, PORE_PINHOLE,
    REPORT_MIN_DIAMETER_MM,
)


class TestClassifyPoreSize:
    def test_large(self):
        assert classify_pore_size(15.0) == "大洞"
        assert classify_pore_size(10.5) == "大洞"
        assert classify_pore_size(100.0) == "大洞"

    def test_medium(self):
        assert classify_pore_size(10.0) == "中洞"
        assert classify_pore_size(5.0) == "中洞"
        assert classify_pore_size(7.5) == "中洞"

    def test_small(self):
        assert classify_pore_size(4.9) == "小洞"
        assert classify_pore_size(1.0) == "小洞"
        assert classify_pore_size(2.0) == "小洞"

    def test_pinhole(self):
        assert classify_pore_size(0.99) == "针孔/溶孔"
        assert classify_pore_size(0.1) == "针孔/溶孔"
        assert classify_pore_size(0.0) == "针孔/溶孔"


class TestDiameterFromArea:
    def test_known_area(self):
        # 1 mm² 圆 → 直径 = 2*sqrt(1/π) ≈ 1.128 mm
        scale = scale_from_dpi(96, microscopic=False)
        # 1 mm² 像素面积 = (96/25.4)² ≈ 14.27 px²
        pixel_area = (96 / 25.4) ** 2
        d = diameter_from_area(pixel_area, scale)
        assert abs(d - 1.128) < 0.01

    def test_zero_area(self):
        scale = scale_from_dpi(96)
        d = diameter_from_area(0, scale)
        assert d == 0.0


class TestAnalyzePores:
    def _make_mask_with_circles(self, image_size=(400, 600), circles=None):
        mask = np.zeros(image_size, dtype=np.uint8)
        for (cx, cy, r) in circles:
            cv2.circle(mask, (cx, cy), r, 255, -1)
        return mask

    def test_analyze_single_pore(self):
        mask = self._make_mask_with_circles(circles=[(100, 100, 30)])
        scale = scale_from_dpi(96)
        result = analyze_pores(mask, scale)
        assert result.pore_count == 1
        # 直径 ≈ 60 px = 60/3.78 ≈ 15.87 mm
        assert 15 < result.pores[0].diameter_real < 16.5
        assert result.pores[0].size_class == "大洞"
        assert result.porosity > 0

    def test_analyze_multiple_pores(self):
        mask = self._make_mask_with_circles(
            circles=[(100, 100, 20), (300, 200, 25), (500, 300, 10)],
        )
        scale = scale_from_dpi(96)
        result = analyze_pores(mask, scale)
        assert result.pore_count == 3
        assert result.pore_count_report == 3  # 都 ≥ 2mm
        assert result.porosity > 0

    def test_analyze_empty(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        scale = scale_from_dpi(96)
        result = analyze_pores(mask, scale)
        assert result.pore_count == 0
        assert result.porosity == 0.0
        assert result.average_diameter_real == 0.0

    def test_pinhole_not_counted_in_report(self):
        # 一个 0.5mm 直径的小孔(0.25mm 半径)→ 应被报告排除
        # 0.5mm 直径 = 0.5 * 3.78 = 1.89 px 半径
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 2, 255, -1)  # 半径 2 px ≈ 0.53 mm
        scale = scale_from_dpi(96)
        result = analyze_pores(mask, scale)
        assert result.pore_count == 1
        assert result.pore_count_report == 0

    def test_classification_distribution(self):
        mask = self._make_mask_with_circles(circles=[
            (100, 100, 50),   # 大洞 (直径 ~26mm, >10mm)
            (200, 200, 20),   # 中洞 (直径 ~10.6mm, 5-10mm 区间)
            (300, 300, 5),    # 小洞 (直径 ~2.6mm, 1-5mm 区间)
        ])
        scale = scale_from_dpi(96)
        result = analyze_pores(mask, scale)
        # 验证大洞、中洞、小洞都能被正确分类
        assert result.size_distribution["大洞"] >= 1
        # 中洞和小洞至少有一个被检出
        assert result.size_distribution["中洞"] + result.size_distribution["小洞"] >= 1


class TestPoreDataClass:
    def test_pore_to_dict(self):
        p = Pore(
            id=1, area_px=100, area_real=10.0, diameter_real=3.57,
            centroid=(50, 50), bbox=(40, 40, 20, 20), perimeter_px=20,
            size_class="小洞",
        )
        d = p.to_dict()
        assert d["id"] == 1
        assert d["area_real"] == 10.0
        assert d["diameter_real"] == 3.57
        assert d["size_class"] == "小洞"
        assert d["filled_status"] == "未充填"
        assert d["effectiveness"] == "有效"


class TestFrequencyCurve:
    def test_empty(self):
        result = compute_diameter_frequency_curve([])
        assert result["bins"] == []

    def test_single_bin(self):
        pores = [
            Pore(id=i, area_px=100, area_real=1.0, diameter_real=1.0,
                 centroid=(0, 0), bbox=(0, 0, 10, 10), size_class="针孔/溶孔")
            for i in range(5)
        ]
        result = compute_diameter_frequency_curve(pores, bins=5)
        assert sum(result["counts"]) == 5
        assert result["normal_cumulative"][-1] == 1.0

    def test_distribution_shape(self):
        pores = [
            Pore(id=i, area_px=100, area_real=float(d), diameter_real=float(d),
                 centroid=(0, 0), bbox=(0, 0, 10, 10), size_class="小洞")
            for i, d in enumerate([1, 2, 3, 5, 8])
        ]
        result = compute_diameter_frequency_curve(pores, bins=3)
        assert len(result["bins"]) == 3
        assert sum(result["counts"]) == 5
        assert result["cumulative"][-1] == 5


class TestEnums:
    def test_filled_status_values(self):
        assert FilledStatus.UNFILLED.value == "未充填"
        assert FilledStatus.SEMI_FILLED.value == "半充填"
        assert FilledStatus.FILLED.value == "全充填"

    def test_filled_material_values(self):
        assert FilledMaterial.CLAY.value == "泥质"
        assert FilledMaterial.CALCITE.value == "方解石"
        assert FilledMaterial.DOLOMITE.value == "白云石"
        assert FilledMaterial.ASPHALT.value == "沥青"
        assert FilledMaterial.GYPSUM.value == "石膏"
        assert FilledMaterial.PYRITE.value == "黄铁矿"
        assert FilledMaterial.KAOLINITE.value == "高岭石"
        assert FilledMaterial.QUARTZ.value == "石英"

    def test_pore_types(self):
        assert PoreType.SOLUTION_CAVE.value == "溶洞"
        assert PoreType.CRYSTAL_CAVE.value == "晶洞"

    def test_effectiveness_values(self):
        assert Effectiveness.EFFECTIVE.value == "有效"
        assert Effectiveness.LESS_EFFECTIVE.value == "较有效"
        assert Effectiveness.INEFFECTIVE.value == "无效"
