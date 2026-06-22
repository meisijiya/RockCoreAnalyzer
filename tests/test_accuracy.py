"""准确率与端到端测试.

使用合成图作为 ground truth,验证算法识别准确率 ≥ 80%.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rockpore.core.calibration import scale_from_dpi
from rockpore.core.accuracy import (
    evaluate_accuracy, detect_pores_robust,
    compute_pixel_metrics, match_pores, _compute_pred_individual_masks,
)
from rockpore.core.analysis import analyze_pores
from rockpore.core.synthetic import (
    generate_synthetic_rock, SyntheticPore, make_default_synthetic,
)


class TestPixelMetrics:
    def test_perfect_match(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        p, r, iou = compute_pixel_metrics(mask, mask)
        assert p == 1.0 and r == 1.0 and iou == 1.0

    def test_no_overlap(self):
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[0:10, 0:10] = 255
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[50:60, 50:60] = 255
        p, r, iou = compute_pixel_metrics(pred, gt)
        assert p == 0.0 and r == 0.0 and iou == 0.0

    def test_partial_overlap(self):
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[20:40, 20:40] = 255  # 400 px
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[30:50, 30:50] = 255  # 400 px
        p, r, iou = compute_pixel_metrics(pred, gt)
        # 交集 10x10=100, 并集 700
        assert abs(p - 0.25) < 0.01
        assert abs(r - 0.25) < 0.01
        assert abs(iou - 100/700) < 0.01


class TestSyntheticImage:
    def test_default_synthetic_generation(self):
        pores, img, gt = make_default_synthetic()
        assert len(pores) == 15
        assert img.shape[:2] == (400, 600)
        assert gt.shape == (400, 600)
        # ground truth 应有 15 个连通域
        n, _, _, _ = cv2.connectedComponentsWithStats(gt, 8)
        assert n - 1 == 15

    def test_default_synthetic_accuracy(self):
        """主测试: 验证算法在标准合成图上达到 ≥80% 准确率."""
        pores, img, gt = make_default_synthetic()
        scale = scale_from_dpi(96)
        report = evaluate_accuracy(img, gt, scale)
        # 打印详情便于调试
        print(f"\n合成图准确率报告:")
        print(f"  Pixel IoU: {report.pixel_iou * 100:.2f}%")
        print(f"  Detection F1: {report.detection_f1 * 100:.2f}%")
        print(f"  Diameter MAE: {report.diameter_mae:.3f} mm")
        print(f"  Composite: {report.composite_score}")
        # 主断言:综合得分 ≥ 0.80
        assert report.composite_score >= 0.80, (
            f"识别准确率 {report.composite_score} 未达 80% 目标"
        )

    def test_multi_difficulty_accuracy(self):
        """多难度集测试."""
        from tests.multi_accuracy import make_easy_set, make_hard_set, make_dense_set
        scale = scale_from_dpi(96)
        all_pass = True
        results = {}
        for name, (img, gt) in [("easy", make_easy_set()), ("hard", make_hard_set()),
                                 ("dense", make_dense_set())]:
            report = evaluate_accuracy(img, gt, scale)
            results[name] = report.composite_score
            print(f"  {name}: composite = {report.composite_score}")
            if report.composite_score < 0.80:
                all_pass = False
        # 至少 2/3 通过
        passing = sum(1 for s in results.values() if s >= 0.80)
        assert passing >= 2, f"多难度集通过 {passing}/3,失败: {results}"


class TestRealImage:
    def test_lf_runs(self):
        """真实岩心图能跑通(不评估准确率,仅功能验证)."""
        path = Path(__file__).resolve().parent.parent / "lf.jpg"
        if not path.exists():
            pytest.skip("lf.jpg 不存在,跳过真实图测试")
        img = cv2.imread(str(path))
        assert img is not None
        scale = scale_from_dpi(96)
        mask, result = detect_pores_robust(img, scale, min_diameter_real=1.0)
        assert result.pore_count > 0
        # 验证字段有效性
        assert result.porosity >= 0
        assert result.average_diameter_real >= 0
        # 至少有 1 个达到报告级
        assert result.pore_count_report >= 1


class TestReport:
    def test_html_generation(self):
        from rockpore.core.report import ReportData, generate_report
        _, img, _ = make_default_synthetic()
        scale = scale_from_dpi(96)
        mask, result = detect_pores_robust(img, scale)
        data = ReportData(
            project_name="测试项目",
            sample_id="S001",
            analyst="测试员",
            image_path="test.png",
            image_size=(img.shape[1], img.shape[0]),
            scale_info=f"{scale.pixels_per_unit:.3f} px/mm",
            analysis_result=result,
            original_image=img,
            annotated_image=img,
        )
        html = generate_report(data)
        assert "<html" in html
        assert "岩心孔洞分析报告" in html
        assert "测试项目" in html
        assert "S001" in html
        # 应包含所有关键字段
        assert "孔洞总个数" in html
        assert "面孔率" in html
        assert "平均等效直径" in html
        assert "直径频率分布" in html
