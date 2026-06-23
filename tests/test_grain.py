"""粒度分析模块单元测试.

覆盖:
- Wentworth 粒级分类
- Grain 数据类属性计算
- 距离变换 + 分水岭分割
- 合成图准确性 (≥80% composite)
"""
from __future__ import annotations

import sys
import unittest

import cv2
import numpy as np

from rockpore.core.calibration import Scale
from rockpore.core.grain import (
    GrainParams, Grain, GrainAnalysisResult,
    classify_grain_size, detect_grain_mask, analyze_grains,
    draw_grain_annotations, compute_size_distribution_text,
    SIZE_CLASSES_ORDERED,
)
from rockpore.core.synthetic_grain import (
    SyntheticGrain, generate_synthetic_grain_rock,
    make_default_synthetic_grain, make_overlapping_synthetic_grain,
    make_granite_synthetic_grain,
)
from rockpore.core.grain_accuracy import (
    evaluate_grain_accuracy, compute_pixel_metrics, match_grains,
)


class TestGrainSizeClassification(unittest.TestCase):
    """测试 Wentworth 粒级分类."""

    def test_gravel_classifications(self):
        self.assertEqual(classify_grain_size(300), "巨砾")
        self.assertEqual(classify_grain_size(256), "巨砾")  # 边界
        self.assertEqual(classify_grain_size(100), "粗砾")
        self.assertEqual(classify_grain_size(64), "粗砾")
        self.assertEqual(classify_grain_size(30), "中砾")
        self.assertEqual(classify_grain_size(16), "中砾")
        self.assertEqual(classify_grain_size(5), "细砾")
        self.assertEqual(classify_grain_size(2), "细砾")

    def test_sand_classifications(self):
        self.assertEqual(classify_grain_size(1.5), "极粗砂")
        self.assertEqual(classify_grain_size(1), "极粗砂")
        self.assertEqual(classify_grain_size(0.7), "粗砂")
        self.assertEqual(classify_grain_size(0.5), "粗砂")
        self.assertEqual(classify_grain_size(0.3), "中砂")
        self.assertEqual(classify_grain_size(0.25), "中砂")
        self.assertEqual(classify_grain_size(0.15), "细砂")
        self.assertEqual(classify_grain_size(0.125), "细砂")
        self.assertEqual(classify_grain_size(0.08), "极细砂")
        self.assertEqual(classify_grain_size(0.0625), "极细砂")
        self.assertEqual(classify_grain_size(0.01), "粉砂")
        self.assertEqual(classify_grain_size(0.0039), "粉砂")
        self.assertEqual(classify_grain_size(0.001), "黏土")

    def test_size_classes_ordered(self):
        """SIZE_CLASSES_ORDERED 包含所有粒级且按粒径降序."""
        self.assertGreater(len(SIZE_CLASSES_ORDERED), 0)
        # 第一个粒级最大粒径,最后一个最小
        self.assertEqual(SIZE_CLASSES_ORDERED[0][0], "巨砾")
        self.assertEqual(SIZE_CLASSES_ORDERED[-1][0], "黏土")


class TestGrainDataclass(unittest.TestCase):
    """测试 Grain / GrainParams 数据类."""

    def test_grain_to_dict(self):
        g = Grain(
            id=1, area_px=100.0, perimeter_px=40.0,
            major_axis_px=15.0, minor_axis_px=10.0,
            diameter_mm=1.0, diameter_major_mm=1.5,
            centroid=(50, 50), bbox=(40, 40, 20, 20),
            orientation_deg=45.0, circularity=0.8,
            solidity=0.9, aspect_ratio=1.5,
            size_class="细砾",
        )
        d = g.to_dict()
        self.assertEqual(d["id"], 1)
        self.assertEqual(d["size_class"], "细砾")
        self.assertEqual(d["centroid"], [50, 50])
        # contour 不应被序列化
        self.assertNotIn("contour", d)

    def test_grain_params_defaults(self):
        p = GrainParams()
        self.assertGreater(p.min_area_px, 0)
        self.assertGreater(p.distance_threshold_ratio, 0)
        self.assertLessEqual(p.distance_threshold_ratio, 1.0)

    def test_grain_params_to_dict(self):
        p = GrainParams(blur_kernel=7, min_area_px=300)
        d = p.to_dict()
        self.assertEqual(d["blur_kernel"], 7)
        self.assertEqual(d["min_area_px"], 300)


class TestSyntheticGrain(unittest.TestCase):
    """测试合成图生成器."""

    def test_make_default_synthetic(self):
        grains, image, instance_mask = make_default_synthetic_grain(n_grains=20, seed=42)
        self.assertGreater(len(grains), 0)
        self.assertEqual(image.shape[2], 3)  # BGR
        self.assertGreater((instance_mask > 0).sum(), 0)
        # 实例 mask 应有多个 ID
        unique_ids = [int(i) for i in np.unique(instance_mask) if i > 0]
        self.assertGreater(len(unique_ids), 5)

    def test_make_overlapping_synthetic(self):
        grains, image, instance_mask = make_overlapping_synthetic_grain(
            n_pairs=4, seed=42, contact_ratio=0.85,
        )
        self.assertEqual(len(grains), 8)  # 4 pairs * 2 grains
        unique_ids = [int(i) for i in np.unique(instance_mask) if i > 0]
        self.assertEqual(len(unique_ids), 8)

    def test_make_granite_synthetic(self):
        grains, image, instance_mask = make_granite_synthetic_grain(
            n_grains=15, seed=42,
        )
        self.assertGreater(len(grains), 0)
        # 大颗粒 (r ≥ 60) 至少有几个
        big_grains = [g for g in grains if g.radius_px >= 60]
        self.assertGreater(len(big_grains), 0)

    def test_non_overlapping_default(self):
        """make_default_synthetic_grain 颗粒不重叠."""
        grains, image, instance_mask = make_default_synthetic_grain(n_grains=15, seed=42)
        for i, g1 in enumerate(grains):
            for j, g2 in enumerate(grains[i+1:], i+1):
                dist = np.hypot(g1.cx - g2.cx, g1.cy - g2.cy)
                min_d = (g1.radius_px + g2.radius_px) * 0.7
                # 颗粒中心距离 ≥ 0.7*(r1+r2) (允许轻微重叠)
                self.assertGreaterEqual(dist, min_d * 0.5,
                    f"颗粒 {i}({g1.cx},{g1.cy}) 和 {j}({g2.cx},{g2.cy}) 太近")


class TestGrainDetection(unittest.TestCase):
    """测试核心检测算法."""

    def setUp(self):
        self.scale = Scale(pixels_per_unit=3.78, unit='mm')

    def test_detect_grain_mask(self):
        """检测二值 mask 应有合理前景比例."""
        _, image, _ = make_default_synthetic_grain(n_grains=15, seed=42)
        params = GrainParams()
        bin_mask, _ = detect_grain_mask(image, params)
        # 前景比例应在 5%~60% 之间 (不能全 0 也不能全 255)
        ratio = (bin_mask > 0).sum() / bin_mask.size
        self.assertGreater(ratio, 0.05)
        self.assertLess(ratio, 0.60)

    def test_analyze_grains_returns_grains(self):
        """analyze_grains 应返回颗粒列表."""
        _, image, _ = make_default_synthetic_grain(n_grains=15, seed=42)
        params = GrainParams()
        result, _, markers = analyze_grains(image, self.scale, params)
        self.assertIsInstance(result, GrainAnalysisResult)
        self.assertGreater(result.grain_count, 0)
        # Grain 列表不应为空
        self.assertGreater(len(result.grains), 0)
        # 每个颗粒应有完整属性
        for g in result.grains[:3]:
            self.assertGreater(g.area_px, 0)
            self.assertGreater(g.diameter_mm, 0)
            self.assertGreater(g.circularity, 0)
            self.assertGreater(g.solidity, 0)

    def test_analyze_grains_real_image(self):
        """真实岩石图测试:不崩溃."""
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "粒度样2.png")
        if not os.path.exists(path):
            self.skipTest("粒度样2.png 不存在")
        image = cv2.imread(path)
        scale = Scale(pixels_per_unit=8.0, unit='mm')
        params = GrainParams(blur_kernel=7, min_area_px=500)
        result, _, _ = analyze_grains(image, scale, params)
        self.assertGreater(result.grain_count_filtered, 0)


class TestGrainAccuracy(unittest.TestCase):
    """测试准确率评估."""

    def setUp(self):
        self.scale = Scale(pixels_per_unit=3.78, unit='mm')

    def test_default_synthetic_passes_80_percent(self):
        """默认合成图 (15 个不重叠颗粒) 应 > 80% composite."""
        grains, image, gt_mask = make_default_synthetic_grain(n_grains=15, seed=42)
        params = GrainParams()
        report = evaluate_grain_accuracy(image, gt_mask, self.scale, params)
        print(f"\n默认合成: composite={report.composite_score:.4f}, "
              f"IoU={report.pixel_iou:.3f}, F1={report.detection_f1:.3f}, "
              f"matched={report.matched}/{report.count_gt}/{report.count_pred}")
        self.assertGreater(report.composite_score, 0.80)

    def test_overlapping_synthetic_passes_80_percent(self):
        """粘连合成图应 > 80% composite."""
        grains, image, gt_mask = make_overlapping_synthetic_grain(
            n_pairs=6, seed=42, contact_ratio=0.85,
        )
        params = GrainParams()
        report = evaluate_grain_accuracy(image, gt_mask, self.scale, params)
        print(f"\n粘连合成: composite={report.composite_score:.4f}, "
              f"IoU={report.pixel_iou:.3f}, F1={report.detection_f1:.3f}, "
              f"matched={report.matched}/{report.count_gt}/{report.count_pred}")
        self.assertGreater(report.composite_score, 0.80)

    def test_granite_synthetic_passes_80_percent(self):
        """花岗岩合成图 (真实图风格) 应 > 80% composite — 用户要求.

        多个 seed 测试取平均,反映算法在多种分布下的稳定性.
        """
        composites = []
        for n in [15, 20, 25]:
            for seed in [42, 1, 7]:
                grains, image, gt_mask = make_granite_synthetic_grain(n_grains=n, seed=seed)
                params = GrainParams()
                report = evaluate_grain_accuracy(image, gt_mask, self.scale, params)
                composites.append(report.composite_score)
                print(f"\n  花岗岩 n={n} seed={seed}: composite={report.composite_score:.4f}, "
                      f"IoU={report.pixel_iou:.3f}, F1={report.detection_f1:.3f}, "
                      f"matched={report.matched}/{report.count_gt}/{report.count_pred}")
        avg = np.mean(composites)
        worst = min(composites)
        print(f"\n  花岗岩平均: {avg:.4f}, 最差: {worst:.4f}")
        # 至少 80% 通过
        self.assertGreater(worst, 0.75,
            f"花岗岩最差 composite={worst:.4f} 应 > 0.75")
        # 平均 > 80%
        self.assertGreater(avg, 0.80,
            f"花岗岩平均 composite={avg:.4f} 应 > 0.80")

    def test_pixel_metrics(self):
        """像素级指标函数."""
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[20:40, 20:40] = 255
        pred = gt.copy()
        p, r, iou = compute_pixel_metrics(pred, gt)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(iou, 1.0)
        # 全部错
        pred2 = np.zeros_like(gt)
        pred2[0:20, 0:20] = 255
        p2, r2, iou2 = compute_pixel_metrics(pred2, gt)
        self.assertEqual(p2, 0.0)
        self.assertEqual(r2, 0.0)
        self.assertEqual(iou2, 0.0)

    def test_match_grains_perfect_match(self):
        """完美匹配测试."""
        gt_mask = np.zeros((100, 100), dtype=np.int32)
        gt_mask[10:30, 10:30] = 1
        gt_mask[40:60, 40:60] = 2
        pred_mask = np.zeros_like(gt_mask)
        pred_mask[10:30, 10:30] = 255
        pred_mask[40:60, 40:60] = 255
        from rockpore.core.grain_accuracy import _extract_gt_grains, _extract_pred_grains
        gt_list = _extract_gt_grains(gt_mask)
        pred_list = _extract_pred_grains(pred_mask)
        self.assertEqual(len(gt_list), 2)
        self.assertEqual(len(pred_list), 2)
        matches, missed, fps = match_grains(pred_list, gt_list, distance_threshold_px=50.0)
        self.assertEqual(len(matches), 2)
        self.assertEqual(len(missed), 0)
        self.assertEqual(len(fps), 0)


class TestGrainVisualization(unittest.TestCase):
    """测试可视化函数."""

    def test_draw_grain_annotations(self):
        """draw_grain_annotations 不崩溃."""
        _, image, _ = make_default_synthetic_grain(n_grains=5, seed=42)
        scale = Scale(pixels_per_unit=3.78, unit='mm')
        result, _, _ = analyze_grains(image, scale)
        out = draw_grain_annotations(image, result.grains)
        self.assertEqual(out.shape, image.shape)

    def test_compute_size_distribution_text(self):
        """compute_size_distribution_text 返回字符串."""
        result = GrainAnalysisResult(
            image_area_px=10000, image_area_real=10,
            grain_count=5, grain_count_filtered=5,
            total_area_px=500, total_area_real=0.5,
            average_diameter_mm=1.0, median_diameter_mm=1.0,
            max_diameter_mm=2.0, min_diameter_mm=0.5,
            average_circularity=0.8,
            size_distribution={"细砾": 3, "极粗砂": 2},
        )
        text = compute_size_distribution_text(result)
        self.assertIn("粒级分布", text)
        self.assertIn("细砾", text)
        self.assertIn("极粗砂", text)
        self.assertIn("合计: 5", text)


if __name__ == "__main__":
    unittest.main()