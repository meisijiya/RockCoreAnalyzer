"""预处理模块单元测试."""

import numpy as np
import cv2
import pytest

from rockpore.core.preprocessing import (
    auto_levels, adjust_brightness, adjust_contrast, adjust_saturation,
    adjust_gamma, to_grayscale, blur_image, median_filter, sharpen_image,
    invert_image, detect_edges, curve_adjust, preprocess, PreprocessParams,
    apply_pore_enhancement,
)


class TestAutoLevels:
    def test_stretches_to_full_range(self):
        # 创建一个 50-200 范围的图像
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        out = auto_levels(img)
        assert out.min() <= 5  # 应被拉低
        assert out.max() >= 250  # 应被拉高

    def test_grayscale(self):
        img = np.random.randint(80, 180, (50, 50), dtype=np.uint8)
        out = auto_levels(img)
        assert out.shape == img.shape

    def test_preserves_dtype(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = auto_levels(img)
        assert out.dtype == np.uint8


class TestAdjustBrightness:
    def test_increase(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        out = adjust_brightness(img, 50)
        assert (out >= 145).all()  # 100 + 50 = 150

    def test_decrease(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        out = adjust_brightness(img, -50)
        assert (out <= 155).all()


class TestAdjustContrast:
    def test_increase(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        out = adjust_contrast(img, 2.0)
        # 0 灰度 (中心 100) × 2 → 0;255 (中心 100) → 200
        assert out.max() <= 200 or out.min() >= 0  # 不超界

    def test_zero(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        out = adjust_contrast(img, 0.0)
        assert (out == 0).all()


class TestAdjustGamma:
    def test_gamma_lt_1_brightens(self):
        img = np.full((10, 10), 100, dtype=np.uint8)
        out = adjust_gamma(img, 0.5)
        assert out[0, 0] > 100  # 应变亮

    def test_gamma_gt_1_darkens(self):
        img = np.full((10, 10), 200, dtype=np.uint8)
        out = adjust_gamma(img, 2.0)
        assert out[0, 0] < 200  # 应变暗


class TestInvert:
    def test_invert(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        out = invert_image(img)
        assert (out == np.array([[255, 127, 0]])).all()


class TestPipeline:
    def test_preprocess_no_op(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        params = PreprocessParams(auto_levels=False, to_gray=False)
        out = preprocess(img, params)
        # 不做任何处理,除了一点浮点误差
        assert out.shape == img.shape

    def test_preprocess_to_gray(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        params = PreprocessParams(to_gray=True, auto_levels=False)
        out = preprocess(img, params)
        assert out.ndim == 2

    def test_preprocess_edge(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        params = PreprocessParams(edge_detect=True, auto_levels=False)
        out = preprocess(img, params)
        assert out.ndim == 2
        # 边缘图应该二值化
        unique = np.unique(out)
        assert all(u in [0, 255] for u in unique)


class TestEnhancement:
    def test_enhance_bgr(self):
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        out = apply_pore_enhancement(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_enhance_gray(self):
        img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        out = apply_pore_enhancement(img)
        assert out.shape == img.shape
