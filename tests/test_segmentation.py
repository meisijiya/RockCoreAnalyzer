"""分割模块单元测试."""

import numpy as np
import cv2
import pytest

from rockpore.core.segmentation import (
    SegmentationParams, segment_by_color, flood_fill_region,
    extract_pores, invert_mask, remove_small_components,
    keep_largest_components,
)


class TestSegmentByColor:
    def test_extract_black(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[10:20, 10:20] = (0, 0, 0)
        img[30:40, 30:40] = (255, 255, 255)
        params = SegmentationParams(color=(0, 0, 0), tolerance=10)
        mask = segment_by_color(img, params)
        assert mask.shape == img.shape[:2]
        assert (mask[10:20, 10:20] == 255).all()
        assert (mask[30:40, 30:40] == 0).all()

    def test_extract_with_tolerance(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        # 创建从黑到灰的渐变
        for i in range(50):
            img[i, :] = i * 5
        params = SegmentationParams(color=(0, 0, 0), tolerance=50)
        mask = segment_by_color(img, params)
        # 容差 50 应匹配较暗的像素
        assert (mask > 0).sum() > 0

    def test_grayscale(self):
        img = np.zeros((30, 30), dtype=np.uint8)
        img[5:10, 5:10] = 0
        img[15:20, 15:20] = 200
        params = SegmentationParams(color=(0, 0, 0), tolerance=10)
        mask = segment_by_color(img, params)
        assert (mask[5:10, 5:10] == 255).all()


class TestExtractPores:
    def test_otsu(self):
        # 50% 暗色块,OTSU 应明确分离
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        img[20:80, 20:80] = (10, 10, 10)  # 60% 暗,40% 亮
        mask = extract_pores(img, method="otsu")
        assert (mask[20:80, 20:80] == 255).all()

    def test_adaptive(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = extract_pores(img, method="adaptive")
        assert mask.shape == img.shape[:2]

    def test_auto_alias(self):
        img = np.full((50, 50, 3), 200, dtype=np.uint8)
        img[10:40, 10:40] = (5, 5, 5)  # 60% 暗块
        mask = extract_pores(img, method="auto")
        assert (mask[10:40, 10:40] == 255).all()

    def test_invalid_method(self):
        img = np.zeros((30, 30, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_pores(img, method="invalid")


class TestFloodFill:
    def test_continuous_region(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        params = SegmentationParams(color=(0, 0, 0), tolerance=10, continuous=True)
        # 内部加 seed 属性
        params._seed = (25, 25)
        mask = flood_fill_region(img, (25, 25), params)
        # 全黑图像,应能填出整个连通区域
        assert (mask == 255).sum() > 0

    def test_out_of_bounds(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        params = SegmentationParams(color=(0, 0, 0), tolerance=10)
        params._seed = (100, 100)
        mask = flood_fill_region(img, (100, 100), params)
        assert (mask == 0).all()


class TestMaskUtils:
    def test_invert(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        inv = invert_mask(mask)
        assert (inv[2:5, 2:5] == 0).all()
        assert (inv[0:2, 0:2] == 255).all()

    def test_remove_small(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255  # 大区域 900 px
        mask[0:3, 0:3] = 255  # 小区域 9 px
        out = remove_small_components(mask, min_area=20)
        assert (out[10:40, 10:40] == 255).all()
        assert (out[0:3, 0:3] == 0).all()

    def test_keep_largest(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[0:20, 0:20] = 255  # 区域 1: 400 px
        mask[30:35, 30:35] = 255  # 区域 2: 25 px
        out = keep_largest_components(mask, top_k=1)
        assert (out[0:20, 0:20] == 255).all()
        assert (out[30:35, 30:35] == 0).all()
