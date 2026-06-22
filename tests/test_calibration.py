"""标尺模块单元测试."""

import math
import pytest

from rockpore.core.calibration import (
    Scale, ScaleUnit, pixel_to_mm, mm_to_pixel, scale_from_dpi,
    get_image_dpi,
)


class TestScale:
    def test_mm_per_pixel_96dpi(self):
        s = Scale(pixels_per_unit=96/25.4, unit=ScaleUnit.MILLIMETER, dpi=96)
        # 1 inch = 25.4 mm = 96 px → 1 px = 25.4/96 mm
        expected = 25.4 / 96
        assert abs(s.mm_per_pixel - expected) < 1e-6

    def test_pixels_to_real(self):
        s = scale_from_dpi(96, microscopic=False)
        # 96 px @96dpi = 1 inch = 25.4 mm
        assert abs(s.pixels_to_real(96) - 25.4) < 1e-6

    def test_real_to_pixels(self):
        s = scale_from_dpi(96, microscopic=False)
        assert abs(s.real_to_pixels(25.4) - 96) < 1e-6

    def test_area_conversion(self):
        s = scale_from_dpi(96, microscopic=False)
        # 1 mm² = (96/25.4)² 像素
        expected_px_area = (96 / 25.4) ** 2
        assert abs(s.area_pixels_to_real(expected_px_area) - 1.0) < 1e-4

    def test_microscopic_scale(self):
        s = scale_from_dpi(96, microscopic=True)
        assert s.unit == ScaleUnit.MICROMETER
        # 96 px @96dpi = 1 inch = 25.4 mm = 25400 μm
        assert abs(s.pixels_to_real(96) - 25400) < 1e-6


class TestConvenienceFunctions:
    def test_pixel_to_mm_96dpi(self):
        assert abs(pixel_to_mm(96, 96) - 25.4) < 1e-6

    def test_pixel_to_mm_300dpi(self):
        # 300 px @ 300 dpi = 1 inch = 25.4 mm
        assert abs(pixel_to_mm(300, 300) - 25.4) < 1e-6

    def test_mm_to_pixel(self):
        assert abs(mm_to_pixel(25.4, 96) - 96) < 1e-6

    def test_round_trip(self):
        for px in [10, 100, 1000, 0.5]:
            mm = pixel_to_mm(px, 96)
            assert abs(mm_to_pixel(mm, 96) - px) < 1e-6


class TestGetImageDpi:
    def test_existing_jpg(self, tmp_path):
        import cv2
        import numpy as np
        # 写一张 100x100 灰度图,DPI 默认 96
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        path = str(tmp_path / "test.jpg")
        cv2.imwrite(path, img)
        dpi = get_image_dpi(path)
        # JPEG 写入会保留 DPI 信息(虽然有时归一化)
        assert dpi > 0

    def test_nonexistent_file(self):
        # 不存在文件应返回默认 96
        assert get_image_dpi("/nonexistent/path.jpg") == 96

    def test_bmp_default(self, tmp_path):
        import cv2
        import numpy as np
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        path = str(tmp_path / "test.bmp")
        cv2.imwrite(path, img)
        # BMP 通常不存 DPI,fallback 到 96
        assert get_image_dpi(path) == 96


class TestScaleFromDpi:
    def test_96dpi_macro(self):
        s = scale_from_dpi(96, microscopic=False)
        assert s.dpi == 96
        assert s.unit == ScaleUnit.MILLIMETER

    def test_96dpi_micro(self):
        s = scale_from_dpi(96, microscopic=True)
        assert s.dpi == 96
        assert s.unit == ScaleUnit.MICROMETER
