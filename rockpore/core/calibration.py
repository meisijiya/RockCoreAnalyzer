"""标尺换算模块.

实现像素到毫米/微米的换算,支持宏观(毫米)与微观(微米)两种分析模式.
依据 PDF 1.1 节:宏观标尺以图像原始 DPI 计算,微观标尺由用户根据物镜设置.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np


class ScaleUnit(str, Enum):
    """标尺单位."""
    MILLIMETER = "mm"
    MICROMETER = "μm"


@dataclass(frozen=True)
class Scale:
    """标尺换算参数.
    Attributes:
        pixels_per_unit: 每单位的像素数(宏观分析时为像素/毫米,微观分析时为像素/微米)
        unit: 标尺单位
        dpi: 原始 DPI(可选)
    """
    pixels_per_unit: float
    unit: ScaleUnit
    dpi: int = 96

    @property
    def mm_per_pixel(self) -> float:
        """每像素对应的毫米数."""
        if self.unit == ScaleUnit.MILLIMETER:
            return 1.0 / self.pixels_per_unit
        return 1.0 / (self.pixels_per_unit * 1000.0)

    def pixels_to_real(self, pixels: float) -> float:
        """像素长度转实际长度."""
        return pixels / self.pixels_per_unit

    def real_to_pixels(self, real: float) -> float:
        """实际长度转像素长度."""
        return real * self.pixels_per_unit

    def area_pixels_to_real(self, area_px: float) -> float:
        """像素面积转实际面积(mm²)."""
        mm_per_px = self.mm_per_pixel
        return area_px * (mm_per_px ** 2)


@dataclass(frozen=True)
class Calibration:
    """标尺校准信息.
    包含标尺值与参考长度(像素),用于像素到实际尺寸的精确换算.
    """
    reference_pixels: float
    reference_real: float
    unit: ScaleUnit

    @property
    def pixels_per_unit(self) -> float:
        """每单位的像素数."""
        if self.reference_real <= 0:
            raise ValueError("参考长度必须大于0")
        return self.reference_pixels / self.reference_real

    def to_scale(self) -> Scale:
        """转换为 Scale 对象."""
        return Scale(pixels_per_unit=self.pixels_per_unit, unit=self.unit)


def pixel_to_mm(pixels: float, dpi: int = 96) -> float:
    """便捷函数:像素转毫米.
    依据 PDF 1.1 节宏观标尺定义,1 英寸 = 25.4 mm.
    """
    if dpi <= 0:
        raise ValueError("DPI 必须大于 0")
    inches = pixels / dpi
    return inches * 25.4


def mm_to_pixel(mm: float, dpi: int = 96) -> float:
    """便捷函数:毫米转像素."""
    if dpi <= 0:
        raise ValueError("DPI 必须大于 0")
    inches = mm / 25.4
    return inches * dpi


def scale_from_dpi(dpi: int, microscopic: bool = False) -> Scale:
    """根据 DPI 创建标尺.
    宏观分析(mm,基于图像 DPI),微观分析(μm,需另行校准).
    """
    if microscopic:
        return Scale(pixels_per_unit=dpi / 25.4 / 1000.0, unit=ScaleUnit.MICROMETER, dpi=dpi)
    return Scale(pixels_per_unit=dpi / 25.4, unit=ScaleUnit.MILLIMETER, dpi=dpi)


def get_image_dpi(image_path: str) -> int:
    """读取图像 DPI,失败时或 DPI=0 时返回默认 96.
    使用 PIL 读取,支持中文路径(因 PIL 内部用 UTF-8).
    """
    try:
        from PIL import Image
        # PIL 在 Windows 下处理中文路径可能失败,改用 bytes 方式
        with open(image_path, "rb") as f:
            data = f.read()
        from io import BytesIO
        with Image.open(BytesIO(data)) as img:
            dpi_info = img.info.get("dpi")
            if dpi_info:
                if isinstance(dpi_info, tuple):
                    dpi = int(dpi_info[0])
                else:
                    dpi = int(dpi_info)
                if dpi > 0:
                    return dpi
    except Exception:
        pass
    return 96


__all__ = [
    "Scale", "Calibration", "ScaleUnit",
    "pixel_to_mm", "mm_to_pixel",
    "scale_from_dpi", "get_image_dpi",
]
