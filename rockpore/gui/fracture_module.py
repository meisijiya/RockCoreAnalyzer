"""裂缝分析模块 - 预留接口.

后续实现 TODO:
- 步骤 5: 裂缝提取 (HoughLinesP / Hessian + 阈值)
- 步骤 6: 形态学优化
- 步骤 7: 类型标注 (构造缝/成岩缝/风化缝)
- 步骤 8: 长度/宽度/密度计算
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from .module_base import AnalysisModule, StepDefinition, make_default_fracture_steps
from .pore_module import StepPlaceholder


class FractureModule(AnalysisModule):
    """裂缝分析模块(预留接口,后续实现)."""
    name = "裂缝分析"
    icon = "📏"
    description = "识别岩心裂缝并计算长度、宽度、密度"

    def __init__(self):
        self._steps = make_default_fracture_steps()

    @property
    def steps(self):
        return self._steps

    def create_step_panel(self, step_idx: int, parent=None):
        return StepPlaceholder(
            f"裂缝分析 - 步骤 {step_idx + 1}",
            "🚧 裂缝分析模块正在开发中。\n\n"
            "该模块将提供以下能力:\n"
            "• 裂缝自动检测 (Hough 变换 / Hessian 矩阵增强)\n"
            "• 裂缝长度、宽度测量\n"
            "• 裂缝密度计算 (面密度、线密度、间距)\n"
            "• 裂缝类型标注 (构造/成岩/风化)\n"
            "• 充填程度评价\n\n"
            "请使用「孔洞分析」Tab 体验完整功能。",
            "🚧"
        )

    def run_step(self, step_idx: int, context: dict) -> dict:
        return {}

    def analyze(self, image, scale, context: dict) -> Any:
        # TODO: 实现裂缝分析
        raise NotImplementedError("裂缝分析模块待实现")

    def build_report_data(self, context: dict) -> dict:
        return {}
