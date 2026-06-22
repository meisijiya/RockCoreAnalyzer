"""粒度分析模块 - 预留接口.

后续实现 TODO:
- 步骤 5: 颗粒分割 (距离变换 + 分水岭)
- 步骤 6: 边界修正
- 步骤 7: 粒级筛选
- 步骤 8: 粒度参数计算 (均值/分选/偏度)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .module_base import AnalysisModule, make_default_grain_steps
from .pore_module import StepPlaceholder


class GrainModule(AnalysisModule):
    """粒度分析模块(预留接口,后续实现)."""
    name = "粒度分析"
    icon = "⚪"
    description = "识别颗粒并计算粒度分布、粒度参数"

    def __init__(self):
        self._steps = make_default_grain_steps()

    @property
    def steps(self):
        return self._steps

    def create_step_panel(self, step_idx: int, parent=None):
        return StepPlaceholder(
            f"粒度分析 - 步骤 {step_idx + 1}",
            "🚧 粒度分析模块正在开发中。\n\n"
            "该模块将提供以下能力:\n"
            "• 颗粒自动检测\n"
            "• 距离变换 + 分水岭分割\n"
            "• 粒度分布直方图\n"
            "• 粒度参数 (均值/分选/偏度/峰度)\n"
            "• 沉积环境判别\n\n"
            "请使用「孔洞分析」Tab 体验完整功能。",
            "🚧"
        )

    def run_step(self, step_idx: int, context: dict) -> dict:
        return {}

    def analyze(self, image, scale, context: dict) -> Any:
        raise NotImplementedError("粒度分析模块待实现")

    def build_report_data(self, context: dict) -> dict:
        return {}
