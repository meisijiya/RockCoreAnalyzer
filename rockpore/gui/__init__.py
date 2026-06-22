"""PyQt5 GUI 模块 v2.

现代化教学风格的岩心分析软件,支持多模块 Tab 切换.

模块:
- 主题: theme.py (QSS 样式)
- 画布: canvas_view.py (CanvasView)
- 教学侧栏: teaching_panel.py
- 帮助: help_dialog.py
- 引导: walkthrough.py
- 模块基类: module_base.py
- 孔洞模块: pore_module.py
- 裂缝模块: fracture_module.py
- 粒度模块: grain_module.py
- 主窗口: main_window.py
"""

from .main_window import MainWindow, run_gui
from .module_base import AnalysisModule, StepDefinition
from .pore_module import PoreModule
from .fracture_module import FractureModule
from .grain_module import GrainModule

__all__ = [
    "MainWindow", "run_gui",
    "AnalysisModule", "StepDefinition",
    "PoreModule", "FractureModule", "GrainModule",
]
