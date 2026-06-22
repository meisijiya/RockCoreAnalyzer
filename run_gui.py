#!/usr/bin/env python3
"""启动岩心孔洞分析软件 GUI."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    print("错误: PyQt5 未安装,请运行: pip install PyQt5", file=sys.stderr)
    sys.exit(1)

from rockpore.gui.main_window import run_gui

if __name__ == "__main__":
    sys.exit(run_gui())
