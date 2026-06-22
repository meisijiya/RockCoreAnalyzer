#!/usr/bin/env python3
"""rockpore-cli 启动脚本."""

import sys
from pathlib import Path

# 允许从项目根目录直接运行
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rockpore.cli import main

if __name__ == "__main__":
    sys.exit(main())
