"""教学辅助面板 - 侧栏卡片式教学说明.

每个步骤在右侧显示:
- 步骤标题 + 进度
- "为什么" 卡片(科学依据)
- "怎么做" 卡片(操作指引)
- "地质意义" 卡片(背景知识,可选)
- 下一步建议
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget, QProgressBar,
)

from .module_base import StepDefinition
from .theme import color


class Card(QFrame):
    """通用卡片."""
    def __init__(self, kind: str = "info", parent=None):
        super().__init__(parent)
        self.setObjectName(f"{kind}Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        self._layout = layout
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def add_title(self, text: str):
        title = QLabel(text)
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._layout.addWidget(title)
        return title

    def add_content(self, text: str):
        content = QLabel(text)
        content.setWordWrap(True)
        content.setStyleSheet("font-size: 12px; line-height: 1.5;")
        content.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._layout.addWidget(content)
        return content


class TeachingPanel(QScrollArea):
    """教学说明侧栏.

    包含:
    - 步骤进度条 + 标题
    - "为什么" 卡片
    - "怎么做" 卡片
    - "地质意义" 卡片(可选)
    - 下一步/上一步按钮
    """
    next_clicked = pyqtSignal()
    prev_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("teachingPanel")
        self.setStyleSheet(f"""
            QScrollArea#teachingPanel {{
                background: {color('bg')};
                border: none;
            }}
            QScrollArea#teachingPanel > QWidget > QWidget {{
                background: transparent;
            }}
        """)
        self.setWidgetResizable(True)
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)
        self._build_ui()
        self.set_step(None, 0, 0)

    def _build_ui(self):
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)
        # 步骤进度
        self._step_label = QLabel("步骤 0/0")
        self._step_label.setStyleSheet(f"""
            color: {color('primary')};
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
        """)
        v.addWidget(self._step_label)
        self._progress = QProgressBar()
        self._progress.setMaximum(100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        v.addWidget(self._progress)
        # 标题
        self._title = QLabel("未选择步骤")
        self._title.setStyleSheet(f"""
            color: {color('text')};
            font-size: 18px;
            font-weight: bold;
        """)
        self._title.setWordWrap(True)
        v.addWidget(self._title)
        # 副标题
        self._subtitle = QLabel("")
        self._subtitle.setStyleSheet(f"color: {color('text_secondary')}; font-size: 12px;")
        self._subtitle.setWordWrap(True)
        v.addWidget(self._subtitle)
        v.addSpacing(8)
        # 为什么
        self._why_card = Card("info")
        self._why_card.add_title("💡 为什么做这一步?")
        self._why_text = self._why_card.add_content("")
        v.addWidget(self._why_card)
        # 怎么做
        self._how_card = Card("tip")
        self._how_card.add_title("🛠 怎么做?")
        self._how_text = self._how_card.add_content("")
        v.addWidget(self._how_card)
        # 地质意义
        self._geo_card = Card("warning")
        self._geo_card.add_title("⛏ 地质意义")
        self._geo_text = self._geo_card.add_content("")
        v.addWidget(self._geo_card)
        v.addStretch(1)
        # 导航按钮
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("◀ 上一步")
        self._prev_btn.setObjectName("ghostButton")
        self._prev_btn.clicked.connect(self.prev_clicked)
        nav.addWidget(self._prev_btn)
        self._next_btn = QPushButton("下一步 ▶")
        self._next_btn.setObjectName("primaryButton")
        self._next_btn.clicked.connect(self.next_clicked)
        nav.addWidget(self._next_btn)
        v.addLayout(nav)
        self.setWidget(container)

    def set_step(self, step: Optional[StepDefinition], idx: int, total: int):
        """更新当前步骤的显示."""
        if step is None:
            self._step_label.setText(f"步骤 0/{total}")
            self._progress.setValue(0)
            self._title.setText("未选择步骤")
            self._subtitle.setText("")
            self._why_text.setText("请先打开图像,然后开始分析流程。")
            self._how_text.setText("点击左上角「打开图像」按钮 (Ctrl+O)")
            self._geo_text.setText("")
            self._why_card.setVisible(True)
            self._how_card.setVisible(True)
            self._geo_card.setVisible(False)
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return
        self._step_label.setText(f"步骤 {step.index}/{total}")
        pct = int(step.index / total * 100)
        self._progress.setValue(pct)
        self._title.setText(step.title)
        self._subtitle.setText(step.subtitle)
        self._why_text.setText(step.why)
        self._how_text.setText(step.how)
        if step.geology:
            self._geo_text.setText(step.geology)
            self._geo_card.setVisible(True)
        else:
            self._geo_card.setVisible(False)
        self._prev_btn.setEnabled(step.index > 1)
        self._next_btn.setEnabled(step.index < total)

    def set_next_enabled(self, enabled: bool):
        self._next_btn.setEnabled(enabled)

    def set_prev_enabled(self, enabled: bool):
        self._prev_btn.setEnabled(enabled)
