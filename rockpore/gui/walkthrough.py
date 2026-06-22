"""首次使用引导覆盖层.

用户在第一次启动(或点击"新手引导")时,
高亮显示界面元素并弹出文字提示.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)


class GuideStep:
    """引导步骤."""
    def __init__(
        self,
        target_widget: Optional[QWidget],
        title: str,
        content: str,
        position: str = "auto",  # auto / top / bottom / left / right
        next_label: str = "下一步",
        on_next: Optional[Callable] = None,
    ):
        self.target_widget = target_widget
        self.title = title
        self.content = content
        self.position = position
        self.next_label = next_label
        self.on_next = on_next


class WalkthroughOverlay(QDialog):
    """半透明引导遮罩 + 提示气泡."""

    def __init__(self, steps: List[GuideStep], parent=None):
        super().__init__(parent)
        self.steps = steps
        self.idx = 0
        # 透明遮罩属性
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setModal(True)
        # 全屏
        if parent:
            self.setGeometry(parent.geometry())
        # 提示气泡
        self._build_bubble()
        self._position_bubble()
        self.show()

    def _build_bubble(self):
        self._bubble = QFrame(self)
        self._bubble.setStyleSheet("""
            QFrame {
                background: white;
                border: 2px solid #2c5fa3;
                border-radius: 12px;
            }
            QLabel#bubbleTitle {
                color: #2c5fa3;
                font-size: 16px;
                font-weight: bold;
            }
            QLabel#bubbleContent {
                color: #1f2328;
                font-size: 13px;
                line-height: 1.6;
            }
        """)
        v = QVBoxLayout(self._bubble)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(8)
        self._title_label = QLabel("")
        self._title_label.setObjectName("bubbleTitle")
        self._title_label.setWordWrap(True)
        v.addWidget(self._title_label)
        self._content_label = QLabel("")
        self._content_label.setObjectName("bubbleContent")
        self._content_label.setWordWrap(True)
        self._content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self._content_label)
        # 进度 + 按钮
        h = QHBoxLayout()
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #57606a; font-size: 11px;")
        h.addWidget(self._progress_label)
        h.addStretch(1)
        self._skip_btn = QPushButton("跳过引导")
        self._skip_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #57606a;
                border: none;
                padding: 6px 12px;
            }
            QPushButton:hover { color: #2c5fa3; }
        """)
        self._skip_btn.clicked.connect(self._on_skip)
        h.addWidget(self._skip_btn)
        self._next_btn = QPushButton("下一步")
        self._next_btn.setStyleSheet("""
            QPushButton {
                background: #2c5fa3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1e4880; }
        """)
        self._next_btn.clicked.connect(self._on_next)
        h.addWidget(self._next_btn)
        v.addLayout(h)

    def _position_bubble(self):
        step = self.steps[self.idx]
        bw, bh = 380, 200
        # 默认放在屏幕中央
        x = (self.width() - bw) // 2
        y = (self.height() - bh) // 2
        if step.target_widget is not None:
            target_geom = step.target_widget.mapToGlobal(QPoint(0, 0))
            target_rect = QRect(target_geom, step.target_widget.size())
            if step.position == "right":
                x = target_rect.right() + 20
                y = target_rect.center().y() - bh // 2
            elif step.position == "left":
                x = target_rect.left() - bw - 20
                y = target_rect.center().y() - bh // 2
            elif step.position == "bottom":
                x = target_rect.center().x() - bw // 2
                y = target_rect.bottom() + 20
            elif step.position == "top":
                x = target_rect.center().x() - bw // 2
                y = target_rect.top() - bh - 20
            else:  # auto - 选有空间的一侧
                space_right = self.width() - target_rect.right()
                space_left = target_rect.left()
                space_bottom = self.height() - target_rect.bottom()
                spaces = [
                    ("right", space_right, target_rect.center().y() - bh // 2),
                    ("bottom", space_bottom, target_rect.bottom() + 20),
                    ("left", space_left, target_rect.center().y() - bh // 2),
                    ("top", target_rect.top(), target_rect.top() - bh - 20),
                ]
                pos, sx, sy = max(spaces, key=lambda s: s[1])
                x = sx if pos in ("right", "left") else target_rect.center().x() - bw // 2
                y = sy
        # 边界修正
        x = max(10, min(x, self.width() - bw - 10))
        y = max(10, min(y, self.height() - bh - 10))
        self._bubble.setGeometry(x, y, bw, bh)
        # 更新内容
        self._title_label.setText(step.title)
        self._content_label.setText(step.content)
        self._next_btn.setText(step.next_label)
        self._progress_label.setText(f"{self.idx + 1} / {len(self.steps)}")
        if self.idx == len(self.steps) - 1:
            self._next_btn.setText("完成")

    def _on_next(self):
        step = self.steps[self.idx]
        if step.on_next:
            step.on_next()
        if self.idx < len(self.steps) - 1:
            self.idx += 1
            self._position_bubble()
        else:
            self.close()

    def _on_skip(self):
        self.close()

    def paintEvent(self, event):
        # 半透明背景
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
        # 目标位置挖洞
        step = self.steps[self.idx]
        if step.target_widget is not None:
            target_geom = step.target_widget.mapToGlobal(QPoint(0, 0))
            local = self.mapFromGlobal(target_geom)
            target_rect = QRect(local, step.target_widget.size())
            # 在目标位置画一个高亮框
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(target_rect, Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(44, 95, 163), 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(target_rect)
