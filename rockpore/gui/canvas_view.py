"""重写后的画布控件.

比旧版 ImageCanvas:
- 更好的视觉层次(背景色、网格线)
- 缩放/重置按钮更明显
- 工具栏集成在画布上
- 双击可重置视图
- 状态显示更友好
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal
from PyQt5.QtGui import (
    QColor, QCursor, QFont, QImage, QPainter, QPen, QPixmap, QWheelEvent, QMouseEvent,
    QPaintEvent,
)
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget, QToolButton,
)

from .theme import color


def numpy_to_qimage(image: np.ndarray) -> QImage:
    """numpy BGR/Gray → QImage."""
    if image is None:
        return QImage()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        h, w = image.shape
        return QImage(image.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, c = image.shape
    if c == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    if c == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return QImage(rgb.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
    return QImage()


class CanvasTool(str, Enum):
    """画布工具."""
    VIEW = "view"          # 查看/平移
    PICK = "pick"          # 选择孔洞
    ERASE = "erase"        # 橡皮擦
    ADD = "add"            # 添加
    COLOR_PICK = "color"   # 颜色拾取


class Annotation:
    """单条标注."""
    def __init__(self, id, bbox: Tuple[int, int, int, int], label: str = "",
                 color: Tuple[int, int, int] = (0, 200, 100)):
        self.id = id
        self.bbox = bbox  # (x, y, w, h)
        self.label = label
        self.color = color


class CanvasView(QFrame):
    """重写后的图像画布.

    布局:
    ┌──────────────────────────────────┐
    │ [缩放 100%] [1:1] [重置] [工具]   │ <- 顶部工具条
    ├──────────────────────────────────┤
    │                                  │
    │       图像显示区域                │
    │                                  │
    │                                  │
    ├──────────────────────────────────┤
    │ 位置 (x, y) | 缩放 100%          │ <- 底部状态
    └──────────────────────────────────┘
    """
    pore_clicked = pyqtSignal(int)
    color_picked = pyqtSignal(tuple)  # (B, G, R)
    mask_modified = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("canvasFrame")
        self.setStyleSheet(f"""
            QFrame#canvasFrame {{
                background: #2a2d35;
                border: 1px solid {color('border')};
                border-radius: 8px;
            }}
        """)
        # 数据
        self._image: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._overlay: Optional[np.ndarray] = None
        self._overlay_alpha: float = 0.4
        self._annotations: List[Annotation] = []
        # 视图状态
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._last_pos: Optional[QPoint] = None
        # 工具
        self._tool = CanvasTool.VIEW
        self._brush_radius = 12
        # 构建 UI
        self._build_ui()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 降低最小高度 400→320,让 step_stack 有更多空间(从审查建议)
        self.setMinimumSize(500, 320)
        # 鼠标追踪
        self._canvas.setMouseTracking(True)
        self._canvas.setFocusPolicy(Qt.StrongFocus)

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)
        # 顶部工具条
        top = QHBoxLayout()
        top.setSpacing(4)
        self._zoom_label = QLabel("100%")
        self._zoom_label.setStyleSheet(
            f"color: {color('text_inverse')}; padding: 0 8px; font-weight: bold;"
        )
        top.addWidget(self._zoom_label)
        top.addSpacing(8)
        # 工具按钮组
        self._tool_buttons = {}
        for tool, label, shortcut in [
            (CanvasTool.VIEW, "🖱 查看", "V"),
            (CanvasTool.PICK, "👆 选择", "S"),
            (CanvasTool.ERASE, "🧹 擦除", "E"),
            (CanvasTool.ADD, "➕ 添加", "A"),
            (CanvasTool.COLOR_PICK, "🎯 取色", "C"),
        ]:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setToolTip(f"{label} ({shortcut})")
            btn.setStyleSheet("""
                QToolButton {
                    background: rgba(255,255,255,0.1);
                    color: white;
                    border: 1px solid rgba(255,255,255,0.2);
                    border-radius: 4px;
                    padding: 4px 10px;
                    font-size: 12px;
                }
                QToolButton:hover {
                    background: rgba(255,255,255,0.2);
                }
                QToolButton:checked {
                    background: #ff7849;
                    color: white;
                    border-color: #ff7849;
                }
            """)
            btn.clicked.connect(lambda _, t=tool: self._set_tool(t))
            self._tool_buttons[tool] = btn
            top.addWidget(btn)
        top.addSpacing(8)
        # 笔刷半径
        top.addWidget(QLabel("笔刷:"))
        self._brush_minus = QToolButton()
        self._brush_minus.setText("−")
        self._brush_minus.setFixedSize(28, 28)
        self._brush_minus.setStyleSheet("""
            QToolButton {
                background: rgba(255,255,255,0.1);
                color: white;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
            }
            QToolButton:hover { background: rgba(255,255,255,0.2); }
        """)
        self._brush_minus.clicked.connect(lambda: self._change_brush(-2))
        top.addWidget(self._brush_minus)
        self._brush_label = QLabel(f"{self._brush_radius}px")
        self._brush_label.setStyleSheet("color: white; padding: 0 4px; min-width: 30px;")
        self._brush_label.setAlignment(Qt.AlignCenter)
        top.addWidget(self._brush_label)
        self._brush_plus = QToolButton()
        self._brush_plus.setText("+")
        self._brush_plus.setFixedSize(28, 28)
        self._brush_plus.setStyleSheet(self._brush_minus.styleSheet())
        self._brush_plus.clicked.connect(lambda: self._change_brush(2))
        top.addWidget(self._brush_plus)
        top.addStretch(1)
        # 缩放按钮
        for text, slot, tip in [
            ("放大", lambda: self._zoom_at(self._canvas.rect().center(), 1.2), "Ctrl++"),
            ("缩小", lambda: self._zoom_at(self._canvas.rect().center(), 1/1.2), "Ctrl+-"),
            ("1:1", self._reset_view, "恢复 100% 缩放"),
            ("适应", self._fit_view, "适应窗口"),
        ]:
            btn = QToolButton()
            btn.setText(text)
            btn.setToolTip(tip)
            btn.setStyleSheet("""
                QToolButton {
                    background: rgba(255,255,255,0.1);
                    color: white;
                    border: 1px solid rgba(255,255,255,0.2);
                    border-radius: 4px;
                    padding: 4px 10px;
                    font-size: 12px;
                }
                QToolButton:hover { background: rgba(255,255,255,0.2); }
            """)
            btn.clicked.connect(slot)
            top.addWidget(btn)
        v.addLayout(top)
        # 画布
        self._canvas = _CanvasSurface(self)
        self._canvas.pore_clicked.connect(self.pore_clicked)
        self._canvas.color_picked.connect(self.color_picked)
        self._canvas.mask_modified.connect(self.mask_modified)
        v.addWidget(self._canvas, 1)
        # 底部状态条
        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        self._pos_label = QLabel("位置: -")
        self._pos_label.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 11px;")
        bottom.addWidget(self._pos_label)
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 11px;")
        bottom.addWidget(self._info_label)
        bottom.addStretch(1)
        self._tool_label = QLabel("工具: 查看")
        self._tool_label.setStyleSheet(
            f"color: {color('accent')}; font-size: 11px; font-weight: bold;"
        )
        bottom.addWidget(self._tool_label)
        v.addLayout(bottom)
        # 默认工具
        self._set_tool(CanvasTool.VIEW)

    # ---------------- 公共 API ----------------
    def set_image(self, image: np.ndarray):
        """设置图像."""
        self._canvas.set_image(image)
        self._mask = None
        self._overlay = None
        self._annotations = []
        if image is not None:
            self._fit_view()

    def set_mask(self, mask: np.ndarray, color_bgr=(255, 0, 255), alpha=0.4):
        """设置掩码叠加层."""
        self._mask = mask
        self._overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        self._overlay[mask > 0] = color_bgr
        self._overlay_alpha = alpha
        self._canvas.set_overlay(self._overlay, alpha)
        self._update_info()

    def clear_overlay(self):
        self._mask = None
        self._overlay = None
        self._annotations = []
        self._canvas.set_overlay(None, 0)
        self._update_info()

    def set_annotations(self, annotations: List[Annotation]):
        self._annotations = annotations
        self._canvas.set_annotations(annotations)

    def set_tool(self, tool: CanvasTool):
        self._set_tool(tool)

    def set_brush_radius(self, r: int):
        self._brush_radius = max(2, min(100, r))
        self._brush_label.setText(f"{self._brush_radius}px")
        self._canvas.brush_radius = self._brush_radius

    def get_image(self) -> Optional[np.ndarray]:
        return self._canvas.image

    def get_mask(self) -> Optional[np.ndarray]:
        return self._mask

    # ---------------- 内部 ----------------
    def _set_tool(self, tool: CanvasTool):
        self._tool = tool
        self._canvas.set_tool(tool)
        # 互斥按钮
        for t, btn in self._tool_buttons.items():
            btn.setChecked(t == tool)
        # 更新提示
        tool_names = {
            CanvasTool.VIEW: "查看",
            CanvasTool.PICK: "选择孔洞",
            CanvasTool.ERASE: "擦除",
            CanvasTool.ADD: "添加",
            CanvasTool.COLOR_PICK: "取色",
        }
        self._tool_label.setText(f"工具: {tool_names[tool]}")
        # 光标
        cursors = {
            CanvasTool.VIEW: Qt.OpenHandCursor,
            CanvasTool.PICK: Qt.PointingHandCursor,
            CanvasTool.ERASE: Qt.CrossCursor,
            CanvasTool.ADD: Qt.CrossCursor,
            CanvasTool.COLOR_PICK: Qt.CrossCursor,
        }
        self._canvas.setCursor(cursors[tool])

    def _change_brush(self, delta: int):
        self.set_brush_radius(self._brush_radius + delta)

    def _zoom_at(self, center: QPoint, factor: float):
        self._canvas.zoom_at(center, factor)

    def _reset_view(self):
        self._canvas.set_scale(1.0)
        self._update_zoom_label()

    def _fit_view(self):
        self._canvas.fit_to_widget()

    def _update_zoom_label(self):
        self._zoom_label.setText(f"{int(self._canvas.scale * 100)}%")

    def _update_info(self):
        """更新底部信息条:显示掩码覆盖率.
        I2 修复:移除死代码 (analyze_pores 导入、self._scale 不存在、静默 try/except).
        """
        if self._mask is None:
            self._info_label.setText("")
            return
        h, w = self._mask.shape[:2]
        white = int((self._mask > 0).sum())
        pct = white / (h * w) * 100
        self._info_label.setText(f"掩码覆盖: {white:,} px ({pct:.1f}%)")

    def update_pos_label(self, pos: QPoint):
        if pos:
            self._pos_label.setText(f"位置: ({pos.x()}, {pos.y()})")

    def update_zoom_label(self, scale: float):
        self._zoom_label.setText(f"{int(scale * 100)}%")


class _CanvasSurface(QWidget):
    """实际绘制和接收鼠标事件的画布表面."""
    pore_clicked = pyqtSignal(int)
    color_picked = pyqtSignal(tuple)
    mask_modified = pyqtSignal(np.ndarray)

    def __init__(self, parent: CanvasView):
        super().__init__(parent)
        self.parent_view = parent
        self.image: Optional[np.ndarray] = None
        self.overlay: Optional[np.ndarray] = None
        self.overlay_alpha: float = 0.4
        self.annotations: List[Annotation] = []
        self.tool: CanvasTool = CanvasTool.VIEW
        self.brush_radius: int = 12
        # 视图
        self.scale: float = 1.0
        self.offset = QPoint(0, 0)
        self.last_pos: Optional[QPoint] = None
        # 背景棋盘格
        self._check_pixmap: Optional[QPixmap] = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ----- 外部调用 -----
    def set_image(self, image: Optional[np.ndarray]):
        self.image = image
        self.overlay = None
        self.annotations = []
        self.update()

    def set_overlay(self, overlay: Optional[np.ndarray], alpha: float):
        self.overlay = overlay
        self.overlay_alpha = alpha
        self.update()

    def set_annotations(self, annotations: List[Annotation]):
        self.annotations = annotations
        self.update()

    def set_tool(self, tool: CanvasTool):
        self.tool = tool

    def set_scale(self, scale: float):
        self.scale = scale
        self._center_image()
        self.parent_view.update_zoom_label(scale)
        self.update()

    def zoom_at(self, screen_pos: QPoint, factor: float):
        if self.image is None or self.scale <= 0:
            return
        old = self.scale
        new = max(0.05, min(20.0, old * factor))
        if old == new:
            return
        img_w_old = self.image.shape[1] * old
        img_h_old = self.image.shape[0] * old
        if img_w_old == 0 or img_h_old == 0:
            return
        rel_x = (screen_pos.x() - self.offset.x()) / img_w_old
        rel_y = (screen_pos.y() - self.offset.y()) / img_h_old
        self.scale = new
        new_w = self.image.shape[1] * new
        new_h = self.image.shape[0] * new
        self.offset = QPoint(
            int(screen_pos.x() - rel_x * new_w),
            int(screen_pos.y() - rel_y * new_h),
        )
        self.parent_view.update_zoom_label(new)
        self.update()

    def fit_to_widget(self):
        if self.image is None:
            return
        w, h = self.image.shape[1], self.image.shape[0]
        cw, ch = self.width(), self.height()
        if cw == 0 or ch == 0:
            return
        scale = min(cw / w, ch / h) * 0.95
        self.scale = max(0.05, min(20.0, scale))
        self._center_image()
        self.parent_view.update_zoom_label(self.scale)
        self.update()

    # ----- 鼠标事件 -----
    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.2 if delta > 0 else 1 / 1.2
        self.zoom_at(event.pos(), factor)

    def mousePressEvent(self, event: QMouseEvent):
        if self.image is None:
            return
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and self.tool == CanvasTool.VIEW
        ):
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            self._apply_tool(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent):
        # 更新位置标签
        img_pos = self._screen_to_image(event.pos())
        if img_pos:
            self.parent_view.update_pos_label(img_pos)
        # 平移
        if self.last_pos is not None and self.tool == CanvasTool.VIEW:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            self.offset = QPoint(self.offset.x() + dx, self.offset.y() + dy)
            self.last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self.last_pos = None
            cursors = {
                CanvasTool.VIEW: Qt.OpenHandCursor,
                CanvasTool.PICK: Qt.PointingHandCursor,
                CanvasTool.ERASE: Qt.CrossCursor,
                CanvasTool.ADD: Qt.CrossCursor,
                CanvasTool.COLOR_PICK: Qt.CrossCursor,
            }
            self.setCursor(cursors[self.tool])

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.fit_to_widget()

    def keyPressEvent(self, event):
        # 快捷键
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_0:
                self.set_scale(1.0)
            elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                self.zoom_at(self.rect().center(), 1.2)
            elif event.key() == Qt.Key_Minus:
                self.zoom_at(self.rect().center(), 1 / 1.2)
        elif event.key() == Qt.Key_V:
            self.parent_view._set_tool(CanvasTool.VIEW)
        elif event.key() == Qt.Key_E:
            self.parent_view._set_tool(CanvasTool.ERASE)
        elif event.key() == Qt.Key_A:
            self.parent_view._set_tool(CanvasTool.ADD)
        elif event.key() == Qt.Key_S:
            self.parent_view._set_tool(CanvasTool.PICK)
        elif event.key() == Qt.Key_C:
            self.parent_view._set_tool(CanvasTool.COLOR_PICK)
        elif event.key() == Qt.Key_F:
            self.fit_to_widget()

    # ----- 工具应用 -----
    def _screen_to_image(self, pos: QPoint) -> Optional[QPoint]:
        if self.image is None or self.scale <= 0:
            return None
        ix = int((pos.x() - self.offset.x()) / self.scale)
        iy = int((pos.y() - self.offset.y()) / self.scale)
        if 0 <= ix < self.image.shape[1] and 0 <= iy < self.image.shape[0]:
            return QPoint(ix, iy)
        return None

    def _apply_tool(self, screen_pos: QPoint):
        if self.image is None:
            return
        img_pos = self._screen_to_image(screen_pos)
        if img_pos is None:
            return
        if self.tool == CanvasTool.PICK:
            for ann in self.annotations:
                x, y, w, h = ann.bbox
                if x <= img_pos.x() < x + w and y <= img_pos.y() < y + h:
                    self.pore_clicked.emit(ann.id)
                    return
            self.pore_clicked.emit(-1)
            return
        if self.tool == CanvasTool.COLOR_PICK:
            if self.image.ndim == 3:
                b, g, r = self.image[img_pos.y(), img_pos.x()]
                self.color_picked.emit((int(b), int(g), int(r)))
            return
        if self.tool in (CanvasTool.ERASE, CanvasTool.ADD) and self.parent_view._mask is not None:
            mask = self.parent_view._mask.copy()
            r = self.brush_radius
            if self.tool == CanvasTool.ERASE:
                cv2.circle(mask, (img_pos.x(), img_pos.y()), r, 0, -1)
            else:
                cv2.circle(mask, (img_pos.x(), img_pos.y()), r, 255, -1)
            self.parent_view.set_mask(mask)
            self.mask_modified.emit(mask)

    def _center_image(self):
        if self.image is None:
            return
        cw, ch = self.width(), self.height()
        w = self.image.shape[1] * self.scale
        h = self.image.shape[0] * self.scale
        self.offset = QPoint(int((cw - w) / 2), int((ch - h) / 2))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 保持当前居中(若 scale 为适应)
        if self.scale <= 1.0:
            self._center_image()

    # ----- 绘制 -----
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        # 背景棋盘格
        painter.fillRect(self.rect(), QColor(42, 45, 53))
        if self.image is None:
            painter.setPen(QColor(200, 200, 200))
            font = QFont("Microsoft YaHei", 14)
            painter.setFont(font)
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                "📷  请先打开岩心图像 (Ctrl+O)\n\n"
                "支持格式: JPG / PNG / BMP / TIF\n"
                "建议分辨率: 500×400 以上",
            )
            return
        h, w = self.image.shape[:2]
        scaled_w = int(w * self.scale)
        scaled_h = int(h * self.scale)
        target = QRect(self.offset.x(), self.offset.y(), scaled_w, scaled_h)
        # 原图
        qimg = numpy_to_qimage(self.image)
        painter.drawImage(target, qimg)
        # 叠加层
        if self.overlay is not None:
            painter.setOpacity(self.overlay_alpha)
            overlay_qimg = numpy_to_qimage(self.overlay)
            painter.drawImage(target, overlay_qimg)
            painter.setOpacity(1.0)
        # 标注
        if self.annotations:
            for ann in self.annotations:
                x, y, ww, hh = ann.bbox
                rx = self.offset.x() + int(x * self.scale)
                ry = self.offset.y() + int(y * self.scale)
                rw = max(2, int(ww * self.scale))
                rh = max(2, int(hh * self.scale))
                r, g, b = ann.color
                pen = QPen(QColor(r, g, b), 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rx, ry, rw, rh)
                # 编号
                if ann.id is not None:
                    text = f"#{ann.id}" + (f" {ann.label}" if ann.label else "")
                    font = QFont("Arial", 10, QFont.Bold)
                    painter.setFont(font)
                    fm = painter.fontMetrics()
                    tw = fm.horizontalAdvance(text) + 8
                    th = fm.height() + 4
                    # 文字背景
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(r, g, b, 220))
                    painter.drawRoundedRect(rx, ry - th - 2, tw, th, 3, 3)
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(rx + 4, ry - 5, text)
        # 十字标线(取色器工具时)
        if self.tool == CanvasTool.COLOR_PICK:
            mx, my = self.mapFromGlobal(QCursor.pos())
            if self.rect().contains(QPoint(mx, my)):
                pen = QPen(QColor(255, 200, 0, 180), 1, Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(mx, 0, mx, self.height())
                painter.drawLine(0, my, self.width(), my)
