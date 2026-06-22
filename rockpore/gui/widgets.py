"""自定义 PyQt5 控件.

提供图像画布、缩放控件、参数面板等可复用 UI 组件.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPixmap, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QLineEdit, QTextEdit, QScrollArea, QSizePolicy,
)


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


class ImageCanvas(QWidget):
    """可缩放、可标注的图像画布.
    支持:
    - 鼠标拖动平移
    - 滚轮缩放
    - 在图像上叠加掩码(半透明彩色)
    - 绘制边界框与编号
    - 鼠标点击选择孔洞
    """
    pore_clicked = pyqtSignal(int)  # 孔洞 id(若 -1 则为背景)
    mask_modified = pyqtSignal(np.ndarray)  # 掩码被修改

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image: Optional[np.ndarray] = None  # BGR
        self.overlay: Optional[np.ndarray] = None  # BGR 半透明
        self.mask: Optional[np.ndarray] = None  # 单通道 0/255
        self.annotations: list = []  # [(id, bbox, label), ...]
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.last_pos: Optional[QPoint] = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        self.tool = "view"  # view / erase / add / pick
        self.erase_radius = 10

    def set_image(self, image: np.ndarray):
        """设置显示的图像."""
        self.image = image
        self.overlay = None
        self.mask = None
        self.annotations = []
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.update()

    def set_mask(self, mask: np.ndarray, color: tuple = (255, 0, 255), alpha: float = 0.4):
        """设置掩码叠加层."""
        self.mask = mask
        self.overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        self.overlay[mask > 0] = color
        self.overlay_alpha = alpha
        self.update()

    def clear_overlay(self):
        self.overlay = None
        self.mask = None
        self.update()

    def set_annotations(self, annotations: list):
        """annotations: [(id, (x, y, w, h), label), ...]"""
        self.annotations = annotations
        self.update()

    def set_tool(self, tool: str):
        """设置工具: view/erase/add/pick."""
        self.tool = tool

    def wheelEvent(self, event):
        if self.image is None:
            return
        delta = event.angleDelta().y()
        old = self.scale_factor
        if delta > 0:
            self.scale_factor = min(8.0, self.scale_factor * 1.2)
        else:
            self.scale_factor = max(0.1, self.scale_factor / 1.2)
        # 围绕鼠标位置缩放
        if old > 0:
            mouse = event.pos()
            img_w_old = self.image.shape[1] * old
            img_h_old = self.image.shape[0] * old
            rel_x = (mouse.x() - self.offset.x()) / img_w_old
            rel_y = (mouse.y() - self.offset.y()) / img_h_old
            new_w = self.image.shape[1] * self.scale_factor
            new_h = self.image.shape[0] * self.scale_factor
            self.offset = QPoint(int(mouse.x() - rel_x * new_w),
                                  int(mouse.y() - rel_y * new_h))
        self.update()

    def mousePressEvent(self, event):
        if self.image is None:
            return
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.tool == "view"):
            self.last_pos = event.pos()
        elif event.button() == Qt.LeftButton and self.tool in ("erase", "add", "pick"):
            self._apply_tool(event.pos())

    def mouseMoveEvent(self, event):
        if self.last_pos is not None and self.tool == "view":
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            self.offset = QPoint(self.offset.x() + dx, self.offset.y() + dy)
            self.last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton) and self.tool == "view":
            self.last_pos = None

    def _screen_to_image(self, pos: QPoint) -> Optional[QPoint]:
        """将屏幕坐标转换为图像坐标."""
        if self.image is None or self.scale_factor <= 0:
            return None
        ix = int((pos.x() - self.offset.x()) / self.scale_factor)
        iy = int((pos.y() - self.offset.y()) / self.scale_factor)
        if 0 <= ix < self.image.shape[1] and 0 <= iy < self.image.shape[0]:
            return QPoint(ix, iy)
        return None

    def _apply_tool(self, screen_pos: QPoint):
        img_pos = self._screen_to_image(screen_pos)
        if img_pos is None or self.mask is None:
            return
        if self.tool == "pick":
            # 检查点击位置在哪个孔洞内
            for ann in self.annotations:
                ann_id, bbox, _ = ann
                x, y, w, h = bbox
                if x <= img_pos.x() < x + w and y <= img_pos.y() < y + h:
                    self.pore_clicked.emit(ann_id)
                    return
            self.pore_clicked.emit(-1)
            return
        # erase / add
        radius = self.erase_radius
        if self.tool == "erase":
            cv2.circle(self.mask, (img_pos.x(), img_pos.y()), radius, 0, -1)
        elif self.tool == "add":
            cv2.circle(self.mask, (img_pos.x(), img_pos.y()), radius, 255, -1)
        # 重新构建 overlay
        self.set_mask(self.mask)
        self.mask_modified.emit(self.mask)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(50, 50, 50))
        if self.image is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "请先打开岩心图像 (Ctrl+O)")
            return
        h, w = self.image.shape[:2]
        scaled_w = int(w * self.scale_factor)
        scaled_h = int(h * self.scale_factor)
        target = QRect(self.offset.x(), self.offset.y(), scaled_w, scaled_h)
        # 绘制原图
        qimg = numpy_to_qimage(self.image)
        painter.drawImage(target, qimg)
        # 绘制叠加层
        if self.overlay is not None:
            overlay_qimg = numpy_to_qimage(self.overlay)
            painter.setOpacity(self.overlay_alpha)
            painter.drawImage(target, overlay_qimg)
            painter.setOpacity(1.0)
        # 绘制边界框和编号
        if self.annotations:
            painter.setPen(QPen(QColor(0, 255, 0), 1))
            font = QFont("Arial", 9)
            painter.setFont(font)
            for ann_id, bbox, label in self.annotations:
                x, y, ww, hh = bbox
                rx = self.offset.x() + int(x * self.scale_factor)
                ry = self.offset.y() + int(y * self.scale_factor)
                rw = int(ww * self.scale_factor)
                rh = int(hh * self.scale_factor)
                painter.drawRect(rx, ry, rw, rh)
                if ann_id is not None:
                    text = f"#{ann_id}" if not label else f"#{ann_id} {label}"
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(rx + 1, ry + 12, text)
                    painter.setPen(QColor(0, 255, 255))
                    painter.drawText(rx, ry + 11, text)
        # 状态栏
        if self.tool != "view":
            painter.setPen(QColor(255, 255, 0))
            painter.drawText(10, 20, f"工具: {self.tool} (滚轮缩放, 拖动平移)")


class ScalePanel(QGroupBox):
    """标尺设置面板."""
    scale_changed = pyqtSignal(int, bool)  # (dpi, microscopic)

    def __init__(self, parent=None):
        super().__init__("标尺设置", parent)
        layout = QFormLayout()
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(1, 10000)
        self.dpi_spin.setValue(96)
        self.dpi_spin.setSuffix(" DPI")
        self.micro_check = QCheckBox("微观分析(微米单位)")
        layout.addRow("图像 DPI:", self.dpi_spin)
        layout.addRow(self.micro_check)
        self.setLayout(layout)
        self.dpi_spin.valueChanged.connect(lambda v: self.scale_changed.emit(v, self.micro_check.isChecked()))
        self.micro_check.stateChanged.connect(lambda _: self.scale_changed.emit(self.dpi_spin.value(), self.micro_check.isChecked()))


class PreprocessPanel(QGroupBox):
    """图像预处理面板."""
    preprocessed = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__("图像预处理", parent)
        layout = QFormLayout()
        self.brightness = QDoubleSpinBox()
        self.brightness.setRange(-100, 100)
        self.brightness.setValue(0)
        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0, 3)
        self.contrast.setSingleStep(0.1)
        self.contrast.setValue(1.0)
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.1, 5.0)
        self.gamma.setSingleStep(0.1)
        self.gamma.setValue(1.0)
        self.blur = QSpinBox()
        self.blur.setRange(0, 31)
        self.blur.setValue(0)
        self.auto_levels = QCheckBox("自动色阶")
        self.auto_levels.setChecked(True)
        self.sharpen = QCheckBox("锐化")
        self.to_gray = QCheckBox("转灰度")
        self.invert = QCheckBox("底片效果")
        apply_btn = QPushButton("应用")
        apply_btn.clicked.connect(self._on_apply)
        layout.addRow("亮度:", self.brightness)
        layout.addRow("对比度:", self.contrast)
        layout.addRow("Gamma:", self.gamma)
        layout.addRow("模糊核:", self.blur)
        layout.addRow(self.auto_levels)
        layout.addRow(self.sharpen)
        layout.addRow(self.to_gray)
        layout.addRow(self.invert)
        layout.addRow(apply_btn)
        self.setLayout(layout)
        self._input_image = None

    def set_input(self, image: np.ndarray):
        self._input_image = image

    def _on_apply(self):
        if self._input_image is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        from rockpore.core.preprocessing import preprocess, PreprocessParams
        params = PreprocessParams(
            brightness=self.brightness.value(),
            contrast=self.contrast.value(),
            gamma=self.gamma.value(),
            blur_kernel=self.blur.value(),
            sharpen=self.sharpen.isChecked(),
            auto_levels=self.auto_levels.isChecked(),
            to_gray=self.to_gray.isChecked(),
            invert=self.invert.isChecked(),
        )
        out = preprocess(self._input_image, params)
        self.preprocessed.emit(out)


class SegmentationPanel(QGroupBox):
    """孔洞提取面板."""
    extracted = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__("孔洞提取", parent)
        layout = QFormLayout()
        self.method = QComboBox()
        self.method.addItems(["auto(OTSU)", "adaptive", "triangle", "color(手动)"])
        self.tolerance = QSpinBox()
        self.tolerance.setRange(0, 255)
        self.tolerance.setValue(30)
        self.continuous = QCheckBox("连续区域")
        self.continuous.setChecked(True)
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 10000)
        self.min_area.setValue(10)
        extract_btn = QPushButton("提取孔洞")
        extract_btn.clicked.connect(self._on_extract)
        layout.addRow("方法:", self.method)
        layout.addRow("颜色匹配度:", self.tolerance)
        layout.addRow(self.continuous)
        layout.addRow("最小面积(像素):", self.min_area)
        layout.addRow(extract_btn)
        self.setLayout(layout)
        self._input_image = None

    def set_input(self, image: np.ndarray):
        self._input_image = image

    def _on_extract(self):
        if self._input_image is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        from rockpore.core.segmentation import (
            SegmentationParams, segment_by_color, extract_pores,
        )
        from rockpore.core.morphology import remove_noise, morphological_open
        method_map = {"auto(OTSU)": "otsu", "adaptive": "adaptive", "triangle": "triangle"}
        m = self.method.currentText()
        if m in method_map:
            mask = extract_pores(self._input_image, method=method_map[m])
        else:
            params = SegmentationParams(
                color=(0, 0, 0),
                tolerance=self.tolerance.value(),
                continuous=self.continuous.isChecked(),
            )
            mask = segment_by_color(self._input_image, params)
        # 形态学清理
        mask = morphological_open(mask, kernel_size=2)
        if self.min_area.value() > 0:
            mask = remove_noise(mask, min_area=self.min_area.value())
        self.extracted.emit(mask)


class MorphologyPanel(QGroupBox):
    """二次编辑(数学形态学)面板."""
    edited = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__("二次编辑", parent)
        layout = QFormLayout()
        self.op = QComboBox()
        self.op.addItems(["膨胀", "腐蚀", "开运算", "闭运算", "去噪", "填充孔洞"])
        self.kernel = QSpinBox()
        self.kernel.setRange(1, 31)
        self.kernel.setValue(3)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 20)
        self.iterations.setValue(1)
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 100000)
        self.min_area.setValue(10)
        self.max_area = QSpinBox()
        self.max_area.setRange(0, 1000000)
        self.max_area.setValue(0)
        apply_btn = QPushButton("应用")
        apply_btn.clicked.connect(self._on_apply)
        layout.addRow("操作:", self.op)
        layout.addRow("核大小:", self.kernel)
        layout.addRow("迭代次数:", self.iterations)
        layout.addRow("去噪最小面积:", self.min_area)
        layout.addRow("去噪最大面积:", self.max_area)
        layout.addRow(apply_btn)
        self.setLayout(layout)
        self._input_mask = None

    def set_input(self, mask: np.ndarray):
        self._input_mask = mask

    def _on_apply(self):
        if self._input_mask is None:
            QMessageBox.warning(self, "提示", "请先提取孔洞")
            return
        from rockpore.core.morphology import (
            dilate_region, erode_region, morphological_open, morphological_close,
            remove_noise, fill_holes,
        )
        op = self.op.currentText()
        if op == "膨胀":
            out = dilate_region(self._input_mask, self.kernel.value(), self.iterations.value())
        elif op == "腐蚀":
            out = erode_region(self._input_mask, self.kernel.value(), self.iterations.value())
        elif op == "开运算":
            out = morphological_open(self._input_mask, self.kernel.value())
        elif op == "闭运算":
            out = morphological_close(self._input_mask, self.kernel.value())
        elif op == "去噪":
            out = remove_noise(
                self._input_mask,
                min_area=self.min_area.value(),
                max_area=self.max_area.value() if self.max_area.value() > 0 else 0,
            )
        elif op == "填充孔洞":
            out = fill_holes(self._input_mask, min_hole_area=self.min_area.value())
        else:
            out = self._input_mask
        self.edited.emit(out)


class AnalysisPanel(QGroupBox):
    """孔洞分析与特征参数设置面板."""
    def __init__(self, parent=None):
        super().__init__("孔洞特征参数", parent)
        layout = QFormLayout()
        self.pore_id = QLineEdit("")
        self.pore_id.setReadOnly(True)
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(0, 1000)
        self.diameter.setDecimals(3)
        self.diameter.setSuffix(" mm")
        self.area = QDoubleSpinBox()
        self.area.setRange(0, 100000)
        self.area.setDecimals(3)
        self.area.setSuffix(" mm²")
        self.filled_status = QComboBox()
        from rockpore.core.analysis import FilledStatus
        self.filled_status.addItems([s.value for s in FilledStatus])
        self.filled_material = QComboBox()
        from rockpore.core.analysis import FilledMaterial
        self.filled_material.addItems([s.value for s in FilledMaterial])
        self.effectiveness = QComboBox()
        from rockpore.core.analysis import Effectiveness
        self.effectiveness.addItems([s.value for s in Effectiveness])
        modify_btn = QPushButton("修改参数")
        self.modify_btn = modify_btn
        layout.addRow("孔洞编号:", self.pore_id)
        layout.addRow("等效直径:", self.diameter)
        layout.addRow("面积:", self.area)
        layout.addRow("填充情况:", self.filled_status)
        layout.addRow("填充物:", self.filled_material)
        layout.addRow("有效性:", self.effectiveness)
        layout.addRow(modify_btn)
        self.setLayout(layout)
        self._current_pore = None

    def set_pore(self, pore):
        """显示某个孔洞的参数."""
        self._current_pore = pore
        if pore is None:
            self.pore_id.setText("")
            return
        self.pore_id.setText(str(pore.id))
        self.diameter.setValue(pore.diameter_real)
        self.area.setValue(pore.area_real)
        idx = self.filled_status.findText(pore.filled_status.value)
        if idx >= 0:
            self.filled_status.setCurrentIndex(idx)
        idx = self.filled_material.findText(pore.filled_material.value)
        if idx >= 0:
            self.filled_material.setCurrentIndex(idx)
        idx = self.effectiveness.findText(pore.effectiveness.value)
        if idx >= 0:
            self.effectiveness.setCurrentIndex(idx)

    def get_current_params(self):
        """读取当前 UI 中的参数(用于修改)."""
        from rockpore.core.analysis import FilledStatus, FilledMaterial, Effectiveness
        return {
            "filled_status": FilledStatus(self.filled_status.currentText()),
            "filled_material": FilledMaterial(self.filled_material.currentText()),
            "effectiveness": Effectiveness(self.effectiveness.currentText()),
        }
