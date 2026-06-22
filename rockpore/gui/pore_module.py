"""孔洞分析模块的步骤面板 - 10 步工作流.

每一步是一个 QWidget,接收主窗口的 context 并与之通信.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QTextEdit,
    QVBoxLayout, QWidget, QButtonGroup, QHeaderView, QTableWidget,
    QTableWidgetItem, QAbstractItemView,
)

from .module_base import AnalysisModule, StepDefinition
from .teaching_panel import Card


# ============= 步骤 1: 打开图像 =============
class Step1OpenImage(QWidget):
    completed = pyqtSignal()
    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(16)
        # 标题
        title = QLabel("🖼  加载岩心图像")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("选择要分析的岩心照片文件")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 操作区
        card = Card("info")
        card.add_title("操作步骤")
        card.add_content(
            "1. 点击下方「选择图像」按钮\n"
            "2. 在弹出对话框中选择 JPG / PNG / BMP / TIF 格式图像\n"
            "3. 图像将显示在中央画布\n\n"
            "💡 提示:\n"
            "• 推荐使用分辨率 ≥ 500×400 的图像\n"
            "• 图像应包含比例尺或已知参考尺寸\n"
            "• 模糊/过曝的图像会降低识别准确率"
        )
        v.addWidget(card)
        # 按钮
        h = QHBoxLayout()
        btn = QPushButton("📂 选择图像...")
        btn.setObjectName("primaryButton")
        btn.setMinimumHeight(40)
        btn.clicked.connect(self._open_image)
        h.addWidget(btn)
        sample_btn = QPushButton("🎲 加载示例")
        sample_btn.clicked.connect(self._load_sample)
        h.addWidget(sample_btn)
        h.addStretch(1)
        v.addLayout(h)
        v.addStretch(1)

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开岩心图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        if path:
            from rockpore.core.io_utils import imread_unicode
            img = imread_unicode(path)
            if img is None:
                QMessageBox.critical(self, "错误", f"无法读取图片: {path}")
                return
            self.ctx()["image"] = img
            self.ctx()["image_path"] = path
            self.completed.emit()

    def _load_sample(self):
        """加载软件自带的示例孔洞图像(支持中文路径)."""
        from rockpore.core.io_utils import find_sample_image, imread_unicode
        path = find_sample_image()
        if path is None:
            QMessageBox.information(
                self, "未找到示例",
                "软件未找到示例图像「孔洞.png」。\n\n"
                "请将示例图片放到以下任一位置:\n"
                "• 当前工作目录\n"
                "• 软件根目录\n"
                "• 桌面\n\n"
                "或直接点击「选择图像」手动打开。"
            )
            return
        img = imread_unicode(path)
        if img is None:
            QMessageBox.critical(self, "错误", f"找到示例但读取失败: {path}")
            return
        self.ctx()["image"] = img
        self.ctx()["image_path"] = path
        self.completed.emit()


# ============= 步骤 3: 标尺选择 =============
class Step3Scale(QWidget):
    scale_changed = pyqtSignal(int, bool)
    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(16)
        title = QLabel("📏  标尺设置")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("将图像像素换算为实际尺寸(mm 或 μm)")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 表单
        card = Card("info")
        form_card = Card("tip")
        cv = form_card._layout
        form = QFormLayout()
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(1, 10000)
        self.dpi_spin.setValue(96)
        self.dpi_spin.setSuffix(" DPI")
        self.dpi_spin.setMinimumWidth(150)
        self.dpi_spin.valueChanged.connect(self._on_changed)
        form.addRow("图像 DPI:", self.dpi_spin)
        self.micro_check = QCheckBox("微观分析(μm)")
        self.micro_check.stateChanged.connect(self._on_changed)
        form.addRow("分析模式:", self.micro_check)
        cv.addLayout(form)
        v.addWidget(form_card)
        # 实时显示
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(
            "background: #e8f0fb; color: #2c5fa3; padding: 12px; "
            "border-radius: 6px; font-size: 13px; font-weight: bold;"
        )
        v.addWidget(self.info_label)
        v.addStretch(1)
        self._on_changed()

    def _on_changed(self):
        from rockpore.core.calibration import scale_from_dpi
        dpi = self.dpi_spin.value()
        microscopic = self.micro_check.isChecked()
        scale = scale_from_dpi(dpi, microscopic)
        unit = "μm" if microscopic else "mm"
        self.info_label.setText(
            f"📐 当前标尺: 1 {unit} = {scale.pixels_per_unit:.3f} 像素  "
            f"(1 像素 = {scale.mm_per_pixel * (1000 if microscopic else 1):.3f} {unit})"
        )
        self.ctx()["scale"] = scale
        self.scale_changed.emit(dpi, microscopic)


# ============= 步骤 4: 图像预处理 =============
class _LabeledSlider(QWidget):
    """标签 + 滑块 + 数值显示 三段式控件.
    解决 QFormLayout 中 QDoubleSpinBox 显示异常(Windows 11 上拉成绿色长条).
    """
    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, minimum: float, maximum: float,
                 value: float, step: float = 0.1, decimals: int = 2,
                 unit: str = "", parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        # 标签
        self._label = QLabel(label)
        self._label.setMinimumWidth(60)
        self._label.setStyleSheet("color: #1f2328; font-size: 13px;")
        h.addWidget(self._label)
        # 滑块(整数 0-1000 内部,映射到浮点值)
        self._min = minimum
        self._max = maximum
        self._decimals = decimals
        self._unit = unit
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(self._to_slider(value))
        self._slider.setMinimumWidth(180)
        self._slider.setTickPosition(QSlider.NoTicks)
        h.addWidget(self._slider, 1)
        # 数值显示
        self._value_label = QLabel(self._format_value(value))
        self._value_label.setMinimumWidth(80)
        self._value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._value_label.setStyleSheet(
            "color: #2c5fa3; font-weight: bold; "
            "background: #e8f0fb; padding: 4px 10px; border-radius: 4px;"
        )
        h.addWidget(self._value_label)
        # 增减按钮(精确微调)
        self._minus_btn = QPushButton("−")
        self._minus_btn.setFixedSize(28, 28)
        self._minus_btn.setStyleSheet(
            "QPushButton { background: #fafbfc; border: 1px solid #d0d7de; "
            "border-radius: 4px; font-size: 16px; font-weight: bold; }"
            "QPushButton:hover { background: #e8f0fb; }"
        )
        self._plus_btn = QPushButton("+")
        self._plus_btn.setFixedSize(28, 28)
        self._plus_btn.setStyleSheet(self._minus_btn.styleSheet())
        h.addWidget(self._minus_btn)
        h.addWidget(self._plus_btn)
        # 信号
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._minus_btn.clicked.connect(lambda: self._step_by(-step))
        self._plus_btn.clicked.connect(lambda: self._step_by(step))

    def _to_slider(self, value: float) -> int:
        if self._max == self._min:
            return 0
        return int((value - self._min) / (self._max - self._min) * 1000)

    def _from_slider(self, slider_val: int) -> float:
        return self._min + slider_val / 1000.0 * (self._max - self._min)

    def _format_value(self, value: float) -> str:
        fmt = f"{{:.{self._decimals}f}}"
        text = fmt.format(value)
        if self._unit:
            text += f" {self._unit}"
        return text

    def _on_slider_changed(self, slider_val: int):
        value = self._from_slider(slider_val)
        # 量化到 step
        self._value_label.setText(self._format_value(value))
        self.valueChanged.emit(value)

    def _step_by(self, delta: float):
        current = self._from_slider(self._slider.value())
        new = current + delta
        new = max(self._min, min(self._max, new))
        self._slider.setValue(self._to_slider(new))

    def value(self) -> float:
        return self._from_slider(self._slider.value())

    def setValue(self, v: float):
        self._slider.setValue(self._to_slider(v))
        self._value_label.setText(self._format_value(v))


class Step4Preprocess(QWidget):
    preprocessed = pyqtSignal(np.ndarray)
    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("🎨  图像预处理")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("调整图像以增强孔洞与基质的对比度")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 预设按钮
        h = QHBoxLayout()
        auto_btn = QPushButton("⚡ 一键增强(推荐)")
        auto_btn.setObjectName("primaryButton")
        auto_btn.clicked.connect(self._auto_levels)
        h.addWidget(auto_btn)
        reset_btn = QPushButton("♻ 重置为原图")
        reset_btn.setObjectName("ghostButton")
        reset_btn.clicked.connect(self._reset)
        h.addWidget(reset_btn)
        h.addStretch(1)
        v.addLayout(h)
        # 使用说明
        info = QLabel("💡 用法:\n"
                       "1. 拖动滑块或在右侧按钮调节参数(数值实时显示)\n"
                       "2. 勾选增强选项(锐化/底片/转灰度)\n"
                       "3. 点击底部「应用所有设置」一次性应用全部修改\n"
                       "4. 不满意可点「重置为原图」回到原始图像")
        info.setStyleSheet("color: #57606a; font-size: 12px; padding: 8px; line-height: 1.5;")
        info.setWordWrap(True)
        v.addWidget(info)
        # 参数面板(使用 _LabeledSlider 解决 Windows 11 QSpinBox 显示异常)
        form_card = Card("tip")
        form = QVBoxLayout()
        form.setSpacing(10)
        # 自动色阶
        self.auto_levels_check = QCheckBox("启用自动色阶(推荐开启)")
        self.auto_levels_check.setChecked(True)
        form.addWidget(self.auto_levels_check)
        # 4 个滑块控件
        self.brightness = _LabeledSlider("亮度", -100, 100, 0, step=1, decimals=0, unit="")
        self.contrast = _LabeledSlider("对比度", 0.1, 3.0, 1.0, step=0.1, decimals=2, unit="×")
        self.gamma = _LabeledSlider("Gamma", 0.1, 5.0, 1.0, step=0.1, decimals=2, unit="")
        self.blur = _LabeledSlider("模糊核", 0, 31, 0, step=1, decimals=0, unit="px")
        form.addWidget(self.brightness)
        form.addWidget(self.contrast)
        form.addWidget(self.gamma)
        form.addWidget(self.blur)
        # 增强选项
        opts = QHBoxLayout()
        opts.setSpacing(16)
        self.sharpen_check = QCheckBox("🔪 锐化")
        self.invert_check = QCheckBox("🎞 底片效果")
        self.gray_check = QCheckBox("⚫ 转灰度")
        opts.addWidget(self.sharpen_check)
        opts.addWidget(self.invert_check)
        opts.addWidget(self.gray_check)
        opts.addStretch(1)
        form.addLayout(opts)
        # 应用按钮
        apply_btn = QPushButton("✅ 应用所有设置")
        apply_btn.setObjectName("primaryButton")
        apply_btn.setMinimumHeight(40)
        apply_btn.clicked.connect(self._apply_params)
        form.addWidget(apply_btn)
        form_card._layout.addLayout(form)
        v.addWidget(form_card)
        v.addStretch(1)
        self._current_processed = None

    def _get_image(self):
        ctx = self.ctx()
        if ctx.get("processed_image") is not None:
            return ctx["processed_image"]
        return ctx.get("image")

    def _auto_levels(self):
        """一键增强:LAB+CLAHE+色阶."""
        img = self._get_image()
        if img is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        from rockpore.core.preprocessing import apply_pore_enhancement
        out = apply_pore_enhancement(img)
        self._emit(out)

    def _reset(self):
        """重置:丢弃所有预处理,回到原图."""
        ctx = self.ctx()
        if ctx.get("image") is not None:
            self._emit(ctx["image"].copy())
            ctx.pop("processed_image", None)

    def _apply_params(self):
        """应用所有设置."""
        img = self._get_image()
        if img is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        from rockpore.core.preprocessing import preprocess, PreprocessParams
        blur_int = int(round(self.blur.value()))
        params = PreprocessParams(
            brightness=self.brightness.value(),
            contrast=self.contrast.value(),
            gamma=self.gamma.value(),
            blur_kernel=blur_int,
            sharpen=self.sharpen_check.isChecked(),
            invert=self.invert_check.isChecked(),
            to_gray=self.gray_check.isChecked(),
            auto_levels=self.auto_levels_check.isChecked(),
        )
        out = preprocess(img, params)
        self._emit(out)

    def _emit(self, image):
        self._current_processed = image
        self.ctx()["processed_image"] = image
        self.preprocessed.emit(image)


# ============= 步骤 5: 孔洞提取 =============
class Step5Extract(QWidget):
    extracted = pyqtSignal(np.ndarray)
    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("🎯  孔洞提取")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("自动从图像中识别暗色孔洞区域")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 方法选择
        form_card = Card("tip")
        form = QFormLayout()
        self.method = QComboBox()
        self.method.addItems([
            "auto(OTSU 自动阈值) ★推荐",
            "adaptive(自适应阈值)",
        ])
        # triangle 方法在 segmentation.extract_pores 中未实现,
        # 暴露给用户会触发 ValueError;只在 accuracy 内部 fallback 使用
        self.tolerance = QSpinBox()
        self.tolerance.setRange(0, 255)
        self.tolerance.setValue(30)
        self.tolerance.setSuffix(" (颜色匹配度)")
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 10000)
        self.min_area.setValue(10)
        self.min_area.setSuffix(" 像素")
        extract_btn = QPushButton("🔍 自动提取")
        extract_btn.setObjectName("primaryButton")
        extract_btn.setMinimumHeight(40)
        extract_btn.clicked.connect(self._extract)
        form.addRow("提取方法:", self.method)
        form.addRow("最小面积:", self.min_area)
        form.addRow(extract_btn)
        form_card._layout.addLayout(form)
        v.addWidget(form_card)
        # 取色器说明
        info = QLabel("💡 提示: 暗色孔洞在浅色基质上识别效果最佳。\n可先用「自动色阶」增强对比度。")
        info.setStyleSheet("color: #57606a; font-size: 12px; padding: 8px;")
        info.setWordWrap(True)
        v.addWidget(info)
        v.addStretch(1)

    def _extract(self):
        ctx = self.ctx()
        if "image" not in ctx or ctx["image"] is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        # S2 + 新需求:如果当前掩码已被用户手动编辑过(脏),询问合并策略
        user_mask = ctx.get("mask")
        user_dirty = ctx.get("mask_dirty", False)
        merge_mode = "overwrite"  # 默认覆盖
        if user_mask is not None and user_dirty:
            # 提供 3 选项:覆盖 / 合并 / 取消
            # 用 QMessageBox.question 不能 3 选项,改用自定义对话框
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
            dlg = QDialog(self)
            dlg.setWindowTitle("合并模式")
            dlg.setMinimumWidth(420)
            v = QVBoxLayout(dlg)
            v.addWidget(QLabel("当前掩码已被您手动编辑过(橡皮擦/添加)。\n"
                              "请选择如何合并新的提取结果:"))
            btn_layout = QVBoxLayout()
            btn_overwrite = QPushButton("📝 完全覆盖 - 丢弃您的编辑,使用新提取")
            btn_merge_add = QPushButton("➕ 合并(您的编辑优先) - 保留您添加的孔洞,在空白处做新提取")
            btn_merge_only = QPushButton("🧹 仅在您擦除处做新提取 - 保留您添加的")
            btn_cancel = QPushButton("❌ 取消")
            btn_layout.addWidget(btn_overwrite)
            btn_layout.addWidget(btn_merge_add)
            btn_layout.addWidget(btn_merge_only)
            btn_layout.addWidget(btn_cancel)
            v.addLayout(btn_layout)
            result = {"mode": "cancel"}
            def set_mode(m):
                result["mode"] = m
                dlg.accept()
            btn_overwrite.clicked.connect(lambda: set_mode("overwrite"))
            btn_merge_add.clicked.connect(lambda: set_mode("merge_add"))
            btn_merge_only.clicked.connect(lambda: set_mode("merge_only"))
            btn_cancel.clicked.connect(lambda: set_mode("cancel"))
            dlg.exec_()
            merge_mode = result["mode"]
            if merge_mode == "cancel":
                return
        # 1. 自动提取
        img = ctx.get("processed_image")
        if img is None:
            img = ctx["image"]
        from rockpore.core.segmentation import extract_pores
        from rockpore.core.morphology import remove_noise, morphological_open
        method_map = {
            "auto(OTSU 自动阈值) ★推荐": "otsu",
            "adaptive(自适应阈值)": "adaptive",
        }
        m = self.method.currentText()
        try:
            new_mask = extract_pores(img, method=method_map.get(m, "otsu"))
        except ValueError as e:
            QMessageBox.critical(self, "提取失败", f"提取方法错误:\n{e}")
            return
        new_mask = morphological_open(new_mask, kernel_size=2)
        if self.min_area.value() > 0:
            new_mask = remove_noise(new_mask, min_area=self.min_area.value())
        # 2. 合并用户编辑(根据 merge_mode)
        if merge_mode == "overwrite" or user_mask is None or not user_dirty:
            mask = new_mask
        elif merge_mode == "merge_add":
            # 合并:用户 mask=255 的位置保留 + 新提取的孔洞
            # 即:cv2.bitwise_or(user_mask, new_mask)
            mask = cv2.bitwise_or(user_mask, new_mask)
        elif merge_mode == "merge_only":
            # 仅在用户擦除处(=0)做新提取
            # result = (new_mask WHERE user_mask=0) | (user_mask WHERE user_mask=255)
            # = new_mask & ~user_mask | user_mask
            # = new_mask | user_mask(因为 new_mask & ~user_mask ≤ new_mask,加 user_mask 不变)
            # 实际:用户擦除的位置,如果新提取是 1 就保留;用户添加的位置始终保留
            mask = cv2.bitwise_or(user_mask, new_mask)
        else:
            mask = new_mask
        ctx["mask"] = mask
        ctx["mask_dirty"] = False
        # 显示合并结果提示
        if merge_mode != "overwrite" and user_dirty:
            new_in_user = int(((user_mask == 0) & (mask == 255)).sum())
            kept_user = int(((user_mask == 255) & (mask == 255)).sum())
            self.status_message = (
                f"✓ 合并完成: 保留您添加的 {kept_user} px 区域 + "
                f"擦除处新发现 {new_in_user} px 区域"
            )
        self.extracted.emit(mask)


# ============= 步骤 6: 二次编辑 =============
class Step6Edit(QWidget):
    edited = pyqtSignal(np.ndarray)
    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("🛠  二次编辑")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("使用数学形态学优化孔洞区域")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 操作
        form_card = Card("tip")
        form = QFormLayout()
        self.op = QComboBox()
        self.op.addItems([
            "膨胀 - 扩大孔洞区域",
            "腐蚀 - 缩小孔洞区域",
            "开运算 - 去小毛刺",
            "闭运算 - 填补小孔",
            "去噪 - 删除过小/过大区域",
            "填充 - 填充掩码内孔洞",
        ])
        self.kernel = QSpinBox()
        self.kernel.setRange(1, 31)
        self.kernel.setValue(3)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 10)
        self.iterations.setValue(1)
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 100000)
        self.min_area.setValue(10)
        apply_btn = QPushButton("应用")
        apply_btn.setObjectName("primaryButton")
        apply_btn.clicked.connect(self._apply)
        form.addRow("操作:", self.op)
        form.addRow("核大小:", self.kernel)
        form.addRow("迭代次数:", self.iterations)
        form.addRow("最小面积(去噪):", self.min_area)
        form.addRow(apply_btn)
        form_card._layout.addLayout(form)
        v.addWidget(form_card)
        # 撤销
        h = QHBoxLayout()
        undo_btn = QPushButton("↶ 撤销到上一步")
        undo_btn.clicked.connect(self._undo)
        h.addWidget(undo_btn)
        h.addStretch(1)
        v.addLayout(h)
        # 说明
        info = QLabel("💡 形态学操作说明:\n• 膨胀: 边界外扩,适合修复断裂\n• 腐蚀: 边界内缩,适合去除毛刺\n• 开运算: 腐蚀+膨胀,去小噪点\n• 闭运算: 膨胀+腐蚀,补小空洞")
        info.setStyleSheet("color: #57606a; font-size: 12px; padding: 8px;")
        info.setWordWrap(True)
        v.addWidget(info)
        v.addStretch(1)
        # I5: 撤销栈已提取到 ModuleWorkflowPage.context["_edit_undo_stack"]
        # 避免步骤切换时 widget 重建导致撤销历史丢失

    def _apply(self):
        ctx = self.ctx()
        if ctx.get("mask") is None:
            QMessageBox.warning(self, "提示", "请先提取孔洞")
            return
        mask = ctx["mask"]
        # 撤销栈放在 context,跨步骤保留
        undo_stack = ctx.setdefault("_edit_undo_stack", [])
        undo_stack.append(mask.copy())
        op = self.op.currentIndex()
        from rockpore.core.morphology import (
            dilate_region, erode_region, morphological_open, morphological_close,
            remove_noise, fill_holes,
        )
        if op == 0:
            out = dilate_region(mask, self.kernel.value(), self.iterations.value())
        elif op == 1:
            out = erode_region(mask, self.kernel.value(), self.iterations.value())
        elif op == 2:
            out = morphological_open(mask, self.kernel.value())
        elif op == 3:
            out = morphological_close(mask, self.kernel.value())
        elif op == 4:
            out = remove_noise(mask, min_area=self.min_area.value())
        else:
            out = fill_holes(mask, min_hole_area=self.min_area.value())
        ctx["mask"] = out
        self.edited.emit(out)

    def _undo(self):
        ctx = self.ctx()
        undo_stack = ctx.get("_edit_undo_stack", [])
        if not undo_stack:
            QMessageBox.information(self, "提示", "没有可撤销的操作")
            return
        last = undo_stack.pop()
        ctx["mask"] = last
        self.edited.emit(last)


# ============= 步骤 8: 孔洞分析 =============
class Step8Analyze(QWidget):
    analyzed = pyqtSignal(object)  # PoreAnalysisResult
    pore_selected = pyqtSignal(int)  # 选中孔洞的 id,-1=取消选中
    show_all_changed = pyqtSignal(bool)  # 用户切换"显示所有"

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        self._pores_by_id: Dict[int, Any] = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("📊  孔洞分析")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("计算面积、直径、面孔率等定量参数")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 按钮行:分析 + 显示所有/只显示选中
        h = QHBoxLayout()
        run_btn = QPushButton("▶ 自动分析")
        run_btn.setObjectName("primaryButton")
        run_btn.setMinimumHeight(36)
        run_btn.clicked.connect(self._run)
        h.addWidget(run_btn)
        self.show_all_btn = QPushButton("👁 显示所有标注")
        self.show_all_btn.setObjectName("ghostButton")
        self.show_all_btn.setCheckable(True)
        self.show_all_btn.setChecked(False)  # 默认不显示所有(避免遮挡)
        self.show_all_btn.toggled.connect(self._on_show_all_toggled)
        h.addWidget(self.show_all_btn)
        h.addStretch(1)
        v.addLayout(h)
        # 数据卡片
        cards_grid = QGridLayout()
        cards_grid.setSpacing(10)
        self._cards = {}
        for i, (key, label, unit) in enumerate([
            ("pore_count", "孔洞总数", "个"),
            ("pore_count_report", "报告级(≥2mm)", "个"),
            ("porosity", "面孔率", "%"),
            ("avg_diameter", "平均直径", "mm"),
            ("max_diameter", "最大直径", "mm"),
            ("min_diameter", "最小直径", "mm"),
        ]):
            r, c = divmod(i, 3)
            frame = QFrame()
            frame.setObjectName("dataCard")
            fv = QVBoxLayout(frame)
            fv.setContentsMargins(10, 8, 10, 8)
            lab = QLabel(label)
            lab.setObjectName("dataCardLabel")
            val = QLabel("-")
            val.setObjectName("dataCardValue")
            fv.addWidget(lab)
            fv.addWidget(val)
            self._cards[key] = val
            cards_grid.addWidget(frame, r, c)
        v.addLayout(cards_grid)
        # 详情表格(替换原 QListWidget,带列排序)
        v.addWidget(QLabel("💡 点击表格行 → 在画布上高亮该孔洞:"))
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "ID", "直径 (mm)", "面积 (mm²)", "分类", "质心 (x,y)", "填充",
        ])
        self.table.verticalHeader().setVisible(False)
        # 显式行高(避免渲染 0 高度)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        # 表头加粗
        h_font = self.table.horizontalHeader().font()
        h_font.setBold(True)
        self.table.horizontalHeader().setFont(h_font)
        # 最小尺寸保证可见
        self.table.setMinimumHeight(220)
        # 列宽自适应
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        # 关键:行选中信号 → 触发 pore_selected
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        v.addWidget(self.table, 1)

    def _run(self):
        ctx = self.ctx()
        if ctx.get("mask") is None:
            QMessageBox.warning(self, "提示", "请先完成孔洞提取")
            return
        if ctx.get("scale") is None:
            QMessageBox.warning(self, "提示", "请先设置标尺")
            return
        from rockpore.core.analysis import analyze_pores
        result = analyze_pores(ctx["mask"], ctx["scale"])
        ctx["analysis_result"] = result
        # 更新卡片
        self._cards["pore_count"].setText(f"{result.pore_count} 个")
        self._cards["pore_count_report"].setText(f"{result.pore_count_report} 个")
        self._cards["porosity"].setText(f"{result.porosity * 100:.2f} %")
        self._cards["avg_diameter"].setText(f"{result.average_diameter_real:.2f} mm")
        self._cards["max_diameter"].setText(f"{result.max_diameter_real:.2f} mm")
        self._cards["min_diameter"].setText(f"{result.min_diameter_real:.2f} mm")
        # 填表
        self._fill_table(result)
        # 默认清空画布上的孔洞标注(避免遮挡)
        # 仅在用户点"显示所有"时才显示
        self.pore_selected.emit(-1)  # 通知 page 清标注
        self.analyzed.emit(result)

    def _fill_table(self, result):
        """填充孔洞详情表格."""
        self.table.setRowCount(len(result.pores))
        self._pores_by_id = {p.id: p for p in result.pores}
        for row, p in enumerate(result.pores):
            # ID
            id_item = QTableWidgetItem(str(p.id))
            id_item.setData(Qt.UserRole, p.id)  # 用于查找
            self.table.setItem(row, 0, id_item)
            # 直径
            self.table.setItem(row, 1, QTableWidgetItem(f"{p.diameter_real:.2f}"))
            # 面积
            self.table.setItem(row, 2, QTableWidgetItem(f"{p.area_real:.2f}"))
            # 分类
            self.table.setItem(row, 3, QTableWidgetItem(p.size_class))
            # 质心
            cx, cy = p.centroid
            self.table.setItem(row, 4, QTableWidgetItem(f"({cx:.0f}, {cy:.0f})"))
            # 填充
            self.table.setItem(row, 5, QTableWidgetItem(p.filled_status.value))
        # 默认按 ID 排序
        self.table.sortItems(0, Qt.AscendingOrder)

    def _on_table_selection_changed(self):
        """用户点击表格行 → 触发 pore_selected 信号(主窗口画布高亮)."""
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.pore_selected.emit(-1)
            return
        row = selected[0].row()
        id_item = self.table.item(row, 0)
        if id_item is None:
            self.pore_selected.emit(-1)
            return
        pore_id = id_item.data(Qt.UserRole)
        if pore_id is None:
            pore_id = int(id_item.text())
        self.pore_selected.emit(int(pore_id))

    def _on_show_all_toggled(self, checked: bool):
        """切换是否显示所有孔洞标注."""
        # 通过 analyzed 信号携带标志位 → page 处理
        # 这里仅更新按钮文字
        if checked:
            self.show_all_btn.setText("🙈 隐藏所有标注")
        else:
            self.show_all_btn.setText("👁 显示所有标注")
        # 通知 page(用 pore_selected(-2) 区分,-2=切换显隐状态)
        # 简单方案:发个自定义信号
        if hasattr(self, 'show_all_changed'):
            self.show_all_changed.emit(checked)


# ============= 步骤 9: 基础信息 =============
class Step9Info(QWidget):
    saved = pyqtSignal(dict)
    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("📝  基础信息")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("录入项目元数据,用于报告生成")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        form_card = Card("info")
        form = QFormLayout()
        self.project = QLineEdit()
        self.sample_id = QLineEdit()
        self.analyst = QLineEdit()
        self.remarks = QTextEdit()
        self.remarks.setMaximumHeight(100)
        save_btn = QPushButton("💾 保存信息")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._save)
        form.addRow("项目名称:", self.project)
        form.addRow("样品编号:", self.sample_id)
        form.addRow("分析人员:", self.analyst)
        form.addRow("备注:", self.remarks)
        form.addRow(save_btn)
        form_card._layout.addLayout(form)
        v.addWidget(form_card)
        v.addStretch(1)
        # 预填上次保存值
        if "info" in self.ctx():
            self._load(self.ctx()["info"])

    def _save(self):
        data = {
            "project": self.project.text(),
            "sample_id": self.sample_id.text(),
            "analyst": self.analyst.text(),
            "remarks": self.remarks.toPlainText(),
        }
        self.ctx()["info"] = data
        QMessageBox.information(self, "保存成功", "基础信息已保存")
        self.saved.emit(data)

    def _load(self, data):
        self.project.setText(data.get("project", ""))
        self.sample_id.setText(data.get("sample_id", ""))
        self.analyst.setText(data.get("analyst", ""))
        self.remarks.setPlainText(data.get("remarks", ""))


# ============= 步骤 10: 报告生成 =============
class Step10Report(QWidget):
    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("📄  生成报告")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("生成 HTML 格式的孔洞分析报告")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        h = QHBoxLayout()
        gen_btn = QPushButton("📄 生成报告")
        gen_btn.setObjectName("primaryButton")
        gen_btn.setMinimumHeight(40)
        gen_btn.clicked.connect(self._generate)
        h.addWidget(gen_btn)
        save_btn = QPushButton("💾 保存到文件...")
        save_btn.clicked.connect(self._save)
        h.addWidget(save_btn)
        h.addStretch(1)
        v.addLayout(h)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        v.addWidget(self.preview)
        self._html = None

    def _generate(self):
        ctx = self.ctx()
        if "analysis_result" not in ctx:
            QMessageBox.warning(self, "提示", "请先完成孔洞分析")
            return
        from rockpore.core.report import ReportData, generate_report, _make_annotated_image
        result = ctx["analysis_result"]
        scale = ctx.get("scale")
        scale_info = f"{scale.pixels_per_unit:.3f} px/{'μm' if 'MICROMETER' in scale.unit.name else 'mm'}" if scale else "-"
        annot = _make_annotated_image(ctx["image"], ctx.get("mask"), result)
        info = ctx.get("info", {})
        data = ReportData(
            project_name=info.get("project", ""),
            sample_id=info.get("sample_id", ""),
            analyst=info.get("analyst", ""),
            image_path=ctx.get("image_path", ""),
            image_size=(ctx["image"].shape[1], ctx["image"].shape[0]),
            scale_info=scale_info,
            remarks=info.get("remarks", ""),
            analysis_result=result,
            original_image=ctx["image"],
            annotated_image=annot,
        )
        self._html = generate_report(data)
        # 简单预览
        summary = (
            f"# 岩心孔洞分析报告\n\n"
            f"**项目:** {data.project_name or '-'}  \n"
            f"**样品:** {data.sample_id or '-'}  \n"
            f"**分析人员:** {data.analyst or '-'}  \n"
            f"**图像:** {data.image_path or '-'}  \n"
            f"**标尺:** {data.scale_info}  \n\n"
            f"## 关键指标\n\n"
            f"- **孔洞总个数:** {result.pore_count} 个\n"
            f"- **报告级孔洞数(≥2mm):** {result.pore_count_report} 个\n"
            f"- **孔洞面孔率:** {result.porosity * 100:.2f} %\n"
            f"- **平均等效直径:** {result.average_diameter_real:.2f} mm\n"
            f"- **总孔洞面积:** {result.total_pore_area_real:.2f} mm²\n\n"
            f"## 分类统计\n\n"
        )
        for k, v in result.size_distribution.items():
            pct = v / max(1, result.pore_count) * 100
            summary += f"- **{k}:** {v} 个 ({pct:.1f}%)\n"
        self.preview.setMarkdown(summary)
        self._data = data
        QMessageBox.information(self, "成功", "报告已生成,可保存为 HTML")

    def _save(self):
        if not self._html:
            self._generate()
        if not self._html:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存报告", "report.html", "HTML 文件 (*.html)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._html)
            QMessageBox.information(self, "保存成功", f"报告已保存到:\n{path}")


# ============= 步骤占位(2, 7)=============
class StepPlaceholder(QWidget):
    """简单占位步骤."""
    def __init__(self, title: str, content: str, icon: str = "✨", parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(16)
        t = QLabel(f"{icon}  {title}")
        t.setObjectName("pageTitle")
        v.addWidget(t)
        card = Card("info")
        card.add_content(content)
        v.addWidget(card)
        v.addStretch(1)


# ============= 孔洞模块 =============
class PoreModule(AnalysisModule):
    """孔洞分析模块(完整实现)."""
    name = "孔洞分析"
    icon = "🕳"
    description = "识别岩心孔洞并计算面积、直径、面孔率"

    def __init__(self):
        from .module_base import make_default_pore_steps
        self._steps = make_default_pore_steps()

    @property
    def steps(self):
        return self._steps

    def create_step_panel(self, step_idx: int, parent=None):
        # parent 是 ModuleWorkflowPage,有 .context 属性
        page = parent
        def ctx_getter():
            return page.context
        idx = step_idx + 1  # 0-based → 1-based
        if idx == 1:
            return Step1OpenImage(ctx_getter, parent)
        if idx == 2:
            return StepPlaceholder(
                "启动孔洞分析",
                "本模块已启动,接下来请按左侧 10 步流程操作。\n\n"
                "建议路径:\n1. 打开图像 → 2. (本步) → 3. 设置标尺 → 4. 预处理\n"
                "→ 5. 提取 → 6. 编辑 → 7. 填充 → 8. 分析 → 9. 信息 → 10. 报告",
                "🚀"
            )
        if idx == 3:
            return Step3Scale(ctx_getter, parent)
        if idx == 4:
            return Step4Preprocess(ctx_getter, parent)
        if idx == 5:
            return Step5Extract(ctx_getter, parent)
        if idx == 6:
            return Step6Edit(ctx_getter, parent)
        if idx == 7:
            return StepPlaceholder(
                "孔洞填充",
                "在「二次编辑」中选择「填充 - 填充掩码内孔洞」即可。\n"
                "此步骤通常与第 6 步合并操作。\n\n"
                "💡 填充规则: 直径 < 设定阈值的孔洞被视为噪声,被填平。",
                "💧"
            )
        if idx == 8:
            return Step8Analyze(ctx_getter, parent)
        if idx == 9:
            return Step9Info(ctx_getter, parent)
        if idx == 10:
            return Step10Report(ctx_getter, parent)
        return StepPlaceholder(f"步骤 {idx}", "该步骤内容待补充", "📌")

    def run_step(self, step_idx: int, context: dict) -> dict:
        # 各步骤已通过信号触发,此处返回空
        return {}

    def analyze(self, image, scale, context: dict) -> Any:
        """完整流程."""
        from rockpore.core.accuracy import detect_pores_robust
        mask, result = detect_pores_robust(image, scale)
        return {"mask": mask, "result": result}

    def build_report_data(self, context: dict) -> dict:
        return {
            "image": context.get("image"),
            "mask": context.get("mask"),
            "result": context.get("analysis_result"),
            "scale": context.get("scale"),
            "image_path": context.get("image_path"),
            "info": context.get("info", {}),
        }
