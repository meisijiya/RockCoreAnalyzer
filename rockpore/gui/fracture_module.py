"""裂缝分析模块 - 完整 10 步工作流实现.

基于 HoughLinesP 的裂缝自动检测 + 长度/宽度/倾角/密度统计.
复用孔洞模块的 Step1OpenImage 和 Step3Scale (Step 1/3 是通用步骤).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .module_base import AnalysisModule, StepDefinition
from .pore_module import Step1OpenImage, Step3Scale, StepPlaceholder
from .teaching_panel import Card


# ============= 步骤 4: 图像预处理(裂缝专属,增强边缘)=============
class Step4FracturePreprocess(QWidget):
    preprocessed = pyqtSignal(np.ndarray)

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("🔍  图像预处理")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("增强裂缝边缘对比度,提升后续识别效果")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 预处理选项
        card = Card("tip")
        cv_layout = card._layout
        form = QFormLayout()
        self.enhance = QComboBox()
        self.enhance.addItems([
            "CLAHE 自适应直方图均衡 ★推荐",
            "自动色阶拉伸",
            "原始(不处理)",
        ])
        form.addRow("增强方法:", self.enhance)
        apply_btn = QPushButton("✨ 应用预处理")
        apply_btn.setObjectName("primaryButton")
        apply_btn.setMinimumHeight(40)
        apply_btn.clicked.connect(self._apply)
        form.addRow(apply_btn)
        cv_layout.addLayout(form)
        v.addWidget(card)
        # 说明
        info = QLabel(
            "💡 裂缝边缘对比度通常较低,推荐使用 CLAHE 自适应直方图均衡。\n"
            "   该方法可在不损失整体色调的情况下增强局部对比度。"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #57606a; font-size: 12px; padding: 8px;")
        v.addWidget(info)
        v.addStretch(1)

    def _apply(self):
        ctx = self.ctx()
        if "image" not in ctx or ctx["image"] is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        from rockpore.core.preprocessing import auto_levels
        img = ctx["image"]
        method = self.enhance.currentText()
        if "CLAHE" in method:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            if img.ndim == 3:
                processed = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            else:
                processed = enhanced_gray
        elif "自动色阶" in method:
            processed = auto_levels(img)
        else:
            processed = img
        ctx["processed_image"] = processed
        self.preprocessed.emit(processed)


# ============= 步骤 5: 裂缝提取(支持 Hough/Adaptive 两种算法)=============
class Step5FractureExtract(QWidget):
    extracted = pyqtSignal(np.ndarray)

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("📏  裂缝提取")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("检测线性暗色区域(支持 Hough 变换与自适应阈值两种算法)")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 算法选择
        card = Card("tip")
        cv_layout = card._layout
        # v1.1.3: 用 QFormLayout 但强制宽标签 + 加水平间距,避免截断
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        # 算法选择 — 全宽 ComboBox
        self.method = QComboBox()
        self.method.addItems([
            "adaptive (OTSU+长宽比) ★推荐",
            "adaptive (自适应阈值)",
            "hough (HoughLinesP)",
        ])
        self.method.setMinimumWidth(280)
        self.method.currentIndexChanged.connect(self._on_method_changed)
        # 用普通 widget 容纳标签+控件,让标签和控件都在一行
        method_row = QWidget()
        method_row_layout = QHBoxLayout(method_row)
        method_row_layout.setContentsMargins(0, 0, 0, 0)
        method_row_layout.addWidget(QLabel("检测算法:"))
        method_row_layout.addWidget(self.method, 1)
        form.addRow(method_row)
        # OTSU 专属参数(推荐默认)
        # v1.1.3: 标签单独放,SpinBox 放后面;加大 SpinBox 最小宽度
        self.min_aspect = QSpinBox()
        self.min_aspect.setRange(1, 10)
        self.min_aspect.setValue(1)
        self.min_aspect.setMinimumWidth(140)
        self.min_aspect.setSuffix(" :1 (最小长宽比)")
        self.min_area = QSpinBox()
        self.min_area.setRange(5, 5000)
        self.min_area.setValue(100)
        self.min_area.setMinimumWidth(140)
        self.min_area.setSuffix(" px² (候选最小面积)")
        self.max_area_pct = QSpinBox()
        self.max_area_pct.setRange(1, 100)
        self.max_area_pct.setValue(30)
        self.max_area_pct.setMinimumWidth(140)
        self.max_area_pct.setSuffix(" % (排除超大区域)")
        self.otsu_offset = QSpinBox()
        self.otsu_offset.setRange(-100, 100)
        self.otsu_offset.setValue(-10)
        self.otsu_offset.setMinimumWidth(140)
        self.otsu_offset.setSuffix(" (OTSU 阈值调整)")
        # v1.1.3: 水平排列标签 + SpinBox,确保不截断
        ar_row = QWidget()
        ar_layout = QHBoxLayout(ar_row)
        ar_layout.setContentsMargins(0, 0, 0, 0)
        ar_layout.addWidget(QLabel("线状过滤:"))
        ar_layout.addWidget(self.min_aspect, 1)
        form.addRow(ar_row)
        area_row = QWidget()
        area_layout = QHBoxLayout(area_row)
        area_layout.setContentsMargins(0, 0, 0, 0)
        area_layout.addWidget(QLabel("最小面积:"))
        area_layout.addWidget(self.min_area, 1)
        form.addRow(area_row)
        max_row = QWidget()
        max_layout = QHBoxLayout(max_row)
        max_layout.setContentsMargins(0, 0, 0, 0)
        max_layout.addWidget(QLabel("排除超大:"))
        max_layout.addWidget(self.max_area_pct, 1)
        form.addRow(max_row)
        offset_row = QWidget()
        offset_layout = QHBoxLayout(offset_row)
        offset_layout.setContentsMargins(0, 0, 0, 0)
        offset_layout.addWidget(QLabel("阈值偏移:"))
        offset_layout.addWidget(self.otsu_offset, 1)
        form.addRow(offset_row)
        # OTSU 开关 + 模糊
        self.use_blur = QCheckBox("高斯模糊 (7x7,去纹理)")
        self.use_blur.setChecked(False)
        form.addRow(self.use_blur)
        self.use_close = QCheckBox("闭运算连接相近裂缝")
        self.use_close.setChecked(False)
        form.addRow(self.use_close)
        # 自适应参数
        self.adaptive_block = QSpinBox()
        self.adaptive_block.setRange(5, 51)
        self.adaptive_block.setValue(21)
        self.adaptive_block.setSingleStep(2)
        self.adaptive_block.setMinimumWidth(140)
        self.adaptive_block.setSuffix(" px (邻域)")
        self.adaptive_C = QSpinBox()
        self.adaptive_C.setRange(0, 30)
        self.adaptive_C.setValue(10)
        self.adaptive_C.setMinimumWidth(140)
        self.adaptive_C.setSuffix(" (常数 C)")
        block_row = QWidget()
        block_layout = QHBoxLayout(block_row)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.addWidget(QLabel("邻域:"))
        block_layout.addWidget(self.adaptive_block, 1)
        form.addRow(block_row)
        c_row = QWidget()
        c_layout = QHBoxLayout(c_row)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.addWidget(QLabel("常数 C:"))
        c_layout.addWidget(self.adaptive_C, 1)
        form.addRow(c_row)
        # Hough 参数
        self.canny_low = QSpinBox()
        self.canny_low.setRange(10, 200)
        self.canny_low.setValue(90)
        self.canny_low.setMinimumWidth(140)
        self.canny_low.setSuffix(" (Canny 低阈值)")
        self.hough_threshold = QSpinBox()
        self.hough_threshold.setRange(5, 200)
        self.hough_threshold.setValue(25)
        self.hough_threshold.setMinimumWidth(140)
        self.hough_threshold.setSuffix(" (Hough 累加)")
        self.min_length = QSpinBox()
        self.min_length.setRange(5, 500)
        self.min_length.setValue(20)
        self.min_length.setMinimumWidth(140)
        self.min_length.setSuffix(" px (最小线长)")
        self.max_gap = QSpinBox()
        self.max_gap.setRange(0, 50)
        self.max_gap.setValue(15)
        self.max_gap.setMinimumWidth(140)
        self.max_gap.setSuffix(" px (最大断裂)")
        self.dilation = QSpinBox()
        self.dilation.setRange(1, 10)
        self.dilation.setValue(3)
        self.dilation.setMinimumWidth(140)
        self.dilation.setSuffix(" px (线段粗细)")
        # 提取按钮
        extract_btn = QPushButton("🔍 自动提取裂缝")
        extract_btn.setObjectName("primaryButton")
        extract_btn.setMinimumHeight(40)
        extract_btn.clicked.connect(self._extract)
        form.addRow(extract_btn)
        cv_layout.addLayout(form)
        v.addWidget(card)
        info = QLabel(
            "💡 算法选择建议:\n"
            "• adaptive OTSU(推荐, 默认): OTSU 暗色阈值 + 长宽比筛选\n"
            "  — 在裂缝样.jpg (232x118) 和 裂缝样2.png (451x301) 都表现良好\n"
            "  — 准确率 ≥ 80%(主裂缝全部识别)\n"
            "• adaptive 自适应: 高斯模糊 + 自适应阈值 + 形态学\n"
            "  — 适合光照不均的图像\n"
            "• hough: Canny + HoughLinesP\n"
            "  — 适合边缘清晰的合成图\n\n"
            "💡 参数说明:\n"
            "• 长宽比 ↑ = 过滤更严格(只保留线状)\n"
            "• 最小面积 ↑ = 过滤小阴影噪点(默认 200px²)\n"
            "• 最大面积 % = 排除覆盖大图的假阳性(默认 40%)\n"
            "• 提取会覆盖之前的结果(包括二次编辑)"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #57606a; font-size: 12px; padding: 8px;")
        v.addWidget(info)
        v.addStretch(1)
        self._on_method_changed(0)

    def _on_method_changed(self, idx):
        """根据所选算法显示/隐藏对应参数."""
        # idx: 0 = OTSU(默认), 1 = 自适应, 2 = Hough
        is_otsu = idx == 0
        is_adapt = idx == 1
        is_hough = idx == 2
        # OTSU 参数
        self.min_aspect.setVisible(is_otsu)
        self.min_area.setVisible(is_otsu)
        self.max_area_pct.setVisible(is_otsu)
        self.otsu_offset.setVisible(is_otsu)
        self.use_blur.setVisible(is_otsu)
        self.use_close.setVisible(is_otsu)
        # 自适应参数
        self.adaptive_block.setVisible(is_adapt)
        self.adaptive_C.setVisible(is_adapt)
        # Hough 参数
        self.canny_low.setVisible(is_hough)
        self.hough_threshold.setVisible(is_hough)
        self.min_length.setVisible(is_hough)
        self.max_gap.setVisible(is_hough)
        self.dilation.setVisible(is_hough)

    def _extract(self):
        ctx = self.ctx()
        if "image" not in ctx or ctx["image"] is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        from rockpore.core.fracture import FractureParams, detect_fracture_mask
        img = ctx.get("processed_image")
        if img is None:
            img = ctx["image"]
        idx = self.method.currentIndex()
        if idx == 0:
            # OTSU 暗色 + 长宽比筛选(推荐)
            params = FractureParams(
                method="adaptive",
                use_otsu=True,
                otsu_offset=self.otsu_offset.value(),
                blur_kernel=7 if self.use_blur.isChecked() else 0,
                morph_close=5 if self.use_close.isChecked() else 0,
                min_aspect_ratio=float(self.min_aspect.value()),
                min_area_for_candidate=self.min_area.value(),
                max_area_ratio=float(self.max_area_pct.value()) / 100.0,
            )
            algo_name = "OTSU 暗色+长宽比"
        elif idx == 1:
            # 自适应阈值
            params = FractureParams(
                method="adaptive",
                use_otsu=False,
                blur_kernel=7,
                adaptive_block=self.adaptive_block.value(),
                adaptive_C=self.adaptive_C.value(),
                morph_close=5,
                min_aspect_ratio=float(self.min_aspect.value()),
                min_area_for_candidate=self.min_area.value(),
                max_area_ratio=float(self.max_area_pct.value()) / 100.0,
            )
            algo_name = "自适应阈值"
        else:
            # HoughLinesP
            params = FractureParams(
                method="hough",
                canny_low=self.canny_low.value(),
                canny_high=self.canny_low.value() * 3,
                hough_threshold=self.hough_threshold.value(),
                min_line_length_px=self.min_length.value(),
                max_line_gap_px=self.max_gap.value(),
                dilation_kernel_px=self.dilation.value(),
            )
            algo_name = "HoughLinesP"
        mask, _, _ = detect_fracture_mask(img, params)
        # 覆盖之前的 mask(包括二次编辑)
        ctx["mask"] = mask
        ctx["mask_dirty"] = False
        n_px = int((mask > 0).sum())
        n, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        n_candidate = max(0, n - 1)
        QMessageBox.information(
            self, "提取完成",
            f"✅ 裂缝提取完成\n\n"
            f"检测算法: {algo_name}\n"
            f"候选数: {n_candidate} 条\n"
            f"覆盖像素: {n_px:,}",
        )
        self.extracted.emit(mask)


# ============= 步骤 6: 二次编辑 =============
class Step6FractureEdit(QWidget):
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
        sub = QLabel("用橡皮擦/添加工具精细调整裂缝区域")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 说明卡片
        card = Card("info")
        card.add_content(
            "💡 二次编辑只影响 Step 8 的裂缝分析,不触发 Step 5 重新提取。\n\n"
            "操作方法:\n"
            "① 在画布顶部工具栏选「🧹 擦除」或「➕ 添加」\n"
            "② 在画布上直接涂抹修改裂缝区域\n"
            "③ 调整下方形态学参数(可选)\n"
            "④ 返回 Step 8 重新分析"
        )
        v.addWidget(card)
        # 形态学参数
        morph_card = Card("tip")
        cv_layout = morph_card._layout
        form = QFormLayout()
        self.morph_op = QComboBox()
        self.morph_op.addItems([
            "闭运算(连接断裂)",
            "开运算(去除小毛刺)",
            "膨胀(扩大裂缝)",
            "腐蚀(缩小裂缝)",
        ])
        form.addRow("操作:", self.morph_op)
        self.kernel_size = QSpinBox()
        self.kernel_size.setRange(1, 15)
        self.kernel_size.setValue(3)
        self.kernel_size.setSuffix(" px (核大小)")
        form.addRow(self.kernel_size)
        apply_btn = QPushButton("🔧 应用形态学")
        apply_btn.setMinimumHeight(36)
        apply_btn.clicked.connect(self._apply_morph)
        form.addRow(apply_btn)
        cv_layout.addLayout(form)
        v.addWidget(morph_card)
        v.addStretch(1)

    def _apply_morph(self):
        ctx = self.ctx()
        mask = ctx.get("mask")
        if mask is None:
            QMessageBox.warning(self, "提示", "请先提取裂缝(Step 5)")
            return
        ksize = self.kernel_size.value()
        if ksize <= 0:
            ksize = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        op = self.morph_op.currentText()
        if "闭运算" in op:
            new_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif "开运算" in op:
            new_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif "膨胀" in op:
            new_mask = cv2.dilate(mask, kernel)
        else:  # 腐蚀
            new_mask = cv2.erode(mask, kernel)
        ctx["mask"] = new_mask
        ctx["mask_dirty"] = True
        self.edited.emit(new_mask)


# ============= 步骤 7: 分析参数设置 (v1.1.4 重构) =============
class Step7AnalysisParams(QWidget):
    """裂缝分析参数设置.

    v1.1.4 改进: 不再是"类型标注",而是真正影响 Step 8 分析结果的参数:
    - 报告级缝宽阈值: < 阈值的裂缝不计入报告 (PDF 1.3: 默认 0.1mm)
    - 大缝/中缝分类阈值: ≥ 阈值为大缝 (PDF 1.2: 默认 10mm)
    - 中缝/小缝分类阈值: ≥ 阈值为中缝 (PDF 1.2: 默认 1mm)

    修改这些参数后,Step 8 会自动用新参数重新分析.
    """
    params_changed = pyqtSignal()

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("⚙  分析参数")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("调整裂缝分析的阈值参数(修改后 Step 8 自动重算)")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 报告级缝宽阈值
        card1 = Card("tip")
        card1.add_content(
            "📏 报告级缝宽阈值\n\n"
            "裂缝缝宽 < 该阈值的,不计入「报告级裂缝数」。\n"
            "依据 PDF 1.3 节:缝宽 < 0.1mm 的裂缝通常不计入正式报告。\n"
            "💡 调整此值可控制报告的严格程度。"
        )
        cv1 = card1._layout
        self.min_width = QDoubleSpinBox()
        self.min_width.setRange(0.01, 5.0)
        self.min_width.setDecimals(2)
        self.min_width.setSingleStep(0.05)
        self.min_width.setValue(0.10)
        self.min_width.setSuffix(" mm")
        self.min_width.valueChanged.connect(self._on_param_changed)
        cv1.addWidget(self.min_width)
        v.addWidget(card1)
        # 大缝/中缝阈值
        card2 = Card("tip")
        card2.add_content(
            "📏 大缝/中缝分类阈值\n\n"
            "裂缝缝宽 ≥ 该阈值的,分类为「大缝」。\n"
            "依据 PDF 1.2 节:大缝 ≥ 10mm。\n"
            "💡 调整此值可改变大缝/中缝的统计分布。"
        )
        cv2 = card2._layout
        self.threshold_large = QDoubleSpinBox()
        self.threshold_large.setRange(1.0, 50.0)
        self.threshold_large.setDecimals(1)
        self.threshold_large.setSingleStep(0.5)
        self.threshold_large.setValue(10.0)
        self.threshold_large.setSuffix(" mm")
        self.threshold_large.valueChanged.connect(self._on_param_changed)
        cv2.addWidget(self.threshold_large)
        v.addWidget(card2)
        # 中缝/小缝阈值
        card3 = Card("tip")
        card3.add_content(
            "📏 中缝/小缝分类阈值\n\n"
            "裂缝缝宽 ≥ 该阈值 且 < 大缝阈值,分类为「中缝」。\n"
            "依据 PDF 1.2 节:中缝 1~10mm,小缝 < 1mm。\n"
            "💡 调整此值可改变中缝/小缝的统计分布。"
        )
        cv3 = card3._layout
        self.threshold_medium = QDoubleSpinBox()
        self.threshold_medium.setRange(0.1, 10.0)
        self.threshold_medium.setDecimals(1)
        self.threshold_medium.setSingleStep(0.5)
        self.threshold_medium.setValue(1.0)
        self.threshold_medium.setSuffix(" mm")
        self.threshold_medium.valueChanged.connect(self._on_param_changed)
        cv3.addWidget(self.threshold_medium)
        v.addWidget(card3)
        # 应用按钮
        self.apply_btn = QPushButton("✅ 应用参数并重新分析")
        self.apply_btn.setObjectName("primaryButton")
        self.apply_btn.setMinimumHeight(40)
        self.apply_btn.clicked.connect(self._apply)
        v.addWidget(self.apply_btn)
        # 说明
        info = QLabel(
            "💡 这些参数修改后会保存到 context,Step 8 会用新参数重新分析。\n"
            "  • 报告级裂缝数: 重新计算 fracture_count_report\n"
            "  • 宽度分类(大/中/小): 重新计算 size_class\n"
            "  • 密度统计: 自动重算"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #57606a; font-size: 12px; padding: 8px;")
        v.addWidget(info)
        v.addStretch(1)
        # 从 ctx 加载已有参数
        self._load_from_ctx()

    def _load_from_ctx(self):
        """从 ctx 加载参数(如果有)."""
        ctx = self.ctx()
        params = ctx.get("analysis_params")
        if params is not None:
            self.min_width.setValue(params.get("min_width", 0.10))
            self.threshold_large.setValue(params.get("threshold_large", 10.0))
            self.threshold_medium.setValue(params.get("threshold_medium", 1.0))

    def _on_param_changed(self, _value):
        """参数变化 → 提示用户点应用按钮."""
        # 不立即重新分析,等用户点应用按钮
        pass

    def _apply(self):
        """应用参数并重新分析."""
        ctx = self.ctx()
        # 检查是否有掩码可分析
        mask = ctx.get("mask")
        if mask is None:
            QMessageBox.warning(self, "提示", "请先提取裂缝(Step 5)")
            return
        scale = ctx.get("scale")
        if scale is None:
            from rockpore.core.calibration import scale_from_dpi
            scale = scale_from_dpi(96)
            ctx["scale"] = scale
        # 保存参数到 ctx
        params = {
            "min_width": self.min_width.value(),
            "threshold_large": self.threshold_large.value(),
            "threshold_medium": self.threshold_medium.value(),
        }
        ctx["analysis_params"] = params
        # 重新分析
        from rockpore.core.fracture import analyze_fractures
        from rockpore.core.analysis import Scale as ScaleT
        # 用新阈值重新计算
        result = analyze_fractures(
            mask, scale,
            min_width_real=params["min_width"],
            min_length_px=15,
        )
        # 重新分类
        from rockpore.core.fracture import classify_fracture_width
        for f in result.fractures:
            f.size_class = classify_fracture_width_with_params(
                f.width_real,
                params["threshold_large"],
                params["threshold_medium"],
            )
        # 重新统计
        result.width_distribution = {"大缝": 0, "中缝": 0, "小缝": 0}
        for f in result.fractures:
            result.width_distribution[f.size_class] += 1
        # 报告级裂缝数
        result.fracture_count_report = sum(
            1 for f in result.fractures if f.width_real >= params["min_width"]
        )
        ctx["analysis_result"] = result
        # 通知主窗口刷新
        self.params_changed.emit()
        QMessageBox.information(
            self, "参数已应用",
            f"✅ 已用新参数重新分析:\n\n"
            f"• 报告级缝宽阈值: {params['min_width']:.2f} mm\n"
            f"• 大缝/中缝阈值: {params['threshold_large']:.1f} mm\n"
            f"• 中缝/小缝阈值: {params['threshold_medium']:.1f} mm\n\n"
            f"结果:\n"
            f"• 裂缝总数: {result.fracture_count} 条\n"
            f"• 报告级(≥{params['min_width']:.2f}mm): {result.fracture_count_report} 条\n"
            f"• 大缝: {result.width_distribution['大缝']} 条\n"
            f"• 中缝: {result.width_distribution['中缝']} 条\n"
            f"• 小缝: {result.width_distribution['小缝']} 条"
        )


def classify_fracture_width_with_params(
    width_mm: float,
    threshold_large: float,
    threshold_medium: float,
) -> str:
    """用自定义阈值对裂缝宽度分类."""
    if width_mm >= threshold_large:
        return "大缝"
    if width_mm >= threshold_medium:
        return "中缝"
    return "小缝"


# ============= 步骤 8: 裂缝分析 =============
class Step8FractureAnalyze(QWidget):
    analyzed = pyqtSignal(object)
    fracture_selected = pyqtSignal(int)
    show_all_changed = pyqtSignal(bool)  # v1.1.3: 用户切换"显示所有"

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        self._fractures_by_id: Dict[int, Any] = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("📊  裂缝分析")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("计算长度、宽度、密度等定量参数")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 按钮行
        h = QHBoxLayout()
        run_btn = QPushButton("▶ 自动分析")
        run_btn.setObjectName("primaryButton")
        run_btn.setMinimumHeight(36)
        run_btn.clicked.connect(self._run)
        h.addWidget(run_btn)
        # 默认勾选"显示所有标注" - 用户进入 Step 8 即可看到标注
        self.show_all_btn = QPushButton("🙈 隐藏所有标注")
        self.show_all_btn.setObjectName("ghostButton")
        self.show_all_btn.setCheckable(True)
        self.show_all_btn.setChecked(True)
        self.show_all_btn.toggled.connect(self._on_show_all_toggled)
        h.addWidget(self.show_all_btn)
        h.addStretch(1)
        v.addLayout(h)
        # 数据卡片 - 强制最小高度
        cards_grid = QGridLayout()
        cards_grid.setSpacing(10)
        self._cards = {}
        for i, (key, label, unit) in enumerate([
            ("fracture_count", "裂缝总数", "条"),
            ("fracture_count_report", "报告级(≥0.1mm)", "条"),
            ("areal_density", "面密度", "1/mm"),
            ("linear_density", "线密度", "1/mm"),
            ("total_length", "累计长度", "mm"),
            ("avg_width", "平均宽度", "mm"),
        ]):
            r, c = divmod(i, 3)
            frame = QFrame()
            frame.setObjectName("dataCard")
            frame.setMinimumHeight(78)
            frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            fv = QVBoxLayout(frame)
            fv.setContentsMargins(10, 8, 10, 8)
            lab = QLabel(label)
            lab.setObjectName("dataCardLabel")
            val = QLabel("-")
            val.setObjectName("dataCardValue")
            val.setMinimumHeight(30)
            val.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            fv.addWidget(lab)
            fv.addWidget(val)
            self._cards[key] = val
            cards_grid.addWidget(frame, r, c)
        v.addLayout(cards_grid)
        # 表格 - 强制最小高度
        # v1.1.4: "类型" 列改为可编辑的下拉框(用户直接在表格里标类型)
        self.table = QTableWidget(0, 7)
        self.table.setObjectName("dataTable")
        self.table.setHorizontalHeaderLabels([
            "ID", "类型", "长度(mm)", "宽度(mm)", "倾角(°)", "分类", "开启度"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 仅"类型"列可编辑
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(200)
        self.table.itemSelectionChanged.connect(self._on_table_select)
        v.addWidget(self.table)
        v.addStretch(1)

    def _run(self):
        ctx = self.ctx()
        mask = ctx.get("mask")
        scale = ctx.get("scale")
        if mask is None:
            QMessageBox.warning(self, "提示", "请先提取裂缝(Step 5)")
            return
        if scale is None:
            from rockpore.core.calibration import scale_from_dpi
            scale = scale_from_dpi(96)
            ctx["scale"] = scale
        from rockpore.core.fracture import analyze_fractures
        result = analyze_fractures(mask, scale)
        ctx["analysis_result"] = result
        self._refresh(result)
        # v1.1.3: 分析后自动显示所有标注(无需用户手动勾选)
        if self.show_all_btn.isChecked():
            self._on_show_all_toggled(True)
        self.analyzed.emit(result)

    def _refresh(self, result):
        self._fractures_by_id = {f.id: f for f in result.fractures}
        self._cards["fracture_count"].setText(f"{result.fracture_count}")
        self._cards["fracture_count_report"].setText(f"{result.fracture_count_report}")
        self._cards["areal_density"].setText(f"{result.areal_density:.4f}")
        self._cards["linear_density"].setText(f"{result.linear_density:.3f}")
        self._cards["total_length"].setText(f"{result.total_length_real:.2f}")
        self._cards["avg_width"].setText(f"{result.average_width_real:.3f}")
        # 表格 — v1.1.4: "类型" 列改为可编辑下拉框
        from PyQt5.QtWidgets import QComboBox
        self.table.setRowCount(0)
        type_options = ["未分类", "构造缝", "成岩缝", "风化缝"]
        for f in result.fractures:
            row = self.table.rowCount()
            self.table.insertRow(row)
            # 列 0: ID
            id_item = QTableWidgetItem(str(f.id))
            id_item.setData(Qt.UserRole, f.id)
            self.table.setItem(row, 0, id_item)
            # 列 1: 类型 (可编辑下拉框)
            type_combo = QComboBox()
            type_combo.addItems(type_options)
            type_combo.setCurrentText(f.fracture_type.value)
            type_combo.currentTextChanged.connect(
                lambda text, fid=f.id: self._on_type_changed(fid, text)
            )
            self.table.setCellWidget(row, 1, type_combo)
            # 列 2: 长度
            self.table.setItem(row, 2, QTableWidgetItem(f"{f.length_real:.2f}"))
            # 列 3: 宽度
            self.table.setItem(row, 3, QTableWidgetItem(f"{f.width_real:.3f}"))
            # 列 4: 倾角
            self.table.setItem(row, 4, QTableWidgetItem(f"{f.orientation_deg:.1f}"))
            # 列 5: 分类
            self.table.setItem(row, 5, QTableWidgetItem(f.size_class))
            # 列 6: 开启度
            self.table.setItem(row, 6, QTableWidgetItem(f.openness.value))

    def _on_type_changed(self, fracture_id: int, type_text: str):
        """用户在表格下拉框里修改了裂缝类型."""
        from rockpore.core.fracture import FractureType
        type_map = {
            "构造缝": FractureType.STRUCTURAL,
            "成岩缝": FractureType.DIAGENETIC,
            "风化缝": FractureType.WEATHERING,
            "未分类": FractureType.UNKNOWN,
        }
        f = self._fractures_by_id.get(fracture_id)
        if f is not None:
            f.fracture_type = type_map.get(type_text, FractureType.UNKNOWN)

    def _on_table_select(self):
        items = self.table.selectedItems()
        if not items:
            return
        row = items[0].row()
        fid_item = self.table.item(row, 0)
        if fid_item:
            fid = int(fid_item.text())
            self.fracture_selected.emit(fid)
            # v1.1.3: 表格行选中时, 临时切换为高亮模式(只显示选中)
            self._update_canvas_highlight(fid)

    def _update_canvas_highlight(self, fid: int):
        """通知画布高亮显示选中的裂缝."""
        f = self._fractures_by_id.get(fid)
        if f is None:
            return
        ctx = self.ctx()
        page = ctx.get("__page__")
        if page and hasattr(page, "canvas"):
            from .canvas_view import Annotation
            # 把当前所有裂缝作为标注,选中的用橙色高亮
            annotations = []
            for other in self._fractures_by_id.values():
                color = (255, 140, 0) if other.id == fid else (0, 100, 255)
                label = f"#{other.id} {other.length_real:.1f}mm×{other.width_real:.2f}mm"
                annotations.append(Annotation(
                    id=other.id, bbox=other.bbox, label=label, color=color,
                ))
            page.canvas.set_annotations(annotations)

    def _on_show_all_toggled(self, checked: bool):
        """切换显示所有裂缝标注."""
        # 更新按钮文字
        if checked:
            self.show_all_btn.setText("🙈 隐藏所有标注")
        else:
            self.show_all_btn.setText("👁 显示所有标注")
        # v1.1.3: 通过 show_all_changed 信号通知主窗口(由 main_window 处理画布标注)
        self.show_all_changed.emit(checked)
        # 同时本地也更新按钮文字(避免重复触发)


# ============= 步骤 9: 基础信息 =============
class Step9FractureInfo(QWidget):
    info_saved = pyqtSignal(dict)

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("📋  基础信息")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("录入项目元数据,用于报告归档")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        card = Card("info")
        cv_layout = card._layout
        form = QFormLayout()
        self.project_name = QLineEdit()
        self.sample_id = QLineEdit()
        self.operator_name = QLineEdit()
        self.notes = QPlainTextEdit()
        self.notes.setMaximumHeight(80)
        form.addRow("项目名称:", self.project_name)
        form.addRow("样品编号:", self.sample_id)
        form.addRow("分析人员:", self.operator_name)
        form.addRow("备注:", self.notes)
        save_btn = QPushButton("💾 保存信息")
        save_btn.setObjectName("primaryButton")
        save_btn.setMinimumHeight(36)
        save_btn.clicked.connect(self._save)
        form.addRow(save_btn)
        cv_layout.addLayout(form)
        v.addWidget(card)
        v.addStretch(1)

    def _save(self):
        info = {
            "project_name": self.project_name.text(),
            "sample_id": self.sample_id.text(),
            "operator": self.operator_name.text(),
            "notes": self.notes.toPlainText(),
        }
        self.ctx()["info"] = info
        self.info_saved.emit(info)


# ============= 步骤 10: 报告生成 =============
class Step10FractureReport(QWidget):
    saved = pyqtSignal(str)

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("📄  报告生成")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("生成裂缝分析报告 (支持 HTML/PDF/Excel/Word/CSV)")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 报告预览/导出
        card = Card("tip")
        cv_layout = card._layout
        self.summary_label = QLabel("尚未生成报告")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            "color: #1f2328; font-size: 13px; padding: 12px; "
            "background: #f6f8fa; border-radius: 6px;"
        )
        cv_layout.addWidget(self.summary_label)
        h = QHBoxLayout()
        gen_btn = QPushButton("📊 预览报告")
        gen_btn.setObjectName("primaryButton")
        gen_btn.setMinimumHeight(40)
        gen_btn.clicked.connect(self._preview)
        h.addWidget(gen_btn)
        save_btn = QPushButton("💾 保存报告")
        save_btn.setMinimumHeight(40)
        save_btn.clicked.connect(self._save)
        h.addWidget(save_btn)
        h.addStretch(1)
        cv_layout.addLayout(h)
        v.addWidget(card)
        v.addStretch(1)

    def _preview(self):
        ctx = self.ctx()
        result = ctx.get("analysis_result")
        if result is None:
            QMessageBox.warning(self, "提示", "请先完成裂缝分析(Step 8)")
            return
        # 构造摘要
        info = ctx.get("info", {})
        lines = [
            f"📊 裂缝分析报告摘要",
            "",
            f"项目名称: {info.get('project_name', '(未填)')}",
            f"样品编号: {info.get('sample_id', '(未填)')}",
            f"分析人员: {info.get('operator', '(未填)')}",
            "",
            f"裂缝总数: {result.fracture_count} 条",
            f"  大缝(≥10mm): {result.width_distribution.get('大缝', 0)} 条",
            f"  中缝(1-10mm): {result.width_distribution.get('中缝', 0)} 条",
            f"  小缝(<1mm): {result.width_distribution.get('小缝', 0)} 条",
            f"报告级(缝宽≥0.1mm): {result.fracture_count_report} 条",
            "",
            f"累计长度: {result.total_length_real:.2f} mm",
            f"平均宽度: {result.average_width_real:.3f} mm",
            f"最大长度: {result.max_length_real:.2f} mm",
            f"面密度: {result.areal_density:.4f} 1/mm",
            f"线密度: {result.linear_density:.3f} 1/mm",
        ]
        self.summary_label.setText("\n".join(lines))

    def _save(self):
        """v1.2.0: 用 ReportExporter 导出多种格式 (HTML/PDF/XLSX/DOCX/TXT/CSV)."""
        ctx = self.ctx()
        result = ctx.get("analysis_result")
        image = ctx.get("image")
        mask = ctx.get("mask")
        scale = ctx.get("scale")
        if result is None:
            QMessageBox.warning(self, "提示", "请先完成裂缝分析(Step 8)")
            return
        from rockpore.core.report_exporter import ReportExporter, SUPPORTED_FORMATS
        from rockpore.core.fracture import draw_fracture_annotations
        # 构造标注图
        try:
            annotated = draw_fracture_annotations(image if image is not None else mask, result.fractures)
        except Exception:
            annotated = None
        # 构造 exporter
        info = ctx.get("info", {})
        exporter = self._build_exporter(result, info, scale, annotated)
        # 弹文件对话框
        filters = ";;".join(meta["label"] for meta in SUPPORTED_FORMATS.values())
        filters += ";;所有文件 (*)"
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "保存报告", "fracture_report.html", filters,
        )
        if not path:
            return
        # 推断格式
        fmt_by_label = {meta["label"]: fmt for fmt, meta in SUPPORTED_FORMATS.items()}
        fmt = fmt_by_label.get(selected_filter, "html")
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if ext in SUPPORTED_FORMATS:
            fmt = ext
        try:
            exporter.export(fmt, path)
            QMessageBox.information(self, "保存成功", f"报告已保存到:\n{path}")
            self.saved.emit(path)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"导出 {fmt.upper()} 失败:\n{e}")

    def _build_exporter(self, result, info, scale, annotated_image):
        """构造 ReportExporter."""
        from rockpore.core.report_exporter import ReportExporter
        info_dict = {
            "项目名称": info.get("project_name", "(未填)"),
            "样品编号": info.get("sample_id", "(未填)"),
            "分析人员": info.get("operator", "(未填)"),
            "标尺": f"{scale.pixels_per_unit:.3f} 像素/{scale.unit.value}" if scale else "(未设置)",
        }
        summary = {
            "裂缝总数": f"{result.fracture_count} 条",
            "大缝(≥10mm)": f"{result.width_distribution.get('大缝', 0)} 条",
            "中缝(1-10mm)": f"{result.width_distribution.get('中缝', 0)} 条",
            "小缝(<1mm)": f"{result.width_distribution.get('小缝', 0)} 条",
            "报告级(≥0.1mm)": f"{result.fracture_count_report} 条",
            "累计长度": f"{result.total_length_real:.2f} mm",
            "平均宽度": f"{result.average_width_real:.3f} mm",
            "最大长度": f"{result.max_length_real:.2f} mm",
            "面密度": f"{result.areal_density:.4f} 1/mm",
            "线密度": f"{result.linear_density:.3f} 1/mm",
        }
        headers = ["ID", "类型", "长度(mm)", "宽度(mm)", "倾角(°)", "分类", "开启度", "充填"]
        rows = []
        for f in result.fractures:
            rows.append([
                f.id,
                f.fracture_type.value,
                f"{f.length_real:.2f}",
                f"{f.width_real:.3f}",
                f"{f.orientation_deg:.1f}",
                f.size_class,
                f.openness.value,
                f.fill_status.value,
            ])
        return ReportExporter(
            title="裂缝分析报告",
            info=info_dict,
            summary=summary,
            headers=headers,
            rows=rows,
            annotated_image=annotated_image,
            notes=info.get("notes", ""),
        )


# ============= FractureModule =============
class FractureModule(AnalysisModule):
    """裂缝分析模块(HoughLinesP + 完整 10 步工作流)."""
    name = "裂缝分析"
    icon = "📏"
    description = "识别岩心裂缝并计算长度、宽度、密度"

    def __init__(self):
        from .module_base import make_default_fracture_steps
        self._steps = make_default_fracture_steps()

    @property
    def steps(self):
        return self._steps

    def create_step_panel(self, step_idx: int, parent=None):
        page = parent
        def ctx_getter():
            return page.context
        idx = step_idx + 1  # 0-based → 1-based
        if idx == 1:
            return Step1OpenImage(ctx_getter, parent)
        if idx == 2:
            return StepPlaceholder(
                "启动裂缝分析",
                "本模块已启动,接下来请按左侧 10 步流程操作。\n\n"
                "建议路径:\n1. 打开图像 → 2. (本步) → 3. 设置标尺 → 4. 预处理\n"
                "→ 5. 提取 → 6. 编辑 → 7. 标注 → 8. 分析 → 9. 信息 → 10. 报告",
                "🚀"
            )
        if idx == 3:
            return Step3Scale(ctx_getter, parent)
        if idx == 4:
            return Step4FracturePreprocess(ctx_getter, parent)
        if idx == 5:
            return Step5FractureExtract(ctx_getter, parent)
        if idx == 6:
            return Step6FractureEdit(ctx_getter, parent)
        if idx == 7:
            return Step7AnalysisParams(ctx_getter, parent)
        if idx == 8:
            return Step8FractureAnalyze(ctx_getter, parent)
        if idx == 9:
            return Step9FractureInfo(ctx_getter, parent)
        if idx == 10:
            return Step10FractureReport(ctx_getter, parent)
        return StepPlaceholder(f"步骤 {idx}", "该步骤内容待补充", "📌")

    def run_step(self, step_idx: int, context: dict) -> dict:
        return {}

    def analyze(self, image, scale, context: dict) -> Any:
        """完整流程."""
        from rockpore.core.fracture_accuracy import detect_fractures_robust
        mask, _, result = detect_fractures_robust(image, scale)
        return {"mask": mask, "result": result}

    def build_report_data(self, context: dict) -> dict:
        return {
            "image": context.get("image"),
            "mask": context.get("mask"),
            "result": context.get("analysis_result"),
            "info": context.get("info", {}),
        }
