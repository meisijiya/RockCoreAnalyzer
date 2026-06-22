"""裂缝分析模块 - 完整 10 步工作流实现.

基于 HoughLinesP 的裂缝自动检测 + 长度/宽度/倾角/密度统计.
复用孔洞模块的 Step1OpenImage 和 Step3Scale (Step 1/3 是通用步骤).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
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
        form = QFormLayout()
        self.method = QComboBox()
        self.method.addItems([
            "adaptive(自适应阈值) ★推荐 — 真实岩石图",
            "hough(HoughLinesP) — 边缘清晰的图像",
        ])
        self.method.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("检测算法:", self.method)
        # Hough 参数
        self.canny_low = QSpinBox()
        self.canny_low.setRange(10, 200)
        self.canny_low.setValue(90)
        self.canny_low.setSuffix(" (Canny 低阈值)")
        form.addRow(self.canny_low)
        self.hough_threshold = QSpinBox()
        self.hough_threshold.setRange(5, 200)
        self.hough_threshold.setValue(25)
        self.hough_threshold.setSuffix(" (Hough 累加)")
        form.addRow(self.hough_threshold)
        self.min_length = QSpinBox()
        self.min_length.setRange(5, 500)
        self.min_length.setValue(20)
        self.min_length.setSuffix(" px (最小线长)")
        form.addRow(self.min_length)
        self.max_gap = QSpinBox()
        self.max_gap.setRange(0, 50)
        self.max_gap.setValue(15)
        self.max_gap.setSuffix(" px (最大断裂)")
        form.addRow(self.max_gap)
        self.dilation = QSpinBox()
        self.dilation.setRange(1, 10)
        self.dilation.setValue(3)
        self.dilation.setSuffix(" px (线段粗细)")
        form.addRow(self.dilation)
        # 自适应参数
        self.blur_kernel = QSpinBox()
        self.blur_kernel.setRange(3, 15)
        self.blur_kernel.setValue(7)
        self.blur_kernel.setSuffix(" px (高斯模糊)")
        form.addRow(self.blur_kernel)
        self.adaptive_block = QSpinBox()
        self.adaptive_block.setRange(5, 51)
        self.adaptive_block.setValue(21)
        self.adaptive_block.setSingleStep(2)
        self.adaptive_block.setSuffix(" px (邻域大小)")
        form.addRow(self.adaptive_block)
        self.adaptive_C = QSpinBox()
        self.adaptive_C.setRange(0, 30)
        self.adaptive_C.setValue(10)
        self.adaptive_C.setSuffix(" (常数 C)")
        form.addRow(self.adaptive_C)
        self.morph_close = QSpinBox()
        self.morph_close.setRange(0, 15)
        self.morph_close.setValue(5)
        self.morph_close.setSuffix(" px (闭运算)")
        form.addRow(self.morph_close)
        self.min_aspect = QSpinBox()
        self.min_aspect.setRange(1, 10)
        self.min_aspect.setValue(2)
        self.min_aspect.setSuffix(" :1 (长宽比)")
        form.addRow(self.min_aspect)
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
            "• adaptive(推荐): 高斯模糊 + 自适应阈值 + 形态学 + 长宽比筛选\n"
            "  — 适合真实岩石图(纹理多、对比度低),鲁棒性更好\n"
            "• hough: Canny 边缘 + HoughLinesP 概率霍夫变换\n"
            "  — 适合边缘清晰的图像(合成图、岩心薄片)\n\n"
            "💡 参数说明:\n"
            "• 提取会覆盖之前的结果(包括二次编辑)"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #57606a; font-size: 12px; padding: 8px;")
        v.addWidget(info)
        v.addStretch(1)
        # 默认显示 adaptive 参数
        self._on_method_changed(0)

    def _on_method_changed(self, idx):
        """根据所选算法显示/隐藏对应参数."""
        is_hough = idx == 1
        # Hough 参数
        self.canny_low.setVisible(is_hough)
        self.hough_threshold.setVisible(is_hough)
        self.min_length.setVisible(is_hough)
        self.max_gap.setVisible(is_hough)
        self.dilation.setVisible(is_hough)
        # Adaptive 参数
        is_adapt = idx == 0
        self.blur_kernel.setVisible(is_adapt)
        self.adaptive_block.setVisible(is_adapt)
        self.adaptive_C.setVisible(is_adapt)
        self.morph_close.setVisible(is_adapt)
        self.min_aspect.setVisible(is_adapt)

    def _extract(self):
        ctx = self.ctx()
        if "image" not in ctx or ctx["image"] is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        from rockpore.core.fracture import FractureParams, detect_fracture_mask
        from rockpore.core.fracture_accuracy import detect_fractures_robust
        img = ctx.get("processed_image")
        if img is None:
            img = ctx["image"]
        is_hough = self.method.currentIndex() == 1
        if is_hough:
            params = FractureParams(
                method="hough",
                canny_low=self.canny_low.value(),
                canny_high=self.canny_low.value() * 3,
                hough_threshold=self.hough_threshold.value(),
                min_line_length_px=self.min_length.value(),
                max_line_gap_px=self.max_gap.value(),
                dilation_kernel_px=self.dilation.value(),
            )
        else:
            params = FractureParams(
                method="adaptive",
                blur_kernel=self.blur_kernel.value(),
                adaptive_block=self.adaptive_block.value(),
                adaptive_C=self.adaptive_C.value(),
                morph_close=self.morph_close.value(),
                min_aspect_ratio=float(self.min_aspect.value()),
            )
        mask, _, _ = detect_fracture_mask(img, params)
        # 覆盖之前的 mask(包括二次编辑)
        ctx["mask"] = mask
        ctx["mask_dirty"] = False
        # 提示检测结果
        n_px = int((mask > 0).sum())
        n, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        n_candidate = max(0, n - 1)
        QMessageBox.information(
            self, "提取完成",
            f"✅ 裂缝提取完成\n\n"
            f"检测算法: {'HoughLinesP' if is_hough else '自适应阈值'}\n"
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


# ============= 步骤 7: 裂缝类型标注 =============
class Step7FractureType(QWidget):
    typed = pyqtSignal(dict)

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("🏷  类型标注")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("为每条裂缝标注成因类型(可选,默认均为未分类)")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 类型说明
        card = Card("info")
        card.add_content(
            "🔍 裂缝类型分类(PDF 1.2 节):\n\n"
            "① 构造缝 — 受构造应力形成,通常规则平直\n"
            "② 成岩缝 — 沉积过程中形成,常被方解石/泥质充填\n"
            "③ 风化缝 — 暴露地表后风化形成,形状不规则\n\n"
            "💡 批量标注:在下方设置默认类型,点击「应用」即可批量更新所有裂缝"
        )
        v.addWidget(card)
        # 批量标注控件
        type_card = Card("tip")
        cv_layout = type_card._layout
        form = QFormLayout()
        self.bulk_type = QComboBox()
        self.bulk_type.addItems(["未分类", "构造缝", "成岩缝", "风化缝"])
        form.addRow("默认类型:", self.bulk_type)
        self.bulk_fill = QComboBox()
        self.bulk_fill.addItems(["未充填", "半充填", "全充填"])
        form.addRow("默认充填:", self.bulk_fill)
        self.bulk_open = QComboBox()
        self.bulk_open.addItems(["开启", "半开启", "闭合"])
        form.addRow("默认开启度:", self.bulk_open)
        apply_btn = QPushButton("✅ 应用批量标注")
        apply_btn.setObjectName("primaryButton")
        apply_btn.setMinimumHeight(40)
        apply_btn.clicked.connect(self._apply_bulk)
        form.addRow(apply_btn)
        cv_layout.addLayout(form)
        v.addWidget(type_card)
        v.addStretch(1)

    def _apply_bulk(self):
        """批量标注当前结果中的所有裂缝."""
        ctx = self.ctx()
        result = ctx.get("analysis_result")
        if result is None or result.fracture_count == 0:
            QMessageBox.information(self, "提示", "请先在 Step 8 完成裂缝分析")
            return
        from rockpore.core.fracture import (
            FractureType, FractureFill, FractureOpenness,
        )
        type_map = {"构造缝": FractureType.STRUCTURAL,
                    "成岩缝": FractureType.DIAGENETIC,
                    "风化缝": FractureType.WEATHERING,
                    "未分类": FractureType.UNKNOWN}
        fill_map = {"未充填": FractureFill.UNFILLED,
                    "半充填": FractureFill.SEMI_FILLED,
                    "全充填": FractureFill.FILLED}
        open_map = {"开启": FractureOpenness.OPEN,
                    "半开启": FractureOpenness.SEMI_OPEN,
                    "闭合": FractureOpenness.CLOSED}
        for f in result.fractures:
            f.fracture_type = type_map[self.bulk_type.currentText()]
            f.fill_status = fill_map[self.bulk_fill.currentText()]
            f.openness = open_map[self.bulk_open.currentText()]
        self.typed.emit({})


# ============= 步骤 8: 裂缝分析 =============
class Step8FractureAnalyze(QWidget):
    analyzed = pyqtSignal(object)
    fracture_selected = pyqtSignal(int)

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
        self.show_all_btn = QPushButton("👁 显示所有标注")
        self.show_all_btn.setObjectName("ghostButton")
        self.show_all_btn.setCheckable(True)
        self.show_all_btn.setChecked(False)
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
        self.table = QTableWidget(0, 7)
        self.table.setObjectName("dataTable")
        self.table.setHorizontalHeaderLabels([
            "ID", "类型", "长度(mm)", "宽度(mm)", "倾角(°)", "分类", "状态"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
        self.analyzed.emit(result)

    def _refresh(self, result):
        self._fractures_by_id = {f.id: f for f in result.fractures}
        self._cards["fracture_count"].setText(f"{result.fracture_count}")
        self._cards["fracture_count_report"].setText(f"{result.fracture_count_report}")
        self._cards["areal_density"].setText(f"{result.areal_density:.4f}")
        self._cards["linear_density"].setText(f"{result.linear_density:.3f}")
        self._cards["total_length"].setText(f"{result.total_length_real:.2f}")
        self._cards["avg_width"].setText(f"{result.average_width_real:.3f}")
        # 表格
        self.table.setRowCount(0)
        for f in result.fractures:
            row = self.table.rowCount()
            self.table.insertRow(row)
            cells = [
                str(f.id),
                f.fracture_type.value,
                f"{f.length_real:.2f}",
                f"{f.width_real:.3f}",
                f"{f.orientation_deg:.1f}",
                f.size_class,
                f.openness.value,
            ]
            for col, val in enumerate(cells):
                item = QTableWidgetItem(val)
                if col == 0:
                    item.setData(Qt.UserRole, f.id)
                self.table.setItem(row, col, item)

    def _on_table_select(self):
        items = self.table.selectedItems()
        if not items:
            return
        row = items[0].row()
        fid_item = self.table.item(row, 0)
        if fid_item:
            fid = int(fid_item.text())
            self.fracture_selected.emit(fid)
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
                label = f"{other.length_real:.1f}mm × {other.width_real:.2f}mm"
                annotations.append(Annotation(
                    id=other.id, bbox=other.bbox, label=label, color=color,
                ))
            page.canvas.set_annotations(annotations)

    def _on_show_all_toggled(self, checked: bool):
        """切换显示所有裂缝标注."""
        ctx = self.ctx()
        page = ctx.get("__page__")
        if not page or not hasattr(page, "canvas"):
            return
        from .canvas_view import Annotation
        if checked:
            result = ctx.get("analysis_result")
            if result:
                annotations = [
                    Annotation(
                        id=f.id,
                        bbox=f.bbox,
                        label=f"{f.length_real:.1f}mm × {f.width_real:.2f}mm",
                        color=(0, 100, 255),
                    )
                    for f in result.fractures
                ]
                page.canvas.set_annotations(annotations)
        else:
            page.canvas.set_annotations([])


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
        sub = QLabel("生成裂缝分析 HTML 报告")
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
        save_btn = QPushButton("💾 保存 HTML")
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
        ctx = self.ctx()
        result = ctx.get("analysis_result")
        image = ctx.get("image")
        mask = ctx.get("mask")
        scale = ctx.get("scale")
        if result is None:
            QMessageBox.warning(self, "提示", "请先完成裂缝分析(Step 8)")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存报告", "fracture_report.html",
            "HTML 文件 (*.html);;所有文件 (*)",
        )
        if not path:
            return
        from rockpore.core.report import generate_report, ReportData
        from rockpore.core.fracture import draw_fracture_annotations
        # 构造标注图
        annotated = draw_fracture_annotations(image if image is not None else mask, result.fractures)
        # 写入临时 png
        import tempfile, os
        tmp_png = os.path.join(tempfile.gettempdir(), "fracture_annotated.png")
        cv2.imwrite(tmp_png, annotated)
        # 复用通用 report 框架(简化: 使用 generate_report)
        info = ctx.get("info", {})
        project_name = info.get("project_name", "裂缝分析")
        # 用一个最小化的 HTML 报告
        from rockpore.core.fracture import draw_fracture_annotations as dfa
        # 直接构造 HTML
        html = self._build_html(result, info, annotated, scale)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        QMessageBox.information(self, "保存成功", f"报告已保存到:\n{path}")
        self.saved.emit(path)

    def _build_html(self, result, info, annotated_image, scale):
        """构造裂缝分析报告 HTML."""
        import base64
        # 把图像嵌入 base64
        import cv2
        _, buf = cv2.imencode(".png", annotated_image)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        rows = []
        for f in result.fractures:
            rows.append(
                f"<tr><td>{f.id}</td><td>{f.fracture_type.value}</td>"
                f"<td>{f.length_real:.2f}</td><td>{f.width_real:.3f}</td>"
                f"<td>{f.orientation_deg:.1f}</td><td>{f.size_class}</td>"
                f"<td>{f.openness.value}</td><td>{f.fill_status.value}</td></tr>"
            )
        html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>裂缝分析报告 — {info.get('project_name', '未命名项目')}</title>
<style>
body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin: 30px; color: #1f2328; }}
h1 {{ color: #2c5fa3; border-bottom: 3px solid #2c5fa3; padding-bottom: 10px; }}
h2 {{ color: #2c5fa3; margin-top: 30px; }}
.meta {{ background: #f6f8fa; padding: 15px; border-radius: 8px; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 20px 0; }}
.kpi {{ background: #e8f0fb; padding: 15px; border-radius: 8px; border-left: 4px solid #2c5fa3; }}
.kpi-label {{ font-size: 12px; color: #57606a; }}
.kpi-value {{ font-size: 24px; font-weight: bold; color: #2c5fa3; margin-top: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #d0d7de; padding: 8px; text-align: center; font-size: 13px; }}
th {{ background: #f6f8fa; color: #1f2328; font-weight: 600; }}
tr:hover {{ background: #f6f8fa; }}
img {{ max-width: 100%; border: 1px solid #d0d7de; border-radius: 6px; }}
</style></head><body>
<h1>📏 裂缝分析报告</h1>

<div class="meta">
  <p><b>项目名称:</b> {info.get('project_name', '(未填)')}</p>
  <p><b>样品编号:</b> {info.get('sample_id', '(未填)')}</p>
  <p><b>分析人员:</b> {info.get('operator', '(未填)')}</p>
  <p><b>分析日期:</b> {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <p><b>标尺:</b> {scale.pixels_per_unit:.3f} 像素/{scale.unit.value}</p>
</div>

<h2>关键指标</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-label">裂缝总数</div><div class="kpi-value">{result.fracture_count}</div></div>
  <div class="kpi"><div class="kpi-label">报告级(≥0.1mm)</div><div class="kpi-value">{result.fracture_count_report}</div></div>
  <div class="kpi"><div class="kpi-label">累计长度</div><div class="kpi-value">{result.total_length_real:.2f} mm</div></div>
  <div class="kpi"><div class="kpi-label">平均宽度</div><div class="kpi-value">{result.average_width_real:.3f} mm</div></div>
  <div class="kpi"><div class="kpi-label">面密度</div><div class="kpi-value">{result.areal_density:.4f} 1/mm</div></div>
  <div class="kpi"><div class="kpi-label">线密度</div><div class="kpi-value">{result.linear_density:.3f} 1/mm</div></div>
</div>

<h2>宽度分布</h2>
<table><tr><th>大缝(≥10mm)</th><th>中缝(1-10mm)</th><th>小缝(&lt;1mm)</th></tr>
<tr><td>{result.width_distribution.get('大缝', 0)}</td>
    <td>{result.width_distribution.get('中缝', 0)}</td>
    <td>{result.width_distribution.get('小缝', 0)}</td></tr>
</table>

<h2>裂缝标注图</h2>
<img src="data:image/png;base64,{b64}" alt="裂缝标注图">

<h2>裂缝详细参数</h2>
<table><tr><th>ID</th><th>类型</th><th>长度(mm)</th><th>宽度(mm)</th><th>倾角(°)</th><th>分类</th><th>开启度</th><th>充填</th></tr>
{''.join(rows)}
</table>

<h2>备注</h2>
<p>{info.get('notes', '(无)')}</p>

<p style="margin-top: 40px; color: #57606a; font-size: 12px;">本报告由 岩心孔洞分析软件 自动生成</p>
</body></html>"""
        return html


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
            return Step7FractureType(ctx_getter, parent)
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
