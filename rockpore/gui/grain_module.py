"""粒度分析模块 - 完整 10 步工作流.

实现:
- Step 1: 打开图像 (复用 Step1OpenImage)
- Step 2: 启动分析
- Step 3: 标尺选择 (复用 Step3Scale)
- Step 4: 图像预处理
- Step 5: 颗粒提取 (OTSU + 距离变换 + 分水岭)
- Step 6: 颗粒编辑 (拖拽式擦除/添加)
- Step 7: 分析参数 (粒径筛选)
- Step 8: 粒度统计 (表格 + 统计参数)
- Step 9: 基础信息
- Step 10: 报告生成
"""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from .module_base import AnalysisModule, make_default_grain_steps, StepDefinition
from .pore_module import StepPlaceholder, Step1OpenImage, Step3Scale


class GrainModule(AnalysisModule):
    """粒度分析模块 (距离变换 + 分水岭)."""
    name = "粒度分析"
    icon = "⚪"
    description = "识别颗粒并计算粒度分布、粒度参数"

    def __init__(self):
        self._steps = make_default_grain_steps()

    @property
    def steps(self):
        return self._steps

    def create_step_panel(self, step_idx: int, parent=None):
        page = parent

        def ctx_getter():
            return page.context

        idx = step_idx + 1  # 0-based -> 1-based
        if idx == 1:
            return Step1OpenImage(ctx_getter, parent)
        if idx == 2:
            return StepPlaceholder(
                "启动粒度分析",
                "本模块已启动,接下来请按左侧 10 步流程操作。\n\n"
                "建议路径:\n1. 打开图像 → 2. (本步) → 3. 设置标尺 → 4. 预处理\n"
                "→ 5. 颗粒提取 → 6. 颗粒编辑 → 7. 参数 → 8. 分析 → 9. 信息 → 10. 报告",
                "🚀"
            )
        if idx == 3:
            return Step3Scale(ctx_getter, parent)
        if idx == 4:
            return Step4GrainPreprocess(ctx_getter, parent)
        if idx == 5:
            return Step5GrainExtract(ctx_getter, parent)
        if idx == 6:
            return Step6GrainEdit(ctx_getter, parent)
        if idx == 7:
            return Step7GrainParams(ctx_getter, parent)
        if idx == 8:
            return Step8GrainAnalyze(ctx_getter, parent)
        if idx == 9:
            return Step9GrainInfo(ctx_getter, parent)
        if idx == 10:
            return Step10GrainReport(ctx_getter, parent)
        return StepPlaceholder(f"步骤 {idx}", "该步骤内容待补充", "📌")

    def run_step(self, step_idx: int, context: dict) -> dict:
        return {}

    def analyze(self, image, scale, context: dict) -> Any:
        """完整流程: 检测 + 分析."""
        from rockpore.core.grain import GrainParams, detect_grain_mask, analyze_grains
        params = GrainParams(
            blur_kernel=context.get("grain_blur", 7),
            morph_close=context.get("grain_morph_close", 7),
            morph_open=context.get("grain_morph_open", 3),
            min_area_px=context.get("grain_min_area", 200),
            distance_threshold_ratio=context.get("grain_dtr", 0.30),
        )
        mask = context.get("mask")
        if mask is None:
            mask, _ = detect_grain_mask(image, params)
        result, _, markers = analyze_grains(image, scale, params, mask=mask)
        return {"mask": mask, "result": result, "markers": markers}

    def build_report_data(self, context: dict) -> dict:
        return {
            "image": context.get("image"),
            "mask": context.get("mask"),
            "result": context.get("analysis_result"),
            "info": context.get("info", {}),
        }


# ============= 步骤 4: 图像预处理 =============
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QComboBox, QFormLayout, QMessageBox,
    QSizePolicy,
)
from PyQt5.QtCore import pyqtSignal, Qt
from .teaching_panel import Card
import cv2


class Step4GrainPreprocess(QWidget):
    """步骤 4: 粒度图像预处理."""
    preprocessed = pyqtSignal(np.ndarray)

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("🔧  图像预处理")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("增强颗粒与基质的对比度")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 说明卡片
        teach_card = Card("info")
        teach_card.add_title("预处理目的")
        teach_card.add_content(
            "将浅色矿物颗粒与深色基质分离。\n\n"
            "方法: CLAHE 局部直方图均衡化 + 可选高斯模糊。\n\n"
            "• CLAHE: 增强局部对比度,让颗粒边界更清晰\n"
            "• 模糊: 平滑颗粒内部纹理,避免过分割\n\n"
            "地质意义: 花岗岩/砂岩的颗粒边界由深色矿物(黑云母/角闪石)勾勒,预处理需保持边界完整。"
        )
        v.addWidget(teach_card)
        # 参数卡片
        param_card = Card("tip")
        cv_layout = param_card._layout
        form = QFormLayout()
        self.clahe_clip = QSpinBox()
        self.clahe_clip.setRange(1, 10)
        self.clahe_clip.setValue(2)
        self.clahe_clip.setSuffix(" (CLAHE 裁切限值)")
        self.clahe_clip.setMinimumWidth(140)
        form.addRow("CLAHE 增强:", self.clahe_clip)
        self.blur_kernel = QSpinBox()
        self.blur_kernel.setRange(0, 25)
        self.blur_kernel.setValue(7)
        self.blur_kernel.setSuffix(" px (模糊核大小, 0=不模糊)")
        self.blur_kernel.setMinimumWidth(140)
        form.addRow("高斯模糊:", self.blur_kernel)
        cv_layout.addLayout(form)
        v.addWidget(param_card)
        # 按钮行: 预览 + 重置
        h = QHBoxLayout()
        preview_btn = QPushButton("👁 预览预处理")
        preview_btn.setObjectName("primaryButton")
        preview_btn.setMinimumHeight(36)
        preview_btn.clicked.connect(self._preview)
        h.addWidget(preview_btn)
        reset_btn = QPushButton("♻ 重置为原图")
        reset_btn.setObjectName("ghostButton")
        reset_btn.setMinimumHeight(36)
        reset_btn.clicked.connect(self._reset)
        h.addWidget(reset_btn)
        h.addStretch(1)
        v.addLayout(h)
        v.addStretch(1)

    def _preview(self):
        """预览:在画布直接显示预处理结果(灰度转BGR)."""
        ctx = self.ctx()
        image = ctx.get("image")
        if image is None:
            QMessageBox.warning(self, "提示", "请先打开图像(Step 1)")
            return
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip.value(), tileGridSize=(8, 8))
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        gray = clahe.apply(gray)
        bk = self.blur_kernel.value()
        if bk > 0:
            gray = cv2.GaussianBlur(gray, (bk | 1, bk | 1), 0)
        # 保存参数 + 灰度预处理图
        ctx["preprocessed_gray"] = gray
        ctx["grain_blur"] = self.blur_kernel.value()
        # 转 BGR 用于画布显示
        if len(gray.shape) == 2:
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            display = gray.copy()
        self.preprocessed.emit(display)

    def _reset(self):
        """重置为原图."""
        ctx = self.ctx()
        image = ctx.get("image")
        if image is None:
            return
        ctx.pop("preprocessed_gray", None)
        self.preprocessed.emit(image.copy())


# ============= 步骤 5: 颗粒提取 =============
class Step5GrainExtract(QWidget):
    """步骤 5: 颗粒提取 (OTSU + 距离变换 + 分水岭)."""
    extracted = pyqtSignal(np.ndarray)

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("🔬  颗粒提取")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("OTSU 正向阈值 + 距离变换 + 分水岭分割")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 说明卡片
        teach_card = Card("info")
        teach_card.add_title("提取原理")
        teach_card.add_content(
            "算法流程:\n"
            "1. OTSU 正向阈值: 浅色矿物颗粒(亮) → 255(前景)\n"
            "2. 闭运算: 连接断裂的颗粒边界\n"
            "3. 距离变换: 计算每个像素到边界的距离\n"
            "4. 局部最大值: 找颗粒中心点\n"
            "5. 分水岭: 从中心向边界扩展,分离粘连颗粒\n\n"
            "适用: 花岗岩/砂岩等颗粒支撑岩石。"
        )
        v.addWidget(teach_card)
        # 参数
        param_card = Card("tip")
        cv_layout = param_card._layout
        form = QFormLayout()
        self.blur_kernel = QSpinBox()
        self.blur_kernel.setRange(0, 25)
        self.blur_kernel.setValue(7)
        self.blur_kernel.setSuffix(" (模糊核)")
        self.blur_kernel.setMinimumWidth(140)
        form.addRow("高斯模糊:", self.blur_kernel)
        self.morph_close = QSpinBox()
        self.morph_close.setRange(0, 20)
        self.morph_close.setValue(7)
        self.morph_close.setSuffix(" (闭运算)")
        self.morph_close.setMinimumWidth(140)
        form.addRow("闭运算:", self.morph_close)
        self.morph_open = QSpinBox()
        self.morph_open.setRange(0, 10)
        self.morph_open.setValue(3)
        self.morph_open.setSuffix(" (开运算)")
        self.morph_open.setMinimumWidth(140)
        form.addRow("开运算:", self.morph_open)
        self.min_area = QSpinBox()
        self.min_area.setRange(10, 5000)
        self.min_area.setValue(200)
        self.min_area.setSuffix(" px² (最小面积)")
        self.min_area.setMinimumWidth(140)
        form.addRow("最小面积:", self.min_area)
        self.dtr = QSpinBox()
        self.dtr.setRange(5, 80)
        self.dtr.setValue(30)
        self.dtr.setSuffix(" % (山峰阈值)")
        self.dtr.setMinimumWidth(140)
        form.addRow("距离山峰:", self.dtr)
        cv_layout.addLayout(form)
        v.addWidget(param_card)
        # 按钮
        h = QHBoxLayout()
        extract_btn = QPushButton("▶ 提取颗粒")
        extract_btn.setObjectName("primaryButton")
        extract_btn.setMinimumHeight(40)
        extract_btn.clicked.connect(self._extract)
        h.addWidget(extract_btn)
        h.addStretch(1)
        v.addLayout(h)
        v.addStretch(1)

    def _extract(self):
        ctx = self.ctx()
        # v1.2.0 改进: Step 4 预览过的灰度图真正传给 Step 5 (Plan A)
        # 跟 pore/fracture 一致 — Step 4 不只是给人看
        # 修复: 不能用 `or` 因为 numpy 数组的 or 会触发 ambiguous 错误
        image = ctx.get("preprocessed_gray")
        if image is None:
            image = ctx.get("image")
        if image is None:
            QMessageBox.warning(self, "提示", "请先打开图像(Step 1)")
            return
        # v1.2.0 修复: 重提取前清掉旧状态,避免残留标注和数据
        # 之前只 emit 'extracted' 信号,虽然 _on_mask_extracted 会清标注,
        # 但用户报告: "重提取后图片标记不会重置",可能消息框阻塞或事件时序
        # 导致画布没及时刷新。直接在 _extract 内部清,更可靠。
        ctx["analysis_result"] = None  # 强制 Step 8 重新计算
        from rockpore.core.grain import GrainParams, detect_grain_mask, analyze_grains
        from rockpore.core.calibration import Scale

        scale = ctx.get("scale", Scale(pixels_per_unit=8.0, unit='mm'))
        params = GrainParams(
            blur_kernel=self.blur_kernel.value(),
            morph_close=self.morph_close.value(),
            morph_open=self.morph_open.value(),
            min_area_px=self.min_area.value(),
            distance_threshold_ratio=self.dtr.value() / 100.0,
        )
        bin_mask, _ = detect_grain_mask(image, params)
        result, _, markers = analyze_grains(image, scale, params, mask=bin_mask)
        ctx["mask"] = bin_mask
        ctx["markers"] = markers
        # v1.2.0 修复: 画布显示实际颗粒分割区域(markers > 1),
        # 而不是原始 OTSU 二值图。用户反馈: 「提取到 2 个颗粒,
        # 但整个图片都快标记完了」 — 因为 bin_mask 是 OTSU 原始
        # 前景(可能占 60% 图像),而 markers 只有实际颗粒区域。
        # 视觉与数量对不上 → 改用 markers 区域显示
        # v1.2.0 进一步修复: 0 颗时不显示误导性大区域
        # dtr=80% 时,虽然 0 颗,但 markers>1 还有 26 万像素
        # (watershed 还是会分配大区域,被 max_area_ratio 过滤)
        # 0 颗时显示空 mask
        if result.grain_count_filtered == 0:
            grain_region_mask = np.zeros_like(bin_mask)
        else:
            grain_region_mask = ((markers > 1).astype(np.uint8)) * 255
        ctx["grain_params"] = params
        ctx["grain_blur"] = self.blur_kernel.value()
        ctx["grain_morph_close"] = self.morph_close.value()
        ctx["grain_morph_open"] = self.morph_open.value()
        ctx["grain_min_area"] = self.min_area.value()
        ctx["grain_dtr"] = self.dtr.value() / 100.0
        # 直接清掉画布上的旧标注(更可靠,不等消息框)
        page = self.parent()
        if page and hasattr(page, "canvas"):
            page.canvas.set_annotations([])
            # v1.2.0: 显示实际颗粒分割区域 (markers > 1),
            # 不是 OTSU 原始 bin_mask。让视觉与颗粒数对得上
            page.canvas.set_mask(grain_region_mask)
        QMessageBox.information(self, "提取完成",
            f"提取到 {result.grain_count_filtered} 个颗粒\n"
            f"平均粒径: {result.average_diameter_mm:.1f} mm\n"
            f"粒级分布: {dict((k, v) for k, v in result.size_distribution.items() if v > 0)}\n"
            f"\n提示: 距离山峰={int(self.dtr.value())}% "
            f"{'较高,只识别大颗粒 (默认30%)' if self.dtr.value() > 40 else '正常' if self.dtr.value() >= 20 else '较低,可能识别过多小颗粒'}"
            f"{' ⚠ 0 颗!距离山峰过高或最小面积过大,建议调低' if result.grain_count_filtered == 0 else ''}")
        # v1.2.0 修复: emit 实际颗粒分割区域而不是 OTSU 二值图
        # (避免 _on_mask_extracted 又把视觉覆盖回 bin_mask)
        self.extracted.emit(grain_region_mask)
        # v1.2.0 改进: 提取后自动跳到 Step 8 并重跑分析
        # 用户反馈: 「重提取后看不到新数据」
        # 之前要手动到 Step 8 点「自动分析」,现在自动完成
        if page and hasattr(page, "goto_step") and hasattr(page, "step_panels"):
            page.goto_step(7)  # Step 8 (0-based index 7)
            if 7 < len(page.step_panels):
                step8 = page.step_panels[7]
                if hasattr(step8, "_run"):
                    step8._run()


# ============= 步骤 6: 颗粒编辑 =============
class Step6GrainEdit(QWidget):
    """步骤 6: 颗粒边界编辑 (拖拽式)."""
    edited = pyqtSignal(np.ndarray)

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("🛠  颗粒编辑")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("用橡皮擦/添加工具精细调整颗粒区域")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        card = Card("info")
        card.add_content(
            "💡 编辑只影响 Step 8 的颗粒分析,不触发 Step 5 重新提取。\n\n"
            "操作方法:\n"
            "① 在画布顶部工具栏选「🧹 擦除」或「➕ 添加」\n"
            "② 在画布上按住鼠标拖拽涂抹修改颗粒区域\n"
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
            "闭运算(连接相邻颗粒)",
            "开运算(去除小噪点)",
            "膨胀(扩大颗粒)",
            "腐蚀(缩小颗粒)",
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
            QMessageBox.warning(self, "提示", "请先提取颗粒(Step 5)")
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
        else:
            new_mask = cv2.erode(mask, kernel)
        ctx["mask"] = new_mask
        ctx["mask_dirty"] = True
        self.edited.emit(new_mask)


# ============= 步骤 7: 分析参数 =============
class Step7GrainParams(QWidget):
    """步骤 7: 粒度分析参数 (粒径筛选)."""
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
        sub = QLabel("调整粒度分析的阈值参数(修改后 Step 8 自动重算)")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 说明卡片
        card = Card("info")
        card.add_title("参数说明")
        card.add_content(
            "粒径筛选:\n"
            "• 最小粒径(mm): 小于此值的颗粒不计入统计\n"
            "• 最大粒径(mm): 大于此值的颗粒不计入统计\n\n"
            "密实度/圆度筛选:\n"
            "• 最小密实度: 排除不规则形状(密实度=面积/凸包面积)\n"
            "• 最小圆度: 排除过扁/过长颗粒(圆度=4πA/P²)\n\n"
            "地质意义: 粒度分析关注颗粒粒径分布,密实度/圆度反映搬运距离与磨圆程度。"
        )
        v.addWidget(card)
        card = Card("tip")
        cv_layout = card._layout
        form = QFormLayout()
        self.min_diameter = QSpinBox()
        self.min_diameter.setRange(0, 1000)
        self.min_diameter.setValue(1)
        self.min_diameter.setSuffix(" mm (最小粒径)")
        self.min_diameter.setMinimumWidth(140)
        form.addRow("最小粒径:", self.min_diameter)
        self.max_diameter = QSpinBox()
        self.max_diameter.setRange(1, 5000)
        self.max_diameter.setValue(5000)
        self.max_diameter.setSuffix(" mm (最大粒径)")
        self.max_diameter.setMinimumWidth(140)
        form.addRow("最大粒径:", self.max_diameter)
        self.min_solidity = QSpinBox()
        self.min_solidity.setRange(1, 99)
        self.min_solidity.setValue(40)
        self.min_solidity.setSuffix(" % (最小密实度)")
        self.min_solidity.setMinimumWidth(140)
        form.addRow("最小密实度:", self.min_solidity)
        self.min_circularity = QSpinBox()
        self.min_circularity.setRange(1, 99)
        self.min_circularity.setValue(20)
        self.min_circularity.setSuffix(" % (最小圆度)")
        self.min_circularity.setMinimumWidth(140)
        form.addRow("最小圆度:", self.min_circularity)
        cv_layout.addLayout(form)
        v.addWidget(card)
        # 应用按钮
        apply_btn = QPushButton("✅ 应用参数")
        apply_btn.setObjectName("primaryButton")
        apply_btn.setMinimumHeight(36)
        apply_btn.clicked.connect(self._apply)
        v.addWidget(apply_btn)
        v.addStretch(1)

    def _apply(self):
        ctx = self.ctx()
        ctx["grain_min_diameter_mm"] = self.min_diameter.value()
        ctx["grain_max_diameter_mm"] = self.max_diameter.value()
        ctx["grain_min_solidity"] = self.min_solidity.value() / 100.0
        ctx["grain_min_circularity"] = self.min_circularity.value() / 100.0
        self.params_changed.emit()


# ============= 步骤 8: 粒度统计 =============
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView


class Step8GrainAnalyze(QWidget):
    """步骤 8: 粒度统计 (表格 + 统计参数)."""
    analyzed = pyqtSignal(object)
    grain_selected = pyqtSignal(int)
    show_all_changed = pyqtSignal(bool)  # v1.2.0: 用户切换"显示所有"

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        self._grains_by_id: Dict[int, Any] = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("📊  粒度统计")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("计算颗粒尺寸、圆度、密实度等定量参数")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 按钮行
        h = QHBoxLayout()
        run_btn = QPushButton("▶ 自动分析")
        run_btn.setObjectName("primaryButton")
        run_btn.setMinimumHeight(36)
        run_btn.clicked.connect(self._run)
        h.addWidget(run_btn)
        # v1.2.0: 显示所有标注开关 (与 fracture 一致)
        self.show_all_btn = QPushButton("🙈 隐藏所有标注")
        self.show_all_btn.setObjectName("ghostButton")
        self.show_all_btn.setCheckable(True)
        self.show_all_btn.setChecked(True)
        self.show_all_btn.toggled.connect(self._on_show_all_toggled)
        h.addWidget(self.show_all_btn)
        h.addStretch(1)
        v.addLayout(h)
        # 统计卡片
        from PyQt5.QtWidgets import QGridLayout, QFrame
        cards_grid = QGridLayout()
        cards_grid.setSpacing(10)
        self._cards = {}
        for i, (key, label, unit) in enumerate([
            ("grain_count", "颗粒总数", "颗"),
            ("avg_diameter", "平均粒径", "mm"),
            ("median_diameter", "中位粒径", "mm"),
            ("max_diameter", "最大粒径", "mm"),
            ("total_area", "颗粒总面积", "mm²"),
            ("avg_circularity", "平均圆度", ""),
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
            fv.addWidget(lab)
            # 单位嵌入数值文本(与 fracture 一致),
            # 避免单独 unit_lbl 与 val 视觉错位
            val = QLabel("-")
            val.setObjectName("dataCardValue")
            val.setMinimumHeight(30)
            val.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            fv.addWidget(val)
            cards_grid.addWidget(frame, r, c)
            self._cards[key] = val
        v.addLayout(cards_grid)
        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "#", "粒径(mm)", "长轴(mm)", "面积(px²)", "圆度", "密实度",
            "长宽比", "粒级", "质心"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMinimumHeight(250)
        self.table.itemSelectionChanged.connect(self._on_selection)
        v.addWidget(self.table, 1)

    def _run(self):
        ctx = self.ctx()
        # v1.2.0: 优先用 Step 4 预览过的灰度图,与 Step 5 保持一致
        # 修复: 不能用 `or` 因为 numpy 数组的 or 会触发 ambiguous 错误
        image = ctx.get("preprocessed_gray")
        if image is None:
            image = ctx.get("image")
        mask = ctx.get("mask")
        if image is None or mask is None:
            QMessageBox.warning(self, "提示", "请先完成 Step 5 颗粒提取")
            return
        from rockpore.core.grain import GrainParams, analyze_grains
        from rockpore.core.calibration import Scale

        scale = ctx.get("scale", Scale(pixels_per_unit=8.0, unit='mm'))
        params = GrainParams(
            blur_kernel=ctx.get("grain_blur", 7),
            morph_close=ctx.get("grain_morph_close", 7),
            morph_open=ctx.get("grain_morph_open", 3),
            min_area_px=ctx.get("grain_min_area", 200),
            distance_threshold_ratio=ctx.get("grain_dtr", 0.30),
            min_solidity=ctx.get("grain_min_solidity", 0.4),
            circularity_min=ctx.get("grain_min_circularity", 0.2),
        )
        result, _, _ = analyze_grains(image, scale, params, mask=mask)
        ctx["analysis_result"] = result
        # 显示统计卡片(单位嵌入文本,与 fracture 一致)
        self._cards["grain_count"].setText(f"{result.grain_count_filtered} 颗")
        self._cards["avg_diameter"].setText(f"{result.average_diameter_mm:.1f} mm")
        self._cards["median_diameter"].setText(f"{result.median_diameter_mm:.1f} mm")
        self._cards["max_diameter"].setText(f"{result.max_diameter_mm:.1f} mm")
        self._cards["total_area"].setText(f"{result.total_area_real:.0f} mm²")
        self._cards["avg_circularity"].setText(f"{result.average_circularity:.2f}")
        # 填充表格
        self._grains_by_id.clear()
        self.table.setRowCount(len(result.grains))
        for row, g in enumerate(result.grains):
            self._grains_by_id[g.id] = g
            self.table.setItem(row, 0, QTableWidgetItem(str(g.id)))
            self.table.setItem(row, 1, QTableWidgetItem(f"{g.diameter_major_mm:.1f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{g.diameter_major_mm:.1f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{g.area_px:.0f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{g.circularity:.3f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{g.solidity:.3f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{g.aspect_ratio:.2f}"))
            self.table.setItem(row, 7, QTableWidgetItem(g.size_class))
            self.table.setItem(row, 8, QTableWidgetItem(f"({g.centroid[0]:.0f},{g.centroid[1]:.0f})"))
        self.analyzed.emit(result)

    def _on_selection(self):
        rows = set(idx.row() for idx in self.table.selectedIndexes())
        if rows:
            row = list(rows)[0]
            # v1.2.0 修复: 用 _grains_by_id 取真实 ID,不再用 row+1 假设
            # (行号可能与 ID 不一致,如排序后)
            if 0 <= row < len(self._grains_by_id):
                # 取该行的第一列(就是 ID)
                id_item = self.table.item(row, 0)
                if id_item:
                    gid = int(id_item.text())
                    self.grain_selected.emit(gid)

    def _on_show_all_toggled(self, checked: bool):
        """v1.2.0: 切换"显示所有"按钮 (与 fracture 一致)."""
        self.show_all_changed.emit(checked)
        if checked:
            self.show_all_btn.setText("🙈 隐藏所有标注")
        else:
            self.show_all_btn.setText("👁 显示所有标注")


# ============= 步骤 9: 基础信息 =============
from PyQt5.QtWidgets import QLineEdit, QTextEdit


class Step9GrainInfo(QWidget):
    """步骤 9: 基础信息录入."""
    completed = pyqtSignal()

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("📋  基础信息")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("录入样品元数据")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        card = Card("info")
        cv_layout = card._layout
        form = QFormLayout()
        self.well_name = QLineEdit()
        self.well_name.setPlaceholderText("如: 长页地1井")
        form.addRow("井号:", self.well_name)
        self.depth = QLineEdit()
        self.depth.setPlaceholderText("如: 1234.5 m")
        form.addRow("深度:", self.depth)
        self.rock_type = QLineEdit()
        self.rock_type.setPlaceholderText("如: 中粒花岗岩")
        form.addRow("岩性:", self.rock_type)
        self.remarks = QTextEdit()
        self.remarks.setPlaceholderText("备注信息...")
        self.remarks.setMaximumHeight(80)
        form.addRow("备注:", self.remarks)
        cv_layout.addLayout(form)
        v.addWidget(card)
        save_btn = QPushButton("💾 保存信息")
        save_btn.setObjectName("primaryButton")
        save_btn.setMinimumHeight(36)
        save_btn.clicked.connect(self._save)
        v.addWidget(save_btn)
        v.addStretch(1)

    def _save(self):
        ctx = self.ctx()
        ctx["info"] = {
            "well_name": self.well_name.text(),
            "depth": self.depth.text(),
            "rock_type": self.rock_type.text(),
            "remarks": self.remarks.toPlainText(),
        }
        self.completed.emit()


# ============= 步骤 10: 报告生成 =============
from PyQt5.QtWidgets import QTextBrowser, QFileDialog


class Step10GrainReport(QWidget):
    """步骤 10: 生成粒度分析报告."""
    completed = pyqtSignal()

    def __init__(self, ctx_getter, parent=None):
        super().__init__(parent)
        self.ctx = ctx_getter
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        title = QLabel("📄  报告生成")
        title.setObjectName("pageTitle")
        v.addWidget(title)
        sub = QLabel("生成标准化粒度分析报告")
        sub.setObjectName("pageSubtitle")
        v.addWidget(sub)
        # 按钮行
        h = QHBoxLayout()
        generate_btn = QPushButton("📝 生成报告")
        generate_btn.setObjectName("primaryButton")
        generate_btn.setMinimumHeight(36)
        generate_btn.clicked.connect(self._generate)
        h.addWidget(generate_btn)
        save_btn = QPushButton("💾 保存报告")
        save_btn.setMinimumHeight(36)
        save_btn.clicked.connect(self._save)
        h.addWidget(save_btn)
        h.addStretch(1)
        v.addLayout(h)
        # 报告预览
        self.report_view = QTextBrowser()
        self.report_view.setMinimumHeight(300)
        v.addWidget(self.report_view, 1)

    def _generate(self):
        ctx = self.ctx()
        result = ctx.get("analysis_result")
        if result is None:
            QMessageBox.warning(self, "提示", "请先完成 Step 8 粒度分析")
            return
        info = ctx.get("info", {})
        # v1.2.0: 用 ReportExporter 构造数据,同时在 QTextBrowser 中显示纯文本预览
        from rockpore.core.report_exporter import ReportExporter
        exporter = self._build_exporter(result, info)
        # 在预览框显示纯文本
        self.report_view.setPlainText(exporter._build_text())
        # 缓存 exporter 以供 _save 用
        self._exporter = exporter

    def _build_exporter(self, result, info):
        """构造 ReportExporter (供 _generate 和 _save 共用)."""
        from rockpore.core.report_exporter import ReportExporter
        info_dict = {
            "井号": info.get("well_name", "未录入"),
            "深度": info.get("depth", "未录入"),
            "岩性": info.get("rock_type", "未录入"),
        }
        # 粒级分布 (过滤掉 0 颗的)
        size_dist_text = {k: v for k, v in result.size_distribution.items() if v > 0}
        summary = {
            "颗粒总数": f"{result.grain_count_filtered} 颗",
            "平均粒径": f"{result.average_diameter_mm:.1f} mm",
            "中位粒径": f"{result.median_diameter_mm:.1f} mm",
            "最大粒径": f"{result.max_diameter_mm:.1f} mm",
            "最小粒径": f"{result.min_diameter_mm:.1f} mm",
            "平均圆度": f"{result.average_circularity:.3f}",
            "颗粒总面积": f"{result.total_area_real:.0f} mm²",
            "图像面积": f"{result.image_area_real:.0f} mm²",
        }
        # 粒级分布作为单独 section
        headers = ["#", "粒径(mm)", "长轴(mm)", "面积(px²)", "圆度", "密实度", "长宽比", "粒级"]
        rows = []
        for g in result.grains:
            rows.append([
                g.id,
                f"{g.diameter_mm:.2f}",
                f"{g.diameter_major_mm:.2f}",
                f"{g.area_px:.0f}",
                f"{g.circularity:.3f}",
                f"{g.solidity:.3f}",
                f"{g.aspect_ratio:.2f}",
                g.size_class,
            ])
        return ReportExporter(
            title="岩石粒度分析报告",
            info=info_dict,
            summary=summary,
            headers=headers,
            rows=rows,
            notes=info.get("remarks", ""),
        )

    def _save(self):
        if not hasattr(self, "_exporter") or self._exporter is None:
            QMessageBox.warning(self, "提示", "请先生成报告")
            return
        from rockpore.core.report_exporter import SUPPORTED_FORMATS
        # 构造文件类型过滤器
        filters = ";;".join(
            f"{meta['label']}" for fmt, meta in SUPPORTED_FORMATS.items()
        )
        filters += ";;所有文件 (*)"
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "保存报告", "粒度分析报告.html", filters
        )
        if not path:
            return
        # 从 selected_filter 推断格式
        # 反向:label → fmt
        fmt_by_label = {meta["label"]: fmt for fmt, meta in SUPPORTED_FORMATS.items()}
        fmt = fmt_by_label.get(selected_filter, "html")
        # 或从扩展名推断
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if ext in SUPPORTED_FORMATS:
            fmt = ext
        try:
            self._exporter.export(fmt, path)
            QMessageBox.information(self, "保存成功", f"报告已保存到:\n{path}")
            self.completed.emit()
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"导出 {fmt.upper()} 失败:\n{e}")