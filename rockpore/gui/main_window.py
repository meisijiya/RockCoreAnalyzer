"""主窗口 v2 - 现代化教学风格 + 多模块 Tab 架构.

布局:
┌────────────────────────────────────────────────────────────┐
│  菜单栏: 文件 / 处理 / 分析 / 报告 / 帮助                    │
├────────────────────────────────────────────────────────────┤
│  工具栏: 📂 打开 | 💾 保存 | 🔍 分析 | ❓ 帮助 | 新手引导     │
├────────────────────────────────────────────────────────────┤
│  Tabs: [🕳 孔洞分析] [📏 裂缝分析] [⚪ 粒度分析]              │
├──────────┬───────────────────────────────────┬──────────────┤
│ 步骤导航  │                                   │  教学说明侧栏 │
│ ① 打开    │                                   │  💡 为什么    │
│ ② 启动    │          画布                     │  🛠 怎么做    │
│ ③ 标尺    │     (显示图像/标注)              │  ⛏ 地质意义  │
│ ④ 预处理  │                                   │              │
│ ⑤ 提取    │                                   │  [上一步]     │
│ ⑥ 编辑    │                                   │  [下一步 ▶]   │
│ ⑦ 填充    │                                   │              │
│ ⑧ 分析    │                                   │              │
│ ⑨ 信息    │                                   │              │
│ ⑩ 报告    │                                   │              │
├──────────┴───────────────────────────────────┴──────────────┤
│  状态栏: 就绪                                                 │
└────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSettings, QTimer
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QAction, QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMenuBar, QMessageBox,
    QPushButton, QSizePolicy, QSplitter, QStackedWidget, QStatusBar,
    QTabWidget, QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from .canvas_view import Annotation, CanvasView, CanvasTool
from .help_dialog import HelpDialog
from .module_base import AnalysisModule, StepDefinition
from .pore_module import PoreModule
from .fracture_module import FractureModule
from .grain_module import GrainModule
from .teaching_panel import TeachingPanel
from .theme import apply_theme, color
from .walkthrough import GuideStep, WalkthroughOverlay


# 单个模块的工作流页面
class ModuleWorkflowPage(QWidget):
    """单个模块的工作流页面.

    包含:
    - 左侧步骤导航列表
    - 中央步骤内容(画布在上,步骤面板在下)
    - 右侧教学说明侧栏
    """

    def __init__(self, module: AnalysisModule, parent=None):
        super().__init__(parent)
        self.module = module
        # 每个模块独立的 context
        self.context: Dict = {
            "image": None,
            "image_path": "",
            "scale": None,
            "mask": None,
            "mask_dirty": False,        # 标记:True=用户手动修改过(避免被自动覆盖)
            "processed_image": None,
            "analysis_result": None,
            "info": {},
            "_edit_undo_stack": [],     # Step6 形态学编辑的撤销栈(提到 page 层避免丢失)
        }
        # 当前步骤索引 (0-based)
        self.current_step_idx = 0
        # 步骤面板栈
        self.step_panels: List[QWidget] = []
        self._build_ui()
        # 画布手动修改 → 同步到 context(关键修复 S1)
        self.canvas.mask_modified.connect(self._on_canvas_mask_modified)

    def _build_ui(self):
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        # 三栏: 步骤导航 | 主体 | 教学
        splitter = QSplitter(Qt.Horizontal)
        # 左侧步骤导航
        self.step_list = QListWidget()
        self.step_list.setMaximumWidth(180)
        self.step_list.setMinimumWidth(140)
        self.step_list.setStyleSheet(f"""
            QListWidget {{
                background: {color('bg_panel')};
                border: none;
                border-right: 1px solid {color('border')};
                outline: 0;
                padding: 8px 4px;
            }}
            QListWidget::item {{
                padding: 10px 12px;
                border-radius: 6px;
                margin: 2px 4px;
                color: {color('text_secondary')};
            }}
            QListWidget::item:hover {{
                background: {color('bg_hover')};
            }}
            QListWidget::item:selected {{
                background: {color('primary_light')};
                color: {color('primary')};
                font-weight: bold;
                border-left: 3px solid {color('primary')};
            }}
        """)
        for i, step in enumerate(self.module.steps):
            item = QListWidgetItem(f"{step.index}. {step.title}")
            self.step_list.addItem(item)
        self.step_list.setCurrentRow(0)
        self.step_list.currentRowChanged.connect(self._on_step_changed)
        splitter.addWidget(self.step_list)
        # 中间:画布 + 步骤面板
        center_splitter = QSplitter(Qt.Vertical)
        # 画布
        self.canvas = CanvasView()
        center_splitter.addWidget(self.canvas)
        # 步骤面板栈
        self.step_stack = QStackedWidget()
        self.step_stack.setStyleSheet(f"""
            QStackedWidget {{
                background: {color('bg')};
                border-top: 1px solid {color('border')};
            }}
        """)
        for step in self.module.steps:
            panel = self.module.create_step_panel(step.index - 1, parent=self)
            self.step_panels.append(panel)
            self.step_stack.addWidget(panel)
            # 连接通用信号
            self._wire_panel_signals(panel, step)
        center_splitter.addWidget(self.step_stack)
        center_splitter.setSizes([500, 300])
        splitter.addWidget(center_splitter)
        # 右侧教学面板
        self.teaching = TeachingPanel()
        self.teaching.next_clicked.connect(self._goto_next)
        self.teaching.prev_clicked.connect(self._goto_prev)
        splitter.addWidget(self.teaching)
        splitter.setSizes([180, 600, 320])
        h.addWidget(splitter)
        # 初始步骤
        self._update_teaching()

    def _wire_panel_signals(self, panel, step: StepDefinition):
        """连接步骤面板的信号到画布和导航."""
        # Step1 的 completed: 加载图像后跳到第 2 步
        if hasattr(panel, "completed"):
            panel.completed.connect(self._on_step1_completed)
        # Step4 preprocessed: 更新画布
        if hasattr(panel, "preprocessed"):
            panel.preprocessed.connect(self._on_image_updated)
        # Step5 extracted: 更新掩码 + 跳到第 6 步
        if hasattr(panel, "extracted"):
            panel.extracted.connect(self._on_mask_extracted)
        # Step6 edited: 更新掩码
        if hasattr(panel, "edited"):
            panel.edited.connect(self._on_mask_updated)
        # Step8 analyzed: 更新标注 + 跳到第 9 步
        if hasattr(panel, "analyzed"):
            panel.analyzed.connect(self._on_analysis_done)
        # Step9 saved: 跳到第 10 步
        if hasattr(panel, "saved"):
            panel.saved.connect(self._on_info_saved)

    def _on_step1_completed(self):
        """步骤 1 完成: 显示图像 + 跳到第 2 步(启动分析)."""
        img = self.context.get("image")
        if img is not None:
            self.canvas.set_image(img)
            self.goto_step(1)  # 跳到第 2 步

    def _on_image_updated(self, image: np.ndarray):
        """步骤 4 完成: 更新画布显示预处理结果."""
        self.canvas.set_image(image)

    def _on_mask_extracted(self, mask: np.ndarray):
        """步骤 5 完成: 显示掩码 + 清标注(避免旧标注残留)."""
        self.canvas.set_mask(mask)
        self.canvas.set_annotations([])  # I4: 清空旧标注
        self.context["mask_dirty"] = False  # 自动提取的掩码不算"脏"

    def _on_mask_updated(self, mask: np.ndarray):
        """步骤 6 完成: 更新掩码显示."""
        self.canvas.set_mask(mask)

    def _on_canvas_mask_modified(self, mask: np.ndarray):
        """画布手动涂抹 → 同步到 context(关键 S1 修复).
        用户在画布上用橡皮擦/添加工具修改的掩码会触发此信号,
        把最新掩码写回 context['mask'],标记 mask_dirty=True.
        这样 Step6 形态学编辑会读取到包含涂抹的版本,
        Step8 孔洞分析也能识别用户手动画的区域.
        """
        self.context["mask"] = mask
        self.context["mask_dirty"] = True

    def _on_analysis_done(self, result):
        """步骤 8 完成: 标注孔洞 + 跳到第 9 步."""
        if result is None:
            return
        from .canvas_view import Annotation
        annotations = [
            Annotation(
                id=p.id, bbox=p.bbox, label=f"{p.diameter_real:.1f}mm",
                color=(0, 200, 100),
            )
            for p in result.pores
        ]
        self.canvas.set_annotations(annotations)

    def _on_info_saved(self, data: dict):
        """步骤 9 完成: 跳到第 10 步(报告)."""
        self.goto_step(9)

    def _on_step_changed(self, row: int):
        if row < 0 or row >= len(self.module.steps):
            return
        self.current_step_idx = row
        self.step_stack.setCurrentIndex(row)
        self._update_teaching()

    def _update_teaching(self):
        step = self.module.steps[self.current_step_idx]
        self.teaching.set_step(step, step.index, len(self.module.steps))

    def _goto_next(self):
        if self.current_step_idx < len(self.module.steps) - 1:
            self.step_list.setCurrentRow(self.current_step_idx + 1)

    def _goto_prev(self):
        if self.current_step_idx > 0:
            self.step_list.setCurrentRow(self.current_step_idx - 1)

    def goto_step(self, idx: int):
        if 0 <= idx < len(self.module.steps):
            self.step_list.setCurrentRow(idx)

    def on_image_loaded(self, image: np.ndarray, path: str):
        """图像被主窗口加载后通知各页面更新.
        重置所有分析相关的 context 字段,避免旧图的处理结果污染新图.
        """
        self.context["image"] = image
        self.context["image_path"] = path
        self.context["mask"] = None
        self.context["mask_dirty"] = False
        self.context["processed_image"] = None
        self.context["analysis_result"] = None
        self.context["info"] = {}
        self.context["_edit_undo_stack"] = []
        self.canvas.set_image(image)
        self.canvas.set_annotations([])  # 清掉旧标注


class _StepPanelHost:
    """步骤面板 host,提供 context 访问."""

    def __init__(self, page: ModuleWorkflowPage):
        self._page = page

    @property
    def context(self):
        return self._page.context

    def __call__(self):
        return self._page.context


class MainWindow(QMainWindow):
    """主窗口 v2."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("岩心孔洞分析软件 v1.0")
        self.resize(1400, 900)
        # 模块注册
        self.modules: List[AnalysisModule] = [
            PoreModule(),
            FractureModule(),
            GrainModule(),
        ]
        # 每模块的 workflow 页面
        self.module_pages: Dict[str, ModuleWorkflowPage] = {}
        # 当前选中的模块
        self.current_module_idx = 0
        # 构建 UI
        self._build_central()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        # 设置
        self._settings = QSettings("RockPore", "Analysis")
        # 首次启动时显示引导
        if not self._settings.value("walkthrough_done", False, type=bool):
            QTimer.singleShot(800, self._show_walkthrough)

    def _build_central(self):
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(False)
        for module in self.modules:
            page = ModuleWorkflowPage(module, parent=self)
            self.module_pages[module.name] = page
            self.tabs.addTab(page, f"{module.icon}  {module.name}")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)
        # 选第一个
        self.tabs.setCurrentIndex(0)

    def _build_menu(self):
        mb: QMenuBar = self.menuBar()
        # 文件
        file_menu = mb.addMenu("文件(&F)")
        a_open = QAction("📂 打开图像(&O)", self)
        a_open.setShortcut(QKeySequence.Open)
        a_open.triggered.connect(self.open_image)
        file_menu.addAction(a_open)
        file_menu.addSeparator()
        a_save_mask = QAction("保存掩码", self)
        a_save_mask.triggered.connect(self.save_current_mask)
        file_menu.addAction(a_save_mask)
        a_save_annot = QAction("保存标注图", self)
        a_save_annot.triggered.connect(self.save_annotated)
        file_menu.addAction(a_save_annot)
        a_export = QAction("导出报告...", self)
        a_export.setShortcut("Ctrl+S")
        a_export.triggered.connect(self.export_current_report)
        file_menu.addAction(a_export)
        file_menu.addSeparator()
        a_quit = QAction("退出", self)
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(self.close)
        file_menu.addAction(a_quit)
        # 视图
        view_menu = mb.addMenu("视图(&V)")
        for i, mod in enumerate(self.modules):
            a = QAction(f"{mod.icon} {mod.name}", self)
            a.setShortcut(f"Ctrl+{i+1}")
            a.triggered.connect(lambda _, idx=i: self.tabs.setCurrentIndex(idx))
            view_menu.addAction(a)
        view_menu.addSeparator()
        a_zoom_in = QAction("放大", self)
        a_zoom_in.setShortcut("Ctrl++")
        a_zoom_in.triggered.connect(self._zoom_in)
        view_menu.addAction(a_zoom_in)
        a_zoom_out = QAction("缩小", self)
        a_zoom_out.setShortcut("Ctrl+-")
        a_zoom_out.triggered.connect(self._zoom_out)
        view_menu.addAction(a_zoom_out)
        a_zoom_reset = QAction("1:1 缩放", self)
        a_zoom_reset.setShortcut("Ctrl+0")
        a_zoom_reset.triggered.connect(self._zoom_reset)
        view_menu.addAction(a_zoom_reset)
        a_zoom_fit = QAction("适应窗口", self)
        a_zoom_fit.setShortcut("Ctrl+F")
        a_zoom_fit.triggered.connect(self._zoom_fit)
        view_menu.addAction(a_zoom_fit)
        # 帮助
        help_menu = mb.addMenu("帮助(&H)")
        a_help = QAction("❓ 帮助", self)
        a_help.setShortcut("F1")
        a_help.triggered.connect(self._show_help)
        help_menu.addAction(a_help)
        a_tour = QAction("🎓 新手引导", self)
        a_tour.triggered.connect(self._show_walkthrough)
        help_menu.addAction(a_tour)
        help_menu.addSeparator()
        a_about = QAction("关于", self)
        a_about.triggered.connect(self._show_about)
        help_menu.addAction(a_about)

    def _build_toolbar(self):
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize())
        self.addToolBar(tb)
        # 主操作
        for label, tip, slot, obj_name in [
            ("📂 打开", "打开图像 (Ctrl+O)", self.open_image, "primaryButton"),
            ("💾 保存", "保存当前模块报告 (Ctrl+S)", self.export_current_report, ""),
            ("🔄 重做", "重新分析当前模块", self._redo_current, "ghostButton"),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            if obj_name:
                btn.setObjectName(obj_name)
            btn.clicked.connect(slot)
            tb.addWidget(btn)
        tb.addSeparator()
        # 帮助
        a_help = QPushButton("❓")
        a_help.setToolTip("帮助 (F1)")
        a_help.setFixedSize(36, 36)
        a_help.clicked.connect(self._show_help)
        tb.addWidget(a_help)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_label = QLabel("🟢 就绪 - 请打开岩心图像")
        sb.addWidget(self.status_label, 1)
        self.module_label = QLabel("当前模块: 孔洞分析")
        sb.addPermanentWidget(self.module_label)

    # -------------- 文件操作 --------------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开岩心图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        if not path:
            return
        from rockpore.core.io_utils import imread_unicode
        img = imread_unicode(path)
        if img is None:
            QMessageBox.critical(self, "错误", f"无法读取图片: {path}")
            return
        # 通知所有模块
        for page in self.module_pages.values():
            page.on_image_loaded(img, path)
        # 自动初始化标尺(在当前模块)
        self._init_scale_for(path)
        self.status_label.setText(
            f"✅ 已打开: {os.path.basename(path)} ({img.shape[1]}×{img.shape[0]})"
        )
        # 跳到下一步
        self.tabs.currentWidget().goto_step(2)  # 标尺步骤

    def _init_scale_for(self, path: str):
        from rockpore.core.calibration import get_image_dpi, scale_from_dpi
        dpi = get_image_dpi(path)
        scale = scale_from_dpi(dpi)
        for page in self.module_pages.values():
            page.context["scale"] = scale

    def save_current_mask(self):
        page = self.tabs.currentWidget()
        mask = page.canvas.get_mask()
        if mask is None:
            QMessageBox.warning(self, "提示", "当前模块尚未生成掩码")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存掩码", "mask.png", "PNG (*.png)"
        )
        if path:
            from rockpore.core.io_utils import imwrite_unicode
            imwrite_unicode(path, mask)
            self.status_label.setText(f"✅ 掩码已保存: {path}")

    def save_annotated(self):
        page = self.tabs.currentWidget()
        result = page.context.get("analysis_result")
        if result is None or page.context.get("image") is None:
            QMessageBox.warning(self, "提示", "请先完成分析")
            return
        from rockpore.core.report import _make_annotated_image
        path, _ = QFileDialog.getSaveFileName(
            self, "保存标注图", "annotated.png", "PNG (*.png)"
        )
        if path:
            annot = _make_annotated_image(
                page.context["image"],
                page.context.get("mask"),
                result,
            )
            from rockpore.core.io_utils import imwrite_unicode
            imwrite_unicode(path, annot)
            self.status_label.setText(f"✅ 标注图已保存: {path}")

    def export_current_report(self):
        page = self.tabs.currentWidget()
        result = page.context.get("analysis_result")
        if result is None:
            QMessageBox.warning(self, "提示", "请先完成分析")
            return
        from rockpore.core.report import ReportData, generate_report, _make_annotated_image
        scale = page.context.get("scale")
        scale_info = f"{scale.pixels_per_unit:.3f} px/{'μm' if scale and 'MICROMETER' in scale.unit.name else 'mm'}" if scale else "-"
        annot = _make_annotated_image(page.context["image"], page.context.get("mask"), result)
        info = page.context.get("info", {})
        data = ReportData(
            project_name=info.get("project", ""),
            sample_id=info.get("sample_id", ""),
            analyst=info.get("analyst", ""),
            image_path=page.context.get("image_path", ""),
            image_size=(page.context["image"].shape[1], page.context["image"].shape[0]),
            scale_info=scale_info,
            remarks=info.get("remarks", ""),
            analysis_result=result,
            original_image=page.context["image"],
            annotated_image=annot,
        )
        html = generate_report(data)
        path, _ = QFileDialog.getSaveFileName(
            self, "保存报告", f"report_{self.modules[self.current_module_idx].name}.html",
            "HTML (*.html)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            self.status_label.setText(f"✅ 报告已保存: {path}")

    def _redo_current(self):
        page = self.tabs.currentWidget()
        if page.context.get("image") is not None and page.context.get("scale") is not None:
            from rockpore.core.accuracy import detect_pores_robust
            mask, result = detect_pores_robust(
                page.context["image"], page.context["scale"]
            )
            page.context["mask"] = mask
            page.context["analysis_result"] = result
            page.canvas.set_mask(mask)
            self.status_label.setText("🔄 已重新分析")

    # -------------- 视图 --------------
    def _on_tab_changed(self, idx: int):
        self.current_module_idx = idx
        mod = self.modules[idx]
        self.module_label.setText(f"当前模块: {mod.name}")

    def _zoom_in(self):
        page = self.tabs.currentWidget()
        page.canvas._canvas.zoom_at(page.canvas._canvas.rect().center(), 1.2)

    def _zoom_out(self):
        page = self.tabs.currentWidget()
        page.canvas._canvas.zoom_at(page.canvas._canvas.rect().center(), 1 / 1.2)

    def _zoom_reset(self):
        page = self.tabs.currentWidget()
        page.canvas._canvas.set_scale(1.0)

    def _zoom_fit(self):
        page = self.tabs.currentWidget()
        page.canvas._canvas.fit_to_widget()

    # -------------- 帮助 --------------
    def _show_help(self):
        dlg = HelpDialog(self)
        dlg.exec_()

    def _show_about(self):
        QMessageBox.about(
            self, "关于",
            "<h2>岩心孔洞分析软件 v1.0</h2>"
            "<p>基于 Python + OpenCV + PyQt5 的教学型岩心图像分析软件</p>"
            "<p>支持: 孔洞分析 ✅ | 裂缝分析 🚧 | 粒度分析 🚧</p>"
            "<p>综合准确率 ≥ 97%</p>"
            "<p style='color: #57606a;'>仅供学习与教学使用</p>"
        )

    def _show_walkthrough(self):
        steps = [
            GuideStep(
                self.tabs, "👋 欢迎使用岩心孔洞分析软件",
                "本软件提供 3 大分析模块:\n"
                "• 🕳 孔洞分析(完整实现)\n"
                "• 📏 裂缝分析(开发中)\n"
                "• ⚪ 粒度分析(开发中)\n\n"
                "每个模块采用标准 10 步工作流,适合教学。",
                position="auto",
            ),
            GuideStep(
                self.module_pages["孔洞分析"].step_list,
                "📋 左侧步骤导航",
                "点击任意步骤可跳转到对应操作。\n"
                "当前步骤高亮显示,已完成的步骤会打勾。\n\n"
                "推荐按顺序操作: 1→2→3→...→10",
                position="right",
            ),
            GuideStep(
                self.tabs.currentWidget().canvas,
                "🖼 中央画布",
                "• 滚轮: 缩放\n"
                "• 拖动: 平移\n"
                "• 顶部工具栏: 选择/擦除/添加/取色\n"
                "• 底部状态栏: 位置/缩放",
                position="top",
            ),
            GuideStep(
                self.tabs.currentWidget().teaching,
                "💡 右侧教学说明",
                "每个步骤都有:\n"
                "• 为什么做这一步(科学依据)\n"
                "• 怎么做(操作指引)\n"
                "• 地质意义(背景知识)\n\n"
                "让学生不仅会操作,还理解原理。",
                position="left",
            ),
            GuideStep(
                self.menuBar(),
                "📋 顶部菜单",
                "• 文件: 打开/保存/导出\n"
                "• 视图: 切换模块、缩放\n"
                "• 帮助: 帮助文档、新手引导\n\n"
                "💡 按 F1 随时打开帮助。",
                position="bottom",
                next_label="完成引导",
            ),
        ]
        try:
            self._walkthrough = WalkthroughOverlay(steps, parent=self)
            self._settings.setValue("walkthrough_done", True)
        except Exception as e:
            print(f"引导启动失败: {e}")


def run_gui():
    """启动 GUI 入口."""
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec_()
