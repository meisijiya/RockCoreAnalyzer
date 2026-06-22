"""现代教学风格 Qt 主题 (QSS).

设计目标:
- 亮色主题,学生长时间使用不刺眼
- 主色:深蓝 (#2c5fa3) 表示专业/可靠
- 强调色:橙色 (#ff7849) 表示行动/警示
- 成功色:绿色 (#28a745)
- 圆角 6px,扁平化,符合现代审美
- 关键控件(QPushButton, QGroupBox, QTabWidget)加大内边距,触控友好
"""

# ============= 颜色调色板 =============
PALETTE = {
    "primary": "#2c5fa3",       # 主色 - 深蓝
    "primary_hover": "#1e4880",
    "primary_pressed": "#163a66",
    "primary_light": "#e8f0fb",
    "accent": "#ff7849",        # 强调色 - 橙
    "accent_hover": "#e85d2a",
    "success": "#28a745",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "info": "#17a2b8",
    "bg": "#f5f7fa",            # 主背景 - 浅灰
    "bg_panel": "#ffffff",      # 面板背景
    "bg_subtle": "#fafbfc",     # 微妙背景
    "bg_hover": "#eef2f7",
    "bg_pressed": "#dde5ed",
    "border": "#d0d7de",        # 边框
    "border_strong": "#8c959f",
    "text": "#1f2328",          # 主文字
    "text_secondary": "#57606a",
    "text_muted": "#8c959f",
    "text_inverse": "#ffffff",
    "shadow": "rgba(0, 0, 0, 0.08)",
}

# ============= 完整 QSS 样式表 =============
GLOBAL_QSS = f"""
/* 全局 */
* {{
    font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
    font-size: 13px;
    color: {PALETTE['text']};
}}

QMainWindow, QDialog {{
    background: {PALETTE['bg']};
}}

/* Tab Widget - 顶部模块切换 */
QTabWidget::pane {{
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    background: {PALETTE['bg_panel']};
    top: -1px;
}}

QTabBar {{
    background: transparent;
    qproperty-drawBase: 0;
}}

QTabBar::tab {{
    background: transparent;
    color: {PALETTE['text_secondary']};
    padding: 10px 24px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    min-width: 120px;
}}

QTabBar::tab:hover {{
    background: {PALETTE['bg_hover']};
    color: {PALETTE['primary']};
}}

QTabBar::tab:selected {{
    background: {PALETTE['bg_panel']};
    color: {PALETTE['primary']};
    border: 1px solid {PALETTE['border']};
    border-bottom: 2px solid {PALETTE['primary']};
    font-weight: bold;
}}

/* GroupBox - 分组框 */
QGroupBox {{
    background: {PALETTE['bg_panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {PALETTE['primary']};
    background: transparent;
    font-size: 13px;
}}

/* 按钮 */
QPushButton {{
    background: {PALETTE['bg_panel']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 8px 16px;
    min-height: 20px;
    font-size: 13px;
}}

QPushButton:hover {{
    background: {PALETTE['bg_hover']};
    border-color: {PALETTE['border_strong']};
}}

QPushButton:pressed {{
    background: {PALETTE['bg_pressed']};
}}

QPushButton:disabled {{
    background: {PALETTE['bg_subtle']};
    color: {PALETTE['text_muted']};
}}

QPushButton#primaryButton {{
    background: {PALETTE['primary']};
    color: {PALETTE['text_inverse']};
    border: 1px solid {PALETTE['primary']};
    font-weight: bold;
}}

QPushButton#primaryButton:hover {{
    background: {PALETTE['primary_hover']};
    border-color: {PALETTE['primary_hover']};
}}

QPushButton#primaryButton:pressed {{
    background: {PALETTE['primary_pressed']};
}}

QPushButton#accentButton {{
    background: {PALETTE['accent']};
    color: {PALETTE['text_inverse']};
    border: 1px solid {PALETTE['accent']};
    font-weight: bold;
}}

QPushButton#accentButton:hover {{
    background: {PALETTE['accent_hover']};
}}

QPushButton#successButton {{
    background: {PALETTE['success']};
    color: {PALETTE['text_inverse']};
    border: 1px solid {PALETTE['success']};
    font-weight: bold;
}}

QPushButton#ghostButton {{
    background: transparent;
    border: 1px dashed {PALETTE['border']};
    color: {PALETTE['text_secondary']};
}}

QPushButton#ghostButton:hover {{
    background: {PALETTE['bg_hover']};
}}

/* 输入控件 */
QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
    background: {PALETTE['bg_panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {PALETTE['primary_light']};
    selection-color: {PALETTE['primary']};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 2px solid {PALETTE['primary']};
    padding: 5px 9px;
}}

/* SpinBox 紧凑显示(不撑满整行)+ 上下箭头按钮 */
QSpinBox, QDoubleSpinBox {{
    min-width: 90px;
    max-width: 130px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    height: 14px;
    border-left: 1px solid {PALETTE['border']};
    border-bottom: 1px solid {PALETTE['border']};
    background: {PALETTE['bg_subtle']};
    border-top-right-radius: 5px;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
    background: {PALETTE['primary_light']};
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    height: 14px;
    border-left: 1px solid {PALETTE['border']};
    background: {PALETTE['bg_subtle']};
    border-bottom-right-radius: 5px;
}}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {PALETTE['primary_light']};
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {PALETTE['text_secondary']};
}}

QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
    border-bottom-color: {PALETTE['primary']};
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {PALETTE['text_secondary']};
}}

QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
    border-top-color: {PALETTE['primary']};
}}

/* 滑块样式 - 用于参数调节 */
QSlider::groove:horizontal {{
    height: 6px;
    background: {PALETTE['bg_subtle']};
    border: 1px solid {PALETTE['border']};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {PALETTE['primary']};
    border: 2px solid {PALETTE['bg_panel']};
    width: 18px;
    margin: -7px 0;
    border-radius: 10px;
}}

QSlider::handle:horizontal:hover {{
    background: {PALETTE['primary_hover']};
}}

QSlider::sub-page:horizontal {{
    background: {PALETTE['primary_light']};
    border-radius: 3px;
}}

QLineEdit:read-only {{
    background: {PALETTE['bg_subtle']};
    color: {PALETTE['text_secondary']};
}}

/* 下拉框 */
QComboBox {{
    background: {PALETTE['bg_panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 20px;
}}

QComboBox:hover {{
    border-color: {PALETTE['border_strong']};
}}

QComboBox:focus {{
    border: 2px solid {PALETTE['primary']};
    padding: 5px 9px;
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {PALETTE['text_secondary']};
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background: {PALETTE['bg_panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    selection-background-color: {PALETTE['primary_light']};
    selection-color: {PALETTE['primary']};
    padding: 4px;
}}

/* 复选框 */
QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1.5px solid {PALETTE['border_strong']};
    border-radius: 4px;
    background: {PALETTE['bg_panel']};
}}

QCheckBox::indicator:hover {{
    border-color: {PALETTE['primary']};
}}

QCheckBox::indicator:checked {{
    background: {PALETTE['primary']};
    border-color: {PALETTE['primary']};
    image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMiAxMiI+PHBhdGggZD0iTTEgNkw0IDkgMTEgMiIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIi8+PC9zdmc+);
}}

/* 进度条 */
QProgressBar {{
    background: {PALETTE['bg_subtle']};
    border: 1px solid {PALETTE['border']};
    border-radius: 4px;
    text-align: center;
    color: {PALETTE['text']};
    height: 16px;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {PALETTE['primary']}, stop:1 {PALETTE['accent']});
    border-radius: 3px;
}}

/* 状态栏 */
QStatusBar {{
    background: {PALETTE['bg_panel']};
    color: {PALETTE['text_secondary']};
    border-top: 1px solid {PALETTE['border']};
}}

QStatusBar::item {{
    border: none;
}}

/* 菜单栏 */
QMenuBar {{
    background: {PALETTE['bg_panel']};
    color: {PALETTE['text']};
    border-bottom: 1px solid {PALETTE['border']};
    padding: 2px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 6px 12px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background: {PALETTE['bg_hover']};
}}

QMenu {{
    background: {PALETTE['bg_panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 24px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background: {PALETTE['primary_light']};
    color: {PALETTE['primary']};
}}

QMenu::separator {{
    height: 1px;
    background: {PALETTE['border']};
    margin: 4px 8px;
}}

/* 工具栏 */
QToolBar {{
    background: {PALETTE['bg_panel']};
    border-bottom: 1px solid {PALETTE['border']};
    spacing: 4px;
    padding: 4px;
}}

QToolBar::separator {{
    background: {PALETTE['border']};
    width: 1px;
    margin: 4px 6px;
}}

QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px 10px;
    color: {PALETTE['text']};
}}

QToolButton:hover {{
    background: {PALETTE['bg_hover']};
    border-color: {PALETTE['border']};
}}

QToolButton:checked {{
    background: {PALETTE['primary_light']};
    color: {PALETTE['primary']};
    border-color: {PALETTE['primary']};
    font-weight: bold;
}}

/* 滚动条 */
QScrollBar:vertical {{
    background: {PALETTE['bg_subtle']};
    width: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background: {PALETTE['border_strong']};
    border-radius: 5px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background: {PALETTE['text_secondary']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {PALETTE['bg_subtle']};
    height: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background: {PALETTE['border_strong']};
    border-radius: 5px;
    min-width: 20px;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* 提示卡片 */
QFrame#infoCard {{
    background: {PALETTE['primary_light']};
    border: 1px solid {PALETTE['primary']};
    border-left: 4px solid {PALETTE['primary']};
    border-radius: 6px;
}}

QFrame#warningCard {{
    background: #fff8e1;
    border: 1px solid {PALETTE['warning']};
    border-left: 4px solid {PALETTE['warning']};
    border-radius: 6px;
}}

QFrame#successCard {{
    background: #e8f5e9;
    border: 1px solid {PALETTE['success']};
    border-left: 4px solid {PALETTE['success']};
    border-radius: 6px;
}}

QFrame#tipCard {{
    background: {PALETTE['bg_subtle']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
}}

/* 步骤指示器 (左/顶) */
QFrame#stepIndicator {{
    background: {PALETTE['bg_panel']};
    border-right: 1px solid {PALETTE['border']};
}}

QLabel#stepNumber {{
    color: {PALETTE['text_muted']};
    font-size: 12px;
    font-weight: bold;
}}

QLabel#stepTitle {{
    color: {PALETTE['text']};
    font-size: 13px;
}}

QLabel#stepNumberActive {{
    color: {PALETTE['primary']};
    font-size: 14px;
    font-weight: bold;
}}

QLabel#stepTitleActive {{
    color: {PALETTE['primary']};
    font-size: 14px;
    font-weight: bold;
}}

/* 标题 */
QLabel#pageTitle {{
    color: {PALETTE['text']};
    font-size: 18px;
    font-weight: bold;
}}

QLabel#pageSubtitle {{
    color: {PALETTE['text_secondary']};
    font-size: 13px;
}}

/* 数据卡片 */
QFrame#dataCard {{
    background: {PALETTE['bg_panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 12px;
}}

QLabel#dataCardLabel {{
    color: {PALETTE['text_secondary']};
    font-size: 12px;
}}

QLabel#dataCardValue {{
    color: {PALETTE['primary']};
    font-size: 22px;
    font-weight: bold;
}}

/* Splitter */
QSplitter::handle {{
    background: {PALETTE['border']};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

/* 表格 */
QTableWidget, QTableView {{
    background: {PALETTE['bg_panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    gridline-color: {PALETTE['border']};
}}

QHeaderView::section {{
    background: {PALETTE['bg_subtle']};
    padding: 8px;
    border: none;
    border-right: 1px solid {PALETTE['border']};
    border-bottom: 1px solid {PALETTE['border']};
    color: {PALETTE['text']};
    font-weight: bold;
}}

QTableWidget::item, QTableView::item {{
    padding: 6px;
}}

QTableWidget::item:selected, QTableView::item:selected {{
    background: {PALETTE['primary_light']};
    color: {PALETTE['primary']};
}}
"""


def apply_theme(app):
    """应用主题到 QApplication."""
    app.setStyleSheet(GLOBAL_QSS)


# ============= 调色板访问函数 =============
def color(name: str) -> str:
    return PALETTE.get(name, "#000000")
