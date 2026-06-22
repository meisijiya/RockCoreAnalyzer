"""帮助面板 - 右上角 ? 按钮.

点击展开一个浮动窗口,显示:
- 软件简介
- 模块列表
- 快捷键
- 常见问题
- 关于
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QTabWidget, QTextBrowser,
    QVBoxLayout, QWidget, QToolButton,
)

from .theme import color


HELP_CONTENT = {
    "简介": """
<h2>岩心孔洞分析软件 v1.0</h2>
<p>本软件用于岩心图像的孔洞、裂缝、粒度分析与报告生成,适用于石油地质、
沉积学、岩石学等教学与科研场景。</p>

<h3>核心特性</h3>
<ul>
<li>🎯 高准确率孔洞识别(综合 97%+)</li>
<li>📏 自动标尺换算(像素↔mm/μm)</li>
<li>🖼️ 10 步标准化工作流</li>
<li>📊 HTML 报告自动生成</li>
<li>🛠️ CLI 批量处理 + 单元测试覆盖</li>
</ul>

<h3>三种分析模块</h3>
<table border="1" cellpadding="6" style="border-collapse: collapse;">
<tr><th>模块</th><th>说明</th><th>状态</th></tr>
<tr><td>🕳 孔洞分析</td><td>识别岩心孔洞并计算面积、直径、面孔率</td><td>✅ 已实现</td></tr>
<tr><td>📏 裂缝分析</td><td>识别裂缝并计算长度、宽度、密度</td><td>🚧 预留接口</td></tr>
<tr><td>⚪ 粒度分析</td><td>识别颗粒并计算粒度分布</td><td>🚧 预留接口</td></tr>
</table>
""",
    "快捷键": """
<h3>全局快捷键</h3>
<table border="1" cellpadding="4">
<tr><th>快捷键</th><th>功能</th></tr>
<tr><td>Ctrl+O</td><td>打开图像</td></tr>
<tr><td>Ctrl+S</td><td>保存当前模块报告</td></tr>
<tr><td>Ctrl++</td><td>放大画布</td></tr>
<tr><td>Ctrl+-</td><td>缩小画布</td></tr>
<tr><td>Ctrl+0</td><td>画布 1:1 缩放</td></tr>
<tr><td>Ctrl+F</td><td>画布适应窗口</td></tr>
<tr><td>Ctrl+1/2/3</td><td>切换到第 1/2/3 个模块 Tab</td></tr>
<tr><td>F1</td><td>打开本帮助</td></tr>
<tr><td>Esc</td><td>退出当前工具</td></tr>
</table>

<h3>画布工具快捷键</h3>
<table border="1" cellpadding="4">
<tr><th>快捷键</th><th>工具</th></tr>
<tr><td>V</td><td>查看/平移</td></tr>
<tr><td>S</td><td>选择孔洞</td></tr>
<tr><td>E</td><td>橡皮擦</td></tr>
<tr><td>A</td><td>添加</td></tr>
<tr><td>C</td><td>取色器</td></tr>
<tr><td>F</td><td>适应窗口</td></tr>
</table>
""",
    "孔洞分类": """
<h3>孔洞按大小分类 (PDF 1.2 节)</h3>
<table border="1" cellpadding="4">
<tr><th>类型</th><th>直径范围</th><th>地质学含义</th></tr>
<tr><td>大洞</td><td>&gt; 10 mm</td><td>主要储集空间</td></tr>
<tr><td>中洞</td><td>5 ~ 10 mm</td><td>次要储集空间</td></tr>
<tr><td>小洞</td><td>1 ~ 4.9 mm</td><td>细小孔洞</td></tr>
<tr><td>针孔/溶孔</td><td>&lt; 1 mm</td><td>微孔隙</td></tr>
</table>

<h3>报告标准 (PDF 1.3 节)</h3>
<ul>
<li>直径 &lt; 2mm 的孔洞在报告中<strong>不计数</strong></li>
<li>缝宽 &lt; 0.1mm 的裂隙不计数</li>
<li>长度 &lt; 2cm 的分枝裂隙不计数</li>
</ul>

<h3>有效性评价 (PDF 1.2 节)</h3>
<table border="1" cellpadding="4">
<tr><th>等级</th><th>含义</th></tr>
<tr><td>有效</td><td>孔洞未充填,流通性好</td></tr>
<tr><td>较有效</td><td>孔洞半充填</td></tr>
<tr><td>无效</td><td>孔洞全充填,无流通性</td></tr>
</table>

<h3>填充物类型</h3>
<p>泥质、方解石、白云石、沥青、石膏、黄铁矿、高岭石、石英</p>
""",
    "常见问题": """
<h3>Q1: 识别准确率如何?</h3>
<p>在标准岩心合成图(15 个孔洞)上,综合准确率达 97.29%
(像素 IoU 94.58% + Detection F1 100%)。真实岩心图可能因
图像质量、孔洞与背景对比度而有所差异。</p>

<h3>Q2: 报告标准为何是 ≥2mm?</h3>
<p>依据《岩心孔洞软件》PDF 1.3 节报告说明:缝宽 &lt; 0.1mm、
长度 &lt; 2cm 的分枝裂隙、直径 &lt; 2mm 的孔洞不计数。
这是地质行业惯例,避免微小孔洞干扰储层评价。</p>

<h3>Q3: 多张子图合成的图片如何处理?</h3>
<p>当前算法将整张图视为单张分析。如果您的图片包含多个独立子图
(如 lf.jpg 包含主图+2 张层状图),建议先在图像编辑软件中
分割为独立图片再分别分析。</p>

<h3>Q4: 裂缝分析与孔洞分析的区别?</h3>
<p>孔洞是近似圆形的等轴状空间,裂缝是细长的线状空间。
本软件使用 OTSU 阈值法提取孔洞区域;裂缝将使用 Hough 变换
或 Hessian 矩阵增强(后续版本)。</p>

<h3>Q5: 微观分析如何使用?</h3>
<p>勾选「微观分析(微米单位)」后,标尺单位从 mm 变为 μm。
适合显微镜下拍摄的岩心薄片分析。需在「重新定标」中设置
物镜倍数。</p>

<h3>Q6: 如何提高识别准确率?</h3>
<ol>
<li>图像预处理阶段使用「自动色阶」增强对比度</li>
<li>在「二次编辑」中先用开运算去噪,再用闭运算连接</li>
<li>合理设置最小面积阈值,过滤过小噪点</li>
<li>若光照不均,尝试「查找边缘」或「CLAHE」增强</li>
</ol>
""",
    "关于": """
<h2>关于本软件</h2>
<p><strong>名称:</strong> 岩心孔洞分析软件 (Rock Core Pore Analysis)</p>
<p><strong>版本:</strong> v1.0</p>
<p><strong>技术栈:</strong> Python + OpenCV + PyQt5</p>
<p><strong>许可:</strong> 仅供学习与教学使用</p>

<h3>依赖</h3>
<ul>
<li>opencv-python-headless ≥ 4.5.0</li>
<li>numpy ≥ 1.21.0</li>
<li>PyQt5 ≥ 5.15.0</li>
<li>Pillow ≥ 9.0.0</li>
<li>scipy, scikit-image, matplotlib (可选)</li>
</ul>

<h3>参考资料</h3>
<ul>
<li>《岩心裂缝、孔洞、粒度虚拟仿真教学系统》—— 实验指导书</li>
<li>《岩心孔洞软件》—— 需求文档</li>
</ul>

<h3>项目结构</h3>
<pre style="background: #f5f5f5; padding: 10px; border-radius: 4px;">
rockpore/
├── core/           # 核心算法
│   ├── calibration.py    # 标尺换算
│   ├── preprocessing.py  # 图像预处理
│   ├── segmentation.py   # 区域分割
│   ├── morphology.py     # 形态学
│   ├── analysis.py       # 孔洞分析
│   ├── accuracy.py       # 准确率评估
│   └── report.py         # HTML 报告
├── gui/            # PyQt5 桌面 GUI
│   ├── theme.py
│   ├── canvas_view.py
│   ├── teaching_panel.py
│   ├── module_base.py
│   └── main_window.py
└── cli.py          # CLI 入口
</pre>
""",
}


class HelpDialog(QDialog):
    """帮助对话框."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("帮助 - 岩心孔洞分析软件")
        self.resize(700, 500)
        self.setStyleSheet(f"""
            QDialog {{
                background: {color('bg_panel')};
            }}
            QTabWidget::pane {{
                border: 1px solid {color('border')};
                border-radius: 6px;
            }}
            QTabBar::tab {{
                padding: 8px 16px;
                background: transparent;
                color: {color('text_secondary')};
            }}
            QTabBar::tab:selected {{
                color: {color('primary')};
                border-bottom: 2px solid {color('primary')};
                font-weight: bold;
            }}
        """)
        v = QVBoxLayout(self)
        tabs = QTabWidget()
        for name, html in HELP_CONTENT.items():
            browser = QTextBrowser()
            browser.setHtml(html)
            browser.setOpenExternalLinks(True)
            tabs.addTab(browser, name)
        v.addWidget(tabs)
        # 关闭按钮
        h = QHBoxLayout()
        h.addStretch(1)
        close_btn = QPushButton("关闭")
        close_btn.setObjectName("primaryButton")
        close_btn.clicked.connect(self.close)
        h.addWidget(close_btn)
        v.addLayout(h)
