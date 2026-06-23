"""分析模块抽象基类.

统一孔洞/裂缝/粒度三种分析的工作流接口.
每个模块是一个独立的 AnalysisModule 子类,注册到 MainWindow 后
会在顶部 Tab 中显示一个 Tab 页,内含相同结构的 10 步工作流.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np


@dataclass
class StepDefinition:
    """工作流步骤定义.

    Attributes:
        index: 步骤序号(从 1 开始)
        title: 步骤标题(简短,4-8 字)
        subtitle: 副标题(描述这一步要做什么)
        why: 为什么做这一步(地质意义/科学依据)
        how: 怎么做(操作指引)
        geology: 地质学背景知识(可选)
    """
    index: int
    title: str
    subtitle: str
    why: str
    how: str
    geology: str = ""


class AnalysisModule(ABC):
    """分析模块抽象基类.

    子类必须实现:
        - name: 模块名
        - icon: Tab 图标(emoji 或资源路径)
        - steps: 步骤定义列表
        - create_step_panel(step_idx, parent) -> QWidget
        - run_step(step_idx, context) -> dict
        - analyze(image, scale, context) -> Any
        - generate_report(data) -> ReportData

    上下文 (context dict) 用于在步骤之间共享数据:
        - "image": 原始图像
        - "image_path": 图像路径
        - "scale": 标尺对象
        - "mask": 当前掩码
        - "intermediate": 任意中间结果
    """

    name: str = "未命名模块"
    icon: str = "📊"
    description: str = ""

    @property
    @abstractmethod
    def steps(self) -> List[StepDefinition]:
        """返回该模块的步骤定义列表."""
        ...

    @abstractmethod
    def create_step_panel(self, step_idx: int, parent=None):
        """创建指定步骤的交互面板 QWidget."""
        ...

    @abstractmethod
    def run_step(self, step_idx: int, context: dict) -> dict:
        """执行指定步骤的核心逻辑,返回需要更新到 context 的字段.

        Returns:
            dict: 例如 {"mask": mask_array, "result": analysis_result}
        """
        ...

    @abstractmethod
    def analyze(self, image: np.ndarray, scale, context: dict) -> Any:
        """完整分析流程(一步到位,用于 CLI/批处理)."""
        ...

    @abstractmethod
    def build_report_data(self, context: dict) -> dict:
        """构建报告数据(传给 core.report.generate_report)."""
        ...


def make_default_pore_steps() -> List[StepDefinition]:
    """孔洞分析默认步骤定义(对应 PDF 6.1 节)."""
    return [
        StepDefinition(
            index=1, title="打开图像", subtitle="加载岩心照片",
            why="所有分析都基于原始图像数据,需先打开待分析的岩心照片。",
            how="点击「打开图像」或菜单「文件→打开」(Ctrl+O)。支持 jpg/png/bmp/tif。",
            geology="岩心照片应包含比例尺或已知参考尺寸,用于后续换算为实际尺寸(mm)。",
        ),
        StepDefinition(
            index=2, title="启动分析", subtitle="进入 10 步工作流",
            why="按标准流程操作,确保分析结果可复现、可对比。",
            how="点击「启动分析」按钮,进入 10 步工作流界面。",
            geology="规范的实验流程是科研与教学的基础。",
        ),
        StepDefinition(
            index=3, title="标尺选择", subtitle="设置图像物理比例",
            why="图像中像素需换算为实际尺寸(mm),才能进行定量分析。",
            how="在右侧面板设置 DPI(默认 96),或切换宏观(mm)/微观(μm)模式。",
            geology="宏观分析单位 mm,微观分析单位 μm,1 英寸 = 25.4 mm。",
        ),
        StepDefinition(
            index=4, title="图像预处理", subtitle="增强孔洞与基质对比",
            why="原始图像常因光照不均、对比度低导致孔洞难以识别,需预处理增强。",
            how="调节亮度/对比度/Gamma,或点击「自动色阶」一键优化。",
            geology="色阶拉伸可放大暗色孔洞与浅色基质的灰度差异。",
        ),
        StepDefinition(
            index=5, title="孔洞提取", subtitle="自动识别暗色孔洞区域",
            why="将孔洞从基质中分离,生成二值掩码。每次提取会覆盖之前的掩码(包括 Step 6 的擦除/添加)。",
            how="选择提取方法(OTSU/自适应/颜色匹配),调整最小面积阈值。",
            geology="碳酸盐岩孔洞通常为深色,OTSU 阈值法基于灰度直方图自动分割。",
        ),
        StepDefinition(
            index=6, title="二次编辑", subtitle="手工调整 + 形态学优化",
            why="用橡皮擦/添加工具精细调整自动提取的孔洞区域(增加或减少孔洞),或用形态学运算去噪/连接断裂。",
            how="① 顶部工具栏选「🧹 擦除」或「➕ 添加」直接在画布上涂抹;② 下方表单选膨胀/腐蚀/开/闭运算调整核大小。",
            geology="用户的擦除/添加**只影响 Step 8 的孔洞分析**——擦除的位置不会识别为孔洞,添加的位置会被当作新孔洞;但不影响 Step 5 重新提取。",
        ),
        StepDefinition(
            index=7, title="孔洞填充", subtitle="填充掩码内小孔",
            why="掩码内部的小洞可能是采样噪声,填充可提高面积统计精度。",
            how="设置填充阈值,预览效果后确认。",
            geology="依据 PDF 1.3 节,直径 < 2mm 的孔洞在报告中不计数。",
        ),
        StepDefinition(
            index=8, title="孔洞分析", subtitle="计算面积、直径、面孔率(含用户的擦除/添加)",
            why="基于当前掩码(含 Step 6 的擦除/添加)计算每个连通域的面积、等效圆直径、分类。",
            how="点击「自动分析」,系统统计孔洞并显示。**表格中行数 = 最终纳入分析的孔洞数**。",
            geology="孔洞按直径分类:大洞>10mm,中洞5-10mm,小洞1-5mm,针孔<1mm。",
        ),
        StepDefinition(
            index=9, title="基础信息", subtitle="录入项目元数据",
            why="报告需包含项目/样品/分析人员等元数据,便于归档。",
            how="填写项目名称、样品编号、分析人员、备注,点击保存。",
            geology="完整记录是地质研究可追溯性的基础。",
        ),
        StepDefinition(
            index=10, title="报告生成", subtitle="生成 HTML 分析报告",
            why="标准化报告便于学生提交与教师批改。",
            how="点击「生成报告」预览,「保存报告」导出 HTML。",
            geology="报告包括孔洞统计、分类、频率分布、详细参数表。",
        ),
    ]


def make_default_fracture_steps() -> List[StepDefinition]:
    """裂缝分析默认步骤定义(预留接口,后续实现)."""
    return [
        StepDefinition(
            index=1, title="打开图像", subtitle="加载岩心照片",
            why="裂缝分析与孔洞分析使用相同的图像输入。",
            how="点击「打开图像」加载待分析图像。",
        ),
        StepDefinition(
            index=2, title="启动分析", subtitle="进入裂缝分析工作流",
            why="裂缝识别与孔洞识别算法不同,需独立流程。",
            how="点击「启动分析」进入。",
        ),
        StepDefinition(
            index=3, title="标尺选择", subtitle="设置图像物理比例",
            why="裂缝长度/宽度需换算为 mm。",
            how="设置 DPI(默认 96)。",
            geology="裂缝密度:线密度=条数/长度,面密度=累计长度/面积。",
        ),
        StepDefinition(
            index=4, title="图像预处理", subtitle="增强裂缝边缘",
            why="裂缝边缘对比度通常较低,需锐化。",
            how="使用「查找边缘」或「锐化」功能。",
        ),
        StepDefinition(
            index=5, title="裂缝提取", subtitle="识别线性暗色区域",
            why="裂缝呈细长线状,需用特定算法(如 Hough 变换)。",
            how="TODO: 实现 HoughLinesP 或 Hessian 增强 + 阈值。",
        ),
        StepDefinition(
            index=6, title="二次编辑", subtitle="合并断裂、去除毛刺",
            why="初步检测的裂缝可能有断裂或假阳性。",
            how="使用形态学闭运算连接断裂。",
        ),
        StepDefinition(
            index=7, title="分析参数", subtitle="调整报告阈值与分类阈值",
            why="裂缝分析的统计结果受阈值影响:报告级缝宽阈值(默认0.1mm)决定哪些裂缝计入正式报告;大/中缝阈值(默认10mm)影响宽度分类。",
            how="设置三个阈值参数,点击「应用参数并重新分析」,Step 8 会自动重算所有统计。",
            geology="依据 PDF 1.2-1.3 节:缝宽 < 0.1mm 不计入报告;大缝 ≥ 10mm,中缝 1-10mm,小缝 < 1mm。",
        ),
        StepDefinition(
            index=8, title="裂缝分析", subtitle="计算长度、宽度、密度",
            why="基于当前阈值对识别出的裂缝进行统计分析。",
            how="点击「自动分析」,系统根据 Step 7 设定的阈值计算每条裂缝的长度、宽度、宽度分类、统计密度。",
        ),
        StepDefinition(
            index=9, title="基础信息", subtitle="录入项目元数据",
            why="报告需包含项目元数据。",
            how="填写项目信息。",
        ),
        StepDefinition(
            index=10, title="报告生成", subtitle="生成裂缝分析报告",
            why="标准化报告。",
            how="点击「生成报告」。",
        ),
    ]


def make_default_grain_steps() -> List[StepDefinition]:
    """粒度分析默认步骤定义(预留接口,后续实现)."""
    return [
        StepDefinition(
            index=1, title="打开图像", subtitle="加载岩心照片",
            why="粒度分析基于砾岩/砂岩的颗粒图像。",
            how="打开图像。",
        ),
        StepDefinition(
            index=2, title="启动分析", subtitle="进入粒度分析工作流",
            why="粒度分析关注颗粒大小分布。",
            how="点击「启动分析」。",
        ),
        StepDefinition(
            index=3, title="标尺选择", subtitle="设置图像物理比例",
            why="颗粒尺寸需换算为 mm。",
            how="设置 DPI。",
            geology="粒度分类:砾>2mm,砂 0.0625-2mm,粉砂 0.0039-0.0625mm,泥<0.0039mm。",
        ),
        StepDefinition(
            index=4, title="图像预处理", subtitle="分割准备",
            why="颗粒与基质需清晰分离。",
            how="调节对比度,做色阶拉伸。",
        ),
        StepDefinition(
            index=5, title="颗粒分割", subtitle="分水岭分割粘连颗粒",
            why="颗粒常粘连,需分水岭分离。",
            how="TODO: 距离变换 + 分水岭。",
        ),
        StepDefinition(
            index=6, title="边界编辑", subtitle="修正分割错误",
            why="自动分割可能有错误,需手工修正。",
            how="使用橡皮擦/添加工具。",
        ),
        StepDefinition(
            index=7, title="颗粒筛选", subtitle="按尺寸筛选",
            why="用户可能只关心某粒级范围。",
            how="设置最大/最小粒径。",
        ),
        StepDefinition(
            index=8, title="粒度统计", subtitle="计算粒度参数",
            why="粒度均值、分选性、偏度等是沉积相分析的关键。",
            how="计算每个颗粒的等效圆直径。",
            geology="粒度参数反映搬运介质能量与沉积环境。",
        ),
        StepDefinition(
            index=9, title="基础信息", subtitle="录入项目元数据",
            why="报告元数据。",
            how="填写项目信息。",
        ),
        StepDefinition(
            index=10, title="报告生成", subtitle="生成粒度分析报告",
            why="标准化粒度报告。",
            how="点击「生成报告」。",
        ),
    ]
