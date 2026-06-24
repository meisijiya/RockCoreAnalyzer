"""教学面板 + 步骤定义测试.

验证 3 模块 × 10 步的 30 个 step 都有完整的 why/how/geology 字段.
教学面板能正确 set_step 并不崩.
"""

from __future__ import annotations

import os
import sys

# 允许无显示器 (CI 环境)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from rockpore.gui.module_base import (
    make_default_fracture_steps,
    make_default_grain_steps,
    make_default_pore_steps,
)


# ===== 步骤定义完整性测试 =====

@pytest.mark.parametrize("module_name,step_factory", [
    ("pore", make_default_pore_steps),
    ("fracture", make_default_fracture_steps),
    ("grain", make_default_grain_steps),
])
class TestStepDefinitions:
    """三个模块 30 个步骤的完整性."""

    def test_has_10_steps(self, module_name, step_factory):
        steps = step_factory()
        assert len(steps) == 10, f"{module_name} 应有 10 步,实际 {len(steps)}"

    def test_steps_sequential(self, module_name, step_factory):
        steps = step_factory()
        for i, s in enumerate(steps, start=1):
            assert s.index == i, f"{module_name} 步骤 {i} 编号错误"

    def test_steps_have_titles(self, module_name, step_factory):
        steps = step_factory()
        for s in steps:
            assert s.title and len(s.title) <= 10, (
                f"{module_name} Step {s.index} 标题缺失或过长: '{s.title}'"
            )

    def test_steps_have_why(self, module_name, step_factory):
        steps = step_factory()
        for s in steps:
            assert s.why and len(s.why) >= 30, (
                f"{module_name} Step {s.index} 'why' 太短 ({len(s.why)} 字):\n{s.why}"
            )

    def test_steps_have_how(self, module_name, step_factory):
        steps = step_factory()
        for s in steps:
            assert s.how and len(s.how) >= 50, (
                f"{module_name} Step {s.index} 'how' 太短 ({len(s.how)} 字):\n{s.how}"
            )

    def test_steps_have_geology(self, module_name, step_factory):
        steps = step_factory()
        for s in steps:
            assert s.geology and len(s.geology) >= 10, (
                f"{module_name} Step {s.index} 'geology' 缺失:\n{s.geology}"
            )

    def test_step_titles_unique(self, module_name, step_factory):
        """同模块的步骤标题应基本唯一 (除「打开图像」「基础信息」「报告生成」共用)."""
        steps = step_factory()
        titles = [s.title for s in steps]
        duplicates = set([t for t in titles if titles.count(t) > 1])
        common = {"打开图像", "基础信息", "报告生成"}
        unexpected_dupes = duplicates - common
        assert not unexpected_dupes, (
            f"{module_name} 步骤标题意外重复: {unexpected_dupes}"
        )


# ===== 教学面板渲染测试 (off-screen) =====

class TestTeachingPanel:
    """教学面板能正确 set_step 并不崩."""

    def test_panel_loads(self):
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from rockpore.gui.teaching_panel import TeachingPanel
        panel = TeachingPanel()
        panel.show()
        # 不崩即可
        assert panel is not None

    def test_panel_set_empty_step(self):
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from rockpore.gui.teaching_panel import TeachingPanel
        panel = TeachingPanel()
        panel.set_step(None, 0, 10)
        assert panel._title.text() == "未选择步骤"

    def test_panel_set_pore_steps(self):
        """3 模块 30 步的 set_step 全部不崩."""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from rockpore.gui.teaching_panel import TeachingPanel
        panel = TeachingPanel()

        for factory, name in [
            (make_default_pore_steps, "pore"),
            (make_default_fracture_steps, "fracture"),
            (make_default_grain_steps, "grain"),
        ]:
            steps = factory()
            for s in steps:
                panel.set_step(s, s.index, len(steps))
                # 验证内容已更新
                assert panel._title.text() == s.title
                assert len(panel._why_text.text()) > 30
                assert len(panel._how_text.text()) > 50
                assert panel._geo_text.text() != ""

    def test_panel_geometry_long_text(self):
        """长文本不溢出 (面板用 QScrollArea 自动滚动)."""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from rockpore.gui.teaching_panel import TeachingPanel
        panel = TeachingPanel()
        # 拿最长的 step
        s = make_default_pore_steps()[4]  # 孔洞提取,有 355 字的 how
        panel.set_step(s, 5, 10)
        # 验证 widget 高度合理 (QScrollArea 应该能容纳)
        # 这里只检查 _why_text 的内容长度
        assert len(panel._why_text.text()) > 100
        assert len(panel._how_text.text()) > 300


# ===== 关键内容关键字测试 (确保内容真的有讲算法/地质) =====

class TestStepContentKeywords:
    """每个 step 应包含关键术语,证明真的讲到了算法/地质."""

    @pytest.mark.parametrize("step_idx,keywords", [
        (4, ["OTSU", "灰度", "直方图"]),   # 孔洞 Step 4 预处理
        (5, ["OTSU", "分割", "阈值"]),     # 孔洞 Step 5 提取
        (8, ["连通域", "等效圆", "直径"]),  # 孔洞 Step 8 分析
        (5, ["OTSU", "长宽比", "Hough"]),  # 裂缝 Step 5
        (7, ["0.1mm", "10mm", "PDF"]),     # 裂缝 Step 7 阈值
        (5, ["距离变换", "分水岭", "山峰"]),  # 粒度 Step 5
        (8, ["Folk", "Wentworth", "Mz"]),  # 粒度 Step 8 统计
    ])
    def test_pore_fracture_grain_keyword(self, step_idx, keywords):
        """模糊检查,任一 factory 的对应 step 应包含这些关键字."""
        all_steps = (
            make_default_pore_steps()
            + make_default_fracture_steps()
            + make_default_grain_steps()
        )
        # 找包含关键字的 step
        for s in all_steps:
            if s.index == step_idx:
                combined = s.why + s.how + s.geology
                # 至少 1 个关键字命中
                hits = sum(1 for k in keywords if k in combined)
                if hits >= 1:
                    return
        # 如果全 3 模块都没命中,失败
        pytest.fail(f"Step {step_idx} 内容不含任何期望关键字: {keywords}")
