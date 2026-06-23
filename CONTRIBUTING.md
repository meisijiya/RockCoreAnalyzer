# 贡献指南 (CONTRIBUTING)

欢迎为 **岩心图像分析软件 (Rock Core Analyzer)** 贡献代码、报告 bug、改进文档！

## 📌 重要：v1.2.0 已冻结

**v1.2.0 是当前稳定版本，已冻结。** 后续新功能/改进通过 **Pull Request** 进入 `master` 分支。

- ✅ **稳定版本**: `v1.2.0` tag + `release/v1.2.0` 分支（不接受新代码，只修关键 bug）
- 🚧 **开发分支**: `master`（所有 PR 提到这里）

## 🐛 报告 Bug

在 [GitHub Issues](https://github.com/meisijiya/RockCoreAnalyzer/issues) 提交，请包含:

1. **环境**: OS / Python 版本 / `pip list` 输出
2. **复现步骤**: 最小化操作序列
3. **期望 vs 实际**: 截图或错误堆栈
4. **测试图**: 如果跟算法相关，附图（确保合法）

**Bug 模板**:

```markdown
### 环境
- OS: Ubuntu 24.04 (WSL2)
- Python: 3.12
- v1.2.0

### 复现
1. 打开 粒度样2.png
2. Step 5 设置 dtr=80%
3. 点"提取颗粒"

### 实际
弹出 0 颗，但画布显示 53% 青色
```

## 🚀 提交新功能 (PR)

### 1. 找待实现功能

查看 `README.md` 的 **「🔮 待实现功能」** 章节，里面有优先级标注的待办项。

### 2. 准备工作

```bash
# Fork 并克隆
git clone https://github.com/<your-username>/RockCoreAnalyzer.git
cd RockCoreAnalyzer

# 同步上游 master
git remote add upstream https://github.com/meisijiya/RockCoreAnalyzer.git
git fetch upstream
git checkout master
git merge upstream/master

# 建特性分支
git checkout -b feature/your-feature-name
```

### 3. 开发要求

#### 代码风格
- 遵循 PEP 8
- 公共函数/类加 **函数级中文注释**（项目约定）
- 新模块有 `__all__` 导出

#### 测试要求
- 新功能必须有单元测试（`tests/test_<module>.py`）
- 现有测试不能回归（142 个测试保持 100% 通过）
- 测试覆盖核心逻辑

#### 文档要求
- 更新 `docs/CHANGELOG.md` 标注新功能
- 重大功能在 `docs/` 加独立沉淀文档
- README 表格（如算法支持矩阵）同步更新

### 4. 提交

```bash
# 写清楚的 commit message
git commit -m "feat(v1.3.0): 实现多模块协同分析

- 同一岩心跨模块对比
- 跨模块综合报告
- 14 个新测试 100% 通过"

# 推送你的 fork
git push origin feature/your-feature-name
```

**Commit 规范**:
- `feat(version)`: 新功能
- `fix(version)`: Bug 修复
- `refactor`: 重构（无功能变化）
- `docs`: 文档
- `test`: 测试
- `chore`: 杂项（依赖、CI 等）

### 5. 创建 PR

在 GitHub 上从 `<your-fork>/feature/xxx` 提 PR 到 `meisijiya/RockCoreAnalyzer:master`。

**PR 描述模板**:

```markdown
## 关联 Issue
Closes #123

## 改动
- 新增 `rockpore/core/multi_module.py` 跨模块分析
- 改 `main_window.py` 加"综合分析"按钮
- 加 14 个测试到 `tests/test_multi_module.py`

## 测试
- 142 → 156 个测试，0 回归
- pytest 输出: `156 passed in 5.32s`

## 截图
（如果改了 UI）

## 文档
- CHANGELOG.md 更新
- 新增 `docs/V1.3.0_MULTI_MODULE.md`
```

## 📋 PR 审核流程

1. **CI 自动跑测试** — 必须 142+ 个测试 100% 通过
2. **代码风格检查** — 遵循 PEP 8
3. **人工 Review** — 维护者审核
4. **合并** — 审核通过后 squash merge

## 🎯 待实现功能优先级

详细的待实现功能列表见 `README.md` 的 **「🔮 待实现功能」** 章节。

**High Priority** (欢迎 PR):
- 同一岩心跨模块协同分析
- 多尺度粒度分析（大/中/小颗粒分别统计）
- 批处理多图像（已部分实现，可完善）

**Medium Priority**:
- 颗粒方向性分析（各向异性）
- 沉积相自动判别（Folk & Ward 参数）
- 图像直方图分析

**Low Priority** (研究性质):
- 深度学习辅助（U-Net）
- 三维可视化
- Web 版（Flask/FastAPI）

## 💡 提交建议

如果你想贡献但不知道从哪开始:

1. **从 `good first issue` 标签开始** — 适合新手的简单任务
2. **修复文档错别字** — 任何 PR 都会被接受
3. **加新测试** — 提高覆盖率
4. **改进错误信息** — 让崩溃提示更友好

## 🛡️ 行为准则

- **尊重**: 建设性讨论，避免人身攻击
- **包容**: 欢迎不同背景的贡献者
- **学习**: 提问不丢人，困惑时大胆说

## 📞 联系

- **GitHub Issues**: 报告 bug / 提建议
- **Pull Request**: 贡献代码
- **Discussions** (如有): 讨论架构/方向

## 📜 许可

贡献的代码沿用项目原有许可：仅供学习与教学使用。

---

*感谢你的贡献！每一个 PR 都让这个项目变得更好。* ✨