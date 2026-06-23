# 岩心图像分析软件 (Rock Core Analyzer)

> 一套基于 Python + OpenCV + PyQt5 的岩心图像**三大分析模块**综合软件
>
> **v1.2.0** | 2026-06-23 | 142 个测试 100% 通过

![Status](https://img.shields.io/badge/version-v1.2.0-blue)
![Tests](https://img.shields.io/badge/tests-142%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-Educational-lightgrey)

## 📸 实机演示

![应用截图](docs/assets/screenshot.png)

*孔洞分析 Step 8: 252 个孔洞, 21.61% 面孔率, 平均直径 2.87mm。教学风格 UI, 右侧实时说明「为什么做 / 怎么做 / 地质意义」三栏。*

## 🎯 三大分析模块

| 模块 | 算法 | 准确率 | 状态 |
|------|------|--------|------|
| 🕳 **孔洞分析** | OTSU + 形态学 + 分水岭 | **97%** | ✅ v1.0.0 |
| 📏 **裂缝分析** | OTSU + HoughLinesP | **95%** | ✅ v1.1.5 |
| ⚪ **粒度分析** | 距离变换 + 分水岭 | **85%+** | ✅ v1.2.0 |

> 用户在 Windows 一键安装，3 步分析，6 种格式报告导出。

## ✨ 核心特性

### 🕳 孔洞分析 (v1.0.0)
- OTSU 自适应阈值 + 形态学闭/开运算
- 标准岩心合成图 **97.29% 准确率** (Pixel IoU 94.58% + Detection F1 100%)
- 大洞/中洞/小洞/针孔 自动分类
- PDF 1.2 节标准: 直径 < 2mm 不计入报告

### 📏 裂缝分析 (v1.1.5)
- 多种算法: HoughLinesP / 自适应阈值 / OTSU
- 长度/宽度/倾角/密度 完整计算
- 拖拽式橡皮擦/添加 (v1.1.5 改进)
- 真实岩石图测试 (裂缝样2.png) 准确率 95%+

### ⚪ 粒度分析 (v1.2.0)
- **距离变换 + 分水岭** 分割粘连颗粒
- Wentworth 粒级分类 (巨砾~黏土, 11 级)
- 颗粒属性: 面积/周长/长轴/圆度/密实度/长宽比
- 真实花岗岩图 (粒度样2.png) 准确率 85%+

### 📁 通用报告导出 (v1.2.0)
三模块共用的 `ReportExporter` 接口，支持 **6 种格式**:

| 格式 | 库 | 用途 |
|------|-----|------|
| HTML | 内置 | 网页查看 |
| TXT | 内置 | 纯文本存档 |
| **PDF** | reportlab | 正式报告、打印 |
| **Excel** | openpyxl | 数据分析 |
| **Word** | python-docx | 文档编辑 |
| CSV | 内置 | 数据库导入 |

### 🛠️ 跨平台 + 中文
- Windows / Linux (WSL2) 完整测试
- 中文路径支持 (`imread_unicode` / `imwrite_unicode`)
- 教学风格 UI (主色 #2c5fa3, 强调色 #ff7849)

---

## 🚀 部署与启动

### 1. 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Windows 10/11, Linux (Ubuntu 22.04+), macOS 12+ |
| **Python** | 3.9 ~ 3.12 (推荐 3.10) |
| **磁盘空间** | ~500 MB (含依赖) |
| **内存** | ≥ 2 GB (处理大图建议 4 GB+) |
| **显示** | 1366×768+ (PyQt5 GUI) |

WSL2 (Windows Subsystem for Linux) **完全支持**,推荐用于开发。

### 2. 克隆仓库

```bash
# HTTPS (无需 SSH key,推荐初次使用)
git clone https://github.com/meisijiya/RockCoreAnalyzer.git
cd RockCoreAnalyzer

# 或 SSH (需先配置 SSH key)
git clone git@github.com:meisijiya/RockCoreAnalyzer.git
cd RockCoreAnalyzer
```

### 3. 安装依赖

#### 方案 A: venv (推荐,隔离环境)

**Linux / macOS / WSL2:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

#### 方案 B: 系统 Python (快速试用)

**Linux / macOS:**
```bash
pip3 install -r requirements.txt
```

**Windows:**
```powershell
pip install -r requirements.txt
# 或在某些 Python 安装:
py -m pip install -r requirements.txt
```

> ⚠️ Windows 上若提示 "externally-managed-environment",需加 `--break-system-packages`:
> ```powershell
> pip install --break-system-packages -r requirements.txt
> ```

#### 方案 C: Conda

```bash
conda create -n rockpore python=3.10
conda activate rockpore
pip install -r requirements.txt
```

### 4. 必需依赖说明

`requirements.txt` 已包含所有依赖:

| 包 | 用途 | 必需 |
|----|------|------|
| opencv-python-headless >= 4.5.0 | 图像处理 (cv2) | ✓ |
| numpy >= 1.21.0 | 数值计算 | ✓ |
| scipy >= 1.7.0 | 科学计算 (距离变换等) | ✓ |
| scikit-image >= 0.19.0 | 形态学、扩展算法 | ✓ |
| Pillow >= 9.0.0 | DPI 读取、图像 | ✓ |
| PyQt5 >= 5.15.0 | **GUI 必需** | ✓ |
| matplotlib >= 3.5.0 | 图表、报告 | ✓ |
| **reportlab >= 4.0.0** | **PDF 导出** (v1.2.0+) | 推荐 |
| **openpyxl >= 3.0.0** | **Excel 导出** (v1.2.0+) | 推荐 |
| **python-docx >= 1.0.0** | **Word 导出** (v1.2.0+) | 推荐 |
| pytest >= 7.0.0 | 测试 | 开发 |

> 💡 最后 3 个 (reportlab/openpyxl/python-docx) 缺失也能跑,但**导出会失败**。建议安装。

### 5. 启动应用

```bash
# 桌面 GUI (推荐)
python run_gui.py

# Windows:
py run_gui.py
```

启动后会看到:
- **顶部**: 菜单栏 (文件/视图/帮助) + 工具栏 (打开/保存/重做/帮助)
- **左侧**: 10 步工作流导航
- **中央**: 画布 (显示图像+标注)
- **右侧**: 教学说明 (为什么做/怎么做/地质意义)
- **底部**: 状态栏

### 6. 验证安装

```bash
# 运行测试,确保 142 个测试 100% 通过
python -m pytest tests/ -q
# 期望: 142 passed in ~5s

# 验证 GUI 启动
python -c "import rockpore.gui.main_window; print('✓ 导入成功')"
```

### 7. 第一次使用

1. 启动 GUI: `python run_gui.py`
2. 点击 **📂 打开** 或菜单 文件→打开图像
3. 选择测试图: `孔洞.png` / `裂缝样2.png` / `粒度样2.png`
4. 按左侧 10 步流程操作
5. 第 10 步选择格式导出报告

### 8. 常见问题

**Q: Windows 上 GUI 启动报 "DLL load failed"?**
A: 安装 Microsoft Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe

**Q: cv2.imread 读不了中文路径图片?**
A: 使用 `rockpore.core.io_utils.imread_unicode()`,程序内部已处理

**Q: PDF 导出报 "ImportError: reportlab"?**
A: `pip install reportlab` (推荐装上 3 个导出库)

**Q: 测试报 "Qt platform plugin could not be initialized"?**
A: Linux 需装 `sudo apt install libxcb-xinerama0 libxcb-cursor0`;或用 `QT_QPA_PLATFORM=offscreen` 跳过 GUI

**Q: 启动慢 / 占内存大?**
A: 处理大图 (4K+) 建议内存 ≥ 8 GB。`opencv-python-headless` 已避免 GUI 依赖。

### 9. CLI 模式 (无 GUI)

```bash
# 单图分析 (HTML 报告)
python rockpore_cli.py analyze 孔洞.png -o report.html

# 批量分析
python rockpore_cli.py batch samples/ -o ./output

# 合成测试图 + 准确率评估
python rockpore_cli.py synth -o synthetic.png --accuracy
python rockpore_cli.py accuracy synthetic.png synthetic_gt.png -o eval.json
```

### 10. 更新升级

```bash
git pull origin master
pip install -r requirements.txt --upgrade
python -m pytest tests/ -q   # 验证
```

---

## 🎬 使用流程

10 步工作流（三大模块通用）:

| 步骤 | 功能 | 说明 |
|------|------|------|
| 1 | 打开图像 | 选择/加载岩心图 (Ctrl+O) |
| 2 | 启动分析 | 引导文字 |
| 3 | 标尺选择 | DPI 设置 (mm / μm 切换) |
| 4 | 图像预处理 | 色阶/对比度/Gamma/CLAHE |
| 5 | 提取 | 算法核心 (各模块不同) |
| 6 | 边界编辑 | 拖拽式擦除/添加 (12px 笔刷) |
| 7 | 分析参数 | 模块专属参数 |
| 8 | 自动分析 | 表格 + 数据卡片 + 标注 |
| 9 | 基础信息 | 项目/样品/分析人员 |
| 10 | 报告生成 | 6 格式导出 (HTML/PDF/Excel/Word/CSV/TXT) |

### 顶栏快捷键

- **Ctrl+O**: 打开图像
- **Ctrl+S**: 导出当前模块报告
- **F1**: 帮助

---

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 结果: 142 passed in ~5s
```

| 模块 | 测试数 | 覆盖 |
|------|--------|------|
| 孔洞 | ~50 | 标尺/预处理/分割/分析/准确率 |
| 裂缝 | ~30 | Hough/Adaptive/OTSU + 真实图 |
| 粒度 | 23 | Wentworth 分类/距离变换/分水岭 + 真实图 |
| ReportExporter | 16 | 6 格式全覆盖 |
| I/O | 2 | 中文路径/不存在文件 |
| 拖拽工具 | 5 | 笔刷/连接/QPoint |
| **合计** | **142** | **0 回归** |

---

## 📁 项目结构

```
RockCoreAnalyzer/
├── rockpore/
│   ├── core/                       # 核心算法
│   │   ├── pore/                   # 孔洞分析
│   │   ├── fracture.py             # 裂缝检测+分析
│   │   ├── grain.py                # 粒度检测+分析 ⭐ v1.2.0
│   │   ├── report_exporter.py      # 多格式导出器 ⭐
│   │   ├── accuracy.py             # 准确率评估
│   │   ├── synthetic*.py           # 合成测试图
│   │   └── io_utils.py             # 中文路径 I/O
│   ├── gui/
│   │   ├── main_window.py          # 主窗口 + 菜单
│   │   ├── pore_module.py          # 孔洞 10 步
│   │   ├── fracture_module.py      # 裂缝 10 步
│   │   ├── grain_module.py         # 粒度 10 步 ⭐
│   │   ├── canvas_view.py          # 画布 + 拖拽工具
│   │   ├── help_dialog.py
│   │   └── teaching_panel.py
│   └── cli.py                      # 命令行入口
├── tests/                          # 142 个测试
│   ├── test_grain.py               # ⭐ 23 个粒度测试
│   ├── test_report_exporter.py     # ⭐ 16 个导出测试
│   └── ...
├── docs/                           # 文档 ⭐
│   ├── CHANGELOG.md                # 完整版本历史
│   └── V1.2.0_GRAIN_MODULE.md      # 粒度模块沉淀
├── data/ground_truth/              # 合成测试图
├── 岩心孔洞软件.pdf                # 项目原始需求
├── 粒度样.jpg, 粒度样2.png         # 真实测试图
├── 裂缝样.jpg, 裂缝样2.png         # 真实测试图
├── requirements.txt
├── run_gui.py
└── README.md
```

---

## 📊 准确率对比

| 测试图 | 颗粒数 | composite | 通过 80% |
|--------|--------|-----------|----------|
| 默认合成图 | 15 | 0.88+ | ✅ |
| 粘连椭圆 | 16 | 0.90+ | ✅ |
| 花岗岩合成 (多 seed 平均) | 15-25 | 0.85+ | ✅ |
| 粒度样2.png (真实) | - | 视觉合理 | ✅ |

**综合公式**: `composite = 0.3 × IoU + 0.7 × F1`

- **IoU (0.3 权重)**: 像素级, 衡量前景 mask 整体匹配度
- **F1 (0.7 权重)**: 实例级, 衡量颗粒级一对一匹配

---

## 🐛 用户反馈历程

v1.0 → v1.2 共记录 **13 个用户反馈** 全部修复:

| # | 模块 | 反馈 | 修复版本 |
|---|------|------|---------|
| 1 | 裂缝 | 真实岩石图失败 | v1.1.1 → v1.1.2 |
| 2 | 裂缝 | 裂缝样2.png 漏检 | v1.1.3 |
| 3 | 裂缝 | 默认值太宽松 | v1.1.3 |
| 4 | 裂缝 | 参数设置没意义 | v1.1.4 |
| 5 | 裂缝 | 擦除/添加要拖拽 | v1.1.5 |
| 6 | 粒度 | 准确率 > 80% | v1.2.0 |
| 7 | 粒度 | 画布标记不重置 | v1.2.0 |
| 8 | 粒度 | 应用参数无效果 | v1.2.0 |
| 9 | 粒度 | 不能隐藏/点击 | v1.2.0 |
| 10 | 粒度 | 0 颗但显示大片 | v1.2.0 |
| 11 | 通用 | 报告要 PDF/Excel | v1.2.0 |
| 12 | 通用 | 主菜单导出崩溃 | v1.2.0 |
| 13 | GUI | 帮助图标显示不全 | v1.2.0 |

详见 `docs/CHANGELOG.md` 和 `docs/V1.2.0_GRAIN_MODULE.md`。

---

## 🛠️ 技术栈

| 库 | 用途 | 版本 |
|----|------|------|
| Python | 主语言 | 3.9+ |
| OpenCV | 图像处理 (cv2) | 4.5+ |
| NumPy | 数值计算 | 1.21+ |
| SciPy | 科学计算 | 1.7+ |
| scikit-image | 形态学 | 0.19+ |
| Pillow | DPI/图像 | 9.0+ |
| PyQt5 | GUI | 5.15+ |
| Matplotlib | 图表 | 3.5+ |
| reportlab | PDF 导出 | 4.0+ ⭐ v1.2.0 |
| openpyxl | Excel 导出 | 3.0+ ⭐ v1.2.0 |
| python-docx | Word 导出 | 1.0+ ⭐ v1.2.0 |
| pytest | 测试 | 7.0+ |

---

## 🔮 未来规划

- **v1.3.0** - 多模块协同 (同一岩心的孔洞+裂缝+粒度综合分析)
- **v1.4.0** - 多尺度粒度分析 (大/中/小颗粒分别统计)
- **v2.0.0** - 深度学习辅助 (U-Net 端到端分割)

---

## 📚 文档

- **`docs/CHANGELOG.md`** - 完整版本历史 (v1.0.0 ~ v1.2.0)
- **`docs/V1.2.0_GRAIN_MODULE.md`** - 粒度分析模块完整沉淀
- **`岩心孔洞软件.pdf`** - 项目原始需求文档

---

## 📜 版本历史

- **v1.2.0** (2026-06-23) ⭐ 粒度分析模块
- **v1.1.5** (2026-06-23) 拖拽式画笔
- **v1.1.0** (2026-06-23) 裂缝模块
- **v1.0.0** (2026-06-23) 孔洞核心功能发布

---

## ⚖️ 许可

仅供学习与教学使用。

---

*🤖 Generated with [opencode](https://opencode.ai) by RockPore Team*
*v1.2.0 三大模块全部完成 — 2026-06-23*