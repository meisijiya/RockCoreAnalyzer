# 岩心孔洞分析软件 (Rock Core Pore Analysis)

一套基于 Python + OpenCV + PyQt5 的岩心图像孔洞识别、分析与报告生成软件。

实现《岩心孔洞软件》PDF 文档描述的 10 步工作流,支持岩心孔洞/裂缝的自动识别、特征参数计算、HTML 报告生成。

## 核心特性

- 🎯 **高准确率孔洞识别** - 在标准岩心合成图上达到 **97% 准确率**(Pixel IoU 94.58% + Detection F1 100%)
- 📏 **标尺自适应** - 自动读取图像 DPI,支持宏观(mm)与微观(μm)两种分析模式
- 🖼️ **10 步工作流** - 完整覆盖 PDF 6.1 节描述的图像分析流程
- 🖥️ **PyQt5 桌面 GUI** - 交互式图像画布,支持缩放/平移/橡皮擦/选择
- 📊 **HTML 报告** - 包含基础信息、关键指标、分类统计、频率分布曲线、孔洞详细参数表
- 🛠️ **CLI 工具** - 命令行批量分析、合成图生成、准确率评估
- 🧪 **68 个单元测试** - 覆盖标尺、预处理、分割、形态学、分析、报告、准确率

## 快速开始

### 安装依赖

```bash
# 系统包(可选,用于 PDF/图像处理)
sudo apt install poppler-utils

# Python 依赖
pip install -r requirements.txt
```

> WSL2 用户注意: 使用 `pip install --break-system-packages` 或先创建 venv。

### 启动 GUI

```bash
python3 run_gui.py
```

### 命令行分析

```bash
# 分析单张图
python3 rockpore_cli.py analyze samples/lf.jpg -o report.html

# 批量分析目录
python3 rockpore_cli.py batch samples/ -o ./output

# 准确率评估(需要 ground truth)
python3 rockpore_cli.py synth -o synthetic.png --accuracy
python3 rockpore_cli.py accuracy synthetic.png synthetic_gt.png -o eval.json
```

## 项目结构

```
.
├── rockpore/                      # 主包
│   ├── core/                      # 核心算法
│   │   ├── calibration.py         # 标尺换算 (像素↔mm/μm)
│   │   ├── preprocessing.py       # 图像预处理 (色阶/对比度/Gamma/滤波)
│   │   ├── segmentation.py        # 区域分割/颜色匹配
│   │   ├── morphology.py          # 数学形态学(膨胀/腐蚀/开闭/去噪/填充)
│   │   ├── analysis.py            # 孔洞分析与分类
│   │   ├── accuracy.py            # 准确率评估(IoU/F1)
│   │   ├── synthetic.py           # 合成测试图生成器
│   │   └── report.py              # HTML 报告生成
│   ├── gui/                       # PyQt5 桌面 GUI
│   │   ├── main_window.py         # 主窗口(10 步工作流)
│   │   └── widgets.py             # 自定义控件
│   └── cli.py                     # CLI 入口
├── tests/                         # 68 个单元测试
│   ├── test_calibration.py
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_analysis.py
│   ├── test_accuracy.py           # 端到端准确率测试
│   └── multi_accuracy.py          # 多难度集评估
├── data/ground_truth/             # 合成测试图 + GT
├── docs/                          # 文档
├── run_gui.py                     # GUI 启动脚本
├── rockpore_cli.py                # CLI 启动脚本
├── requirements.txt
└── README.md
```

## 10 步工作流

| 步骤 | PDF 对应 | 实现位置 |
|------|---------|---------|
| 1. 启动图像分析模块 | 6.1 步骤 1 | `MainWindow.open_image()` |
| 2. 启动孔洞分析 | 6.1 步骤 2 | `MainWindow.goto_step(2)` |
| 3. 标尺选择 | 6.1 步骤 3 | `ScalePanel` + `calibration.py` |
| 4. 图像预处理 | 6.1 步骤 4 | `PreprocessPanel` + `preprocessing.py` |
| 5. 孔洞提取 | 6.1 步骤 5 | `SegmentationPanel` + `segmentation.py` |
| 6. 二次编辑 | 6.1 步骤 6 | `MorphologyPanel` + `morphology.py` |
| 7. 孔洞填充 | 6.1 步骤 7 | `MorphologyPanel` (填充模式) |
| 8. 孔洞分析与特征参数 | 6.1 步骤 8 | `AnalysisPanel` + `analysis.py` |
| 9. 基础信息设置 | 6.1 步骤 9 | `PoreInfoDialog` + 步骤 9 面板 |
| 10. 报告浏览 | 6.1 步骤 10 | `generate_html_report()` + `report.py` |

## 算法核心

### 孔洞识别流水线

```
输入岩心图像
   ↓
灰度化 (BGR → GRAY)
   ↓
OTSU 自适应阈值 (暗色 = 孔洞)
   ↓
   ├─ 如果前景占比 > 50% 或 < 0.5%: 降级到 Triangle 阈值
   ↓
形态学闭运算 (连接断裂)
   ↓
形态学开运算 (去小毛刺)
   ↓
自适应分水岭 (分离粘连,仅在必要时)
   ↓
基于实际直径过滤 (< 1mm 视为噪点)
   ↓
掩码 → 孔洞分析(面积/直径/分类)
```

### 孔洞分类 (PDF 1.2 节)

| 类型 | 直径范围 |
|------|---------|
| 大洞 | > 10 mm |
| 中洞 | 5 ~ 10 mm |
| 小洞 | 1 ~ 4.9 mm |
| 针孔/溶孔 | < 1 mm |

报告标准(PDF 1.3 节): **直径 < 2mm 的孔洞不计数**。

### 关键公式 (PDF 1.1-1.2 节)

- 裂缝宽度: `W = A / L`
- 单个孔洞等效圆直径: `Dr = 2 × √(A / π)`
- 平均孔洞等效圆直径: `Dr = (Σ Di) / n`
- 孔洞面孔率: `Σ A_pore / A_image`

## 准确率评估

软件提供客观的准确率评估系统,使用合成测试图作为 ground truth。

### 综合评分

```python
composite_score = 0.5 × pixel_iou + 0.5 × detection_f1
```

- **pixel_iou**: 预测掩码 vs 真值掩码的像素级交并比
- **detection_f1**: 孔洞级 Precision/Recall 的调和平均(IoU ≥ 0.3 视为匹配)

### 测试结果

| 测试集 | 像素 IoU | Detection F1 | 综合得分 | 通过 80% |
|--------|----------|--------------|----------|----------|
| default (15 孔洞) | 94.58% | 100.00% | **97.29%** | ✓ |
| easy (6 孔洞) | 94.93% | 100.00% | **97.46%** | ✓ |
| hard (21 孔洞) | 92.50% | 97.56% | **95.03%** | ✓ |
| dense (7 孔洞) | 92.38% | 92.31% | **92.34%** | ✓ |

## CLI 用法

```
rockpore-cli [-h] {analyze,accuracy,batch,synth} ...
```

### analyze - 单图分析

```bash
rockpore-cli analyze <IMAGE> [OPTIONS]
  --dpi INT            DPI (默认从图像读取)
  --microscopic        微观分析(微米单位)
  --min-diameter FLOAT 报告级最小直径 (默认 2.0 mm)
  --output PATH        输出 (.html/.json/.png)
  --save-mask PATH     保存掩码
  --project STR        项目名称
  --sample-id STR      样品编号
  --analyst STR        分析人员
  --remarks STR        备注
  --verbose            显示孔洞详情
```

### accuracy - 准确率评估

```bash
rockpore-cli accuracy <IMAGE> <GT_MASK> [OPTIONS]
  --dpi INT     DPI (默认 96)
  --enhance     使用增强流水线
  --output PATH 保存评估 JSON
```

### batch - 批量分析

```bash
rockpore-cli batch <INPUT_DIR> [--output-dir DIR]
```

### synth - 生成合成测试图

```bash
rockpore-cli synth [--output PATH] [--accuracy]
```

## 测试

```bash
# 运行所有测试
python3 -m pytest tests/ -v

# 仅准确率测试
python3 -m pytest tests/test_accuracy.py -v

# 多难度集评估
python3 tests/multi_accuracy.py
```

## 报告示例

HTML 报告包含以下内容(参考 `output/` 目录):

- **基础信息** - 项目/样品/分析人员/日期/图像路径
- **关键指标卡片** - 孔洞总数、报告级孔洞数、面孔率、平均直径
- **分类统计表** - 大洞/中洞/小洞/针孔 的数量与占比
- **直径统计** - 平均/最大/最小/标准差
- **直径频率分布** - SVG 直方图(等效圆直径)
- **分析图** - 原图 + 标注图(洋红掩码 + 绿色边界框 + 编号)
- **孔洞详细参数** - ID/面积/直径/周长/质心/分类/填充情况/填充物/有效性

## 已知限制

1. **多子图合成图**: 包含多个独立子图的合成图(如 lf.jpg)需要先手工分割。
2. **裂缝 vs 孔洞**: 当前算法将所有暗色连通域视为孔洞;细长连通域可能是裂缝。
3. **背景/托盘干扰**: 真实拍摄图常含机械缝隙/托盘等干扰,推荐先做岩心区域提取。
4. **深度学习**: 当前为经典图像处理,如需 > 95% 复杂场景准确率,可加 U-Net 微调。

## 依赖说明

| 包 | 用途 | 必需 |
|------|------|------|
| opencv-python-headless | 图像处理 | ✓ |
| numpy | 数值计算 | ✓ |
| PyQt5 | GUI | ✓ |
| Pillow | DPI 读取 | ✓ |
| scipy | 科学计算 | 可选 |
| scikit-image | 扩展 | 可选 |
| pytest | 测试 | 开发 |

## 许可

仅供学习与教学使用。
