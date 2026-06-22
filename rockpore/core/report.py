"""报告生成模块.

依据 PDF 6.1 步骤 10"报告浏览"和 1.3 节"报告说明"要求,
生成 HTML 格式的孔洞分析报告,包含:
- 基础信息
- 孔洞统计表(总个数、总面积、面孔率、等效直径)
- 频率分布曲线数据
- 孔洞详细参数列表
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

from .analysis import (
    PoreAnalysisResult, compute_diameter_frequency_curve
)


@dataclass
class ReportData:
    """报告数据."""
    project_name: str = ""
    sample_id: str = ""
    analyst: str = ""
    image_path: str = ""
    image_size: tuple = (0, 0)
    scale_info: str = ""
    analysis_date: str = ""
    remarks: str = ""
    analysis_result: Optional[PoreAnalysisResult] = None
    original_image: Optional[np.ndarray] = None
    annotated_image: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


def _to_data_url(image: np.ndarray) -> str:
    """将 numpy 图像转 base64 data URL."""
    if image is None:
        return ""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _make_annotated_image(
    image: np.ndarray,
    mask: np.ndarray,
    result: PoreAnalysisResult,
) -> np.ndarray:
    """在原图上绘制检测结果:半透明覆盖 + 编号 + 边界框."""
    if image is None:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()
    # 透明叠加
    overlay = out.copy()
    overlay[mask > 0] = (255, 0, 255)  # 洋红
    cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
    # 绘制编号
    for pore in result.pores:
        cx, cy = pore.centroid
        x, y, w, h = pore.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(
            out, str(pore.id), (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
        )
    return out


def _format_pore_table(pores: List) -> str:
    """孔洞详细参数 HTML 表格."""
    rows = []
    for p in pores:
        rows.append(
            "<tr>"
            f"<td>{p.id}</td>"
            f"<td>{p.area_real:.3f}</td>"
            f"<td>{p.diameter_real:.3f}</td>"
            f"<td>{p.perimeter_px:.1f}</td>"
            f"<td>({p.centroid[0]:.0f}, {p.centroid[1]:.0f})</td>"
            f"<td>{p.size_class}</td>"
            f"<td>{p.filled_status.value}</td>"
            f"<td>{p.filled_material.value}</td>"
            f"<td>{p.effectiveness.value}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _frequency_chart_svg(freq: Dict[str, list]) -> str:
    """绘制频率直方图 SVG."""
    bins = freq.get("bins", [])
    counts = freq.get("counts", [])
    if not bins:
        return "<p style='color:#888'>无孔洞数据</p>"
    w, h = 480, 220
    pad = 30
    bar_w = (w - 2 * pad) / max(1, len(bins))
    max_c = max(counts) if counts else 1
    bars = []
    for i, c in enumerate(counts):
        bh = (c / max_c) * (h - 2 * pad)
        x = pad + i * bar_w
        y = h - pad - bh
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.8:.1f}" '
            f'height="{bh:.1f}" fill="#4a90e2"/>'
        )
        bars.append(
            f'<text x="{x + bar_w * 0.4:.1f}" y="{h - pad + 12:.1f}" '
            f'font-size="9" text-anchor="middle" fill="#333">{bins[i]:.1f}</text>'
        )
    axis = (
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#333"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="#333"/>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#fafafa;border:1px solid #ddd;">{axis}{"".join(bars)}</svg>'
    )


def generate_report(data: ReportData) -> str:
    """生成完整 HTML 报告."""
    if data.analysis_date is None or data.analysis_date == "":
        data.analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = data.analysis_result
    if result is None:
        return "<html><body><h1>无分析结果</h1></body></html>"
    freq = compute_diameter_frequency_curve(result.pores, bins=min(10, max(2, result.pore_count)))
    if data.annotated_image is None and data.original_image is not None:
        # 重建标注图(在 UI 端会预先传入)
        data.annotated_image = data.original_image
    img_orig = _to_data_url(data.original_image)
    img_annot = _to_data_url(data.annotated_image)
    pore_table = _format_pore_table(result.pores)
    size_dist = result.size_distribution
    size_dist_html = "".join(
        f"<tr><td>{k}</td><td>{v}</td><td>{v / max(1, result.pore_count) * 100:.1f}%</td></tr>"
        for k, v in size_dist.items()
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>岩心孔洞分析报告 - {data.sample_id or '未命名'}</title>
<style>
body{{font-family:'Microsoft YaHei',Arial,sans-serif;margin:0;padding:20px;background:#f5f5f5;color:#333;}}
h1{{text-align:center;color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px;}}
h2{{color:#2c3e50;background:#ecf0f1;padding:8px 12px;border-left:4px solid #3498db;margin-top:30px;}}
.section{{background:#fff;padding:20px;margin:10px 0;border-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
.info-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;}}
.info-item{{padding:8px;background:#fafafa;border-radius:3px;}}
.info-item .label{{font-weight:bold;color:#7f8c8d;display:block;font-size:12px;}}
.info-item .value{{font-size:14px;color:#2c3e50;}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px;}}
th{{background:#3498db;color:#fff;padding:8px;text-align:left;}}
td{{padding:6px 8px;border-bottom:1px solid #eee;}}
tr:nth-child(even){{background:#f9f9f9;}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:15px 0;}}
.metric{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:15px;border-radius:6px;text-align:center;}}
.metric .num{{font-size:24px;font-weight:bold;display:block;}}
.metric .lbl{{font-size:12px;opacity:0.9;}}
.images{{display:flex;gap:20px;flex-wrap:wrap;}}
.images > div{{flex:1;min-width:300px;}}
.images img{{max-width:100%;border:1px solid #ddd;border-radius:3px;}}
.footer{{text-align:center;color:#7f8c8d;font-size:12px;margin-top:30px;padding:10px;}}
</style>
</head>
<body>
<h1>岩心孔洞分析报告</h1>

<div class="section">
<h2>基础信息</h2>
<div class="info-grid">
<div class="info-item"><span class="label">项目名称</span><span class="value">{data.project_name or '-'}</span></div>
<div class="info-item"><span class="label">样品编号</span><span class="value">{data.sample_id or '-'}</span></div>
<div class="info-item"><span class="label">分析人员</span><span class="value">{data.analyst or '-'}</span></div>
<div class="info-item"><span class="label">分析日期</span><span class="value">{data.analysis_date}</span></div>
<div class="info-item"><span class="label">图像路径</span><span class="value">{data.image_path or '-'}</span></div>
<div class="info-item"><span class="label">图像尺寸</span><span class="value">{data.image_size[0]} × {data.image_size[1]} 像素</span></div>
<div class="info-item"><span class="label">标尺</span><span class="value">{data.scale_info or '-'}</span></div>
<div class="info-item"><span class="label">备注</span><span class="value">{data.remarks or '-'}</span></div>
</div>
</div>

<div class="section">
<h2>关键指标</h2>
<div class="metrics">
<div class="metric"><span class="num">{result.pore_count}</span><span class="lbl">孔洞总个数(全)</span></div>
<div class="metric"><span class="num">{result.pore_count_report}</span><span class="lbl">报告级孔洞数(≥2mm)</span></div>
<div class="metric"><span class="num">{result.porosity * 100:.2f}%</span><span class="lbl">孔洞面孔率</span></div>
<div class="metric"><span class="num">{result.average_diameter_real:.2f}</span><span class="lbl">平均等效直径(mm)</span></div>
</div>
</div>

<div class="section">
<h2>孔洞分类统计</h2>
<table>
<tr><th>分类</th><th>个数</th><th>占比</th></tr>
{size_dist_html}
</table>
</div>

<div class="section">
<h2>直径统计</h2>
<table>
<tr><th>指标</th><th>值</th></tr>
{''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in result.diameter_statistics.items())}
<tr><td>孔洞总面积(mm²)</td><td>{result.total_pore_area_real:.3f}</td></tr>
<tr><td>平均孔洞面积(mm²)</td><td>{result.average_area_real:.3f}</td></tr>
</table>
</div>

<div class="section">
<h2>直径频率分布</h2>
{_frequency_chart_svg(freq)}
</div>

<div class="section">
<h2>分析图</h2>
<div class="images">
<div><h3>原图</h3><img src="{img_orig}"/></div>
<div><h3>标注图</h3><img src="{img_annot}"/></div>
</div>
</div>

<div class="section">
<h2>孔洞详细参数</h2>
<table>
<tr><th>编号</th><th>面积(mm²)</th><th>等效直径(mm)</th><th>周长(像素)</th><th>质心(x,y)</th><th>分类</th><th>填充情况</th><th>填充物</th><th>有效性</th></tr>
{pore_table}
</table>
</div>

<div class="footer">报告由 岩心孔洞分析软件 v1.0 自动生成 · {data.analysis_date}</div>
</body>
</html>"""
    return html


def save_report(data: ReportData, output_path: str) -> str:
    """生成报告并保存到文件,返回文件路径."""
    html = generate_report(data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


__all__ = ["ReportData", "generate_report", "save_report", "_make_annotated_image"]
